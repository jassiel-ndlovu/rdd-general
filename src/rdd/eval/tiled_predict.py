"""Score a tile-trained checkpoint on whole validation images (US15, Tier B).

The number this exists to produce
---------------------------------
``rdd.data.to_yolo_tiled`` tiles **both** halves of the split, so the metrics a
tiled training run reports are self-consistent but are not comparable with Tier
A's: they are measured over 1280 px crops, against a validation set that has
been cut up the same way, and Tier A's were measured over whole images. Setting
0.64 from one beside 0.65 from the other would be a mistake, and its own
``data.yaml`` header says so.

The comparable figure comes from this module: run the model on overlapping
tiles of each **original whole** validation image, map every detection back to
full-image coordinates, merge across tiles, and score the merged detections
against the **same whole-image ground truth Tier A was scored on**. Then the
only thing that changed between the two numbers is the model and its front end,
which is the comparison the Tier B intervention is supposed to be judged by.

Three things make that harder than it sounds, and each one is a decision here.

**1. Ultralytics cannot score this.** ``model.val()`` scores what its own
dataloader and its own post-processing produced. Our detections come out of
``rdd.data.tiling.merge_detections``, so the average precision has to be
computed from a detection list. It is, in ``rdd.eval.apmetrics`` -- as an exact
port of Ultralytics' own estimator rather than a fresh one or pycocotools,
because a Tier B number computed by a different estimator than Tier A's is not
comparable with it. That module's docstring makes the argument in full, and
``--ap-engine both`` adds a pycocotools cross-check beside the reported number.

**2. The tiling must be the export's tiling.** A model trained on tile 1280 /
overlap 0.2 / above 1200 and evaluated at tile 640 is being evaluated out of its
own domain. ``--tiling-from <tiled data.yaml>`` reads the parameters out of the
header ``to_yolo_tiled`` writes, so the evaluation cannot silently disagree with
the training export. Passing them by hand is still allowed, and the values used
are recorded in the report.

**3. ``model.predict()`` is not ``model.val()``.** This was the expensive
surprise. Scoring prediction output reproduced none of Tier A's recorded
per-class AP50 figures, missing C1 by 0.046 -- and the fault was not in the
average precision, which matches Ultralytics' own bit for bit, but in the
detections. Validation runs NMS with ``multi_label=True``, never clips boxes
before matching, and -- the part worth almost all of the 0.046 -- feeds the
network a **rectangular, stride-padded canvas**: a square image at imgsz 640
goes in on 672x672 with a 16 px grey border, not on 640x640. ``canvas_shape``
and ``ValEquivalentPredictor`` reproduce all three, so this path differs from
Tier A's evaluation in the tiling and nothing else.

**4. Inference is eight times the work.** Norway's 4040x2035 frames give 8 tiles
each at 1280/0.2 and there are 1,224 of them in the validation split, so the
pass is some 20,000 forward passes rather than 5,398. This streams: one source
image at a time, its tiles predicted in batches of ``--tile-batch``, crops
dropped as soon as the batch returns, and only the merged boxes kept. Nothing
accumulates but a few hundred bytes of arrays per image -- which matters,
because holding a whole evaluation in memory is what OOM-killed the g4dn during
US13.

Staging, for the same reason: ``--stage predict`` writes the merged detections
to a JSONL dump and ``--stage score`` scores a dump. Inference is the expensive
half and the scoring half is seconds, so a re-score (per country, a different
merge mode, a cross-check engine) never re-runs the model.

Validating the path before trusting it
--------------------------------------
``--untiled`` passes every image through whole -- one tile equal to the image --
which turns this module into exactly Tier A's evaluation with Tier A's weights,
routed through our own scoring. With ``--expect-from`` pointing at a recorded
``revalidation.json`` it then asserts the recorded per-class AP50 and exits
non-zero if it misses:

    rddpython -m rdd.eval.tiled_predict --untiled \
        --weights /opt/rdd/runs/tierA_base6_s0/weights/best.pt \
        --data    /opt/rdd/data/yolo/core/data.yaml \
        --out     /tmp/gate \
        --expect-from /opt/rdd/runs/tierA_base6_s0/reval/revalidation.json

That is the gate for this evaluation path: if those six numbers do not come
back, the path is wrong and its Tier B figure must not be reported. Running it
needs the validation images, so it runs where the dataset is.

Usage, once the gate has passed:

    rddpython -m rdd.eval.tiled_predict \
        --weights /opt/rdd/runs/tierB_tiled1280_s0/weights/best.pt \
        --data    /opt/rdd/data/yolo/core/data.yaml \
        --tiling-from /opt/rdd/data/yolo/tiled1280/data.yaml \
        --out     /opt/rdd/runs/tierB_tiled1280_s0/tiled_eval

The report is ``tiled_evaluation.json``, in the shape
``rdd.eval.revalidate`` writes: ``pooled``, ``per_class``, ``per_country`` and
``per_class_by_country``, so ``rdd.eval.aggregate`` reads it unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from rdd.data.tiling import (
    OVERLAP_DEFAULT,
    TILE_DEFAULT,
    TileDetection,
    merge_detections,
    tile_grid,
)
from rdd.eval.apmetrics import ImageStats, score, score_by_country
from rdd.eval.revalidate import country_of, load_labels, val_dirs

# Ultralytics' own validation defaults, so that a whole-image run through this
# path differs from `model.val()` in the scoring only. conf is the floor, not
# the reporting threshold: average precision is an integral over confidence and
# truncating at 0.25 would silently lower it.
CONF_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 300

# Per-image detection cap, applied after merging. Ultralytics caps at 300 per
# image; here MAX_DET is a cap per *tile*, which is the right analogue -- a
# Norway frame has 8 tiles and so up to 2,400 raw detections, and the merge
# removes the duplicates among them rather than the genuine extras.
IOU_MATCH = 0.5          # recorded for provenance; apmetrics uses 0.5:0.05:0.95

_TILING_HEADER = re.compile(
    r"#\s*tiling\s*:\s*tile=(\d+)\s+overlap=([\d.]+)\s+tile_above=(\w+)")


# ---------------------------------------------------------------------------
# The canvas the validator actually uses
# ---------------------------------------------------------------------------
def canvas_shape(h0: int, w0: int, imgsz: int = 640, stride: int = 32,
                 rect: bool = True) -> tuple[int, int]:
    """(height, width) of the letterboxed canvas validation feeds the network.

    Not ``(imgsz, imgsz)``, and getting that wrong is worth 0.046 AP50 on C1.
    Validation runs with ``rect=True`` and ``pad=0.5``, so
    ``BaseDataset.set_rectangle`` pads each batch out to
    ``ceil(ratio * imgsz / stride + 0.5) * stride``: a **square** image at imgsz
    640 goes in on a **672x672** canvas with a 16 px grey border on every side,
    and a 4040x2035 frame on 352x672. The boxes come back in the right places
    either way -- the cost is that every confidence is different, because the
    network sees a different canvas, and confidences are what average precision
    integrates over.

    The true shape is a property of the *batch*, which Ultralytics forms after
    sorting the whole split by aspect ratio. Computed per image, as here, it
    gives the identical answer whenever a batch is aspect-homogeneous: that
    covers every square image in RDD2022 (512 to 1080 px) and every tile of the
    tiled front end, since the grid shifts edge tiles inward rather than
    cropping them short. It can differ by one stride step for Norway's whole
    frames, whose three aspect ratios are close but not equal.
    """
    if not rect:
        return imgsz, imgsz
    ar = h0 / w0
    shape = [ar, 1.0] if ar < 1 else ([1.0, 1.0 / ar] if ar > 1 else [1.0, 1.0])
    h, w = (int(math.ceil(s * imgsz / stride + 0.5) * stride) for s in shape)
    return h, w


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class ValEquivalentPredictor:
    """Runs a YOLO checkpoint over a list of crops **the way the validator does**.

    Not ``model.predict()``. That was the first implementation of this class and
    it was wrong, in a way only the gate caught: it reproduced none of the
    recorded Tier A per-class AP50 figures, missing C1 by 0.046 while the
    average precision underneath matched Ultralytics' own bit for bit. Three
    differences between the prediction path and the validation path account for
    it, in descending order of how much they cost:

    1. **The canvas.** Validation runs ``rect=True, pad=0.5``, so a square
       image at imgsz 640 enters the network on a 672x672 canvas with a 16 px
       grey border a side; prediction letterboxes with ``auto=True`` to a plain
       stride multiple. Both put the boxes back in the right places -- the
       16 px measured out of the returned coordinates exactly -- but the
       network sees different pixels and returns different confidences, and
       average precision is an integral over confidence. Worth almost all of
       the 0.046. See ``canvas_shape``.
    2. **NMS.** Validation passes ``multi_label=True``, prediction ``False``:
       with it a box may be kept under more than one class. C1 and C2 differ by
       orientation alone and are this project's confusable pair, so this lands
       on exactly the classes that matter, worth about 0.005 on C2.
    3. **Clipping.** ``ops.scale_boxes`` clips to the frame. The validator
       never clips -- it matches in letterboxed model space against equally
       unclipped ground truth -- so a detection running off the edge keeps its
       full area in the IoU. Small on 512 px frames; not necessarily small on
       Norway's, where a crack at the frame edge is the common case.

    So this calls Ultralytics' own ``LetterBox`` and ``non_max_suppression`` in
    the validator's order with the validator's arguments, over an
    ``AutoBackend`` built the way the validator builds it, and maps boxes back
    with ``scale_boxes``' arithmetic minus its clip. Crops arrive as **BGR
    numpy views** -- the convention ``imread`` returns and the dataloader works
    in -- so no pixel is copied until the resize.
    """

    def __init__(self, weights: Path, imgsz: int = 640, conf: float = CONF_FLOOR,
                 iou: float = NMS_IOU, max_det: int = MAX_DET,
                 device: str | None = None, half: bool = False,
                 rect: bool = True) -> None:
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.torch_utils import select_device

        # AutoBackend unpickles the checkpoint, so a custom module in
        # it must be registered before the load or it fails on an
        # unknown class.
        from rdd.models import register as register_custom_modules
        register_custom_modules()

        self.torch = torch
        self.device = select_device(device or "cpu", verbose=False)
        self.net = AutoBackend(str(weights), device=self.device, dnn=False,
                               fp16=bool(half), fuse=True, verbose=False)
        self.net.eval()
        self.names = (self.net.names if isinstance(self.net.names, dict)
                      else dict(enumerate(self.net.names)))
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self.half = bool(half)
        self.rect = rect
        stride = getattr(self.net, "stride", 32)
        # AutoBackend gives an int for a .pt checkpoint; other backends can
        # hand back the model's stride tensor.
        self.stride = max(32, int(stride.max() if hasattr(stride, "max")
                                  else stride))
        self._LetterBox = LetterBox
        self._boxes: dict[tuple[int, int], LetterBox] = {}
        self.net.warmup(imgsz=(1, 3, imgsz, imgsz))

    def _canvas(self, h0: int, w0: int) -> tuple[int, int]:
        """The shape the validator would actually feed the network."""
        return canvas_shape(h0, w0, self.imgsz, self.stride, self.rect)

    # -- the validator's dataloader, for one crop ---------------------------
    def _prepare(self, crop):
        import cv2

        h0, w0 = crop.shape[:2]
        r = self.imgsz / max(h0, w0)
        img = crop
        if r != 1:
            # The validator's BaseDataset.load_image arithmetic, ceil and all.
            w = min(math.ceil(w0 * r), self.imgsz)
            h = min(math.ceil(h0 * r), self.imgsz)
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rh, rw = img.shape[0] / h0, img.shape[1] / w0
        canvas = self._canvas(h0, w0)
        lb = self._boxes.get(canvas)
        if lb is None:
            lb = self._boxes[canvas] = self._LetterBox(new_shape=canvas,
                                                       scaleup=False)
        labels = {"img": img, "ratio_pad": (rh, rw)}
        params = lb.get_params(labels)
        out = lb.apply_image(dict(labels), params)["img"]
        ratio_pad = ((rh * params["ratio"][1], rw * params["ratio"][0]),
                     (params["left"], params["top"]))
        return out, (h0, w0), ratio_pad

    def __call__(self, crops: list) -> list[list[tuple]]:
        from ultralytics.utils.nms import non_max_suppression

        prepared = [self._prepare(c) for c in crops]
        # A forward pass needs one shape. Crops of one image share theirs, so
        # this regroups only in the mixed case rather than forcing a padding
        # the validator would not have used.
        groups: dict[tuple, list[int]] = {}
        for i, p in enumerate(prepared):
            groups.setdefault(p[0].shape, []).append(i)

        dets: list = [None] * len(prepared)
        for idxs in groups.values():
            batch = np.stack([prepared[i][0] for i in idxs], 0)   # BGR
            t = self.torch.from_numpy(
                np.ascontiguousarray(batch.transpose(0, 3, 1, 2)[:, ::-1]))
            t = t.to(self.device)
            t = (t.half() if self.half else t.float()) / 255
            with self.torch.inference_mode():
                preds = self.net(t)
                outputs = non_max_suppression(
                    preds, self.conf, self.iou, nc=0, multi_label=True,
                    agnostic=False, max_det=self.max_det)
            for i, det in zip(idxs, outputs):
                dets[i] = det
            del preds, outputs, t, batch

        out = []
        for det, (_prep, shape0, ratio_pad) in zip(dets, prepared):
            if det is None or not len(det):
                out.append([])
                continue
            # ops.scale_boxes' arithmetic, with its final clip_boxes left out.
            # The validator never clips: _prepare_pred returns the boxes in
            # letterboxed model space and matches them there, against ground
            # truth that is equally unclipped, so a detection running off the
            # frame keeps its full area in the IoU. Clipping it here would
            # shrink that area and lose matches at the frame edge -- which is
            # where longitudinal cracks live, and was worth 0.04 AP50 on C1.
            (gain_y, gain_x), (pad_x, pad_y) = ratio_pad
            boxes = det[:, :4].clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain_x
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain_y
            out.append([
                (int(c), float(s), *(float(v) for v in xyxy))
                for xyxy, s, c in zip(boxes.tolist(), det[:, 4].tolist(),
                                      det[:, 5].tolist())
            ])
        del dets, prepared
        return out


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------
def predict_image(img, width: int, height: int, predictor, *,
                  tile: int, overlap: float, tile_above: int | None,
                  merge_mode: str, tile_batch: int,
                  merge_kw: dict | None = None) -> tuple[list[tuple], int]:
    """Tiled inference over one decoded image -> merged detections, n_tiles.

    ``img`` is an HWC BGR array, as ``imread`` returns it. Crops are numpy
    slices, which are views: a Norway frame is decoded once and its eight tiles
    cost nothing until the predictor resizes them.

    Returns ``[(cls, score, x0, y0, x1, y1), ...]`` in full-image pixels.

    An image at or below the tiling threshold yields a single whole-image tile,
    and then nothing is cropped and nothing is merged: the pixels go to the
    model exactly as Tier A sent them. That is the ``--tile-above`` pass-through
    the training export uses for the 512-720 px subsets, and it has to behave
    identically here or the two halves of the corpus would be evaluated under
    different rules.
    """
    tiles = tile_grid(width, height, tile, overlap, tile_above)
    whole = (len(tiles) == 1 and tiles[0].width == width
             and tiles[0].height == height)

    dets: list[TileDetection] = []
    if whole:
        for cls, sc, x0, y0, x1, y1 in predictor([img])[0]:
            dets.append(TileDetection(tiles[0].index, cls, sc, x0, y0, x1, y1))
    else:
        for start in range(0, len(tiles), max(1, tile_batch)):
            batch = tiles[start:start + max(1, tile_batch)]
            crops = [img[t.y0:t.y1, t.x0:t.x1] for t in batch]
            try:
                for t, found in zip(batch, predictor(crops)):
                    for cls, sc, x0, y0, x1, y1 in found:
                        dets.append(TileDetection(t.index, cls, sc,
                                                  x0, y0, x1, y1))
            finally:
                del crops

    if whole:
        # One tile covering the image: merging could only de-duplicate what the
        # detector's own NMS already ruled on, so it is skipped rather than
        # applied with nothing to do.
        merged = [(d.cls, d.score, d.x0, d.y0, d.x1, d.y1) for d in dets]
        merged.sort(key=lambda m: -m[1])
    else:
        merged = [
            (m.cls, m.score, m.x0, m.y0, m.x1, m.y1)
            for m in merge_detections(dets, tiles, width, height,
                                      mode=merge_mode, **(merge_kw or {}))
        ]
    return merged, len(tiles)


# ---------------------------------------------------------------------------
# Stage 1 -- predict
# ---------------------------------------------------------------------------
def stage_predict(args, predictor) -> tuple[list[ImageStats], dict]:
    """Stream the validation images, dump merged detections, keep the stats."""
    # The validator's own reader, not PIL: a JPEG decoded by two libraries can
    # differ in the last bit, and the gate is checked to four decimal places.
    from ultralytics.utils.patches import imread

    val_img_dir, val_lbl_dir = val_dirs(args.data)
    images = sorted(val_img_dir.glob("*.jpg"))
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no .jpg images under {val_img_dir}")

    dump = None
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        dump = args.dump.open("w", encoding="utf-8")

    stats: list[ImageStats] = []
    n_tiles_total = n_dets = 0
    t0 = time.time()
    print(f"== tiled inference over {len(images)} whole images ==")
    print(f"   tile={args.tile} overlap={args.overlap} "
          f"tile_above={args.tile_above} merge={args.merge_mode} "
          f"imgsz={args.imgsz} conf={args.conf}", flush=True)

    try:
        for n, path in enumerate(images, start=1):
            im = imread(str(path))
            if im is None:
                raise SystemExit(f"could not decode {path}")
            h, w = im.shape[:2]
            merged, n_tiles = predict_image(
                im, w, h, predictor,
                tile=args.tile, overlap=args.overlap,
                tile_above=args.tile_above, merge_mode=args.merge_mode,
                tile_batch=args.tile_batch, merge_kw=args.merge_kw)
            del im

            stem = path.stem
            gts = load_labels(val_lbl_dir / f"{stem}.txt", w, h)
            st = ImageStats(
                stem=stem, country=country_of(stem),
                pred_cls=[m[0] for m in merged],
                conf=[m[1] for m in merged],
                pred_boxes=[list(m[2:]) for m in merged],
                target_cls=[g[0] for g in gts],
                target_boxes=[list(g[1:5]) for g in gts],
            )
            stats.append(st)
            n_tiles_total += n_tiles
            n_dets += len(merged)

            if dump is not None:
                dump.write(json.dumps({
                    "stem": stem, "w": w, "h": h, "n_tiles": n_tiles,
                    "dets": [[m[0], round(m[1], 6), *(round(v, 2) for v in m[2:])]
                             for m in merged],
                    "gt": [[g[0], *(round(v, 2) for v in g[1:5])] for g in gts],
                }) + "\n")

            if n % 200 == 0:
                el = time.time() - t0
                # flush: redirected to a file this is block-buffered, and a
                # pass over 5,398 whole images runs long enough that an
                # operator watching the log needs to see it move.
                print(f"   {n}/{len(images)} images  {n_tiles_total} tiles  "
                      f"{n_dets} merged dets  {el:.0f}s "
                      f"({n / el:.1f} img/s)", flush=True)
                # The merged detections are kept; nothing else should be.
                gc.collect()
    finally:
        if dump is not None:
            dump.close()

    el = time.time() - t0
    prov = {
        "n_images": len(stats),
        "n_tiles": n_tiles_total,
        "tiles_per_image": round(n_tiles_total / max(1, len(stats)), 3),
        "n_merged_detections": n_dets,
        "seconds": round(el, 1),
    }
    print(f"   done: {len(stats)} images, {n_tiles_total} tiles "
          f"({prov['tiles_per_image']} per image), {n_dets} merged detections, "
          f"{el:.0f}s")
    if args.dump:
        print(f"   detections dumped to {args.dump}")
    return stats, prov


def load_dump(path: Path, limit: int = 0) -> list[ImageStats]:
    """Rebuild the per-image stats from a ``--dump`` file, no model needed."""
    stats = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            dets, gts = rec["dets"], rec["gt"]
            stats.append(ImageStats(
                stem=rec["stem"], country=country_of(rec["stem"]),
                pred_cls=[d[0] for d in dets],
                conf=[d[1] for d in dets],
                pred_boxes=[d[2:6] for d in dets],
                target_cls=[g[0] for g in gts],
                target_boxes=[g[1:5] for g in gts],
            ))
    return stats


# ---------------------------------------------------------------------------
# Stage 2 -- score
# ---------------------------------------------------------------------------
def stage_score(stats: list[ImageStats], names: dict[int, str],
                ap_engine: str) -> dict:
    block = score(stats, names)
    report = {"pooled": block["pooled"], "per_class": block["per_class"]}
    per_country, crossed = score_by_country(stats, names)
    report["per_country"] = per_country
    report["per_class_by_country"] = crossed

    print("\n== pooled ==")
    for k, v in report["pooled"].items():
        print(f"   {k:12s} {v:.4f}")
    print("\n== per class ==")
    for c, v in report["per_class"].items():
        print(f"   {c:26s} AP50={v['AP50']:.4f} "
              f"AP50_95={v['AP50_95']:.4f} R={v['recall']:.4f} "
              f"n={v['n_instances']}")
    print("\n== per country ==")
    for c, v in per_country.items():
        print(f"   {c:18s} n={v['n_images']:5d} mAP50={v['mAP50']:.4f} "
              f"mAP50_95={v['mAP50_95']:.4f}")

    if ap_engine == "both":
        try:
            from rdd.eval.apmetrics import score_pycocotools
            cross = score_pycocotools(stats, names)
        except ImportError:
            print("\n   pycocotools not installed; cross-check skipped")
        else:
            report["cross_check_pycocotools"] = cross
            print("\n== pycocotools cross-check (not the reported number) ==")
            for c, v in cross["per_class"].items():
                ours = report["per_class"].get(c, {}).get("AP50")
                delta = f"{v['AP50'] - ours:+.4f}" if ours is not None else "   n/a"
                print(f"   {c:26s} AP50={v['AP50']:.4f}  vs ours {delta}")
    return report


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def check_expected(report: dict, expected: dict[str, float], tol: float,
                   full: dict[str, dict] | None = None) -> bool:
    """Compare per-class AP50 against a recorded run. True if every class agrees.

    This is the acceptance test for the whole evaluation path, and it is
    deliberately all-or-nothing: an AP computation that is right for five
    classes and wrong for the sixth is not a scoring path with one bug, it is a
    scoring path that cannot be trusted for any of them.
    """
    print(f"\n== gate: per-class AP50 against the recorded run (tol {tol}) ==")
    ok = True
    for cls, want in sorted(expected.items()):
        got = report["per_class"].get(cls, {}).get("AP50")
        if got is None:
            print(f"   [MISS] {cls:26s} expected {want:.4f}, class absent")
            ok = False
            continue
        delta = got - want
        good = abs(delta) <= tol
        ok &= good
        print(f"   [{'ok' if good else 'FAIL':4s}] {cls:26s} "
              f"expected {want:.4f}  got {got:.4f}  delta {delta:+.4f}")
    print("   gate PASSED" if ok else "   gate FAILED -- do not report this path")

    # AP50 is what the gate turns on, because it is the figure the study
    # reports. The other three recorded numbers are printed beside it as
    # corroboration: they are not conditions, but a path that matched AP50 and
    # missed recall would be matching by luck.
    if full:
        print("\n   corroboration (not gated):")
        for cls in sorted(full):
            want, got = full[cls], report["per_class"].get(cls, {})
            parts = [f"{k} {got[k] - want[k]:+.4f}"
                     for k in ("AP50_95", "precision", "recall")
                     if k in want and k in got]
            if parts:
                print(f"     {cls:26s} " + "  ".join(parts))
    return ok


def expected_from(path: Path) -> tuple[dict[str, float], dict[str, dict]]:
    """(per-class AP50 to gate on, the whole recorded per-class block)."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    pc = rec.get("per_class") or {}
    return ({k: float(v["AP50"]) for k, v in pc.items() if "AP50" in v}, pc)


def tiling_from_header(path: Path) -> tuple[int, float, int | None]:
    """Read tile/overlap/tile_above out of a ``to_yolo_tiled`` data.yaml header.

    The export writes them as a comment, which is the only record of how the
    training view was built. Reading them back is what stops an evaluation from
    silently using a different grid than the one the model was trained on.
    """
    text = path.read_text(encoding="utf-8")
    m = _TILING_HEADER.search(text)
    if not m:
        raise SystemExit(
            f"{path} has no '# tiling :' header line -- is it a tiled export "
            f"written by rdd.data.to_yolo_tiled?")
    tile, overlap, above = int(m.group(1)), float(m.group(2)), m.group(3)
    return tile, overlap, (None if above == "None" else int(above))


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", type=Path,
                    help="checkpoint to evaluate (not needed with --score-only)")
    ap.add_argument("--data", type=Path,
                    help="data.yaml of the UNTILED whole-image view: its val "
                         "images are what we tile and its val labels are the "
                         "ground truth, so the score is Tier A comparable")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="model input size per tile (not the tile size)")

    ap.add_argument("--tile", type=int, default=TILE_DEFAULT)
    ap.add_argument("--overlap", type=float, default=OVERLAP_DEFAULT)
    ap.add_argument("--tile-above", type=int, default=None,
                    help="largest side an image may have and still pass "
                         "through whole")
    ap.add_argument("--tiling-from", type=Path,
                    help="a tiled data.yaml: take tile/overlap/tile-above from "
                         "its header, so evaluation matches the training export")
    ap.add_argument("--untiled", action="store_true",
                    help="pass every image through whole. Turns this into Tier "
                         "A's evaluation routed through our scoring, which is "
                         "how the path is validated -- see --expect-from")

    ap.add_argument("--merge-mode", choices=("stitch", "nms"), default="stitch")
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--merge-contain", type=float, default=0.7)
    ap.add_argument("--merge-align", type=float, default=0.5)
    ap.add_argument("--merge-gap", type=float, default=8.0)
    ap.add_argument("--merge-edge", type=float, default=4.0)

    ap.add_argument("--conf", type=float, default=CONF_FLOOR)
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU)
    ap.add_argument("--max-det", type=int, default=MAX_DET,
                    help="per tile, not per image")
    ap.add_argument("--tile-batch", type=int, default=8,
                    help="tiles per forward pass. 8 is one Norway frame at "
                         "1280/0.2; raise it on a bigger card, lower it if "
                         "memory is tight")
    ap.add_argument("--square-canvas", action="store_true",
                    help="letterbox to a plain imgsz x imgsz square instead of "
                         "the rectangular, stride-padded canvas validation "
                         "actually uses. Not Tier A comparable; see "
                         "canvas_shape()")
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many images (a smoke test)")

    ap.add_argument("--stage", choices=("predict", "score", "both"),
                    default="both")
    ap.add_argument("--dump", type=Path,
                    help="JSONL of merged detections and ground truth, so a "
                         "re-score never re-runs the model")
    ap.add_argument("--score-only", type=Path,
                    help="score an existing --dump instead of inferring")
    ap.add_argument("--ap-engine", choices=("ultralytics", "both"),
                    default="ultralytics",
                    help="'both' adds a pycocotools cross-check beside the "
                         "reported number; see rdd.eval.apmetrics")
    ap.add_argument("--expect-from", type=Path,
                    help="a recorded revalidation.json whose per-class AP50 "
                         "this run must reproduce; non-zero exit if it does not")
    ap.add_argument("--expect-tol", type=float, default=0.0005,
                    help="allowed absolute AP50 difference. The default is half "
                         "the last recorded decimal place")
    ap.add_argument("--names", type=Path,
                    help="data.yaml to take class names from when --score-only "
                         "leaves no model loaded")
    args = ap.parse_args(argv)

    if args.stage == "score" and not args.score_only:
        ap.error("--stage score needs --score-only <dump.jsonl>: there is "
                 "nothing to score without either a dump or an inference pass")
    if args.tiling_from:
        args.tile, args.overlap, args.tile_above = tiling_from_header(args.tiling_from)
        print(f"tiling from {args.tiling_from}: tile={args.tile} "
              f"overlap={args.overlap} tile_above={args.tile_above}")
    if args.untiled:
        # Every real image is smaller than this, so the grid is one whole-image
        # tile and the crop/merge path is bypassed entirely.
        args.tile_above = 10 ** 9
        args.tile = max(args.tile, 10 ** 9)
    args.merge_kw = dict(iou_thr=args.merge_iou, contain_thr=args.merge_contain,
                         align_thr=args.merge_align, gap_tol=args.merge_gap,
                         edge_tol=args.merge_edge)

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "tiled_evaluation.json"

    names: dict[int, str] = {}
    stats: list[ImageStats]
    prov: dict = {}

    if args.score_only:
        stats = load_dump(args.score_only, args.limit)
        print(f"scoring {len(stats)} images from {args.score_only}")
        src = args.names or args.data
        if src:
            import yaml
            cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
            names = {int(k): v for k, v in cfg["names"].items()}
        prov = {"scored_from": str(args.score_only), "n_images": len(stats)}
    else:
        if not (args.weights and args.data):
            ap.error("--weights and --data are required unless --score-only")
        predictor = ValEquivalentPredictor(
            args.weights, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
            max_det=args.max_det, device=args.device, half=args.half,
            rect=not args.square_canvas)
        names = predictor.names
        stats, prov = stage_predict(args, predictor)
        del predictor
        gc.collect()
        if args.stage == "predict":
            # Inference only: write what we have and leave the scoring to a
            # second process, which is how a 15 GB box survives both halves.
            out_json.write_text(json.dumps({"inference": prov}, indent=2),
                                encoding="utf-8")
            print(f"\nwrote {out_json} (inference only)")
            return 0

    report = {
        "weights": str(args.weights) if args.weights else None,
        "data": str(args.data) if args.data else None,
        "imgsz": args.imgsz,
        "iou_match": IOU_MATCH,
        "ap_engine": "ultralytics-8.4.131-port (rdd.eval.apmetrics)",
        "canvas": "square" if args.square_canvas else "rect(pad=0.5)",
        "tiling": {
            "tile": args.tile, "overlap": args.overlap,
            "tile_above": args.tile_above, "untiled": bool(args.untiled),
            "merge_mode": args.merge_mode, **args.merge_kw,
        },
        "inference": prov,
        "data_yaml_header": "".join(
            ln + "\n" for ln in args.data.read_text(encoding="utf-8").splitlines()
            if ln.startswith("#")) if args.data else "",
    }
    report.update(stage_score(stats, names, args.ap_engine))

    rc = 0
    if args.expect_from:
        expected, recorded = expected_from(args.expect_from)
        passed = check_expected(report, expected, args.expect_tol, recorded)
        report["gate"] = {
            "expected_from": str(args.expect_from),
            "tolerance": args.expect_tol,
            "expected_per_class_AP50": expected,
            "passed": passed,
        }
        rc = 0 if passed else 1

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
