"""
Exercise the tiled evaluation path end to end, with no GPU and no dataset (US15).

What this has to establish
--------------------------
``rdd.eval.tiled_predict`` produces the one number Tier B is judged by: a score
for a tile-trained checkpoint over whole validation images, set beside Tier A's.
Two independent things have to be right for that number to mean anything.

**The average precision.** It cannot come from ``model.val()`` -- the detections
are ours, not its dataloader's -- so ``rdd.eval.apmetrics`` computes it, as a
port of Ultralytics' own estimator (that module's docstring says why it is a
port and not pycocotools). Sections A and B check the port: against hand
computable cases, against the installed Ultralytics where there is one, and
against pycocotools where there is one.

**The plumbing.** Grid, crop, map back, merge, match against whole-image ground
truth. Section C runs the real CLI over a renderable copy of the synthetic
fixture with a stub detector in place of a model, and section D covers the
degenerate cases ``rdd.verify.tile_roundtrip`` already covers geometrically --
an image with no boxes, a box cut by a seam, an image below the tiling
threshold -- this time through the scoring path rather than the geometry.

Section E pins one rule that is neither of those and broke the path once: the
shape of the canvas validation feeds the network. It is not ``imgsz x imgsz``,
and letterboxing to a square instead cost 0.046 AP50 on C1 with every box
still in the right place. The gate caught that, but the gate needs the
validation images, so the rule is also asserted here where nothing is needed.

The fixture's own images are placeholder bytes (five bytes of JPEG magic), which
costs ``tile_roundtrip`` nothing because every check there is geometric. This
file crops pixels, so it renders real JPEGs at the fixture's dimensions and
reuses the fixture's boxes and filenames. The boxes under test are still the
ones the pipeline produces.

A stub detector, not a checkpoint: the question here is whether a detection that
enters the path comes out the other end correctly scored, and a real model would
make that question depend on what it happened to detect. The oracle stub emits
exactly the ground truth, clipped per tile, so any departure from a perfect
score is this code's fault.

Usage
-----
    python -m rdd.verify.tiled_eval --root /tmp/fixture

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np

from rdd.data.tiling import Tile, clip_box_to_tile, tile_grid
from rdd.eval import apmetrics, tiled_predict
from rdd.eval.apmetrics import ImageStats, ap_per_class, box_iou, match_predictions
from rdd.verify.tile_roundtrip import Checks, load_fixture

TOL = 1e-9


def _imread(path: Path) -> np.ndarray:
    """BGR array, the convention rdd.eval.tiled_predict works in.

    Uses Ultralytics' reader when it is installed, so the verifier exercises
    the same decode the gate does, and PIL otherwise -- no check here depends
    on the pixels, only on their shape.
    """
    try:
        from ultralytics.utils.patches import imread
        return imread(str(path))
    except Exception:                                       # noqa: BLE001
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()

# The Tier B training export's configuration, and Norway's frame size at it.
TILE, OVERLAP, ABOVE = 1280, 0.2, 1200
NORWAY = (4040, 2035)


# ---------------------------------------------------------------------------
# Stub detectors
# ---------------------------------------------------------------------------
class ScriptedPredictor:
    """Stands in for ``UltralyticsTilePredictor``.

    ``begin(tiles, boxes)`` arms it for one image; each call then consumes the
    next tiles in grid order and returns detections in tile-local coordinates.
    Relying on call order is deliberate -- check C0 asserts that contract, so if
    ``predict_image`` ever stops calling the model once per tile in order, these
    checks fail loudly rather than quietly scoring the wrong tiles.
    """

    def __init__(self, score_for=None) -> None:
        self.tiles: list[Tile] = []
        self.boxes: list[tuple] = []
        self.cursor = 0
        self.calls: list[int] = []            # batch sizes, for C0
        self.seen: list[int] = []             # tile indices, in order
        self.score_for = score_for or (lambda cls, box, tile: 1.0)

    def begin(self, tiles, boxes) -> None:
        self.tiles, self.boxes, self.cursor = tiles, boxes, 0
        self.calls, self.seen = [], []

    def __call__(self, crops: list) -> list[list[tuple]]:
        self.calls.append(len(crops))
        out = []
        for _ in crops:
            t = self.tiles[self.cursor]
            self.cursor += 1
            self.seen.append(t.index)
            found = []
            for cls, box in self.boxes:
                clipped = clip_box_to_tile(box, t)
                if clipped is None:
                    continue
                x0, y0, x1, y1, _retained = clipped
                found.append((cls, self.score_for(cls, box, t), x0, y0, x1, y1))
            out.append(found)
        return out


class EmptyPredictor:
    def __init__(self) -> None:
        self.n = 0

    def begin(self, tiles, boxes) -> None:
        pass

    def __call__(self, crops: list) -> list[list[tuple]]:
        self.n += len(crops)
        return [[] for _ in crops]


# ---------------------------------------------------------------------------
# A. the average precision itself
# ---------------------------------------------------------------------------
def _stats(preds, gts, stem="Japan_000000") -> ImageStats:
    return ImageStats(
        stem=stem, country=stem.rsplit("_", 1)[0],
        pred_cls=[p[0] for p in preds], conf=[p[1] for p in preds],
        pred_boxes=[list(p[2:]) for p in preds],
        target_cls=[g[0] for g in gts], target_boxes=[list(g[1:]) for g in gts])


def check_ap(ch: Checks) -> None:
    names = {0: "C1", 1: "C2"}

    # A1 one ground truth, one exact detection. The answer is 0.995, not 1.0,
    # and that is the point: compute_ap appends a sentinel precision of 0 at
    # recall 1, so the trapezoid loses half of the last 1/100th of the
    # interval. It is why a flawless Ultralytics run reports 0.995 and never
    # 1.000, and reproducing that ceiling exactly is the first evidence the
    # port is the same estimator. COCO's mean-of-101 on the identical curve
    # gives 0.9901 instead.
    r = apmetrics.score([_stats([(0, 0.9, 0, 0, 10, 10)], [(0, 0, 0, 10, 10)])],
                        names)
    ch.check(abs(r["per_class"]["C1"]["AP50"] - 0.995) < TOL,
             "A1 a perfect detection scores exactly 0.995",
             f"got {r['per_class']['C1']['AP50']} -- Ultralytics' ceiling; "
             f"mean-of-101 would give 0.9901")

    # A2 a detection too far off to match at IoU 0.5 (IoU here is 100/300).
    r = apmetrics.score([_stats([(0, 0.9, 5, 0, 15, 10)], [(0, 0, 0, 10, 10)])],
                        names)
    ch.check(r["per_class"]["C1"]["AP50"] == 0.0,
             "A2 a detection below IoU 0.5 scores AP50 = 0.0",
             f"IoU {float(box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0]):.3f}")

    # A3 half the ground truth found. The recall curve stops at 0.5 and the
    # envelope falls to zero there, so AP is half the ceiling exactly.
    r = apmetrics.score([_stats([(0, 0.9, 0, 0, 10, 10)],
                                [(0, 0, 0, 10, 10), (0, 90, 90, 100, 100)])],
                        names)
    ch.check(abs(r["per_class"]["C1"]["AP50"] - 0.495) < TOL,
             "A3 one of two instances found scores exactly 0.495",
             f"got {r['per_class']['C1']['AP50']} -- half of 0.995, which is "
             f"what a recall curve ending at 0.5 integrates to")

    # A3b rank matters: the same two detections cost more when the false
    # positive outranks the true one. A scoring path that ignored confidence
    # order would return the same number twice.
    tp_first = apmetrics.score([_stats(
        [(0, 0.9, 0, 0, 10, 10), (0, 0.8, 50, 50, 60, 60)],
        [(0, 0, 0, 10, 10)])], names)["per_class"]["C1"]["AP50"]
    fp_first = apmetrics.score([_stats(
        [(0, 0.9, 50, 50, 60, 60), (0, 0.8, 0, 0, 10, 10)],
        [(0, 0, 0, 10, 10)])], names)["per_class"]["C1"]["AP50"]
    ch.check(fp_first < tp_first,
             "A3b a false positive ranked above a true one costs AP",
             f"TP first {tp_first:.4f}, FP first {fp_first:.4f}")

    # A4 a class with ground truth and no predictions is reported as zero.
    r = apmetrics.score([_stats([(0, 0.9, 0, 0, 10, 10)],
                                [(0, 0, 0, 10, 10), (1, 20, 20, 30, 30)])], names)
    ch.check(r["per_class"].get("C2", {}).get("AP50") == 0.0
             and r["per_class"]["C2"]["n_instances"] == 1,
             "A4 a class with labels and no detections scores 0.0, not absent")

    # A5 a class predicted but never annotated is omitted entirely -- the
    # ap_class_index rule rdd.eval.revalidate.per_class_from documents.
    r = apmetrics.score([_stats([(1, 0.9, 20, 20, 30, 30)], [(0, 0, 0, 10, 10)])],
                        names)
    ch.check("C2" not in r["per_class"] and "C1" in r["per_class"],
             "A5 a class with no labels is omitted, not scored 0.0")

    # A6 pooled mAP is the mean over classes that have labels.
    r = apmetrics.score([_stats([(0, 0.9, 0, 0, 10, 10), (1, 0.9, 20, 20, 30, 30)],
                                [(0, 0, 0, 10, 10), (1, 20, 20, 30, 30)])], names)
    ch.check(abs(r["pooled"]["mAP50"] - 0.995) < TOL
             and r["pooled"]["recall"] == 1.0,
             "A6 two perfect classes give pooled mAP50 0.995, recall 1.0",
             "pooled mAP is the mean over classes with labels")

    # A7 no detections at all anywhere.
    r = apmetrics.score([_stats([], [(0, 0, 0, 10, 10)])], names)
    ch.check(r["per_class"]["C1"]["AP50"] == 0.0 and r["pooled"]["mAP50"] == 0.0,
             "A7 an evaluation with no detections scores 0.0 and does not crash")


def check_matching(ch: Checks) -> None:
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])

    # B1 two detections contend for one ground truth: the earlier one in the
    # list wins, whatever their IoUs. That is worth being exact about, because
    # the obvious reading of Ultralytics' code ("sort the pairs by descending
    # IoU, then take them greedily") is wrong: the np.unique that de-duplicates
    # detections re-sorts the pairs into detection order, so the second
    # de-duplication -- the one over ground truths -- resolves ties by position
    # in the detection list, not by IoU. The list arrives sorted by descending
    # confidence (NMS emits it that way, and so does
    # rdd.data.tiling.merge_detections), so in practice the ground truth goes
    # to the more confident detection even when the other one fits better.
    better_second = np.array([[0.0, 0.0, 9.0, 9.0], [0.0, 0.0, 10.0, 10.0]])
    better_first = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]])
    a = match_predictions(np.array([0, 0]), np.array([0]), box_iou(gt, better_second))
    b = match_predictions(np.array([0, 0]), np.array([0]), box_iou(gt, better_first))
    ch.check(bool(a[0, 0]) and not bool(a[1, 0])
             and bool(b[0, 0]) and not bool(b[1, 0]),
             "B1 two detections for one label: the first in the list takes it",
             "IoU 0.81 before 1.00 still wins -- confidence order decides, not fit")

    # B1b the IoU ordering does still decide which *label* a detection is
    # credited with when it overlaps two of them: each detection keeps its
    # best-IoU pairing before the contention above is resolved.
    two_gt = np.array([[0.0, 0.0, 10.0, 10.0], [8.0, 0.0, 20.0, 10.0]])
    one_pred = np.array([[7.0, 0.0, 19.0, 10.0]])       # fits the second label
    tp2 = match_predictions(np.array([0]), np.array([0, 0]), box_iou(two_gt, one_pred))
    ch.check(bool(tp2[0, 0]),
             "B1b a detection is credited against its best-IoU label")

    # B2 class-aware: a perfect box of the wrong class is not a match.
    tp = match_predictions(np.array([1]), np.array([0]), box_iou(gt, gt))
    ch.check(not bool(tp[0, 0]), "B2 a perfect box of the wrong class never matches",
             "C1 and C2 differ only by orientation, so this must hold")

    # B3 one ground truth absorbs one detection and no more.
    pred = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.5, 9.5]])
    tp = match_predictions(np.array([0, 0]), np.array([0]), box_iou(gt, pred))
    ch.check(int(tp[:, 0].sum()) == 1,
             "B3 a ground truth is matched at most once")

    # B4 the IoU vector is 0.5:0.05:0.95 and a borderline box falls out of the
    # strict columns -- the behaviour AP50_95 depends on.
    pred = np.array([[0.0, 0.0, 10.0, 7.0]])            # IoU 0.7
    tp = match_predictions(np.array([0]), np.array([0]), box_iou(gt, pred))
    ch.check(bool(tp[0, 0]) and bool(tp[0, 3]) and not bool(tp[0, 5]),
             "B4 a 0.70 IoU box is a TP to the 0.65 column and not beyond",
             f"columns {''.join('1' if v else '0' for v in tp[0])}")

    # B5 box_iou runs in single precision, because Ultralytics' does. Where
    # A8 cannot run (no Ultralytics installed) this is what guards the choice:
    # the same boxes in double precision give a different number, so the dtype
    # is not cosmetic -- it decides a box sitting exactly on a threshold.
    a = np.array([[600.0, 400.0, 1600.0, 460.0]])        # Norway-scale crack
    b = np.array([[600.0, 400.0, 1341.0, 460.0]])
    f32 = box_iou(a, b)
    lt, rb = np.maximum(a[:, :2], b[:, :2]), np.minimum(a[:, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    f64 = inter / ((a[:, 2:] - a[:, :2]).prod(1) + (b[:, 2:] - b[:, :2]).prod(1)
                   - inter + 1e-7)
    ch.check(f32.dtype == np.float32 and float(f32[0, 0]) != float(f64[0]),
             "B5 box_iou is float32, as Ultralytics' is",
             f"float32 {float(f32[0, 0]):.9f} vs float64 {float(f64[0]):.9f}")


def check_against_ultralytics(ch: Checks) -> None:
    """Numerical equivalence with the installed Ultralytics, if there is one."""
    try:
        import torch
        from ultralytics.utils.metrics import ap_per_class as ul_ap_per_class
        from ultralytics.utils.metrics import box_iou as ul_box_iou
    except Exception as e:                                  # noqa: BLE001
        ch.note("A8 equivalence with ultralytics",
                f"skipped: {type(e).__name__} -- not installed here. Run this "
                f"verifier on the instance or in CI, where it is pinned.")
        return

    rng = np.random.default_rng(0)
    worst_iou = worst_ap = 0.0
    for _ in range(25):
        n_gt, n_pred = int(rng.integers(0, 12)), int(rng.integers(0, 20))
        gt = rng.uniform(0, 500, (max(n_gt, 1), 2))
        gt = np.hstack([gt, gt + rng.uniform(5, 120, (max(n_gt, 1), 2))])[:n_gt]
        pr = rng.uniform(0, 500, (max(n_pred, 1), 2))
        pr = np.hstack([pr, pr + rng.uniform(5, 120, (max(n_pred, 1), 2))])[:n_pred]
        if n_gt and n_pred:
            mine = box_iou(gt, pr)
            theirs = ul_box_iou(torch.from_numpy(gt), torch.from_numpy(pr)).numpy()
            worst_iou = max(worst_iou, float(np.abs(mine - theirs).max()))

        tp = rng.random((n_pred, 10)) > 0.5
        conf = rng.random(n_pred)
        pred_cls = rng.integers(0, 6, n_pred)
        target_cls = rng.integers(0, 6, n_gt)
        if n_gt == 0:
            continue
        mine = ap_per_class(tp, conf, pred_cls, target_cls)
        t = ul_ap_per_class(tp.astype(float), conf, pred_cls, target_cls)
        _tp, _fp, p, r, _f1, ap, idx = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
        worst_ap = max(
            worst_ap,
            float(np.abs(mine["ap"] - ap).max()),
            float(np.abs(mine["p"] - p).max()),
            float(np.abs(mine["r"] - r).max()),
            float(np.abs(mine["ap_class_index"] - idx).max()),
        )

    ch.check(worst_iou == 0.0, "A8 box_iou matches ultralytics bit for bit",
             f"max abs difference {worst_iou:.3e} over 25 random cases "
             f"(float32, as theirs is)")
    ch.check(worst_ap == 0.0, "A9 ap_per_class matches ultralytics bit for bit",
             f"max abs difference {worst_ap:.3e} over 25 random cases, "
             f"covering ap, precision, recall and ap_class_index")


def check_against_pycocotools(ch: Checks) -> None:
    """A loose cross-check: the two estimators should not be far apart."""
    try:
        import pycocotools                                  # noqa: F401
    except Exception as e:                                  # noqa: BLE001
        ch.note("A10 pycocotools cross-check",
                f"skipped: {type(e).__name__} -- not installed here. It is "
                f"pinned in requirements.txt, so it runs on the instance.")
        return

    rng = np.random.default_rng(1)
    names = {0: "C1", 1: "C2"}
    stats = []
    for i in range(40):
        n = int(rng.integers(1, 6))
        gts = []
        for _ in range(n):
            x, y = rng.uniform(0, 400, 2)
            gts.append((int(rng.integers(0, 2)), x, y, x + 60, y + 40))
        preds = [(g[0], float(rng.uniform(0.3, 1.0)),
                  g[1] + rng.normal(0, 4), g[2] + rng.normal(0, 4),
                  g[3] + rng.normal(0, 4), g[4] + rng.normal(0, 4))
                 for g in gts if rng.random() > 0.2]
        stats.append(_stats(preds, gts, stem=f"Japan_{i:06d}"))

    ours = apmetrics.score(stats, names)["per_class"]
    theirs = apmetrics.score_pycocotools(stats, names)["per_class"]
    gaps = {c: abs(ours[c]["AP50"] - theirs[c]["AP50"]) for c in ours if c in theirs}
    ch.check(gaps and max(gaps.values()) < 0.05,
             "A10 pycocotools agrees with the port to within 0.05 AP50",
             ", ".join(f"{c} delta {v:.4f}" for c, v in gaps.items())
             + "  (they differ by construction; see rdd.eval.apmetrics)")


# ---------------------------------------------------------------------------
# C. the whole path, on a renderable copy of the fixture
# ---------------------------------------------------------------------------
def render_fixture(coco: dict, root: Path) -> tuple[Path, dict[int, str]]:
    """Write a YOLO val tree with real pixels, from the fixture's boxes.

    The fixture ships placeholder image bytes, so a path that crops cannot use
    it directly. Dimensions, filenames and boxes are the fixture's; only the
    pixels are invented, and no check here depends on what they contain.
    """
    from PIL import Image

    present = {a["category_id"] for a in coco["annotations"]}
    cats = [c for c in sorted(coco["categories"], key=lambda c: c["id"])
            if c["id"] in present]
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = {i: c["name"] for i, c in enumerate(cats)}

    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image.setdefault(a["image_id"], []).append(
            (cat_to_idx[a["category_id"]], (x, y, x + w, y + h)))

    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        w, h = im["width"], im["height"]
        # A gradient, so a crop is visibly a crop if anyone ever looks.
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        Image.fromarray(arr).save(img_dir / f"{stem}.jpg", "JPEG", quality=90)
        lines = [
            f"{c} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} "
            f"{(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
            for c, b in by_image.get(im["id"], [])
        ]
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_text = (
        "# Rendered by rdd.verify.tiled_eval from the synthetic fixture.\n"
        f"path: {root.resolve()}\n"
        "train: images/val\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items())
    )
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return root / "data.yaml", names


class _StubbedRun:
    """Runs ``tiled_predict.stage_predict`` with a stub in place of the model."""

    def __init__(self, data_yaml: Path, tmp: Path, names: dict[int, str]) -> None:
        self.data_yaml = data_yaml
        self.tmp = tmp
        self.names = names

    def run(self, predictor, **over) -> tuple[dict, list]:
        import types

        args = types.SimpleNamespace(
            data=self.data_yaml, limit=0, dump=None,
            tile=256, overlap=0.25, tile_above=None, merge_mode="stitch",
            tile_batch=8, merge_kw={},
        )
        for k, v in over.items():
            setattr(args, k, v)

        # The stub needs to know which image it is looking at, and
        # stage_predict does not tell it; wrap the loop's call so the oracle is
        # armed per image, in the same order the CLI walks them.
        from rdd.eval.revalidate import load_labels

        img_dir, lbl_dir = tiled_predict.val_dirs(self.data_yaml)
        images = sorted(img_dir.glob("*.jpg"))
        stats, records = [], []
        for path in images:
            im = _imread(path)
            h, w = im.shape[:2]
            gts = load_labels(lbl_dir / f"{path.stem}.txt", w, h)
            tiles = tile_grid(w, h, args.tile, args.overlap, args.tile_above)
            if hasattr(predictor, "begin"):
                predictor.begin(tiles, [(g[0], tuple(g[1:5])) for g in gts])
            merged, n_tiles = tiled_predict.predict_image(
                im, w, h, predictor, tile=args.tile, overlap=args.overlap,
                tile_above=args.tile_above, merge_mode=args.merge_mode,
                tile_batch=args.tile_batch, merge_kw=args.merge_kw)
            if hasattr(predictor, "seen"):
                self.order_ok = getattr(self, "order_ok", True) and (
                    predictor.seen == [t.index for t in tiles])
                self.sorted_ok = getattr(self, "sorted_ok", True) and all(
                    merged[i][1] >= merged[i + 1][1] for i in range(len(merged) - 1))
            stats.append(ImageStats(
                stem=path.stem, country=path.stem.rsplit("_", 1)[0],
                pred_cls=[m[0] for m in merged], conf=[m[1] for m in merged],
                pred_boxes=[list(m[2:]) for m in merged],
                target_cls=[g[0] for g in gts],
                target_boxes=[list(g[1:5]) for g in gts]))
            records.append((path.stem, n_tiles, merged, gts))
        return apmetrics.score(stats, self.names), records


def check_end_to_end(ch: Checks, coco: dict, tmp: Path) -> None:
    data_yaml, names = render_fixture(coco, tmp / "rendered")
    runner = _StubbedRun(data_yaml, tmp, names)

    # C0 the contract the stub depends on, and that the batching must keep:
    # every tile predicted exactly once, in grid order, whatever the batch
    # size. The second half matters for the scoring, not the inference -- B1
    # showed that contention between two detections for one label is resolved
    # by list order, so merged detections must leave this path sorted by
    # descending confidence.
    oracle = ScriptedPredictor()
    report, records = runner.run(oracle, tile=256, overlap=0.25, tile_batch=3)
    n_tiles = sum(r[1] for r in records)
    ch.check(runner.order_ok and runner.sorted_ok,
             "C0 every tile is predicted once, in order; output is conf-sorted",
             f"{n_tiles} tiles over {len(records)} images, batches of 3")

    # C1 an oracle detector scores the ceiling through the whole path: grid,
    # crop, map back, merge, match against whole-image ground truth. Anything
    # below 0.995 is this code losing or displacing a detection.
    ch.check(abs(report["pooled"]["mAP50"] - 0.995) < TOL
             and all(abs(v["AP50"] - 0.995) < TOL
                     for v in report["per_class"].values()),
             "C1 oracle detections score the 0.995 ceiling over the fixture",
             f"{len(records)} images, {n_tiles} tiles, "
             f"classes {sorted(report['per_class'])}")

    # C2 batching changes nothing: one tile per forward pass or eight.
    r1, rec1 = runner.run(ScriptedPredictor(), tile=256, overlap=0.25, tile_batch=1)
    r8, rec8 = runner.run(ScriptedPredictor(), tile=256, overlap=0.25, tile_batch=8)
    same = (json.dumps(r1, sort_keys=True) == json.dumps(r8, sort_keys=True)
            and [r[2] for r in rec1] == [r[2] for r in rec8])
    ch.check(same, "C2 --tile-batch changes no detection and no number",
             "1 vs 8 tiles per forward pass")

    # C3 images with no labels -- the fixture's genuine negative, plus the two
    # whose boxes voc_to_coco drops upstream (a zero-width box and an unknown
    # label). Each is still tiled, and the oracle finds nothing on it, so it
    # contributes neither a false positive nor a missed box.
    neg = [r for r in records if not r[3]]
    ch.check(len(neg) == 3 and all(r[2] == [] and r[1] >= 1 for r in neg),
             "C3 an image with no boxes yields tiles and no detections",
             ", ".join(f"{r[0]} ({r[1]} tiles)" for r in neg))

    # C4 an image below the threshold passes through whole, exactly as Tier A
    # sent it: one tile, no crop, no merge.
    r_whole, rec_whole = runner.run(ScriptedPredictor(), tile=640, overlap=0.2,
                                    tile_above=1200)
    ch.check(all(r[1] == 1 for r in rec_whole)
             and abs(r_whole["pooled"]["mAP50"] - 0.995) < TOL,
             "C4 every fixture image passes through whole above the threshold",
             "600x600 against tile_above 1200: one tile each, same 0.995")

    # C5 a detector that finds nothing scores zero everywhere, and every class
    # with labels is still listed -- a silent empty report is the failure mode
    # that would look like a working run.
    empty = EmptyPredictor()
    r_empty, _ = runner.run(empty, tile=256, overlap=0.25)
    ch.check(r_empty["pooled"]["mAP50"] == 0.0
             and set(r_empty["per_class"]) == set(report["per_class"])
             and empty.n > 0,
             "C5 a detector that finds nothing scores 0.0 with classes intact",
             f"{empty.n} crops predicted, {len(r_empty['per_class'])} classes listed")

    # C6 the dump round trip: scoring a dump must give the same numbers as
    # scoring in-process, or the two-stage mode is not the same evaluation.
    dump = tmp / "dets.jsonl"
    with dump.open("w", encoding="utf-8") as fh:
        for stem, n_t, merged, gts in records:
            fh.write(json.dumps({
                "stem": stem, "w": 600, "h": 600, "n_tiles": n_t,
                "dets": [[m[0], round(m[1], 6), *(round(v, 2) for v in m[2:])]
                         for m in merged],
                "gt": [[g[0], *(round(v, 2) for v in g[1:5])] for g in gts],
            }) + "\n")
    re_scored = apmetrics.score(tiled_predict.load_dump(dump), names)
    ch.check(json.dumps(re_scored, sort_keys=True)
             == json.dumps(report, sort_keys=True),
             "C6 --score-only on a dump reproduces the in-process numbers")

    # C7 the per-country crossing partitions the images and nothing is lost.
    stats = tiled_predict.load_dump(dump)
    per_country, crossed = apmetrics.score_by_country(stats, names)
    ch.check(sum(v["n_images"] for v in per_country.values()) == len(stats)
             and set(per_country) == {"Japan", "Czech"}
             and set(crossed) == set(per_country),
             "C7 per-country blocks partition the images",
             f"{ {k: v['n_images'] for k, v in per_country.items()} }")


# ---------------------------------------------------------------------------
# D. the seam, at Norway's scale
# ---------------------------------------------------------------------------
def check_seam(ch: Checks) -> None:
    """A crack cut by a seam, scored. This is what the stitch is for.

    ``tile_roundtrip`` check E establishes the geometry: plain NMS leaves two
    half-cracks because their mutual IoU is low. What it cannot show is the
    consequence for the reported number, which is here: the same detections
    score 1.0 stitched and 0.995 under NMS, the difference being the surviving
    half-crack counted as a false positive.
    """
    w, h = NORWAY
    tiles = tile_grid(w, h, TILE, OVERLAP, ABOVE)
    # A 1,000 px longitudinal crack crossing the seam between tile 0 (x 0-1280)
    # and tile 1 (x 1024-2304): each tile sees a piece, and the pieces overlap
    # by only 256 px of a 1,000 px box.
    crack = (600.0, 400.0, 1600.0, 460.0)
    ch.check(len(tiles) == 8, "D1 a Norway frame at 1280/0.2 gives 8 tiles",
             f"{len(tiles)} tiles, "
             f"{len({t.col for t in tiles})}x{len({t.row for t in tiles})}")

    # Scores differ so the ordering under NMS is deterministic: the right-hand
    # piece, which overlaps the truth more, ranks first.
    def score_for(cls, box, tile):
        return 0.95 if tile.index == 1 else 0.90

    img = np.zeros((h, w, 3), dtype=np.uint8)
    out = {}
    for mode in ("stitch", "nms"):
        stub = ScriptedPredictor(score_for=score_for)
        stub.begin(tiles, [(0, crack)])
        merged, n_tiles = tiled_predict.predict_image(
            img, w, h, stub, tile=TILE, overlap=OVERLAP, tile_above=ABOVE,
            merge_mode=mode, tile_batch=8, merge_kw={})
        st = ImageStats(
            stem="Norway_000001", country="Norway",
            pred_cls=[m[0] for m in merged], conf=[m[1] for m in merged],
            pred_boxes=[list(m[2:]) for m in merged],
            target_cls=[0], target_boxes=[list(crack)])
        out[mode] = (merged, apmetrics.score([st], {0: "C1"}))
    del img

    stitched, r_stitch = out["stitch"]
    nmsed, r_nms = out["nms"]
    one_box = [m for m in stitched if m[0] == 0]
    recovered = (len(one_box) == 1
                 and all(abs(a - b) < 1e-6 for a, b in zip(one_box[0][2:], crack)))
    ch.check(recovered,
             "D2 the stitch recovers the cut crack to its exact coordinates",
             f"{len(one_box)} box {tuple(round(v, 1) for v in one_box[0][2:])}"
             if one_box else "none")
    s50, s95 = (r_stitch["per_class"]["C1"]["AP50"],
                r_stitch["per_class"]["C1"]["AP50_95"])
    n50, n95 = (r_nms["per_class"]["C1"]["AP50"],
                r_nms["per_class"]["C1"]["AP50_95"])
    ch.check(abs(s50 - 0.995) < TOL and abs(s95 - 0.995) < TOL,
             "D3 the stitched detection is a true positive at every IoU",
             f"AP50 {s50:.4f}, AP50_95 {s95:.4f} -- the box is exact, so the "
             f"strict columns hold too")
    # At IoU 0.5 the better half alone still clears the threshold, so AP50
    # cannot tell the two modes apart; the cost of NMS here is localisation,
    # which is exactly what AP50_95 measures, plus a second box that is a
    # false positive however it is scored.
    ch.check(len(nmsed) == 2 and len(stitched) == 1 and n95 < s95 and n50 <= s50,
             "D4 under NMS the half-crack survives and costs AP50_95",
             f"NMS: {len(nmsed)} boxes, AP50 {n50:.4f}, AP50_95 {n95:.4f}; "
             f"stitched: 1 box, {s50:.4f} / {s95:.4f}")

    # D5 a box ending exactly on a seam is not stitched across it: the two
    # tiles' pieces abut but neither is cut, so they stay separate objects.
    seam_x = float(tiles[1].x0)
    left = (seam_x - 300.0, 100.0, seam_x, 160.0)
    stub = ScriptedPredictor()
    stub.begin(tiles, [(0, left)])
    merged, _ = tiled_predict.predict_image(
        np.zeros((h, w, 3), dtype=np.uint8), w, h, stub, tile=TILE, overlap=OVERLAP,
        tile_above=ABOVE, merge_mode="stitch", tile_batch=8, merge_kw={})
    ch.check(len(merged) == 1
             and all(abs(a - b) < 1e-6 for a, b in zip(merged[0][2:], left)),
             "D5 a box ending exactly on a seam round-trips unchanged",
             f"{len(merged)} box {tuple(round(v, 1) for v in merged[0][2:])}")

    # D6 an image below the threshold is one whole-image tile even at Norway's
    # own size, so --tile-above governs and nothing is cropped.
    ch.check(len(tile_grid(w, h, TILE, OVERLAP, 10 ** 9)) == 1,
             "D6 --untiled collapses the grid to one whole-image tile")


# ---------------------------------------------------------------------------
# E. the validator's preprocessing contract
# ---------------------------------------------------------------------------
def check_canvas(ch: Checks) -> None:
    """The canvas rule, guarded where no checkpoint and no dataset are needed.

    This is the rule that silently broke the first version of the evaluation
    path: it letterboxed to imgsz x imgsz, which is the obvious thing and not
    what validation does. The gate caught it, but the gate needs the images, so
    the rule is pinned here too.
    """
    from rdd.eval.tiled_predict import canvas_shape

    # Every square size in RDD2022 (512 to 1080) and every tile of the front
    # end lands on the same 672x672 canvas: 640 plus a 16 px border a side.
    squares = {canvas_shape(s, s) for s in (512, 540, 600, 640, 720, 1024,
                                            1080, 1280)}
    ch.check(squares == {(672, 672)},
             "E1 a square image goes in on 672x672, not 640x640",
             f"{sorted(squares)} -- pad=0.5 of a 32 stride, each side")

    # Norway's three frame sizes, which are the only non-square ones.
    norway = {(2035, 4040): (352, 672), (2041, 3643): (384, 672),
              (2044, 3650): (384, 672)}
    got = {hw: canvas_shape(*hw) for hw in norway}
    ch.check(got == norway, "E2 Norway's whole frames go in on a wide canvas",
             ", ".join(f"{w}x{h} -> {got[(h, w)][1]}x{got[(h, w)][0]}"
                       for (h, w) in sorted(norway)))

    ch.check(canvas_shape(2035, 4040, rect=False) == (640, 640),
             "E3 --square-canvas turns the rectangular canvas off",
             "kept for ablation; not Tier A comparable")

    # A tall image is the mirror case, and the formula must not favour width.
    ch.check(canvas_shape(4040, 2035) == (672, 352),
             "E4 a portrait frame is the transpose of a landscape one")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path,
                    help="fixture root (built if absent)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the rendered images and the dump")
    args = ap.parse_args(argv)

    coco = load_fixture(args.root)
    print(f"fixture {args.root}")
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} boxes")
    print(f"  tiled evaluation config: tile={TILE} overlap={OVERLAP} "
          f"tile_above={ABOVE}, Norway frame {NORWAY[0]}x{NORWAY[1]}")

    ch = Checks()
    tmpdir = Path(tempfile.mkdtemp(prefix="tiled_eval_")) if not args.keep \
        else args.root / "_tiled_eval"
    tmpdir.mkdir(parents=True, exist_ok=True)

    sections = [
        ("A. average precision, against hand-computable cases",
         lambda: check_ap(ch)),
        ("A. average precision, against the other implementations",
         lambda: (check_against_ultralytics(ch), check_against_pycocotools(ch))),
        ("B. per-image matching", lambda: check_matching(ch)),
        ("C. the whole path, on a rendered copy of the fixture",
         lambda: check_end_to_end(ch, coco, tmpdir)),
        ("D. the seam, at Norway's scale", lambda: check_seam(ch)),
        ("E. the validator's preprocessing contract",
         lambda: check_canvas(ch)),
    ]
    for title, run in sections:
        at = len(ch.rows)
        run()
        print(f"\n{title}")
        for ok, name, detail in ch.rows[at:]:
            tag = "[ok]  " if ok else ("[FAIL]" if ok is False else "[note]")
            print(f"  {tag} {name:<58} {detail}")

    if not args.keep:
        import shutil
        with contextlib.redirect_stderr(io.StringIO()):
            shutil.rmtree(tmpdir, ignore_errors=True)

    n_checks = sum(1 for r in ch.rows if r[0] is not None)
    print("\n" + "=" * 78)
    if ch.failed:
        print(f"FAILED {len(ch.failed)} of {n_checks} checks:")
        for _, name, detail in ch.failed:
            print(f"  {name} {detail}")
        return 1
    print(f"all {n_checks} checks passed")
    print("No checkpoint was loaded: the detector is a stub, so every number "
          "here is this code's doing.")
    print("The remaining gate -- reproducing tierA_base6_s0's recorded "
          "per-class AP50 -- needs the validation images; see "
          "rdd.eval.tiled_predict --untiled --expect-from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
