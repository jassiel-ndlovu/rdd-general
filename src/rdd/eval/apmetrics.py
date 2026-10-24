"""Average precision over detection lists this project produced itself.

Why this module exists
----------------------
Tier B's resolution front end (US15) runs the detector on overlapping tiles and
merges the per-tile detections back to full-image coordinates with
``rdd.data.tiling.merge_detections``. Those merged detections do not come out of
Ultralytics' dataloader, so ``model.val()`` cannot score them: it only ever
scores what its own validator produced. The score therefore has to be computed
here, from a detection list and a ground-truth list.

That raises the question this module is really about: **which** average
precision. Tier A's recorded numbers (``tierA_base6_s0`` per-class AP50 0.5851,
0.5431, 0.6845, 0.5073, 0.7181, 0.8406) were produced by Ultralytics
8.4.131's own estimator. pycocotools would compute a *different* number from the
very same detections -- different matching, different interpolation, a 100
detection-per-image cap -- typically a few points apart. A Tier B figure
computed by a different estimator than Tier A's is not comparable with it, and
comparability is the entire purpose of the tiled evaluation path. So the
reported estimator is a faithful port of Ultralytics' own, and pycocotools is
offered beside it as an independent cross-check, never as the headline number.

The port is exact, not approximate. It reproduces, from
``ultralytics==8.4.131``:

* ``utils/metrics.box_iou``          -- including the ``+1e-7`` in the
  denominator, which moves IoU in the last decimal places and therefore can
  move a match across the 0.5 threshold;
* ``engine/validator.match_predictions`` (``use_scipy=False``, the default) --
  wrong-class pairs zeroed first, then an assignment that is *not* the greedy
  one it looks like. The pairs are sorted by descending IoU, but the
  ``np.unique`` that de-duplicates detections re-sorts them into detection
  order, so only the first de-duplication is decided by IoU (which label a
  detection is credited against) while the second -- which of two detections
  keeps a contested label -- falls to position in the detection list. That list
  arrives sorted by descending confidence, from NMS and from
  ``merge_detections`` alike, so a contested label goes to the more confident
  detection even when the other fits better. Run once per IoU threshold;
* ``utils/metrics.compute_ap``       -- sentinel points, the precision envelope,
  and 101-point interpolation integrated with the trapezoid rule. Note this is
  *not* the mean of 101 interpolated points that COCO's own definition uses;
  the trapezoid of the same 101 points differs slightly, and matching Tier A
  means matching what Tier A ran;
* ``utils/metrics.ap_per_class``     -- 1000-point precision/recall curves, the
  operating point taken at the maximum of a box-filtered mean F1 curve, and
  ``unique_classes`` from the *ground truth*, so a class with no annotations in
  the subset under evaluation is omitted rather than reported as zero (the
  ``ap_class_index`` behaviour ``rdd.eval.revalidate.per_class_from`` documents).

``rdd.verify.tiled_eval`` asserts the equivalence numerically against the
installed Ultralytics where there is one, and against hand-computable cases
where there is not.

Everything here is pure numpy over small arrays. No torch, no GPU, no images,
so it runs on a laptop and in CI.
"""

from __future__ import annotations

import numpy as np

# Ultralytics' validator IoU vector for mAP@0.5:0.95.
IOUV = np.linspace(0.5, 0.95, 10)

# The eps values are Ultralytics'. They are not cosmetic: _IOU_EPS shifts IoU in
# the last decimals and can decide a borderline match, and _AP_EPS is the
# divisor guard in the recall curve.
_IOU_EPS = 1e-7
_AP_EPS = 1e-16


# ---------------------------------------------------------------------------
# Per-image matching
# ---------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, (N,4) x (M,4) -> (N,M), xyxy.

    Port of ``ultralytics.utils.metrics.box_iou``, including two details that
    look like noise and are not:

    * the ``+_IOU_EPS`` in the denominator, and
    * **float32**. Ultralytics' version calls ``.float()`` on its inputs, with
      the comment "Need .float() to get accurate iou values", so the whole
      computation runs in single precision. Computing it in float64 instead
      shifts IoU by ~2e-7, which is enough to move a box sitting exactly on an
      IoU threshold across it -- and boxes on exact thresholds are not
      hypothetical when annotations are whole pixels.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    area_b = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (area_a + area_b - inter + np.float32(_IOU_EPS))


def match_predictions(pred_cls: np.ndarray, true_cls: np.ndarray,
                      iou: np.ndarray, iouv: np.ndarray = IOUV) -> np.ndarray:
    """(N_pred, len(iouv)) bool TP matrix for one image.

    Port of ``BaseValidator.match_predictions`` with ``use_scipy=False``.
    ``iou`` is (N_true, N_pred), the orientation ``box_iou(gt, pred)`` returns.

    The assignment reads as "greedy by IoU" and is not quite that, which is the
    main reason this is a port rather than a reimplementation. Pairs above the
    threshold are sorted by descending IoU, and the first ``np.unique`` keeps
    one pair per detection -- so each detection does take its best-IoU label.
    But that ``np.unique`` returns indices in sorted *detection* order, which
    destroys the IoU ordering, so the second ``np.unique`` -- one pair per
    label -- keeps whichever detection comes first in the list. The list is in
    descending confidence, so a contested label goes to the more confident
    detection, not the better-fitting one.

    Two consequences worth knowing. Detections must reach this function in
    descending confidence, which ``merge_detections`` guarantees. And this is
    not the same rule as ``rdd.eval.revalidate.match``, which is greedy by
    confidence throughout and exists for the false-negative dump, not for AP.
    """
    n_pred = int(pred_cls.shape[0])
    correct = np.zeros((n_pred, len(iouv)), dtype=bool)
    if n_pred == 0 or true_cls.shape[0] == 0:
        return correct

    # Zero out pairs whose classes disagree, before thresholding.
    iou = np.asarray(iou, dtype=np.float64) * (
        np.asarray(true_cls).reshape(-1, 1) == np.asarray(pred_cls).reshape(1, -1))

    for i, threshold in enumerate(np.asarray(iouv, dtype=np.float64).tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T     # (K, 2): gt, pred
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


# ---------------------------------------------------------------------------
# Average precision
# ---------------------------------------------------------------------------
def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f. Port of ``ultralytics.utils.metrics.smooth``."""
    nf = round(len(y) * f * 2) // 2 + 1          # must be odd
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def _trapezoid(y, x):
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP from one recall/precision curve. Port of ``compute_ap``, method="interp".

    The sentinels matter. ``recall`` is extended with its own last value and
    then 1.0, and ``precision`` with 0.0 twice, so the envelope falls to zero
    past the last detection instead of being extrapolated flat -- the difference
    between a conservative AP and an optimistic one for a class whose recall
    curve stops well short of 1.
    """
    recall = np.asarray(recall, dtype=np.float64).ravel()
    precision = np.asarray(precision, dtype=np.float64).ravel()
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))     # precision envelope
    x = np.linspace(0, 1, 101)                               # 101-point interp
    return float(_trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                 target_cls: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class AP, precision and recall. Port of ``ap_per_class``.

    Returns the five arrays this project reports, all indexed by
    ``ap_class_index``:

        ap        (C, len(IOUV))  AP at each IoU threshold
        p, r, f1  (C,)            at the maximum of the smoothed mean F1 curve
        ap_class_index (C,)       the classes that have ground truth

    A class present in ``pred_cls`` but absent from ``target_cls`` does not
    appear at all: there is no recall to measure, and reporting 0.0 for it would
    read as "the detector failed on this class" rather than "this subset has
    none of it".
    """
    tp = np.asarray(tp, dtype=np.float64).reshape(len(conf), -1) if len(conf) \
        else np.zeros((0, len(IOUV)))
    conf = np.asarray(conf, dtype=np.float64).ravel()
    pred_cls = np.asarray(pred_cls).ravel()
    target_cls = np.asarray(target_cls).ravel()

    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    n_iou = tp.shape[1]

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, n_iou))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = int(i.sum())
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + _AP_EPS)
        # Negative x and xp because confidence decreases along the curve.
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        for j in range(n_iou):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + _AP_EPS)
    # The reported operating point: the confidence at which the mean F1 over
    # classes peaks, found on a box-filtered curve so that a single noisy
    # confidence bin cannot claim it.
    i = _smooth(f1_curve.mean(0), 0.1).argmax() if nc else 0
    return {
        "ap": ap,
        "p": p_curve[:, i] if nc else np.zeros(0),
        "r": r_curve[:, i] if nc else np.zeros(0),
        "f1": f1_curve[:, i] if nc else np.zeros(0),
        "ap_class_index": unique_classes.astype(int),
        "nt": nt,
    }


# ---------------------------------------------------------------------------
# The shape the project's JSON wants
# ---------------------------------------------------------------------------
class ImageStats:
    """One image's contribution to a score: the arrays, not the pixels.

    Held per image so that a subset (one country, say) can be scored by
    concatenating a selection of these rather than by running inference again.
    Each instance is a few hundred bytes; 5,398 of them are a few megabytes,
    which is what makes the per-class-by-country crossing free once inference
    has run.
    """

    __slots__ = ("stem", "country", "conf", "pred_cls", "pred_boxes",
                 "target_cls", "target_boxes", "tp")

    def __init__(self, stem: str, country: str,
                 pred_cls: np.ndarray, conf: np.ndarray, pred_boxes: np.ndarray,
                 target_cls: np.ndarray, target_boxes: np.ndarray) -> None:
        self.stem = stem
        self.country = country
        self.pred_cls = np.asarray(pred_cls, dtype=np.int64).ravel()
        self.conf = np.asarray(conf, dtype=np.float64).ravel()
        self.pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        self.target_cls = np.asarray(target_cls, dtype=np.int64).ravel()
        self.target_boxes = np.asarray(target_boxes, dtype=np.float64).reshape(-1, 4)
        self.tp = match_predictions(
            self.pred_cls, self.target_cls,
            box_iou(self.target_boxes, self.pred_boxes)
            if len(self.target_boxes) and len(self.pred_boxes)
            else np.zeros((len(self.target_boxes), len(self.pred_boxes))))


def score(stats: list[ImageStats], names: dict[int, str]) -> dict:
    """Pooled and per-class metrics over a list of per-image stats.

    The JSON shape is ``rdd.eval.revalidate``'s, so that
    ``rdd.eval.aggregate`` consumes it with no change:

        {"pooled": {mAP50, mAP50_95, precision, recall},
         "per_class": {name: {AP50, AP50_95, precision, recall}}}
    """
    if not stats:
        return {"pooled": {"mAP50": 0.0, "mAP50_95": 0.0,
                           "precision": 0.0, "recall": 0.0},
                "per_class": {}, "n_images": 0}

    tp = np.concatenate([s.tp for s in stats], 0) if any(len(s.tp) for s in stats) \
        else np.zeros((0, len(IOUV)), dtype=bool)
    conf = np.concatenate([s.conf for s in stats], 0)
    pred_cls = np.concatenate([s.pred_cls for s in stats], 0)
    target_cls = np.concatenate([s.target_cls for s in stats], 0)

    res = ap_per_class(tp, conf, pred_cls, target_cls)
    ap, p, r, idx = res["ap"], res["p"], res["r"], res["ap_class_index"]

    per_class = {}
    for i, ci in enumerate(idx):
        per_class[names.get(int(ci), str(int(ci)))] = {
            "AP50": round(float(ap[i, 0]), 4),
            "AP50_95": round(float(ap[i].mean()), 4),
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "n_instances": int(res["nt"][i]),
        }

    pooled = {
        "mAP50": round(float(ap[:, 0].mean()) if len(ap) else 0.0, 4),
        "mAP50_95": round(float(ap.mean()) if len(ap) else 0.0, 4),
        "precision": round(float(p.mean()) if len(p) else 0.0, 4),
        "recall": round(float(r.mean()) if len(r) else 0.0, 4),
    }
    return {"pooled": pooled, "per_class": per_class, "n_images": len(stats)}


def score_by_country(stats: list[ImageStats], names: dict[int, str]) -> tuple[dict, dict]:
    """(per_country, per_class_by_country), in ``revalidate``'s shape.

    ``per_country`` carries the pooled block plus ``f1``, matching
    ``rdd.train.train_yolo.per_country_eval``; the per-class block is split out
    into ``per_class_by_country`` exactly as ``revalidate.stage_metrics`` splits
    it, so the aggregator sees the structure it already reads.
    """
    groups: dict[str, list[ImageStats]] = {}
    for s in stats:
        groups.setdefault(s.country, []).append(s)

    per_country, crossed = {}, {}
    for country, group in sorted(groups.items()):
        block = score(group, names)
        pooled = block["pooled"]
        mp, mr = pooled["precision"], pooled["recall"]
        per_country[country] = {
            "n_images": len(group),
            **pooled,
            "f1": round(2 * mp * mr / (mp + mr), 4) if (mp + mr) else 0.0,
        }
        crossed[country] = block["per_class"]
    return per_country, crossed


# ---------------------------------------------------------------------------
# Independent cross-check: pycocotools
# ---------------------------------------------------------------------------
def score_pycocotools(stats: list[ImageStats], names: dict[int, str]) -> dict:
    """The same detections scored by pycocotools, as a cross-check.

    This is deliberately NOT the reported number -- see the module docstring.
    COCOeval differs from Ultralytics in at least three ways that each move AP
    by a fraction of a point: matching is greedy by descending score rather than
    by descending IoU, AP is the mean of 101 interpolated precisions rather than
    their trapezoid, and ``maxDets`` caps each image at 100 detections. Agreement
    to a point or two is the most that should be expected; a large disagreement
    means something is wrong with the detections or the ground truth, which is
    what makes it worth computing.

    Raises ImportError if pycocotools is absent (it is pinned in
    requirements.txt, so it is present on the instance and in CI).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat_ids = sorted(names)
    gt = {
        "info": {}, "licenses": [],
        "images": [], "annotations": [],
        "categories": [{"id": c, "name": names[c]} for c in cat_ids],
    }
    dets = []
    ann_id = 1
    for i, s in enumerate(stats, start=1):
        # Width/height are only used by COCOeval for area ranges, which this
        # report does not break down, so the image box is sized to the data.
        gt["images"].append({"id": i, "file_name": f"{s.stem}.jpg",
                             "width": 1, "height": 1})
        for cls, (x0, y0, x1, y1) in zip(s.target_cls.tolist(),
                                         s.target_boxes.tolist()):
            w, h = x1 - x0, y1 - y0
            gt["annotations"].append({
                "id": ann_id, "image_id": i, "category_id": int(cls),
                "bbox": [x0, y0, w, h], "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1
        for cls, sc, (x0, y0, x1, y1) in zip(s.pred_cls.tolist(),
                                             s.conf.tolist(),
                                             s.pred_boxes.tolist()):
            dets.append({"image_id": i, "category_id": int(cls),
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(sc)})

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        if not dets:
            return {"per_class": {}, "pooled": {"mAP50": 0.0, "mAP50_95": 0.0},
                    "note": "no detections"}
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: (T=10 IoU, R=101 recall, K=classes, A=areas, M=maxDets)
    prec = ev.eval["precision"]
    present = {int(c) for s in stats for c in s.target_cls.tolist()}
    per_class = {}
    for k, c in enumerate(cat_ids):
        if c not in present:
            continue                     # same omission rule as the port
        p50 = prec[0, :, k, 0, 2]
        pall = prec[:, :, k, 0, 2]
        per_class[names[c]] = {
            "AP50": round(float(p50[p50 > -1].mean()) if (p50 > -1).any() else 0.0, 4),
            "AP50_95": round(float(pall[pall > -1].mean()) if (pall > -1).any() else 0.0, 4),
        }
    vals50 = [v["AP50"] for v in per_class.values()]
    vals95 = [v["AP50_95"] for v in per_class.values()]
    return {
        "per_class": per_class,
        "pooled": {
            "mAP50": round(sum(vals50) / len(vals50), 4) if vals50 else 0.0,
            "mAP50_95": round(sum(vals95) / len(vals95), 4) if vals95 else 0.0,
        },
    }
