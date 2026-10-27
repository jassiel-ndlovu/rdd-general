"""
Overlapping-tile resolution front end: geometry and merging (US15, Tier B).

RDD2022 mixes 512x512 frames with Norway's 3650x2044 frames and training
resizes everything to 640 px, so a Norwegian hairline crack three pixels wide
arrives at the network as half a pixel. Norway scores mAP50 0.244 against 0.902
for the best subset, and US14a found the C4 failure to be missed detections
rather than confusions, which points at resolution handling.

This module is the geometry half of that front end: the tile grid, the box
clipping, and the merge of per-tile detections back to full-image coordinates.
It works from image dimensions and box coordinates only, so it runs against the
canonical COCO store with no images, no GPU and no training. Cropping pixels is
in rdd.data.to_yolo_tiled. The CLI also measures k, the ratio of tiled to
untiled samples, which the Tier B plan uses for budgeting.

Three design decisions.

1. Edge tiles shift inward rather than being padded or emitted at a smaller
   size. Padding introduces a training artefact that does not occur at
   inference, and mixed tile sizes are letterboxed back to square by the
   trainer, reintroducing the resizing this exists to avoid. The price is that
   the last row and column overlap by more than requested, which inflates k
   slightly and is harmless to the merge. Images smaller than one tile pass
   through whole at native size, and tile_above sets the threshold above which
   an image is gridded at all.

2. Clipped boxes keeping less than min_fraction of their original area are
   dropped rather than kept as fragments. This matters most for C1 and C2,
   which are long and thin and straddle boundaries constantly: a grid that
   shreds a 900 px crack into five stubs teaches the model that C1 is a stub.
   The cost is that a dropped fragment leaves visible damage with no label, so
   the tile becomes a partial negative. tile_image reports dropped fragments
   per tile so a dataset builder can skip those tiles. The right value needs
   the US16 ablation on real images.

3. The merge joins boxes rather than only suppressing them. Plain NMS is not
   sufficient: an elongated crack cut by a seam produces two boxes that are
   each roughly half the object, abutting, with low mutual IoU, so NMS keeps
   both and the output has two half-cracks where the truth has one. The merge
   recognises three relations, class-wise and only across tiles. Duplicate,
   where mutual IoU >= iou_thr. Truncated view, where one box is cut off at an
   interior edge and covered by the other to contain_thr of its own area,
   measured as intersection over the smaller area rather than over union. And
   seam continuation, where both boxes run against facing interior edges,
   abut within gap_tol, and line up on the perpendicular axis within
   align_thr. Relations are transitive, each group becomes the bounding box of
   its members, and the group score is the maximum rather than the mean, since
   averaging would penalise an object for having been cut. Detections from the
   same tile are never joined. mode="nms" gives plain NMS instead, so US16 can
   ablate the stitch against the obvious baseline.

Known limitations. Two genuinely distinct collinear cracks meeting at a seam
fuse into one box, and in tile-local coordinates the geometry cannot tell that
from one crack cut in two. Relation 2 can fuse a small same-class box nested
inside a larger one when the small one touches a tile edge. An object crossing
a tile corner appears in up to four tiles and relies on the intermediate
detections to join transitively. The union of a curved crack's fragments
over-covers it, as any axis-aligned box must. No threshold here has been tuned
against real detections, only exercised on synthetic geometry in
rdd.verify.tile_roundtrip.

Usage
-----
    python -m rdd.data.tiling \\
        --coco /opt/rdd/data/coco/rdd2022_train.json \\
        --tile 640 --overlap 0.2 --min-box-fraction 0.3 \\
        --out  /opt/rdd/data/tiling/plan_t640_o20.json

    # just the multiplier k, without the per-image detail
    python -m rdd.data.tiling --coco ... --out ... --summary-only
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

TILE_DEFAULT = 640
OVERLAP_DEFAULT = 0.2
MIN_FRACTION_DEFAULT = 0.3


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tile:
    """A crop rectangle in full-image pixel coordinates, half-open on x1/y1."""

    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets along one axis.

    The last start is pulled back to ``length - tile`` so the final tile ends
    exactly on the image boundary, see decision 1 in the module docstring. It
    is appended only when the regular stride has not already covered the axis,
    so a grid that divides exactly gains no redundant tile.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_grid(width: int, height: int, tile: int = TILE_DEFAULT,
              overlap: float = OVERLAP_DEFAULT,
              tile_above: int | None = None) -> list[Tile]:
    """Tile grid for one image.

    ``overlap`` is a fraction of the tile size, so 0.2 at tile 640 strides
    512 px. ``tile_above`` is the largest side an image may have and still be
    passed through as a single whole-image tile; it defaults to ``tile``, i.e.
    grid anything that does not already fit inside one tile.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")

    threshold = tile if tile_above is None else tile_above
    if max(width, height) <= threshold:
        # Smaller than one tile (or below the "worth tiling" threshold): one
        # tile equal to the image. Native size, no pad, no upscale.
        return [Tile(0, 0, 0, 0, 0, width, height)]

    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = _axis_starts(width, tile, stride)
    ys = _axis_starts(height, tile, stride)

    tiles: list[Tile] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            # min() only bites on an axis shorter than the tile; on a gridded
            # axis the shifted last start guarantees x + tile <= width.
            tiles.append(Tile(len(tiles), row, col,
                              x, y, min(x + tile, width), min(y + tile, height)))
    return tiles


# ---------------------------------------------------------------------------
# Box clipping
# ---------------------------------------------------------------------------
Box = tuple[float, float, float, float]       # x0, y0, x1, y1


@dataclass(frozen=True)
class ClippedBox:
    """A ground-truth box clipped into one tile, in TILE-LOCAL coordinates."""

    cls: int
    x0: float
    y0: float
    x1: float
    y1: float
    retained: float      # clipped area / original area, in (0, 1]
    source: int          # index of the box this came from, in the caller's list

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def clip_box_to_tile(box: Box, tile: Tile
                     ) -> tuple[float, float, float, float, float] | None:
    """Intersect ``box`` with ``tile``.

    Returns ``(x0, y0, x1, y1, retained)`` in tile-local coordinates, or None if
    the box does not intersect the tile's interior.

    Tiles are half-open on x1/y1, so a box whose right edge lands exactly on a
    seam belongs to the tile on whose interior it lies and is not also handed to
    the neighbour as a zero-width sliver. A degenerate box (zero or negative
    area) yields None rather than a division by zero; ``rdd.data.voc_to_coco``
    already drops those upstream, but nothing here should depend on that.
    """
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return None

    ix0 = max(box[0], tile.x0)
    iy0 = max(box[1], tile.y0)
    ix1 = min(box[2], tile.x1)
    iy1 = min(box[3], tile.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None

    retained = ((ix1 - ix0) * (iy1 - iy0)) / area
    # float() throughout: the tile bounds are ints and the box bounds are not,
    # so min()/max() would otherwise hand back a mix of the two.
    return (float(ix0 - tile.x0), float(iy0 - tile.y0),
            float(ix1 - tile.x0), float(iy1 - tile.y0), retained)


def clip_boxes(boxes: list[tuple[int, Box]], tile: Tile,
               min_fraction: float = MIN_FRACTION_DEFAULT
               ) -> tuple[list[ClippedBox], list[ClippedBox]]:
    """Clip ``(cls, box)`` pairs into ``tile``.

    Returns ``(kept, dropped)``. ``dropped`` are the fragments that do intersect
    the tile but retain less than ``min_fraction`` of their original area. They
    are returned rather than discarded because they mark tiles containing
    visible damage with no label (decision 2).
    """
    kept: list[ClippedBox] = []
    dropped: list[ClippedBox] = []
    for i, (cls, box) in enumerate(boxes):
        clipped = clip_box_to_tile(box, tile)
        if clipped is None:
            continue
        x0, y0, x1, y1, retained = clipped
        entry = ClippedBox(cls, x0, y0, x1, y1, retained, i)
        (kept if retained >= min_fraction else dropped).append(entry)
    return kept, dropped


@dataclass(frozen=True)
class TilePlan:
    """One tile plus the labels it would carry."""

    tile: Tile
    kept: list[ClippedBox]
    dropped: list[ClippedBox]


def tile_image(width: int, height: int, boxes: list[tuple[int, Box]],
               tile: int = TILE_DEFAULT, overlap: float = OVERLAP_DEFAULT,
               min_fraction: float = MIN_FRACTION_DEFAULT,
               tile_above: int | None = None) -> list[TilePlan]:
    """Full plan for one image: every tile, with its clipped labels."""
    return [
        TilePlan(t, *clip_boxes(boxes, t, min_fraction))
        for t in tile_grid(width, height, tile, overlap, tile_above)
    ]


# ---------------------------------------------------------------------------
# Detection merging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TileDetection:
    """A detection as the model emits it: TILE-LOCAL coordinates."""

    tile: int
    cls: int
    score: float
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class MergedDetection:
    """A detection in full-image coordinates, after merging."""

    cls: int
    score: float
    x0: float
    y0: float
    x1: float
    y1: float
    parts: tuple[int, ...]       # indices into the input detection list
    tiles: tuple[int, ...]       # tiles those parts came from

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def to_full_image(det: TileDetection, tile: Tile) -> Box:
    """Tile-local detection -> full-image coordinates."""
    return (det.x0 + tile.x0, det.y0 + tile.y0,
            det.x1 + tile.x0, det.y1 + tile.y0)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou(a: Box, b: Box) -> float:
    inter = _overlap(a[0], a[2], b[0], b[2]) * _overlap(a[1], a[3], b[1], b[3])
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def io_min(a: Box, b: Box) -> float:
    """Intersection over the *smaller* box's area.

    The right measure for "is this the truncated view of that object": a stub
    cut off at a tile edge is fully inside the neighbour's box, which IoU scores
    low and this scores 1.0. See relation 2 in the module docstring.
    """
    inter = _overlap(a[0], a[2], b[0], b[2]) * _overlap(a[1], a[3], b[1], b[3])
    if inter <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _truncated(box: Box, tile: Tile, side: str,
               width: int, height: int, edge_tol: float) -> bool:
    """Does ``box`` run up against ``tile``'s ``side`` edge, interior to the image?

    An edge that coincides with the image border is not a seam: a box reaching it
    is genuinely the end of the object, not a cut, and must not be stitched to
    anything across it.
    """
    if side == "right":
        return tile.x1 < width and box[2] >= tile.x1 - edge_tol
    if side == "left":
        return tile.x0 > 0 and box[0] <= tile.x0 + edge_tol
    if side == "bottom":
        return tile.y1 < height and box[3] >= tile.y1 - edge_tol
    if side == "top":
        return tile.y0 > 0 and box[1] <= tile.y0 + edge_tol
    raise ValueError(f"unknown side {side!r}")


def is_truncated(box: Box, tile: Tile, width: int, height: int,
                 edge_tol: float = 4.0) -> bool:
    """Is ``box`` cut off at any interior edge of its tile, i.e. a partial view?"""
    return any(_truncated(box, tile, side, width, height, edge_tol)
               for side in ("left", "right", "top", "bottom"))


def is_seam_continuation(a: Box, tile_a: Tile, b: Box, tile_b: Tile,
                         width: int, height: int, *,
                         align_thr: float = 0.5, gap_tol: float = 8.0,
                         edge_tol: float = 4.0) -> bool:
    """Are these two boxes one object cut by the seam between their tiles?

    Both must be cut off at the facing interior edges of their own tiles, must
    abut or overlap across the seam, and must line up on the perpendicular axis.
    Checked for a vertical seam (side-by-side boxes) and a horizontal one
    (stacked boxes); see decision 3 in the module docstring.
    """
    # --- vertical seam: boxes side by side, the damage runs across x ---------
    left, lt, right, rt = ((a, tile_a, b, tile_b) if a[0] <= b[0]
                           else (b, tile_b, a, tile_a))
    if (_truncated(left, lt, "right", width, height, edge_tol)
            and _truncated(right, rt, "left", width, height, edge_tol)):
        gap = right[0] - left[2]                    # negative where they overlap
        span = min(left[3] - left[1], right[3] - right[1])
        if gap <= gap_tol and span > 0:
            if _overlap(left[1], left[3], right[1], right[3]) / span >= align_thr:
                return True

    # --- horizontal seam: boxes stacked, the damage runs down y --------------
    top, tt, bot, bt = ((a, tile_a, b, tile_b) if a[1] <= b[1]
                        else (b, tile_b, a, tile_a))
    if (_truncated(top, tt, "bottom", width, height, edge_tol)
            and _truncated(bot, bt, "top", width, height, edge_tol)):
        gap = bot[1] - top[3]
        span = min(top[2] - top[0], bot[2] - bot[0])
        if gap <= gap_tol and span > 0:
            if _overlap(top[0], top[2], bot[0], bot[2]) / span >= align_thr:
                return True

    return False


def merge_detections(dets: list[TileDetection], tiles: list[Tile],
                     width: int, height: int, *, mode: str = "stitch",
                     iou_thr: float = 0.5, contain_thr: float = 0.7,
                     align_thr: float = 0.5, gap_tol: float = 8.0,
                     edge_tol: float = 4.0) -> list[MergedDetection]:
    """Map per-tile detections to full-image coordinates and de-duplicate.

    ``mode="stitch"`` (default) joins duplicates AND seam continuations into one
    box per group, the behaviour the elongated crack classes need.
    ``mode="nms"`` is plain class-wise greedy NMS over the mapped boxes, kept so
    the US16 ablation has a baseline. Both are class-wise: a C1 box is never
    merged with a C2 box, because orientation is definitional between those two.

    Output is sorted by descending score.
    """
    if mode not in ("stitch", "nms"):
        raise ValueError(f"mode must be 'stitch' or 'nms', got {mode!r}")

    by_index = {t.index: t for t in tiles}
    full: list[Box] = []
    for d in dets:
        t = by_index.get(d.tile)
        if t is None:
            raise KeyError(
                f"detection references tile {d.tile}, which is not in the grid")
        full.append(to_full_image(d, t))

    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    out: list[MergedDetection] = []

    if mode == "nms":
        suppressed = [False] * len(dets)
        for i in order:
            if suppressed[i]:
                continue
            for j in order:
                if j == i or suppressed[j] or dets[j].cls != dets[i].cls:
                    continue
                if iou(full[i], full[j]) >= iou_thr:
                    suppressed[j] = True
            out.append(MergedDetection(dets[i].cls, dets[i].score, *full[i],
                                       parts=(i,), tiles=(dets[i].tile,)))
        return out

    # --- stitch: union-find over duplicate and seam-continuation relations ---
    parent = list(range(len(dets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for i, d in enumerate(dets):
        by_class[d.cls].append(i)

    for members in by_class.values():
        for a in range(len(members)):
            i = members[a]
            for b in range(a + 1, len(members)):
                j = members[b]
                # Never join two detections from the same tile: the detector's
                # own NMS already ruled on those.
                if dets[i].tile == dets[j].tile:
                    continue
                ti, tj = by_index[dets[i].tile], by_index[dets[j].tile]
                # 1. duplicate
                if iou(full[i], full[j]) >= iou_thr:
                    union(i, j)
                    continue
                # 2. truncated view of an object the neighbour sees whole
                if io_min(full[i], full[j]) >= contain_thr and (
                        is_truncated(full[i], ti, width, height, edge_tol)
                        or is_truncated(full[j], tj, width, height, edge_tol)):
                    union(i, j)
                    continue
                # 3. one object cut by the seam between the two tiles
                if is_seam_continuation(full[i], ti, full[j], tj,
                                        width, height, align_thr=align_thr,
                                        gap_tol=gap_tol, edge_tol=edge_tol):
                    union(i, j)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(dets)):
        groups[find(i)].append(i)

    for members in groups.values():
        members.sort()
        x0 = min(full[i][0] for i in members)
        y0 = min(full[i][1] for i in members)
        x1 = max(full[i][2] for i in members)
        y1 = max(full[i][3] for i in members)
        # Max, not mean: a cut object is only partly visible in each tile and the
        # better-placed tile carries the less degraded estimate.
        score = max(dets[i].score for i in members)
        out.append(MergedDetection(
            dets[members[0]].cls, score,
            max(0.0, x0), max(0.0, y0),
            min(float(width), x1), min(float(height), y1),
            parts=tuple(members),
            tiles=tuple(sorted({dets[i].tile for i in members}))))

    out.sort(key=lambda m: -m.score)
    return out


# ---------------------------------------------------------------------------
# CLI: plan a tiling of the canonical store, and measure k
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan an overlapping-tile front end over the canonical COCO "
                    "store and measure the sample multiplier k. Reads geometry "
                    "only, no image is opened.")
    ap.add_argument("--coco", required=True, type=Path,
                    help="canonical COCO store, e.g. rdd2022_train.json")
    ap.add_argument("--out", required=True, type=Path, help="plan JSON to write")
    ap.add_argument("--tile", type=int, default=TILE_DEFAULT)
    ap.add_argument("--overlap", type=float, default=OVERLAP_DEFAULT,
                    help="fraction of the tile size (0.2 at tile 640 strides 512)")
    ap.add_argument("--min-box-fraction", type=float, default=MIN_FRACTION_DEFAULT,
                    help="drop a clipped box retaining less than this fraction of "
                         "its original area (default 0.3)")
    ap.add_argument("--tile-above", type=int, default=None,
                    help="only grid images whose larger side exceeds this many "
                         "pixels; smaller ones pass through whole. Defaults to "
                         "--tile. Raising it caps k to the high-resolution "
                         "subsets, see docs/13-tier-b-plan.md section 5.")
    ap.add_argument("--summary-only", action="store_true",
                    help="write the per-country summary without the per-image "
                         "tile detail (the detail is what the crop step will "
                         "consume, and it is large)")
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    anns_by_image: dict[int, list] = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    per_country: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    per_class_in: collections.Counter = collections.Counter()
    per_class_kept: collections.Counter = collections.Counter()
    per_class_dropped: collections.Counter = collections.Counter()
    skipped_no_size = 0
    plan_images = []

    for im in coco["images"]:
        W, H = int(im.get("width") or 0), int(im.get("height") or 0)
        if W <= 0 or H <= 0:
            # Without dimensions there is no geometry to plan. The store should
            # never contain these; count them rather than guess a size.
            skipped_no_size += 1
            continue

        boxes: list[tuple[int, Box]] = []
        for a in anns_by_image.get(im["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append((a["category_id"], (x, y, x + w, y + h)))
            per_class_in[a["category_id"]] += 1

        plans = tile_image(W, H, boxes, args.tile, args.overlap,
                           args.min_box_fraction, args.tile_above)

        country = im.get("country", "unknown")
        c = per_country[country]
        c["images"] += 1
        c["tiles"] += len(plans)
        c["boxes_in"] += len(boxes)
        if len(plans) > 1:
            c["gridded"] += 1
        for p in plans:
            c["boxes_kept"] += len(p.kept)
            c["fragments_dropped"] += len(p.dropped)
            if p.kept:
                c["tiles_with_labels"] += 1
            elif p.dropped:
                # Visible damage, no label: a tile a dataset builder may want to
                # exclude rather than present as a negative (decision 2).
                c["tiles_unlabelled_damage"] += 1
            else:
                c["tiles_negative"] += 1
            for b in p.kept:
                per_class_kept[b.cls] += 1
            for b in p.dropped:
                per_class_dropped[b.cls] += 1

        if not args.summary_only:
            plan_images.append({
                "image_id": im["id"],
                "file_name": im.get("file_name"),
                "rel_path": im.get("rel_path"),
                "country": country,
                "width": W,
                "height": H,
                "tiles": [
                    {
                        "index": p.tile.index,
                        "row": p.tile.row,
                        "col": p.tile.col,
                        "crop": p.tile.as_list(),
                        # Tile-local pixel coordinates; the YOLO export
                        # normalises per tile, as rdd.data.to_yolo does per image.
                        "boxes": [[b.cls] + [round(v, 2) for v in b.as_list()]
                                  for b in p.kept],
                        "dropped": len(p.dropped),
                    }
                    for p in plans
                ],
            })

    total: collections.Counter = collections.Counter()
    for c in per_country.values():
        total.update(c)
    k = total["tiles"] / max(total["images"], 1)

    report = {
        "source_coco": str(args.coco),
        "config": {
            "tile": args.tile,
            "overlap": args.overlap,
            "stride": max(1, int(round(args.tile * (1.0 - args.overlap)))),
            "min_box_fraction": args.min_box_fraction,
            "tile_above": args.tile_above if args.tile_above is not None else args.tile,
        },
        "totals": dict(total),
        "k": round(k, 3),
        "images_skipped_no_size": skipped_no_size,
        "per_country": {name: dict(c) for name, c in sorted(per_country.items())},
        "per_class": {
            cat_name.get(cid, str(cid)): {
                "boxes_in": per_class_in[cid],
                "tile_boxes_kept": per_class_kept.get(cid, 0),
                "fragments_dropped": per_class_dropped.get(cid, 0),
            }
            for cid in sorted(set(per_class_in) | set(per_class_kept)
                              | set(per_class_dropped))
        },
    }
    if not args.summary_only:
        report["images"] = plan_images

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2 if args.summary_only else None),
                        encoding="utf-8")

    # --- report -------------------------------------------------------------
    print("=" * 78)
    print(f"tile {args.tile}  overlap {args.overlap}  "
          f"stride {report['config']['stride']}  "
          f"min_box_fraction {args.min_box_fraction}  "
          f"tile_above {report['config']['tile_above']}")
    print("=" * 78)
    print(f"{'country':<16}{'images':>8}{'gridded':>9}{'tiles':>9}{'k':>7}"
          f"{'boxes in':>10}{'kept':>8}{'dropped':>9}")
    for name, c in sorted(per_country.items()):
        kc = c["tiles"] / max(c["images"], 1)
        print(f"{name:<16}{c['images']:>8}{c.get('gridded', 0):>9}{c['tiles']:>9}"
              f"{kc:>7.2f}{c['boxes_in']:>10}{c['boxes_kept']:>8}"
              f"{c.get('fragments_dropped', 0):>9}")
    print("-" * 78)
    print(f"{'TOTAL':<16}{total['images']:>8}{total.get('gridded', 0):>9}"
          f"{total['tiles']:>9}{k:>7.2f}{total['boxes_in']:>10}"
          f"{total['boxes_kept']:>8}{total.get('fragments_dropped', 0):>9}")
    print("=" * 78)
    print(f"  k = {k:.3f} tiled samples per source image.")
    print("  Step-matched training divides the epoch count by k "
          "(docs/13-tier-b-plan.md section 5).")
    print(f"  tiles with labels            {total.get('tiles_with_labels', 0):>9}")
    print(f"  tiles that are negatives     {total.get('tiles_negative', 0):>9}")
    print(f"  tiles with UNLABELLED damage {total.get('tiles_unlabelled_damage', 0):>9}"
          "  (every box in them fell below the area threshold)")
    if skipped_no_size:
        print(f"  images skipped, no size      {skipped_no_size:>9}")

    print("\n  per class:")
    print(f"    {'class':<26}{'boxes in':>10}{'tile boxes':>12}"
          f"{'fragments dropped':>20}")
    for name, st in report["per_class"].items():
        print(f"    {name:<26}{st['boxes_in']:>10}{st['tile_boxes_kept']:>12}"
              f"{st['fragments_dropped']:>20}")

    print(f"\nwrote {args.out}")
    if args.summary_only:
        print("  (--summary-only: no per-image tile detail written)")
    print("\nNOTHING WAS CROPPED. This is geometry only: no image was opened and")
    print("no tiled dataset was written. The crop step needs the image set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
