"""
Exercise the tiling front end end to end on the synthetic fixture (US15).

Why this is a verifier and not a handful of unit cases
-----------------------------------------------------
The three pieces of ``rdd.data.tiling``, grid, clip, merge, are each easy to
get right in isolation and easy to get wrong together. The interesting failures
are compositional: a grid whose last tile is shifted inward changes which boxes
straddle a seam, which changes which fragments the area threshold drops, which
changes whether the merge can reconnect them. So this runs the whole chain over
the fixture the repository already ships, at several tile sizes, and checks the
invariants that have to hold whatever the configuration:

  A. the grid covers the image exactly, with no tile out of bounds
  B. every kept box lies inside its tile and retains at least the threshold
  C. the merge is lossless, and a box whose fragments all survived clipping
     round-trips back to its original coordinates exactly
  D. the degenerate cases behave: a box exactly on a seam, a tile with no boxes
     at all, a zero-area box, an image smaller than one tile
  E. the Norway-scale geometry the fixture does not contain, including the case
     that motivates the stitch: one crack cut by a seam, where plain NMS leaves
     two half-cracks and the stitch recovers one

Sections A to D use the fixture, built by ``rdd.data._make_fixture`` and
converted by ``rdd.data.voc_to_coco``, so the boxes under test are the ones the
pipeline actually produces rather than ones written to suit the test. The
fixture's images are placeholder bytes, which costs nothing here: every check in
this file is geometric and no pixel is ever read.

Section E is synthetic because the fixture's images are all 600x600 and the
whole point of the front end is 3650x2044. Those cases are hand-built at
Norway's dimensions.

Usage
-----
    python -m rdd.verify.tile_roundtrip --root /tmp/fixture

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
from pathlib import Path

from rdd.data import _make_fixture, voc_to_coco
from rdd.data.tiling import (
    Tile,
    TileDetection,
    clip_box_to_tile,
    iou,
    is_truncated,
    merge_detections,
    tile_grid,
    tile_image,
    to_full_image,
)


def _full(dets: list[TileDetection], tiles: list[Tile]) -> list[tuple]:
    by_index = {t.index: t for t in tiles}
    return [to_full_image(d, by_index[d.tile]) for d in dets]

TOL = 1e-6

# (tile, overlap, min_box_fraction). The first is the real configuration; the
# others shrink the tile below the fixture's 600x600 so a genuine grid is
# exercised, with the last one small enough to cut the fixture's longest boxes
# into three.
CONFIGS = [(640, 0.2, 0.3), (256, 0.25, 0.3), (128, 0.5, 0.3)]

# Norway's dimensions, from docs/13-tier-b-plan.md.
NORWAY = (3650, 2044)


class Checks:
    """Collects pass/fail lines so every check runs before anything exits."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def note(self, name: str, detail: str) -> None:
        self.rows.append((None, name, detail))

    @property
    def failed(self) -> list[tuple[bool, str, str]]:
        return [r for r in self.rows if r[0] is False]


# ---------------------------------------------------------------------------
# A. grid
# ---------------------------------------------------------------------------
def check_grid(ch: Checks) -> None:
    sizes = [(600, 600), (640, 640), (512, 512), NORWAY, (3650, 640),
             (100, 4000), (641, 641), (1, 1)]
    n_tiles = 0
    in_bounds = full_size = covered = ends_on_edge = True
    for w, h in sizes:
        for tile, overlap, _ in CONFIGS:
            tiles = tile_grid(w, h, tile, overlap)
            n_tiles += len(tiles)
            xs, ys = bytearray(w), bytearray(h)
            for t in tiles:
                if not (0 <= t.x0 < t.x1 <= w and 0 <= t.y0 < t.y1 <= h):
                    in_bounds = False
                # Every tile is tile-sized unless that axis is shorter than one
                # tile, in which case it is the whole axis (decision 1).
                if t.width != min(tile, w) or t.height != min(tile, h):
                    full_size = False
                for i in range(t.x0, t.x1):
                    xs[i] = 1
                for i in range(t.y0, t.y1):
                    ys[i] = 1
            # The grid is a product of the two axes, so complete 1-D coverage on
            # both is exactly complete 2-D coverage.
            if not (all(xs) and all(ys)):
                covered = False
            if max(t.x1 for t in tiles) != w or max(t.y1 for t in tiles) != h:
                ends_on_edge = False

    ch.check(in_bounds, "A1 every tile lies inside the image",
             f"{n_tiles} tiles over {len(sizes)} sizes x {len(CONFIGS)} configs")
    ch.check(full_size, "A2 every tile is tile-sized (short axis excepted)")
    ch.check(covered, "A3 the grid covers the image with no gap")
    ch.check(ends_on_edge, "A4 the last tile ends on the image boundary",
             "shifted inward, not padded")

    # An image at or below one tile is passed through whole.
    whole = [tile_grid(600, 600, 640, 0.2), tile_grid(640, 640, 640, 0.2),
             tile_grid(3650, 2044, 640, 0.2, tile_above=4000)]
    ch.check(all(len(t) == 1 and t[0].x0 == 0 and t[0].y0 == 0 for t in whole),
             "A5 image below the threshold is one whole-image tile",
             "600x600 and 640x640 at tile 640; 3650x2044 at --tile-above 4000")

    # Interior starts advance by exactly the requested stride.
    tiles = tile_grid(*NORWAY, 640, 0.2)
    cols = sorted({t.x0 for t in tiles})
    ch.check(all(b - a == 512 for a, b in zip(cols, cols[1:-1])),
             "A6 interior stride is tile*(1-overlap)",
             f"cols {cols} (last shifted inward)")
    ch.check(len(tiles) == 28, "A7 Norway frame at 640/0.2 gives 7x4 tiles",
             f"{len(cols)} cols x {len({t.y0 for t in tiles})} rows "
             f"= {len(tiles)}")


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
def load_fixture(root: Path) -> dict:
    """Build the fixture if needed, convert it, and return the COCO store."""
    if not any(root.rglob("*.xml")):
        root.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _make_fixture.build(root)
    out = root / "fixture_coco.json"
    # --no-strict: the fixture deliberately contains an unknown label (D99).
    buf = io.StringIO()
    # The fixture's unknown label is deliberate, so neither the converter's
    # report nor its warning belongs in this verifier's output.
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = voc_to_coco.main(["--root", str(root), "--source", "rdd2022",
                               "--split", "train", "--out", str(out),
                               "--no-strict"])
    if rc != 0:
        raise SystemExit(f"voc_to_coco failed on the fixture:\n{buf.getvalue()}")
    return json.loads(out.read_text(encoding="utf-8"))


def image_boxes(coco: dict) -> list[tuple[dict, list[tuple[int, tuple]]]]:
    by_image: dict[int, list] = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]].append((a["category_id"], (x, y, x + w, y + h)))
    return [(im, by_image.get(im["id"], [])) for im in coco["images"]]


# ---------------------------------------------------------------------------
# B. clipping, on the fixture
# ---------------------------------------------------------------------------
def check_clipping(ch: Checks, coco: dict) -> None:
    inside = at_least_threshold = dropped_below = full_when_it_fits = True
    n_kept = n_dropped = 0
    zero_area_seen = 0

    for im, boxes in image_boxes(coco):
        W, H = im["width"], im["height"]
        for tile, overlap, min_frac in CONFIGS:
            plans = tile_image(W, H, boxes, tile, overlap, min_frac)
            fits_whole: set[int] = set()
            for p in plans:
                for b in p.kept:
                    n_kept += 1
                    if not (0 <= b.x0 < b.x1 <= p.tile.width
                            and 0 <= b.y0 < b.y1 <= p.tile.height):
                        inside = False
                    if b.retained < min_frac - TOL:
                        at_least_threshold = False
                    if b.retained >= 1.0 - TOL:
                        fits_whole.add(b.source)
                for b in p.dropped:
                    n_dropped += 1
                    if not 0 < b.retained < min_frac:
                        dropped_below = False

            # Any box that fits inside a tile must be kept whole by at least one
            # tile, that is what the overlap is for.
            for i, (_, box) in enumerate(boxes):
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bw <= 0 or bh <= 0:
                    zero_area_seen += 1
                    continue
                if bw <= tile * (1 - overlap) and bh <= tile * (1 - overlap):
                    if i not in fits_whole:
                        full_when_it_fits = False

    ch.check(inside, "B1 every kept box lies inside its tile",
             f"{n_kept} kept boxes")
    ch.check(at_least_threshold, "B2 every kept box retains >= min_fraction")
    ch.check(dropped_below, "B3 every dropped fragment retains < min_fraction",
             f"{n_dropped} fragments dropped")
    ch.check(full_when_it_fits,
             "B4 a box smaller than the stride is kept whole somewhere",
             "what the overlap buys")
    ch.check(zero_area_seen == 0,
             "B5 no zero-area box reaches the tiler",
             "voc_to_coco drops the fixture's degenerate box upstream")


# ---------------------------------------------------------------------------
# C. round trip, on the fixture
# ---------------------------------------------------------------------------
def check_roundtrip(ch: Checks, coco: dict) -> None:
    lossless = exact_recovery = True
    n_dets = n_merged = 0
    exact = 0
    fragmented: list[str] = []
    empty_ok = True

    for im, boxes in image_boxes(coco):
        W, H = im["width"], im["height"]
        for tile, overlap, min_frac in CONFIGS:
            plans = tile_image(W, H, boxes, tile, overlap, min_frac)
            tiles = [p.tile for p in plans]

            # The clipped ground truth, handed back as if the model had emitted
            # it from each tile at confidence 1.0.
            dets: list[TileDetection] = []
            source_of: list[int] = []
            dropped_sources: set[int] = set()
            for p in plans:
                for b in p.kept:
                    dets.append(TileDetection(p.tile.index, b.cls, 1.0,
                                              b.x0, b.y0, b.x1, b.y1))
                    source_of.append(b.source)
                for b in p.dropped:
                    dropped_sources.add(b.source)

            merged = merge_detections(dets, tiles, W, H)
            n_dets += len(dets)
            n_merged += len(merged)

            if not boxes:
                # A tile with no boxes at all: tiles still produced, nothing to
                # merge, no crash.
                if dets or merged or not tiles:
                    empty_ok = False

            # C1 lossless: every part accounted for exactly once, and no merged
            # box larger than the union of its own parts.
            seen = [i for m in merged for i in m.parts]
            if sorted(seen) != list(range(len(dets))):
                lossless = False
            for m in merged:
                px0 = min(dets[i].x0 + tiles[dets[i].tile].x0 for i in m.parts)
                py0 = min(dets[i].y0 + tiles[dets[i].tile].y0 for i in m.parts)
                px1 = max(dets[i].x1 + tiles[dets[i].tile].x0 for i in m.parts)
                py1 = max(dets[i].y1 + tiles[dets[i].tile].y0 for i in m.parts)
                if (abs(m.x0 - px0) > TOL or abs(m.y0 - py0) > TOL
                        or abs(m.x1 - px1) > TOL or abs(m.y1 - py1) > TOL):
                    lossless = False
                if len({dets[i].cls for i in m.parts}) != 1:
                    lossless = False

            # C2 exact recovery: a source box none of whose fragments were
            # dropped must come back as exactly one box, at its original
            # coordinates.
            groups: dict[int, list] = collections.defaultdict(list)
            for m in merged:
                for i in m.parts:
                    groups[source_of[i]].append(m)
            for src, (cls, box) in enumerate(boxes):
                if src in dropped_sources:
                    if len({id(m) for m in groups.get(src, [])}) > 1:
                        fragmented.append(
                            f"{im['file_name']} box {src} at tile {tile}: "
                            f"{len({id(m) for m in groups[src]})} pieces")
                    continue
                recovered = {id(m): m for m in groups.get(src, [])}
                if len(recovered) != 1:
                    exact_recovery = False
                    continue
                m = next(iter(recovered.values()))
                if (abs(m.x0 - box[0]) > TOL or abs(m.y0 - box[1]) > TOL
                        or abs(m.x1 - box[2]) > TOL or abs(m.y1 - box[3]) > TOL
                        or m.cls != cls):
                    exact_recovery = False
                else:
                    exact += 1

    ch.check(lossless, "C1 the merge is lossless and adds no area",
             f"{n_dets} tile detections -> {n_merged} merged")
    ch.check(exact_recovery,
             "C2 an unclipped box round-trips to its exact coordinates",
             f"{exact} boxes recovered to within {TOL}")
    ch.check(empty_ok, "C3 an image with no boxes yields tiles and no detections",
             "the fixture's genuine negative")
    if fragmented:
        ch.note("C4 boxes left in pieces after a fragment was dropped",
                f"{len(fragmented)}: " + "; ".join(fragmented[:3]))
    else:
        ch.note("C4 boxes left in pieces after a fragment was dropped", "none")


# ---------------------------------------------------------------------------
# D. degenerate geometry
# ---------------------------------------------------------------------------
def check_degenerate(ch: Checks) -> None:
    tiles = tile_grid(1024, 512, 512, 0.0)          # seam at x = 512, no overlap
    left, right = tiles[0], tiles[1]

    # A box whose right edge is exactly on the seam.
    on_seam = (300.0, 100.0, 512.0, 200.0)
    in_left = clip_box_to_tile(on_seam, left)
    in_right = clip_box_to_tile(on_seam, right)
    ch.check(in_left is not None and abs(in_left[4] - 1.0) < TOL
             and in_right is None,
             "D1 a box ending exactly on a seam belongs to one tile",
             "half-open on x1/y1, so no zero-width sliver next door")

    # A box exactly equal to a tile.
    exact_tile = (512.0, 0.0, 1024.0, 512.0)
    got = clip_box_to_tile(exact_tile, right)
    ch.check(got is not None and abs(got[4] - 1.0) < TOL
             and got[:4] == (0.0, 0.0, 512.0, 512.0),
             "D2 a box exactly filling a tile clips to the whole tile")

    # Degenerate boxes.
    ch.check(clip_box_to_tile((5.0, 5.0, 5.0, 400.0), left) is None
             and clip_box_to_tile((5.0, 5.0, 100.0, 5.0), left) is None
             and clip_box_to_tile((100.0, 10.0, 50.0, 200.0), left) is None,
             "D3 zero-width, zero-height and inverted boxes clip to nothing",
             "no division by zero")

    # A box entirely outside the tile.
    ch.check(clip_box_to_tile((600.0, 10.0, 700.0, 20.0), left) is None,
             "D4 a box outside the tile clips to nothing")

    # No detections at all.
    ch.check(merge_detections([], tiles, 1024, 512) == [],
             "D5 merging an empty detection list returns nothing")

    # One whole-image tile: the merge is the identity map.
    one = tile_grid(600, 600, 640, 0.2)
    d = TileDetection(0, 0, 0.9, 10.0, 20.0, 120.0, 300.0)
    merged = merge_detections([d], one, 600, 600)
    ch.check(len(merged) == 1 and merged[0].as_list() == [10.0, 20.0, 120.0, 300.0],
             "D6 on a single whole-image tile the merge is the identity")

    # Two detections from the same tile are never joined, however close.
    same = [TileDetection(0, 0, 0.9, 10.0, 10.0, 100.0, 100.0),
            TileDetection(0, 0, 0.8, 12.0, 12.0, 102.0, 102.0)]
    ch.check(len(merge_detections(same, one, 600, 600)) == 2,
             "D7 detections from one tile are never merged with each other",
             "the detector's own NMS already ruled on those")

    # An edge that is the image border is not a seam.
    border = Tile(0, 0, 0, 0, 0, 512, 512)
    ch.check(not is_truncated((400.0, 0.0, 512.0, 100.0), border, 512, 512)
             and is_truncated((400.0, 0.0, 512.0, 100.0), border, 1024, 512),
             "D8 a box at the image border counts as truncated only "
             "when a tile follows")


# ---------------------------------------------------------------------------
# E. Norway-scale cases the fixture cannot provide
# ---------------------------------------------------------------------------
def check_norway(ch: Checks) -> None:
    W, H = NORWAY
    tiles = tile_grid(W, H, 640, 0.2)
    by_pos = {(t.row, t.col): t for t in tiles}

    def local(t: Tile, box) -> tuple[float, float, float, float]:
        return (box[0] - t.x0, box[1] - t.y0, box[2] - t.x0, box[3] - t.y0)

    # --- E1: one hairline C1 crack, 3 px wide and 900 px long, crossing two
    # horizontal seams. This is the case the whole module exists for.
    crack = (1000.0, 300.0, 1003.0, 1200.0)
    rows = [by_pos[(r, 1)] for r in (0, 1, 2)]
    dets = [TileDetection(t.index, 0, 0.6 + 0.1 * i, *local(t, (
        crack[0], max(crack[1], t.y0), crack[2], min(crack[3], t.y1))))
        for i, t in enumerate(rows)]

    stitched = merge_detections(dets, tiles, W, H)
    plain = merge_detections(dets, tiles, W, H, mode="nms")
    ok = (len(stitched) == 1
          and max(abs(a - b) for a, b in zip(stitched[0].as_list(), crack)) < TOL)
    ch.check(ok, "E1 a crack cut by two seams merges back to one box",
             f"3 tile detections -> {len(stitched)} stitched, "
             f"{len(plain)} under plain NMS")
    worst = max(iou(full_a, full_b)
                for i, full_a in enumerate(_full(dets, tiles))
                for full_b in _full(dets, tiles)[i + 1:])
    ch.check(len(plain) == 3,
             "E1b plain NMS leaves the crack in pieces",
             f"highest mutual IoU between the pieces is {worst:.2f}, "
             "below any usable threshold")

    # --- E2: a compact C4 pothole sitting in the overlap band, seen whole by
    # two tiles. Both modes must collapse it to one box.
    pothole = (2080.0, 530.0, 2140.0, 590.0)
    pair = [by_pos[(0, 3)], by_pos[(1, 4)]]
    dets = [TileDetection(t.index, 3, 0.8, *local(t, pothole)) for t in pair]
    ch.check(len(merge_detections(dets, tiles, W, H)) == 1
             and len(merge_detections(dets, tiles, W, H, mode="nms")) == 1,
             "E2 a pothole seen whole by two tiles collapses to one box",
             "duplicate relation, which plain NMS also gets right")

    # --- E3: a truncated view of an object the neighbour sees whole. 100 px of
    # C2 crack, 40 px of it visible in the tile to the left.
    crack2 = (600.0, 800.0, 700.0, 806.0)
    a = by_pos[(1, 0)]       # x 0..640   -> sees 600..640
    b = by_pos[(1, 1)]       # x 512..1152 -> sees all of it
    dets = [TileDetection(a.index, 1, 0.5, *local(a, (600.0, 800.0, 640.0, 806.0))),
            TileDetection(b.index, 1, 0.9, *local(b, crack2))]
    merged = merge_detections(dets, tiles, W, H)
    nms = merge_detections(dets, tiles, W, H, mode="nms")
    ch.check(len(merged) == 1 and abs(merged[0].score - 0.9) < TOL
             and max(abs(x - y) for x, y in zip(merged[0].as_list(), crack2)) < TOL,
             "E3 a stub cut at a seam joins the whole view of the same object",
             f"IoU is 0.4 so NMS leaves {len(nms)}; score kept at the max")

    # --- E4: two parallel cracks 100 px apart, each cut by the same seam. The
    # alignment test must keep them separate.
    t0, t1 = by_pos[(0, 1)], by_pos[(1, 1)]
    dets = [
        TileDetection(t0.index, 0, 0.7, *local(t0, (1000.0, 300.0, 1003.0, 640.0))),
        TileDetection(t1.index, 0, 0.7, *local(t1, (1100.0, 512.0, 1103.0, 900.0))),
    ]
    ch.check(len(merge_detections(dets, tiles, W, H)) == 2,
             "E4 two parallel cracks across one seam stay separate",
             "perpendicular overlap is 0, below align_thr")

    # --- E5: the documented limitation. Two distinct cracks that genuinely abut
    # at a seam are indistinguishable from one crack cut in two, and fuse.
    dets = [
        TileDetection(t0.index, 0, 0.7, *local(t0, (1000.0, 100.0, 1003.0, 640.0))),
        TileDetection(t1.index, 0, 0.7, *local(t1, (1000.0, 512.0, 1003.0, 900.0))),
    ]
    fused = merge_detections(dets, tiles, W, H)
    ch.note("E5 two distinct cracks abutting at a seam fuse into one",
            f"{len(fused)} box spanning y "
            f"{fused[0].y0:.0f}-{fused[0].y1:.0f}, the known limitation, "
            "measurable only on real images (US16)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Exercise rdd.data.tiling end to end on the synthetic fixture.")
    ap.add_argument("--root", required=True, type=Path,
                    help="fixture root; built by rdd.data._make_fixture if empty")
    args = ap.parse_args(argv)

    coco = load_fixture(args.root)
    print(f"fixture {args.root}")
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} boxes "
          f"(1 genuine negative, 1 degenerate box and 1 unknown label dropped "
          f"upstream by voc_to_coco)")
    print("  configs (tile, overlap, min_box_fraction): "
          + ", ".join(str(c) for c in CONFIGS))

    ch = Checks()
    sections = [
        ("A. tile grid", lambda: check_grid(ch)),
        ("B. box clipping, on the fixture", lambda: check_clipping(ch, coco)),
        ("C. detection merge round trip, on the fixture",
         lambda: check_roundtrip(ch, coco)),
        ("D. degenerate geometry", lambda: check_degenerate(ch)),
        ("E. Norway-scale cases (3650x2044, synthetic)",
         lambda: check_norway(ch)),
    ]
    for title, run in sections:
        at = len(ch.rows)
        run()
        print(f"\n{title}")
        for ok, name, detail in ch.rows[at:]:
            tag = "[ok]  " if ok else ("[FAIL]" if ok is False else "[note]")
            print(f"  {tag} {name:<58} {detail}")

    n_checks = sum(1 for r in ch.rows if r[0] is not None)
    print("\n" + "=" * 78)
    if ch.failed:
        print(f"FAILED {len(ch.failed)} of {n_checks} checks:")
        for _, name, detail in ch.failed:
            print(f"  {name} {detail}")
        return 1
    print(f"all {n_checks} checks passed")
    print("Geometry only: no image was opened and no tiled dataset was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
