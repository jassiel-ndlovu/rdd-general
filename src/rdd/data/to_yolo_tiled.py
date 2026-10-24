"""Export a tiled YOLO view of the canonical store.

`rdd.data.tiling` decides *where* the tiles go and which boxes survive clipping.
This module is what actually opens the images, writes the crops, and emits the
YOLO label files beside them. The split between the two is deliberate: the
geometry is testable on a synthetic fixture with no image set, and is (see
`rdd.verify.tile_roundtrip`); only this module needs the 26 GB of pixels.

Usage:
    rddpython -m rdd.data.to_yolo_tiled \
        --coco   /opt/rdd/data/coco/rdd2022_train.json \
        --splits /opt/rdd/data/splits/rdd2022_core.json \
        --root   /opt/rdd/data/rdd2022/RDD2022 \
        --out    /opt/rdd/data/yolo/tiled1280 \
        --tile 1280 --overlap 0.2 --tile-above 1200

Three decisions worth knowing about, because they affect what the resulting
numbers mean.

**Both halves are tiled.** It would be possible to tile only the training split
and validate on whole images, but a model trained on 1280 px crops and validated
on whole 3650 px frames downscaled to 640 is being validated out of its own
domain, and the per-epoch curves become uninterpretable. So validation is tiled
too, which makes the training-time metrics self-consistent. It also means those
metrics are NOT comparable with Tier A's: the comparable number comes from
running tiled inference over the original whole validation images and merging
the detections back, which is a separate step.

**Tile crops are written, not symlinked.** `to_yolo.py` symlinks whole images
because the pixels are identical to the source. A crop is new data, so there is
nothing to link to.

**Filenames keep the country parseable.** Per-country evaluation recovers the
country as `stem.rsplit("_", 1)[0]`, so a tile suffix must not introduce a new
underscore. Tiles are named `<stem>-t<NN>`, which leaves `Norway_000123-t07`
parsing to `Norway` exactly as the untiled `Norway_000123` does.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from rdd.data.tiling import (
    MIN_FRACTION_DEFAULT,
    OVERLAP_DEFAULT,
    TILE_DEFAULT,
    tile_image,
)

JPEG_QUALITY = 95   # high: this is a resolution experiment, so compression
                    # artefacts are the last thing we want to introduce


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path, help="image root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tile", type=int, default=TILE_DEFAULT)
    ap.add_argument("--overlap", type=float, default=OVERLAP_DEFAULT)
    ap.add_argument("--min-box-fraction", type=float, default=MIN_FRACTION_DEFAULT)
    ap.add_argument("--tile-above", type=int, default=None)
    ap.add_argument("--drop-empty-fraction", type=float, default=0.0,
                    help="discard this fraction of the tiles that carry no "
                         "label, chosen deterministically. 0.0 keeps them all. "
                         "At tile 640 the corpus is 83%% negative tiles, which "
                         "pushes a detector towards predicting nothing -- see "
                         "docs/13-tier-b-plan.md section 5")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many source images (a smoke test)")
    args = ap.parse_args(argv)

    from PIL import Image

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))

    # Drop categories with no instances, exactly as to_yolo.py does. C7_rutting
    # is declared in the frozen class map but has zero annotations in RDD2022,
    # so keeping it would emit nc: 7 against Tier A's nc: 6 and shift nothing but
    # break the comparison. Dropping it leaves indices 0-5 identical.
    present = {a["category_id"] for a in coco["annotations"]}
    all_cats = sorted(coco["categories"], key=lambda c: c["id"])
    cats = [c for c in all_cats if c["id"] in present]
    excluded = [c["name"] for c in all_cats if c["id"] not in present]
    names = [c["name"] for c in cats]
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    images = {im["id"]: im for im in coco["images"]}
    boxes_by_image: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]           # COCO bbox is xywh
        boxes_by_image[a["image_id"]].append(
            (cat_to_idx[a["category_id"]], (x, y, x + w, y + h)))

    assign: dict[int, str] = {}
    for half in ("train", "val"):
        for iid in splits["splits"][half]["image_ids"]:
            assign[iid] = half

    for half in ("train", "val"):
        (args.out / "images" / half).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / half).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    per_country = Counter()
    missing: list[str] = []
    done = 0

    for iid, im in sorted(images.items()):
        half = assign.get(iid)
        if half is None:
            stats["image_not_in_split"] += 1      # China_Drone, excluded as aerial
            continue

        src = args.root / im["rel_path"]
        if not src.exists():
            missing.append(im["rel_path"])
            if len(missing) > 50:
                print("too many missing images; is --root wrong?")
                for m in missing[:10]:
                    print("  ", m)
                return 1
            continue

        plans = tile_image(
            im["width"], im["height"], boxes_by_image.get(iid, []),
            tile=args.tile, overlap=args.overlap,
            min_fraction=args.min_box_fraction, tile_above=args.tile_above,
        )

        stem = Path(im["file_name"]).stem
        country = stem.rsplit("_", 1)[0]
        whole = len(plans) == 1 and plans[0].tile.width == im["width"] \
            and plans[0].tile.height == im["height"]

        # Only decode the pixels when something actually needs cropping. A
        # pass-through image is linked, so opening it would be wasted work on
        # roughly 28,000 of the 36,000 source images.
        img = None
        try:
            if not whole:
                img = Image.open(src)
                img = img.convert("RGB")
            for p in plans:
                t = p.tile
                # Deterministic negative dropping, keyed on the tile name.
                # crc32, not hash(): Python randomises string hashing per
                # process, so hash() would select a different subset on every
                # run and the export would stop being reproducible.
                #
                # Only tiles CUT from an image are eligible. A pass-through
                # image is byte-for-byte what the untiled export had, so its
                # negatives are the baseline's own and dropping them would
                # change the training distribution in a second, unrelated way.
                # Every artificial negative comes from a grid, and those are
                # the ones that need thinning: at tile 1280 Norway yields
                # 49,784 empty tiles against 5,712 carrying a label, and that
                # one subset then accounts for 63% of the whole training set.
                if not p.kept and not whole and args.drop_empty_fraction > 0:
                    key = zlib.crc32(f"{stem}-t{t.index}".encode()) % 1000
                    if key / 1000.0 < args.drop_empty_fraction:
                        stats["negative_tiles_dropped"] += 1
                        continue

                name = stem if whole else f"{stem}-t{t.index:02d}"
                dst_img = args.out / "images" / half / f"{name}.jpg"
                dst_lbl = args.out / "labels" / half / f"{name}.txt"

                if not (dst_img.exists() or dst_img.is_symlink()):
                    if whole:
                        # Nothing was cropped, so the pixels are the source's.
                        # Re-encoding ~28,000 pass-through images would cost an
                        # hour and add a generation of JPEG loss to a
                        # resolution experiment, for no gain. Link instead, as
                        # to_yolo.py does.
                        try:
                            dst_img.symlink_to(src.resolve())
                        except OSError:
                            import shutil
                            shutil.copy2(src, dst_img)
                    else:
                        img.crop((t.x0, t.y0, t.x1, t.y1)).save(
                            dst_img, "JPEG", quality=JPEG_QUALITY)

                lines = []
                for b in p.kept:
                    # Tile-local pixels -> normalised xywh against the tile.
                    cx = (b.x0 + b.x1) / 2 / t.width
                    cy = (b.y0 + b.y1) / 2 / t.height
                    bw = (b.x1 - b.x0) / t.width
                    bh = (b.y1 - b.y0) / t.height
                    lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                # An empty file, not a missing one: YOLO reads absence as "not
                # yet exported" in some tooling and as "negative" in others, and
                # the project convention is to be explicit.
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""),
                                   encoding="utf-8")

                stats[f"tiles_{half}"] += 1
                per_country[country] += 1
                stats["tiles_with_labels" if p.kept else "tiles_negative"] += 1
                stats["boxes"] += len(p.kept)
                stats["boxes_dropped_as_fragments"] += len(p.dropped)
                if p.dropped and not p.kept:
                    stats["tiles_with_unlabelled_damage"] += 1
        finally:
            if img is not None:
                img.close()

        done += 1
        if done % 2000 == 0:
            print(f"  {done} source images -> {stats['tiles_train'] + stats['tiles_val']} tiles")
        if args.limit and done >= args.limit:
            print(f"  stopping at --limit {args.limit}")
            break

    header = (
        "# Generated by rdd.data.to_yolo_tiled -- do not edit by hand.\n"
        f"# source COCO : {args.coco}\n"
        f"# split manifest: {args.splits}\n"
        f"# train sha256 : {splits['splits']['train']['sha256']}\n"
        f"# val   sha256 : {splits['splits']['val']['sha256']}\n"
        f"# tiling      : tile={args.tile} overlap={args.overlap} "
        f"tile_above={args.tile_above} min_box_fraction={args.min_box_fraction}\n"
        "# NOTE: both halves are tiled, so metrics computed against this view\n"
        "#       are NOT comparable with the untiled Tier A numbers. The\n"
        "#       comparable figure comes from tiled inference over the original\n"
        "#       whole validation images.\n"
        + (f"# excluded (zero instances in this split): {excluded}\n"
           if excluded else "")
    )
    (args.out / "data.yaml").write_text(
        header
        + f"path: {args.out.resolve()}\n"
        + "train: images/train\n"
        + "val: images/val\n"
        + f"nc: {len(names)}\n"
        + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")

    print("\n== tiled export ==")
    for k in sorted(stats):
        print(f"  {k:34s} {stats[k]}")
    total = stats["tiles_train"] + stats["tiles_val"]
    if done:
        print(f"  {'k (tiles per source image)':34s} {total / done:.3f}")
    if missing:
        print(f"  {'missing source images':34s} {len(missing)}")
    print("\n  per country:")
    for c, n in per_country.most_common():
        print(f"    {c:20s} {n}")
    print(f"\nwrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
