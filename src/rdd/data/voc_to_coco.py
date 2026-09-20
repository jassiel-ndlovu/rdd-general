"""
Convert Pascal VOC annotations into the project's canonical COCO store (M1/O2).

Design decisions worth knowing
------------------------------
1. COCO JSON is the canonical format, not YOLO TXT. COCO is the only common
   format that carries per-annotation *attributes*, and the project needs to
   attach a severity grade, a source dataset, a country and the original class
   code to every box. YOLO TXT and VOC XML both discard that. Training formats
   are generated from this store, never hand-edited.

2. Unknown labels are a hard error. Silently dropping an unrecognised class is
   how a dataset quietly loses a category; the class map in
   ``configs/classmap.yaml`` must be updated deliberately instead.

3. An image with zero annotations is kept, not discarded. Roughly 46% of
   RDD2022 images carry no damage; they are genuine negatives and the model
   needs them. Images are only dropped if the file itself is missing.

4. Boxes are validated against the image dimensions declared in the XML, and
   degenerate boxes (zero width or height, or inverted coordinates) are
   reported and skipped rather than silently written out.

Usage
-----
    python -m rdd.data.voc_to_coco \\
        --root /opt/rdd/data/rdd2022/RDD2022 \\
        --source rdd2022 \\
        --split train \\
        --out /opt/rdd/data/coco/rdd2022_train.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("pyyaml is required:  pip install pyyaml", file=sys.stderr)
    raise


CONFIG_DEFAULT = Path(__file__).resolve().parents[3] / "configs" / "classmap.yaml"


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------
def load_classmap(path: Path, source: str):
    """Return (name->id, source_label->name, ignore_token, class_records)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    if source not in cfg["sources"]:
        known = ", ".join(cfg["sources"])
        raise SystemExit(f"unknown source '{source}'. Known sources: {known}")

    name_to_id = {c["name"]: c["id"] for c in cfg["classes"]}
    label_map = cfg["sources"][source]["label_map"]
    ignore = cfg.get("ignore_token", "__IGNORE__")

    # Every mapped target must exist in the taxonomy, or the map is incoherent.
    for src_label, target in label_map.items():
        if target != ignore and target not in name_to_id:
            raise SystemExit(
                f"class map error: source label '{src_label}' maps to "
                f"'{target}', which is not a declared class in {path.name}")

    return name_to_id, label_map, ignore, cfg["classes"]


# ---------------------------------------------------------------------------
# VOC parsing
# ---------------------------------------------------------------------------
def parse_voc(xml_path: Path):
    """Extract (filename, width, height, [(label, xmin, ymin, xmax, ymax)])."""
    root = ET.parse(xml_path).getroot()

    filename = (root.findtext("filename") or xml_path.stem + ".jpg").strip()
    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None else 0
    height = int(float(size.findtext("height"))) if size is not None else 0

    boxes = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or not label:
            continue
        try:
            xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((label, xmin, ymin, xmax, ymax))

    return filename, width, height, boxes


def infer_view(country: str, default: str = "forward") -> str:
    """Capture geometry, derived from the subset name.

    RDD2022's China_Drone subset is top-down aerial imagery; every other subset
    is a forward-facing vehicle view. Mixing the two without a marker is a known
    confound, so the view must be recorded per subset rather than set once for
    the whole tree.
    """
    return "topdown" if "drone" in country.lower() else default


def infer_country(xml_path: Path) -> str:
    """RDD2022 lays out <root>/<Country>/<split>/annotations/xmls/*.xml.

    The country is the directory immediately above the split directory, so
    anchor on the split rather than counting components from either end --
    the root path depth varies between machines.
    """
    parts = xml_path.parts
    lowered = [p.lower() for p in parts]
    for split in ("train", "test"):
        if split in lowered:
            i = lowered.index(split)
            if i > 0:
                return parts[i - 1]
    # Fall back to the first directory above the file that is not structural.
    structural = {"xmls", "annotations", "images"}
    for part in reversed(parts[:-1]):
        if part.lower() not in structural:
            return part
    return "unknown"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert(root: Path, source: str, split: str, out: Path,
            config: Path, view: str, strict: bool) -> int:
    name_to_id, label_map, ignore, class_records = load_classmap(config, source)

    xml_files = sorted(root.rglob(f"*/{split}/annotations/xmls/*.xml"))
    if not xml_files:
        xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"no XML annotations found under {root}")

    print(f"found {len(xml_files)} annotation files under {root}")

    images, annotations = [], []
    per_class = collections.Counter()
    per_country = collections.Counter()
    unknown_labels = collections.Counter()
    degenerate = 0
    missing_images = 0
    img_id = 0
    ann_id = 0

    for xml_path in xml_files:
        filename, width, height, boxes = parse_voc(xml_path)
        country = infer_country(xml_path)

        # The image lives in a sibling images/ directory in the RDD layout.
        img_path = xml_path.parent.parent.parent / "images" / filename
        if not img_path.exists():
            alt = list(root.rglob(filename))
            if alt:
                img_path = alt[0]
            else:
                missing_images += 1
                continue

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
            # Project extensions: these are what COCO is being used for.
            "source_dataset": source,
            "country": country,
            "view": infer_view(country, default=view),
            "split_hint": split,
            "rel_path": str(img_path.relative_to(root)).replace("\\", "/"),
        })
        per_country[country] += 1

        for label, xmin, ymin, xmax, ymax in boxes:
            if label not in label_map:
                unknown_labels[label] += 1
                continue
            target = label_map[label]
            if target == ignore:
                continue

            if xmax <= xmin or ymax <= ymin:
                degenerate += 1
                continue
            # Clamp to the declared image bounds; RDD XMLs occasionally exceed them.
            if width and height:
                xmin = max(0.0, min(xmin, width)); xmax = max(0.0, min(xmax, width))
                ymin = max(0.0, min(ymin, height)); ymax = max(0.0, min(ymax, height))
                if xmax <= xmin or ymax <= ymin:
                    degenerate += 1
                    continue

            w, h = xmax - xmin, ymax - ymin
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": name_to_id[target],
                "bbox": [round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                # Project extensions.
                "orig_class": label,
                "severity": None,      # filled later by the severity derivation step
                "source_dataset": source,
                "country": country,
            })
            per_class[target] += 1

    if unknown_labels:
        print("\nUNKNOWN LABELS (not in the class map):", file=sys.stderr)
        for lab, n in unknown_labels.most_common():
            print(f"    {lab:<12} {n:>7} occurrence(s)", file=sys.stderr)
        if strict:
            print("\nRefusing to write. Add these to configs/classmap.yaml "
                  "(map them to a class or to __IGNORE__), or pass --no-strict.",
                  file=sys.stderr)
            return 1

    coco = {
        "info": {
            "description": f"RDD project canonical store -- {source}/{split}",
            "version": "1",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "classmap_version": yaml.safe_load(config.read_text(encoding="utf-8"))["version"],
        },
        "licenses": [{"id": 1, "name": "CC BY-SA 4.0 / CC BY 4.0 -- see source dataset"}],
        "categories": [
            {"id": c["id"], "name": c["name"], "supercategory": c["tier"]}
            for c in class_records
        ],
        "images": images,
        "annotations": annotations,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco), encoding="utf-8")

    # --- report -------------------------------------------------------------
    print(f"\nwrote {out}")
    print(f"  images       {len(images):>8}")
    print(f"  annotations  {len(annotations):>8}")

    labelled = {a["image_id"] for a in annotations}
    print(f"  unlabelled   {len(images) - len(labelled):>8}  "
          f"({100 * (len(images) - len(labelled)) / max(len(images), 1):.1f}% -- kept as negatives)")
    if missing_images:
        print(f"  MISSING imgs {missing_images:>8}  (annotation had no matching image file)")
    if degenerate:
        print(f"  degenerate   {degenerate:>8}  boxes skipped (zero-area or inverted)")

    print("\n  per class:")
    for c in class_records:
        n = per_class.get(c["name"], 0)
        exp = c.get("expected_instances")
        flag = ""
        if exp and n:
            flag = "  ok" if n == exp else f"  (published total for all splits: {exp})"
        print(f"    {c['name']:<26} {n:>8}{flag}")

    print("\n  per country:")
    for k, n in sorted(per_country.items()):
        print(f"    {k:<26} {n:>8}")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert Pascal VOC annotations to the canonical COCO store.")
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root, e.g. /opt/rdd/data/rdd2022/RDD2022")
    ap.add_argument("--source", required=True,
                    help="source key in configs/classmap.yaml (rdd2022, rdd2018, ...)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default=CONFIG_DEFAULT, type=Path)
    ap.add_argument("--view", default="forward",
                    choices=["forward", "topdown"],
                    help="default capture geometry. Subsets whose name contains "
                         "'drone' are forced to topdown regardless, so one run "
                         "over the whole tree records the view correctly.")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="warn on unknown labels instead of refusing to write")
    args = ap.parse_args(argv)

    return convert(args.root, args.source, args.split, args.out,
                   args.config, args.view, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
