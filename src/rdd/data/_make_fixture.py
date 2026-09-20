"""Build a tiny synthetic RDD-shaped VOC tree, so the converter can be tested
without the 13 GB archive. Used by the CI smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

XML = """<annotation>
  <folder>images</folder>
  <filename>{fn}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  {objects}
</annotation>
"""

OBJ = """<object>
    <name>{label}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
  </object>"""


def build(root: Path) -> None:
    # Two countries, mirroring the real layout <root>/<Country>/train/{annotations/xmls,images}
    spec = {
        "Japan": [
            ("Japan_000000.jpg", 600, 600, [("D00", 10, 20, 120, 300), ("D20", 200, 210, 330, 400)]),
            ("Japan_000001.jpg", 600, 600, [("D40", 50, 60, 150, 170)]),
            ("Japan_000002.jpg", 600, 600, []),                      # genuine negative
            ("Japan_000003.jpg", 600, 600, [("D10", 5, 5, 5, 400)]),  # degenerate: zero width
            ("Japan_000004.jpg", 600, 600, [("D99", 1, 1, 50, 50)]),  # unknown label
        ],
        "Czech": [
            ("Czech_000000.jpg", 600, 600, [("D10", 30, 40, 300, 90)]),
            ("Czech_000001.jpg", 600, 600, [("D00", 400, 10, 590, 580)]),
        ],
    }

    for country, records in spec.items():
        xml_dir = root / country / "train" / "annotations" / "xmls"
        img_dir = root / country / "train" / "images"
        xml_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        for fn, w, h, boxes in records:
            objects = "\n  ".join(
                OBJ.format(label=l, x1=x1, y1=y1, x2=x2, y2=y2) for l, x1, y1, x2, y2 in boxes)
            (xml_dir / fn.replace(".jpg", ".xml")).write_text(
                XML.format(fn=fn, w=w, h=h, objects=objects), encoding="utf-8")
            # The converter only checks that the image file exists.
            (img_dir / fn).write_bytes(b"\xff\xd8\xff\xdb placeholder")

    print(f"fixture written to {root}")
    print("  Japan: 5 images (1 negative, 1 degenerate box, 1 unknown label)")
    print("  Czech: 2 images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    build(ap.parse_args().root)
