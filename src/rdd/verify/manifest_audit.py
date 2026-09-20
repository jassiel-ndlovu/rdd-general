"""
Audit the RDD2022 file manifest without downloading the 13.3 GB archive.

Why this exists
---------------
Objective O1 (audit per-category data availability) and objective O3 (recover
the dropped D43/D44 marking labels) both need to know what is actually inside
RDD2022 before committing bandwidth, disk and GPU time to it.

The CRDDC organisers publish a Windows ``tree /f`` listing of the archive as a
separate ~3 MB text file. Parsing it answers several questions for 3 MB instead
of 13 GB:

  * how many images and annotation files each country subset really contains,
    and whether those counts reconcile with the published paper;
  * what the filenames look like, which is the precondition for the O3
    filename-join hypothesis;
  * whether any subset ships test annotations (it must not).

Manifest format
---------------
``tree /f`` emits ASCII box drawing. Directory lines carry a ``+---`` or
``\\---`` marker whose column position encodes depth; file lines are plain
names indented under the directory that precedes them::

    +---China_Drone                                    depth 0
    |   \\---train                                      depth 1
    |       +---annotations                            depth 2
    |       |   \\---xmls                               depth 3
    |       |           China_Drone_000000.xml         file in .../xmls

Because the listing is depth-first and a directory's files always follow its
header, attaching each file to the most recently seen directory is correct.

Usage
-----
    python -m rdd.verify.manifest_audit --manifest File_List_CRDDC_RDD2022.txt
    python -m rdd.verify.manifest_audit --manifest ... --json report.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

# Per-country training image counts published in Arya et al. (2024), Figure 4.
# Used as an independent cross-check on our parse.
PUBLISHED_TRAIN_IMAGES = {
    "Japan": 10506, "India": 7706, "Czech": 2829, "Norway": 8161,
    "United_States": 4805, "China_MotorBike": 1977, "China_Drone": 2401,
}
PUBLISHED_TEST_IMAGES = {
    "Japan": 2627, "India": 1959, "Czech": 709, "Norway": 2040,
    "United_States": 1200, "China_MotorBike": 500, "China_Drone": 0,
}
PUBLISHED_TRAIN_LABELS = {
    "Japan": 16470, "India": 6831, "Czech": 1745, "Norway": 11229,
    "United_States": 11014, "China_MotorBike": 4650, "China_Drone": 3068,
}

IMAGE_EXT = (".jpg", ".jpeg", ".png")
ANNOT_EXT = (".xml",)

# Matches the directory marker and reports the column it starts at.
DIR_MARKER = re.compile(r"[+\\]---")


def parse_manifest(path: Path):
    """Return {dir_path: {"images": n, "annotations": n, "samples": [...]}}."""
    text = io.open(path, encoding="utf-8", errors="replace").read()

    stack: list[str] = []
    current = "<root>"
    per_dir: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    samples: dict[str, list[str]] = collections.defaultdict(list)

    for raw in text.splitlines():
        if not raw.strip():
            continue

        m = DIR_MARKER.search(raw)
        if m and raw[: m.start()].strip(" |") == "":
            # Directory header. Column position / 4 gives the nesting depth.
            depth = m.start() // 4
            name = raw[m.end():].strip()
            del stack[depth:]
            stack.append(name)
            current = "/".join(stack)
            continue

        # File line: strip the box-drawing decoration and keep the basename.
        name = raw.lstrip("| \t").strip()
        if not name or name.startswith(("+", "\\")):
            continue
        low = name.lower()
        if low.endswith(IMAGE_EXT):
            per_dir[current]["images"] += 1
            if len(samples[current]) < 4:
                samples[current].append(name)
        elif low.endswith(ANNOT_EXT):
            per_dir[current]["annotations"] += 1
            if len(samples[current]) < 4:
                samples[current].append(name)

    return {d: {**dict(c), "samples": samples[d]} for d, c in per_dir.items()}


def roll_up(per_dir: dict) -> dict:
    """Aggregate per-directory counts into per-country / per-split totals."""
    out: dict[str, dict] = collections.defaultdict(
        lambda: {"train_images": 0, "train_annotations": 0,
                 "test_images": 0, "test_annotations": 0, "samples": []}
    )
    for d, c in per_dir.items():
        parts = [p for p in d.split("/") if p and p != "<root>"]
        if not parts:
            continue
        country = parts[0]
        lowered = [p.lower() for p in parts]
        if "train" in lowered:
            split = "train"
        elif "test" in lowered:
            split = "test"
        else:
            continue
        out[country][f"{split}_images"] += c.get("images", 0)
        out[country][f"{split}_annotations"] += c.get("annotations", 0)
        if c.get("samples") and len(out[country]["samples"]) < 6:
            out[country]["samples"].extend(c["samples"][:2])
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit the RDD2022 archive from its published file manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--json", type=Path, help="write a machine-readable report here")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    per_dir = parse_manifest(args.manifest)
    totals = roll_up(per_dir)

    print("=" * 74)
    print("RDD2022 manifest audit")
    print("=" * 74)
    hdr = f"{'subset':<18}{'train img':>10}{'train xml':>10}{'test img':>10}{'test xml':>10}"
    print(hdr)
    print("-" * len(hdr))

    grand = collections.Counter()
    for country in sorted(totals):
        t = totals[country]
        print(f"{country:<18}{t['train_images']:>10}{t['train_annotations']:>10}"
              f"{t['test_images']:>10}{t['test_annotations']:>10}")
        for k in ("train_images", "train_annotations", "test_images", "test_annotations"):
            grand[k] += t[k]
    print("-" * len(hdr))
    print(f"{'TOTAL':<18}{grand['train_images']:>10}{grand['train_annotations']:>10}"
          f"{grand['test_images']:>10}{grand['test_annotations']:>10}")

    print("\nReconciliation against Arya et al. (2024), Figure 4")
    print("-" * 74)
    discrepancies = 0
    for country in sorted(PUBLISHED_TRAIN_IMAGES):
        exp_tr = PUBLISHED_TRAIN_IMAGES[country]
        exp_te = PUBLISHED_TEST_IMAGES[country]
        got_tr = totals.get(country, {}).get("train_images", 0)
        got_te = totals.get(country, {}).get("test_images", 0)
        ok = (got_tr == exp_tr) and (got_te == exp_te)
        discrepancies += 0 if ok else 1
        print(f"  {country:<18} train {got_tr:>6}/{exp_tr:<6} test {got_te:>5}/{exp_te:<5}"
              f"  {'ok' if ok else 'MISMATCH'}")

    print("\nOne XML per training image?  (a missing XML is an unlabelled image)")
    print("-" * 74)
    for country in sorted(totals):
        t = totals[country]
        diff = t["train_images"] - t["train_annotations"]
        note = "1:1" if diff == 0 else f"{diff} image(s) without an XML"
        print(f"  {country:<18} images {t['train_images']:>6}  xmls {t['train_annotations']:>6}   {note}")

    print("\nTest split must ship WITHOUT annotations")
    print("-" * 74)
    for country in sorted(totals):
        n = totals[country]["test_annotations"]
        print(f"  {country:<18} test xmls: {n:>5}   {'ok' if n == 0 else 'UNEXPECTED'}")

    print("\nFilename patterns  (the input to the O3 marking-recovery join)")
    print("-" * 74)
    for country in sorted(totals):
        print(f"  {country:<18} {totals[country]['samples'][:3]}")

    if args.json:
        args.json.write_text(json.dumps(
            {"totals": totals,
             "published_train_images": PUBLISHED_TRAIN_IMAGES,
             "published_test_images": PUBLISHED_TEST_IMAGES,
             "published_train_labels": PUBLISHED_TRAIN_LABELS,
             "discrepancies": discrepancies}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    print("\n" + "=" * 74)
    print(f"{discrepancies} subset(s) disagree with the published counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
