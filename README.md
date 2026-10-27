# RDD: A General Road Damage Detector

Implementation code for the 20/20 Computer Vision group project,
*A General Road Damage Detector over Publicly Available Damage Categories*.

The project trains a single detector over a broader road damage taxonomy than the
dominant public benchmark uses, and measures what that generality costs on the
classes the benchmark does cover.

This repository holds the code. Research findings, infrastructure specifics, cost
figures and planning documents are kept separately. If a comment refers to a
document you cannot find, that is why.

## Layout

```
rdd/
├── configs/
│   ├── classmap.yaml          canonical class map, frozen ids
│   ├── yolo11s-p2.yaml        stride-4 detection head
│   ├── yolo11s-strip.yaml     strip attention
│   ├── yolo11s-dysample.yaml  dynamic upsampling
│   └── yolo11s-combined.yaml  all three
├── infra/
│   ├── config.sh              names and defaults
│   ├── bootstrap.sh           S3, IAM, security group, budget
│   ├── launch-instance.sh     start a GPU box
│   ├── launch-clone.sh        start one from an image of a running box
│   ├── stop-instance.sh       stop compute, keep the disk
│   ├── terminate-instance.sh  destroy, requires typing TERMINATE
│   ├── status.sh              what is running and what it costs
│   ├── track.sh               live progress across all instances
│   ├── sync-code.sh           laptop to S3 to instance
│   ├── fetch-datasets.sh      download source data
│   ├── run-tierA.sh           baseline experiment set
│   ├── run-tierB.sh           tiling experiments
│   ├── run-us17.sh            architecture variants
│   ├── run-us21.sh            rutting detection and segmentation
│   ├── run-us22.sh            zero-shot evaluation
│   └── userdata.sh            instance first boot
└── src/rdd/
    ├── verify/                dataset audits and progress reporting
    ├── data/
    │   ├── voc_to_coco.py     VOC to canonical COCO store
    │   ├── dedup.py           perceptual hash de-duplication
    │   ├── splits.py          grouped splits, hashed
    │   ├── to_yolo.py         COCO to YOLO view
    │   ├── to_yolo_tiled.py   tiled YOLO view
    │   ├── rutting_to_yolo.py Sekilab rutting corpus
    │   └── stellenbosch_to_yolo.py  Stellenbosch pothole corpus
    ├── models/                custom detector modules
    ├── train/
    │   ├── train_yolo.py      baseline and variant trainer
    │   └── train_rutting.py   rutting detection and segmentation
    └── eval/
        ├── aggregate.py       mean and standard deviation across seeds
        ├── aggregate_seeds.py per-class-by-country spreads
        ├── revalidate.py      evaluation only, per class and per country
        ├── compare_variants.py  architecture comparison table
        ├── latency.py         inference latency per variant
        ├── zeroshot.py        held-out corpus evaluation
        └── distractor.py      prediction counts on uncovered damage
```

## Getting started

Nothing here needs cloud access or a dataset download.

```bash
git clone https://github.com/jassiel-ndlovu/rdd-general.git
cd rdd-general
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Build a synthetic dataset and run the converter over it:

```bash
PYTHONPATH=src python -m rdd.data._make_fixture --root /tmp/fixture
```

```bash
PYTHONPATH=src python -m rdd.data.voc_to_coco --root /tmp/fixture --source rdd2022 --split train --out /tmp/fixture/out.json --no-strict
```

If both succeed the environment works. CI runs the same smoke test.

`requirements.txt` pins `ultralytics`, which pulls in PyTorch at roughly 900 MB. To
work on the data pipeline only, install everything except that line.

## Cloud infrastructure

`infra/` provisions and manages an AWS GPU training environment: an S3 bucket, a
scoped IAM role, a security group with no inbound rules, a budget with alerts, and
a GPU instance that stops itself after a period of inactivity.

These scripts will not work against the project's own account and are not meant to.
Account id, bucket name and region are derived at runtime, so an equivalent
environment can be stood up in any account:

```bash
export AWS_REGION=us-east-1
bash infra/bootstrap.sh          # idempotent, near-free while idle
bash infra/status.sh             # what exists and what it costs
bash infra/launch-instance.sh --job baseline --dry-run
```

`bootstrap.sh` is safe to re-run. `launch-instance.sh` confirms the hourly cost
before starting anything, and `--dry-run` shows what it would do without doing it.

These scripts create billable resources. Run `status.sh` before and after, and do
not leave an instance running unattended. The idle watchdog is a backstop, not a
substitute for checking.

## Conventions

Breaking one of these invalidates work, sometimes without any visible error.

| | |
|---|---|
| Canonical annotation format | COCO JSON with project attributes (`severity`, `country`, `view`, `orig_class`) |
| YOLO label files | Generated, never hand-edited. Change the COCO store and re-export |
| Class ids | Frozen in `configs/classmap.yaml`. CI enforces this |
| Label map | Derived from the labels present in the data, not the one shipped with the dataset |
| Splits | By country and duplicate cluster, never random image. De-duplicate first |
| Seeds | Three per configuration, reported as mean and standard deviation |
| Augmentation | No vertical flip and no rotation. Orientation distinguishes longitudinal from transverse cracks, so rotating an image relabels it. CI enforces this |
| Metrics | Per class and per country, never a single pooled number |
| Smoke tests | Any run with `--fraction < 1.0` writes `valid_for_reporting: false` |
| Tagging | Every cloud resource gets `Project=RDD-General`, and every script filters on it |

## Contributing

`main` is protected and nobody pushes to it directly.

```bash
git switch main && git pull
git switch -c us12-four-class-control     # name the branch after the backlog item
# work
git push -u origin us12-four-class-control
gh pr create --fill
```

Every pull request needs project manager approval before it can merge. The pull
request template asks for the command you ran and its output.

CI runs on every pull request and checks LF line endings in shell scripts, shell
syntax, that every pipeline module imports, the fixture smoke test, that the frozen
class ids are unchanged, and that the augmentation constraints are still in place.

Tell the team before launching a GPU instance. One instance at a time.
