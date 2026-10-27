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

## Getting the data

Three public datasets. None of them is redistributed here, and all three are
fetched by script.

| Dataset | Used for | Licence | Fetch |
|---|---|---|---|
| RDD2022 | C1 to C6, training and validation | CC BY 4.0 on figshare; the authors' GitHub README says CC BY-SA 4.0 | `bash infra/fetch-datasets.sh` |
| Road Rutting (Sekilab) | C7 rutting | CC BY 4.0, doi:10.17632/v8mbnvx8pv.1 | `bash infra/run-us21.sh` |
| Stellenbosch potholes | zero-shot evaluation | CC0 1.0 | `bash infra/run-us22.sh` |

The RDD2022 licence is stated inconsistently by its own publishers, so treat it
as the more restrictive CC BY-SA 4.0 until that is resolved. The rutting
archive is verified against the SHA-256 its publisher provides; the Stellenbosch
archive has no published hash, so its size and tar index are checked instead and
the extracted image count is compared against the published 11,497.

RDD2022 is a 13.3 GB download and `fetch-datasets.sh` is meant to run on the
training instance, where the transfer and the S3 upload are both in-region.

Once the raw data is in place, build the canonical store, the splits and the
YOLO views:

```bash
bash infra/run-m1.sh      # VOC to COCO, de-duplication, grouped splits, YOLO export
```

The splits are grouped by country and duplicate cluster, never by random image,
and are fixed by SHA-256. Those hashes are recorded in every `revalidation.json`
under `reports/`, so a rebuilt split can be checked against the one the reported
results used.

## Reproducing the results

Every training run sets its seed explicitly. The Tier A baseline runs three
seeds (0, 1, 2) and everything else runs seed 0, which is what the report states
alongside each number.

```bash
bash infra/run-tierA.sh       # six-class baseline x3 seeds, four-class control
bash infra/run-tierB.sh       # tiled resolution front end
bash infra/run-us17.sh abcd   # the four architecture variants
bash infra/run-us21.sh        # rutting detection and segmentation
bash infra/run-us22.sh        # zero-shot evaluation on held-out imagery
bash infra/latency.sh         # per-variant latency, on an idle GPU
```

Each run writes `summary.json` and `reval/revalidation.json` beside its
checkpoints and syncs them to S3. The aggregation and comparison steps are:

```bash
PYTHONPATH=src python -m rdd.eval.aggregate --runs runs --out runs/tierA_summary.json
PYTHONPATH=src python -m rdd.eval.aggregate_seeds \
    --reval reports/tier-a/tierA_base6_s0.revalidation.json \
    --reval reports/tier-a/tierA_base6_s1.revalidation.json \
    --reval reports/tier-a/tierA_base6_s2.revalidation.json \
    --out   reports/tier-a/aggregate_seeds.json
PYTHONPATH=src python -m rdd.eval.compare_variants \
    --baseline reports/tier-a/tierA_base6_s0.revalidation.json \
    --aggregate reports/tier-a/aggregate.json \
    --variant us17b=reports/tier-b/us17b_strip_s0.revalidation.json
```

A full reproduction needs a GPU: the Tier A set is about 40 GPU-hours and each
variant is 15 to 17 hours on an NVIDIA T4. The data pipeline, the aggregation
and the comparison tables all run on a laptop with no GPU.

## Results in this repository

Every number in the report is read from one of these files rather than
transcribed by hand.

| Report section | Files |
|---|---|
| Per class, per country, cost of generality | `reports/tier-a/aggregate.json`, `aggregate_seeds.json`, `tierA_base6_s{0,1,2}.revalidation.json`, `tierA_ctrl4_s0.json` |
| Why potholes are missed | `reports/tier-a/false_negatives_C4_pothole.json` |
| Architecture variants and latency | `reports/tier-b/us17{a,b,c,d}_*.json`, `us17_latency.json` |
| Rutting | `reports/tier-c/us21_rutting_*.json` |
| Zero-shot transfer | `reports/tier-c/us22_zeroshot*.json`, `us22_conversion.json` |

`reports/tier-a/` and `reports/tier-b/` also hold the generated LaTeX tables, so
the report's tables and the JSON cannot disagree.

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
