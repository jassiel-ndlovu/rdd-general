# RDD — A General Road Damage Detector

Implementation scaffolding for the 20/20 Computer Vision Group project,
*A General Road Damage Detector over Publicly Available Damage Categories*.

The project trains a single detector across a broader road-damage taxonomy than the
dominant public benchmark uses, and measures what that generality costs on the classes
the benchmark does cover.

> **Scope of this repository.** This is the code. The project's research findings,
> infrastructure specifics, cost model and planning documents are held separately and are
> not published here. If a comment refers to a document you cannot find, that is why — ask
> the project manager.

---

## Layout

```
rdd/
├── configs/
│   └── classmap.yaml         canonical class map (FROZEN ids)
├── infra/
│   ├── config.sh             single source of truth for names and defaults
│   ├── bootstrap.sh          S3 + IAM + security group + budget (idempotent)
│   ├── launch-instance.sh    start a GPU box (confirms cost first)
│   ├── stop-instance.sh      stop compute, keep the disk
│   ├── terminate-instance.sh destroy (requires typing TERMINATE)
│   ├── status.sh             what is running and what it costs
│   ├── check-quota.sh        quota state + live spot prices
│   ├── sync-code.sh          laptop -> S3 -> instance
│   ├── fetch-datasets.sh     download source data (run on the instance)
│   ├── run-m1.sh             data pipeline, end to end
│   ├── run-tierA.sh          baseline experiment set
│   ├── progress.sh           live training progress and ETA
│   ├── userdata.sh           instance first-boot provisioning
│   └── policies/*.json       IAM trust and permission policies
└── src/rdd/
    ├── verify/
    │   ├── manifest_audit.py      audit the dataset before downloading it
    │   ├── sequence_locality.py   measure whether filename order is temporal
    │   ├── inspect_dedup.py       render duplicate clusters for inspection
    │   └── progress.py            training progress, pace, ETA, cost
    ├── data/
    │   ├── voc_to_coco.py         VOC -> canonical COCO store (strict)
    │   ├── dedup.py               perceptual-hash de-duplication (banded LSH)
    │   ├── splits.py              grouped splits, hashed
    │   ├── to_yolo.py             COCO -> YOLO view
    │   └── _make_fixture.py       synthetic test data
    ├── train/train_yolo.py        baseline trainer, S3 checkpointing
    └── eval/aggregate.py          mean±std, per-class, per-country
```

---

## Getting started

Nothing below needs cloud access or a dataset download. This is the fastest way to confirm
your environment works and to exercise the pipeline end to end.

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

If both succeed you have a working environment. This is the same smoke test CI runs.

> `requirements.txt` pins `ultralytics`, which pulls in PyTorch (~900 MB). If you only
> intend to work on the data pipeline, install everything *except* that line.

---

## Cloud infrastructure

`infra/` provisions and manages an AWS GPU training environment: an S3 bucket, a scoped IAM
role, a security group with **zero inbound rules**, a budget with alerts, and a GPU instance
that stops itself after a period of inactivity.

**These scripts will not work against the project's own account, and are not meant to.**
They are parameterised — account id, bucket name and region are all derived at runtime — so
that anyone can stand up an equivalent environment in their own account:

```bash
export AWS_REGION=us-east-1
bash infra/bootstrap.sh          # idempotent; near-free while idle
bash infra/status.sh             # what exists and what it costs
bash infra/launch-instance.sh --job baseline --dry-run
```

`bootstrap.sh` is safe to re-run. `launch-instance.sh` confirms the hourly cost before it
starts anything, and `--dry-run` shows what it would do without doing it.

> ⚠️ **These scripts create billable resources.** `launch-instance.sh` starts a GPU
> instance. Run `status.sh` before and after, and never leave an instance running
> unattended — the idle watchdog is a backstop, not a substitute for checking.

---

## Conventions

These are not stylistic preferences. Breaking one invalidates work, sometimes silently.

| | |
|---|---|
| Canonical annotation format | COCO JSON with project attributes (`severity`, `country`, `view`, `orig_class`) |
| YOLO label files | **Generated, never hand-edited.** Change the COCO store and re-export |
| Class ids | **Frozen** in `configs/classmap.yaml`; never renumber. CI enforces this |
| Label map | Derived from the labels actually present in the data. Do not substitute the one shipped with the dataset |
| Splits | By **country and duplicate cluster**, never random image; de-duplicate first |
| Seeds | Three per configuration; report mean ± std |
| Augmentation | **No vertical flip, no rotation** — orientation distinguishes longitudinal from transverse cracks, so rotating an image relabels it. CI enforces this |
| Metrics | Per-class **and** per-country, always; never a single pooled number |
| Smoke tests | Any run with `--fraction < 1.0` writes `valid_for_reporting: false`. Never report such a number |
| Tagging | Every cloud resource gets `Project=RDD-General`; every script filters on it |

---

## Contributing

`main` is protected. Nobody pushes to it directly, including the project manager.

```bash
git switch main && git pull
git switch -c us12-four-class-control     # name the branch after the backlog item
# ... work ...
git push -u origin us12-four-class-control
gh pr create --fill
```

Every pull request needs project-manager approval before it can merge. The pull request
template asks for **evidence** — the command you ran and its output. "It works" is not
evidence.

CI runs on every pull request and checks: LF line endings in shell scripts, shell syntax,
that every pipeline module imports, the fixture smoke test, that the frozen class ids are
unchanged, and that the augmentation constraints are still in place.

**Before you launch a GPU instance, tell the team.** One instance at a time.

<!-- codeowners verification, delete this branch afterwards -->
