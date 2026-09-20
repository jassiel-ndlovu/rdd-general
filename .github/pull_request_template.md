## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Backlog item

<!-- Taiga reference, e.g. US-4 / TASK-12. If there is no backlog item, say why. -->

## Evidence

<!--
Paste the command you ran and its output. "It works" is not evidence.
For a pipeline change, the fixture run is usually enough:

    PYTHONPATH=src python -m rdd.data._make_fixture --root /tmp/fixture
    PYTHONPATH=src python -m rdd.data.voc_to_coco --root /tmp/fixture \
        --source rdd2022 --split train --out /tmp/fixture/out.json --no-strict
-->

```
```

## If this changes a safeguard

<!-- Delete this section if it does not apply. -->

- [ ] I observed the safeguard's effect change — not just the code
- [ ] The number it controls actually moved, and I have said what it moved from and to

> A fix whose effect you have not observed is not a fix. This rule was written after a
> correctly-reasoned change to the de-duplication logic turned out to be inert, because a
> flag elsewhere was still overriding it. The cluster count did not move, which is the only
> reason it was caught.

## If this reports results

- [ ] Class count, seed and split SHA-256 stated
- [ ] Run was full, i.e. `valid_for_reporting: true` — not a fraction
- [ ] Per-class **and** per-country numbers, not a single pooled figure
- [ ] No comparison drawn against a published hidden-test score

## Checklist

- [ ] No data, weights, images or credentials committed
- [ ] Shell scripts are LF, not CRLF
- [ ] Frozen class ids in `configs/classmap.yaml` untouched
- [ ] Augmentation constraints untouched — no vertical flip, no rotation
- [ ] Generated YOLO label files not hand-edited; the COCO store was changed and re-exported
- [ ] If this launched a GPU instance, the team was told and `infra/status.sh` shows it stopped
