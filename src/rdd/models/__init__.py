"""Make this project's custom modules visible to Ultralytics.

parse_model resolves a module name from a model YAML through its own globals,
so a class defined elsewhere is not found. register() injects our classes into
that namespace and adds them to torch.serialization's allow-list, without which
a checkpoint containing one cannot be unpickled.

Call it before building or loading a model. Every evaluation path needs it, not
only the trainer. Idempotent.
"""
from __future__ import annotations

from rdd.models.modules import DySample, StripAttention

__all__ = ["DySample", "StripAttention", "register"]

CUSTOM = (StripAttention, DySample)

_done = False


def register() -> None:
    """Make the custom modules resolvable by name, and loadable from a .pt."""
    global _done
    if _done:
        return

    from ultralytics.nn import tasks

    for cls in CUSTOM:
        setattr(tasks, cls.__name__, cls)

    # Checkpoints pickle the class, so a restricted torch.load needs it allowed.
    # Absent on older torch builds, hence the guard rather than an import at top.
    try:
        import torch.serialization as ts

        if hasattr(ts, "add_safe_globals"):
            ts.add_safe_globals(list(CUSTOM))
    except Exception:                                 # noqa: BLE001
        pass

    _done = True


def transfer_weights(model, checkpoint: str, max_shift: int = 6) -> tuple[int, int]:
    """Transfer pretrained weights into a modified architecture.

    Ultralytics' .load() matches state-dict keys verbatim, and keys carry the
    layer index, so inserting a module renumbers every later layer and their
    weights are dropped without warning. Measured here at roughly 40% of the
    model for two variants. A variant initialised from 60% of the pretrained
    weights is not comparable with a fully pretrained baseline.

    A key that does not match by name is retried at a lower index, accepting
    only an exact shape match within max_shift. Insertions move layers later,
    so candidates are searched downwards and the nearest index wins.

    Returns (transferred, total) over the destination state dict.
    """
    import re
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    src_sd = src.float().state_dict() if hasattr(src, "state_dict") else src

    dst = model.model if hasattr(model, "model") else model
    dst_sd = dst.state_dict()

    # suffix -> {index: tensor}, for the positional retry
    by_suffix: dict[str, dict[int, "torch.Tensor"]] = {}
    for k, v in src_sd.items():
        m = re.match(r"^(model\.)(\d+)(\..*)$", k)
        if m:
            by_suffix.setdefault(m.group(1) + "*" + m.group(3), {})[int(m.group(2))] = v

    out, n_named, n_shifted = {}, 0, 0
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            out[k] = src_sd[k]
            n_named += 1
            continue
        m = re.match(r"^(model\.)(\d+)(\..*)$", k)
        if not m:
            continue
        cand = by_suffix.get(m.group(1) + "*" + m.group(3), {})
        i = int(m.group(2))
        for j in range(i, max(-1, i - max_shift - 1), -1):
            if j in cand and cand[j].shape == v.shape:
                out[k] = cand[j]
                n_shifted += 1
                break

    missing = dst.load_state_dict(out, strict=False)
    total = len(dst_sd)
    print(f"  transferred {len(out)}/{total} tensors "
          f"({n_named} by name, {n_shifted} by position); "
          f"{len(missing.missing_keys)} left initialised")
    return len(out), total
