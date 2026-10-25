"""Make this project's custom modules visible to Ultralytics.

``ultralytics.nn.tasks.parse_model`` turns a module name from a model YAML into
a class with ``globals()[name]`` — its own module globals. A class defined
anywhere else is simply not found, and the YAML fails with a ``KeyError`` that
names the module and nothing else.

So ``register()`` injects our classes into that namespace. It also registers
them with ``torch.serialization`` where that is available, because a checkpoint
containing a custom module cannot be unpickled under PyTorch's restricted load
unless the class is on the allow-list.

Call it before building *or loading* a model. Loading matters as much as
building: a trained ``.pt`` holding a ``StripAttention`` is unreadable without
this, so every evaluation path needs it too, not only the trainer.

Idempotent, so calling it from several entry points costs nothing.
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
    """Transfer pretrained weights into a modified architecture, by name and
    then by position.

    Ultralytics' own ``.load()`` matches state-dict keys verbatim. Keys carry
    the layer's index — ``model.19.cv1.conv.weight`` — so inserting a module
    renumbers every layer after it and none of them match any more. Measured on
    this project's variants: the stride-4 head transferred 297 of a possible
    499 tensors and strip attention 288, meaning roughly 40% of the pretrained
    weights were silently discarded and those layers started random.

    That is a confound, not a detail. The baseline this is compared against was
    fully pretrained, so a variant that scores worse could be losing on
    initialisation rather than on architecture.

    This recovers the shifted layers: a key that does not match by name is
    retried against the same suffix at a lower index, accepting only an exact
    shape match and only a shift within ``max_shift``. Insertions move layers
    later, never earlier, so candidates are searched downwards; the nearest
    index wins, which is the correct one when a single block has been inserted.

    Returns (transferred, total) counted over the destination state dict.
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
