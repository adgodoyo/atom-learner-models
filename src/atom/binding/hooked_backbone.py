"""Hooking the shared LoRA basis onto a frozen backbone's q/v projections.

DeltaW^l(u) = alpha * sum_i c_i(u) A_i^l B_i^l -- ONE shared mixing
vector c(u) from the hypernetwork, broadcast across a PER-LAYER LoRABasis on every targeted
q/v projection (the basis pairs A^l/B^l differ by layer and by q-vs-v; c does not -- the
hypernetwork runs once per example, not once per layer).

Threading c(u) through a frozen HF model's forward is the practical problem: nn.Linear's
forward takes only `x`, and the backbone must stay untouched (frozen means frozen -- not
just requires_grad=False, its own forward code is never edited). Solution: replace each
targeted nn.Linear in place with a thin wrapper that reads the current mixing vector off a
shared, externally-set MixingContext at call time -- the training loop sets `context.c`
once per batch (or per K-sample-expanded batch) immediately before invoking the model, the
same out-of-band-forward-state pattern HF's own cache objects use.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.binding.basis import LoRABasis


class MixingContext:
    """Mutable holder for the current forward's mixing vectors c: (batch, m).

    `batch` must match the leading (pre-sequence) dim of whatever forward call is about to
    run through the hooked layers. The training loop owns setting this immediately before
    each `model(...)` call; nothing inside the model touches it.
    """

    def __init__(self) -> None:
        self.c: torch.Tensor | None = None

    def set(self, c: torch.Tensor) -> None:
        self.c = c

    def clear(self) -> None:
        self.c = None


class LoRAHookedLinear(nn.Module):
    """Wraps one frozen nn.Linear: forward = frozen(x) + basis.apply(x, context.c).

    `frozen`'s own parameters are forced requires_grad=False here (not left to caller
    discipline -- a wrapped layer is frozen by construction). Only `basis`'s A/B (and
    upstream, the hypernetwork producing c) are trainable. c is broadcast from (batch, m) to
    (batch*seq, m) by repeat_interleave along the sequence dimension: personalisation is
    per-example (per-u-sample), not per-token, so every token of one sequence shares one c.
    """

    def __init__(self, frozen: nn.Linear, basis: LoRABasis, context: MixingContext) -> None:
        super().__init__()
        for p in frozen.parameters():
            p.requires_grad = False
        self.frozen = frozen
        self.basis = basis
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.frozen(x)
        c = self.context.c
        if c is None:
            raise RuntimeError(
                "LoRAHookedLinear called with no mixing vector set -- "
                "call context.set(c) before the model forward."
            )
        *lead, in_dim = x.shape
        n_tokens = 1
        for d in lead:
            n_tokens *= d
        x_flat = x.reshape(n_tokens, in_dim)

        batch = c.shape[0]
        if batch == 0 or n_tokens % batch != 0:
            raise ValueError(
                f"mixing vector batch {batch} does not evenly divide the forward's "
                f"{n_tokens} flattened positions (leading dims {lead})"
            )
        repeat = n_tokens // batch
        c_flat = c.repeat_interleave(repeat, dim=0) if repeat > 1 else c

        # The frozen backbone runs in bf16; the basis stays fp32 for the low-rank
        # branch's numerical stability (standard LoRA practice -- bf16's ~3 decimal digits
        # of mantissa is a bad place to accumulate a rank-8x32 sum). Up-cast the activation
        # into the branch, down-cast the delta back to the frozen output's own dtype before
        # adding, so `base + delta` never silently promotes the whole residual stream to fp32.
        # `context.c` is set by the training loop and may be on a different device than
        # the model (e.g. computed on CPU by the hypernetwork before the K-sample forward);
        # basis params are the authority on both device and dtype for this branch.
        basis_dtype, basis_device = self.basis.A.dtype, self.basis.A.device
        x_for_basis = x_flat.to(device=basis_device, dtype=basis_dtype)
        c_for_basis = c_flat.to(device=basis_device, dtype=basis_dtype)
        delta = self.basis.apply(x_for_basis, c_for_basis)
        if delta.dtype != base.dtype:
            delta = delta.to(base.dtype)
        return base + delta.reshape(*lead, self.basis.out_dim)


def _is_text_qv(name: str) -> bool:
    """Text-decoder q/v only -- gemma-4-E4B is multimodal and its vision-tower q/v are never
    LoRA targets (same rule as codec.backbone.find_qv_modules, kept in sync deliberately)."""
    is_qv = name.endswith(("q_proj", "v_proj", "wq", "wv", "query", "value"))
    is_text = not any(k in name for k in ("vision", "audio", "multi_modal", "mm_"))
    return is_qv and is_text


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


def attach_lora_basis(
    model: nn.Module, m: int = 32, r: int = 8, alpha: float | None = None, seed: int = 0,
) -> tuple[dict[str, LoRABasis], MixingContext]:
    """Replace every text q/v nn.Linear in `model` (in place) with a LoRAHookedLinear.

    One LoRABasis PER targeted layer (its own A^l/B^l) but ONE shared MixingContext, so a
    single c(u) computed once per batch by the hypernetwork reaches every layer unchanged.
    Returns {module_name: basis} (for the optimiser's param groups, and for the
    checkpoint/resume test to assert checkpointed state) and the context (for the
    training loop to set c on before each forward).

    Freezes EVERY parameter already on `model` first -- not just the ones inside the
    wrapped q/v Linears. Without this, only the specific wrapped Linears end up frozen (LoRAHookedLinear's own __init__ does
    that part), but everything else already on the model -- embeddings, k_proj, o_proj,
    MLP, norms, the LM head -- silently keeps its default requires_grad=True from HF's own
    from_pretrained, so `loss.backward()` computes gradients for the ENTIRE multi-billion-
    parameter backbone, not just the ~40M basis+hypernetwork params the architecture
    actually calls trainable. On gemma4-e2b that would be 4.97B unintended trainable
    params -- enough to OOM a backward pass that should never need that memory.
    Freezing here, once, before any wrapping happens, prevents it:
    `LoRABasis`'s own A/B are created fresh afterward with their normal requires_grad=True
    default, so nothing about the intended trainable set changes.

    Raises if no text q/v Linear modules are found -- silently attaching nothing to an
    unexpected architecture would be a much worse failure mode than a loud one here.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    context = MixingContext()
    bases: dict[str, LoRABasis] = {}
    targets = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _is_text_qv(name)
    ]
    if not targets:
        raise ValueError("no text q/v nn.Linear modules found to attach LoRA to")

    for i, (name, linear) in enumerate(targets):
        basis = LoRABasis(
            in_dim=linear.in_features, out_dim=linear.out_features,
            m=m, r=r, alpha=alpha, seed=seed + i,
        )
        # Basis lives on the same device as the frozen linear it wraps (real backbones may
        # be on mps/cuda), but stays fp32 -- only device moves here, dtype is handled per
        # forward call in LoRAHookedLinear (see its docstring on why bf16 stays out of the
        # low-rank branch).
        basis = basis.to(device=linear.weight.device)
        hooked = LoRAHookedLinear(linear, basis, context)
        _set_submodule(model, name, hooked)
        bases[name] = basis

    return bases, context


def _hooked_layers(model: nn.Module) -> list[tuple[str, "LoRAHookedLinear"]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRAHookedLinear)]


def build_second_basis(model: nn.Module, seed: int = 1000) -> dict[str, LoRABasis]:
    """Basis half of the global-LoRA control arm (`atom.binding.GlobalMixingVector` is the
    mixing-vector half): a
    SECOND, independent set of `LoRABasis` objects -- one per layer `attach_lora_basis`
    already wrapped, matching each layer's own `(m, r, alpha)` exactly -- for a control
    arm that needs its OWN trained weights on the SAME already-hooked backbone.

    Why this exists rather than a second `attach_lora_basis(model, ...)` call: that call
    replaces `nn.Linear` in place, so a second call on an ALREADY-wrapped model finds zero
    `nn.Linear` targets left to wrap (they are all `LoRAHookedLinear` now) and raises. A
    genuinely independent basis therefore cannot come from re-attaching -- and loading a
    second full backbone (~4B params) just to host one alternate LoRA basis would double
    peak VRAM for no reason, since the FROZEN weights `LoRAHookedLinear.frozen` wraps are
    identical either way. This reads each existing layer's shape straight off
    `LoRAHookedLinear.frozen` (still the original, unwrapped `nn.Linear` -- `attach_lora_
    basis` stores it, never discards it) and off the CURRENTLY active basis's own
    `(m, r, alpha)`, so the new basis is shape-matched without needing those hyperparameters
    passed in twice. Use with `set_active_basis` to switch which basis a forward pass
    reads -- `MixingContext` (the shared `c` channel `context.set(...)` writes to) is
    untouched; only which `A`/`B` matrices `c` multiplies against changes.
    """
    hooked_layers = _hooked_layers(model)
    if not hooked_layers:
        raise ValueError("build_second_basis: no LoRAHookedLinear layers found on model -- "
                         "call attach_lora_basis first")
    bases: dict[str, LoRABasis] = {}
    for i, (name, mod) in enumerate(hooked_layers):
        existing = mod.basis
        basis = LoRABasis(in_dim=mod.frozen.in_features, out_dim=mod.frozen.out_features,
                          m=existing.m, r=existing.r, alpha=existing.alpha, seed=seed + i)
        bases[name] = basis.to(device=mod.frozen.weight.device)
    return bases


def set_active_basis(model: nn.Module, bases: dict[str, LoRABasis]) -> None:
    """Swap which `LoRABasis` each `LoRAHookedLinear` in `model` currently reads from --
    O(n_layers) attribute reassignment, no forward pass touched. `bases` must cover every
    hooked layer name in `model` (`attach_lora_basis`'s or `build_second_basis`'s own
    return value); a partial `bases` dict raises rather than silently leaving some layers
    on their previous basis, which would make a control-arm forward pass a silent mix of
    both arms' weights."""
    hooked_layers = _hooked_layers(model)
    missing = [name for name, _ in hooked_layers if name not in bases]
    if missing:
        raise KeyError(f"set_active_basis: no basis provided for layers {missing}")
    for name, mod in hooked_layers:
        mod.basis = bases[name]
