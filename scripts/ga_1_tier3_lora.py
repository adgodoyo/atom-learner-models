#!/usr/bin/env python3
"""Shared helpers for the fine-tuned LLM selector (Figure 12): plain (unconditioned) LoRA on the
backbone's text q/v projections.

The binding adapter in `atom.binding.hooked_backbone` is hypernetwork-conditioned: `DeltaW(u)`
varies per learner through a mixing vector `c(u)`. The selector needs the opposite -- one adapter
for the whole selection policy -- so this module implements the plain form and reuses that
module's `_is_text_qv` targeting rule verbatim: text-decoder q/v only, never the multimodal
tower's. Fixed by the pre-registered design, not tuned here: r=16, alpha=32, q/v only, backbone
frozen.
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch
from torch import nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atom.binding.hooked_backbone import _is_text_qv, _set_submodule  # noqa: E402

LORA_R = 16
LORA_ALPHA = 32


class PlainLoRALinear(nn.Module):
    """frozen(x) + (alpha/r) * B(A(x)).  B zero-init ⇒ an exact no-op at step 0."""

    def __init__(self, frozen: nn.Linear, r: int = LORA_R, alpha: int = LORA_ALPHA,
                 seed: int = 0):
        super().__init__()
        for p in frozen.parameters():
            p.requires_grad = False
        self.frozen = frozen
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        gen = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(self.r, frozen.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
        self.lora_a = nn.Parameter(a.to(frozen.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(frozen.out_features, self.r,
                                               dtype=frozen.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.frozen(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_a), self.lora_b)
        return base + self.scaling * delta


def attach_plain_lora(model: nn.Module, r: int = LORA_R, alpha: int = LORA_ALPHA,
                      seed: int = 0) -> dict[str, PlainLoRALinear]:
    """Freeze everything, then wrap every text q/v Linear in place.

    Freezing the WHOLE model first is deliberate: wrapping only the targeted Linears leaves embeddings, k/o projections, MLP, norms and
    the LM head at HF's default `requires_grad=True`, so backward silently computes
    gradients for the entire backbone. Assert the trainable count afterwards.
    """
    for p in model.parameters():
        p.requires_grad = False
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and _is_text_qv(n)]
    if not targets:
        raise RuntimeError("no text q/v projections found to adapt")
    out: dict[str, PlainLoRALinear] = {}
    for i, (name, mod) in enumerate(targets):
        wrapped = PlainLoRALinear(mod, r=r, alpha=alpha, seed=seed + i)
        wrapped.to(mod.weight.device)
        _set_submodule(model, name, wrapped)
        out[name] = wrapped
    return out


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state_dict(adapters: dict[str, PlainLoRALinear]) -> dict[str, torch.Tensor]:
    state = {}
    for name, mod in adapters.items():
        state[f"{name}.lora_a"] = mod.lora_a.detach().cpu().clone()
        state[f"{name}.lora_b"] = mod.lora_b.detach().cpu().clone()
    return state


def load_adapter_state(adapters: dict[str, PlainLoRALinear],
                       state: dict[str, torch.Tensor]) -> None:
    missing = [n for n in adapters if f"{n}.lora_a" not in state]
    if missing:
        raise KeyError(f"adapter state is missing {len(missing)} layers, e.g. {missing[:3]}")
    with torch.no_grad():
        for name, mod in adapters.items():
            mod.lora_a.copy_(state[f"{name}.lora_a"].to(mod.lora_a.device, mod.lora_a.dtype))
            mod.lora_b.copy_(state[f"{name}.lora_b"].to(mod.lora_b.device, mod.lora_b.dtype))


class adapters_disabled:
    """Context manager: run the frozen base model through the same wrapped graph.

    Used for the KL anchor's `pi_base`. Scaling is zeroed rather than the modules being
    unwrapped, so the base and adapted passes traverse an identical code path -- if they
    did not, any difference between them could be an artifact of the wrapper rather than
    of the adapter, and the anchor term would be measuring the wrong thing.
    """

    def __init__(self, adapters: dict[str, PlainLoRALinear]):
        self.adapters = adapters
        self.saved: dict[str, float] = {}

    def __enter__(self):
        for name, mod in self.adapters.items():
            self.saved[name] = mod.scaling
            mod.scaling = 0.0
        return self

    def __exit__(self, *exc):
        for name, mod in self.adapters.items():
            mod.scaling = self.saved[name]
        return False
