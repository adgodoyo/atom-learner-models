#!/usr/bin/env python3
"""Before/after panel for the belief-conditioned question writer (Figure 13), run on the
belief-blind fine-tune's panel harness unchanged.

The primary pre-registered reading is the swapped-history DiD compared against the belief-blind
fine-tune's -1.759 [-2.042, -1.501], on the same harness and the same states. Because "same
harness" is a requirement of that comparison, this file does not re-implement the panel: it
imports `ga_1_tier4_panel_common` and runs its `main`, replacing exactly the things that must
differ for a belief-conditioned adapter and asserting each replacement target still has the
expected shape. Every metric, bootstrap seed, caption, the states, the derangement and the DiD
estimator itself are the shared code path.

The replacements
  1. `attach_plain_lora` -> an m=8 / r=8 / alpha=32 basis. The belief-blind fine-tune was m=1
     with its mixing vector pinned to the constant 1.0, which made it belief-blind by
     construction.
  2. `LORA_M` 1 -> 8, so the panel's "adapter off" line still writes a correctly shaped zero
     vector when it re-embeds through the frozen codec.
  3. `build_tier1_prompt` -> the same function, wrapped to record which HISTORY each prompt was
     built from. The prompt is unchanged; this is only the join that lets the mixing vector
     follow the prompt.
  4. `generate_sampled` -> per-row `c = hyper(u)` from that history's belief, instead of a
     constant. Zero-shot stays exactly as in the shared panel -- the same loaded weights with
     the mixing vector set to zero, so Delta-W = 0 and no second checkpoint is involved.

Why the join is on the prompt and not on row order. The belief must describe the history the
model is actually reading, and in the swapped arm that is the partner's history: swapping the
prompt but not the weights would measure a mixed condition rather than a person swap. Joining on
the prompt text makes that automatic and order-independent, and a collision check fires if two
different histories ever produce the same prompt.

The no_history arm changes meaning here. An empty history gives the prior belief, identical for
every learner, so `u` and therefore `c` are identical across learners too: here that arm is
person-blind in both the prompt and the weights. It is the empty-belief case excluded from
training, it is out of the training distribution under the warm draw, and it is a degenerate
reference line only. A disclosure says so.

Input: `results/ga_1_tier4_rungc/adapter_rungc_epoch1.pt`; outputs as in the shared panel.
Licence boundary: generated text stays local.
"""
from __future__ import annotations

import inspect
import pathlib
import sys
from typing import Any, Sequence

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import ga_1_tier4_panel_common as panel  # noqa: E402
from ga_1_common import NumericProgress, stable_uint64  # noqa: E402
from ga_1_tier4_rungc import (  # noqa: E402
    HYPER_HIDDEN,
    HYPER_KIND,
    LORA_ALPHA,
    LORA_M,
    LORA_R,
    SEED,
    U_DIM,
)
from rl_2 import _frozen_filter_state  # noqa: E402

_PROMPT_EVENTS: dict[str, list] = {}      # prompt text -> the history it was built from
_QUERY_BY_EVENTS: dict[tuple, Any] = {}   # that history -> the query codec its belief is read at
_STATE: dict[str, Any] = {}               # "field", "hyper", "build_prompt", "fallback_query"


def events_key(events) -> tuple:
    """A value key for a history, stable across the list copies the panel makes."""
    return tuple((int(e["problem_id"]), int(e["order"])) for e in events)


def attach_belief_lora(model: Any, seed: Any) -> tuple[dict, Any]:
    """Same call shape as `attach_plain_lora`, but the belief-conditioned basis."""
    from atom.binding import attach_lora_basis

    return attach_lora_basis(model, m=LORA_M, r=LORA_R, alpha=LORA_ALPHA, seed=SEED)


def recording_prompt(events, episode_text, tok, max_history_tokens):
    """`build_tier1_prompt` unchanged, plus the prompt -> history record."""
    text = _STATE["build_prompt"](events, episode_text, tok,
                                  max_history_tokens=max_history_tokens)
    key = list(events)
    prior = _PROMPT_EVENTS.get(text)
    if prior is not None and [id(e) for e in prior] != [id(e) for e in key]:
        if [int(e["order"]) for e in prior] != [int(e["order"]) for e in key]:
            raise RuntimeError("prompt-collision check: two different histories "
                               "produced the same prompt, so the belief join is ambiguous")
    _PROMPT_EVENTS[text] = key
    return text


def belief_for_prompt(text: str) -> torch.Tensor:
    """u = [mu || log_sigma] for the history this prompt shows, from the FROZEN substrate.

    The query codec matters: `_filter_state` appends it as the last item and reads the belief
    at that slot, so a belief read at a fixed query is NOT the belief training used. Each
    history therefore carries its OWN episode's query, exactly as `belief_vector` did during
    training -- and in the swapped arm that is the partner's, so the swap moves the person in
    the prompt and in the weights together rather than leaving a mixed condition.
    """
    events = _PROMPT_EVENTS.get(text)
    if events is None:
        raise RuntimeError("belief join: a prompt reached generation without a history "
                           "record; the panel built it outside the wrapped constructor")
    key = events_key(events)
    if key and key not in _QUERY_BY_EVENTS:
        raise RuntimeError("belief join: a non-empty history has no query codec; the "
                           "panel drew a state this file did not enumerate")
    # The empty history is the no_history arm only. Training never saw it (warm starts
    # {3,5,10}), so no query is "the" right one; the first panel state's is used and the arm
    # is reported as the degenerate, out-of-distribution reference line it is.
    query = _QUERY_BY_EVENTS[key] if key else _STATE["fallback_query"]
    mu, log_sigma = _frozen_filter_state(_STATE["field"]["model"],
                                         sorted(events, key=lambda x: x["order"]),
                                         query, "cpu")
    return torch.cat((mu.detach().flatten(), log_sigma.detach().flatten())).float()


def mixing_vector(batch_prompts: Sequence[str], device: str, dtype: torch.dtype) -> torch.Tensor:
    """c = hyper(u) for one batch, as a tensor on `device` in `dtype`.

    The hypernetwork stays on the CPU in FP32 and only its OUTPUT crosses to the backbone --
    the same rule training used, where the net was FP32 and only `c` was cast. Keeping the
    whole 64->64->8 MLP on the CPU costs nothing at batch sizes like these and removes the
    device-mismatch class entirely: there is exactly one crossing, and it is this line.
    """
    u = torch.stack([belief_for_prompt(p) for p in batch_prompts])      # CPU, fp32
    with torch.no_grad():
        c = _STATE["hyper"](u)                                          # CPU, fp32
    return c.to(device=device, dtype=dtype)


def generate_belief_conditioned(bundle: dict, context: Any, prompts: Sequence[str],
                                adapter_on: bool, batch_seed_tag: str, batch_size: int,
                                max_prompt_tokens: int, progress: NumericProgress,
                                stage: str) -> list[str]:
    """The shared panel's `generate_sampled` with ONE thing changed: the mixing vector.

    Decoding is otherwise identical -- same temperature, top_p, per-batch content-addressed
    seeding, left padding and degenerate-output check -- because the point of this run is a
    comparison against the belief-blind fine-tune on the same harness.
    """
    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    old_pad, old_trunc = tok.padding_side, tok.truncation_side
    tok.padding_side, tok.truncation_side = "left", "left"
    dtype = next(model.parameters()).dtype
    out: list[str] = []
    try:
        model.eval()
        for offset in range(0, len(prompts), batch_size):
            batch = list(prompts[offset:offset + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_prompt_tokens).to(device)
            width = int(enc["input_ids"].shape[1])
            if adapter_on:
                context.set(mixing_vector(batch, device, dtype))
            else:
                context.set(torch.zeros((len(batch), LORA_M), device=device, dtype=dtype))
            torch.manual_seed(stable_uint64(f"{batch_seed_tag}|{offset}") % (2 ** 31))
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=True, temperature=panel.TEMPERATURE,
                                     top_p=panel.TOP_P, min_new_tokens=panel.MIN_NEW_TOKENS,
                                     max_new_tokens=panel.MAX_NEW_TOKENS, use_cache=True,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for row in gen:
                out.append(tok.decode(row[width:], skip_special_tokens=True).strip())
            progress.maybe_heartbeat(stage, min(offset + len(batch), len(prompts)), len(prompts),
                                     enforce_overrun=True)
    finally:
        tok.padding_side, tok.truncation_side = old_pad, old_trunc
    if any(not x for x in out):
        raise RuntimeError("degenerate-output check: an empty generation after decoding")
    return out


def assert_patch_targets_unchanged() -> None:
    """Refuse to run if the shared panel is not the shape this file patches.

    A silent divergence would break the one thing the comparison depends on, so this is an
    assertion and not a comment.
    """
    ours = list(inspect.signature(generate_belief_conditioned).parameters)
    theirs = list(inspect.signature(panel.generate_sampled).parameters)
    if ours != theirs:
        raise RuntimeError(f"generate_sampled signature drift:\n  panel: {theirs}\n  ours : {ours}")
    if panel.LORA_M != 1:
        raise RuntimeError(f"expected the shared panel's LORA_M == 1, found {panel.LORA_M}")
    if list(inspect.signature(panel.attach_plain_lora).parameters) != ["model", "seed"]:
        raise RuntimeError("attach_plain_lora signature drift; the belief-conditioned patch is unsafe")
    if list(inspect.signature(panel.build_tier1_prompt).parameters)[:3] != [
            "events", "episode_text", "tokenizer"]:
        raise RuntimeError("build_tier1_prompt signature drift; the belief join is unsafe")


def main() -> int:
    assert_patch_targets_unchanged()

    from atom.binding import Hypernetwork

    ckpt_path = REPO / "results/ga_1_tier4_rungc/adapter_rungc_epoch1.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("hyper_cfg", {"u_dim": U_DIM, "kind": HYPER_KIND, "hidden": HYPER_HIDDEN})
    net = Hypernetwork(int(cfg["u_dim"]), LORA_M, kind=str(cfg["kind"]),
                       hidden=int(cfg["hidden"]), seed=SEED)
    net.load_state_dict(ckpt["hyper"])
    net.eval()                      # stays FP32; only its output c is cast, as in training
    _STATE["hyper"] = net

    real_field = panel.load_field
    _STATE["build_prompt"] = panel.build_tier1_prompt

    def field_hook(substrate: str, raw_dir: str, device: str):
        f = real_field(substrate, raw_dir, device)
        _STATE["field"] = f
        # Enumerate the same states the panel will, and record each history's own query codec.
        # This mirrors the panel's spec loop exactly; if it ever drifts, `belief_for_prompt`
        # fails closed on the missing key rather than silently reading at the wrong query.
        for start in panel.PANEL_STARTS:
            for ep in f["scenarios"][f"val_start{start}"]:
                _QUERY_BY_EVENTS[events_key(ep["prefix_events"])] = ep["targets"][0]["codec"]
        first = f["scenarios"][f"val_start{panel.PANEL_STARTS[0]}"][0]
        _STATE["fallback_query"] = first["targets"][0]["codec"]
        return f

    panel.LORA_M = LORA_M
    panel.load_field = field_hook
    panel.attach_plain_lora = attach_belief_lora
    panel.build_tier1_prompt = recording_prompt
    panel.generate_sampled = generate_belief_conditioned
    return panel.main()


if __name__ == "__main__":
    raise SystemExit(main())
