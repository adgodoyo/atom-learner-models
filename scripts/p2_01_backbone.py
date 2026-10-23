#!/usr/bin/env python
"""Backbone due diligence for gemma4-e4b, the frozen backbone of the codec and simulators.

Confirms that the backbone loads locally, that mid-stack hidden states are accessible and that
the text-decoder q/v LoRA slots used by the binding adapter exist; records tok/s and peak memory;
and tabulates recorded metadata for the other candidate backbones. Never downloads unless
--allow-download is given. Output: `reports/p2_01_backbone.md`.
Run: `python scripts/p2_01_backbone.py [--model gemma4-e4b] [--allow-download]`.
Serves §5.2 (codec backbone).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from atom.codec.backbone import (  # noqa: E402
    DEFAULT_BACKBONE,
    MODEL_REGISTRY,
    find_qv_modules,
    get_device,
    is_cached,
    load_backbone,
    measure_throughput,
    mid_stack_layer,
    probe_hidden_states,
    read_prior_metadata,
)
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

PRIOR_KEYS = ["gemma4-e4b", "qwen2.5-3b", "llama3.2-3b", "pythia-2.8b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BACKBONE, choices=list(MODEL_REGISTRY))
    ap.add_argument("--allow-download", action="store_true",
                    help="permit a (gated, ~16 GB) HF download if not cached")
    args = ap.parse_args()

    logger = RunLogger(config={"task": "p2_01_backbone", "model": args.model,
                               "device": get_device(), "allow_download": args.allow_download},
                       data_version="none")
    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    cached, cache_path = is_cached(hf_id)
    logger.heartbeat(step=0, force=True, stage="cache check", model=args.model,
                     hf_id=hf_id, cached=cached)

    metrics: dict[str, object] = {}
    # Prior-evidence rows (no model load).
    for k in PRIOR_KEYS:
        pm = read_prior_metadata(k)
        if pm:
            params = f"{pm['n_params_B']}B" if pm["n_params_B"] is not None else "?B"
            metrics[f"{k} (prior)"] = (
                f"{pm['n_layers']} L · {pm['hidden_dim']} d · {params} · "
                f"access✓ · {pm['dtype']}/{pm['device']}"
            )

    live_ok = False
    notes = [f"chosen backbone: {args.model} ({hf_id}); cached={cached}"]
    if cached or args.allow_download:
        logger.heartbeat(step=1, force=True, stage="loading backbone (may take ~1 min)")
        bundle = load_backbone(args.model, allow_download=args.allow_download)
        logger.heartbeat(step=2, force=True, stage="loaded", n_layers=bundle["n_layers"],
                         hidden_dim=bundle["hidden_dim"], params_B=bundle["n_params_B"],
                         load_s=bundle["load_s"])
        probe = probe_hidden_states(bundle)
        qv = find_qv_modules(bundle["model"])
        thr = measure_throughput(bundle)
        n_layers = probe["n_hidden_tensors"]
        # Layer is NOT chosen here (that is the layer x pooling x k sweep). Report the
        # candidate bands: mean-pooling peaks early-mid (~.25-.40), last-token late (~.60-.71).
        mean_band = (mid_stack_layer(n_layers, 0.25), mid_stack_layer(n_layers, 0.40))
        last_band = (mid_stack_layer(n_layers, 0.60), mid_stack_layer(n_layers, 0.71))
        live_ok = probe["ok"] and qv["has_q"] and qv["has_v"]

        metrics[f"{args.model} (LIVE)"] = (
            f"{n_layers} L (0-{n_layers - 1}) · {probe['hidden_dim']} d · {bundle['n_params_B']}B · "
            f"access{'✓' if probe['ok'] else '✗'} · "
            f"{thr['tok_s']} tok/s · {thr['peak_mem_gb']} GB · "
            f"text q/v{'✓' if qv['has_q'] and qv['has_v'] else '✗'}"
        )
        logger.log_metrics(model=args.model, **probe, **thr,
                           has_text_q=qv["has_q"], has_text_v=qv["has_v"],
                           n_text_q=qv["n_text_q"], n_text_v=qv["n_text_v"],
                           mean_band=str(mean_band), last_band=str(last_band),
                           load_s=bundle["load_s"], n_params_B=bundle["n_params_B"])
        notes += [
            f"hidden-state access: {probe['n_hidden_tensors']} layer tensors (0-{n_layers - 1}), "
            f"dim {probe['hidden_dim']}",
            f"codec layer chosen by the separate (layer×pooling×k) sweep — candidates: "
            f"mean-pool L{mean_band[0]}-{mean_band[1]} (depth .25-.40), "
            f"last-token L{last_band[0]}-{last_band[1]} incl. L30 prior (depth .60-.71)",
            f"text-decoder q/v LoRA slots (binding-adapter targets): {qv['n_text_q']} q + {qv['n_text_v']} v, "
            f"e.g. {qv['q_example'][:1]}",
            f"multimodal: vision q/v excluded from LoRA targets, e.g. {qv['q_nontext']}",
            f"throughput {thr['tok_s']} tok/s on {bundle['device']} (full forward, all layers), "
            f"peak {thr['peak_mem_gb']} GB, load {bundle['load_s']}s",
        ]
        status = "ok" if live_ok else "gate-fail"
    else:
        notes += [
            "gemma4-e4b is NOT in the local HF cache — live due-diligence is BLOCKED.",
            "Re-run with --allow-download (needs Gemma gated access on the HF account) to pull it,",
            "or point ATOM at an existing local copy. No download was triggered.",
        ]
        status = "blocked"

    logger.summary(status=status, live_ok=live_ok, cached=cached)
    passed = live_ok
    gate_report(
        path="reports/p2_01_backbone.md",
        title="P2-1 - Backbone due-diligence (gemma4-e4b)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "device": logger.config.get("device", "-"),
              "status": status},
        metrics=metrics,
        passed=passed,
        gate_desc="chosen backbone loads; mid-stack hidden-state access + q/v LoRA slots "
                  "confirmed; tok/s + peak mem recorded; backbone table produced.",
        notes=notes,
    )
    print(f"\nP2-1: {status.upper()} · report: reports/p2_01_backbone.md")
    if status == "blocked":
        print("  gemma4-e4b not cached; re-run with --allow-download (gated) to pull ~16 GB.")
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
