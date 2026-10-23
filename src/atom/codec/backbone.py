"""Frozen backbone loading and hidden-state access.

The backbone is `gemma4-e4b` in bf16, the model the Meaning Contracts codec was validated
on. This module loads it (and, for the backbone comparison table, reads the Meaning
Contracts code's cached metadata for the other backbones), exposes mid-stack hidden
states, and never auto-downloads: `is_cached` checks the local HF cache first so a ~16 GB
pull is an explicit decision, not a side effect.

Follows the Meaning Contracts extraction code (`extraction/extract_activations.py`): same
nested-config handling (Gemma PLE nests dims under `text_config`), same MPS discipline.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any

# Model registry (hf ids from the Meaning Contracts extract_activations.py MODEL_CONFIGS).
# `license` is tracked here so every citation reads it from one place. Gemma releases ship
# under Google's own custom "Gemma Terms of Use" (a restricted-use license, not
# Apache/MIT/OSI-approved), and the registry records that for the gemma4 models too.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gemma4-e4b": {"hf_id": "google/gemma-4-E4B", "dtype": "bfloat16",
                   "license": "Gemma Terms of Use (Google, custom -- not Apache/MIT/OSI)",
                   "notes": "Gemma 4 E4B PLE; dims nested under text_config"},
    "gemma4-e2b": {"hf_id": "google/gemma-4-E2B", "dtype": "bfloat16",
                   "license": "Gemma Terms of Use (Google, custom -- not Apache/MIT/OSI)",
                   "notes": "Gemma 4 E2B (fast-iteration twin of e4b); "
                           "same PLE/GQA-style shared-KV family, smaller effective params "
                           "-- P5-1's pilot backbone"},
    "gemma3-4b": {"hf_id": "google/gemma-3-4b-pt", "dtype": "bfloat16",
                  "license": "Gemma Terms of Use (Google, custom -- not Apache/MIT/OSI)",
                  "notes": "fallback"},
    "qwen2.5-3b": {"hf_id": "Qwen/Qwen2.5-3B", "dtype": "bfloat16",
                   "license": "Apache License 2.0", "notes": "standard"},
    "llama3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B", "dtype": "bfloat16",
                    "license": "Llama 3.2 Community License (Meta, custom -- gated)",
                    "notes": "gated"},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "dtype": "bfloat16",
                    "license": "Apache License 2.0", "notes": "v3 paper"},
}
DEFAULT_BACKBONE = "gemma4-e4b"

# Meaning Contracts' cached activations (for the backbone comparison table without re-running).
_REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_ACTIVATIONS_DIR = _REPO_ROOT.parent / "final_codebase" / "results" / "activations"


def get_device(force: str | None = None) -> str:
    if force:
        return force
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def is_cached(hf_id: str) -> tuple[bool, str | None]:
    """True + local path iff the model snapshot (with weights) is in the HF cache. No download."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(hf_id, local_files_only=True)
    except Exception:
        return False, None
    has_weights = bool(
        glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "*.bin"))
    )
    return has_weights, (path if has_weights else None)


def _cfg_attr(cfg, attr: str):
    """Read a config attr, falling back to text_config (Gemma PLE nests dims there)."""
    val = getattr(cfg, attr, None)
    if val is None and hasattr(cfg, "text_config"):
        val = getattr(cfg.text_config, attr, None)
    return val


def load_backbone(
    model_key: str = DEFAULT_BACKBONE,
    device: str | None = None,
    dtype: str | None = None,
    hf_token: str | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Load tokenizer + frozen model. Refuses to download unless allow_download=True."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    device = device or get_device()
    dtype = dtype or cfg["dtype"]
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    cached, _ = is_cached(hf_id)
    if not cached and not allow_download:
        raise RuntimeError(
            f"{hf_id} is not in the local HF cache and allow_download=False. "
            "Refusing to trigger a ~16 GB gated download implicitly."
        )

    tok = AutoTokenizer.from_pretrained(hf_id, token=hf_token,
                                        local_files_only=not allow_download)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    # CUDA-only: `device_map` streams each shard straight to the GPU during
    # deserialization (via `accelerate`), instead of fully materialising the model on
    # CPU then `.to(device)`-copying it -- the latter transiently needs ~2x the model's
    # size in system RAM, which OOMs/thrashes a memory-constrained machine (e.g. 30 GB RAM,
    # no swap, vs. a ~15 GB bf16 gemma4-e4b checkpoint) well before anything reaches a
    # 46 GB GPU. MPS/CPU keep the plain `.to(device)` path (device_map's CUDA placement
    # logic does not apply there).
    load_kwargs = dict(dtype=torch_dtype, low_cpu_mem_usage=True, token=hf_token,
                       local_files_only=not allow_download)
    if device == "cuda":
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()
    if device != "cuda":
        model.to(device)
    load_s = time.time() - t0

    n_hidden = _cfg_attr(model.config, "num_hidden_layers")
    n_layers = (n_hidden + 1) if n_hidden is not None else None  # +1 embedding layer
    hidden_dim = _cfg_attr(model.config, "hidden_size")
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    return {
        "model": model, "tokenizer": tok, "hf_id": hf_id, "device": device,
        "dtype": dtype, "n_layers": n_layers, "hidden_dim": hidden_dim,
        "n_params_B": round(n_params, 3), "load_s": round(load_s, 1),
    }


def probe_hidden_states(bundle: dict, text: str = "What is 1/2 + 1/4?") -> dict[str, Any]:
    """Confirm output_hidden_states works; resolve true layer count / hidden dim from a pass."""
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    inp = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states
    return {"ok": hs is not None and len(hs) > 1,
            "n_hidden_tensors": len(hs), "hidden_dim": int(hs[0].shape[-1])}


def mid_stack_layer(n_layers: int, depth: float = 0.5) -> int:
    """Layer index nearest a target normalised depth (0 = embedding, 1 = last)."""
    return int(round(depth * (n_layers - 1)))


def find_qv_modules(model, max_examples: int = 4) -> dict[str, Any]:
    """Find the TEXT decoder's q/v projections (the LoRA targets for the binding).

    gemma-4-E4B is multimodal (it has a vision_tower with its own q/v); those are reported
    separately under `q_nontext` and are never treated as the LoRA targets.
    """
    names = [n for n, _ in model.named_modules()]
    q_all = [n for n in names if n.endswith(("q_proj", "wq", "query"))]
    v_all = [n for n in names if n.endswith(("v_proj", "wv", "value"))]

    def _text(n: str) -> bool:
        return not any(k in n for k in ("vision", "audio", "multi_modal", "mm_"))

    q_text = [n for n in q_all if _text(n)]
    v_text = [n for n in v_all if _text(n)]
    return {"has_q": bool(q_text), "has_v": bool(v_text),
            "n_text_q": len(q_text), "n_text_v": len(v_text),
            "q_example": q_text[:max_examples], "v_example": v_text[:max_examples],
            "q_nontext": [n for n in q_all if not _text(n)][:2]}


def measure_throughput(bundle: dict, n_tokens: int = 256, repeats: int = 3) -> dict[str, Any]:
    """Forward-pass throughput (tok/s) and peak device memory for a single sequence."""
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    text = "The student solved the problem. " * (n_tokens // 6)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).to(device)
    real_tokens = int(inp["input_ids"].shape[1])

    with torch.no_grad():  # warm-up (MPS graph compilation settles after a couple passes)
        for _ in range(2):
            model(**inp, output_hidden_states=True)
    if device == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            model(**inp, output_hidden_states=True)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    peak_gb = None
    if device == "mps":
        peak_gb = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    elif device == "cuda":
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    return {"tok_s": round(real_tokens * repeats / elapsed, 1),
            "prompt_tokens": real_tokens, "peak_mem_gb": peak_gb}


def read_prior_metadata(model_key: str) -> dict[str, Any] | None:
    """Pull layers/dim/params/dtype from the Meaning Contracts cached metadata (no model load)."""
    path = PRIOR_ACTIVATIONS_DIR / model_key / f"{model_key}_metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        m = json.load(f)
    return {"hf_id": m.get("hf_id"), "n_layers": m.get("num_layers"),
            "hidden_dim": m.get("hidden_dim"), "n_params_B": m.get("parameters_B"),
            "dtype": m.get("dtype"), "device": m.get("device"),
            "hidden_access": True}  # activations exist ⇒ access worked
