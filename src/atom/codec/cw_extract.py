"""Unified one-pass extraction of the last-token, mean and boundary-mean readouts per scope.

One causal forward pass over `prefix + code` yields the last-token and mean-over-code
readouts (mirrors `extract.encode_one`'s convention) and the boundary-state trajectory
(`boundary.py`) in the same pass, so no submission is forwarded through the backbone more
than once per scope. `prefix` = the `qa_prefix_full` prefix for the prompt‖code scope, or
"" for the code-alone scope (matching `extract.extract_texts`'s convention: an empty
prefix pools over every content token).

`BoundaryCache` mirrors `extract.RawStateCache`'s resumable, hash-keyed design, extended
with a third array (`boundary_mean`) alongside `last`/`mean`. The full per-boundary
trajectory (variable length) is not persisted here: this cache serves the probe battery's
fixed-size readouts only; the episode encoder calls `boundary.extract_boundary_states`
directly when it needs the per-step sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from atom.codec.boundary import boundary_char_offsets, boundary_token_positions
from atom.codec.extract import answer_token_span


def extract_scope(bundle: dict, code: str, prefix: str = "", boundary_cap: int = 32
                  ) -> dict[str, np.ndarray]:
    """One forward pass -> {'last':(L,D), 'mean':(L,D), 'boundary_mean':(L,D)}."""
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    full = prefix + code
    if prefix:
        start, end, full_ids = answer_token_span(tok, prefix, full)
        content_tok_base = start  # answer_token_span already counts special tokens
    else:
        full_ids = tok(full, add_special_tokens=True).input_ids
        start, end = 0, len(full_ids)
        # `start=0` is deliberate (mean/last pool from the very first token, including
        # BOS, matching the convention of the reported mean/last readouts). But
        # boundary positions are computed against a SEPARATE, no-special-tokens
        # tokenization of `code` alone (boundary_token_positions' own contract), so its
        # tok_base must be the number of special tokens `add_special_tokens=True`
        # actually prepended -- NOT 0. With `tok_base=0` every boundary would be read one
        # token early (typically the BOS token itself for the first boundary); on
        # google/gemma-4-E4B `add_special_tokens=True` prepends exactly one BOS token. Computed as a length difference, not
        # hardcoded to 1, so it stays correct if a tokenizer ever prepends more.
        content_ids = tok(code, add_special_tokens=False).input_ids
        content_tok_base = len(full_ids) - len(content_ids)

    offsets = boundary_char_offsets(code, cap=boundary_cap) or [len(code)]
    positions = boundary_token_positions(tok, code, offsets, tok_base=content_tok_base)
    positions = sorted(set(p for p in positions if start <= p < end))
    if not positions:
        positions = [end - 1]

    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    n_layers, dim = len(hs), hs[0].shape[-1]
    last = np.empty((n_layers, dim), np.float32)
    mean = np.empty((n_layers, dim), np.float32)
    b_sum = np.zeros((n_layers, dim), np.float32)
    for li, h in enumerate(hs):
        h0 = h[0].float()
        last[li] = h0[end - 1].cpu().numpy()
        mean[li] = h0[start:end].mean(dim=0).cpu().numpy()
        for p in positions:
            b_sum[li] += h0[p].cpu().numpy()
    del out, hs
    if device == "mps":
        torch.mps.empty_cache()
    return {"last": last, "mean": mean, "boundary_mean": b_sum / len(positions)}


class BoundaryCache:
    """Hash-keyed, resumable store of (last, mean, boundary_mean) -- mirrors RawStateCache."""

    def __init__(self, out_dir: str | Path, n_layers: int, hidden_dim: int, capacity: int,
                 model: str, scope: str, dtype: str = "float16"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.paths = {k: self.dir / f"states_{k}.npy" for k in ("last", "mean", "boundary_mean")}
        self.n_layers, self.hidden_dim, self.dtype = n_layers, hidden_dim, dtype
        self.model, self.scope = model, scope
        shape = (capacity, n_layers, hidden_dim)

        if self.meta_path.exists() and self.paths["last"].exists():
            m = json.loads(self.meta_path.read_text())
            self.row_of_hash = {k: int(v) for k, v in m["row_of_hash"].items()}
            self._next = int(m["completed"])
            self.arrs = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in self.paths.items()}
            self.capacity = self.arrs["last"].shape[0]
        else:
            self.row_of_hash, self._next = {}, 0
            self.arrs = {k: np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)
                        for k, p in self.paths.items()}
            self.capacity = capacity
            self._save_meta()

    def has(self, h: str) -> bool:
        return h in self.row_of_hash

    def _grow(self, new_cap: int) -> None:
        del self.arrs
        shape = (new_cap, self.n_layers, self.hidden_dim)
        new_arrs = {}
        for k, path in self.paths.items():
            old = np.lib.format.open_memmap(path, mode="r")
            tmp = path.with_suffix(".npy.tmp")
            new = np.lib.format.open_memmap(tmp, mode="w+", dtype=self.dtype, shape=shape)
            new[: self._next] = old[: self._next]
            new.flush()
            del old, new
            path.unlink()
            tmp.rename(path)
            new_arrs[k] = np.lib.format.open_memmap(path, mode="r+")
        self.capacity = new_cap
        self.arrs = new_arrs

    def add(self, h: str, vecs: dict[str, np.ndarray] | None) -> None:
        if h in self.row_of_hash:
            return
        if self._next >= self.capacity:
            self._grow(max(self.capacity * 2, self._next + 1))
        r = self._next
        for k, v in vecs.items():
            self.arrs[k][r] = v.astype(self.dtype)
        self.row_of_hash[h] = r
        self._next += 1

    def flush(self) -> None:
        for a in self.arrs.values():
            a.flush()
        self._save_meta()

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps({
            "model": self.model, "scope": self.scope, "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim, "capacity": self.capacity, "dtype": self.dtype,
            "completed": self._next, "row_of_hash": self.row_of_hash,
        }))

    @property
    def n_cached(self) -> int:
        return self._next

    def states(self, readout: str) -> np.ndarray:
        return np.asarray(self.arrs[readout][: self._next])


def probe_metadata(bundle: dict) -> dict[str, Any]:
    from atom.codec.backbone import probe_hidden_states

    probe = probe_hidden_states(bundle)
    return {"n_layers": probe["n_hidden_tensors"], "hidden_dim": probe["hidden_dim"]}
