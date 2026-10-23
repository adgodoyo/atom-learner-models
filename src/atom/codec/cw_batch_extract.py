"""Batched, single-layer extraction for full-corpus-scale encoding.

`cw_extract.extract_scope` is one item per forward pass, all 43 layers stored -- fine at
probe-battery scale (thousands of items, a layer sweep needing every layer). The full
S19+F19 corpus (171k+ unique code texts) is encoded at the already-chosen production layer
only (L15, selected by the probe battery) -- storing 43 layers would be a ~40x disk and
transfer cost for data nothing reads, and one-item-at-a-time forward passes were measured
at 41-104 hours on a laptop for this corpus. This module computes the exact same three
readouts (last, mean, boundary_mean) at ONE layer, batched with length bucketing so
padding waste stays low: the throughput comes from batching a forward-only pass, not from
a bigger card.

Equivalence with `extract_scope` is the whole point (same numbers, faster path) and is
pinned by `tests/test_cw_batch_extract.py` against the real backbone -- but NOT
bit-identically, at ANY batch size, including 1. Two distinct, measured sources: (i) right-padding a real item to a longer batch-mate's length perturbs
its own real-token hidden states by up to ~0.25 absolute at L15, reproduced under BOTH
`sdpa` and `eager` attention (not an sdpa fast-path quirk); (ii) even at batch size 1
(zero padding), merely PASSING an explicit all-ones `attention_mask` -- which this
module always does, while `extract_scope` never passes one at all -- causes a smaller
but still nonzero divergence. Both are genuine properties of this backbone's bf16
batched/masked forward path, not bugs here. Cosine similarity between batched and
single-item vectors measured 0.9998-1.0000 across short/medium/long code at every batch
size tested -- directionally negligible, and every downstream consumer (item-geometry
probes, the codec's own deflated-cosine/whitened-Mahalanobis metrics) reads direction,
not raw coordinates. The equivalence tests therefore check cosine, never elementwise
closeness.

Right-padding is used deliberately (not the tokenizer's left-padding default): under a
causal mask, a real token's hidden state should in principle depend only on itself and
earlier real tokens, so trailing pad tokens after the real content should not influence
them -- right-padding needs no custom position-id surgery this way. In practice, per the
measurement above, this holds only up to the bf16-batching noise floor described above,
not exactly. `attention_mask` is still passed to the model for correctness/robustness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def bucket_by_length(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Greedy length-bucketing: sort ascending, chunk into `batch_size`-sized groups of
    ORIGINAL indices. Minimises padding within each batch; does not reorder across
    batches (caller reassembles by index, order of batches is irrelevant)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def _boundary_positions_for_item(tok, code: str, full_ids: list[int], boundary_cap: int
                                 ) -> list[int]:
    """Content-token boundary positions within `full_ids` (which includes BOS/specials),
    as in `cw_extract.extract_scope`: `tok_base` is the actual
    special-token count, derived as a length difference, never hardcoded."""
    from atom.codec.boundary import boundary_char_offsets, boundary_token_positions

    content_ids = tok(code, add_special_tokens=False).input_ids
    tok_base = len(full_ids) - len(content_ids)
    offsets = boundary_char_offsets(code, cap=boundary_cap) or [len(code)]
    positions = boundary_token_positions(tok, code, offsets, tok_base=tok_base)
    positions = sorted(set(p for p in positions if 0 <= p < len(full_ids)))
    return positions or [len(full_ids) - 1]


def extract_batch_l15(bundle: dict, texts: list[str], layer: int, boundary_cap: int = 32,
                      max_length: int = 4096) -> dict[str, np.ndarray]:
    """One padded, batched forward pass -> {'last','mean','boundary_mean'} each (N, D)
    at `layer` only. `texts` should already be length-bucketed by the caller for the
    padding-waste win; correctness does not depend on it."""
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    saved_padding_side = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok(texts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt", return_offsets_mapping=False, add_special_tokens=True)
    finally:
        tok.padding_side = saved_padding_side

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    lengths = attention_mask.sum(dim=1).tolist()  # real (non-pad) token count per item

    # Call the DECODER, not the full CausalLM head: the full
    # `model(...)` forward computes lm_head logits over the entire vocab for every
    # position in the batch, then divides by `final_logit_softcapping` -- a value we
    # never read (only `hidden_states` is used below). At batch_size=32 this allocates a
    # single ~13 GiB tensor and runs a 48 GB card out of memory; at batch_size=1 the same
    # waste is small enough to go unnoticed (as in `cw_extract.extract_scope`'s
    # single-item calls). `model.get_decoder()` returns the text transformer
    # stack with no LM head, so `output_hidden_states=True` is produced directly without
    # ever materialising logits -- same hidden states, no wasted compute or memory.
    decoder = model.get_decoder()
    with torch.no_grad():
        out = decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h = out.hidden_states[layer].float()  # (N, T, D)
    del out
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    n, dim = len(texts), h.shape[-1]
    last = np.empty((n, dim), np.float32)
    mean = np.empty((n, dim), np.float32)
    b_mean = np.empty((n, dim), np.float32)

    for i, text in enumerate(texts):
        length = int(lengths[i])
        last[i] = h[i, length - 1].cpu().numpy()
        mean[i] = h[i, :length].mean(dim=0).cpu().numpy()

        full_ids = input_ids[i, :length].tolist()
        positions = _boundary_positions_for_item(tok, text, full_ids, boundary_cap)
        b_mean[i] = h[i, positions].mean(dim=0).cpu().numpy()

    return {"last": last, "mean": mean, "boundary_mean": b_mean}


def extract_many_l15(bundle: dict, texts: list[str], layer: int, batch_size: int,
                     boundary_cap: int = 32, max_length: int = 4096,
                     progress_cb: Any = None) -> dict[str, np.ndarray]:
    """Length-bucketed batched extraction over an arbitrary list of texts, reassembled
    to the CALLER's original order. `progress_cb(n_done, n_total)` if given."""
    tok = bundle["tokenizer"]
    lengths = [len(tok(t, add_special_tokens=True).input_ids) for t in texts]
    buckets = bucket_by_length(lengths, batch_size)

    n, dim = len(texts), bundle["hidden_dim"]
    last = np.empty((n, dim), np.float32)
    mean = np.empty((n, dim), np.float32)
    b_mean = np.empty((n, dim), np.float32)

    done = 0
    for idx in buckets:
        batch_texts = [texts[i] for i in idx]
        out = extract_batch_l15(bundle, batch_texts, layer=layer, boundary_cap=boundary_cap,
                                max_length=max_length)
        for j, i in enumerate(idx):
            last[i] = out["last"][j]
            mean[i] = out["mean"][j]
            b_mean[i] = out["boundary_mean"][j]
        done += len(idx)
        if progress_cb is not None:
            progress_cb(done, n)

    return {"last": last, "mean": mean, "boundary_mean": b_mean}


class SingleLayerVectorCache:
    """Hash-keyed, resumable store of (last, mean, boundary_mean) at ONE layer -- same
    growable-memmap design as `cw_extract.BoundaryCache`, minus the layer dimension
    (the full-corpus encode only ever wants L15; storing all ~43 layers the way
    `BoundaryCache` does would be a ~40x waste of disk/S3 transfer for numbers nothing
    reads at this scale)."""

    READOUTS = ("last", "mean", "boundary_mean")

    def __init__(self, out_dir: str | Path, hidden_dim: int, capacity: int, model: str,
                scope: str, dtype: str = "float16"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.paths = {k: self.dir / f"states_{k}.npy" for k in self.READOUTS}
        self.hidden_dim, self.dtype = hidden_dim, dtype
        self.model, self.scope = model, scope
        shape = (capacity, hidden_dim)

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
        shape = (new_cap, self.hidden_dim)
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

    def add(self, h: str, vecs: dict[str, np.ndarray]) -> None:
        if h in self.row_of_hash:
            return
        if self._next >= self.capacity:
            self._grow(max(self.capacity * 2, self._next + 1))
        r = self._next
        for k in self.READOUTS:
            self.arrs[k][r] = vecs[k].astype(self.dtype)
        self.row_of_hash[h] = r
        self._next += 1

    def flush(self) -> None:
        for a in self.arrs.values():
            a.flush()
        self._save_meta()

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps({
            "model": self.model, "scope": self.scope, "hidden_dim": self.hidden_dim,
            "capacity": self.capacity, "dtype": self.dtype,
            "completed": self._next, "row_of_hash": self.row_of_hash,
        }))

    @property
    def n_cached(self) -> int:
        return self._next

    def states(self, readout: str) -> np.ndarray:
        return np.asarray(self.arrs[readout][: self._next])


def extract_many_l15_into_cache(bundle: dict, hash_to_text: dict[str, str], layer: int,
                                batch_size: int, cache: SingleLayerVectorCache,
                                boundary_cap: int = 32, max_length: int = 4096,
                                sync_every: int | None = None,
                                sync_cb: Callable[[], None] | None = None,
                                progress_cb: Any = None) -> None:
    """Encodes every hash in `hash_to_text` NOT already in `cache`, length-bucketed,
    flushing to disk (and calling `sync_cb()`, if given, every >= `sync_every` NEWLY
    encoded items) as it goes. This is the resumability + periodic-offsite-sync
    primitive: a lost machine loses at most `sync_every` items of already-computed
    work, never the whole run (an end-of-run-only upload could leave hours of GPU work
    unrecoverable)."""
    todo = [h for h in hash_to_text if not cache.has(h)]
    if not todo:
        return
    tok = bundle["tokenizer"]
    lengths = [len(tok(hash_to_text[h], add_special_tokens=True).input_ids) for h in todo]
    buckets = bucket_by_length(lengths, batch_size)

    since_sync, done = 0, 0
    for idx in buckets:
        hs = [todo[i] for i in idx]
        texts = [hash_to_text[h] for h in hs]
        out = extract_batch_l15(bundle, texts, layer=layer, boundary_cap=boundary_cap,
                                max_length=max_length)
        for j, h in enumerate(hs):
            cache.add(h, {"last": out["last"][j], "mean": out["mean"][j],
                         "boundary_mean": out["boundary_mean"][j]})
        since_sync += len(hs)
        done += len(hs)
        if progress_cb is not None:
            progress_cb(done, len(todo))
        if sync_every is not None and since_sync >= sync_every:
            cache.flush()
            if sync_cb is not None:
                sync_cb()
            since_sync = 0
    cache.flush()
    if sync_cb is not None:
        sync_cb()
