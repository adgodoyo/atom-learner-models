"""Extracts raw pooled states (answer-in-context) into a resumable, hash-keyed cache.

The cache holds the *raw, undeflated* pooled states; the codec transform (deflate +
whiten) is fit and applied later, so k, whitening and even the metric stay retunable
without any model forward.

Encoding follows the answer-in-context rule: the object is `question ‖ answer`; we read
hidden states only at the **answer's token positions**. The answer span is found by the
shared-token-prefix length between the `prefix` (`question ‖ delimiter`) and the `full`
string, which is robust to boundary-token merges. Two readouts per layer, both from one
causal forward pass: last-token (the final answer token) and mean-over-answer-span.

Data licence: the cache stores derived vectors + hashed keys + interaction ids only,
never raw question/answer text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TEMPLATE = "{question}\nAnswer: {answer}"


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def qa_prefix_full(question: str, answer: str, template: str = DEFAULT_TEMPLATE) -> tuple[str, str]:
    """Return (prefix, full): prefix = everything before the answer, full = prefix+answer."""
    full = template.format(question=question, answer=answer)
    prefix = template.format(question=question, answer="")
    if not full.startswith(prefix):  # non-suffix template: fall back to locating the answer
        cut = full.rfind(answer) if answer else len(full)
        prefix = full[:cut] if cut >= 0 else full
    return prefix, full


def answer_token_span(tokenizer, prefix: str, full: str) -> tuple[int, int, list[int]]:
    """(start, end, full_ids): answer tokens are full_ids[start:end].

    start = length of the token prefix shared by `prefix` and `full` (robust to the answer's
    first token merging with the delimiter). Degenerate/empty answers fall back to the last
    token so pooling is never empty.
    """
    pre_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    full_ids = tokenizer(full, add_special_tokens=True).input_ids
    start = 0
    for a, b in zip(pre_ids, full_ids):
        if a == b:
            start += 1
        else:
            break
    end = len(full_ids)
    if start >= end:
        start = max(0, end - 1)
    return start, end, full_ids


def encode_one(bundle: dict, prefix: str, full: str) -> tuple[np.ndarray, np.ndarray]:
    """One causal forward pass -> (last (L, D), mean (L, D)) pooled over the answer span."""
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    start, end, full_ids = answer_token_span(tok, prefix, full)
    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of L tensors, each (1, T, D)
    n_layers, dim = len(hs), hs[0].shape[-1]
    last = np.empty((n_layers, dim), np.float32)
    mean = np.empty((n_layers, dim), np.float32)
    for li, h in enumerate(hs):
        # Production generation can leave adapter-conditioned hidden states attached
        # to a graph even though this frozen diagnostic forward is under no_grad().
        # Extraction is a numeric readout only: sever that graph before crossing the
        # PyTorch -> NumPy boundary (and avoid retaining it across codec rows).
        h0 = h[0].detach().float()
        last[li] = h0[end - 1].cpu().numpy()
        mean[li] = h0[start:end].mean(dim=0).cpu().numpy()
    del out, hs
    if device == "mps":
        torch.mps.empty_cache()
    return last, mean


class RawStateCache:
    """Hash-keyed, resumable store of raw pooled states (last + mean) at all layers."""

    def __init__(self, out_dir: str | Path, n_layers: int, hidden_dim: int, capacity: int,
                 model: str, template: str, dtype: str = "float16"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.last_path = self.dir / "states_last.npy"
        self.mean_path = self.dir / "states_mean.npy"
        self.n_layers, self.hidden_dim, self.capacity = n_layers, hidden_dim, capacity
        self.dtype, self.model, self.template = dtype, model, template
        shape = (capacity, n_layers, hidden_dim)

        if self.meta_path.exists() and self.last_path.exists():
            m = json.loads(self.meta_path.read_text())
            self.row_of_hash = {k: int(v) for k, v in m["row_of_hash"].items()}
            self.ids_of_hash = m.get("ids_of_hash", {})
            self._next = int(m["completed"])
            self.last = np.lib.format.open_memmap(self.last_path, mode="r+")
            self.mean = np.lib.format.open_memmap(self.mean_path, mode="r+")
            self.capacity = self.last.shape[0]  # actual on-disk capacity, not the passed arg
        else:
            self.row_of_hash, self.ids_of_hash, self._next = {}, {}, 0
            self.last = np.lib.format.open_memmap(self.last_path, mode="w+", dtype=dtype, shape=shape)
            self.mean = np.lib.format.open_memmap(self.mean_path, mode="w+", dtype=dtype, shape=shape)
            self._save_meta()

    def has(self, h: str) -> bool:
        return h in self.row_of_hash

    def _grow(self, new_cap: int) -> None:
        """Grow the memmaps to new_cap rows, preserving what's written. Amortised O(1)."""
        del self.last, self.mean  # release handles before replacing the files
        shape = (new_cap, self.n_layers, self.hidden_dim)
        for path in (self.last_path, self.mean_path):
            old = np.lib.format.open_memmap(path, mode="r")
            tmp = path.with_suffix(".npy.tmp")
            new = np.lib.format.open_memmap(tmp, mode="w+", dtype=self.dtype, shape=shape)
            new[: self._next] = old[: self._next]
            new.flush()
            del old, new
            path.unlink()
            tmp.rename(path)
        self.capacity = new_cap
        self.last = np.lib.format.open_memmap(self.last_path, mode="r+")
        self.mean = np.lib.format.open_memmap(self.mean_path, mode="r+")

    def add(self, h: str, last_vec: np.ndarray, mean_vec: np.ndarray, ids: list | None = None) -> None:
        if h not in self.row_of_hash:
            if self._next >= self.capacity:
                self._grow(max(self.capacity * 2, self._next + 1))
            r = self._next
            self.last[r] = last_vec.astype(self.dtype)
            self.mean[r] = mean_vec.astype(self.dtype)
            self.row_of_hash[h] = r
            self._next += 1
        if ids:
            self.ids_of_hash.setdefault(h, [])
            self.ids_of_hash[h] = sorted(set(self.ids_of_hash[h]) | set(map(int, ids)))

    def flush(self) -> None:
        self.last.flush()
        self.mean.flush()
        self._save_meta()

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps({
            "model": self.model, "template": self.template, "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim, "capacity": self.capacity, "dtype": self.dtype,
            "completed": self._next, "row_of_hash": self.row_of_hash,
            "ids_of_hash": self.ids_of_hash,
        }))

    @property
    def n_cached(self) -> int:
        return self._next

    def states(self, pooling: str) -> np.ndarray:
        arr = self.last if pooling == "last" else self.mean
        return np.asarray(arr[: self._next])


def build_items(interactions, problems, n: int, seed: int, template: str = DEFAULT_TEMPLATE):
    """Sample n interactions, join cleaned problem text, return [(id, prefix, full, hash)]."""
    samp = interactions.sample(min(n, len(interactions)), random_state=seed)
    body_col = "problem_body_clean" if "problem_body_clean" in problems.columns else "problem_body"
    pmap = dict(zip(problems["problem_id"], problems[body_col].astype(str)))
    items = []
    for row in samp.itertuples(index=False):
        q = pmap.get(int(row.problem_id), "")
        a = "" if row.answer_text is None or (isinstance(row.answer_text, float)) else str(row.answer_text)
        prefix, full = qa_prefix_full(q, a, template)
        items.append((int(row.id), prefix, full, text_hash(full)))
    return items


def extract_calibration(interactions, problems, bundle, out_dir, n=1000, seed=42,
                        template=DEFAULT_TEMPLATE, logger=None) -> dict[str, Any]:
    """Encode a calibration slice into a resumable raw-state cache. Returns run stats."""
    from atom.codec.backbone import probe_hidden_states

    probe = probe_hidden_states(bundle)
    n_layers, hidden_dim = probe["n_hidden_tensors"], probe["hidden_dim"]

    items = build_items(interactions, problems, n, seed, template)
    unique: dict[str, tuple[str, str]] = {}
    ids_by_hash: dict[str, list] = {}
    for iid, prefix, full, h in items:
        unique.setdefault(h, (prefix, full))
        ids_by_hash.setdefault(h, []).append(iid)

    cache = RawStateCache(out_dir, n_layers, hidden_dim, capacity=len(unique),
                          model=bundle["hf_id"], template=template)
    n_encoded = 0
    for i, (h, (prefix, full)) in enumerate(unique.items()):
        if cache.has(h):
            cache.add(h, None, None, ids=ids_by_hash[h])  # refresh ids only
            continue
        last, mean = encode_one(bundle, prefix, full)
        cache.add(h, last, mean, ids=ids_by_hash[h])
        n_encoded += 1
        if logger and (n_encoded % 25 == 0):
            cache.flush()
            logger.heartbeat(step=n_encoded, total=len(unique), stage="encoding")
    cache.flush()
    return {"n_interactions": len(items), "n_unique": len(unique),
            "n_encoded": n_encoded, "n_cached": cache.n_cached,
            "n_layers": n_layers, "hidden_dim": hidden_dim, "out_dir": str(cache.dir)}


def extract_texts(bundle, items, out_dir, logger=None) -> dict[str, Any]:
    """Encode standalone texts (e.g. problem bodies) into a resumable raw-state cache.

    items: list of (id, text). Each text is encoded on its own (empty prefix -> pooling runs
    over all content tokens), giving last-token and mean-over-text readouts at every layer.
    Used for Job A (problem geometry): item relatedness without curriculum labels.
    """
    from atom.codec.backbone import probe_hidden_states

    probe = probe_hidden_states(bundle)
    n_layers, hidden_dim = probe["n_hidden_tensors"], probe["hidden_dim"]

    unique: dict[str, str] = {}
    ids_by_hash: dict[str, list] = {}
    for iid, text in items:
        h = text_hash(text)
        unique.setdefault(h, text)
        ids_by_hash.setdefault(h, []).append(int(iid))

    cache = RawStateCache(out_dir, n_layers, hidden_dim, capacity=len(unique),
                          model=bundle["hf_id"], template="<standalone-text>")
    n_encoded = 0
    for h, text in unique.items():
        if cache.has(h):
            cache.add(h, None, None, ids=ids_by_hash[h])
            continue
        last, mean = encode_one(bundle, "", text)  # empty prefix -> pool over all content tokens
        cache.add(h, last, mean, ids=ids_by_hash[h])
        n_encoded += 1
        if logger and (n_encoded % 25 == 0):
            cache.flush()
            logger.heartbeat(step=n_encoded, total=len(unique), stage="encoding")
    cache.flush()
    return {"n_items": len(items), "n_unique": len(unique), "n_encoded": n_encoded,
            "n_cached": cache.n_cached, "n_layers": n_layers, "hidden_dim": hidden_dim,
            "out_dir": str(cache.dir)}
