"""The codec transform: strip dominant components + whiten.

Raw pooled states are cached; THIS
transform is stored separately and applied on load, so k and whitening retune post-hoc
without any model forward. Fit stores, per (layer, pooling): the mean and the top-K_max
eigenvectors/eigenvalues of the centred covariance. Apply:

    center (x - mu)  ->  drop the top-k eigen-directions (deflation)  ->  optionally
    whiten the retained directions by 1/sqrt(eigenvalue).

Euclidean distance in the whitened output equals the truncated-Mahalanobis distance the
Meaning Contracts analysis uses; with whiten=False and cosine it reproduces that pipeline's
deflated-cosine metric. Follows the Meaning Contracts code's `analysis/utils.py`
(`top_k_deflation`, `svd_truncated`, `mahalanobis_truncated`); float64 stats, SEED=42.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 42
CODEC_VERSION = "codec-v1"


@dataclass
class CodecTransform:
    """A fitted, versioned codec transform for one (layer, pooling)."""

    mean: np.ndarray       # (D,)
    eigvecs: np.ndarray    # (D, K_max) columns = eigenvectors, descending eigenvalue
    eigvals: np.ndarray    # (K_max,) covariance eigenvalues, descending
    layer: int | None = None
    pooling: str | None = None
    version: str = CODEC_VERSION
    seed: int = SEED
    fit_n: int = 0
    source: str = ""

    @property
    def k_max(self) -> int:
        return self.eigvecs.shape[1]

    @property
    def dim(self) -> int:
        return self.eigvecs.shape[0]

    @property
    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.mean.astype(np.float64).tobytes())
        h.update(self.eigvecs.astype(np.float64).tobytes())
        h.update(self.eigvals.astype(np.float64).tobytes())
        return h.hexdigest()[:12]

    # -- fitting --------------------------------------------------------------
    @classmethod
    def fit(cls, X: np.ndarray, k_max: int = 50, layer: int | None = None,
            pooling: str | None = None, seed: int = SEED, source: str = "",
            svd_solver: str = "randomized") -> "CodecTransform":
        """Fit from raw pooled states X (N, D). k_max eigen-directions are retained.

        `svd_solver`: `"randomized"` (default) uses
        `sklearn.utils.extmath.randomized_svd`, an approximate method whose exact numeric
        output for a given component depends on `k_max` itself (the random projection
        matrix's shape changes with `k_max`, so it consumes the seeded RNG stream
        differently) -- **measured NOT nested**: refitting the real FA problem corpus at
        k_max=300 vs. 50 gave eigenvector cosine similarity as low as 0.994 for the last
        few components of a 50-truncation, concentrated exactly where the spectrum has no
        elbow (eigenvalue ratios ~1.01-1.04 between consecutive components 44-49 -- a
        near-degenerate block, where individual eigenvector DIRECTIONS are inherently
        ill-conditioned even though the eigenVALUES and the subspace they span are
        stable). `"full"` uses `np.linalg.svd` (deterministic, economy SVD) and is
        **exactly self-nested by construction**: verified empirically (cosine similarity
        1.0 to full float64 precision, any two k_max truncations of the same fit) -- any
        wider future refit's first k_max columns are IDENTICAL to a narrower one's,
        because both are simply prefixes of the one canonical decomposition, never two
        independent randomised approximations."""
        X64 = np.asarray(X, dtype=np.float64)
        if X64.ndim != 2:
            raise ValueError(f"expected (N, D) states, got shape {X64.shape}")
        n = X64.shape[0]
        k_max = min(k_max, X64.shape[1], max(1, n - 1))
        mean = X64.mean(axis=0)
        Xc = X64 - mean
        if svd_solver == "randomized":
            from sklearn.utils.extmath import randomized_svd
            np.random.seed(seed)
            _, S, Vt = randomized_svd(Xc, n_components=k_max, random_state=seed)
        elif svd_solver == "full":
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            S, Vt = S[:k_max], Vt[:k_max]
        else:
            raise ValueError(f"unknown svd_solver {svd_solver!r}; expected 'randomized' or 'full'")
        eigvecs = Vt.T.copy()                      # (D, k_max)
        eigvals = (S ** 2) / max(n - 1, 1)         # covariance eigenvalues
        return cls(mean=mean, eigvecs=eigvecs, eigvals=eigvals, layer=layer,
                   pooling=pooling, seed=seed, fit_n=n, source=source)

    # -- applying -------------------------------------------------------------
    def apply(self, X: np.ndarray, k: int = 5, rank: int | None = None,
              whiten: bool = True) -> np.ndarray:
        """Codec-encode raw states X (N, D) -> (N, rank-k).

        Drops the top-k eigen-directions (deflation); keeps directions [k, rank);
        whitens the kept directions by 1/sqrt(eigenvalue) if whiten.
        """
        rank = self.k_max if rank is None else min(rank, self.k_max)
        if k < 0 or k >= rank:
            raise ValueError(f"need 0 <= k < rank <= k_max; got k={k}, rank={rank}, k_max={self.k_max}")
        Xc = np.asarray(X, dtype=np.float64) - self.mean
        coords = Xc @ self.eigvecs[:, k:rank]                    # (N, rank-k)
        if whiten:
            coords = coords / np.sqrt(np.maximum(self.eigvals[k:rank], 1e-12))
        return coords

    # -- persistence (versioned) ---------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p, version=self.version, seed=self.seed, fit_n=self.fit_n,
            layer=(-1 if self.layer is None else self.layer),
            pooling=(self.pooling or ""), source=self.source,
            mean=self.mean, eigvecs=self.eigvecs, eigvals=self.eigvals,
        )
        # np.savez appends .npz; return the actual path.
        return p if p.suffix == ".npz" else p.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "CodecTransform":
        z = np.load(Path(path), allow_pickle=False)
        layer = int(z["layer"])
        return cls(
            mean=z["mean"], eigvecs=z["eigvecs"], eigvals=z["eigvals"],
            layer=(None if layer < 0 else layer),
            pooling=(str(z["pooling"]) or None), version=str(z["version"]),
            seed=int(z["seed"]), fit_n=int(z["fit_n"]), source=str(z["source"]),
        )
