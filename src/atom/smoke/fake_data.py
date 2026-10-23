"""Synthetic learners for the smoke test.

A deliberately trivial knowledge-tracing world: each learner has a scalar ability, each
item a scalar difficulty plus a random "codec" vector standing in for the real semantic
encoder. Correctness is Bernoulli(sigmoid(ability - difficulty)). The signal is learnable
(a filter that accumulates a learner's history should beat chance), which is all the smoke
needs: it exercises data -> filter -> head -> eval -> report without any real model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FakeData:
    codec: torch.Tensor       # [N, T, D] per-step item feature vectors ("codec" stand-in)
    y: torch.Tensor           # [N, T] correctness in {0, 1}
    item_idx: torch.Tensor    # [N, T] which item was served
    ability: torch.Tensor     # [N] latent learner ability (ground truth, for reference)
    difficulty: torch.Tensor  # [M] latent item difficulty (ground truth)
    item_codec: torch.Tensor  # [M, D] the fixed per-item vectors

    @property
    def n_learners(self) -> int:
        return self.y.shape[0]

    @property
    def seq_len(self) -> int:
        return self.y.shape[1]

    @property
    def codec_dim(self) -> int:
        return self.codec.shape[2]


def make_fake_data(
    n_learners: int = 20,
    n_items: int = 50,
    seq_len: int = 60,
    codec_dim: int = 16,
    ability_sd: float = 1.0,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> FakeData:
    g = torch.Generator().manual_seed(seed)

    ability = torch.randn(n_learners, generator=g) * ability_sd
    difficulty = torch.randn(n_items, generator=g) * difficulty_sd
    # Fixed per-item vectors, scaled so their norm does not blow up with dimension.
    item_codec = torch.randn(n_items, codec_dim, generator=g) / (codec_dim ** 0.5)

    item_idx = torch.randint(0, n_items, (n_learners, seq_len), generator=g)
    logits = ability[:, None] - difficulty[item_idx]          # [N, T]
    p = torch.sigmoid(logits)
    y = torch.bernoulli(p, generator=g)                        # [N, T]
    codec = item_codec[item_idx]                               # [N, T, D]

    return FakeData(
        codec=codec,
        y=y,
        item_idx=item_idx,
        ability=ability,
        difficulty=difficulty,
        item_codec=item_codec,
    )


def temporal_split(seq_len: int, train_frac: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-learner temporal masks: predict later steps from earlier ones (no leakage)."""
    split = max(1, int(round(seq_len * train_frac)))
    steps = torch.arange(seq_len)
    train_mask = steps < split
    val_mask = steps >= split
    return train_mask, val_mask
