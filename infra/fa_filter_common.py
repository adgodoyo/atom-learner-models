"""Shared helpers for `fc_1a_factored_heads` and `wd_1_wrong_detection_panel`.

Provides the FA belief-filter configuration (`B4_CONFIG`: codec dimension, episode and
filter hidden widths, latent dimension), the path of the 64-dimensional FA item codec,
re-exports `chunked_filter_forward` from `atom.simulator.fa_train_step`, and puts the release's
`src/` directory on `sys.path`.
"""
from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from atom.simulator.fa_train_step import (  # noqa: E402
    chunked_filter_forward,
    gather_belief_for_batch,
)
REPO = pathlib.Path(__file__).resolve().parents[1]
B4_CONFIG = {"codec_dim": 64, "episode_hidden": 256, "filter_hidden": 256, "latent_dim": 256}
CODEC_PATH = REPO / "artifacts" / "fa_item_codec_dim64.npz"
