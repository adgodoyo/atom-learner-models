"""Attempt-level fast loop for CodeWorkout: a small GRU consumes the attempt-level events
of one episode and emits an episode summary e_ep (thesis Section 5.3). The design is
per-attempt and causal because the evaluation scores every attempt of every problem with
plain BCE, separates attempt-level from episode-level base rates, and perturbs single
attempts in a causality canary; none of these can be satisfied by one episode-start
prediction broadcast across attempts.

Naming note, deliberate: FA's `episode_v2.EpisodeEncoderV2` operates at the single-EVENT
level (FA has no multi-attempt working-session concept below one interaction) -- calling
this module an "episode encoder" too would collide two different things under one name.
This module summarises the sequence of ATTEMPTS inside one CW episode (submit ->
compiler/test feedback -> revise, `cw_sessions.py`'s `episode_id`) two different ways:

  (a) a per-episode POOLED summary (may see the whole episode -- by the time the
      cross-episode filter reads a completed episode's summary, that episode is already
      in the past, so there is no leak here). This plays the role of "one event" fed
      into the SAME corpus-agnostic cross-episode filter FA uses (`filter/factory.py`).
  (b) a per-ATTEMPT causal PREFIX state. The state available for predicting attempt k's
      own outcome must be built ONLY from attempts 1..k-1: not attempt k's own code
      vector, diff, or scalar features, since `compile_ok`/`score_frac` are the label in
      near-closed-form (and even the code text alone makes compile-error
      near-deterministic -- a head that reads attempt k would be a static analyser, not a
      model of the learner; it would also be useless to the planner, which never
      generates text inside a rollout, so at prediction time attempt k's code does not
      exist yet). Concretely: the GRU's hidden state ENTERING step k (i.e. exiting step
      k-1), never the state exiting step k. Attempt 1's prefix is the fixed zero init
      state -- IDENTICAL across every episode regardless of content, the
      "attempt-prefix causality" analogue of FA's `b_t = f(H_<t)`.

Both are produced from ONE unpacked `nn.GRU` forward pass (no `pack_padded_sequence`):
a GRU's recurrence is strictly causal (h_t depends only on h_{t-1} and x_t), so garbage
at right-padded positions t >= length can never influence any REAL position's hidden
state -- padding safety and prefix causality are the same property, and gathering the
true final state at `length[i]-1` per row is exactly as pad-invariant as a packed
implementation (a regression test asserts it).

RULE (the leakage rule of thesis Section 4.5; Appendix A.5 records the leak it forbids):
no per-attempt feature may be normalised by, or otherwise derived from, an episode-level
aggregate. A feature such as `attempt_frac = row_t / (episode_length - 1)`
violates this -- the denominator is not known until the episode ends, so attempt 1's
value alone makes the FULL episode length exactly recoverable, and every prefix from
attempt 2 onward would know whether it was approaching the episode's end (base rates make
this a ~3.4x label lift, not a rounding error: attempt-level success is ~24% overall but
~82-84% on an episode's OWN final attempt, since students mostly stop when they pass).
The feature used is therefore `attempt_idx` (raw, causal, 0-based position, log1p+clamp
compressed the same way `n_compile_errors` is); the name deliberately avoids "frac",
which invites a length denominator. A flip-attempt-k canary cannot catch a length leak
(it perturbs one attempt's OUTCOME, never the episode's length), so the regression tests
also carry a prefix-truncation-invariance canary (`test_prefix_truncation_invariance...`)
that subsumes it: prefix states computed from an episode truncated after attempt k must
be bit-identical to those from the full episode, for every k.
"""

from __future__ import annotations

import torch
from torch import nn

N_SCALAR_FEATS = 4  # [compile_ok, score_frac, n_compile_errors (log1p, clipped), attempt_idx (log1p, clipped)]


class AttemptEncoder(nn.Module):
    def __init__(self, code_dim: int, diff_dim: int, proj_dim: int, hidden: int,
                 summary_dim: int, n_scalar_feats: int = N_SCALAR_FEATS):
        super().__init__()
        self.hidden = hidden
        self.code_proj = nn.Linear(code_dim, proj_dim)
        self.diff_proj = nn.Linear(diff_dim, proj_dim)
        self.scalar_proj = nn.Linear(n_scalar_feats, proj_dim)
        self.gru = nn.GRU(input_size=proj_dim * 3, hidden_size=hidden, batch_first=True)
        self.summary_head = nn.Linear(hidden, summary_dim)

    def forward(self, code_vec: torch.Tensor, diff_vec: torch.Tensor,
                scalar_feats: torch.Tensor, lengths: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """code_vec/diff_vec: [N_episodes, T_max, dim]. scalar_feats:
        [N_episodes, T_max, n_scalar_feats]. lengths: [N_episodes] (int, >= 1, true
        attempt count per episode).

        Returns `(prefix_states, summary)`:
          `prefix_states`: [N_episodes, T_max, hidden] -- `prefix_states[:, k]` is the
              CAUSAL state available for predicting attempt k (0-indexed), built only
              from attempts `0..k-1`. `prefix_states[:, 0]` is the fixed zero init state
              for every episode (the degenerate case: at T=1 this is the ONLY
              value ever read, identical across episodes, so the model's dependence on
              the prefix collapses to a constant -- the two-input FA shape).
          `summary`: [N_episodes, summary_dim] -- the pooled whole-episode summary
              (gathered at each row's own true final position, `length[i]-1`), for the
              cross-episode filter.
        """
        if bool((lengths < 1).any()):
            raise ValueError("every episode must have at least one attempt (lengths >= 1)")
        n, t_max = code_vec.shape[0], code_vec.shape[1]
        x = torch.cat([self.code_proj(code_vec), self.diff_proj(diff_vec),
                      self.scalar_proj(scalar_feats)], dim=-1)
        out, _ = self.gru(x)  # out: [N, T_max, hidden] -- state EXITING each step, unpacked

        init = out.new_zeros(n, 1, self.hidden)
        prefix_states = torch.cat([init, out[:, :-1]], dim=1) if t_max > 1 else init

        idx = (lengths - 1).view(n, 1, 1).expand(n, 1, out.shape[-1])
        final_state = out.gather(1, idx).squeeze(1)  # [N, hidden], pad-invariant (see docstring)
        summary = self.summary_head(final_state)
        return prefix_states, summary


def build_scalar_feats(compile_ok: torch.Tensor, score_frac: torch.Tensor,
                       n_compile_errors: torch.Tensor, attempt_idx: torch.Tensor) -> torch.Tensor:
    """Stacks the four per-attempt scalar channels in the fixed order
    `AttemptEncoder` expects. All inputs share the same shape (any number of leading
    dims, e.g. [N_episodes, T_max]); `n_compile_errors` and `attempt_idx` are both
    log1p-compressed and clipped -- raw counts/indices are heavy-tailed, and this keeps
    a single pathological attempt (or a very long episode) from dominating the GRU's
    input scale. `attempt_idx` MUST be the raw, causal 0-based position (never an
    episode-length-normalised fraction -- see this module's docstring for why); the clamp
    is 3.5, chosen because the measured mean is ~4.1 attempts/episode, so this
    keeps resolution where the real mass sits rather than saturating early."""
    n_err = torch.log1p(n_compile_errors.float()).clamp(max=5.0)
    idx = torch.log1p(attempt_idx.float()).clamp(max=3.5)
    return torch.stack(
        [compile_ok.float(), score_frac.float(), n_err, idx], dim=-1)
