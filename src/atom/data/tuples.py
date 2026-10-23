"""Event tuples + learner-level chronological splits for FoundationalASSIST.

Split design: disjoint train/val/test **learner** sets, with each learner's events ordered
chronologically. Test learners are genuinely held out, so prediction can be measured from
their very first interaction, as a cold-start curve requires. Leakage is prevented by
construction and verified by an audit + a deliberately-leaky canary.

Licence rule: the event tuple carries no raw text -- only hashed learner IDs, labels and
behavioural channels, plus the interaction `id` to rejoin answer text on the local machine
when encoding with the codec. The split manifest (hashed learner -> split) is safe to
commit and is frozen so every stage reuses the identical split. (`fa_manifest` builds the
canonical manifest and lockbox on top of these split identities.)
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "learner", "id", "t", "problem_id", "timestamp", "correct", "hint_count", "saw_answer",
]
SPLITS = ("train", "val", "test")


def hash_learner(uid) -> str:
    """Stable 64-bit hex hash of a learner id (de-identifies before any export)."""
    return hashlib.sha1(str(uid).encode("utf-8")).hexdigest()[:16]


def build_events(interactions: pd.DataFrame) -> pd.DataFrame:
    """Canonical per-interaction event stream, chronologically ordered within learner.

    Requires columns: user_id, id, problem_id, end_time, hint_count, saw_answer, and either
    `correct` or `discrete_score`. Rows with an unparseable/missing timestamp are dropped
    (they cannot be placed in time). Events are sorted by (timestamp, id) so ties are broken
    deterministically, then `t` is the 0-based position in the learner's history.
    """
    df = interactions
    end_time = df["end_time"]
    if not pd.api.types.is_datetime64_any_dtype(end_time):
        end_time = pd.to_datetime(end_time, errors="coerce", utc=True)

    if "correct" in df.columns:
        correct = df["correct"]
    else:
        correct = (df["discrete_score"] == 1).astype("int8")

    ev = pd.DataFrame({
        "learner": df["user_id"].map(hash_learner),
        "id": df["id"].astype("int64"),
        "problem_id": df["problem_id"].astype("int64"),
        "timestamp": pd.to_datetime(end_time, utc=True),
        "correct": np.asarray(correct).astype("int8"),
        "hint_count": df["hint_count"].astype("int32"),
        "saw_answer": df["saw_answer"].astype(bool).astype("int8"),
    })
    ev = ev.dropna(subset=["timestamp"])
    # mergesort = stable, so the id tiebreak within equal timestamps is respected.
    ev = ev.sort_values(["learner", "timestamp", "id"], kind="mergesort").reset_index(drop=True)
    ev["t"] = ev.groupby("learner", sort=False).cumcount().astype("int32")
    return ev[EVENT_COLUMNS]


def make_learner_split(learners, fracs=(0.8, 0.1, 0.1), seed: int = 0) -> pd.DataFrame:
    """Deterministic disjoint train/val/test partition over unique (hashed) learners.

    Returns DataFrame[learner, split]. Sorting the unique learners before the seeded
    permutation makes the split reproducible regardless of input row order.
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fracs must sum to 1.0, got {fracs} (sum {sum(fracs)})")
    uniq = np.array(sorted(set(learners)))
    n = len(uniq)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(n * fracs[0]))
    n_va = int(round(n * fracs[1]))
    split = np.empty(n, dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_va]] = "val"
    split[perm[n_tr + n_va:]] = "test"
    return pd.DataFrame({"learner": uniq, "split": split})


def select_lm_subsample(
    events: pd.DataFrame, frac: float, seed: int, n_learners: int | None = None,
) -> pd.DataFrame:
    """Deterministic (seeded) event subsample of the text staged for LM-loss training.

    ONE primitive for both the pilot (frac=0.05, n_learners=200) and the full run
    (frac=0.10-0.20, n_learners=None), so the two staged samples are drawn the same way,
    not by convergent hand-written logic. `events` is whatever population the caller passes in (callers are
    expected to pre-filter to `split == "train"` -- LM-loss training data must never touch
    val/test). The learner draw and the event draw
    use distinct RNG streams (seed, seed+1) so they don't correlate by construction.
    """
    pool = events
    if n_learners is not None:
        learners = np.array(sorted(pool["learner"].unique()))
        rng = np.random.default_rng(seed)
        chosen_learners = set(learners[rng.permutation(len(learners))[:n_learners]])
        pool = pool[pool["learner"].isin(chosen_learners)]

    ids = np.array(sorted(pool["id"].unique()))
    rng = np.random.default_rng(seed + 1)
    n_take = int(round(len(ids) * frac))
    chosen_ids = set(ids[rng.permutation(len(ids))[:n_take]])
    return pool[pool["id"].isin(chosen_ids)].copy()


def apply_split(events: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach a `split` column to events by hashed learner."""
    return events.merge(manifest, on="learner", how="left")


def write_split_manifest(manifest: pd.DataFrame, path) -> None:
    manifest.to_csv(path, index=False)


def load_split_manifest(path) -> pd.DataFrame:
    return pd.read_csv(path)


# -- leakage audits (the split gate) -----------------------------------------

def audit_disjoint(manifest: pd.DataFrame) -> dict:
    """No learner may appear in more than one split."""
    per_learner_splits = manifest.groupby("learner")["split"].nunique()
    dup_rows = int(manifest["learner"].duplicated().sum())
    in_multiple = int((per_learner_splits > 1).sum())
    return {
        "disjoint": bool(dup_rows == 0 and in_multiple == 0),
        "duplicate_rows": dup_rows,
        "learners_in_multiple_splits": in_multiple,
    }


def audit_monotonic_time(events: pd.DataFrame) -> dict:
    """Within each learner, timestamps must be non-decreasing along `t`."""
    ev = events.sort_values(["learner", "t"])
    dt = ev["timestamp"].diff()
    same_learner = ev["learner"].eq(ev["learner"].shift())
    backwards = int(((dt < pd.Timedelta(0)) & same_learner).sum())
    return {"monotonic": backwards == 0, "backwards_steps": backwards}


def audit_split_coverage(events: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Every learner present in the events must map to exactly one split."""
    covered = set(manifest["learner"])
    ev_learners = set(events["learner"].unique())
    missing = ev_learners - covered
    return {"all_covered": len(missing) == 0, "missing_learners": len(missing)}


def inject_leak(manifest: pd.DataFrame) -> pd.DataFrame:
    """Canary: copy one test learner into train, which must break disjointness."""
    test_learner = manifest.loc[manifest["split"] == "test", "learner"].iloc[0]
    leak = pd.DataFrame({"learner": [test_learner], "split": ["train"]})
    return pd.concat([manifest, leak], ignore_index=True)
