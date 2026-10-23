"""Canonical FA event manifest, splits and lockbox seal.

This is a separate build from `fa.py`/`tuples.py`, not a patch on them. Those modules feed
the current item's codec into the belief and map missing scores to 0; neither behaviour may
propagate into the canonical manifest. This module implements the mandatory preprocessing
order:

  1. load raw tables (caller's job -- `atom.data.fa.load_interactions`/`load_problems`)
  2. clean question text (`atom.data.fa_clean`)
  3. remove exact duplicate event IDs before sequences/lag features are built
  4. preserve distinct IDs for repeated (learner, problem) exposures
  5. resolve platform score to {0, 1, missing} -- never map missing to 0
  6. derive the four-way support label
  7. chronological order within learner from valid timestamps; missing-time rows documented,
     not chronologically ordered
  8. gap features computed only after dedup + ordering
  9. label-validity / missingness indicators
  10. freeze + hash the resulting manifest (done by the caller script, see `lockbox_hash`)

The belief boundary -- b_t = f(H_<t), the current question enters only the emission path,
never the recurrent belief -- is the belief filter's concern, not this module's; this
module's job is to hand the filter a clean, leakage-audited event stream and frozen splits
so that boundary can be implemented correctly downstream.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from atom.data.tuples import hash_learner

SUPPORT_CLASSES = ("NONE", "HINT_ONLY", "REVEAL_ONLY", "HINT_AND_REVEAL")

MANIFEST_COLUMNS = [
    "learner", "id", "problem_id", "problem_group_id",
    "timestamp", "has_valid_time", "t",
    "score", "score_missing",
    "support", "support_code", "hint_count", "saw_answer",
    "gap_log1p_min", "first_event", "missing_gap",
    "first_exposure",
]


# -- step 3: exact-ID dedup --------------------------------------------------

def dedup_interactions(df: pd.DataFrame, id_col: str = "id") -> tuple[pd.DataFrame, dict]:
    """Drop later copies of exact-duplicate event IDs, keeping the first occurrence.

    "First" is the original file row order (source order), which is what the raw
    `Interactions.csv` gives us -- duplicate-ID groups in this corpus are byte-identical
    across every other column (verified: 15,677/15,677 groups), so which copy survives
    does not change any label; only the count of rows entering sequence assembly.
    """
    n_before = len(df)
    dup_mask = df[id_col].duplicated(keep=False)
    n_in_dup_groups = int(dup_mask.sum())
    n_dup_groups = int(df.loc[dup_mask, id_col].nunique())
    out = df.drop_duplicates(subset=[id_col], keep="first").copy()
    stats = {
        "n_before": n_before,
        "n_after": len(out),
        "n_dropped": n_before - len(out),
        "n_rows_in_dup_groups": n_in_dup_groups,
        "n_dup_groups": n_dup_groups,
    }
    return out, stats


def assert_unique_ids(df: pd.DataFrame, id_col: str = "id") -> None:
    """Mandatory post-dedup assertion (the target of the planted-duplicate canary test)."""
    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        raise AssertionError(f"{int(dup.sum())} rows still share a duplicated {id_col!r} after dedup")


def inject_duplicate_row(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Planted-duplicate canary: re-duplicate one already-deduplicated row's id.

    A dedup pipeline that is silently broken (e.g. dedup fitted before an upstream
    concat, or a stale cached table) must still reject this. Used only in tests / the
    gate script's canary check -- never on the manifest that ships.
    """
    rng = np.random.default_rng(seed)
    row = df.iloc[[int(rng.integers(0, len(df)))]].copy()
    return pd.concat([df, row], ignore_index=True)


# -- step 5: score resolution -------------------------------------------------

def resolve_score(discrete_score: pd.Series) -> pd.Series:
    """Map platform score to nullable {0, 1, <NA>} -- missing stays missing, never 0."""
    out = pd.Series(pd.NA, index=discrete_score.index, dtype="Int8")
    out[discrete_score == 1] = 1
    out[discrete_score == 0] = 0
    return out


# -- step 6: four-class support label -----------------------------------------

def support_label(hint_count: pd.Series, saw_answer: pd.Series) -> pd.DataFrame:
    """Categorical support mode from (hint_count > 0, saw_answer): NONE/HINT_ONLY/REVEAL_ONLY/HINT_AND_REVEAL."""
    hint = (hint_count.fillna(0) > 0)
    reveal = saw_answer.astype(bool)
    code = hint.astype("int8") + 2 * reveal.astype("int8")  # 0/1/2/3 = NONE/HINT/REVEAL/BOTH
    label = pd.Categorical.from_codes(code.to_numpy(), categories=list(SUPPORT_CLASSES))
    return pd.DataFrame({"support": label, "support_code": code.astype("int8")})


# -- near-duplicate problem grouping ------------------------------------------

def build_problem_catalog(problems_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per unique `problem_id` with a `problem_group_id` for near-duplicate items.

    Grouping signal: exact match on normalised (stripped, lower-cased) cleaned body text,
    the same signal already used by `scripts/p2_07_encode_problems.py` to dedupe the codec
    cache. This catches identical-body variants across problem-set duplicates; it does not
    catch paraphrases -- a disclosed limitation, not a silent one.

    Requires `problems_clean` from `load_problems(clean_body=True)`.
    """
    cat = problems_clean.drop_duplicates("problem_id", keep="first").copy()
    norm = cat["problem_body_clean"].fillna("").str.strip().str.lower()
    # sha256 (not pandas' hash_pandas_object) so the group id is stable across processes/runs.
    cat["problem_group_id"] = norm.map(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    )
    keep = ["problem_id", "problem_group_id"]
    if "problem_type" in cat.columns:
        keep.append("problem_type")
    return cat[keep]


# -- steps 4, 7, 8, 9: canonical event manifest -------------------------------

def build_canonical_events(
    interactions: pd.DataFrame, problem_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the canonical, leakage-audited event manifest from deduplicated interactions.

    `interactions` must already be exact-ID deduplicated (call `dedup_interactions` first).
    `problem_catalog` is `build_problem_catalog`'s output (problem_id -> problem_group_id).

    History position `t` (and the gap/first-event features derived from it) is assigned only
    over each learner's VALID-timestamp rows, sorted (timestamp, id). Missing-time rows keep
    every other field (score, support, hint_count, saw_answer -- all usable by current-event
    heads) but get `t = NA`, `has_valid_time = False`, and are excluded from the recurrent
    history; their relative order cannot be established from the schema, so no chronological
    claim is made about them (step 7).
    """
    df = interactions.merge(problem_catalog, on="problem_id", how="left")
    if df["problem_group_id"].isna().any():
        missing = int(df["problem_group_id"].isna().sum())
        raise AssertionError(f"{missing} events reference a problem_id absent from the catalog")

    ts = df["end_time"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)
    has_valid_time = ts.notna()

    score = resolve_score(df["discrete_score"])
    sup = support_label(df["hint_count"], df["saw_answer"])

    ev = pd.DataFrame({
        "learner": df["user_id"].map(hash_learner),
        "id": df["id"].astype("int64"),
        "problem_id": df["problem_id"].astype("int64"),
        "problem_group_id": df["problem_group_id"],
        "timestamp": ts,
        "has_valid_time": has_valid_time,
        "score": score,
        "score_missing": score.isna(),
        "support": sup["support"],
        "support_code": sup["support_code"],
        "hint_count": df["hint_count"].astype("int32"),
        "saw_answer": df["saw_answer"].astype(bool).astype("int8"),
    })

    # Chronological position + gap features, valid-time rows only, per learner.
    ev["t"] = pd.array([pd.NA] * len(ev), dtype="Int32")
    ev["gap_log1p_min"] = np.nan
    ev["first_event"] = False
    ev["missing_gap"] = True
    ev["first_exposure"] = pd.array([pd.NA] * len(ev), dtype="boolean")

    valid = ev[has_valid_time].copy()
    valid = valid.sort_values(["learner", "timestamp", "id"], kind="mergesort")
    valid["t"] = valid.groupby("learner", sort=False).cumcount().astype("int32")

    prev_ts = valid.groupby("learner", sort=False)["timestamp"].shift(1)
    gap_min = (valid["timestamp"] - prev_ts).dt.total_seconds() / 60.0
    valid["first_event"] = prev_ts.isna()
    valid["gap_log1p_min"] = np.log1p(gap_min.clip(lower=0))
    valid["missing_gap"] = valid["first_event"]

    prev_problem_seen = (
        valid.groupby(["learner", "problem_id"], sort=False).cumcount() == 0
    )
    valid["first_exposure"] = prev_problem_seen.astype("boolean")

    for col in ["t", "gap_log1p_min", "first_event", "missing_gap", "first_exposure"]:
        ev.loc[valid.index, col] = valid[col]

    stats = {
        "n_events": len(ev),
        "n_valid_time": int(has_valid_time.sum()),
        "n_missing_time": int((~has_valid_time).sum()),
        "n_score_missing": int(score.isna().sum()),
        "support_counts": ev["support"].value_counts().to_dict(),
    }
    return ev[MANIFEST_COLUMNS], stats


# -- audits --------------------------------------------------------------------

def audit_no_duplicate_ids(events: pd.DataFrame) -> dict:
    dup = events["id"].duplicated(keep=False)
    return {"unique_ids": bool(not dup.any()), "n_duplicated": int(dup.sum())}


def audit_score_never_imputed(events: pd.DataFrame) -> dict:
    """`score_missing` rows must carry score == <NA>, never 0 or 1."""
    bad = events.loc[events["score_missing"], "score"].notna().sum()
    return {"score_never_imputed": bool(bad == 0), "n_bad": int(bad)}


def audit_monotonic_time(events: pd.DataFrame) -> dict:
    valid = events[events["has_valid_time"]].sort_values(["learner", "t"])
    dt = valid["timestamp"].diff()
    same_learner = valid["learner"].eq(valid["learner"].shift())
    backwards = int(((dt < pd.Timedelta(0)) & same_learner).sum())
    return {"monotonic": backwards == 0, "backwards_steps": backwards}


def audit_missing_time_excluded(events: pd.DataFrame) -> dict:
    missing = events[~events["has_valid_time"]]
    bad = int((missing["t"].notna()).sum())
    return {"missing_time_excluded": bool(bad == 0), "n_bad": bad}


# -- two-axis split + lockbox ---------------------------------------------------

def make_learner_split_v2(
    old_manifest: pd.DataFrame, lockbox_frac: float, seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Learner axis: {train, val, development_exposed, lockbox}.

    Reuses the original split identities (`tuples.py`) for train/val/test -- val and test
    were already used to adjudicate gates during development and cannot be un-exposed by
    relabelling. The original test learners therefore become `development_exposed` (its
    history disclosed wherever cited, never called an untouched final test). A genuinely
    new lockbox has no untouched pool to draw from except the original TRAIN learners
    (their rows fed gradients for earlier models, but nothing built on this manifest --
    new manifest, new splits, new filter -- has touched them, so they are the only
    learners that can be sealed). Carved deterministically, seeded, disjoint from
    the retained train remainder.
    """
    old_train = old_manifest.loc[old_manifest["split"] == "train", "learner"].to_numpy()
    old_val = old_manifest.loc[old_manifest["split"] == "val", "learner"]
    old_test = old_manifest.loc[old_manifest["split"] == "test", "learner"]

    rng = np.random.default_rng(seed)
    uniq_train = np.array(sorted(set(old_train)))
    perm = rng.permutation(len(uniq_train))
    n_lockbox = int(round(len(uniq_train) * lockbox_frac))
    lockbox_learners = uniq_train[perm[:n_lockbox]]
    new_train_learners = uniq_train[perm[n_lockbox:]]

    manifest = pd.concat([
        pd.DataFrame({"learner": new_train_learners, "split": "train"}),
        pd.DataFrame({"learner": sorted(set(old_val)), "split": "val"}),
        pd.DataFrame({"learner": sorted(set(old_test)), "split": "development_exposed"}),
        pd.DataFrame({"learner": lockbox_learners, "split": "lockbox"}),
    ], ignore_index=True)

    stats = {
        "n_train": len(new_train_learners), "n_val": len(set(old_val)),
        "n_development_exposed": len(set(old_test)), "n_lockbox": len(lockbox_learners),
    }
    return manifest, stats


def make_item_split_v2(
    problem_catalog: pd.DataFrame, heldout_frac: float, seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Item axis over `problem_group_id`: {train, heldout}. A group never splits across the two."""
    groups = np.array(sorted(problem_catalog["problem_group_id"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_heldout = int(round(len(groups) * heldout_frac))
    heldout = set(groups[perm[:n_heldout]])
    manifest = pd.DataFrame({
        "problem_group_id": groups,
        "split": np.where(np.isin(groups, list(heldout)), "heldout", "train"),
    })
    stats = {
        "n_groups": len(groups), "n_train_groups": len(groups) - n_heldout,
        "n_heldout_groups": n_heldout,
    }
    return manifest, stats


def audit_split_disjoint(manifest: pd.DataFrame, key: str) -> dict:
    per_key_splits = manifest.groupby(key)["split"].nunique()
    dup_rows = int(manifest[key].duplicated().sum())
    in_multiple = int((per_key_splits > 1).sum())
    return {
        "disjoint": bool(dup_rows == 0 and in_multiple == 0),
        "duplicate_rows": dup_rows,
        "in_multiple_splits": in_multiple,
    }


def inject_split_leak(manifest: pd.DataFrame, from_split: str, to_split: str) -> pd.DataFrame:
    """Canary: copy one `from_split` key into `to_split`, which must break disjointness."""
    key_col = manifest.columns[0]
    row = manifest.loc[manifest["split"] == from_split].iloc[[0]].copy()
    row["split"] = to_split
    return pd.concat([manifest, row], ignore_index=True)


def attach_splits(
    events: pd.DataFrame, learner_split: pd.DataFrame, item_split: pd.DataFrame,
) -> pd.DataFrame:
    out = events.merge(learner_split, on="learner", how="left", suffixes=("", "_learner"))
    out = out.rename(columns={"split": "split_learner"})
    out = out.merge(item_split, on="problem_group_id", how="left")
    out = out.rename(columns={"split": "split_item"})
    return out


def audit_lockbox_isolated(events_with_splits: pd.DataFrame) -> dict:
    """The strict lockbox cell (lockbox learner x heldout item) must not overlap dev/train use.

    Not a leakage risk by construction (learner and item axes are independent partitions),
    but a required sanity check: every lockbox-learner row must carry `split_learner ==
    'lockbox'` and no lockbox learner may appear under any other learner-split label.
    """
    learners = events_with_splits.groupby("learner")["split_learner"].nunique()
    return {"learner_split_consistent": bool((learners <= 1).all())}


# -- lockbox seal ----------------------------------------------------------------

def load_v2_events(manifest_path, learner_split_path, item_split_path) -> pd.DataFrame:
    """Load the clean canonical manifest and restrict to `has_valid_time` rows (the only
    ones with a defined `t` -- see `build_canonical_events`). Adds a `split` column aliased
    to `split_learner` (baseline scripts split on the learner axis only; the item-axis
    holdout is evaluated separately) -- NEVER includes `lockbox`
    rows' *content* in anything printed; callers must filter to `train`/`val` before scoring
    (development_exposed/lockbox are informational-only / sealed respectively, by the
    split design).

    `manifest_path` is the output of `fa_00_manifest_rebuild.py`, which already
    carries `split_learner`/`split_item` (it calls `attach_splits` itself before saving) --
    this function reuses those columns rather than re-attaching, so a stale or hand-edited
    `learner_split_path`/`item_split_path` can never silently diverge from what the manifest
    was actually sealed against. If a caller passes a manifest WITHOUT those columns (e.g. a
    raw `build_canonical_events` output), the split files are attached fresh instead.
    """
    ev = pd.read_parquet(manifest_path)
    ev = ev[ev["has_valid_time"]].copy()
    if "split_learner" not in ev.columns or "split_item" not in ev.columns:
        learner_split = pd.read_csv(learner_split_path)
        item_split = pd.read_csv(item_split_path)
        ev = attach_splits(ev, learner_split, item_split)
    ev["split"] = ev["split_learner"]
    return ev


def exclude_score_missing_from_history(ev: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-harness switch: an AKT-MATCHED view of the event stream, for a fair simulator-vs-AKT comparison.

    AKT v2 (`scripts/fa_01_akt_v2.py`) excludes score-missing events from its stream
    ENTIRELY (`ev[~ev["score_missing"]]`) rather than masking them, because pyKT's AKT
    couples a step's response into the joint input embedding -- there is no well-defined
    maskable placeholder there the way there is for a GRU/transformer filter. The recurrent
    belief filter and the simulator instead keep those events as recurrent INPUT and mask them out of the
    LOSS only (`filter/batch.py::build_padded_batch_v2`'s `label_valid`). The two streams
    therefore differ in the *histories* each model conditions on, not merely in which
    positions are scored -- every event downstream of an excluded one sits in a shorter,
    gap-free context for AKT that it never had for the filter/the simulator. That difference
    cannot be repaired by subsetting a metric after the fact, because the conditioning
    already differed at prediction time; the only honest fix is to score the simulator on
    a stream built the SAME way AKT's was.

    This function builds that view: drop `score_missing` rows entirely (not just flag
    them), then re-derive `t`/`gap_log1p_min`/`first_event`/`missing_gap` on the reduced
    per-learner sequence -- otherwise the dropped rows would leave silent holes in `t`
    that `filter/batch.py`'s padded-batch builders would place as empty padding rather
    than closing the gap, which is NOT what AKT's own exclusion does (its chunker just
    never sees the excluded rows, so the remaining ones sit contiguously). Missing-time
    rows are dropped from this view too (they already carry no defined `t`, same as the
    original manifest). Output feeds straight into `build_padded_batch_v2` unchanged --
    this is a preprocessing view of `ev`, not a change to the batch builder itself.

    Does NOT change the simulator's model code or the manifest's masking decision, which
    stands (this is an evaluation-harness capability only).
    """
    kept = ev[ev["has_valid_time"] & ~ev["score_missing"]].copy()
    kept = kept.sort_values(["learner", "timestamp", "id"], kind="mergesort")
    kept["t"] = kept.groupby("learner", sort=False).cumcount().astype("int32")

    prev_ts = kept.groupby("learner", sort=False)["timestamp"].shift(1)
    gap_min = (kept["timestamp"] - prev_ts).dt.total_seconds() / 60.0
    kept["first_event"] = prev_ts.isna()
    kept["gap_log1p_min"] = np.log1p(gap_min.clip(lower=0))
    kept["missing_gap"] = kept["first_event"]
    return kept


def lockbox_hash(events_with_splits: pd.DataFrame) -> dict:
    """Seal the strict lockbox cell (lockbox learner AND heldout item) without reading its rows.

    Returns only a content hash + counts -- never the rows (lockbox rows are never printed;
    only the hash is committed). The hash covers the sorted (learner, id) pairs, so any
    future accidental mutation of that cell is detectable without re-reading its labels.
    """
    cell = events_with_splits[
        (events_with_splits["split_learner"] == "lockbox")
        & (events_with_splits["split_item"] == "heldout")
    ]
    pairs = sorted(zip(cell["learner"].tolist(), cell["id"].tolist()))
    blob = "\n".join(f"{l},{i}" for l, i in pairs).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "n_rows": len(cell),
        "n_learners": int(cell["learner"].nunique()),
        "n_item_groups": int(cell["problem_group_id"].nunique()),
    }
