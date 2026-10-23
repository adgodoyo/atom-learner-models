"""Shared helpers for `ga_1_tier4_rungc_panel` (writing the question, Figure 13): lexical
memorisation and internal-spread statistics, seeded sampled decoding, and the panel `main()`
that generates questions with and without the learner's history and with a swapped history,
projects them through the frozen codec, and scores pointing, manifold support and
history-sensitivity.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
from typing import Any, Sequence
import numpy as np
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cw_planner_suite import _candidate_scores  # noqa: E402
from ga_1_common import (  # noqa: E402
    EXPECTED_HF_ID,
    LocalGenerationStore,
    NumericProgress,
    bootstrap_mean,
    build_tier1_prompt,
    embed_questions,
    generated_manifold_metrics,
    load_local_gemma,
    load_problem_statements,
    load_raw_episode_text,
    normalise_text,
    release_backbone,
    stable_hash,
    stable_uint64,
    stop_instance_if_requested,
    validate_no_external_api_surface,
)
from ga_1_tier4_common import SUBSTRATE, load_field, state_eps, state_rows  # noqa: E402
from ga_1_tier4_generation_common import LORA_M, SFT_SEEDS, attach_plain_lora
from rl_2 import _bootstrap_seed_learner, _frozen_filter_state  # noqa: E402
from atom.binding.derangement import assert_derangement, derangement_permutation  # noqa: E402
from atom.eval.cw_generation import dist_n  # noqa: E402
TEMPERATURE = 0.8
TOP_P = 1.0
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 8
HISTORY_TOKENS = 3000
ARMS = ("zero_shot", "tuned")
MODES = ("history", "no_history", "swapped_history")
PANEL_STARTS = (3, 5, 10)


def token_set(text: str) -> set[str]:
    return set(normalise_text(text).split())


def max_pool_similarity(text: str, pool: Sequence[str]) -> tuple[float, int]:
    """Max token-Jaccard against the 50 real statements, with the argmax item.

    Deliberately a blunt lexical measure: memorisation is being DISCLOSED, so the statistic
    must be one a reader can check by eye against the projection table, not a learned score.
    """
    t = token_set(text)
    best, arg = 0.0, -1
    for i, ref in enumerate(pool):
        r = token_set(ref)
        union = len(t | r)
        j = (len(t & r) / union) if union else 0.0
        if j > best:
            best, arg = j, i
    return best, arg


def mean_pairwise_distance(vectors: np.ndarray, cap: int = 400) -> float:
    """A set's own internal codec spread -- the reference line history-sensitivity is read
    against (as operationalised in the pre-registration)."""
    x = np.asarray(vectors, dtype=np.float64)
    if len(x) > cap:
        x = x[np.linspace(0, len(x) - 1, cap, dtype=int)]
    if len(x) < 2:
        return float("nan")
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    iu = np.triu_indices(len(x), k=1)
    return float(d[iu].mean())


def generate_sampled(bundle: dict, context: Any, prompts: Sequence[str], adapter_on: bool,
                     batch_seed_tag: str, batch_size: int, max_prompt_tokens: int,
                     progress: NumericProgress, stage: str) -> list[str]:
    """Sampled decoding, temperature 0.8, deterministic per batch.

    `generate` draws from one RNG stream per call, so the seed is set per batch from the
    batch's own content-addressed tag; batch composition is deterministic, so the whole
    schedule reproduces.  Greedy is never used here -- see the pre-registration.
    """
    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    old_pad, old_trunc = tok.padding_side, tok.truncation_side
    tok.padding_side, tok.truncation_side = "left", "left"
    dtype = next(model.parameters()).dtype
    out: list[str] = []
    try:
        model.eval()
        for offset in range(0, len(prompts), batch_size):
            batch = list(prompts[offset:offset + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_prompt_tokens).to(device)
            width = int(enc["input_ids"].shape[1])
            context.set(torch.full((len(batch), LORA_M), 1.0 if adapter_on else 0.0,
                                   device=device, dtype=dtype))
            torch.manual_seed(stable_uint64(f"{batch_seed_tag}|{offset}") % (2 ** 31))
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                     min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS,
                                     use_cache=True, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
            for row in gen:
                out.append(tok.decode(row[width:], skip_special_tokens=True).strip())
            progress.maybe_heartbeat(stage, min(offset + len(batch), len(prompts)), len(prompts),
                                     enforce_overrun=True)
    finally:
        tok.padding_side, tok.truncation_side = old_pad, old_trunc
    if any(not x for x in out):
        raise RuntimeError("degenerate-output check: an empty generation after decoding")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--substrate", default=SUBSTRATE)
    ap.add_argument("--sft-dir", default="results/ga_1_tier4_sft")
    ap.add_argument("--out-dir", default="results/ga_1_tier4_panel")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--val-learners", type=int, default=110)
    ap.add_argument("--n-seeds", type=int, default=len(SFT_SEEDS))
    ap.add_argument("--pilot-learners", type=int, default=0)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--fuse-minutes", type=int, default=360)
    ap.add_argument("--overrun-minutes", type=int, default=0)
    ap.add_argument("--rearm-fuse", action="store_true")
    ap.add_argument("--stop-on-complete", action="store_true")
    args = ap.parse_args()

    validate_no_external_api_surface([pathlib.Path(__file__)])
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    progress = NumericProgress(out_dir=out, bucket=args.bucket, out_prefix=args.out_prefix,
                               fuse_minutes=args.fuse_minutes, overrun_minutes=args.overrun_minutes,
                               rearm_fuse=args.rearm_fuse)
    t0 = time.time()

    field = load_field(args.substrate, args.raw_dir, "cpu")
    statements = load_problem_statements()
    episode_text = load_raw_episode_text(args.raw_dir)
    real_texts = [str(v["statement"]) for v in statements.values()]
    codec_z = np.load(REPO / "artifacts" / "cw_item_codec_dim64.npz",
                      allow_pickle=False)
    problem_codecs = np.asarray(codec_z["codec"], dtype=np.float32)
    problem_ids = np.asarray(codec_z["problem_id"], dtype=np.int64)
    progress.heartbeat("field", 1, 1)

    specs: list[dict[str, Any]] = []
    for start in PANEL_STARTS:
        eps = field["scenarios"][f"val_start{start}"]
        if args.pilot_learners:
            eps = eps[:args.pilot_learners]
        for ep in eps:
            specs.append({"learner": str(ep["learner"]), "start": int(start),
                          "events": list(ep["prefix_events"]), "episode": ep})
    # The swapped-history partner: a seed-fixed DERANGEMENT over learners WITHIN a start, so
    # history length and curriculum position are held fixed and only the person moves.  The
    # audited `atom.binding.derangement_permutation` is used rather than a randperm -- a
    # random permutation leaves fixed points, and a fixed point here would silently make the
    # swap arm identical to the history arm for that learner.
    swap_of: dict[tuple[str, int], list[Any]] = {}
    for start in PANEL_STARTS:
        rows = [x for x in specs if x["start"] == start]
        if len(rows) < 2:
            continue
        gen = torch.Generator().manual_seed(int(stable_uint64(f"ga1-tier4-swap|{start}") % (2 ** 31)))
        perm = derangement_permutation(len(rows), generator=gen)
        assert_derangement(perm)
        for i, r in enumerate(rows):
            partner = rows[int(perm[i])]
            if partner["learner"] == r["learner"]:
                raise RuntimeError("swap check: a learner was paired with itself")
            swap_of[(r["learner"], start)] = list(partner["events"])
            r["swap_partner"] = partner["learner"]
    progress.heartbeat("panel_states", len(specs), len(specs))

    bundle = load_local_gemma(args.device)
    if bundle["hf_id"] != EXPECTED_HF_ID:
        raise RuntimeError("generator mismatch")
    tok = bundle["tokenizer"]
    bases, context = attach_plain_lora(bundle["model"], SFT_SEEDS[0])

    hist_prompts, swap_prompts = [], []
    for spec in specs:
        p = build_tier1_prompt(spec["events"], episode_text, tok, max_history_tokens=HISTORY_TOKENS)
        w = build_tier1_prompt(swap_of[(spec["learner"], spec["start"])], episode_text, tok,
                               max_history_tokens=HISTORY_TOKENS)
        for text in (p, w):
            if "CURRENT MENU" in text or "TOTAL DIAGNOSTIC BUDGET" in text:
                raise RuntimeError("menu/budget leak check fired")
        hist_prompts.append(p)
        swap_prompts.append(w)
    # The no-history prompt is ONE identical string, so one row per (learner, start) would
    # only resample it.  Generate 110 -- one per learner -- and reuse across that learner's
    # starts.  The pairing is not true CRN, and the bootstrap is learner-clustered, so no
    # statistic changes meaning; the caption says so.
    nohist_prompt = build_tier1_prompt([], episode_text, tok, max_history_tokens=HISTORY_TOKENS)
    if "CURRENT MENU" in nohist_prompt or "TOTAL DIAGNOSTIC BUDGET" in nohist_prompt:
        raise RuntimeError("menu/budget leak check fired")
    nohist_learners = sorted({s0["learner"] for s0 in specs})
    nohist_prompts = [nohist_prompt] * len(nohist_learners)

    store = LocalGenerationStore(out / "generations.jsonl")
    rows: list[dict[str, Any]] = []
    for seed in SFT_SEEDS[:args.n_seeds]:
        adapter_path = REPO / args.sft_dir / f"adapter_seed{seed}.pt"
        saved = torch.load(adapter_path, map_location=args.device, weights_only=False)
        for arm in ARMS:
            if arm == "tuned":
                for name, b in bases.items():
                    b.load_state_dict(saved["bases"][name])
            for mode in MODES:
                prompts = {"history": hist_prompts, "swapped_history": swap_prompts,
                           "no_history": nohist_prompts}[mode]
                texts = generate_sampled(
                    bundle, context, prompts, adapter_on=(arm == "tuned"),
                    batch_seed_tag=f"ga1-tier4-decode|{seed}|{arm}|{mode}",
                    batch_size=args.batch_size, max_prompt_tokens=args.max_prompt_tokens,
                    progress=progress, stage=f"generate_{arm}_{mode}_seed{seed}")
                if mode == "no_history":
                    by_learner = dict(zip(nohist_learners, texts))
                    pairs = [(spec, by_learner[spec["learner"]]) for spec in specs]
                else:
                    pairs = list(zip(specs, texts))
                for spec, text in pairs:
                    row = {"seed": int(seed), "arm": arm, "mode": mode,
                           "learner": spec["learner"], "start": spec["start"],
                           "question": text}
                    store.append(row)
                    rows.append(row)

    # The codec is the RULER and never moves: always evaluated with the adapter off.
    context.set(torch.zeros((1, LORA_M), device=bundle["device"],
                            dtype=next(bundle["model"].parameters()).dtype))
    unique = sorted({r["question"] for r in rows})
    vectors = embed_questions(bundle, unique,
                              on_row=lambda d, t: progress.maybe_heartbeat("codec", d, t))
    vec_of = {q: vectors[i] for i, q in enumerate(unique)}
    release_backbone(bundle)

    # Pointing: EPIG of the generated question at that state's belief, expressed as its
    # percentile within the SAME state's real-menu EPIG distribution.
    model = field["model"]
    epig_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for n, spec in enumerate(specs, start=1):
        ep = spec["episode"]
        query = ep["targets"][0]["codec"]
        mu, _ = _frozen_filter_state(model, sorted(spec["events"], key=lambda x: x["order"]),
                                     query, "cpu")
        eps = state_eps(spec["learner"], 0, spec["start"], mu.numel(), "cpu")
        scored = _candidate_scores(model, mu, list(ep["candidates"]), ep["targets"], eps, "cpu")
        # The TRUE realised advantage of every menu item at this state -- the
        # training signal itself.  Primary pointing is scored against this, not against EPIG.
        exact = state_rows(model, ep, {"learner": spec["learner"], "budget": 0,
                                       "step": int(spec["start"]), "picks": [],
                                       "remaining": list(range(len(ep["candidates"])))},
                           "cpu", with_epig=False)
        epig_cache[(spec["learner"], spec["start"])] = {
            "mu": mu, "eps": eps, "targets": ep["targets"],
            "menu_epig": np.asarray([scored[i]["epig"] for i in range(len(ep["candidates"]))]),
            "menu_r": np.asarray(exact["r"], dtype=np.float64),
            "menu_pids": np.asarray(exact["problem_ids"], dtype=np.int64),
            "history_pids": {int(e["problem_id"]) for e in spec["events"]},
        }
        progress.maybe_heartbeat("menu_exact_r", n, len(specs))

    for n, row in enumerate(rows, start=1):
        c = epig_cache[(row["learner"], row["start"])]
        codec_vec = torch.tensor(vec_of[row["question"]], dtype=torch.float32)
        scored = _candidate_scores(model, c["mu"], [{"codec": codec_vec}], c["targets"],
                                   c["eps"], "cpu")
        row["epig"] = float(scored[0]["epig"])
        row["epig_percentile"] = float((c["menu_epig"] < row["epig"]).mean())
        row["menu_epig_median"] = float(np.median(c["menu_epig"]))
        sim, arg = max_pool_similarity(row["question"], real_texts)
        row["max_pool_similarity"] = sim
        row["nearest_pool_problem_id"] = int(list(statements)[arg]) if arg >= 0 else -1
        d = np.linalg.norm(problem_codecs - vec_of[row["question"]][None, :], axis=1)
        proj_pid = int(problem_ids[int(d.argmin())])
        row["nearest_codec_problem_id"] = proj_pid
        row["nearest_codec_distance"] = float(d.min())
        # PRIMARY pointing: project to the nearest real pool item; if that item is in
        # this state's own menu, report the percentile of its TRUE realised r within the
        # menu's r distribution.  Projections outside the menu are a MISS and are counted,
        # never silently dropped.
        hit = np.flatnonzero(c["menu_pids"] == proj_pid)
        if len(hit):
            row["realised_r"] = float(c["menu_r"][int(hit[0])])
            row["realised_r_percentile"] = float((c["menu_r"] < row["realised_r"]).mean())
            row["projection_miss"] = 0
        else:
            row["realised_r"] = float("nan")
            row["realised_r_percentile"] = float("nan")
            row["projection_miss"] = 1
        # A plausible way to look history-sensitive while being useless.
        row["nearest_already_in_history"] = int(proj_pid in c["history_pids"])
        progress.maybe_heartbeat("score_generations", n, len(rows))

    # ------------------------------------------------------------------ the panel tables
    def cell(arm: str, mode: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["arm"] == arm and r["mode"] == mode]

    real_dist = {f"dist_{n}": dist_n(real_texts, n) for n in (1, 2)}
    panel: dict[str, Any] = {"real_reference": {"dist": real_dist, "n_problems": len(real_texts)}}
    for arm in ARMS:
        for mode in MODES:
            sub = cell(arm, mode)
            vecs = np.stack([vec_of[r["question"]] for r in sub])
            man = generated_manifold_metrics(vecs, problem_codecs)
            by_seed_pct, by_seed_sim, by_seed_rr = [], [], []
            for seed in SFT_SEEDS[:args.n_seeds]:
                s_rows = [r for r in sub if r["seed"] == seed]
                by_seed_pct.append(np.asarray([r["epig_percentile"] for r in s_rows]))
                by_seed_sim.append(np.asarray([r["max_pool_similarity"] for r in s_rows]))
                # A miss is not a zero and must never be averaged as one; aggregate over
                # HITS and report the miss rate beside the number (keeping the NaNs would make
                # every primary cell NaN).
                _rr = np.asarray([r["realised_r_percentile"] for r in s_rows])
                by_seed_rr.append(_rr[np.isfinite(_rr)])
            panel[f"{arm}|{mode}"] = {
                "n": len(sub),
                "manifold_support_at_real_p95": man["support_fraction_at_real_p95"],
                "nearest_distance_mean_ci": bootstrap_mean(man["nearest_distance"], 20260930),
                "dist_1": dist_n([r["question"] for r in sub], 1),
                "dist_2": dist_n([r["question"] for r in sub], 2),
                "dist_1_frac_of_real": (dist_n([r["question"] for r in sub], 1) or 0.0)
                / max(real_dist["dist_1"] or 1e-9, 1e-9),
                "dist_2_frac_of_real": (dist_n([r["question"] for r in sub], 2) or 0.0)
                / max(real_dist["dist_2"] or 1e-9, 1e-9),
                "unique_normalised_fraction": len({normalise_text(r["question"]) for r in sub})
                / max(len(sub), 1),
                # PRIMARY pointing; EPIG percentile is secondary below.
                "realised_r_percentile_PRIMARY": _bootstrap_seed_learner(by_seed_rr, 20260936),
                "projection_miss_rate": float(np.mean([r["projection_miss"] for r in sub])),
                "nearest_already_in_history_fraction": float(
                    np.mean([r["nearest_already_in_history"] for r in sub])),
                "within_start_internal_spread": {
                    str(st): mean_pairwise_distance(
                        np.stack([vec_of[r["question"]] for r in sub if r["start"] == st]))
                    for st in PANEL_STARTS if any(r["start"] == st for r in sub)},
                "epig_percentile_secondary": _bootstrap_seed_learner(by_seed_pct, 20260931),
                "max_pool_similarity": _bootstrap_seed_learner(by_seed_sim, 20260932),
                "exact_pool_match_fraction": float(np.mean(
                    [normalise_text(r["question"]) in {normalise_text(t) for t in real_texts}
                     for r in sub])),
                "internal_mean_pairwise_codec_distance": mean_pairwise_distance(vecs),
            }
    for arm in ARMS:
        h = cell(arm, "history")
        by_mode = {m: {(r["seed"], r["learner"], r["start"]): r for r in cell(arm, m)}
                   for m in ("no_history", "swapped_history")}
        sens: dict[str, Any] = {}
        for mode, seed_off in (("swapped_history", 20260937), ("no_history", 20260933)):
            per_seed = []
            for seed in SFT_SEEDS[:args.n_seeds]:
                vals = [float(np.linalg.norm(
                    vec_of[r["question"]]
                    - vec_of[by_mode[mode][(seed, r["learner"], r["start"])]["question"]]))
                    for r in h if r["seed"] == seed]
                per_seed.append(np.asarray(vals))
            sens[f"paired_codec_distance_vs_{mode}"] = _bootstrap_seed_learner(per_seed, seed_off)
        # The SWAP is the headline -- it holds curriculum position fixed and moves
        # only who the student is.  The no-history contrast is the secondary reference line.
        sens["headline"] = "paired_codec_distance_vs_swapped_history"
        sens["no_history_internal_spread"] = panel[f"{arm}|no_history"][
            "internal_mean_pairwise_codec_distance"]
        sens["history_internal_spread"] = panel[f"{arm}|history"][
            "internal_mean_pairwise_codec_distance"]
        sens["within_start_history_spread"] = panel[f"{arm}|history"]["within_start_internal_spread"]
        sens["within_start_swapped_spread"] = panel[f"{arm}|swapped_history"][
            "within_start_internal_spread"]
        panel[f"{arm}|history_sensitivity"] = sens
    for mode in MODES:
        tuned = {(r["seed"], r["learner"], r["start"]): r for r in cell("tuned", mode)}
        per_seed_pct, per_seed_nn, per_seed_rr = [], [], []
        for seed in SFT_SEEDS[:args.n_seeds]:
            z = [r for r in cell("zero_shot", mode) if r["seed"] == seed]
            keys = [(seed, r["learner"], r["start"]) for r in z]
            per_seed_pct.append(np.asarray([tuned[k]["epig_percentile"] for k in keys])
                                - np.asarray([r["epig_percentile"] for r in z]))
            per_seed_nn.append(np.asarray([tuned[k]["nearest_codec_distance"] for k in keys])
                               - np.asarray([r["nearest_codec_distance"] for r in z]))
            _d = (np.asarray([tuned[k]["realised_r_percentile"] for k in keys])
                  - np.asarray([r["realised_r_percentile"] for r in z]))
            per_seed_rr.append(_d[np.isfinite(_d)])   # paired on hits in BOTH arms
        panel[f"movement|{mode}"] = {
            "tuned_minus_zero_shot_realised_r_percentile_PRIMARY":
                _bootstrap_seed_learner(per_seed_rr, 20260938),
            "tuned_minus_zero_shot_epig_percentile_secondary":
                _bootstrap_seed_learner(per_seed_pct, 20260934),
            "tuned_minus_zero_shot_nearest_distance": _bootstrap_seed_learner(per_seed_nn, 20260935),
        }
    # The personalisation difference-in-differences: does TRAINING widen the gap between a
    # learner's own history and someone else's?  This is the pre-registered mechanism read.
    did = []
    for seed in SFT_SEEDS[:args.n_seeds]:
        gaps = {}
        for arm in ARMS:
            h = [r for r in cell(arm, "history") if r["seed"] == seed]
            sw = {(r["learner"], r["start"]): r for r in cell(arm, "swapped_history")
                  if r["seed"] == seed}
            gaps[arm] = np.asarray([
                float(np.linalg.norm(vec_of[r["question"]]
                                     - vec_of[sw[(r["learner"], r["start"])]["question"]]))
                for r in h])
        did.append(gaps["tuned"] - gaps["zero_shot"])
    panel["movement|personalisation_did_swap"] = {
        "tuned_minus_zero_shot_own_vs_swapped_history_distance":
            _bootstrap_seed_learner(did, 20260939),
        "reading": ("positive = training widened the separation between this learner's own "
                    "history and another learner's at the same start"),
    }

    manifest = {
        "lane": "T4", "rung": "B", "stage": "panel", "substrate": args.substrate,
        "model": EXPECTED_HF_ID, "checkpoint_revision": bundle.get("checkpoint_revision"),
        "decode": {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P,
                   "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS,
                   "note": "greedy decoding was measured to degenerate on this task and is never used"},
        "arms": [f"{a}|{m}" for a in ARMS for m in MODES],
        "n_states": len(specs), "n_generations": len(rows),
        "seeds": list(SFT_SEEDS[:args.n_seeds]), "starts": list(PANEL_STARTS),
        "canaries": {"menu_budget_leak": "pass", "no_history_prompt_identical": "pass",
                     "empty_generation": "pass", "codec_evaluated_with_adapter_off": True},
        "n_training_runs": len(SFT_SEEDS[:args.n_seeds]),
        "disclosures": [
            # The pre-registered design uses ONE training seed.  This string is emitted
            # verbatim into every table, as for the fine-tuned selector.
            ("single training run — no seed-level clustering is possible and none is claimed"
             if args.n_seeds == 1 else
             f"{args.n_seeds} training runs; intervals are seed- and learner-clustered"),
            "MOVEMENT claim about the generative distribution, never a question-quality claim "
            "and never 'novel questions help students'.",
            "On a 50-question corpus partial memorisation is EXPECTED; the memorisation panel "
            "reports it rather than hiding it.",
            "Manifold support 0.90 is a reference line, not a gate; a failure is a result.",
            "The instruments are the mind-model's own codec/EPIG, by design; this is not an "
            "independent judgement of question quality.",
            "Tuned no-history rows are OUT OF THE TRAINING DISTRIBUTION: under the warm draw "
            "the tuned model never saw an empty history. Reference line only.",
            "The utility-model (U-hat) validation used the COLD DRAW (start 0, 11.1% cold "
            "rows). It stands because it is an instrument-validation claim and not a "
            "personalisation claim -- but the two draws do not match, and this says so.",
            "PRIMARY pointing is the realised-r percentile of the projected menu item. EPIG "
            "percentile is SECONDARY: the utility-model validation measured Spearman(EPIG, true r) within a state "
            "at only 0.07-0.10, so an EPIG-percentile null is NOT evidence that the signal "
            "failed to shape generation -- the training signal is r, and the panel scores "
            "against it.",
            "The no-history arm is 110 generations per model (one per learner, reused across "
            "that learner's starts). The no-history prompt is a single identical string, so "
            "one row per (learner, start) would only resample it; pairing is by index under "
            "matched batch seeds and not true CRN, and the bootstrap is learner-clustered, so no "
            "statistic changes meaning.",
            "History-sensitivity's HEADLINE is the swapped-history contrast, not "
            "history-vs-no-history: every history block prints the completed exercise's own "
            "statement and CW's assignment order is fixed, so a with/without gap is "
            "substantially curriculum position rather than the person.",
        ],
        "elapsed_s": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"status": "complete", "manifest": manifest, "panel": panel}, indent=2, default=str) + "\n")
    np.savez_compressed(
        out / "generation_rows.npz",
        seed=np.asarray([r["seed"] for r in rows]),
        arm=np.asarray([r["arm"] for r in rows]), mode=np.asarray([r["mode"] for r in rows]),
        learner=np.asarray([r["learner"] for r in rows]),
        start=np.asarray([r["start"] for r in rows]),
        epig=np.asarray([r["epig"] for r in rows], dtype=np.float32),
        epig_percentile=np.asarray([r["epig_percentile"] for r in rows], dtype=np.float32),
        realised_r=np.asarray([r["realised_r"] for r in rows], dtype=np.float32),
        realised_r_percentile=np.asarray([r["realised_r_percentile"] for r in rows],
                                         dtype=np.float32),
        projection_miss=np.asarray([r["projection_miss"] for r in rows], dtype=np.int64),
        nearest_already_in_history=np.asarray([r["nearest_already_in_history"] for r in rows],
                                              dtype=np.int64),
        max_pool_similarity=np.asarray([r["max_pool_similarity"] for r in rows], dtype=np.float32),
        nearest_codec_problem_id=np.asarray([r["nearest_codec_problem_id"] for r in rows]),
        nearest_codec_distance=np.asarray([r["nearest_codec_distance"] for r in rows],
                                          dtype=np.float32),
        question_hash=np.asarray([stable_hash(r["question"], 32) for r in rows]))
    for name in ("manifest.json", "summary.json", "generation_rows.npz"):
        progress.upload(out / name)
    print(json.dumps(panel, indent=2, default=str))
    stop_instance_if_requested(args.stop_on_complete)
    return 0
