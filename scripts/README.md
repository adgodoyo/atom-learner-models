# scripts/

Entry points, grouped by the part of the thesis they serve. Run them from the repository root
(`uv run python scripts/<name>.py`, or `infra/<name>.py`); `REPRODUCE.md` gives the exact commands.
File names keep the identifiers used during development. Modules not listed here are shared helpers
imported by these scripts.

## Data and splits (Chapter 4)

| Script | What it does |
|---|---|
| `scripts/fa_00_manifest_rebuild.py` | Build the canonical FoundationalASSIST (FA) event manifest and its learner and item splits. |
| `scripts/cw_1_event_graph.py` | Build the CodeWorkout attempt table from the S19 and F19 full releases. |
| `scripts/cw_2_sessions.py` | Sessionise CodeWorkout attempts into episodes, build episode targets, and write the splits. |
| `scripts/cw_4c_step0_1_splits.py` | Adopt the release's official learner split and freeze the 7-item CodeWorkout holdout. |
| `scripts/fetch_codingbat_statements.py` | Fetch the 50 CodeWorkout problem statements from their public CodingBat pages. |
| `scripts/cw_4b_recover_problem_statements.py` | Recover the 50 CodeWorkout problem statements from CodingBat and encode them as `s(problem)`. |
| `scripts/p1_05_data_report.py` | Generates the FoundationalASSIST data report: distributions, session gaps and answer lengths, cross-checked against the dataset summary statistics quoted in the thesis. |

## Semantic codec (§5.2) and Results I (Chapter 6)

| Script | What it does |
|---|---|
| `scripts/p2_01_backbone.py` | Backbone due diligence for gemma4-e4b, the frozen backbone of the codec and simulators. |
| `scripts/p2_02a_extract_calibration.py` | Builds the raw hidden-state calibration cache for the FA codec (answer-in-context encoding). |
| `scripts/p2_07_encode_problems.py` | Encodes all FA problems into the production item codec and freezes the problem transform. |
| `scripts/p5b_02_encode_cw_code.py` | Encodes all CodeWorkout code submissions (code alone) at the production configuration chosen by the probe battery: mean pooling (geometry) and boundary-mean (evidence) at layer 15, both raw and undeflated. |
| `infra/cw_03_encode.py` | Encodes every CodeWorkout code submission and episode-local code diff with the frozen codec. |
| `infra/cw_03_encode_tiktoc_all.py` | Encodes TIKTOC's full attempt-level file (code and episode-local diffs) with the frozen codec. |
| `scripts/p2_05b_probe_battery.py` | Code-representation probe battery on CodeWorkout: which axes of code meaning the frozen codec carries (Table 6). |
| `scripts/fa_pg_probe.py` | Probe whether item difficulty and discrimination are recoverable from the frozen problem-text codec. |
| `scripts/fa_01_majority_irt_v2.py` | Majority and IRT-2PL baselines on the rebuilt FA manifest (v2). |
| `scripts/fa_01_akt_v2.py` | Train and evaluate the AKT baseline on the rebuilt FA manifest (v2), full corpus. |
| `scripts/fa_01b_akt_no_tags.py` | AKT without curriculum tags: the same pyKT AKT and config, with concepts = problem IDs. |
| `infra/p3_05b_akt_full_train.py` | Trains the AKT baseline on the full FoundationalASSIST training split on a GPU instance and reports its validation AUC, accuracy, ECE and Brier score. |

## FA-SIM3, the mathematics simulator (Chapters 5 and 7)

| Script | What it does |
|---|---|
| `scripts/fa_01i_item_encoder_ablation.py` | Item-encoder ablation: does a nonlinear item pathway raise filter AUC, and does the latent:hidden ratio matter once it does?. |
| `scripts/fa_01k_b4_close.py` | Train the outcome-only filter used as FA-SIM3's warm start, and measure it. |
| `scripts/fa_sim3_stage_bundles.py` | Builds the two text bundles FA-SIM3 trains and evaluates on, locally and deterministically. |
| `scripts/fa_sim3_smoke_local.py` | Local smoke test for `infra/fa_sim3_full_run.py`, run on a stub causal LM and tiny synthetic data. |
| `infra/fa_sim3_full_run.py` | Trains FA-SIM3, the mathematics simulator, and evaluates it on validation learners. |
| `scripts/fa_sim3_kt_panel.py` | FA-SIM3 knowledge-tracing and cognitive-accuracy panel, reported in the shape used by the FoundationalASSIST paper; reads persisted artifacts only (no training, no GPU). |
| `scripts/fa_01p_court_baselines.py` | Re-score the AKT, IRT and Majority baselines on FA-SIM3's own 4,977-row evaluation court. |
| `scripts/fa_01p_court_paired.py` | Paired learner-clustered bootstrap contrasts between predictors on the shared FA court. |
| `scripts/fa_sim3_eval_5e5f.py` | Score FA-SIM3's free generations against the learners' real answers, per arm. |
| `scripts/fa_sim3_softmatch_conditional.py` | Tests whether the soft-match generation metrics (edit distance, codec similarity) for FA-SIM3 are a by-product of exact match, by re-analysing already-measured per-row scores. |

## CodeWorkout classifier and generative simulator (Chapters 5 and 7)

| Script | What it does |
|---|---|
| `scripts/cw_sigma.py` | Train and evaluate the CodeWorkout classifier under the K-sample mixture likelihood. |
| `scripts/cw_4f_clean_full_auc.py` | Train the CodeWorkout classifier at its final configuration and report its `P(clean-full)` AUC. |
| `scripts/cw_first_submission_baselines.py` | CodeWorkout first-submission correctness: ATOM against two simple baselines (thesis Table 9). |
| `scripts/cw_classifier_belief_handoff.py` | Export the CodeWorkout classifier's dense validation prefix beliefs for the planner. |
| `infra/cw_sim_full_run.py` | Trains and evaluates the CodeWorkout generative simulator (Table 11). |
| `scripts/cw_sim_smoke_local.py` | Local smoke test for the CodeWorkout generative simulator's training step. |
| `scripts/cw_sim_head_refresh.py` | Retrain the outcome head of the CodeWorkout generative simulator on its frozen belief. |
| `scripts/cw_sim_tv_head_refresh.py` | Retrain the test-vector head of the CodeWorkout generative simulator on a frozen filter. |
| `scripts/cw_sim_test_vector_labels.py` | Per-attempt test-vector labels for the CodeWorkout generative simulator. |
| `scripts/cw_sim_belief_handoff.py` | Export the CodeWorkout generative simulator's dense validation prefix beliefs for the planner. |
| `scripts/cw_sim_train_handoff.py` | Write train-learner belief states from the CodeWorkout generative simulator. |
| `infra/cw_sim_launch2.py` | Code-generation panel for the CodeWorkout generative simulator (Table 11). |
| `infra/cw_sim_launch2_model.py` | Shared helpers for the generation evaluations of the CodeWorkout generative simulator. |
| `infra/cw_sim_launch2_tiktoc.py` | TIKTOC same-data evaluation of the CodeWorkout generative simulator (Table 11). |
| `infra/cw_ext_1_row_selection.py` | Selects the first-submission rows on the 30 CodeWorkout problems not yet generated, and checks the selection. |
| `infra/cw_ext_1_generate.py` | Generates the CodeWorkout generative simulator's code for the first-submission rows on the 30 not-yet-generated CodeWorkout problems and scores it with CodeBLEU. |
| `infra/cw_ext_1_codebleu_score.py` | Standalone CodeBLEU scorer for generated code, run under a separate Python 3.11 interpreter. |
| `scripts/cw_5_functional_equivalence.py` | Functional-equivalence harness: compile and run a Java submission against a problem's TIKTOC test cases. |
| `scripts/cw_classmate_ceiling.py` | Similarity of ATOM's generated code to a learner's real submission, against two learner-blind references. |

## ATOM-Act, the planner (Chapter 8)

| Script | What it does |
|---|---|
| `scripts/corpus_person_signal.py` | Model-free person-signal comparison between CodeWorkout and FoundationalASSIST. |
| `scripts/pl_0_replay_repair.py` | Leak check for the planner's replay rules (§5.7): perturbing a candidate's logged future channels must leave every score and trace bit-identical. |
| `scripts/pl_1_epig_estimator.py` | Validates the planner's EPIG estimator (§5.7) on a synthetic toy with an exact posterior. |
| `scripts/pl_2_positive_control.py` | Planted-world positive control for the planner (§4.5): greedy EPIG must beat random in a world built so that winning is the right answer, otherwise the implementation is broken. |
| `scripts/pl_3_evsi_proper.py` | Pre-registered replay of the EPIG planner against blind baselines on FoundationalASSIST validation learners, on the repaired estimator, harness and belief (context for §8.2). |
| `scripts/pl_3_fa_sim3_binary.py` | The EPIG planner on the belief of FA-SIM3, the mathematics simulator (§8.2): a four-policy replay comparison on FoundationalASSIST validation learners. |
| `scripts/cw_planner_suite.py` | Budget-conditioned question-selection replay on CodeWorkout (the planner suite). |
| `scripts/rl_2.py` | Trains and evaluates the CW planner actor on three beliefs (§8.2): a pointer actor trained on replay per belief substrate, selected on validation, three seeds, against EPIG and blind baselines. |
| `scripts/sim2real_w.py` | Builds and checks the learner-specific menus for the policy trained only inside the simulator (§8.3, Figure 11), as a generation-free, fail-closed smoke check. |
| `scripts/sim2real_w_prepare_events.py` | Prepares the numeric event handoff for the policy trained only inside the simulator (§8.3) without moving licensed raw text. |
| `scripts/sim2real_w_run.py` | Runs the policy trained only inside the simulator (§8.3, Figure 11): warm-started, learner-specific imagined menus, actor training on imagined answers only, then the real replay exam. |
| `scripts/im_s3_dream.py` | Shared helpers for the policy trained only inside the simulator (§8.3, Figure 11): the dream corpus and the imagined-rollout environment. |
| `scripts/ga_1_tier0_run.py` | Zero-shot LLM selector (§8.4): the local Gemma backbone as one more policy in the replay of `rl_2.py`, compared with the trained actor, blind comparators and a privileged upper bound. |
| `scripts/ga_1_tier3_train.py` | Fine-tunes the LLM selector on the simulator's pricing by exact-advantage fine-tuning (Figure 12). |
| `scripts/ga_1_tier3_eval.py` | Evaluates the LLM selector fine-tuned on the simulator's pricing: the personalisation 2x2 and the blind comparator panel (Figure 12). |
| `scripts/ga_1_tier4_utility.py` | Utility model (§8.4): trains a counterfactual utility model Û(s,q) on exact one-step replay advantages and validates it on held-out questions against greedy-EPIG and random. |
| `scripts/ga_1_tier4_rungc.py` | Writing the question (Figure 13): a belief-conditioned, contrastive fine-tune of the local Gemma backbone that generates the next diagnostic question for a learner. |
| `scripts/ga_1_tier4_rungc_panel.py` | Before/after panel for the belief-conditioned question writer (Figure 13), run on the belief-blind fine-tune's panel harness unchanged. |

## Appendix A ablations

| Script | What it does |
|---|---|
| `scripts/p2_02c_sweep.py` | Codec sweep scored by curriculum-free item geometry: chooses the codec layer, pooling and deflation rank k for problem text. |
| `scripts/fa_01g_s1v2_width_sweep.py` | Belief-width sweep: does widening the S1v2 filter (`S1FilterV2`) close the gap to AKT?. |
| `scripts/cw_4i_latent_dim_isolation.py` | Belief-dimension ablation for the CodeWorkout classifier: `latent_dim` 32 vs 64, all else pinned. |
| `scripts/cw_4g_codec_rank_item_mlp.py` | Codec-width x item-encoder grid for the CodeWorkout classifier: P(clean-full) AUC, NLL, Brier, ECE per cell. |
| `scripts/cw_4g_3_item_holdout_check.py` | Item-holdout check for the CodeWorkout codec-width ablation: does a wider codec help on items never trained on?. |
| `scripts/cw_4c_2_did_and_prefix_curve.py` | Item-holdout difference-in-differences, tiered evaluation and prefix-length curves for the CodeWorkout model. |
| `scripts/fa_02a_strata_v2.py` | Derive and freeze the NEAR / FAR / REACH-BACK strata used by the filter-body comparison. |
| `scripts/fa_02_bakeoff.py` | Screen the filter body: GRU against a causal event-transformer and three attention-bias ablations. |
| `infra/fa_sigma_a3.py` | Retrains the filter and outcome head of FA-SIM3, the mathematics simulator, under an alternative belief-width parameterisation and tests whether the belief width tracks evidence. |
| `scripts/p3_04_ou_ablation.py` | OU forgetting-kernel on/off ablation (Appendix A): does explicitly modelling forgetting over real time gaps -- a mean-reverting Ornstein-Uhlenbeck bridge on the belief, applied between a learner's consecutive events -- improve calibration on predictions that immediately follow a long gap, compared with the S1 filter, which has no notion of elapsed real time?. |
| `scripts/fa_01l_time_features_screen.py` | Screen time, session and repeat-exposure features on the 4-class support head and on correctness. |
| `scripts/fc_1a_factored_heads.py` | Factored outcome head: trains a help-mode head and a conditional-correctness head on frozen filters and compares their product against the original 5-class outcome head. |
| `infra/sim_alpha_sweep.py` | Trains one arm of the token-loss-weight sweep for FA-SIM3, the mathematics simulator, and evaluates it on three panels. |
| `scripts/sim_alpha_selection.py` | Selects the token-loss weight alpha for the FA simulator (Appendix A) by applying the pre-registered selection rule to the persisted sweep results; no training and no new evaluation. |
| `scripts/wd_1_wrong_detection_panel.py` | Wrong-detection panel for the FA simulator variants (Appendix A): evaluation only, on existing checkpoints (FA-SIM3, the width-parameterisation variant `a3`, and the `br4_r1/r2/r3` variants), with no further training. |
| `scripts/fa_06_support_baselines.py` | Score two closed-form support baselines (learner-prefix and lag-1+gap) on the FA validation split. |

## Release utilities

| Script | What it does |
|---|---|
| `scripts/materialize_checkpoints.py` | Rebuild the trained checkpoints that the scripts load, from the published safetensors. |
| `scripts/p0_03_smoke.py` | Synthetic end-to-end pipeline smoke test, used by `make smoke`. |

## Shared helpers

| Module | What it provides |
|---|---|
| `infra/cw_03_encode_tiktoc.py` | Encodes the code text of TIKTOC's 17-problem benchmark submissions with the frozen codec. |
| `infra/cw_filter_eval.py` | Shared helpers for the CodeWorkout filter-evaluation scripts. |
| `infra/cw_sim_belief_chunks.py` | Shared helpers for `cw_sim_tv_head_refresh`: learner-aligned chunking of an attempts frame. |
| `infra/cw_sim_core.py` | Shared helpers for training and evaluating the CodeWorkout generative simulator. |
| `infra/fa_filter_common.py` | Shared helpers for `fc_1a_factored_heads` and `wd_1_wrong_detection_panel`. |
| `scripts/cw_classifier_common.py` | Shared helpers for `cw_sigma`, `cw_sim_head_refresh` and `cw_sim_tv_head_refresh`. |
| `scripts/cw_classifier_model_dict.py` | Shared helpers for `cw_classifier_belief_handoff`. |
| `scripts/cw_codebleu_all50.py` | Thesis Table 11, all-50-problem CodeBLEU: the row-weighted mean of the three generation strata. |
| `scripts/cw_fig06_maturity_data.py` | Data for thesis Figure 6: ATOM's CodeWorkout first-submission AUC by belief maturity. |
| `scripts/cw_manifest_io.py` | Shared helpers for loading the CodeWorkout training manifest. |
| `scripts/cw_planner_common.py` | Shared helpers for the CodeWorkout planner scripts. |
| `scripts/cw_sim_model_dict.py` | Shared helpers for loading the CodeWorkout generative simulator's checkpoint. |
| `scripts/cw_sim_smoke_common.py` | Shared helpers for `cw_sim_smoke_local`. |
| `scripts/fa_04b_stage_bundle.py` | Builds a local FA prompt bundle: a seeded subsample of events with their question and answer text. |
| `scripts/fa_bootstrap_utils.py` | Shared helpers for `fc_1a_factored_heads` (Appendix A, factored head). |
| `scripts/fa_stage_corrected_bundles.py` | Rebuilds the FA prompt bundles with corrected question text: problem-type tag, cleaned body, MC options. |
| `scripts/finalize_belief_handoffs.py` | Bring rebuilt planner belief handoffs into the exact form the thesis runs used, and check them. |
| `scripts/ga_1_common.py` | Shared helpers for the LLM-selector experiments of §8.4 (`ga_1_tier0_*`, the zero-shot selector, and the history-only question proposer). |
| `scripts/ga_1_tier0_prompt.py` | Shared helpers for the LLM selectors of §8.4: single-token menu identifiers and the selector prompt renderer, used by the zero-shot selector (`ga_1_tier0_*`) and, unchanged, by the selector fine-tuned on the simulator's pricing (`ga_1_tier3_*`). |
| `scripts/ga_1_tier0_readout.py` | Shared helpers for the zero-shot LLM selector (§8.4): the constrained identifier read-out. |
| `scripts/ga_1_tier3_lora.py` | Shared helpers for the fine-tuned LLM selector (Figure 12): plain (unconditioned) LoRA on the backbone's text q/v projections. |
| `scripts/ga_1_tier3_policy.py` | Shared helpers for the fine-tuned LLM selector (Figure 12): the shuffled LLM selection policy, used by both the trainer and the evaluator. |
| `scripts/ga_1_tier3_states.py` | Shared helpers for the fine-tuned LLM selector (Figure 12): training-state draw, CRN identifier shuffle and exact one-step advantages. |
| `scripts/ga_1_tier4_common.py` | Shared helpers for the utility model and question-writing experiments of §8.4 (`ga_1_tier4_*`): the state draw and the exact-advantage table. |
| `scripts/ga_1_tier4_generation_common.py` | Shared helpers for `ga_1_tier4_rungc` (writing the question, Figure 13): advantage-weighted target sampling, prompt/target encoding with suffix-only loss, left-padded collation with a windowed vocabulary projection, token-budget batching, plain LoRA attachment, and an instantaneous-rate overrun trigger for long training stages. |
| `scripts/ga_1_tier4_panel_common.py` | Shared helpers for `ga_1_tier4_rungc_panel` (writing the question, Figure 13): lexical memorisation and internal-spread statistics, seeded sampled decoding, and the panel `main()` that generates questions with and without the learner's history and with a swapped history, projects them through the frozen codec, and scores pointing, manifold support and history-sensitivity. |
| `scripts/imagination_core.py` | Shared helpers for `sim2real_w_run` (§8.3, the policy trained only inside the simulator): the simulator-only training contract. |
| `scripts/imagination_cw.py` | Shared helpers for `sim2real_w_prepare_events` and `sim2real_w_run` (§8.3, the policy trained only inside the simulator): loading and validating the numeric train handoff of the CodeWorkout generative simulator, the frozen CW item codec, encoding generated code into the frozen 64-D code codec and attempt summary, and the simulator-only pilot/training loop around the `rl_2.py` pointer actor. |
