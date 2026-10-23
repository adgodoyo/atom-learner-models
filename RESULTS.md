# Results

Every number below is printed in the thesis (`paper/ATOM_thesis_Godoy_2026.pdf`). Tables and figures carry the thesis's own numbering. Intervals are learner-clustered 95% bootstrap intervals unless stated. All evaluation is on learners held out of all training: 500 on mathematics (FoundationalASSIST, "FA") and 110 on programming (CodeWorkout, "CW").

**Columns.** *Script* is the entry point in `scripts/` or `infra/`. *Output* is the published aggregate file in `results/`, or, where the script writes a report, the report it regenerates under `reports/` when you run it. [`REPRODUCE.md`](REPRODUCE.md) gives each command, its hardware and its runtime.

## Chapter 6 · Meaning in place of a curriculum

| Thesis | What | Value | Comparison | Script | Output |
|---|---|---|---|---|---|
| Table 6 | Same curriculum skill? (FA problems, codec probe) | 0.819 AUC | raw hidden states 0.749 | `p2_02c_sweep` | `reports/p2_02c_sweep.md` |
| Table 6 | Which of 50 problems does this code solve? | 0.995 [0.989, 0.999] | edit distance 0.917 | `p2_05b_probe_battery` | `reports/p2_05b_probe_battery.md` |
| Table 6 | Who wrote this code? | 0.874 [0.838, 0.907] | edit distance 0.868 (overlapping) | `p2_05b_probe_battery` | as above |
| Table 6 | Do two submissions pass the same tests? | 0.523 [0.469, 0.573] | pre-registered floor 0.70 | `p2_05b_probe_battery` | as above |
| Table 7 | Which problem is harder? (356 held-out problems) | 72.4% | best frontier model 68.6% | `fa_pg_probe` | `results/fa_pg_probe/probe.json` |
| Table 7 | Which problem is more discriminating? | 65.4% | frontier models 31.1%–46.9% | `fa_pg_probe` | as above |
| §6.2 | AKT with all 462 skill tags vs without | 0.6982 vs 0.6922 (tags worth 0.0060) | ATOM's filter 0.6913 on the same court | `fa_01_akt_v2`, `fa_01b_akt_no_tags`, `fa_01i_item_encoder_ablation` | `reports/fa_01b_akt_no_tags.md`, `reports/fa_01i_item_encoder_ablation.md` |

## Chapter 7 · Simulating a learner

| Thesis | What | Value | Comparison | Script | Output |
|---|---|---|---|---|---|
| Table 8 | FA next-response correctness | 0.786 AUC, 73.7% accuracy | AKT (with tags) 0.810, 75.3%; frontier LLMs 0.512–0.559 (published, Worden et al. 2026) | `fa_sim3_kt_panel`, `fa_01p_court_baselines`, `fa_01p_court_paired` | `results/fa_sim3_kt_panel/panel.json`, `results/fa_court_baselines/{panel,paired}.json` |
| §7.1.1 | AKT's paired lead on the same 4,914 rows | 0.0237 [0.0149, 0.0324] | | `fa_01p_court_paired` | `results/fa_court_baselines/paired.json` |
| §7.1.1 | Within a single problem (741 problems) | 0.753 [0.732, 0.777] | learner-blind 0.500 | `fa_sim3_kt_panel` | `results/fa_sim3_kt_panel/panel.json` |
| §7.1.1 | Belief width vs prediction error | correlation 0.484 | | `infra/fa_sim3_full_run` | `results/fa_sim3_full_run/summary.json` |
| Figure 5 | Own belief vs a length-matched stranger's, by history | +0.076 AUC (< 10 events) to +0.229 (200+) | | `paper/figures/fig05_data.py` | `paper/figures/data/fig05_fa_auc_by_history.csv` |
| Table 9 | CW first-submission correctness (4,593 first submissions) | 0.833 AUC | KCGen-KT 0.816 (published) | `cw_first_submission_baselines` | `results/cw_first_submission_baselines/summary.json` |
| Table 9 / §7.1.2 | Two scalars (item difficulty, learner ability) · problem-ID-only GRU · within-problem ATOM | 0.795 · 0.772 · 0.796 | | `cw_first_submission_baselines` | as above |
| §7.1.2 | Seven problems held out of training: ability only · GRU trained without them | 0.805 · 0.783 | | `cw_first_submission_baselines` | as above |

**Note on Table 9's baselines.** The released script reproduces ATOM's numbers (0.8326, within-problem 0.7962) and the two-scalar baseline (0.7948) exactly, as well as Figure 6's points; Figure 6's first-bucket interval depends on the bootstrap seed ([0.322, 0.651] with the released seed). Two baselines were recomputed from their published description because the original code was not kept. The identifier-only recurrent network gives about 0.785 (0.776–0.793 over five seeds) against the printed 0.772, and 0.728 (0.643–0.791) against 0.783 on the held-out problems. Ability alone on the held-out problems reproduces 0.805 when ability uses only the learner's earlier submissions to those problems, and 0.836 with the learner's full history; both are in the output. Every comparison the thesis draws from these rows holds on every seed.
| Figure 6 | CW first-submission AUC by problems completed | 0.489 [0.331, 0.655] at the first problem; 0.746 after 1–8; 0.855 after 9–16 | difficulty only 0.635 | `cw_fig06_maturity_data` | `paper/figures/data/fig06_cw_maturity_auc.csv` |
| Table 10 | FA exact typed answer, outcome class from the model's own filter (n = 300) | 46.0% overall · 65.1 / 21.4 / 13.2 by stratum | best frontier 44.3% (published) | `infra/fa_sim3_full_run` | `results/fa_sim3_full_run/summary.json` |
| Table 10 | Same, true outcome class supplied (n = 300) | 57.3% · 78.1 / 30.5 / 34.0 | | `fa_sim3_eval_5e5f` | `results/fa_sim3_eval_5e5f/summary.json` |
| §7.2.1 | Sampled class agrees with the truth | 53.7% overall, 22.6% on wrong answers | | `fa_sim3_softmatch_conditional` | `results/fa_sim3_softmatch_conditional/` |
| §7.2.1 | Unaided stratum: outcome head vs item-difficulty table | 0.669 vs 0.703 | | `fa_sim3_kt_panel` | `results/fa_sim3_kt_panel/panel.json` |
| Table 11 | CW CodeBLEU, all 50 problems (4,587 rows) | 0.526 | KCGen-KT 0.580 / 0.557, TIKTOC adapted 0.507 (published; different protocol) | `infra/cw_ext_1_*`, `infra/cw_sim_launch2`, then `cw_codebleu_all50` (row-weighted mean of the three strata) | `results/cw_codebleu_all50/summary.json` |
| Table 11 | TIKTOC's own benchmark, zero-shot across cohorts (3,709 rows) | 0.561 | TIKTOC 0.554, OKT-TC 0.522 (published) | `infra/cw_sim_launch2_tiktoc` | `results/cw_sim_launch2_tiktoc/codebleu_panel.json` |
| Table 11 | Seven held-out problems (591 rows) · 13 headline problems | 0.479 · 0.606 | | `infra/cw_sim_launch2` | `results/cw_sim_launch2/{heldout,headline}/summary.json` |
| §7.2.2 | Adapter off · another learner's belief · decoding modes (pooled validation) | 0.564 → 0.407 · 0.563 · within 0.004 | | `infra/cw_sim_launch2` | `results/cw_sim_launch2/result.json` |
| §7.2.2 | Per-test-case AUC | 0.675 | TIKTOC 0.764, Code-DKT-TC 0.661 (published) | `infra/cw_sim_launch2_tiktoc` | `results/cw_sim_launch2_tiktoc/auc_f1_panel.json` |
| Table 12 | FA four arms (4,977 events): NLL of the true answer | 0.4492 · stranger 0.4660 · point 0.4512 · adapter off 3.3007 | costs 0.0168 · 0.0020 · 2.8515 | `infra/fa_sim3_full_run` | `results/fa_sim3_full_run/summary.json` |
| Figures 7–8 | Identity effect by history (+0.0105 → +0.0196); eight readouts | | | `infra/fa_sim3_full_run`, `fa_sim3_eval_5e5f` | `paper/figures/data/` |
| §7.3, Figure 9 | CW code vs the average of real classmates · vs the medoid (character similarity, 1,194 first attempts) | +0.078 [0.067, 0.089] · short by 0.015 [0.006, 0.025]; semantic lead 0.044 at a cold belief → 0.146 past 37 episodes | | `cw_classmate_ceiling` | `results/cw_classmate_ceiling/summary.json`, `paper/figures/data/fig09_cw_ceiling_bins.csv` |

## Chapter 8 · Acting on the belief

| Thesis | What | Value | Comparison | Script | Output |
|---|---|---|---|---|---|
| Table 13 | Outcome variance, person vs item | CW 23.6% vs 6.2%; FA 12.3% vs 18.1%; FA without give-ups 4.6% vs 18.7% | | `corpus_person_signal` | `results/corpus_person_signal/panel.json` |
| Table 14 | Correlation of a k-observation person estimate with remaining items, k = 3 | CW 0.660; FA 0.561; FA without give-ups 0.334 | | `corpus_person_signal` | as above |
| §8.2, Figure 10 | CW, start of course, budget 3: target-set log-loss | trained actor 0.924, greedy value of information 0.944 | random 1.072; difficulty-matched 1.028; maximum entropy 1.031 | `cw_planner_suite` | `results/cw_planner_suite_full_local/summary.json` |
| §8.2 | Actor vs random over 4 starts × 8 budgets | interval-clear better in 66 of 128 scenarios, never worse | | `cw_planner_suite` | as above |
| §8.2 | Actor on three underlying beliefs | about a third of scenarios clear vs random and max-entropy on the strongest belief | | `rl_2` | `results/rl_2/*/summary.json` |
| §8.2 | FA: greedy value of information vs random | see the note below | | | |
| §8.3, Figure 11 | Policy trained only inside the simulator, on real learners | 5 of 6 scenarios favour it (ratios 0.583–0.897); one reverses (1.130); none clears its bar | random = 1.0 | `sim2real_w_prepare_events`, `sim2real_w_run` | `results/sim2real_w/real_exam.json` |
| §8.4 | Zero-shot language model as selector | parses 42.5%; picks the first-listed candidate 90.8% (99.7% without history); loses to random in 36 of 36 | | `ga_1_tier0_run` | `results/ga_1_tier0/summary.json` |
| §8.4, Figure 12 | Fine-tuned on the simulator's pricing | beats random in 6 of 6 (5 clear); 25–74% of oracle headroom; clear of max-epistemic in 3 of 6 | | `ga_1_tier3_train`, `ga_1_tier3_eval` | `results/ga_1_tier3/eval/eval_summary.json` |
| §8.4 | Utility model, seen items · unseen items | 34.2% vs EPIG 21.3% of headroom · 21.5% vs 22.2% | | `ga_1_tier4_utility` | `results/ga_1_tier4_utility/summary.json` |
| §8.4, Figure 13 | Writing the question: on the real-problem manifold · verbatim copies · realised value | 27.2% → 95.6% · 29.7% · 0.5079 [0.4739, 0.5428] (random 0.50) | | `ga_1_tier4_rungc`, `ga_1_tier4_rungc_panel` | `results/ga_1_tier4_rungc_panel/summary.json` |

**Note on the FA planner number in §8.2.**
The thesis prints "worse than random by 0.0159 nats [0.0056, 0.0268]" for greedy value of information on FoundationalASSIST. That figure was computed with an earlier version of the replay harness, before the repairs listed in Appendix A.4. The harness in this repository is the repaired one. It reproduces the direction of the result, not its value:

| Belief the planner reads | Greedy's gain over random in log-loss (nats; negative = greedy worse) |
|---|---|
| The FA filter used during development (repaired harness; that filter is not part of this repository) | −0.0425 [−0.0533, −0.0323] |
| FA-SIM3, the shipped mathematics simulator (`pl_3_fa_sim3_binary`) | +0.0023 [−0.0055, +0.0104], a tie |

In every version, value-of-information questioning does not beat random questioning on FoundationalASSIST. That is the claim the thesis draws from it.

## Appendix A · Ablations (alternatives to a shipped component)

| What | Value | Script |
|---|---|---|
| Item encoder, nonlinear vs affine | +0.0073 AUC (FA); +0.0118 [0.0052, 0.0192] on unseen CW items | `fa_01i_item_encoder_ablation`, `cw_4g_codec_rank_item_mlp`, `cw_4g_3_item_holdout_check` |
| Codec width | 64 dimensions ship on both corpora | `p2_02c_sweep`, `cw_4g_codec_rank_item_mlp` |
| Belief width, FA (32 / 64 / 256 / 512 hidden units) | 0.652 / 0.669 / 0.678 / 0.680 | `fa_01g_s1v2_width_sweep` |
| Belief dimension, CW (64 vs 32) | +0.0060 [−0.0027, +0.0177] on unseen items | `cw_4i_latent_dim_isolation` |
| Filter body: causal event transformer | does not win; temporal bias carries about 96% of its value | `fa_02a_strata_v2`, `fa_02_bakeoff` |
| Width parameterisation | precision-accumulating on CW; free on FA | `infra/fa_sigma_a3`, `cw_sigma` |
| Ornstein–Uhlenbeck forgetting kernel | calibration 0.0183 → 0.0153, costs 0.061 AUC | `p3_04_ou_ablation` |
| Time and session features | −0.0029 nats | `fa_01l_time_features_screen` |
| Factored outcome head | +0.036 on detecting unaided wrong answers (bottom tercile); fails the uncertainty-coupling gate | `fc_1a_factored_heads` |
| Token-loss weight α (sweep and the α = 1.0 retrain) | rule selects 1.0; the thesis reports α = 0.3; the two are statistically indistinguishable | `infra/sim_alpha_sweep`, `sim_alpha_selection`, `wd_1_wrong_detection_panel` |
| Test-vector head refresh (shipped) | all-attempt AUC 0.744 → 0.802; first attempt 0.584 → 0.677 | `cw_sim_tv_head_refresh` |
| Null-prefix convention | 0.677 vs 0.677 | `cw_sim_tv_head_refresh` |

Three Appendix A entries are history and are not reproduced by this repository: the backbone read-out design ("where estimation lives"), the first simulator-trained policy (worse than random in 31 of 32 cells), and the first question-writing design (96% verbatim copies). They describe designs that were replaced, and their numbers stay in the thesis as the record of why.
