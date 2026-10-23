# Reproducing the thesis

Each thesis number, the command that produces it, and what has to run first. [`RESULTS.md`](RESULTS.md) lists the numbers. This file gives the commands in the order you need to run them.

**Conventions.**
- Run every command from the repository root inside the pinned environment (`uv run python …`, or activate `.venv`). Commands are shown as `python …`.
- Paths are relative to the repository root. The corpora sit next to it in `../Datasets/` (see [`DATA.md`](DATA.md)).
- **Hardware.** *Laptop CPU* means a few minutes to an hour on a laptop. *Apple-silicon GPU* means the run used PyTorch's MPS backend (a CUDA GPU also works). *One 24 GB GPU* and *one 48 GB GPU* give the GPU memory the thesis run had. Runtimes are wall-clock times recorded on that hardware, where they were recorded.
- **Arguments.** Rows marked *(arguments not recorded; defaults shown)* show the best-supported command. The script's defaults are the shipped configuration, but the exact invocation behind the thesis number was not logged.

## 0. Setup

```bash
make setup                      # pinned Python 3.11 environment from uv.lock
make weights                    # every published checkpoint, written to the path the code expects
make test && make smoke         # unit tests, and an end-to-end run on synthetic data
huggingface-cli login           # accept the Gemma Terms of Use first
python scripts/p2_01_backbone.py --allow-download    # caches Gemma 4 E4B and checks hidden-state access
```

Place the corpora as described in [`DATA.md`](DATA.md), then:

```bash
make data-fa      # scripts/fa_00_manifest_rebuild.py -> artifacts/fa_manifest_v2.parquet, splits checked
make data-cw      # cw_1_event_graph, cw_2_sessions, cw_4c_step0_1_splits; splits checked
python scripts/p1_05_data_report.py   # optional: corpus statistics (laptop CPU)
```

**Published checkpoints make training optional.** `make weights` rebuilds FA-SIM3, the CodeWorkout classifier, the CodeWorkout generative simulator, its refreshed outcome and test-vector heads, the actors, the LLM selector and the utility models. Every training row below can therefore be skipped. The one exception is FA-SIM3's evaluation. It runs inside `infra/fa_sim3_full_run.py`, and the per-row outputs it writes are not published, so Tables 8, 10 and 12 need that run (see §2).

**Cloud-style trainers.** `infra/fa_sim3_full_run.py`, `infra/sim_alpha_sweep.py`, `infra/cw_03_encode.py` and `infra/p3_05b_akt_full_train.py` require `--bucket <your-bucket>`: an S3 bucket you own, which the script uploads its checkpoints and summaries to. They need AWS credentials and the `aws` CLI, and a failed upload stops the run. Every output is also written locally; each row says where. `infra/cw_sim_full_run.py`, `infra/cw_sim_launch2*.py` and `infra/cw_ext_1_generate.py` accept `--bucket` but do not need it.

> **These scripts schedule a machine shutdown.** `infra/fa_sim3_full_run.py`, `infra/sim_alpha_sweep.py`, `infra/cw_03_encode*.py`, `infra/cw_sim_full_run.py`, `infra/cw_sim_launch2.py`, `infra/cw_sim_launch2_tiktoc.py` and `infra/cw_ext_1_generate.py` always call `sudo shutdown -h +<--fuse-minutes>` when they start, as a guard against runaway cloud jobs. `scripts/cw_sigma.py` and `infra/fa_sigma_a3.py` do so only when `--bucket` is given. Run these scripts on a dedicated GPU machine, or cancel the shutdown with `sudo shutdown -c`. Leave out `--self-terminate`, `--rearm-fuse` and `--stop-on-complete` unless you want the machine to stop.

**Reproducibility caveats.**
- Free-text generation (Tables 10 and 11) reproduces to within about one point across GPU types, not bit for bit. All other numbers are deterministic given the seeds in the scripts.
- CodeBLEU (Table 11) needs Linux x86-64, because TIKTOC's tree-sitter parser is a Linux binary. `infra/cw_ext_1_generate.py` also needs a separate Python 3.11 environment with `tree_sitter==0.20.4`, `numpy` and `pandas` (pass its interpreter with `--codebleu-python`).
- Building the test-vector labels needs a JDK (`javac` and `java` on `PATH`).

The CodeWorkout scripts take the same six feature inputs. Set them once:

```bash
RAW="--raw-dir ../Datasets/CodeWorkout_CMU"
CW="--code-map cache/cw_03_vectors/cw_code_state_hash_map_full.npz \
    --diff-map cache/cw_03_vectors/cw_diff_hash_map_full.npz \
    --codealone-meta cache/cw_03_vectors/codealone/meta.json \
    --codealone-states cache/cw_03_vectors/codealone/states_mean.npy \
    --diff-meta cache/cw_03_vectors/diff/meta.json \
    --diff-states cache/cw_03_vectors/diff/states_mean.npy"
```

## 1. Shared preprocessing, in dependency order

The frozen codec transform (`transforms/problem_mean_L15.npz`), the derived item vectors (`artifacts/problem_codec_mean_L15.npz`, `artifacts/fa_item_codec_dim64.npz`, `artifacts/cw_problem_codec_mean_L15.npz`, `artifacts/cw_item_codec_dim64.npz`) are published, one vector per problem; step 1a only rebuilds them. Steps 1b–1f are required, because their outputs contain or are keyed to licensed data. Step 1f's belief handoffs hold one belief per learner and episode, so they are not published.

| Step | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| 1a FA codec (optional rebuild) | `python scripts/p2_02a_extract_calibration.py --n 1000` → `python scripts/p2_02c_sweep.py --n 1200` → `python scripts/p2_07_encode_problems.py --k-max 300 --svd-solver full --production-rank 50` | Apple-silicon GPU | | `cache/fa_problems/`, `transforms/problem_mean_L15.npz`, `artifacts/problem_codec_mean_L15.npz` |
| 1b CodingBat statements | `python scripts/fetch_codingbat_statements.py` → `python scripts/cw_4b_recover_problem_statements.py` | laptop CPU, then Apple-silicon GPU | minutes | `artifacts/cw_problem_statements.json`, `artifacts/cw_problem_codec_mean_L15.npz` |
| 1c CodeWorkout code and diff encodings (every `$CW` input) | `python infra/cw_03_encode.py --bucket <your-bucket> --raw-dir ../Datasets/CodeWorkout_CMU --fuse-minutes 240 --batch-size 32`, then the copy below | one 48 GB GPU | about 2.3 h | `cache/cw_03_vectors/` |
| 1d TIKTOC code encodings (Table 11, TIKTOC benchmark) | `python infra/cw_03_encode_tiktoc.py --tiktoc-dir ../Datasets/tiktoc-main/data --out-dir cache/cw_03_vectors_tiktoc --fuse-minutes 60` | one GPU | minutes | `cache/cw_03_vectors_tiktoc/` |
| 1e Test-vector labels | `python scripts/cw_sim_test_vector_labels.py` (needs a JDK) *(arguments not recorded; defaults shown; the default cap of 4,000 executed attempts matches the thesis file)* | CPU, multi-core | | `artifacts/cw_sim_test_vector_labels.npz` |
| 1f Belief handoffs (CodeWorkout planner rows) | `python scripts/cw_classifier_belief_handoff.py $RAW $CW --out artifacts/cw_ratchet_val_prefix_beliefs.npz` · `python scripts/cw_sim_belief_handoff.py $RAW $CW --out artifacts/cw_sim_val_prefix_beliefs.npz` · `python scripts/cw_sim_belief_handoff.py $RAW $CW --checkpoint results/cw_sim_head_refresh/final.pt --out artifacts/cw_sim_refreshed_head_val_prefix_beliefs.npz` · `python scripts/cw_sim_train_handoff.py $RAW $CW --out artifacts/cw_sim_refreshed_head_train_prefix_beliefs.npz` → `python scripts/finalize_belief_handoffs.py` | laptop CPU | minutes each | the four handoffs; each script checks its checkpoint's hash, and the last step checks each handoff against the hash of the file the thesis used, which the planner scripts require |

`infra/cw_03_encode.py` writes the vector caches under `--cache-dir` (default `/tmp/cw_03_cache`) and the two hash maps under `/tmp/cw_03_artifacts/`. Move them into the layout the scripts read:

```bash
mkdir -p cache/cw_03_vectors
cp -r /tmp/cw_03_cache/full/codealone /tmp/cw_03_cache/full/diff cache/cw_03_vectors/
cp /tmp/cw_03_artifacts/cw_code_state_hash_map_full.npz /tmp/cw_03_artifacts/cw_diff_hash_map_full.npz cache/cw_03_vectors/
```

**Trained models (Chapter 5; optional, `make weights` replaces them).**

| Model | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| FA outcome filter (FA-SIM3's warm start) | `python scripts/fa_01k_b4_close.py` | Apple-silicon GPU | | `artifacts/checkpoints/fa_outcome_filter.pt` |
| FA-SIM3 staging bundles | `python scripts/fa_04b_stage_bundle.py --frac 0.20 --n-learners 0 --name fa_06_stage2_20pct --report-path reports/fa_06_stage2_20pct_bundle.md` · `python scripts/fa_04b_stage_bundle.py --split val --frac 0.02 --n-learners 0 --seed 0 --name fa_06_val_eval --report-path reports/fa_06_val_eval_bundle.md` · `python scripts/fa_sim3_stage_bundles.py` | laptop CPU | minutes | `staging/bundles/*.tar.gz` (contain licensed text; keep them local) |
| FA-SIM3 smoke test | `python scripts/fa_sim3_smoke_local.py` | laptop CPU | minutes | stub-model check of the full pipeline |
| FA-SIM3 (α = 0.3), training and evaluation | `python infra/fa_sim3_full_run.py --bucket <your-bucket> --fuse-minutes 1560 --generations-local-dir results/fa_sim3_full_run`, then the copy below | one 48 GB GPU | about 19 h (training about 10 h) | `results/fa_sim3_full_run/` |
| CodeWorkout classifier | `python scripts/cw_sigma.py $RAW $CW --likelihood kmixture --sigma-parameterization precision_accumulating --out-dir results/cw_sigma_precision_0305a22/aws/results` | one 24 GB GPU | about 4 min | `…/final.pt` (the path every CodeWorkout script defaults to) |
| CodeWorkout generative simulator | `python infra/cw_sim_full_run.py $RAW $CW --filter-mode joint --max-steps 342 --micro-batch-size 2 --train-row-chunk-size 1 --full-eval --estimated-minutes 720 --fuse-minutes 2400 --out-dir results/cw_sim_base_run` | one 48 GB GPU | about 31 h (20.9 h training, 8.4 h evaluation) | `results/cw_sim_base_run/final.pt` |
| Refreshed outcome head | `python scripts/cw_sim_head_refresh.py $RAW $CW` | laptop CPU | about 2 min | `results/cw_sim_head_refresh/final.pt` |
| Refreshed test-vector head | `python scripts/cw_sim_tv_head_refresh.py $RAW $CW --out-dir results/cw_sim_tv_head_refresh` | laptop CPU | about 6 min | `results/cw_sim_tv_head_refresh/final.pt`, `final_null_prefix.pt` |

`infra/fa_sim3_full_run.py` also reads `staging/bundles/fa_sim1_val_corrected.tar.gz`, the population behind one comparison court. `make weights` writes FA-SIM3 to `artifacts/checkpoints/fa_sim3_final.pt`. After a training run, put its local outputs where the evaluation scripts read them:

```bash
cp /tmp/fa_sim3_predictions.csv.gz      results/fa_sim3_full_run/predictions.csv.gz
cp /tmp/fa_sim3_belief_diagnostics.npz  results/fa_sim3_full_run/belief_diagnostics.npz
cp /tmp/fa_sim3_summary.json            results/fa_sim3_full_run/summary.json
mkdir -p artifacts/checkpoints && cp /tmp/fa_sim3_final.pt artifacts/checkpoints/fa_sim3_final.pt
```

## 2. Chapter 6: meaning in place of a curriculum

| Thesis item | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| Table 6, same curriculum skill (0.819) | `python scripts/p2_02c_sweep.py --n 1200` (step 1a; needs `cache/fa_problems/` from `p2_02a_extract_calibration.py`) | Apple-silicon GPU | | `reports/p2_02c_sweep.md` |
| Table 6, which problem / who wrote it / same tests | `python scripts/p2_05b_probe_battery.py` (encodes the TIKTOC submissions itself) | Apple-silicon GPU | | `reports/p2_05b_probe_battery.md` |
| Table 7, harder / more discriminating problem | `python scripts/fa_pg_probe.py` | laptop CPU | minutes | `results/fa_pg_probe/probe.json` |
| §6.2, AKT with and without skill tags | `python scripts/fa_01_akt_v2.py` · `python scripts/fa_01b_akt_no_tags.py` | Apple-silicon GPU | | `reports/fa_01_akt_v2.md`, `reports/fa_01b_akt_no_tags.md` |
| §6.2, ATOM's filter on the same court (0.6913) | `python scripts/fa_01i_item_encoder_ablation.py` *(arguments not recorded; defaults shown: the full grid)* | Apple-silicon GPU | | `reports/fa_01i_item_encoder_ablation.md` |

`python scripts/fa_01_majority_irt_v2.py` gives the majority and IRT reference rows on the same split (laptop CPU).

## 3. Chapter 7: simulating a learner

All FA rows read `results/fa_sim3_full_run/` (§1). Table 11 reads the generative simulator and its refreshed outcome head.

| Thesis item | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| Table 8, FA correctness; §7.1.1, within-problem AUC; unaided stratum | `python scripts/fa_sim3_kt_panel.py` | laptop CPU | minutes | `results/fa_sim3_kt_panel/panel.json` |
| Table 8, AKT and IRT on the same rows; §7.1.1, AKT's paired lead | `python scripts/fa_01p_court_baselines.py` → `python scripts/fa_01p_court_paired.py` | Apple-silicon GPU (AKT retrain) | about 18 min, then seconds | `results/fa_court_baselines/{panel,paired}.json` |
| §7.1.1, belief width vs error (0.484); Table 10, own-filter row; Table 12, four arms | printed by the FA-SIM3 run (§1) | one 48 GB GPU | (part of the 19 h) | `results/fa_sim3_full_run/summary.json` |
| Figure 5 | `python paper/figures/fig05_data.py` | laptop CPU | minutes | `paper/figures/data/fig05_fa_auc_by_history.csv` |
| Table 9, §7.1.2, CW first-submission correctness and the seven held-out problems | `python scripts/cw_first_submission_baselines.py` (`make eval-cw-classifier`) | laptop CPU | | `results/cw_first_submission_baselines/summary.json` |
| Figure 6 | `python scripts/cw_fig06_maturity_data.py` | laptop CPU | | `paper/figures/data/fig06_cw_maturity_auc.csv` |
| Table 10, true outcome class supplied | `python scripts/fa_sim3_eval_5e5f.py` | Apple-silicon GPU | about 16 min | `results/fa_sim3_eval_5e5f/summary.json`, `sim_scores.csv.gz` |
| §7.2.1, sampled class agrees with truth | `python scripts/fa_sim3_softmatch_conditional.py` (reads the previous row's `sim_scores.csv.gz`) | laptop CPU | seconds | `results/fa_sim3_softmatch_conditional/summary.json` |
| Table 11, 30 further problems (2,802 rows) | `python infra/cw_ext_1_row_selection.py --out results/cw_ext_1/row_selection.json` → `python infra/cw_ext_1_generate.py --out-dir results/cw_ext_1 --codebleu-python <py3.11-env>/bin/python3` | laptop CPU, then one 24 GB GPU on Linux | about 2.6 h | `results/cw_ext_1/result.json` |
| Table 11, 13 headline and 7 held-out problems; §7.2.2, adapter off, another learner's belief, decoding modes | `python infra/cw_sim_launch2.py $RAW $CW --test-vector-labels artifacts/cw_sim_test_vector_labels.npz` *(arguments not recorded; defaults shown)* | one 48 GB GPU, Linux | | `results/cw_sim_launch2/{result.json,headline/,heldout/}` |
| Table 11, all 50 problems (0.526, 4,587 rows) | `python scripts/cw_codebleu_all50.py` (`make codebleu-all50`): the row-weighted mean of the three strata above | laptop CPU | seconds | `results/cw_codebleu_all50/summary.json` |
| Table 11, TIKTOC's own benchmark; §7.2.2, per-test-case AUC | `python infra/cw_sim_launch2_tiktoc.py --tiktoc-dir ../Datasets/tiktoc-main/data` *(arguments not recorded; defaults shown)* | one 48 GB GPU, Linux | | `results/cw_sim_launch2_tiktoc/{codebleu_panel,auc_f1_panel}.json` |
| Figures 7–8 | from the FA-SIM3 run and `fa_sim3_eval_5e5f`; redraw with `make figures` | laptop CPU | seconds | `paper/figures/` |
| §7.3, Figure 9, classmate and medoid references | `python scripts/cw_classmate_ceiling.py --real-codec-vectors <npz>` (reads `results/cw_sim_launch2/headline/generations_raw.jsonl`) | laptop CPU | minutes | `results/cw_classmate_ceiling/summary.json`, `paper/figures/data/fig09_cw_ceiling_bins.csv` |

`--real-codec-vectors` holds the codec coordinates of each row's real submission. Without it, the character-similarity references (+0.078 and the medoid gap) are still computed, but the semantic panel stays empty.

## 4. Chapter 8: acting on the belief

The CodeWorkout planner rows use the classifier (`make weights`) and the belief handoffs from step 1f.

| Thesis item | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| Tables 13–14, outcome variance and k-observation correlation | `python scripts/corpus_person_signal.py` | laptop CPU | about 2 min | `results/corpus_person_signal/panel.json` |
| §8.2, Figure 10, CW planner grid (actor 0.924, greedy 0.944, 66 of 128) | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/cw_planner_suite.py $CW $RAW --checkpoint results/cw_sigma_precision_0305a22/aws/results/final.pt --beliefs artifacts/cw_ratchet_val_prefix_beliefs.npz --beliefs-sha256 4375b0c93d7610fef94dc21d2db657a95aaeb5dc663d102f30fa5e7030a4ecde --out-dir results/cw_planner_suite_full_local --device cpu --train-learners 686 --val-learners 110 --epochs 8` | laptop CPU | | `results/cw_planner_suite_full_local/summary.json` |
| §8.2, actor on three underlying beliefs | `python scripts/rl_2.py $RAW $CW --substrates ratchet-original ratchet-cwsim/cold-head ratchet-cwsim/refreshed-head --n-seeds 3 --train-learners 686 --val-learners 110 --device cuda --out-dir results/rl_2 --resume` (the thesis ran one belief per GPU) | one 24 GB GPU per belief | several hours per belief | `results/rl_2/<belief>/summary.json` |
| §8.2, FA greedy value of information vs random | `python scripts/pl_3_fa_sim3_binary.py` (on FA-SIM3, the shipped simulator; see the note in RESULTS.md on how this relates to the number printed in the thesis) | laptop CPU · one GPU | | `reports/pl_3_evsi_proper.md` · `results/pl_3_fa_sim3_binary/`; see the note in `RESULTS.md` |
| §8.3, Figure 11, simulator-trained policy on real learners | `python scripts/sim2real_w.py --out artifacts/sim2real_w_structural_audit.json` → `python scripts/sim2real_w_prepare_events.py $RAW $CW --checkpoint results/cw_sim_head_refresh/final.pt` → `python scripts/sim2real_w_run.py $CW --device cuda --batch-size 16 --precomputed-events artifacts/sim2real_w_events.npz --structural-audit artifacts/sim2real_w_structural_audit.json --handoff artifacts/cw_sim_refreshed_head_train_prefix_beliefs.npz --checkpoint results/cw_sim_head_refresh/final.pt --base-checkpoint results/cw_sim_base_run/final.pt --out-dir results/sim2real_w` | laptop CPU ×2, then one 48 GB GPU | about 4 h of decoding, plus training | `results/sim2real_w/real_exam.json` |
| §8.4, zero-shot language model as selector | `python scripts/ga_1_tier0_run.py $RAW $CW --device cuda --batch-size 16 --resume --rl2-dir ratchet-original=results/rl_2/ratchet-original --rl2-dir ratchet-cwsim/cold-head=results/rl_2/ratchet-cwsim_cold-head --rl2-dir ratchet-cwsim/refreshed-head=results/rl_2/ratchet-cwsim_refreshed-head --out-dir results/ga_1_tier0` | one 24 GB GPU | about 6 h | `results/ga_1_tier0/summary.json` |
| §8.4, Figure 12, selector fine-tuned on the simulator's pricing | `python scripts/ga_1_tier3_train.py $RAW $CW --device cuda` → `python scripts/ga_1_tier3_eval.py $RAW $CW --device cuda --adapter results/ga_1_tier3/adapter_best.pt --out-dir results/ga_1_tier3/eval` *(arguments not recorded; defaults shown)* | one 48 GB GPU | evaluation about 3.3 h | `results/ga_1_tier3/eval/eval_summary.json` |
| §8.4, utility model, seen and unseen items | `python scripts/ga_1_tier4_utility.py --raw-dir ../Datasets/CodeWorkout_CMU` | laptop CPU | about 4 min | `results/ga_1_tier4_utility/summary.json` |
| §8.4, Figure 13, writing the question | `python scripts/ga_1_tier4_rungc.py --raw-dir ../Datasets/CodeWorkout_CMU --device cuda` → `python scripts/ga_1_tier4_rungc_panel.py` *(arguments not recorded; defaults shown)* | one 24 GB GPU | about 5.0 h, then 1.3 h | `results/ga_1_tier4_rungc_panel/summary.json` |

The question-writing adapter is not published (see [`MODEL_CARD.md`](MODEL_CARD.md)), so the Figure 13 panel needs the `ga_1_tier4_rungc` run first. `make weights` provides the tier-3 adapter, so `ga_1_tier3_eval` can run without retraining.

## 5. Appendix A: ablations

| Thesis item | Command | Hardware | Runtime | Output |
|---|---|---|---|---|
| Item encoder, nonlinear vs affine (FA) | `python scripts/fa_01i_item_encoder_ablation.py` *(arguments not recorded; defaults shown)* | Apple-silicon GPU | | `reports/fa_01i_item_encoder_ablation.md` |
| Item encoder (CW); codec width 64 | `python scripts/cw_4g_codec_rank_item_mlp.py $RAW $CW` · `python scripts/cw_4g_3_item_holdout_check.py $RAW $CW` | laptop CPU | | `reports/cw_4g_codec_rank_item_mlp.md`, `reports/cw_4g_3_item_holdout_check.md` |
| Codec width (FA) | `python scripts/p2_02c_sweep.py --n 1200` | Apple-silicon GPU | | `reports/p2_02c_sweep.md` |
| Belief width, FA (32 / 64 / 256 / 512 hidden units) | `python scripts/fa_01g_s1v2_width_sweep.py` | Apple-silicon GPU | | `reports/fa_01g_s1v2_width_sweep.md` |
| Belief dimension, CW (64 vs 32) | `python scripts/cw_4i_latent_dim_isolation.py $RAW $CW` | laptop CPU | | `reports/cw_4i_latent_dim_isolation.md` |
| Filter body: causal event transformer | `python scripts/fa_02a_strata_v2.py` → `python scripts/fa_02_bakeoff.py` → `python scripts/fa_02_bakeoff.py --arms gru,c_distance_time,c_time --warmup 0.1 --tag _fa2b_warmup` | Apple-silicon GPU | | `reports/fa_02_body_c_screen.md`, `reports/fa_02_body_c_screen_fa2b_warmup.md` |
| Width parameterisation, FA | `python infra/fa_sigma_a3.py --checkpoint-path artifacts/checkpoints/fa_sim3_final.pt --sigma-param precision --free-bits-mu 0.05 --identity-sigma-seed 0.011119465831742543 --identity-reference-share 0.6429` | one 24 GB GPU | about 97 min | `results/fa_sigma_a3/` |
| Width parameterisation, CW | `python scripts/cw_sigma.py $RAW $CW --likelihood kmixture --sigma-parameterization direct` (the precision-accumulating arm is the classifier run in §1) | one 24 GB GPU | minutes | `results/cw_sigma/summary.json` |
| Ornstein–Uhlenbeck forgetting kernel | `python scripts/p3_04_ou_ablation.py` | Apple-silicon GPU | | `reports/p3_04_ou_ablation.md` |
| Time and session features | `python scripts/fa_01l_time_features_screen.py` | Apple-silicon GPU | | `reports/fa_01l_time_features_screen.md` |
| Factored outcome head | `python scripts/fc_1a_factored_heads.py` (the FA-SIM3 arm; its other comparison arms read development checkpoints that are not published) | Apple-silicon GPU | | `results/fc_1a/` |
| Token-loss weight α, sweep | `python infra/sim_alpha_sweep.py --alpha A --max-steps 2500 --n-personalisation-eval-rows 300 --fuse-minutes 90 --bucket <your-bucket>` for A in 0.3, 0.75, 1.0, 1.5, 2.0 | one 48 GB GPU | about 2 h per value (10.1 h for all five) | `results/sim_alpha_sweep/alpha_<A>/` |
| Token-loss weight α, selection and the α = 1.0 retrain | `python scripts/sim_alpha_selection.py` · `python infra/fa_sim3_full_run.py --token-loss-weight 1.0 --fuse-minutes 1400 --bucket <your-bucket> --out-prefix sim-alpha-round2-full` · `python scripts/wd_1_wrong_detection_panel.py` (FA-SIM3 and the α = 1.0 retrain; other columns read development checkpoints that are not published) | laptop CPU · one 48 GB GPU · laptop CPU | seconds · about 18.4 h · about 2 min | selection report, `results/wd_1/summary.json` |
| Test-vector head refresh; null-prefix convention | `python scripts/cw_sim_tv_head_refresh.py $RAW $CW --out-dir results/cw_sim_tv_head_refresh` (§1) | laptop CPU | about 6 min | `results/cw_sim_tv_head_refresh/manifest.json` |

**Appendix A.4 (replay harness).** The harness repairs, the corrected estimator and the planner's positive control are `python scripts/pl_0_replay_repair.py`, `python scripts/pl_1_epig_estimator.py` and `python scripts/pl_2_positive_control.py` (laptop CPU, no arguments).

The α = 1.0 retrain writes its local outputs to the same `/tmp/fa_sim3_*` files as the α = 0.3 run. Run them in separate working copies, or move the first run's outputs before starting the second.

As `RESULTS.md` notes, three Appendix A entries describe replaced designs and are not reproduced here.
