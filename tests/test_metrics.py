import math

import numpy as np

from atom.eval.metrics import (accuracy, auc, best_threshold_f1, bootstrap_mean_ci, brier,
                               ece, f1_score, learner_clustered_auc_ci,
                               learner_clustered_bootstrap_ci, multiclass_brier,
                               multiclass_ece, multiclass_nll, paired_bootstrap_ci)


def test_auc_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert auc(y, s) == 1.0


def test_best_threshold_f1_beats_or_matches_fixed_point_five():
    # All positives score just above 0.5, well-separated from negatives -- 0.5 already
    # gets perfect F1 here, so the calibrated search must find an equally-good point,
    # never a worse one (it is a search over the SAME population, so it cannot do worse
    # than the fixed threshold it is being compared against on any population where 0.5
    # is a valid candidate).
    y = [0, 0, 0, 1, 1, 1]
    p = [0.05, 0.1, 0.3, 0.55, 0.8, 0.95]
    t, best = best_threshold_f1(y, p)
    assert best >= f1_score(y, p, threshold=0.5)
    assert best == 1.0


def test_best_threshold_f1_finds_a_strictly_better_point_than_0_5():
    # Every true positive still ranks ABOVE every true negative (real separability),
    # but the whole distribution sits below 0.5 -- a fixed 0.5 threshold predicts no
    # positives at all (F1=0.0), while a threshold anchored at the lowest positive
    # score recovers perfect separation. This is exactly the case the fixed-threshold
    # F1=0.0 rows in `reports/cw_sim_tiktoc_transfer.md` motivate: F1=0.0 at 0.5 does
    # not mean the head carries no ranking signal, only that it is miscalibrated.
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.15, 0.2, 0.35, 0.4, 0.45]
    fixed = f1_score(y, p, threshold=0.5)
    t, best = best_threshold_f1(y, p)
    assert fixed == 0.0
    assert best > fixed
    assert best == 1.0
    assert t < 0.5


def test_best_threshold_f1_returns_none_for_single_class():
    t, best = best_threshold_f1([0, 0, 0], [0.1, 0.4, 0.9])
    assert t is None
    assert best is None


def test_learner_clustered_auc_ci_perfect_separation_is_tight_around_one():
    # 20 learners, each contributing one perfectly-separable row -- point AUC is exactly
    # 1.0, and every bootstrap resample (whole learners, with replacement) is ALSO
    # perfectly separable, so the CI should be degenerate at 1.0.
    y = [0, 1] * 10
    s = [0.1, 0.9] * 10
    learners = [f"l{i}" for i in range(20)]
    result = learner_clustered_auc_ci(y, s, learners, n_boot=200, seed=0)
    assert result["auc"] == 1.0
    assert result["ci95"][0] == 1.0 and result["ci95"][1] == 1.0
    assert result["n_boot_used"] == 200


def test_learner_clustered_auc_ci_rows_within_a_learner_move_together():
    # Two rows per learner sharing the SAME learner id -- if the clustering is real (not
    # secretly resampling rows), a resample can only ever include an intact (0,1) pair
    # per learner, never break one apart mid-resample, so the CI must stay exactly at
    # the point estimate (every possible resample gives the identical population up to
    # learner multiplicity, which does not change a POOLED AUC when classes are
    # perfectly separable at the row level within every learner).
    y = [0, 1, 0, 1, 0, 1]
    s = [0.2, 0.8, 0.1, 0.9, 0.3, 0.7]
    learners = ["a", "a", "b", "b", "c", "c"]
    result = learner_clustered_auc_ci(y, s, learners, n_boot=100, seed=1)
    assert result["auc"] == 1.0
    assert result["ci95"][0] == 1.0 and result["ci95"][1] == 1.0


def test_learner_clustered_auc_ci_reports_learner_count():
    y = [0, 0, 1, 1, 0, 1]
    s = [0.2, 0.3, 0.6, 0.8, 0.4, 0.7]
    learners = ["a", "a", "b", "c", "c", "d"]
    result = learner_clustered_auc_ci(y, s, learners, n_boot=50, seed=2)
    assert result["n_learners"] == 4


def test_auc_single_class_is_nan():
    assert math.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_accuracy():
    y = [0, 1, 1, 0]
    p = [0.2, 0.9, 0.6, 0.4]
    assert accuracy(y, p) == 1.0


def test_brier_known_value():
    y = [1, 0]
    p = [1.0, 0.0]
    assert brier(y, p) == 0.0
    assert math.isclose(brier([1, 0], [0.0, 1.0]), 1.0)


def test_ece_perfectly_calibrated_is_small():
    # 100 items at p=0.7 with 70% positives -> near-zero calibration error.
    rng = np.random.default_rng(0)
    p = np.full(1000, 0.7)
    y = (rng.random(1000) < 0.7).astype(float)
    assert ece(y, p, n_bins=10) < 0.05


def test_ece_miscalibrated_is_large():
    # Always predicts 0.9 but only 10% are positive.
    p = np.full(1000, 0.9)
    y = np.zeros(1000)
    y[:100] = 1.0
    assert ece(y, p, n_bins=10) > 0.5


def test_bootstrap_mean_ci_matches_sample_mean():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lo, hi = bootstrap_mean_ci(x, n_boot=2000, alpha=0.05, seed=0)
    assert mean == 3.0
    assert lo <= mean <= hi


def test_bootstrap_mean_ci_excludes_zero_for_a_clearly_positive_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=1.0, scale=0.1, size=200)  # clearly positive, small noise
    mean, lo, hi = bootstrap_mean_ci(x, n_boot=2000, alpha=0.05, seed=1)
    assert lo > 0.0


def test_bootstrap_mean_ci_straddles_zero_for_noise_centred_at_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=1.0, size=50)  # noisy, centred at 0, small n
    mean, lo, hi = bootstrap_mean_ci(x, n_boot=2000, alpha=0.05, seed=1)
    assert lo < 0.0 < hi


def test_bootstrap_mean_ci_reproducible_with_same_seed():
    x = np.random.default_rng(2).normal(size=100)
    a = bootstrap_mean_ci(x, n_boot=1000, alpha=0.05, seed=42)
    b = bootstrap_mean_ci(x, n_boot=1000, alpha=0.05, seed=42)
    assert a == b


def test_paired_bootstrap_ci_identical_arrays_centred_on_zero():
    x = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2])
    ci = paired_bootstrap_ci(x, x, n_boot=500, seed=0)
    assert abs(ci["point_delta"]) < 1e-9
    assert not ci["excludes_zero"]


def test_paired_bootstrap_ci_shifted_arrays_exclude_zero():
    x = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.4, 0.6])
    shifted = x + 1.0
    ci = paired_bootstrap_ci(shifted, x, n_boot=500, seed=0)
    assert abs(ci["point_delta"] - 1.0) < 1e-9
    assert ci["excludes_zero"]


def test_learner_clustered_bootstrap_ci_identical_arrays_centred_on_zero():
    learners = np.repeat(np.arange(20), 5)  # 20 learners x 5 rows each = 100 rows
    x = np.random.default_rng(0).normal(size=100)
    ci = learner_clustered_bootstrap_ci(x, x, learners, n_boot=500, seed=0)
    assert abs(ci["point_delta"]) < 1e-9
    assert not ci["excludes_zero"]
    assert ci["n_learners"] == 20
    assert ci["n_rows"] == 100


def test_learner_clustered_bootstrap_ci_shifted_arrays_exclude_zero():
    learners = np.repeat(np.arange(20), 5)
    x = np.random.default_rng(0).normal(size=100)
    shifted = x + 1.0
    ci = learner_clustered_bootstrap_ci(shifted, x, learners, n_boot=500, seed=0)
    assert abs(ci["point_delta"] - 1.0) < 1e-9
    assert ci["excludes_zero"]


def test_learner_clustered_bootstrap_ci_wider_than_row_level_under_within_learner_correlation():
    """The whole point of clustering: when rows within a learner share correlated
    error (here, a per-learner random offset added to every one of their rows), the
    row-level bootstrap understates uncertainty (treats correlated rows as independent
    evidence) while the learner-clustered bootstrap does not. This is the property that
    justifies 1f as the default, not just that the two give similar answers."""
    rng = np.random.default_rng(0)
    n_learners = 30
    rows_per_learner = 10
    learner_offset = rng.normal(scale=1.0, size=n_learners)  # correlated within-learner
    learners = np.repeat(np.arange(n_learners), rows_per_learner)
    noise = rng.normal(scale=0.1, size=n_learners * rows_per_learner)  # small iid noise
    a = learner_offset[learners] + noise
    b = np.zeros_like(a)

    row_ci = paired_bootstrap_ci(a, b, n_boot=1000, seed=1)
    cluster_ci = learner_clustered_bootstrap_ci(a, b, learners, n_boot=1000, seed=1)
    row_width = row_ci["ci95"][1] - row_ci["ci95"][0]
    cluster_width = cluster_ci["ci95"][1] - cluster_ci["ci95"][0]
    assert cluster_width > row_width


def test_learner_clustered_bootstrap_ci_reproducible_with_same_seed():
    learners = np.repeat(np.arange(10), 4)
    x = np.random.default_rng(3).normal(size=40)
    a = learner_clustered_bootstrap_ci(x, x * 0, learners, n_boot=300, seed=7)
    b = learner_clustered_bootstrap_ci(x, x * 0, learners, n_boot=300, seed=7)
    assert a == b


def test_multiclass_nll_perfect_prediction_is_near_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = [0, 1]
    assert multiclass_nll(probs, target) < 1e-6


def test_multiclass_nll_uniform_prediction_matches_log_c():
    probs = np.full((5, 4), 0.25)
    target = [0, 1, 2, 3, 0]
    assert math.isclose(multiclass_nll(probs, target), math.log(4), rel_tol=1e-6)


def test_multiclass_nll_clips_zero_probability_instead_of_diverging():
    probs = np.array([[1.0, 0.0]])
    target = [1]  # predicted probability 0 for the true class
    assert math.isfinite(multiclass_nll(probs, target))


def test_multiclass_brier_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = [0, 2]
    assert multiclass_brier(probs, target) == 0.0


def test_multiclass_brier_matches_hand_computed_value():
    # one row, predicted [0.5, 0.5], true class 0 -> (0.5-1)^2 + (0.5-0)^2 = 0.5
    probs = np.array([[0.5, 0.5]])
    target = [0]
    assert math.isclose(multiclass_brier(probs, target), 0.5, rel_tol=1e-9)


def test_multiclass_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    n = 1000
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.7
    probs[:, 1] = 0.3
    # argmax is always class 0 at confidence 0.7; make it actually right 70% of the time.
    target = np.where(rng.random(n) < 0.7, 0, 1)
    assert multiclass_ece(probs, target, n_bins=10) < 0.05


def test_multiclass_ece_miscalibrated_is_large():
    # always 90% confident in class 0, but only 10% of the time class 0 is right
    n = 1000
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    target = np.ones(n, dtype=int)
    target[:100] = 0
    assert multiclass_ece(probs, target, n_bins=10) > 0.5


def test_multiclass_ece_empty_is_nan():
    assert math.isnan(multiclass_ece(np.zeros((0, 3)), []))
