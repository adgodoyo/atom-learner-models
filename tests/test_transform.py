"""Unit tests for the codec transform (synthetic data with known spectrum)."""

import numpy as np
import pytest

from atom.codec.transform import CodecTransform


def _spectral_data(n=4000, d=20, seed=0):
    """Data whose covariance has 5 dominant directions then a flat tail (~1)."""
    rng = np.random.default_rng(seed)
    variances = np.array([100.0, 50.0, 20.0, 8.0, 8.0] + [1.0] * (d - 5))
    z = rng.standard_normal((n, d)) * np.sqrt(variances)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))  # random orthonormal rotation
    return z @ q.T, variances


def test_eigvals_recovered_and_descending():
    x, variances = _spectral_data()
    ct = CodecTransform.fit(x, k_max=20)
    assert np.all(np.diff(ct.eigvals) <= 1e-6)  # descending
    # top-5 eigenvalues should track the planted dominant variances (loose tolerance).
    assert np.allclose(ct.eigvals[:5], variances[:5], rtol=0.15)


def test_whitening_gives_unit_variance():
    x, _ = _spectral_data()
    ct = CodecTransform.fit(x, k_max=20)
    coords = ct.apply(x, k=0, whiten=True)
    v = coords.var(axis=0)
    assert np.allclose(v, 1.0, atol=0.1)  # every whitened direction ~ unit variance


def test_deflation_removes_dominant_variance():
    x, _ = _spectral_data()
    ct = CodecTransform.fit(x, k_max=20)
    full = ct.apply(x, k=0, whiten=False).var(axis=0).sum()
    deflated = ct.apply(x, k=5, whiten=False).var(axis=0).sum()
    # dropping the top-5 (which hold ~180 of ~195 total variance) leaves < 20%.
    assert deflated < 0.2 * full


def test_apply_shape_and_validation():
    x, _ = _spectral_data()
    ct = CodecTransform.fit(x, k_max=20)
    assert ct.apply(x, k=5, rank=15).shape == (x.shape[0], 10)
    with pytest.raises(ValueError):
        ct.apply(x, k=15, rank=10)


def test_save_load_roundtrip(tmp_path):
    x, _ = _spectral_data()
    ct = CodecTransform.fit(x, k_max=20, layer=12, pooling="mean", source="unit-test")
    out = ct.save(tmp_path / "codec.npz")
    ct2 = CodecTransform.load(out)
    assert ct2.layer == 12 and ct2.pooling == "mean" and ct2.version == ct.version
    assert ct2.hash == ct.hash
    assert np.allclose(ct.apply(x, k=5), ct2.apply(x, k=5))


def test_default_svd_solver_is_randomized_and_unchanged():
    """CW-4g: adding `svd_solver` must not perturb any pre-existing caller's behaviour."""
    x, _ = _spectral_data()
    ct_default = CodecTransform.fit(x, k_max=10)
    ct_explicit = CodecTransform.fit(x, k_max=10, svd_solver="randomized")
    assert np.array_equal(ct_default.eigvecs, ct_explicit.eigvecs)
    assert np.array_equal(ct_default.eigvals, ct_explicit.eigvals)


def test_full_svd_solver_is_exactly_self_nested():
    """CW-4g's nestedness gate, in miniature: a wider full-SVD fit's first k_max columns
    must be IDENTICAL (not merely close) to a narrower fit's -- both are prefixes of the
    same canonical decomposition."""
    x, _ = _spectral_data(n=500, d=20)
    ct_narrow = CodecTransform.fit(x, k_max=8, svd_solver="full")
    ct_wide = CodecTransform.fit(x, k_max=20, svd_solver="full")
    assert np.allclose(ct_narrow.eigvecs, ct_wide.eigvecs[:, :8], atol=1e-10)
    assert np.allclose(ct_narrow.eigvals, ct_wide.eigvals[:8], atol=1e-10)
    assert np.allclose(ct_narrow.apply(x, k=2, rank=8, whiten=False),
                       ct_wide.apply(x, k=2, rank=8, whiten=False), atol=1e-10)


def test_unknown_svd_solver_raises():
    x, _ = _spectral_data()
    with pytest.raises(ValueError):
        CodecTransform.fit(x, k_max=10, svd_solver="bogus")
