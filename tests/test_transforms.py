"""Le trasformazioni ALR/softmax devono essere invertibili e dare quote che
sommano a 1."""
import numpy as np

from consenso.model.transforms import alr_inv_np, alr_np


def test_alr_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        eta = alr_np(p, ref_idx=4)
        back = alr_inv_np(eta, ref_idx=4)
        assert np.allclose(p, back, atol=1e-9)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(7, 4))            # batch di 7, K-1=4 -> K=5
    p = alr_inv_np(eta, ref_idx=2)
    assert p.shape == (7, 5)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert (p > 0).all()


def test_reference_position_invariance():
    p = np.array([0.4, 0.1, 0.25, 0.25])
    for ref in range(4):
        eta = alr_np(p, ref_idx=ref)
        assert np.allclose(alr_inv_np(eta, ref_idx=ref), p, atol=1e-9)
