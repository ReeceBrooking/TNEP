"""Tests for gap_descriptor.py (gap_descriptor_plan.md Task 1): TurboGAP 2-body
and 3-body distance descriptors with fixed centres, exact against direct NumPy
transcriptions of the gapf90 gap.f90 energy loops (gapf90_gap.f90.ref).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_neighbor_dipole import _cfg as _base_cfg, _bulk_frame
from data import assign_type_indices
import gap_descriptor as gd

RCUT, BUFFER = 3.5, 1.0     # buffer 1.0 so the taper is exercised
SIGMA2, SIGMA3 = 0.3, [0.5, 0.4, 0.6]


def _cfg(**kw):
    return _base_cfg(**{"gap2b_rcut": RCUT, "gap3b_rcut": RCUT, "gap_buffer": BUFFER,
                        "gap2b_n_centres": 7, "gap2b_sigma": SIGMA2,
                        "gap3b_n_centres": 9, "gap3b_sigma": list(SIGMA3), **kw})


def _frame():
    at = _bulk_frame()
    return at, assign_type_indices([at], [8, 1])[0]


def _random_centres(rng, cfg):
    cfg.gap2b_centres = rng.uniform(0.5, RCUT, cfg.gap2b_n_centres).tolist()
    lo, hi = np.array([1.0, 0.0, 0.5]), np.array([2 * RCUT, RCUT ** 2, 2 * RCUT])
    cfg.gap3b_centres = rng.uniform(lo, hi, (cfg.gap3b_n_centres, 3)).tolist()
    return cfg


# ── gap.f90 transcriptions (delta = 1, cutoff(s) = 1, e0 = 0, exp kernel) ──

def _fcut_f90(r, rcut, buffer):
    if r < rcut - buffer:
        return 1.0
    return (np.cos(np.pi * (r - rcut + buffer) / buffer) + 1.0) / 2.0


def _pairs(atoms, rcut):
    i, j, D = neighbor_list("ijD", atoms, rcut)
    r = np.linalg.norm(D, axis=1)
    keep = r > 0
    return i[keep], j[keep], D[keep], r[keep]


def _gap2b_energy_f90(atoms, sp, alphas, Qs, sigma, sp1, sp2, rcut, buffer):
    """get_2b_energy_and_forces energy loop (gap.f90:250-281), summed over sites."""
    ii, jj, _, rr = _pairs(atoms, rcut)
    E = 0.0
    for a, b, r in zip(ii, jj, rr):
        if sp[a] != sp1 and sp[a] != sp2:
            continue
        if not ((sp[a] == sp1 and sp[b] == sp2) or (sp[a] == sp2 and sp[b] == sp1)):
            continue
        if r < rcut:
            fcut = _fcut_f90(r, rcut, buffer)
            for s in range(len(alphas)):
                E += alphas[s] * fcut * np.exp(-0.5 * (r - Qs[s]) ** 2 / sigma ** 2)
    return E


def _gap3b_energy_f90(atoms, sp, alphas, Qs, sigma, sp0, sp1, sp2, rcut, buffer):
    """get_3b_energy_and_forces energy loop (gap.f90:566-647), summed over sites."""
    ii, jj, DD, rr = _pairs(atoms, rcut)
    Qs, sigma = np.asarray(Qs, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    pref = 2.0 * np.asarray(alphas, dtype=np.float64)
    E = 0.0
    for i in range(len(atoms)):
        if sp[i] != sp0:
            continue
        ks = np.nonzero(ii == i)[0]
        for a, k in enumerate(ks):
            r12 = rr[k]
            nsk = sp[jj[k]]
            if (nsk != sp1 and nsk != sp2) or r12 > rcut:
                continue
            xyz12 = DD[k]
            for k2 in ks[a + 1:]:
                r13 = rr[k2]
                nsk2 = sp[jj[k2]]
                if not (((nsk == sp1 and nsk2 == sp2) or (nsk == sp2 and nsk2 == sp1)) and r13 < rcut):
                    continue
                xyz13 = DD[k2]
                r23 = np.sqrt(np.sum((xyz13 - xyz12) ** 2))
                if r12 < rcut and r13 < rcut:
                    fcut = _fcut_f90(r12, rcut, buffer) * _fcut_f90(r13, rcut, buffer)
                    q = np.array([r12 + r13, (r12 - r13) ** 2, r23])
                    d = np.sqrt(np.sum((q - Qs) ** 2 / sigma ** 2, axis=1))
                    E += fcut * np.sum(pref * np.exp(-0.5 * d ** 2))
    return E


# ── exactness ───────────────────────────────────────────────────────────

def test_taper_is_exercised():
    at, _ = _frame()
    _, _, _, r = _pairs(at, RCUT)
    assert np.any((r >= RCUT - BUFFER) & (r < RCUT))
    assert np.any(r < RCUT - BUFFER)


@pytest.mark.parametrize("sp1,sp2", [(0, 0), (0, 1), (1, 1)])
def test_gap2b_matches_f90(sp1, sp2):
    rng = np.random.default_rng(11)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    alphas = rng.normal(size=cfg.gap2b_n_centres)
    q2 = gd.gap_raw_numpy(at, ti, cfg)[:, gd.gap_slices(cfg)["two_body"]].astype(np.float64)
    q2 = q2.reshape(len(at), cfg.num_types, cfg.gap2b_n_centres)
    E_ours = alphas @ q2[ti == sp1, sp2].sum(0)
    if sp1 != sp2:
        E_ours += alphas @ q2[ti == sp2, sp1].sum(0)
    E_f90 = _gap2b_energy_f90(at, ti, alphas, cfg.gap2b_centres, cfg.gap2b_sigma,
                              sp1, sp2, RCUT, BUFFER)
    assert abs(E_f90) > 1e-3
    np.testing.assert_allclose(E_ours, E_f90, rtol=1e-6)


@pytest.mark.parametrize("sp0,sp1,sp2", [(0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)])
def test_gap3b_matches_f90(sp0, sp1, sp2):
    rng = np.random.default_rng(13)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    alphas = rng.normal(size=cfg.gap3b_n_centres)
    q3 = gd.gap_raw_numpy(at, ti, cfg)[:, gd.gap_slices(cfg)["three_body"]].astype(np.float64)
    q3 = q3.reshape(len(at), -1, cfg.gap3b_n_centres)
    p = gd.pair_index(cfg.num_types)[sp1, sp2]
    E_ours = 2.0 * (alphas @ q3[ti == sp0, p].sum(0))
    E_f90 = _gap3b_energy_f90(at, ti, alphas, cfg.gap3b_centres, cfg.gap3b_sigma,
                              sp0, sp1, sp2, RCUT, BUFFER)
    assert abs(E_f90) > 1e-3
    np.testing.assert_allclose(E_ours, E_f90, rtol=1e-6)


# ── invariances / layout ────────────────────────────────────────────────

def test_rigid_motion_and_permutation_invariance():
    rng = np.random.default_rng(5)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    q = gd.gap_raw_numpy(at, ti, cfg)
    assert q.dtype == np.float32 and q.shape == (len(at), gd.gap_dim(cfg))
    moved = at.copy()
    moved.rotate(37.0, rng.normal(size=3), rotate_cell=True)
    moved.translate(rng.normal(size=3))
    np.testing.assert_allclose(gd.gap_raw_numpy(moved, ti, cfg), q, rtol=1e-6, atol=1e-6)
    perm = rng.permutation(len(at))
    np.testing.assert_allclose(gd.gap_raw_numpy(at[perm], ti[perm], cfg), q[perm],
                               rtol=1e-6, atol=1e-6)


def test_pair_index_and_widths():
    for T in (1, 2, 3):
        pi = gd.pair_index(T)
        assert pi.shape == (T, T) and np.array_equal(pi, pi.T)
        assert sorted(pi[np.triu_indices(T)]) == list(range(T * (T + 1) // 2))
    cfg = _cfg()
    T, M2, M3 = cfg.num_types, cfg.gap2b_n_centres, cfg.gap3b_n_centres
    sl = gd.gap_slices(cfg)
    assert sl["two_body"] == slice(0, T * M2)
    assert sl["three_body"] == slice(T * M2, T * M2 + T * (T + 1) // 2 * M3)
    assert gd.gap_dim(cfg) == T * M2 + T * (T + 1) // 2 * M3


def test_f_cut_form():
    r = np.array([0.0, RCUT - BUFFER, RCUT - BUFFER / 2, RCUT, RCUT + 1])
    np.testing.assert_allclose(gd.f_cut(r, RCUT, BUFFER), [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-15)


# ── centre selection / validation ───────────────────────────────────────

def test_select_centres():
    cfg = _cfg(seed=3)
    at, _ = _frame()
    c2, c3 = gd.select_centres([at], cfg)
    assert c2.shape == (cfg.gap2b_n_centres,) and c2[0] == 0.5 and c2[-1] == RCUT
    assert c3.shape == (cfg.gap3b_n_centres, 3)
    assert len(np.unique(c3, axis=0)) == cfg.gap3b_n_centres
    assert np.all(c3[:, 0] >= 0) and np.all(c3[:, 0] <= 2 * RCUT)
    assert np.all(c3[:, 1] >= 0) and np.all(c3[:, 1] <= RCUT ** 2)
    assert np.all(c3[:, 2] >= 0) and np.all(c3[:, 2] <= 2 * RCUT)
    c2b, c3b = gd.select_centres([at], _cfg(seed=3))
    np.testing.assert_array_equal(c2b, c2)
    np.testing.assert_array_equal(c3b, c3)
    _, c3c = gd.select_centres([at], _cfg(seed=4))
    assert not np.array_equal(c3c, c3)


@pytest.mark.parametrize("bad", [
    {"gap2b_n_centres": 0}, {"gap3b_n_centres": -1},
    {"gap2b_sigma": 0.0}, {"gap3b_sigma": [0.5, -0.1, 0.5]}, {"gap3b_sigma": [0.5, 0.5]},
    {"gap_buffer": RCUT + 0.1}, {"gap_buffer": 0.0},
    {"gap2b_centres": [1.0, 2.0]}, {"gap3b_centres": [[1.0, 2.0]] * 9},
])
def test_validate_rejects(bad):
    cfg = _cfg(**bad)
    with pytest.raises(ValueError):
        gd.validate_gap_cfg(cfg)


def test_validate_accepts_default_and_with_centres():
    gd.validate_gap_cfg(_cfg())
    gd.validate_gap_cfg(_random_centres(np.random.default_rng(0), _cfg()))
