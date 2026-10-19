"""gap_models.py — body-order site-model descriptors and angular kernels."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest
from ase.io import read
from ase.neighborlist import neighbor_list
import gap_models as gm
from test_gap_dipole import _cfg

WATER = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def _setup(**kw):
    cfg = _cfg(dipole_radial="chebyshev", dipole_cutoff="cosine2", dipole_channels=3,
               dipole_species_pairs="symmetric", gap_site_channels=3, gap_site_lmax=2, **kw)
    cfg.types, cfg.num_types = [8, 1], 2
    at = read(WATER, index="1")
    return cfg, at, np.array([0 if z == 8 else 1 for z in at.numbers])


def _rot(a, b, c):
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return Rz @ Ry @ Rx


def _moved(at, R=None, shift=None):
    b = at.copy()
    if R is not None:
        b.set_cell(at.cell[:] @ R.T, scale_atoms=False); b.positions = at.positions @ R.T
    if shift is not None:
        b.positions = b.positions + shift
    return b


def test_two_body_descriptor_matches_hand_sum_and_is_normalised():
    cfg, at, ti = _setup()
    q = gm.two_body_descriptor(at, ti, cfg)                   # [N, T·C]
    assert q.shape == (96, 2 * 3)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    rc, C = float(cfg.dipole_rc), 3
    i, j, d = neighbor_list("ijd", at, rc)
    keep = i != j; i, j, d = i[keep], j[keep], d[keep]
    x = 2.0 * d / rc - 1.0
    fc = ((1.0 + np.cos(np.pi * d / rc)) / 2.0) ** 2
    R = np.stack([np.ones_like(x), x, 2 * x * x - 1], 1) * fc[:, None]
    ref = np.zeros((96, 2, C)); np.add.at(ref, (i, ti[j]), R)
    ref = ref.reshape(96, -1); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.allclose(q, ref, atol=1e-6)


def test_three_body_descriptor_matches_triplet_loop_and_symmetries():
    cfg, at, ti = _setup()
    q = gm.three_body_descriptor(at, ti, cfg)                 # [N, T(T+1)/2 · C·C · (L+1)]
    assert q.shape == (96, 3 * 9 * 3)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    # explicit triplet loop for one atom
    rc, C, L = float(cfg.dipole_rc), 3, 2
    i, j, D = neighbor_list("ijD", at, rc); keep = i != j; i, j, D = i[keep], j[keep], -D[keep]
    a = 5; sel = i == a; js, ds = j[sel], D[sel]
    r = np.linalg.norm(ds, axis=1); x = 2 * r / rc - 1
    fc = ((1 + np.cos(np.pi * r / rc)) / 2) ** 2
    R = np.stack([np.ones_like(x), x, 2 * x * x - 1], 1) * fc[:, None]
    pidx = np.array([[0, 1], [1, 2]])
    ref = np.zeros((3, C, C, L + 1))
    for p in range(len(js)):
        for k in range(len(js)):
            if p == k: continue
            cos = ds[p] @ ds[k] / (r[p] * r[k])
            P = [1.0, cos, 0.5 * (3 * cos * cos - 1)]
            slot = pidx[ti[js[p]], ti[js[k]]]
            ref[slot] += R[p][:, None, None] * R[k][None, :, None] * np.array(P)[None, None, :]
    ref = ref.reshape(-1); ref /= np.linalg.norm(ref)
    assert np.allclose(q[a], ref, atol=1e-6)
    # invariances
    R3 = _rot(0.6, -0.3, 1.2)
    assert np.allclose(gm.three_body_descriptor(_moved(at, R3), ti, cfg), q, atol=1e-6)
    assert np.allclose(gm.three_body_descriptor(_moved(at, shift=[1.0, -2.0, 0.5]), ti, cfg), q, atol=1e-6)
    assert np.allclose(gm.two_body_descriptor(_moved(at, R3, [0.3, 0.3, 0.3]), ti, cfg),
                       gm.two_body_descriptor(at, ti, cfg), atol=1e-6)
