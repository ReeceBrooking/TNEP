"""gap_models.py — body-order site-model descriptors and angular kernels."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
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
    rc, C = float(cfg.rcut_hard), 3
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
    rc, C, L = float(cfg.rcut_hard), 3, 2
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


def test_angular_cross_tensor_l0_equals_existing_cross_group_and_is_covariant():
    import gap_dipole
    cfg, at, ti = _setup(target_mode=2, gap_cross_channels=2, gap_cross_lmax=0)
    W = gm.angular_cross_tensor(at, ti, cfg)                 # [N, 6, K1(K1+1)/2]
    X = gap_dipole._cross_group(gap_dipole._cross_vectors(at, ti, cfg))   # sym(v⊗v), traceless
    # v⊗v = Σ_{j,k} d_j d_k (INCLUDING j = k); the l=0 angular kernel excludes
    # j = k, so add the bond-axis part back: Σ_j R_a R_b (d_j d_jᵀ)_traceless.
    i, j, d, R = gm._site_pairs(at, cfg, 2)
    tj = ti[j]; Rp = np.zeros((len(R), 2, 2)); Rp[np.arange(len(R)), :, tj] = R; Rp = Rp.reshape(len(R), -1)
    au, bu = np.triu_indices(Rp.shape[1])
    dd = d[:, :, None] * d[:, None, :]; tr = np.trace(dd, axis1=1, axis2=2) / 3
    for k in range(3): dd[:, k, k] -= tr
    voigt = np.stack([dd[:, 0, 0], dd[:, 1, 1], dd[:, 2, 2], dd[:, 0, 1], dd[:, 1, 2], dd[:, 2, 0]], 1)
    diag = np.zeros_like(W); np.add.at(diag, i, voigt[:, :, None] * (Rp[:, au] * Rp[:, bu])[:, None, :])
    assert np.allclose(W + diag, X, atol=1e-5 * np.abs(X).max())
    # covariance: W(Rx) = R W Rᵀ per site; translation invariance
    R3 = _rot(0.4, 0.9, -0.7)
    W2 = gm.angular_cross_tensor(_moved(at, R3), ti, cfg)
    def mat(v): return np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])
    for n in (0, 7, 50):
        for c in range(W.shape[2]):
            assert np.allclose(mat(W2[n, :, c]), R3 @ mat(W[n, :, c]) @ R3.T, atol=1e-5 * np.abs(W).max())
    assert np.allclose(gm.angular_cross_tensor(_moved(at, shift=[2, 1, -1]), ti, cfg), W, atol=1e-6 * np.abs(W).max())


def test_angular_cross_vector_is_odd_and_covariant():
    cfg, at, ti = _setup(target_mode=1, gap_cross_channels=2, gap_cross_lmax=2)
    w = gm.angular_cross_vector(at, ti, cfg)                 # [N, 3, K1²·2]
    assert w.shape[2] == 16 * 2
    R3 = _rot(0.2, 0.5, 0.8)
    w2 = gm.angular_cross_vector(_moved(at, R3), ti, cfg)
    assert np.allclose(w2, np.einsum("ab,nbk->nak", R3, w), atol=1e-5 * np.abs(w).max())
    assert np.allclose(gm.angular_cross_vector(_moved(at, -np.eye(3)), ti, cfg), -w, atol=1e-5 * np.abs(w).max())   # odd


def test_no_triplets_gives_zero_angular_outputs():
    """A dimer has no centre with two neighbours: 3b descriptor and angular
    kernels must come back as correctly shaped zeros, the 2b descriptor intact."""
    from ase import Atoms
    cfg, _, _ = _setup(gap_cross_channels=2, gap_cross_lmax=1)
    at = Atoms("OH", positions=[[10.0, 10.0, 10.0], [10.97, 10.0, 10.0]], cell=[20.0] * 3, pbc=True)
    ti = np.array([0, 1])
    q3 = gm.three_body_descriptor(at, ti, cfg)
    assert q3.shape == (2, 3 * 9 * 3) and not q3.any()
    W = gm.angular_cross_tensor(at, ti, cfg)
    assert W.shape == (2, 6, 10 * 2) and not W.any()
    w = gm.angular_cross_vector(at, ti, cfg)
    assert w.shape == (2, 3, 16 * 1) and not w.any()
    q2 = gm.two_body_descriptor(at, ti, cfg)
    assert np.isfinite(q2).all() and np.allclose(np.linalg.norm(q2, axis=1), 1.0)
