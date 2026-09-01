import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

from spectroscopy import _fd_stencil, _fd_assemble


def _x(D, pts, h):
    x = np.zeros(D)
    for a, s in pts:
        x[a] += s * h
    return x


def test_fd_assemble_exact_on_a_quadratic():
    """FD is exact (to roundoff) for a quadratic, so this pins down the stencil
    ordering, the slice index math and the symmetrisation in _fd_assemble."""
    rng = np.random.default_rng(0)
    D, h = 5, 0.1
    g = rng.normal(size=(D, 3))
    H = rng.normal(size=(D, D, 3))
    H = H + H.transpose(1, 0, 2)          # a true Hessian is symmetric in (a,b)
    mu0 = rng.normal(size=3)

    def f(x):
        return mu0 + x @ g + 0.5 * np.einsum('a,b,abc->c', x, x, H)

    P = np.stack([f(_x(D, pts, h)) for pts in _fd_stencil(D)])
    mu, dmu, d2 = _fd_assemble(P, D, h)

    assert np.allclose(mu, mu0, atol=1e-12)
    assert np.allclose(dmu, g, atol=1e-9)
    assert np.allclose(d2, H, atol=1e-8)
    assert np.allclose(d2, d2.transpose(1, 0, 2), atol=1e-12)


def test_fd_point_count():
    D = 7
    assert len(list(_fd_stencil(D))) == 1 + 2 * D + D * (D - 1)


def test_dipole_derivatives_fd_end_to_end():
    """Real model + real frame: the stencil's base point must reproduce the
    plain prediction, and shapes/symmetry must hold."""
    import ase.io
    from data import assign_type_indices
    from spectroscopy import dipole_derivatives_fd, predict_trajectory_batch
    from test_second_layer import _tiny_cfg, _build

    cfg = _tiny_cfg(num_hidden_layers=1, mixing=False)
    model, _, _ = _build(cfg)
    allowed = set(cfg.types)
    frame = next(f for f in ase.io.read(cfg.data_path, index=':6')
                 if set(f.numbers.tolist()) <= allowed)

    sel = [0, 1]
    mu, dmu, d2 = dipole_derivatives_fd(
        model, [frame], atom_idx=sel, h=0.05, batch_size=32,
        pin_to_cpu=True, descriptor_batch_frames=1, verbose=False)

    assert mu.shape == (1, 3)
    assert dmu.shape == (1, len(sel), 3, 3)
    assert d2.shape == (1, len(sel), 3, len(sel), 3, 3)
    assert np.all(np.isfinite(d2))

    ref = predict_trajectory_batch(
        model, model.builder, [frame],
        assign_type_indices([frame], cfg.types),
        pin_to_cpu=True, descriptor_batch_frames=1)
    assert np.allclose(mu[0], ref[0], atol=1e-6), "base point != plain prediction"

    flat = d2[0].reshape(3 * len(sel), 3 * len(sel), 3)
    assert np.allclose(flat, flat.transpose(1, 0, 2), atol=1e-12)
