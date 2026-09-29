"""Tests for the multi-descriptor plan (multi_descriptor_plan.md).

Task 1: radial_basis_norm="none" on the FIXED GPU SOAP builder.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DescriptorBuilder import make_descriptor_builder
from test_neighbor_dipole import _cfg as _base_cfg, _model, _bulk_frame


def _cfg(**kw):
    # GPU (TF) builder specifically; kwargs still override.
    return _base_cfg(**{"descriptor_mode": 1, "descriptor_precision": "float64", **kw})


def _descriptors(cfg, atoms):
    """[A, Q] numpy descriptors of one frame (forward-only builder path)."""
    desc = make_descriptor_builder(cfg).build_descriptors(
        [atoms], calc_gradients=False)[0][0]
    return np.asarray(desc, dtype=np.float64)


def _self_gradients(cfg, atoms):
    """[A, 3, Q] numpy self gradients ∂q_i/∂r_i (with-grad builder path)."""
    _, grads, _ = make_descriptor_builder(cfg).build_descriptors_self_only([atoms])
    return np.stack([np.asarray(g, dtype=np.float64)[0] for g in grads[0]], axis=0)


def _fd_self_gradient(cfg, atoms, atom, h):
    """[3, Q] central finite difference of q_atom w.r.t. r_atom."""
    out = []
    for c in range(3):
        qs = []
        for s in (+1.0, -1.0):
            at = atoms.copy()
            pos = at.get_positions()
            pos[atom, c] += s * h
            at.set_positions(pos)
            qs.append(_descriptors(cfg, at)[atom])
        out.append((qs[0] - qs[1]) / (2.0 * h))
    return np.stack(out, axis=0)


# ── Task 1: radial_basis_norm="none" on the fixed builder ──────────────────

def test_fixed_soap_norm_none_matches_unnormalised():
    cfg_l2 = _cfg(radial_basis_norm="l2")
    cfg_no = _cfg(radial_basis_norm="none")
    atoms = _bulk_frame()
    q_l2 = _descriptors(cfg_l2, atoms)
    q_no = _descriptors(cfg_no, atoms)                                    # [A, Q]
    n = np.linalg.norm(q_no, axis=1, keepdims=True)
    np.testing.assert_allclose(q_no / n, q_l2, rtol=1e-5, atol=1e-6)      # same direction
    # Magnitude kept and varies with the environment. The raw spectrum is
    # ~4e5 per atom and dominated by the central-atom term, so the spread
    # is modest (O/H ratio ~1.2 on this frame).
    assert np.all(n[:, 0] > 0.0) and n.max() / n.min() > 1.1


@pytest.mark.parametrize("tile", [0, 8000])   # single-shot and pair-tiled gradient paths
def test_fixed_soap_norm_none_gradient_is_unprojected(tile):
    # "tnep" readout so the builder returns descriptor gradients.
    cfg = _cfg(radial_basis_norm="none", dipole_readout="tnep",
               descriptor_pair_tile_size=tile)
    atoms = _bulk_frame()
    g = _self_gradients(cfg, atoms)                                       # [A, 3, Q]
    g_fd = _fd_self_gradient(cfg, atoms, atom=0, h=1e-3)                  # [3, Q]
    # Descriptors leave the builder as float32 at |q| ~ 3e5, so the FD
    # noise floor is eps32·|q|/(2h) ~ 10 on a gradient of scale ~1e5:
    # tolerate 1e-3 of the tensor scale (measured residual ~7e-5 of it).
    np.testing.assert_allclose(g[0], g_fd, rtol=1e-3,
                               atol=1e-3 * np.abs(g_fd).max())
    # Projection identity: applying (I − q̂q̂ᵀ)/‖q‖ to the raw gradient must
    # reproduce the "l2" builder's gradient exactly.
    g_l2 = _self_gradients(_cfg(dipole_readout="tnep",
                                descriptor_pair_tile_size=tile), atoms)
    q = _descriptors(cfg, atoms)[0]; n = np.linalg.norm(q); qh = q / n
    g_proj = (g[0] - (g[0] @ qh)[:, None] * qh) / n
    np.testing.assert_allclose(g_proj, g_l2[0], rtol=1e-5,
                               atol=1e-6 * np.abs(g_l2[0]).max())


def test_fixed_soap_rejects_sqrt():
    with pytest.raises(ValueError, match="radial_basis_norm"):
        _model(_cfg(radial_basis_norm="sqrt"))
    with pytest.raises(ValueError, match="radial_basis_norm"):
        make_descriptor_builder(_cfg(radial_basis_norm="sqrt"))
    # The quippy builder (descriptor_mode=0) ignores radial_basis_norm.
    with pytest.raises(ValueError, match="radial_basis_norm"):
        _model(_cfg(radial_basis_norm="none", descriptor_mode=0))
