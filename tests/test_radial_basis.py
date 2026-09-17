"""Learned SOAP-turbo radial basis (cfg.radial_basis_size)."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import tensorflow as tf
from ase.io import read

from TNEPconfig import TNEPconfig
from DescriptorBuilderGPU import compute_dim_q

CHO_XYZ = os.path.join(ROOT, "datasets", "train.xyz")


def _cho_frames(n=3):
    """First n CHO-only frames of train.xyz (C=6, O=8, H=1)."""
    out = []
    for at in read(CHO_XYZ, index=":40"):
        if set(at.numbers).issubset({6, 8, 1}):
            out.append(at)
        if len(out) == n:
            break
    return out


def _cfg(**kw):
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.dipole_readout = "neighbor"
    cfg.descriptor_mode = 1
    cfg.l_max = 4
    cfg.alpha_max = 4
    cfg.rcut_hard = 4.0
    cfg.rcut_soft = 3.5
    cfg.dipole_rc = 3.5
    cfg.radial_enhancement = 0
    cfg.num_neurons = 8
    cfg.activation = "tanh"
    cfg.descriptor_mixing = False
    cfg.descriptor_preprocess_contract = "off"
    cfg.dipole_channels = 4
    cfg.dipole_species_pairs = "off"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_cutoff = None
    cfg.seed = 7
    cfg.types = [6, 8, 1]
    cfg.num_types = 3
    cfg.type_map = {6: 0, 8: 1, 1: 2}
    cfg.radial_basis_size = None
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.dim_q = compute_dim_q(cfg)
    return cfg


def test_gpu_builder_self_only_neighbor_readout_emits_no_gradients():
    from DescriptorBuilder import make_descriptor_builder
    cfg = _cfg()
    b = make_descriptor_builder(cfg)
    descs, grads, gidx = b.build_descriptors_self_only(_cho_frames(2), batch_size=None)
    assert len(descs) == 2 and descs[0].shape[1] == cfg.dim_q
    assert all(g.shape[0] == 0 for g in grads[0])   # neighbor readout reads no gradients
    assert gidx[0] == [[] for _ in range(len(descs[0]))]


def test_zero_pair_data_stages_on_both_paths():
    """P = 0 (neighbor readout, GPU builder) is a supported invariant through
    pad_and_stack and both chunk-staging paths."""
    from data import (prepare_eval_data, prefetched_chunks, move_data_to_gpu,
                      prestage_chunk_indices, slice_and_complete_chunk)
    cfg = _cfg()
    data = prepare_eval_data(_cho_frames(3), cfg)
    assert tuple(data["grad_values"].shape) == (0, 3, cfg.dim_q)
    ranges = [(0, 2), (2, 3)]
    prestage_chunk_indices(data, ranges)
    for pin in (True, False):
        d = dict(data)
        if not pin:
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)
        chunks = [c for _, _, c in prefetched_chunks(d, ranges, pin_to_cpu=pin)]
        assert [int(c["descriptors"].shape[0]) for c in chunks] == [2, 1]
        assert all(int(c["grad_values"].shape[0]) == 0 for c in chunks)
    c = slice_and_complete_chunk(data, [1, 2])
    assert int(c["pair_struct"].shape[0]) == 0


def test_radial_basis_init_rows_orthonormal_and_identity_at_K_eq_alpha():
    from DescriptorBuilderGPU import radial_basis_init
    M0 = radial_basis_init(4, 8)
    assert M0.shape == (4, 8)
    np.testing.assert_allclose(M0 @ M0.T, np.eye(4), atol=1e-8)
    np.testing.assert_allclose(radial_basis_init(4, 4), np.eye(4), atol=1e-8)
    with pytest.raises(ValueError):
        radial_basis_init(4, 3)
    with pytest.raises(ValueError):
        radial_basis_init(4, 11)


def test_radial_basis_init_maps_K_basis_onto_alpha_basis():
    """M0 @ R_K == R_alpha for the TF radial expansion (single species)."""
    from DescriptorBuilderGPU import (radial_basis_init,
                                      build_orthonormalization_matrix_poly3)
    from DescriptorBuilderGPU_tf import radial_expansion_coeff_poly3_tf
    a, K = 4, 8
    M0 = radial_basis_init(a, K)
    rjs = tf.constant(np.linspace(0.05, 3.99, 200), tf.float64)
    sp = tf.zeros([200], tf.int32); cen = tf.zeros([200], tf.bool)
    kw = dict(n_species=1, rcut_hard=4.0, rcut_soft=3.5, atom_sigma_r=0.5,
              atom_sigma_r_scaling=0.0, amplitude_scaling=1.0, central_weight=1.0,
              radial_enhancement=0, nf=4.0, do_central=True)
    Ra = radial_expansion_coeff_poly3_tf(
        rjs, sp, cen, alpha_max=a,
        W_single=tf.constant(build_orthonormalization_matrix_poly3(a)), **kw).numpy()
    RK = radial_expansion_coeff_poly3_tf(
        rjs, sp, cen, alpha_max=K,
        W_single=tf.constant(build_orthonormalization_matrix_poly3(K)), **kw).numpy()
    np.testing.assert_allclose(M0 @ RK, Ra, atol=1e-7)


def test_radial_basis_channel_map_reproduces_power_spectrum():
    """Pivot-row Gram + gather == the NumPy reference power spectrum, channel
    for channel, on random centre coefficients."""
    from DescriptorBuilderGPU import (radial_basis_channel_map,
                                      make_compress_mask_trivial,
                                      build_multiplicity_array,
                                      power_spectrum_numpy)
    alpha, l_max, S = 4, 4, 3
    n_max = alpha * S; L = l_max + 1; k_max = L * (L + 1) // 2
    cm = radial_basis_channel_map(alpha, l_max, S)
    mi = make_compress_mask_trivial([alpha] * S, l_max)
    assert cm["gather_idx"].shape == (mi["n_compressed"],)
    rng = np.random.default_rng(0)
    cnk = rng.normal(size=(k_max, n_max, 5)) + 1j * rng.normal(size=(k_max, n_max, 5))
    ref = power_spectrum_numpy(
        cnk, build_multiplicity_array(n_max, l_max, mi["skip_mask"]),
        mi["skip_mask"], mi["compressed_idx"], mi["uncompressed_idx"],
        mi["coeffs"], mi["n_compressed"], n_max, l_max)          # [5, Q] normalised
    # Candidate: G[i, p, n', l] = Σ_k Wm[k,l]·Re(c[piv_p,k]·conj c[n',k])
    c = np.transpose(cnk, (2, 1, 0))                             # [N, n_max, k_max]
    piv = c[:, cm["pivot_rows"], :]                              # [N, S, k_max]
    G = np.einsum('ipk,ink,kl->ipnl', piv, np.conj(c), cm["Wm"]).real
    q = G.reshape(5, -1)[:, cm["gather_idx"]] * cm["scale"]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    np.testing.assert_allclose(q, ref, atol=1e-10)


def test_descriptor_input_width():
    from DescriptorBuilderGPU import descriptor_input_width
    cfg = _cfg()
    assert descriptor_input_width(cfg) == cfg.dim_q
    cfg = _cfg(radial_basis_size=8)
    assert descriptor_input_width(cfg) == 3 * 8 * 15 * 2
