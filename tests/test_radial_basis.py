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
    np.testing.assert_allclose(q, ref, atol=1e-8, rtol=0)


def test_descriptor_input_width():
    from DescriptorBuilderGPU import descriptor_input_width
    cfg = _cfg()
    assert descriptor_input_width(cfg) == cfg.dim_q
    cfg = _cfg(radial_basis_size=8)
    assert descriptor_input_width(cfg) == 3 * 8 * 15 * 2


def _soap_from_raw_np(raw, Z, M, alpha, l_max, S):
    """Independent NumPy reference of TNEP.soap_from_raw (one structure).
    raw [A, S·K·k_max·2]; Z [A]; M [T, S, alpha, K]."""
    from DescriptorBuilderGPU import radial_basis_channel_map
    L = l_max + 1; k_max = L * (L + 1) // 2; K = M.shape[-1]
    A = raw.shape[0]
    r = raw.reshape(A, S, K, k_max, 2).astype(np.float64)
    c_raw = r[..., 0] + 1j * r[..., 1]                                   # [A, S, K, k]
    cc = np.einsum('asnk,askm->asnm', M[Z].astype(np.float64), c_raw)   # [A, S, alpha, k]
    cc = cc.reshape(A, S * alpha, k_max)
    cm = radial_basis_channel_map(alpha, l_max, S)
    G = np.einsum('apk,ank,kl->apnl', cc[:, cm["pivot_rows"]], np.conj(cc), cm["Wm"]).real
    q = G.reshape(A, -1)[:, cm["gather_idx"]] * cm["scale"]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n < 1e-5, 1.0, n)


@pytest.mark.parametrize("K,precision", [(4, "float64"), (8, "float64"), (8, "float32")])
def test_builder_raw_coefficients_reproduce_fixed_descriptor(K, precision):
    from DescriptorBuilder import make_descriptor_builder
    from DescriptorBuilderGPU import radial_basis_init, descriptor_input_width
    frames = _cho_frames(3)
    fixed = make_descriptor_builder(_cfg()).build_descriptors_flat(frames, calc_gradients=False)
    cfg = _cfg(radial_basis_size=K, descriptor_precision=precision)
    b = make_descriptor_builder(cfg)
    raw = b.build_descriptors_flat(frames, calc_gradients=False)
    assert raw[0][0].shape[1] == descriptor_input_width(cfg)
    M0 = np.broadcast_to(radial_basis_init(4, K), (3, 3, 4, K))
    for fr, (desc, _, _, _), (r, _, _, _) in zip(frames, fixed, raw):
        Z = np.array([cfg.type_map[int(z)] for z in fr.numbers])
        q = _soap_from_raw_np(r, Z, M0, 4, 4, 3)
        np.testing.assert_allclose(q, desc, atol=2e-5)
    # Neighbor readout + gradients requested (trajectory path): real pair
    # stream, zero-width gradient values.
    out = b.build_descriptors_flat(frames, calc_gradients=True)
    assert out[0][1].shape == (out[0][2].shape[0], 3, 0) and out[0][2].shape[0] > 0
    # Gradient-consuming readout: not implemented under the knob.
    b2 = make_descriptor_builder(_cfg(radial_basis_size=K, dipole_readout="tnep"))
    with pytest.raises(NotImplementedError):
        b2.build_descriptors_flat(frames, calc_gradients=True)


def _model(cfg):
    from TNEP import TNEP
    tf.keras.utils.set_random_seed(0)
    return TNEP(cfg)


def _eval_data(cfg, frames):
    from data import prepare_eval_data
    return prepare_eval_data(frames, cfg)


def _predict(model, data, M=None):
    return model.predict_batch(
        data["descriptors"], data["grad_values"],
        data["pair_atom"], data["pair_gidx"], data["pair_struct"],
        data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        W_atom=data.get("V_atom"), M_radial=M).numpy()


@pytest.mark.parametrize("K", [4, 8])
def test_model_soap_from_raw_matches_numpy_reference(K):
    cfg = _cfg(radial_basis_size=K)
    model = _model(cfg)
    frames = _cho_frames(2)
    data = _eval_data(cfg, frames)
    rng = np.random.default_rng(3)
    M = (model.M_radial.numpy() + 0.1 * rng.normal(size=model.M_radial.shape)).astype(np.float32)
    q_tf = model.soap_from_raw(data["descriptors"], data["Z_int"], tf.constant(M)).numpy()
    for s, fr in enumerate(frames):
        A = len(fr)
        Z = data["Z_int"][s].numpy()[:A]
        ref = _soap_from_raw_np(data["descriptors"][s].numpy()[:A], Z, M, 4, 4, 3)
        np.testing.assert_allclose(q_tf[s, :A], ref, atol=1e-5)
    # padded atoms must be exactly zero (norm guard on an empty row)
    A_max = data["descriptors"].shape[1]
    for s, fr in enumerate(frames):
        assert np.all(q_tf[s, len(fr):] == 0.0)


def test_learned_model_at_init_equals_fixed_model():
    """Same ANN weights, fixed vs learned(K=8) at M0 → identical dipoles."""
    frames = _cho_frames(3)
    cfg_f = _cfg(); m_f = _model(cfg_f); d_f = _eval_data(cfg_f, frames)
    cfg_l = _cfg(radial_basis_size=8); m_l = _model(cfg_l); d_l = _eval_data(cfg_l, frames)
    for w in ("W0", "b0", "W1", "b1"):
        getattr(m_l, w).assign(getattr(m_f, w))
    np.testing.assert_allclose(_predict(m_l, d_l), _predict(m_f, d_f), rtol=1e-4, atol=1e-6)


def test_candidates_match_single_candidate_predictions():
    cfg = _cfg(radial_basis_size=6, dipole_species_pairs="symmetric")
    model = _model(cfg)
    data = _eval_data(cfg, _cho_frames(3))
    rng = np.random.default_rng(5)
    C = 3
    M0 = model.M_radial.numpy()
    Ms = np.stack([M0 + 0.05 * c * rng.normal(size=M0.shape) for c in range(C)]).astype(np.float32)
    W0 = np.stack([model.W0.numpy()] * C); b0 = np.stack([model.b0.numpy()] * C)
    W1 = np.stack([model.W1.numpy()] * C); b1 = np.zeros(C, np.float32)
    cp = np.stack([model.c_pair.numpy()] * C)
    preds = model.predict_batch_candidates(
        data["descriptors"], data["V_atom"], data["Z_int"], data["atom_mask"],
        tf.constant(W0), tf.constant(b0), tf.constant(W1), tf.constant(b1),
        c_pair=tf.constant(cp), M_radial=tf.constant(Ms)).numpy()         # [C, B, 3]
    for c in range(C):
        np.testing.assert_allclose(preds[c], _predict(model, data, tf.constant(Ms[c])),
                                   rtol=1e-4, atol=1e-6)


def test_learned_radial_rejects_gradient_modes_and_cpu_builder():
    with pytest.raises(NotImplementedError):
        _model(_cfg(radial_basis_size=8, dipole_readout="tnep"))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=8, descriptor_mode=0))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=3))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=11))
