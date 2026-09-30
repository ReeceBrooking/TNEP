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
    cfg.dipole_pair_rank = 2
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
    cfg.radial_basis_species_sum = False
    cfg.radial_basis_per_l = False
    cfg.radial_basis_norm = "l2"
    cfg.q_shift = None
    cfg.descriptor_scale = "off"
    cfg.q_scale = None
    cfg.optimizer = "snes"
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
    piv = c[:, cm["rows"], :]                                    # [N, S, k_max]
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
    G = np.einsum('apk,ank,kl->apnl', cc[:, cm["rows"]], np.conj(cc), cm["Wm"]).real
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
        W_atom=data.get("V_atom"), M_radial=M,
        nbr_index=data.get("nbr_index"), nbr_G=data.get("nbr_G")).numpy()


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


@pytest.mark.parametrize("readout", ["neighbor", "bilinear"])
def test_candidates_match_single_candidate_predictions(readout):
    cfg = _cfg(radial_basis_size=6, dipole_species_pairs="symmetric",
               dipole_readout=readout)
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
        c_pair=tf.constant(cp), M_radial=tf.constant(Ms),
        nbr_index=data.get("nbr_index"), nbr_G=data.get("nbr_G")).numpy()  # [C, B, 3]
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
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_species_sum=True))   # needs radial_basis_size
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_per_l=True))


@pytest.mark.parametrize("species_sum", [False, True])
def test_snes_tail_holds_M0_and_roundtrips(species_sum):
    from SNES import _set_model_params
    cfg = _cfg(radial_basis_size=8, radial_basis_species_sum=species_sum)
    model = _model(cfg)
    snes = model.optimizer
    T, S, a, K = 3, 3, 4, 8
    assert snes.n_radial == T * S * a * K
    assert snes.dim == snes.n_anns_total + snes.n_U_pair + snes.n_preprocess + snes.n_radial
    named = snes._split_reconstructed(snes.reconstruct_params_tf(snes.mu))
    np.testing.assert_allclose(named["M_radial"].numpy(), model.M_radial.numpy(), atol=1e-7)
    tov = snes._build_type_of_variable()
    assert tov.shape == (snes.dim,)
    np.testing.assert_array_equal(tov[-snes.n_radial:], np.repeat(np.arange(T), S * a * K))
    # batched reconstruct + assign into a fresh model
    pv = tf.stack([snes.mu, snes.mu + 0.01])
    named_b = snes._split_reconstructed(snes.reconstruct_params_tf(pv))
    assert named_b["M_radial"].shape == (2, T, S, a, K)
    m2 = _model(cfg)
    _set_model_params(m2, *snes.reconstruct_params_tf(snes.mu + 0.01))
    np.testing.assert_allclose(m2.M_radial.numpy(), named_b["M_radial"][1].numpy(), atol=1e-7)


def test_snes_gen0_fitness_matches_fixed_basis():
    """Same seed ⇒ identical ANN μ; learned tail at M0 ⇒ identical fitness."""
    frames = _cho_frames(6)
    out = {}
    for K in (None, 8):
        cfg = _cfg(radial_basis_size=K, pop_size=2)
        model = _model(cfg)
        data = _eval_data(cfg, frames)
        data["types_contained"] = tf.ones((len(frames), 3))
        snes = model.optimizer
        fit = snes.evaluate_population(tf.stack([snes.mu, snes.mu]), data).numpy()
        out[K] = (fit, snes.validate(data, snes.mu))
    np.testing.assert_allclose(out[8][0], out[None][0], rtol=1e-4)
    assert abs(out[8][1] - out[None][1]) < 1e-5 * max(1.0, out[None][1])


def test_save_load_roundtrip_preserves_M_radial_and_predictions(tmp_path):
    from model_io import save_model, load_model
    cfg = _cfg(radial_basis_size=6)
    model = _model(cfg)
    rng = np.random.default_rng(9)
    model.M_radial.assign((model.M_radial.numpy()
                           + 0.2 * rng.normal(size=model.M_radial.shape)).astype(np.float32))
    data = _eval_data(cfg, _cho_frames(2))
    before = _predict(model, data)
    path = str(tmp_path / "m.h5")
    save_model(model, cfg, path)
    loaded = load_model(path)
    assert loaded.cfg.radial_basis_size == 6
    np.testing.assert_allclose(loaded.M_radial.numpy(), model.M_radial.numpy())
    np.testing.assert_allclose(_predict(loaded, data), before, rtol=1e-5, atol=1e-7)


def _soap_from_raw_summed_np(raw, Z, M, alpha, l_max, S):
    """Independent NumPy reference for the species-summed mode (one structure):
    channels summed over s, full (n <= n') power spectrum in Fortran order."""
    L = l_max + 1; k_max = L * (L + 1) // 2; K = M.shape[-1]
    A = raw.shape[0]
    r = raw.reshape(A, S, K, k_max, 2).astype(np.float64)
    c_raw = r[..., 0] + 1j * r[..., 1]                                    # [A, S, K, k]
    cc = np.einsum('asnk,askm->anm', M[Z].astype(np.float64), c_raw)     # [A, alpha, k]
    out = []
    for n in range(alpha):
        for nprime in range(n, alpha):
            for l in range(L):
                k0 = l * (l + 1) // 2
                prod = (cc[:, n, k0:k0 + l + 1] * np.conj(cc[:, nprime, k0:k0 + l + 1])).real
                mult = np.array([1.0] + [2.0] * l) * (np.sqrt(2.0) if n != nprime else 1.0)
                out.append(prod @ mult)
    q = np.stack(out, axis=1)                                             # [A, dim_q]
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(nrm < 1e-5, 1.0, nrm)


def test_species_sum_dim_q_and_channel_map():
    from DescriptorBuilderGPU import (radial_basis_channel_map, compute_dim_q,
                                      power_spectrum_numpy)
    cfg = _cfg(radial_basis_size=8, radial_basis_species_sum=True)
    assert compute_dim_q(cfg) == 4 * 5 // 2 * 5 == 50 and cfg.dim_q == 50
    assert compute_dim_q(_cfg(radial_basis_size=8)) == 165
    cm = radial_basis_channel_map(4, 4, 3, species_sum=True)
    assert cm["gather_idx"].shape == (50,) and cm["n_max"] == 4
    np.testing.assert_array_equal(cm["rows"], np.arange(4))
    # Against power_spectrum_numpy with NO compression over a single alpha block.
    L, k_max, n_max = 5, 15, 4
    n_unc = n_max * (n_max + 1) // 2 * L
    rng = np.random.default_rng(1)
    cnk = rng.normal(size=(k_max, n_max, 5)) + 1j * rng.normal(size=(k_max, n_max, 5))
    from DescriptorBuilderGPU import build_multiplicity_array
    ref = power_spectrum_numpy(
        cnk, build_multiplicity_array(n_max, 4, None), np.zeros(n_unc, bool),
        np.arange(n_unc), np.arange(n_unc), np.ones(n_unc), n_unc, n_max, 4)
    c = np.transpose(cnk, (2, 1, 0))
    G = np.einsum('ipk,ink,kl->ipnl', c[:, cm["rows"], :], np.conj(c), cm["Wm"]).real
    q = G.reshape(5, -1)[:, cm["gather_idx"]] * cm["scale"]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    np.testing.assert_allclose(q, ref, atol=1e-8, rtol=0)


def test_species_sum_block_layout_is_one_block():
    from DescriptorBuilderGPU import descriptor_block_layout
    lay = descriptor_block_layout(_cfg(radial_basis_size=8, radial_basis_species_sum=True))
    assert lay["pair_keys"] == [(0, 0)] and lay["dim_q"] == 50
    assert lay["alpha_eff_per_pair"][(0, 0)] == 10 and lay["N_per_l"] == 10
    assert sorted(np.concatenate([lay["pair_ln_index"][(0, 0)][l] for l in range(5)])) == list(range(50))


@pytest.mark.parametrize("mixing", [False, True])
def test_species_sum_model_matches_numpy_reference(mixing):
    cfg = _cfg(radial_basis_size=8, radial_basis_species_sum=True,
               descriptor_mixing=mixing, descriptor_mixing_regularizer="expm")
    model = _model(cfg)
    assert model.W0.shape[1] == 50
    frames = _cho_frames(2)
    data = _eval_data(cfg, frames)
    rng = np.random.default_rng(4)
    M = (model.M_radial.numpy() + 0.1 * rng.normal(size=model.M_radial.shape)).astype(np.float32)
    q_tf = model.soap_from_raw(data["descriptors"], data["Z_int"], tf.constant(M)).numpy()
    for s, fr in enumerate(frames):
        A = len(fr)
        Z = data["Z_int"][s].numpy()[:A]
        ref = _soap_from_raw_summed_np(data["descriptors"][s].numpy()[:A], Z, M, 4, 4, 3)
        np.testing.assert_allclose(q_tf[s, :A], ref, atol=1e-5)
        assert np.all(q_tf[s, A:] == 0.0)
    # Candidate path (with mixing tail when enabled) agrees with single-candidate predict_batch.
    snes = model.optimizer
    pv = tf.stack([snes.mu, snes.mu + 0.02])
    named = snes._split_reconstructed(snes.reconstruct_params_tf(pv))
    preds = model.predict_batch_candidates(
        data["descriptors"], data["V_atom"], data["Z_int"], data["atom_mask"],
        named["W0"], named["b0"], named["W1"], named["b1"],
        U_pair=named["U_pair"], c_pair=named["c_pair"], M_radial=named["M_radial"]).numpy()
    for c in range(2):
        one = snes._split_reconstructed(snes.reconstruct_params_tf(pv[c]))
        W0 = model._W0_eff(one["W0"], one["U_pair"]) if mixing else one["W0"]
        single = model.predict_batch(
            data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
            data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
            W0, one["b0"], one["W1"], one["b1"], W_atom=data["V_atom"],
            c_pair=one["c_pair"], M_radial=one["M_radial"]).numpy()
        np.testing.assert_allclose(preds[c], single, rtol=1e-4, atol=1e-6)


def test_species_sum_rejects_preprocess_and_roundtrips(tmp_path):
    from model_io import save_model, load_model
    with pytest.raises(NotImplementedError):
        _model(_cfg(radial_basis_size=8, radial_basis_species_sum=True,
                    descriptor_preprocess_contract="angular"))
    cfg = _cfg(radial_basis_size=6, radial_basis_species_sum=True)
    model = _model(cfg)
    data = _eval_data(cfg, _cho_frames(2))
    before = _predict(model, data)
    path = str(tmp_path / "m.h5")
    save_model(model, cfg, path)
    loaded = load_model(path)
    assert loaded.cfg.radial_basis_species_sum is True and loaded.cfg.dim_q == 50
    np.testing.assert_allclose(_predict(loaded, data), before, rtol=1e-5, atol=1e-7)


def _soap_from_raw_per_l_np(raw, Z, M, alpha, l_max, S, species_sum):
    """Independent NumPy reference for radial_basis_per_l: M [T,S,alpha,K,L],
    each l block contracted with its own M[..., l]."""
    L = l_max + 1; k_max = L * (L + 1) // 2; K = M.shape[3]
    A = raw.shape[0]
    r = raw.reshape(A, S, K, k_max, 2).astype(np.float64)
    c_raw = r[..., 0] + 1j * r[..., 1]                                    # [A, S, K, k]
    n_max = alpha if species_sum else S * alpha
    cc = np.zeros((A, n_max, k_max), complex)
    for l in range(L):
        ks = slice(l * (l + 1) // 2, l * (l + 1) // 2 + l + 1)
        Ml = M[Z][..., l].astype(np.float64)                              # [A, S, alpha, K]
        if species_sum:
            cc[:, :, ks] = np.einsum('asnk,askm->anm', Ml, c_raw[..., ks])
        else:
            cc[:, :, ks] = np.einsum('asnk,askm->asnm', Ml, c_raw[..., ks]).reshape(A, n_max, l + 1)
    from DescriptorBuilderGPU import radial_basis_channel_map
    cm = radial_basis_channel_map(alpha, l_max, S, species_sum=species_sum)
    G = np.einsum('apk,ank,kl->apnl', cc[:, cm["rows"]], np.conj(cc), cm["Wm"]).real
    q = G.reshape(A, -1)[:, cm["gather_idx"]] * cm["scale"]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n < 1e-5, 1.0, n)


@pytest.mark.parametrize("species_sum", [False, True])
def test_per_l_radial_basis(species_sum, tmp_path):
    from model_io import save_model, load_model
    cfg = _cfg(radial_basis_size=8, radial_basis_per_l=True,
               radial_basis_species_sum=species_sum, dipole_species_pairs="symmetric")
    model = _model(cfg)
    assert tuple(model.M_radial.shape) == (3, 3, 4, 8, 5)
    frames = _cho_frames(3)
    data = _eval_data(cfg, frames)
    # Init reproduces the shared-M descriptor exactly (M0 tiled over l).
    cfg_s = _cfg(radial_basis_size=8, radial_basis_species_sum=species_sum)
    m_s = _model(cfg_s)
    q_per_l = model.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    q_shared = m_s.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    np.testing.assert_allclose(q_per_l, q_shared, atol=1e-6)
    # Random per-l M against the independent NumPy reference; padded atoms zero.
    rng = np.random.default_rng(6)
    M = (model.M_radial.numpy() + 0.1 * rng.normal(size=model.M_radial.shape)).astype(np.float32)
    q = model.soap_from_raw(data["descriptors"], data["Z_int"], tf.constant(M)).numpy()
    for s, fr in enumerate(frames):
        A = len(fr); Z = data["Z_int"][s].numpy()[:A]
        ref = _soap_from_raw_per_l_np(data["descriptors"][s].numpy()[:A], Z, M, 4, 4, 3, species_sum)
        np.testing.assert_allclose(q[s, :A], ref, atol=1e-5)
        assert np.all(q[s, A:] == 0.0)
    # Different l blocks really use different M: perturb only l=3 -> only l=3 channels change.
    M3 = model.M_radial.numpy().copy(); M3[..., 3] += 0.3 * rng.normal(size=M3[..., 3].shape)
    q3 = model.soap_from_raw(data["descriptors"], data["Z_int"], tf.constant(M3.astype(np.float32))).numpy()
    from DescriptorBuilderGPU import radial_basis_channel_map
    l_of_q = np.array([g % 5 for g in radial_basis_channel_map(4, 4, 3, species_sum=species_sum)["gather_idx"]])
    unnorm_same = np.abs(q3 - q_per_l)   # normalisation couples channels, so compare ratios per atom
    A = len(frames[0])
    assert np.abs(q3[0, :A][:, l_of_q == 3] - q_per_l[0, :A][:, l_of_q == 3]).max() > 1e-3
    # SNES tail + candidates path + save/load.
    snes = model.optimizer
    assert snes.n_radial == 3 * 3 * 4 * 8 * 5
    pv = tf.stack([snes.mu, snes.mu + 0.02])
    named = snes._split_reconstructed(snes.reconstruct_params_tf(pv))
    assert tuple(named["M_radial"].shape) == (2, 3, 3, 4, 8, 5)
    preds = model.predict_batch_candidates(
        data["descriptors"], data["V_atom"], data["Z_int"], data["atom_mask"],
        named["W0"], named["b0"], named["W1"], named["b1"],
        c_pair=named["c_pair"], M_radial=named["M_radial"]).numpy()
    for c in range(2):
        one = snes._split_reconstructed(snes.reconstruct_params_tf(pv[c]))
        single = model.predict_batch(
            data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
            data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
            one["W0"], one["b0"], one["W1"], one["b1"], W_atom=data["V_atom"],
            c_pair=one["c_pair"], M_radial=one["M_radial"]).numpy()
        np.testing.assert_allclose(preds[c], single, rtol=1e-4, atol=1e-6)
    model.M_radial.assign(M)
    before = _predict(model, data)
    path = str(tmp_path / "m.h5"); save_model(model, cfg, path); loaded = load_model(path)
    assert loaded.cfg.radial_basis_per_l is True
    np.testing.assert_allclose(_predict(loaded, data), before, rtol=1e-5, atol=1e-7)


def _rotated(atoms, seed):
    """Copy of atoms rigidly rotated about the cell centre (random rotation)."""
    from scipy.spatial.transform import Rotation
    at = atoms.copy()
    R = Rotation.random(random_state=seed).as_matrix()
    c = at.cell.lengths() / 2
    at.positions = (at.positions - c) @ R.T + c
    at.info["dipole"] = np.zeros(3)   # copy() drops the calculator that held the target
    return at


@pytest.mark.parametrize("species_sum,per_l", [(False, False), (True, True)])
def test_learned_basis_is_rotation_invariant(species_sum, per_l):
    """Every channel of the learned-basis descriptor is invariant under a
    rigid rotation of the frame (random M, not just the fixed-basis init)."""
    cfg = _cfg(radial_basis_size=8, radial_basis_species_sum=species_sum,
               radial_basis_per_l=per_l)
    model = _model(cfg)
    rng = np.random.default_rng(8)
    model.M_radial.assign(
        (model.M_radial.numpy()
         + 0.1 * rng.normal(size=model.M_radial.shape)).astype(np.float32))
    frames = _cho_frames(3)
    data = _eval_data(cfg, frames)
    q = model.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    rot = [_rotated(fr, 11 + i) for i, fr in enumerate(frames)]
    d_rot = _eval_data(cfg, rot)
    q_rot = model.soap_from_raw(d_rot["descriptors"], d_rot["Z_int"]).numpy()
    assert np.abs(q).max() > 1e-4
    np.testing.assert_allclose(q_rot, q, rtol=1e-4, atol=1e-5)
