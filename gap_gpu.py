"""Batched TF forward of the soap-only polarisability GAP (the Raman model): the
TF twin of gap_dipole's NumPy contraction (_phi_frame), restricted to what the
Raman runs use; everything else falls back to NumPy.

    α[x] = Σ_t Σ_ξ Σ_{n∈t} Σ_k v[n,x,k] · δ² · (D_t·D_{M,t}ᵀ)^ξ[n,:] · A_{t,ξ}[:,k]

Descriptors come from the builder in float32; bond basis, kernel and readout run
in float64 because per-atom contributions cancel across hundreds of atoms
(float32/TF32 residuals reached 0.2–300 Å³ on a 290 Å³ cell). Validated against
NumPy in tests/test_gap_gpu.py (max |Δα| ~1e-6 Å³)."""
import numpy as np
import tensorflow as tf

# TF32 matmul turned the cancelling sums into ~300 Å³ errors; the forward is float64 now, but keep it off.
tf.config.experimental.enable_tensor_float_32_execution(False)

from TNEPconfig import resolved_dipole_cutoff, resolved_dipole_rc
import gap_models as gm


def gpu_applicable(model) -> bool:
    cfg = model.cfg
    return (int(cfg.target_mode) == 2 and list(cfg.gap_site_models) == ["soap"]
            and type(model.builder).__name__ == "DescriptorBuilderGPUTF" and model.builder2 is None
            and gm.pair_columns(cfg) and str(cfg.dipole_radial) in ("chebyshev", "chebyshev_warped")
            and resolved_dipole_cutoff(cfg) in ("cosine", "cosine2"))


def _cell_ok(frames, rc) -> bool:
    for a in frames:
        cell = np.asarray(a.cell.array); L = np.diag(cell)
        if not (np.allclose(cell, np.diag(L)) and np.all(a.pbc) and np.all(L >= 2.0 * rc)):
            return False
    return True


def _radial(r, rc, C, radial, cutoff):
    """T_n(x)·f_c(r) → [P, C] (chebyshev / chebyshev_warped; cosine / cosine2)."""
    fc = (1.0 + tf.cos(np.pi * r / rc)) / 2.0
    if cutoff == "cosine2":
        fc = fc * fc
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0 if radial == "chebyshev_warped" else 2.0 * r / rc - 1.0
    T = [tf.ones_like(x), x]
    for _ in range(2, C):
        T.append(2.0 * x * T[-1] - T[-2])
    return tf.stack(T[:C], axis=1) * fc[:, None]


def _bond_basis(d, r, i_seg, rc, C, radial, cutoff, n_seg, rank, unit=False):
    """Σ_j R_n(r_ij)·g(d_ij) scattered to (b, i, t_j) segments → [n_seg, D, G·C]."""
    Rw = _radial(r, rc, C, radial, cutoff)                            # [P, C]
    if unit:
        d = d / r[:, None]
    if rank == 1:
        g = d[:, :, None]                                             # [P, 3, 1]
    else:
        x, y, z = d[:, 0], d[:, 1], d[:, 2]; r2 = x * x + y * y + z * z; zero = tf.zeros_like(r2)
        iso = tf.stack([r2, r2, r2, zero, zero, zero], axis=1)
        tr = tf.stack([x * x, y * y, z * z, x * y, y * z, z * x], axis=1) - iso / 3.0
        g = tf.stack([iso, tr], axis=2)                               # [P, 6, 2]
    G = tf.reshape(g[:, :, :, None] * Rw[:, None, None, :], (tf.shape(d)[0], g.shape[1], g.shape[2] * C))   # [P, D, G·C]
    return tf.math.unsorted_segment_sum(G, i_seg, n_seg)


@tf.function(reduce_retracing=True)
def _forward(pos, cell_L, ti, soap, sparse, A, params):
    """pos [B, N, 3], cell_L [B, 3], ti [N] (same composition in every frame),
    soap [B, N, Q]; sparse: list per species [M_t, Q]; A: list per species
    [M_t, K]; params: dict of python scalars → α [B, 6]."""
    B, N = tf.shape(pos)[0], tf.shape(pos)[1]
    rc, T = params["rc"], params["T"]
    # ── all-pairs minimum-image neighbour list for orthorhombic cells ──
    D = pos[:, :, None, :] - pos[:, None, :, :]                       # r_i − r_j, [B, N, N, 3]
    D = D - cell_L[:, None, None, :] * tf.round(D / cell_L[:, None, None, :])
    r2 = tf.reduce_sum(D * D, axis=-1)
    mask = (r2 < rc * rc) & (r2 > 0.0)
    idx = tf.where(mask)                                              # [P, 3] = (b, i, j)
    d = tf.gather_nd(D, idx); r = tf.sqrt(tf.gather_nd(r2, idx))
    b_i = idx[:, 0] * tf.cast(N, tf.int64) + idx[:, 1]
    seg = b_i * T + tf.cast(tf.gather(ti, idx[:, 2]), tf.int64)       # (b, i, t_j)
    n_seg = tf.cast(B * N, tf.int64) * T
    # ── bond basis [B·N, 6, Ck·T] (k = (group·C + n, t′), C-major) and cross vectors [B·N, 3, C_x·T] ──
    C, Cx = params["C"], params["Cx"]
    V = _bond_basis(d, r, seg, rc, C, params["radial"], params["cutoff"], n_seg, 2, params["unit"])
    V = tf.reshape(tf.transpose(tf.reshape(V, (B * N, T, 6, 2 * C)), (0, 2, 3, 1)), (B * N, 6, 2 * C * T))
    cols = [V]
    if Cx > 0:
        v1 = _bond_basis(d, r, seg, rc, Cx, params["radial"], params["cutoff"], n_seg, 1, params["unit"])
        v1 = tf.reshape(tf.transpose(tf.reshape(v1, (B * N, T, 3, Cx)), (0, 2, 3, 1)), (B * N, 3, Cx * T))
        a_idx, b_idx = params["tri"]
        a, bb = tf.gather(v1, a_idx, axis=2), tf.gather(v1, b_idx, axis=2)            # [B·N, 3, Kx]
        sym = 0.5 * (a[:, :, None, :] * bb[:, None, :, :] + bb[:, :, None, :] * a[:, None, :, :])
        tr3 = (sym[:, 0, 0] + sym[:, 1, 1] + sym[:, 2, 2]) / 3.0
        cross = tf.stack([sym[:, 0, 0] - tr3, sym[:, 1, 1] - tr3, sym[:, 2, 2] - tr3,
                          sym[:, 0, 1], sym[:, 1, 2], sym[:, 2, 0]], axis=1)
        cols.append(cross)
    v = tf.reshape(tf.concat(cols, axis=2), (B, N, 6, params["K"]))    # [B, N, 6, K]
    # ── kernel + readout per species: c[b,n,k] = δ²·(D·D_Mᵀ)^ξ A, α[b,x] = Σ_{n,k} v c ──
    alpha = tf.zeros((B, 6), dtype=soap.dtype)
    for t in range(T):
        sel = tf.cast(tf.equal(ti, t), soap.dtype)                      # [N]
        G = tf.matmul(soap, sparse[t], transpose_b=True)              # [B, N, M_t]
        c = params["delta2"] * tf.matmul(G ** params["xi"], A[t])    # [B, N, K]
        alpha += tf.einsum("bnxk,bnk,n->bx", v, c, sel)
    return alpha


def predict_gpu(model, frames, batch: int = 8) -> np.ndarray:
    """α [F, 6] for frames (same composition and atom order), batched on the GPU."""
    cfg = model.cfg
    import data as tnep_data
    rc = resolved_dipole_rc(cfg)
    tis = [np.asarray(t) for t in tnep_data.assign_type_indices(frames, cfg.types)]
    assert all(np.array_equal(t, tis[0]) for t in tis), "predict_gpu: frames must share composition and atom order"
    assert _cell_ok(frames, rc), "predict_gpu: orthorhombic MIC-safe periodic cells only"
    T = int(cfg.num_types); K1 = int(cfg.gap_cross_channels) * T
    params = dict(rc=rc, T=T, C=int(cfg.dipole_channels), Cx=int(cfg.gap_cross_channels),
                  radial=str(cfg.dipole_radial), cutoff=resolved_dipole_cutoff(cfg),
                  delta2=(float(cfg.gap_delta_soap) * float(cfg.gap_model_delta["soap"])) ** 2,   # as _site_terms
                  xi=int(model.xi), tri=tuple(np.triu_indices(K1)),
                  K=2 * int(cfg.dipole_channels) * T + K1 * (K1 + 1) // 2,
                  unit=bool(getattr(cfg, "dipole_unit_vector", False)))
    sparse = [tf.constant(np.asarray(s, np.float64)) for s in model.site_sparse["soap"]]
    A = []
    for t, (s, a) in enumerate(zip(sparse, model.site_alpha["soap"])):
        A.append(tf.constant(np.asarray(a, np.float64).reshape(int(s.shape[0]), -1)))      # (m, k)
    ti = tf.constant(tis[0], tf.int32)
    out = []
    for s in range(0, len(frames), batch):
        chunk = frames[s:s + batch]
        bf = min(len(chunk), int(getattr(cfg, "descriptor_batch_frames", None) or 4))   # pair tensors are complex128: ~1 GB per 4 bulk frames
        soap = [f[0] for f in model.builder.build_descriptors_flat(chunk, calc_gradients=False, use_tf_nl=True,
                                                                   return_tf=True, batch_frames=bf)]
        soap = tf.stack([tf.cast(tf.convert_to_tensor(x), tf.float64) for x in soap])          # [B, N, Q]
        pos = tf.constant(np.stack([a.positions for a in chunk]), tf.float64)
        L = tf.constant(np.stack([np.diag(a.cell.array) for a in chunk]), tf.float64)
        out.append(_forward(pos, L, ti, soap, sparse, A, params).numpy().astype(np.float64))
    return np.concatenate(out)
