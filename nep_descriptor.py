"""GPUMD NEP descriptors, exactly: per-atom raw basis + the GPUMD contraction.

Constants and formulas are transcribed verbatim from GPUMD commit fee6340c,
src/utilities/nep_utilities.cuh (C3B/C4B/C5B 19-46, Z_COEFFICIENT_L 48-65,
find_fc 339-347, find_fc_and_fcp 349-361, find_fn 500-517, find_fn_and_fnp
519-552, accumulate_s_one 846-895, find_q_one 930-942, find_q 944-1000) and
src/main_nep/nep.cu (neighbour list 60-100: x12 = x[n2] − x[n1]; radial
110-155; angular 157-215).

Layout per atom (float64 inside, float32 out):
    raw_r[i, t', k]     = Σ_{j∈t', r<rc_r} f_k(r_ij; rc_r)                  [T, K_r+1]
    raw_a[i, t', k, c]  = Σ_{j∈t', r<rc_a} f_k(r_ij; rc_a) · P_c(r̂_ij)      [T, K_a+1, 24]
    nep_raw = flatten([raw_r ‖ raw_a])
Contraction (GPUMD nep.cu, c tensors learned in-model):
    q_r[i, n]  = Σ_{t',k} c_r[t_i, t', n, k] · raw_r[i, t', k]
    s[i, n, c] = Σ_{t',k} c_a[t_i, t', n, k] · raw_a[i, t', k, c]
    q_a = find_q_tf(s)  → [L=1..4, (4-body), (5-body)] per n
Descriptor order (parameters.cu:203-211): [q_r] ‖ angular l-major
(ln = l·(n_max_a+1) + n) ‖ 4-body ‖ 5-body.
"""
import numpy as np
import tensorflow as tf
from ase.neighborlist import neighbor_list

# ── constants (nep_utilities.cuh) ──────────────────────────────────────────
C3B = np.array([
    0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
    0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
    1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
    0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
    0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
    2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
    2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
    0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
    0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
    2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
    0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
    0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
    0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
    3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
    0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
    0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
    5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
    4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816],
    dtype=np.float64)
C4B = np.array([-0.007499480826664, -0.134990654879954, 0.067495327439977,
                0.404971964639861, -0.809943929279723], dtype=np.float64)
C5B = np.array([0.026596810706114, 0.053193621412227, 0.026596810706114],
               dtype=np.float64)
Z_COEFFICIENT = {
    1: [[0.0, 1.0], [1.0, 0.0]],
    2: [[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    3: [[0.0, -3.0, 0.0, 5.0], [-1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    4: [[3.0, 0.0, -30.0, 0.0, 35.0], [0.0, -3.0, 0.0, 7.0, 0.0],
        [-1.0, 0.0, 7.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]],
}
NUM_ABC = 24   # 3 + 5 + 7 + 9 for L = 1..4


# ── basis (find_fc, find_fn, find_fc_and_fcp, find_fn_and_fnp) ─────────────
# Bodies are shared between NumPy (float64) and TF (float32): `xp` is np or tf.
def fc(r, rc):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r < rc, 0.5 * np.cos(np.pi * r / rc) + 0.5, 0.0)


def _fn_and_fnp(r, rc, K, xp):
    """(fn [P,K+1], fnp [P,K+1]) = find_fc_and_fcp (nep_utilities.cuh:349-361) +
    find_fn_and_fnp (519-552): f_0 = fc, f_k = (T_k(x)+1)·fc/2, x = 2(r/rc−1)²−1,
    derivative through T_k' = k·U_{k−1} (Chebyshev U recursion u0,u1 as in GPUMD)."""
    inside = r < rc
    fc12 = xp.where(inside, 0.5 * xp.cos(np.pi * r / rc) + 0.5, 0.0)
    fcp12 = xp.where(inside, -0.5 * np.pi * xp.sin(np.pi * r / rc) / rc, 0.0)
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    dx = 2.0 * (r / rc - 1.0) / rc                 # GPUMD: 2(d·rcinv − 1)·rcinv
    fn, fnp = [fc12], [fcp12]
    if K >= 1:
        fn.append((x + 1.0) * 0.5 * fc12)
        fnp.append(dx * fc12 + (x + 1.0) * 0.5 * fcp12)
    u0, u1 = xp.ones_like(x), 2.0 * x               # U_0, U_1
    t_m2, t_m1 = xp.ones_like(x), x                 # T_0, T_1
    for m in range(2, K + 1):
        t_m2, t_m1 = t_m1, 2.0 * x * t_m1 - t_m2
        fnp_tmp = m * u1                            # T_m' = m·U_{m−1}
        u0, u1 = u1, 2.0 * x * u1 - u0
        half = (t_m1 + 1.0) * 0.5
        fnp.append(fnp_tmp * dx * fc12 + half * fcp12)
        fn.append(half * fc12)
    return xp.stack(fn, 1), xp.stack(fnp, 1)


def fn_and_fnp(r, rc, K):
    """NumPy float64: (fn [P,K+1], fnp [P,K+1] = ∂fn/∂r)."""
    return _fn_and_fnp(np.asarray(r, dtype=np.float64), float(rc), int(K), np)


def fn_and_fnp_tf(r, rc, K):
    """TF float32 twin of fn_and_fnp."""
    return _fn_and_fnp(tf.cast(r, tf.float32), float(rc), int(K), tf)


def fn_all(r, rc, K):
    """[P, K+1]: f_0 = fc, f_k = (T_k(x) + 1)·fc/2 with x = 2(r/rc − 1)² − 1."""
    return fn_and_fnp(r, rc, K)[0]


# ── angular polynomials (accumulate_s_one<L> with fn = 1) ──────────────────
def _real_harmonics(x, y, z, xp):
    """(P [P,24], dP [P,3,24]): P_c(r̂) in GPUMD s-index order (L=1: 0..2, L=2: 3..7,
    L=3: 8..14, L=4: 15..23) and ∂P_c/∂(x̂, ŷ, ẑ) with the unit-vector components as
    FREE variables (the caller applies (I − r̂r̂ᵀ)/r for a Cartesian derivative)."""
    one, zero = xp.ones_like(z), xp.zeros_like(z)
    P, dP = [], []
    for L in range(1, 5):
        z_pow = [one]
        for n in range(1, L + 1):
            z_pow.append(z * z_pow[n - 1])
        re, im = x, y                               # (x + iy)^n1
        re_p, im_p = one, zero                      # (x + iy)^(n1 − 1)
        for n1 in range(L + 1):
            n2_start = 0 if (L + n1) % 2 == 0 else 1
            zf, dzf = zero, zero
            for n2 in range(n2_start, L - n1 + 1, 2):
                zf = zf + Z_COEFFICIENT[L][n1][n2] * z_pow[n2]
                if n2 >= 1:
                    dzf = dzf + n2 * Z_COEFFICIENT[L][n1][n2] * z_pow[n2 - 1]
            if n1 == 0:
                P.append(zf); dP.append((zero, zero, dzf))
            else:
                P.append(zf * re); dP.append((n1 * zf * re_p, -n1 * zf * im_p, dzf * re))
                P.append(zf * im); dP.append((n1 * zf * im_p, n1 * zf * re_p, dzf * im))
                re_p, im_p = re, im
                re, im = x * re - y * im, x * im + y * re   # complex_product
    return xp.stack(P, 1), xp.stack([xp.stack(c, 1) for c in dP], -1)


def _np64(*v):
    return [np.asarray(a, dtype=np.float64) for a in v]


def _tf32(*v):
    return [tf.cast(a, tf.float32) for a in v]


def real_harmonics(xhat, yhat, zhat):
    """[P, 24] float64: P_c(r̂)."""
    return _real_harmonics(*_np64(xhat, yhat, zhat), np)[0]


def real_harmonics_tf(xhat, yhat, zhat):
    return _real_harmonics(*_tf32(xhat, yhat, zhat), tf)[0]


def real_harmonics_grad_tf(xhat, yhat, zhat):
    return _real_harmonics(*_tf32(xhat, yhat, zhat), tf)[1]


_C3B_TF = C3B.astype(np.float32)   # NumPy: TF ops accept them; no eager context at import
_C4B_TF = C4B.astype(np.float32)
_C5B_TF = C5B.astype(np.float32)


def find_q_tf(s, four_body, five_body):
    """s [..., 24] float32 → q [..., 4 (+1) (+1)] in GPUMD order L=1..4, 4-body, 5-body."""
    q = []
    for L in range(1, 5):
        i0 = L * L - 1
        q.append(_C3B_TF[i0] * s[..., i0] ** 2
                 + 2.0 * tf.reduce_sum(_C3B_TF[i0 + 1:i0 + 2 * L + 1] * s[..., i0 + 1:i0 + 2 * L + 1] ** 2, -1))
    if four_body:
        s3, s4, s5, s6, s7 = (s[..., k] for k in range(3, 8))
        q.append(_C4B_TF[0] * s3 * s3 * s3 + _C4B_TF[1] * s3 * (s4 * s4 + s5 * s5)
                 + _C4B_TF[2] * s3 * (s6 * s6 + s7 * s7) + _C4B_TF[3] * s6 * (s5 * s5 - s4 * s4)
                 + _C4B_TF[4] * s4 * s5 * s7)
    if five_body:
        s0_sq = s[..., 0] ** 2
        s12 = s[..., 1] ** 2 + s[..., 2] ** 2
        q.append(_C5B_TF[0] * s0_sq * s0_sq + _C5B_TF[1] * s0_sq * s12 + _C5B_TF[2] * s12 * s12)
    return tf.stack(q, -1)


def q3b_from_gram_tf(G, c_a, Zf, num_types, p_idx, q_idx, n_pairs):
    """The L = 1..4 block from the data-static Gram G [N, 4·P] (see nep_gram_on):
    q[c, a, L, n] = Σ_{p≤q} c_a[c, t_a, p, n]·c_a[c, t_a, q, n]·G[a, L, pq],
    one GEMM per centre type over the packed pair axis (no [C, N, n, 24] intermediate).
    c_a is [T, T, n+1, K+1] or per-candidate [C, T, T, n+1, K+1]; p_idx/q_idx from
    gram_pairs. Returns [(C,) N, 4, n+1], l-major like find_q_tf's output."""
    per_cand = len(c_a.shape) == 5
    if not per_cand:
        c_a = c_a[tf.newaxis]
    T, na = int(num_types), int(c_a.shape[3])
    D = int(c_a.shape[2]) * int(c_a.shape[4])
    # c_a [C, T_c, T, n, K] → [C, T_c, D, n] with D = (t\u2032, k) flattened.
    cc = tf.reshape(tf.transpose(c_a, [0, 1, 2, 4, 3]), [-1, T, D, na])
    cp = (tf.gather(cc, p_idx, axis=2) * tf.gather(cc, q_idx, axis=2))  # [C,T_c,P,n]
    N = tf.shape(G)[0]
    Gf = tf.reshape(G, [N * 4, n_pairs])
    sel = tf.one_hot(Zf, T, dtype=G.dtype)                              # [N, T]
    acc = None
    for t in range(T):
        rhs = tf.reshape(tf.transpose(cp[:, t], [1, 0, 2]), [n_pairs, -1])
        qt = tf.reshape(tf.matmul(Gf, rhs), [N, 4, -1, na])             # [N,4,C,n]
        qt = qt * sel[:, tf.newaxis, t, tf.newaxis, tf.newaxis]
        acc = qt if acc is None else acc + qt
    out = tf.transpose(acc, [2, 0, 1, 3])                               # [C,N,4,n]
    return out if per_cand else out[0]


# ── bookkeeping ────────────────────────────────────────────────────────────
# nep_descriptor="sum": the 2-body (radial) and 3-body (angular) expert ANN
# weights, in SNES / Adam / save order.
EXPERT_WEIGHTS = ("W0_r", "b0_r", "W1_r", "W0_a", "b0_a", "W1_a")


def validate_nep_cfg(cfg):
    """Every NEP cfg guard in one place; returns cfg.nep_descriptor."""
    if int(cfg.nep_l_max) != 4:
        raise ValueError(f"cfg.nep_l_max={cfg.nep_l_max}: GPUMD supports only 4.")
    if cfg.nep_five_body and not cfg.nep_four_body:
        raise ValueError("cfg.nep_five_body requires nep_four_body (GPUMD find_q order).")
    if not (float(cfg.nep_rc_radial) > 0.0 and float(cfg.nep_rc_angular) > 0.0):
        raise ValueError("cfg.nep_rc_radial / nep_rc_angular must be > 0.")
    for k in ("nep_n_max_radial", "nep_n_max_angular",
              "nep_basis_size_radial", "nep_basis_size_angular"):
        if int(getattr(cfg, k)) < 0:
            raise ValueError(f"cfg.{k} must be >= 0.")
    mode = str(getattr(cfg, "nep_descriptor", "off"))
    if mode not in ("off", "concat", "replace", "sum"):
        raise ValueError(
            f"cfg.nep_descriptor={mode!r}; expected 'off', 'concat', 'replace' or 'sum'.")
    if mode == "off":
        return mode
    from data import descriptor_gradients_needed   # lazy: data imports nep_descriptor
    if descriptor_gradients_needed(cfg):
        # NEP pair forces exist for the tnep dipole readout only; PES / polarisability
        # need both-atom forces and virials, "sum" would need de_dq through the experts.
        if not (int(cfg.target_mode) == 1 and str(cfg.dipole_readout) == "tnep"
                and mode in ("concat", "replace")):
            raise NotImplementedError(
                f"cfg.nep_descriptor={mode!r}: descriptor gradients are implemented "
                "only for target_mode=1 with dipole_readout='tnep' (concat/replace); "
                "the neighbor readout needs none.")
        if int(cfg.dipole_rij_power) < 1:
            raise ValueError(
                "cfg.dipole_rij_power=0 (self-pairs only) has no NEP pair stream; "
                "the faithful GPUMD tnep readout is dipole_rij_power=2.")
    if mode == "replace":
        if cfg.radial_basis_size is not None:
            raise ValueError(
                "cfg.nep_descriptor='replace' drops the SOAP output: "
                "radial_basis_size would ride the genome (M_radial) with its "
                "output discarded; set radial_basis_size=None.")
        if bool(getattr(cfg, "descriptor_mixing", False)):
            raise ValueError(
                "cfg.nep_descriptor='replace' has no SOAP species-pair blocks; "
                "set cfg.descriptor_mixing off.")
    return mode


def _num_L(cfg):
    return 4 + int(bool(cfg.nep_four_body)) + int(bool(cfg.nep_five_body))


def nep_dim(cfg):
    return (cfg.nep_n_max_radial + 1) + (cfg.nep_n_max_angular + 1) * _num_L(cfg)


def gpumd_num_params(cfg, soap_dim=0):
    """GPUMD's number_of_variables (parameters.cu:203-235): NEP4 typed ANNs, one hidden
    layer, no charge mode — ANN = T·(dim+2)·H + 1, c = T²·[(n_r+1)(K_r+1) + (n_a+1)(K_a+1)].
    Equals SNES.dim for "replace" + dipole_readout="tnep" with mixing / learned basis
    off (descriptor_scale is a zero-parameter W0 reparameterisation). soap_dim adds the
    SOAP columns of "concat" to the ANN input width."""
    T, H = int(cfg.num_types), int(cfg.num_neurons)
    ann = T * (nep_dim(cfg) + int(soap_dim) + 2) * H + 1
    c = T * T * ((cfg.nep_n_max_radial + 1) * (cfg.nep_basis_size_radial + 1)
                 + (cfg.nep_n_max_angular + 1) * (cfg.nep_basis_size_angular + 1))
    return ann + c


def nep_raw_width(cfg):
    """Width of data["nep_raw"], by nep_gram_on(cfg):
        gram off : raw_r [T, K_r+1] ‖ raw_a [T, K_a+1, 24]
        gram on  : raw_r [T, K_r+1] ‖ G [4, P]     (the 3-body Gram block)"""
    T = int(cfg.num_types)
    w_r = T * (cfg.nep_basis_size_radial + 1)
    if nep_gram_on(cfg):
        return w_r + 4 * nep_gram_n_pairs(cfg)
    return w_r + T * (cfg.nep_basis_size_angular + 1) * NUM_ABC


def nep_gram_dim(cfg):
    """D = T·(K_a+1), the angular contraction channels (t\u2032, k) that c_a sums."""
    return int(cfg.num_types) * (int(cfg.nep_basis_size_angular) + 1)


def gram_pairs(cfg):
    """(p, q) upper-triangular pairs over the D angular channels and the packing
    multiplicity (2 off-diagonal: G is symmetric)."""
    iu = np.triu_indices(nep_gram_dim(cfg))
    return iu[0], iu[1], np.where(iu[0] == iu[1], 1.0, 2.0)


def nep_gram_on(cfg):
    """True when the 3-body invariants come from a data-static Gram instead of the
    per-candidate s tensor. q[.., L] = Σ_m w_m·s[.., m]² is quadratic in s and s is
    linear in c_a, so

        q[c, a, n, L] = Σ_{p≤q} c_a[c, t_a, p, n]·c_a[c, t_a, q, n]·G[a, L, pq],
        G[a, L, pq]   = Σ_{m ∈ block L} w_m·raw_a[a, p, m]·raw_a[a, q, m],

    with G geometry-only: built once, one GEMM per centre type per candidate, no
    [C, N, n, 24] intermediate (~8× the Gram block; the whole NEP-path cost).
    Off for nep_four_body / nep_five_body (cubic / quartic in s, no Gram form) and
    for the tnep pair-force readout (needs s itself)."""
    mode = str(getattr(cfg, "nep_descriptor", "off"))
    if mode not in ("concat", "replace", "sum"):
        return False
    if bool(cfg.nep_four_body) or bool(cfg.nep_five_body):
        return False
    pair_forces = (mode in ("concat", "replace")
                   and int(cfg.target_mode) == 1
                   and str(getattr(cfg, "dipole_readout", "neighbor")) == "tnep")
    return not pair_forces


def nep_gram_n_pairs(cfg):
    """P = D(D+1)/2, the packed pair count of the Gram block."""
    D = nep_gram_dim(cfg)
    return D * (D + 1) // 2


def angular_gram_numpy(raw_a, cfg, dtype=np.float32):
    """G[a, L, pq] packed to [N, 4·P] from raw_a [N, D, 24]; the find_q weights
    (C3B, doubled off m = 0) and the off-diagonal 2 are folded in."""
    p, q, mult = gram_pairs(cfg)
    out = []
    for L in range(1, 5):
        i0 = L * L - 1
        w = np.full(2 * L + 1, 2.0); w[0] = 1.0
        blk = raw_a[:, :, i0:i0 + 2 * L + 1]
        G = np.einsum("apm,aqm->apq", blk * (w * C3B[i0:i0 + 2 * L + 1]), blk)
        out.append(G[:, p, q] * mult)
    return np.concatenate(out, 1).astype(dtype)


def nep_slices(cfg):
    """Column ranges into the dim_nep vector: "radial", "angular" (all of it),
    and its sub-blocks "L1".."L4", "four_body", "five_body" (when enabled)."""
    nr, na = cfg.nep_n_max_radial + 1, cfg.nep_n_max_angular + 1
    names = ["L1", "L2", "L3", "L4"] + (["four_body"] if cfg.nep_four_body else []) \
        + (["five_body"] if cfg.nep_five_body else [])
    out = {"radial": slice(0, nr), "angular": slice(nr, nep_dim(cfg))}
    out.update({k: slice(nr + l * na, nr + (l + 1) * na) for l, k in enumerate(names)})
    return out


# ── raw builder + pair stream ──────────────────────────────────────────────
def nep_pair_list(atoms, cfg):
    """(i, j, D) from one ase call at max(rc_r, rc_a): D = r_j − r_i with image offset
    (image-exact, NOT MIC: rc_r = 8 Å exceeds L/2 for the water box), zero-length self
    entries dropped, sorted by centre i."""
    i, j, D = neighbor_list("ijD", atoms, max(float(cfg.nep_rc_radial), float(cfg.nep_rc_angular)))
    keep = np.linalg.norm(D, axis=1) > 0.0
    return i[keep], j[keep], D[keep]


def nep_raw_numpy(atoms, types_int, cfg, pairs=None, dtype=np.float32):
    """[N, W_raw] float32 = flatten([raw_r ‖ raw_a]), or raw_r ‖ G when nep_gram_on(cfg)
    (see nep_raw_width). dtype=np.float64 keeps the float64 accumulation for FD tests."""
    N, T = len(atoms), int(cfg.num_types)
    Kr, Ka = int(cfg.nep_basis_size_radial), int(cfg.nep_basis_size_angular)
    rc_r, rc_a = float(cfg.nep_rc_radial), float(cfg.nep_rc_angular)
    types_int = np.asarray(types_int)
    i, j, D = nep_pair_list(atoms, cfg) if pairs is None else pairs
    r = np.linalg.norm(D, axis=1)
    tj = types_int[j]

    raw_r = np.zeros((N, T, Kr + 1), dtype=np.float64)
    m = r < rc_r
    np.add.at(raw_r, (i[m], tj[m]), fn_all(r[m], rc_r, Kr))

    raw_a = np.zeros((N, T, Ka + 1, NUM_ABC), dtype=np.float64)
    m = r < rc_a
    u = D[m] / r[m][:, None]
    P = real_harmonics(u[:, 0], u[:, 1], u[:, 2])
    np.add.at(raw_a, (i[m], tj[m]), fn_all(r[m], rc_a, Ka)[:, :, None] * P[:, None, :])

    if nep_gram_on(cfg):
        ang = angular_gram_numpy(raw_a.reshape(N, T * (Ka + 1), NUM_ABC), cfg,
                                 dtype=np.float64)
    else:
        ang = raw_a.reshape(N, -1)
    return np.concatenate([raw_r.reshape(N, -1), ang], 1).astype(dtype)


# Per-pair keys every forward site threads (data dict → predict_batch kwargs).
NEP_PAIR_KW = ("nep_pair_atom", "nep_pair_gidx", "nep_pair_struct", "nep_pair_D", "nep_pair_ang")


def nep_pair_kwargs(d):
    """{key: d.get(key)} for NEP_PAIR_KW — None values when the stream is absent."""
    return {k: d.get(k) for k in NEP_PAIR_KW}


def nep_pairs_numpy(atoms, cfg, pairs=None):
    """Image-exact pair stream for the force contraction, keyed as in the data dict:
    nep_pair_atom / nep_pair_gidx [P] int32, nep_pair_D [P,3] float32 (r_j − r_i incl.
    image), nep_pair_ang [P] bool (r < rc_a). nep_pair_ang comes from the float64 D and
    is AUTHORITATIVE: never re-derive r < rc_a from the float32 D in-graph."""
    i, j, D = nep_pair_list(atoms, cfg) if pairs is None else pairs
    return dict(nep_pair_atom=i.astype(np.int32), nep_pair_gidx=j.astype(np.int32),
                nep_pair_D=D.astype(np.float32),
                nep_pair_ang=np.linalg.norm(D, axis=1) < float(cfg.nep_rc_angular))


