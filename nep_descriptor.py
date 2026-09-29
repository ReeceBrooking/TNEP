"""GPUMD NEP descriptors, exactly (multi_descriptor_plan.md §0.3).

NumPy reference for the per-atom raw basis and the GPUMD contraction.
Constants and formulas are transcribed VERBATIM from GPUMD commit fee6340c,
/home/reece/GPUMD/src/utilities/nep_utilities.cuh (C3B/C4B/C5B lines 19-46,
Z_COEFFICIENT_L 48-65, find_fc 339-347, find_fn 500-517, accumulate_s_one
846-895, find_q_one 930-942, find_q 944-1000) and src/main_nep/nep.cu
(neighbour list 60-100: x12 = x[n2] − x[n1]; radial 110-155; angular 157-215).

Layout per atom (float64 inside, float32 out):
    raw_r[i, t', k]     = Σ_{j∈t', r<rc_r} f_k(r_ij; rc_r)                  [T, K_r+1]
    raw_a[i, t', k, c]  = Σ_{j∈t', r<rc_a} f_k(r_ij; rc_a) · P_c(r̂_ij)      [T, K_a+1, 24]
    nep_raw = flatten([raw_r ‖ raw_a])
Contraction (GPUMD nep.cu, c tensors learned in-model):
    q_r[i, n]  = Σ_{t',k} c_r[t_i, t', n, k] · raw_r[i, t', k]
    s[i, n, c] = Σ_{t',k} c_a[t_i, t', n, k] · raw_a[i, t', k, c]
    q_a = find_q(s)  → [L=1..4, (4-body), (5-body)] per n
Descriptor order (parameters.cu:203-211): [q_r] ‖ angular l-major
(ln = l·(n_max_a+1) + n) ‖ 4-body ‖ 5-body.
"""
import numpy as np
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


# ── basis (find_fc, find_fn) ───────────────────────────────────────────────
def fc(r, rc):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r < rc, 0.5 * np.cos(np.pi * r / rc) + 0.5, 0.0)


def fn_all(r, rc, K):
    """[P, K+1]: f_0 = fc, f_k = (T_k(x) + 1)·fc/2 with x = 2(r/rc − 1)² − 1."""
    r = np.asarray(r, dtype=np.float64)
    fc12 = fc(r, rc)
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    out = np.empty((r.shape[0], K + 1), dtype=np.float64)
    out[:, 0] = fc12
    t_m2, t_m1 = np.ones_like(x), x
    for k in range(1, K + 1):
        if k >= 2:
            t_m2, t_m1 = t_m1, 2.0 * x * t_m1 - t_m2
        out[:, k] = (t_m1 + 1.0) * 0.5 * fc12
    return out


# ── angular polynomials (accumulate_s_one<L> with fn = 1) ──────────────────
def real_harmonics(xhat, yhat, zhat):
    """[P, 24]: P_c(r̂) in GPUMD s-index order (L=1: 0..2, L=2: 3..7, L=3: 8..14, L=4: 15..23)."""
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (xhat, yhat, zhat))
    out = np.empty((x.shape[0], NUM_ABC), dtype=np.float64)
    for L in range(1, 5):
        s_index = L * L - 1
        z_pow = [np.ones_like(z)]
        for n in range(1, L + 1):
            z_pow.append(z * z_pow[n - 1])
        real_part, imag_part = x.copy(), y.copy()
        for n1 in range(L + 1):
            n2_start = 0 if (L + n1) % 2 == 0 else 1
            z_factor = np.zeros_like(z)
            for n2 in range(n2_start, L - n1 + 1, 2):
                z_factor = z_factor + Z_COEFFICIENT[L][n1][n2] * z_pow[n2]
            if n1 == 0:
                out[:, s_index] = z_factor; s_index += 1
            else:
                out[:, s_index] = z_factor * real_part; s_index += 1
                out[:, s_index] = z_factor * imag_part; s_index += 1
                real_part, imag_part = (x * real_part - y * imag_part,
                                        x * imag_part + y * real_part)   # complex_product
    return out


# ── invariant forms (find_q_one<L>, find_q) ────────────────────────────────
def find_q(s, four_body, five_body):
    """s [..., 24] → q [..., 4 (+1) (+1)] in GPUMD order L=1..4, 4-body, 5-body."""
    s = np.asarray(s, dtype=np.float64)
    q = []
    for L in range(1, 5):
        i0 = L * L - 1
        q.append(C3B[i0] * s[..., i0] ** 2
                 + 2.0 * np.sum(C3B[i0 + 1:i0 + 2 * L + 1] * s[..., i0 + 1:i0 + 2 * L + 1] ** 2, -1))
    if four_body:
        s3, s4, s5, s6, s7 = (s[..., k] for k in range(3, 8))
        q.append(C4B[0] * s3 * s3 * s3 + C4B[1] * s3 * (s4 * s4 + s5 * s5)
                 + C4B[2] * s3 * (s6 * s6 + s7 * s7) + C4B[3] * s6 * (s5 * s5 - s4 * s4)
                 + C4B[4] * s4 * s5 * s7)
    if five_body:
        s0_sq = s[..., 0] ** 2
        s12 = s[..., 1] ** 2 + s[..., 2] ** 2
        q.append(C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s12 + C5B[2] * s12 * s12)
    return np.stack(q, -1)


# ── bookkeeping ────────────────────────────────────────────────────────────
def validate_nep_cfg(cfg):
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


def _num_L(cfg):
    return 4 + int(bool(cfg.nep_four_body)) + int(bool(cfg.nep_five_body))


def nep_dim(cfg):
    return (cfg.nep_n_max_radial + 1) + (cfg.nep_n_max_angular + 1) * _num_L(cfg)


def nep_raw_width(cfg):
    T = int(cfg.num_types)
    return T * (cfg.nep_basis_size_radial + 1) + T * (cfg.nep_basis_size_angular + 1) * NUM_ABC


def nep_slices(cfg):
    """Column ranges into the dim_nep vector: "radial", "angular" (all of it),
    and its sub-blocks "L1".."L4", "four_body", "five_body" (when enabled)."""
    nr, na = cfg.nep_n_max_radial + 1, cfg.nep_n_max_angular + 1
    names = ["L1", "L2", "L3", "L4"] + (["four_body"] if cfg.nep_four_body else []) \
        + (["five_body"] if cfg.nep_five_body else [])
    out = {"radial": slice(0, nr), "angular": slice(nr, nep_dim(cfg))}
    out.update({k: slice(nr + l * na, nr + (l + 1) * na) for l, k in enumerate(names)})
    return out


# ── raw builder ────────────────────────────────────────────────────────────
def nep_raw_numpy(atoms, types_int, cfg):
    """[N, W_raw] float32 = flatten([raw_r [N,T,K_r+1] ‖ raw_a [N,T,K_a+1,24]])."""
    N, T = len(atoms), int(cfg.num_types)
    Kr, Ka = int(cfg.nep_basis_size_radial), int(cfg.nep_basis_size_angular)
    rc_r, rc_a = float(cfg.nep_rc_radial), float(cfg.nep_rc_angular)
    types_int = np.asarray(types_int)
    i, j, D = neighbor_list("ijD", atoms, max(rc_r, rc_a))   # D = r_j − r_i (+image)
    r = np.linalg.norm(D, axis=1)
    keep = r > 0.0
    i, j, D, r = i[keep], j[keep], D[keep], r[keep]
    tj = types_int[j]

    raw_r = np.zeros((N, T, Kr + 1), dtype=np.float64)
    m = r < rc_r
    np.add.at(raw_r, (i[m], tj[m]), fn_all(r[m], rc_r, Kr))

    raw_a = np.zeros((N, T, Ka + 1, NUM_ABC), dtype=np.float64)
    m = r < rc_a
    u = D[m] / r[m][:, None]
    P = real_harmonics(u[:, 0], u[:, 1], u[:, 2])
    np.add.at(raw_a, (i[m], tj[m]), fn_all(r[m], rc_a, Ka)[:, :, None] * P[:, None, :])

    return np.concatenate([raw_r.reshape(N, -1), raw_a.reshape(N, -1)], 1).astype(np.float32)


def nep_contract_numpy(raw, types_int, c_r, c_a, cfg):
    """[N, dim_nep]: GPUMD's contraction of the raw basis with c_r [T,T,n_r+1,K_r+1],
    c_a [T,T,n_a+1,K_a+1] (index [t_centre, t_neighbour, n, k]). Reference for the TF twin."""
    N, T = raw.shape[0], int(cfg.num_types)
    Kr, Ka = int(cfg.nep_basis_size_radial), int(cfg.nep_basis_size_angular)
    raw = np.asarray(raw, dtype=np.float64)
    types_int = np.asarray(types_int)
    wr = T * (Kr + 1)
    raw_r = raw[:, :wr].reshape(N, T, Kr + 1)
    raw_a = raw[:, wr:].reshape(N, T, Ka + 1, NUM_ABC)
    c_r = np.asarray(c_r, dtype=np.float64)[types_int]          # [N, T, n, k]
    c_a = np.asarray(c_a, dtype=np.float64)[types_int]
    q_r = np.einsum("atk,atnk->an", raw_r, c_r)
    s = np.einsum("atkc,atnk->anc", raw_a, c_a)                  # [N, n, 24]
    q_a = find_q(s, cfg.nep_four_body, cfg.nep_five_body)        # [N, n, num_L]
    return np.concatenate([q_r, q_a.transpose(0, 2, 1).reshape(N, -1)], 1)   # l-major


# ── nep.txt reader ─────────────────────────────────────────────────────────
def read_nep_txt(path):
    """Parse a GPUMD nep.txt → dict(species, nep_* cfg fields, ann [count],
    c_r [T,T,n_r+1,K_r+1], c_a [T,T,n_a+1,K_a+1], q_scaler [dim])."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    head = {}
    n_head = 0
    for ln in lines:
        if ln[0].startswith("nep"):
            head["species"] = ln[2:2 + int(ln[1])]
        elif ln[0] in ("cutoff", "n_max", "basis_size", "l_max", "ANN"):
            head[ln[0]] = [float(v) for v in ln[1:]]
        else:
            try:
                float(ln[0])
            except ValueError:
                raise ValueError(f"{path}: unexpected header line {ln!r}") from None
            break
        n_head += 1
    vals = np.array([float(ln[0]) for ln in lines[n_head:]], dtype=np.float64)
    T = len(head["species"])
    n_r, n_a = (int(v) for v in head["n_max"])
    K_r, K_a = (int(v) for v in head["basis_size"])
    l_max = [int(v) for v in head["l_max"]]
    out = dict(
        species=head["species"],
        nep_rc_radial=head["cutoff"][0], nep_rc_angular=head["cutoff"][1],
        nep_n_max_radial=n_r, nep_n_max_angular=n_a,
        nep_basis_size_radial=K_r, nep_basis_size_angular=K_a,
        nep_l_max=l_max[0],
        nep_four_body=len(l_max) > 1 and l_max[1] == 2,
        nep_five_body=len(l_max) > 2 and l_max[2] == 1,
        nep_num_neurons=int(head["ANN"][0]),
    )
    num_L = 4 + out["nep_four_body"] + out["nep_five_body"]
    dim = (n_r + 1) + (n_a + 1) * num_L
    n_cr, n_ca = T * T * (n_r + 1) * (K_r + 1), T * T * (n_a + 1) * (K_a + 1)
    n_ann = vals.shape[0] - n_cr - n_ca - dim
    if n_ann <= 0:
        raise ValueError(f"{path}: {vals.shape[0]} parameters < c ({n_cr + n_ca}) + q_scaler ({dim}).")
    # GPUMD c index: (n*(K+1)+k)*T² + t1*T + t2  →  [n, k, t1, t2] → [t1, t2, n, k]
    c_r = vals[n_ann:n_ann + n_cr].reshape(n_r + 1, K_r + 1, T, T).transpose(2, 3, 0, 1)
    c_a = vals[n_ann + n_cr:n_ann + n_cr + n_ca].reshape(n_a + 1, K_a + 1, T, T).transpose(2, 3, 0, 1)
    out.update(ann=vals[:n_ann], c_r=np.ascontiguousarray(c_r),
               c_a=np.ascontiguousarray(c_a), q_scaler=vals[-dim:], dim=dim)
    return out
