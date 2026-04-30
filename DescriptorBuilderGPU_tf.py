"""TF GPU SOAP-turbo port (mixed precision: float64 internal, float32 output).

Lifts the validated NumPy backend in DescriptorBuilderGPU.py to TF so the
SOAP descriptor build runs on GPU. The algorithm is identical — only the
inner-loop bodies are switched to tf.* ops so that @tf.function can fold
them into a single GPU graph.

Strategy:
    - Reuse the NumPy CPU-only helpers (basis matrices, compress mask,
      multiplicity array, neighbour-list builder) — these run once per call,
      produce small constants, and would be awkward to port.
    - All hot per-pair compute (radial recursion, angular recursion,
      cnk scatter-sum, power spectrum, derivatives, Cartesian conversion)
      is rewritten in TF.
    - Internal precision is float64 / complex128 for stability; outputs
      are cast to float32 to match the existing trajectory pipeline.
    - A single @tf.function entry point compiles the whole pipeline; with
      reduce_retracing=True it caches across batches with varying P.

Validation: this file's run_phase11_validation() compares its output to
the NumPy reference for all 7 fixtures, both descriptors and gradients.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from ase import Atoms

# Reuse CPU-only helpers and NumPy reference from the main module
from DescriptorBuilderGPU import (
    REFERENCE_SOAP_PARAMS,
    build_orthonormalization_matrix_poly3,
    build_neighbour_list_numpy,
    build_multiplicity_array,
    make_compress_mask_trivial,
    _get_preflm,
    _N_a,
    load_fixture,
    compute_soap_with_grad_from_positions_numpy,
    _make_test_structures,
)


# =========================================================================
# Radial expansion (TF)
# =========================================================================


def _radial_first_integral_tf(
    rjs: tf.Tensor,                 # [P] float64 (normalised by rcut_hard)
    alpha_max: int,
    rcut_soft: float,               # normalised
    atom_sigma_scaled: tf.Tensor,   # [P] float64
) -> tf.Tensor:
    """First-integral recursion over [0, rcut_soft]. Returns [alpha_max, P]."""
    sq2 = tf.constant(np.sqrt(2.0), dtype=tf.float64)
    s2 = atom_sigma_scaled ** 2
    P = tf.shape(rjs)[0]

    I_n = tf.zeros_like(rjs)
    N_n = tf.constant(1.0, dtype=tf.float64)
    N_np1 = tf.constant(_N_a(-2), dtype=tf.float64)
    sqrt_pi_2 = tf.constant(np.sqrt(np.pi / 2.0), dtype=tf.float64)
    I_np1 = (sqrt_pi_2 * atom_sigma_scaled
             * (tf.math.erf((rcut_soft - rjs) / (sq2 * atom_sigma_scaled))
                - tf.math.erf((-rjs) / (sq2 * atom_sigma_scaled))) / N_np1)

    dr = 1.0 - rcut_soft
    if dr == 0.0:
        C1 = tf.zeros_like(rjs)
    else:
        C1 = s2 / dr * tf.math.exp(-0.5 * (rcut_soft - rjs) ** 2 / s2)
    C2 = s2 * tf.math.exp(-0.5 * rjs ** 2 / s2)

    out_list = []
    for n in range(-1, alpha_max + 1):
        C1 = C1 * dr
        N_np2 = tf.constant(_N_a(n), dtype=tf.float64)
        I_np2 = (s2 * (n + 1) * (N_n / N_np2) * I_n
                 - N_np1 * (rjs - 1.0) / N_np2 * I_np1
                 + C1 / N_np2
                 - C2 / N_np2)
        if n > 0:
            out_list.append(I_np2)
        N_n = N_np1
        N_np1 = N_np2
        I_n = I_np1
        I_np1 = I_np2
    return tf.stack(out_list, axis=0)


def _radial_second_integral_tf(
    rjs: tf.Tensor,
    alpha_max: int,
    rcut_soft: float,
    atom_sigma_scaled: tf.Tensor,
    nf: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Soft-cutoff filter contribution. Returns (temp2 [alpha_max, P], pref_f [P])."""
    sq2 = tf.constant(np.sqrt(2.0), dtype=tf.float64)
    s2 = atom_sigma_scaled ** 2
    dr = 1.0 - rcut_soft
    denom = s2 + dr ** 2 / nf ** 2
    atom_sigma_f = atom_sigma_scaled * dr / nf / tf.math.sqrt(denom)
    rj_f = (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs) / denom
    sf2 = atom_sigma_f ** 2
    pref_f = tf.math.exp(-0.5 * (rcut_soft - rjs) ** 2 / denom)

    I_n = tf.zeros_like(rjs)
    N_n = tf.constant(1.0, dtype=tf.float64)
    N_np1 = tf.constant(_N_a(-2), dtype=tf.float64)
    sqrt_pi_2 = tf.constant(np.sqrt(np.pi / 2.0), dtype=tf.float64)
    I_np1 = (sqrt_pi_2 * atom_sigma_f
             * (tf.math.erf((1.0 - rj_f) / (sq2 * atom_sigma_f))
                - tf.math.erf((rcut_soft - rj_f) / (sq2 * atom_sigma_f))) / N_np1)

    if dr == 0.0:
        C2 = tf.zeros_like(rjs)
    else:
        C2 = sf2 / dr * tf.math.exp(-0.5 * (rcut_soft - rj_f) ** 2 / sf2)

    out_list = []
    for n in range(-1, alpha_max + 1):
        C2 = C2 * dr
        N_np2 = tf.constant(_N_a(n), dtype=tf.float64)
        I_np2 = (sf2 * (n + 1) * (N_n / N_np2) * I_n
                 - N_np1 * (rj_f - 1.0) / N_np2 * I_np1
                 - C2 / N_np2)
        if n > 0:
            out_list.append(I_np2)
        N_n = N_np1
        N_np1 = N_np2
        I_n = I_np1
        I_np1 = I_np2
    return tf.stack(out_list, axis=0), pref_f


def _radial_amplitude_with_der_tf(
    rjs: tf.Tensor,
    atom_sigma_scaled: tf.Tensor,
    atom_sigma_scaling: float,
    is_central: tf.Tensor,           # [P] bool
    central_weight: float,
    amplitude_scaling: float,
    radial_enhancement: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Amplitude and its derivative. Mirrors the NumPy version exactly."""
    s2 = atom_sigma_scaled ** 2
    if amplitude_scaling == 0.0:
        amp = 1.0 / atom_sigma_scaled
        amp_der = -tf.constant(atom_sigma_scaling, dtype=tf.float64) / s2
    else:
        env = 1.0 + 2.0 * rjs ** 3 - 3.0 * rjs ** 2
        env_safe = tf.where(env > 1e-10, env, tf.ones_like(env))
        if amplitude_scaling == 1.0:
            amp_pre = (1.0 / atom_sigma_scaled) * env
            amp_der_pre = ((6.0 / atom_sigma_scaled) * (rjs ** 2 - rjs)
                           - (atom_sigma_scaling / atom_sigma_scaled) * amp_pre)
        else:
            amp_pre = (1.0 / atom_sigma_scaled) * env_safe ** amplitude_scaling
            amp_der_pre = ((6.0 * amplitude_scaling / atom_sigma_scaled) * (rjs ** 2 - rjs)
                           * env_safe ** (amplitude_scaling - 1.0)
                           - (atom_sigma_scaling / atom_sigma_scaled) * amp_pre)
        amp = tf.where(env > 1e-10, amp_pre, tf.zeros_like(amp_pre))
        amp_der = tf.where(env > 1e-10, amp_der_pre, tf.zeros_like(amp_der_pre))

    amp = tf.where(is_central, amp * central_weight, amp)
    amp_der = tf.where(is_central, amp_der * central_weight, amp_der)

    if radial_enhancement == 1:
        sqrt_2_pi = tf.constant(np.sqrt(2.0 / np.pi), dtype=tf.float64)
        amp_der_new = (amp * (1.0 + sqrt_2_pi * atom_sigma_scaling)
                       + amp_der * (rjs + sqrt_2_pi * atom_sigma_scaled))
        amp_new = amp * (rjs + sqrt_2_pi * atom_sigma_scaled)
        amp, amp_der = amp_new, amp_der_new
    elif radial_enhancement == 2:
        sqrt_8_pi = tf.constant(np.sqrt(8.0 / np.pi), dtype=tf.float64)
        amp_der_new = (amp * (2.0 * rjs + 2.0 * atom_sigma_scaled * atom_sigma_scaling
                              + sqrt_8_pi * atom_sigma_scaled
                              + sqrt_8_pi * rjs * atom_sigma_scaling)
                       + amp_der * (rjs ** 2 + atom_sigma_scaled ** 2
                                    + sqrt_8_pi * atom_sigma_scaled * rjs))
        amp_new = amp * (rjs ** 2 + atom_sigma_scaled ** 2
                         + sqrt_8_pi * atom_sigma_scaled * rjs)
        amp, amp_der = amp_new, amp_der_new
    return amp, amp_der


def _radial_first_integral_der_tf(
    rjs: tf.Tensor,
    temp1_ext: tf.Tensor,            # [alpha_max+2, P]
    alpha_max: int,
    atom_sigma_scaled: tf.Tensor,
    atom_sigma_scaling: float,
) -> tf.Tensor:
    """Chain-rule derivative of first integral. Returns [alpha_max, P]."""
    s2 = atom_sigma_scaled ** 2
    rj_minus = rjs - 1.0
    sigma_term = atom_sigma_scaling * rj_minus / atom_sigma_scaled
    out_list = []
    for n_f in range(1, alpha_max + 1):
        Na_n = tf.constant(_N_a(n_f), dtype=tf.float64)
        Na_np1 = tf.constant(_N_a(n_f + 1), dtype=tf.float64)
        Na_np2 = tf.constant(_N_a(n_f + 2), dtype=tf.float64)
        py = n_f - 1
        out_list.append(
            rj_minus / s2 * (sigma_term - 1.0) * temp1_ext[py]
            + Na_np1 / s2 / Na_n * (2.0 * sigma_term - 1.0) * temp1_ext[py + 1]
            + atom_sigma_scaling * Na_np2 / atom_sigma_scaled ** 3 / Na_n * temp1_ext[py + 2]
        )
    return tf.stack(out_list, axis=0)


def _radial_second_integral_der_tf(
    rjs: tf.Tensor,
    temp2_ext: tf.Tensor,            # [alpha_max+2, P]
    alpha_max: int,
    rcut_soft: float,
    atom_sigma_scaled: tf.Tensor,
    atom_sigma_scaling: float,
    nf: float,
) -> tf.Tensor:
    """Second-integral derivative including pref_f baked in. Returns [alpha_max, P]."""
    s2 = atom_sigma_scaled ** 2
    dr = 1.0 - rcut_soft
    denom = s2 + dr ** 2 / nf ** 2
    rcut_soft_minus_rj = rcut_soft - rjs

    atom_sigma_f = atom_sigma_scaled * dr / nf / tf.math.sqrt(denom)
    rj_f = (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs) / denom
    sf2 = atom_sigma_f ** 2
    pref_f = tf.math.exp(-0.5 * rcut_soft_minus_rj ** 2 / denom)

    der_pref_f = pref_f * (
        rcut_soft_minus_rj / denom
        + rcut_soft_minus_rj ** 2 / denom ** 2 * atom_sigma_scaled * atom_sigma_scaling
    )
    der_rjf_rj = (
        (2.0 * atom_sigma_scaled * rcut_soft * atom_sigma_scaling
         + dr ** 2 / nf ** 2) / denom
        - (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs)
        * 2.0 * atom_sigma_scaled * atom_sigma_scaling / denom ** 2
    )
    der_sjf_rj = (atom_sigma_scaling * dr / nf / tf.math.sqrt(denom)
                  * (1.0 - atom_sigma_scaled ** 2 / denom))

    rjf_minus = rj_f - 1.0
    sigma_f_term = der_sjf_rj * rjf_minus / atom_sigma_f

    out_list = []
    for n_f in range(1, alpha_max + 1):
        Na_n = tf.constant(_N_a(n_f), dtype=tf.float64)
        Na_np1 = tf.constant(_N_a(n_f + 1), dtype=tf.float64)
        Na_np2 = tf.constant(_N_a(n_f + 2), dtype=tf.float64)
        py = n_f - 1
        out_list.append(
            pref_f * (
                rjf_minus / sf2 * (sigma_f_term - der_rjf_rj) * temp2_ext[py]
                + Na_np1 / sf2 / Na_n * (2.0 * sigma_f_term - der_rjf_rj) * temp2_ext[py + 1]
                + der_sjf_rj * Na_np2 / atom_sigma_f ** 3 / Na_n * temp2_ext[py + 2]
            ) + der_pref_f * temp2_ext[py]
        )
    return tf.stack(out_list, axis=0)


def radial_expansion_coeff_poly3_with_der_tf(
    rjs: tf.Tensor,                     # [P] float64 (raw, unnormalised)
    pair_neighbour_species: tf.Tensor,  # [P] int32
    pair_is_central: tf.Tensor,         # [P] bool
    n_species: int,
    alpha_max: int,
    rcut_hard: float,
    rcut_soft: float,
    atom_sigma_r: float,
    atom_sigma_r_scaling: float,
    amplitude_scaling: float,
    central_weight: float,
    radial_enhancement: int,
    nf: float,
    do_central: bool,
    W_single: tf.Tensor,                # [alpha_max, alpha_max] float64
    global_scaling: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Forward + d/d(rj) of radial expansion. Returns (radial, radial_der), each [n_max, P]."""
    rj_n = rjs / rcut_hard
    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard
    atom_sigma_scaled = atom_sigma_n + atom_sigma_r_scaling * rj_n

    in_cutoff = rj_n < 1.0
    pair_active = in_cutoff
    if not do_central:
        pair_active = pair_active & ~pair_is_central

    amplitude, amplitude_der = _radial_amplitude_with_der_tf(
        rj_n, atom_sigma_scaled, atom_sigma_r_scaling, pair_is_central,
        central_weight, amplitude_scaling, radial_enhancement,
    )

    temp1_ext = _radial_first_integral_tf(
        rj_n, alpha_max + 2, rcut_soft_n, atom_sigma_scaled
    )
    temp2_ext, pref_f = _radial_second_integral_tf(
        rj_n, alpha_max + 2, rcut_soft_n, atom_sigma_scaled, nf
    )
    near_cutoff = (rcut_soft_n - rj_n) < 4.0 * atom_sigma_scaled
    pref_f = tf.where(near_cutoff, pref_f, tf.zeros_like(pref_f))

    combined = temp1_ext[:alpha_max] + pref_f[None, :] * temp2_ext[:alpha_max]
    transformed = tf.linalg.matmul(W_single, combined)
    raw = (amplitude[None, :] * transformed
           * tf.constant(global_scaling * np.sqrt(rcut_hard), dtype=tf.float64))

    temp1_der = _radial_first_integral_der_tf(
        rj_n, temp1_ext, alpha_max, atom_sigma_scaled, atom_sigma_r_scaling
    )
    temp2_der_total = _radial_second_integral_der_tf(
        rj_n, temp2_ext, alpha_max, rcut_soft_n, atom_sigma_scaled,
        atom_sigma_r_scaling, nf,
    )
    temp2_der_total = tf.where(near_cutoff[None, :], temp2_der_total,
                                tf.zeros_like(temp2_der_total))

    der_combined = (
        amplitude[None, :] * (temp1_der + temp2_der_total)
        + amplitude_der[None, :] * (temp1_ext[:alpha_max]
                                    + pref_f[None, :] * temp2_ext[:alpha_max])
    )
    transformed_der = tf.linalg.matmul(W_single, der_combined)
    raw_der = transformed_der * tf.constant(global_scaling / np.sqrt(rcut_hard), dtype=tf.float64)

    pair_active_f = tf.cast(pair_active, tf.float64)
    raw = raw * pair_active_f[None, :]
    raw_der = raw_der * pair_active_f[None, :]

    # Distribute to species blocks
    n_max = n_species * alpha_max
    P = tf.shape(rjs)[0]
    radial_blocks = []
    radial_der_blocks = []
    for s in range(n_species):
        species_mask = tf.cast(tf.equal(pair_neighbour_species, s), tf.float64)
        radial_blocks.append(raw * species_mask[None, :])
        radial_der_blocks.append(raw_der * species_mask[None, :])
    radial = tf.concat(radial_blocks, axis=0)
    radial_der = tf.concat(radial_der_blocks, axis=0)
    return radial, radial_der


# =========================================================================
# Angular expansion (TF)
# =========================================================================


def _get_plm_array_tf(x: tf.Tensor, l_max: int) -> tf.Tensor:
    """Plm via l-recursion. x: [P] float64. Returns [k_max, P]."""
    k_max = (l_max + 1) * (l_max + 2) // 2
    P = tf.shape(x)[0]
    sqrt_1mx2 = tf.math.sqrt(tf.maximum(1.0 - x * x, tf.zeros_like(x)))

    plm_dict = {}                                                  # k → [P] tensor
    plm_dict[0] = tf.ones_like(x)                                  # P_00
    if l_max >= 1:
        plm_dict[1] = x                                            # P_10
        plm_dict[2] = -sqrt_1mx2                                   # P_11
    if l_max >= 2:
        plm_dict[3] = 1.5 * x * x - 0.5                            # P_20
        plm_dict[4] = -3.0 * x * sqrt_1mx2                         # P_21
        plm_dict[5] = 3.0 - 3.0 * x * x                            # P_22

    for l in range(3, l_max + 1):
        for m in range(l - 1):
            k = l * (l + 1) // 2 + m
            k_lm1_m = (l - 1) * l // 2 + m
            k_lm2_m = (l - 2) * (l - 1) // 2 + m
            plm_dict[k] = ((2 * l - 1) * x * plm_dict[k_lm1_m]
                           - (l - 1 + m) * plm_dict[k_lm2_m]) / (l - m)
        k_lm1_lm1 = (l - 1) * l // 2 + (l - 1)
        k_l_lm1 = l * (l + 1) // 2 + (l - 1)
        plm_dict[k_l_lm1] = x * (2 * l - 1) * plm_dict[k_lm1_lm1]
        k_l_l = l * (l + 1) // 2 + l
        plm_dict[k_l_l] = -(2 * l - 1) * sqrt_1mx2 * plm_dict[k_lm1_lm1]

    return tf.stack([plm_dict[k] for k in range(k_max)], axis=0)


def _get_ilexp_tf(x: tf.Tensor, l_max: int) -> tf.Tensor:
    """i_l(x²)·exp(-x²) for l=0..l_max. x: [P] float64. Returns [l_max+1, P]."""
    xcut = 1e-7
    x2 = x * x
    x4 = x2 * x2

    fact = [1.0]
    f = 1.0
    for i in range(1, l_max + 1):
        f = f * (2.0 * i + 1.0)
        fact.append(f)

    safe_x2 = tf.maximum(x2, tf.constant(1e-300, dtype=tf.float64))
    safe_x4 = tf.maximum(x4, tf.constant(1e-300, dtype=tf.float64))
    full_flm2 = tf.abs((1.0 - tf.math.exp(-2.0 * x2)) / (2.0 * safe_x2))
    full_flm1 = tf.abs((x2 - 1.0 + tf.math.exp(-2.0 * x2) * (x2 + 1.0))
                       / (2.0 * safe_x4))
    pos = x > 0
    flm2 = tf.where(pos, full_flm2, tf.ones_like(x))
    flm1 = tf.where(pos, full_flm1, tf.zeros_like(x))

    out_list = []
    if l_max >= 0:
        out_list.append(tf.where(x < xcut, 1.0 - x2, flm2))
    if l_max >= 1:
        out_list.append(tf.where(x2 / 1000.0 < xcut, (x2 - x4) / fact[1], flm1))

    for l in range(2, l_max + 1):
        x_2l = safe_x2 ** l
        taylor = x_2l / fact[l]
        recursion = tf.abs(flm2 - (2.0 * l - 1.0) / safe_x2 * flm1)
        fl = tf.where(taylor * l < xcut, taylor, recursion)
        flm2 = flm1
        flm1 = fl
        out_list.append(fl)
    return tf.stack(out_list, axis=0)


def _get_eimphi_factor_tf(phi: tf.Tensor, m_max: int) -> tf.Tensor:
    """exp(-i·m·φ) for m=0..m_max via Chebyshev. phi: [P] float64. Returns [m_max+1, P] complex128."""
    P = tf.shape(phi)[0]
    cos_phi = tf.math.cos(phi)
    sin_phi = tf.math.sin(phi)
    cosphi2 = 2.0 * cos_phi

    cosm2 = tf.identity(cos_phi)
    sinm2 = -sin_phi
    cosm1 = tf.ones_like(phi)
    sinm1 = tf.zeros_like(phi)

    out_list = [tf.complex(tf.ones_like(phi), tf.zeros_like(phi))]
    for l in range(1, m_max + 1):
        cos0 = cosphi2 * cosm1 - cosm2
        sin0 = cosphi2 * sinm1 - sinm2
        cosm2, cosm1 = cosm1, cos0
        sinm2, sinm1 = sinm1, sin0
        out_list.append(tf.complex(cos0, -sin0))
    return tf.stack(out_list, axis=0)


def _get_plm_array_der_tf(
    plm_extended: tf.Tensor, l_max: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """(plm_div_sin, plm_der_mul_sin) — modified forms for pole-singularity-free derivatives."""
    P = tf.shape(plm_extended)[1]
    k_max = (l_max + 1) * (l_max + 2) // 2
    zeros = tf.zeros((P,), dtype=tf.float64)

    der_dict = {}                                                  # k → [P]
    div_dict = {}
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            if m == 0:
                if l == 0:
                    part1 = zeros
                else:
                    k_l_1 = l * (l + 1) // 2 + 1
                    part1 = -0.5 * plm_extended[k_l_1]
            else:
                k_l_mm1 = l * (l + 1) // 2 + (m - 1)
                part1 = 0.5 * (l + m) * (l - m + 1) * plm_extended[k_l_mm1]
            if m == l:
                part2 = zeros
            else:
                k_l_mp1 = l * (l + 1) // 2 + (m + 1)
                part2 = -0.5 * plm_extended[k_l_mp1]
            der_dict[k] = part1 + part2
    plm_der_mul_sin = tf.stack([der_dict[k] for k in range(k_max)], axis=0)

    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            if m == 0:
                div_dict[k] = zeros
            else:
                k_lp1_mp1 = (l + 1) * (l + 2) // 2 + (m + 1)
                k_lp1_mm1 = (l + 1) * (l + 2) // 2 + (m - 1)
                part1 = 0.5 * (l - m + 1) * (l - m + 2) * plm_extended[k_lp1_mm1]
                part2 = 0.5 * plm_extended[k_lp1_mp1]
                div_dict[k] = part1 + part2
    plm_div_sin = tf.stack([div_dict[k] for k in range(k_max)], axis=0)
    return plm_div_sin, plm_der_mul_sin


def _get_ilexp_der_tf(
    rj: tf.Tensor,
    ilexp_array: tf.Tensor,          # [l_max+1, P]
    l_max: int,
    atom_sigma: tf.Tensor,
    atom_sigma_scaling: float,
) -> tf.Tensor:
    """d(ilexp)/d(rj). Returns [l_max+1, P]."""
    P = tf.shape(rj)[0]
    coeff1 = 2.0 * rj / atom_sigma ** 2
    coeff2 = 1.0 - atom_sigma_scaling * rj / atom_sigma
    safe_rj = tf.maximum(rj, tf.constant(1e-300, dtype=tf.float64))

    out_list = [coeff1 * (ilexp_array[1] - ilexp_array[0])]
    for l in range(1, l_max + 1):
        out_list.append(((-coeff1 - 2.0 * (l + 1) / safe_rj) * ilexp_array[l]
                         + coeff1 * ilexp_array[l - 1]))
    out = tf.stack(out_list, axis=0)
    out = out * coeff2[None, :]
    safe = tf.cast(rj >= 1e-5, tf.float64)
    return out * safe[None, :]


def angular_expansion_coeff_with_der_tf(
    rjs: tf.Tensor,
    thetas: tf.Tensor,
    phis: tf.Tensor,
    pair_active: tf.Tensor,           # [P] bool
    l_max: int,
    atom_sigma_t: float,
    atom_sigma_t_scaling: float,
    rcut: float,
    preflm: tf.Tensor,                # [k_max] float64
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns (exp_coeff, exp_coeff_rad_der, exp_coeff_azi_der, exp_coeff_pol_der), each [k_max, P] complex128."""
    k_max = (l_max + 1) * (l_max + 2) // 2

    x = tf.math.cos(thetas)
    plm_ext = _get_plm_array_tf(x, l_max + 1)
    plm = plm_ext[:k_max]
    plm_div_sin, plm_der_mul_sin = _get_plm_array_der_tf(plm_ext, l_max)

    atom_sigma = atom_sigma_t + atom_sigma_t_scaling * rjs
    rj_by_sigma = rjs / atom_sigma
    prefl = _get_ilexp_tf(rj_by_sigma, l_max)
    prefl_rad_der = _get_ilexp_der_tf(rjs, prefl, l_max, atom_sigma, atom_sigma_t_scaling)
    prefm = _get_eimphi_factor_tf(phis, l_max)

    # Build eimphi[k(l, m)] = prefl[l] · prefm[m] (real * complex)
    eimphi_list = []
    eimphi_rad_der_list = []
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            prefl_l_c = tf.complex(prefl[l], tf.zeros_like(prefl[l]))
            prefl_rad_l_c = tf.complex(prefl_rad_der[l], tf.zeros_like(prefl_rad_der[l]))
            eimphi_list.append(prefl_l_c * prefm[m])
            eimphi_rad_der_list.append(prefl_rad_l_c * prefm[m])
    eimphi = tf.stack(eimphi_list, axis=0)
    eimphi_rad_der = tf.stack(eimphi_rad_der_list, axis=0)
    eimphi_azi_der = eimphi * tf.constant(1j, dtype=tf.complex128)

    amplitude = rcut ** 2 / atom_sigma ** 2
    real_factor = amplitude[None, :] * preflm[:, None] * plm           # [k_max, P] real
    real_factor_c = tf.complex(real_factor, tf.zeros_like(real_factor))

    real_factor_div_sin = amplitude[None, :] * preflm[:, None] * plm_div_sin
    real_factor_div_sin_c = tf.complex(real_factor_div_sin, tf.zeros_like(real_factor_div_sin))
    real_factor_der_mul_sin = amplitude[None, :] * preflm[:, None] * plm_der_mul_sin
    real_factor_der_mul_sin_c = tf.complex(real_factor_der_mul_sin,
                                            tf.zeros_like(real_factor_der_mul_sin))

    exp_coeff = real_factor_c * eimphi
    exp_coeff_rad_der = (
        real_factor_c * eimphi_rad_der
        - 2.0 * tf.complex(amplitude / atom_sigma * atom_sigma_t_scaling
                           * preflm[:, None] * plm,
                           tf.zeros_like(real_factor)) * eimphi
    )
    exp_coeff_azi_der = real_factor_div_sin_c * eimphi_azi_der
    exp_coeff_pol_der = real_factor_der_mul_sin_c * eimphi

    pair_active_c = tf.complex(tf.cast(pair_active, tf.float64),
                                tf.zeros((tf.shape(pair_active)[0],), dtype=tf.float64))
    exp_coeff = exp_coeff * pair_active_c[None, :]
    exp_coeff_rad_der = exp_coeff_rad_der * pair_active_c[None, :]
    exp_coeff_azi_der = exp_coeff_azi_der * pair_active_c[None, :]
    exp_coeff_pol_der = exp_coeff_pol_der * pair_active_c[None, :]
    return exp_coeff, exp_coeff_rad_der, exp_coeff_azi_der, exp_coeff_pol_der


# =========================================================================
# cnk + power spectrum (TF)
# =========================================================================


def aggregate_cnk_with_der_tf(
    R: tf.Tensor,                     # [n_max, P] float64
    A: tf.Tensor,                     # [k_max, P] complex128
    R_der: tf.Tensor,
    A_rad_der: tf.Tensor,
    A_azi_der: tf.Tensor,
    A_pol_der: tf.Tensor,
    pair_struct: tf.Tensor,           # [P] int32
    n_sites: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Forward cnk + 3 per-pair spherical derivatives (NOT scattered)."""
    fac = tf.complex(tf.constant(4.0 * np.pi, dtype=tf.float64),
                     tf.constant(0.0, dtype=tf.float64))
    R_c = tf.complex(R, tf.zeros_like(R))                        # [n_max, P] complex128
    R_der_c = tf.complex(R_der, tf.zeros_like(R_der))

    # cnk[k, n, i] = 4π · Σ_p [pair_struct[p]==i] A[k, p] · R[n, p]
    prod_fwd = fac * A[:, None, :] * R_c[None, :, :]              # [k_max, n_max, P]
    prod_perm = tf.transpose(prod_fwd, [2, 0, 1])                  # [P, k_max, n_max]
    cnk_perm = tf.math.unsorted_segment_sum(prod_perm, pair_struct, n_sites)
    cnk = tf.transpose(cnk_perm, [1, 2, 0])                        # [k_max, n_max, n_sites]

    cnk_rad_der = fac * (A[:, None, :] * R_der_c[None, :, :]
                         + A_rad_der[:, None, :] * R_c[None, :, :])
    cnk_azi_der = fac * A_azi_der[:, None, :] * R_c[None, :, :]
    cnk_pol_der = fac * A_pol_der[:, None, :] * R_c[None, :, :]
    return cnk, cnk_rad_der, cnk_azi_der, cnk_pol_der


def _build_kept_triples(skip_mask: np.ndarray, n_max: int, l_max: int) -> list[tuple]:
    """Pre-enumerate kept (n, n', l, mult_start, mult_count) tuples.

    Returns a Python list — used at @tf.function trace time to unroll the
    power-spectrum loop over only the channels that survive the compress mask.
    """
    out = []
    counter = 0
    counter2 = 0
    for n in range(n_max):
        for nprime in range(n, n_max):
            for l in range(l_max + 1):
                if not bool(skip_mask[counter]):
                    out.append((n, nprime, l, counter2, l + 1))
                    counter2 += l + 1
                counter += 1
    return out


def power_spectrum_with_grad_tf(
    cnk: tf.Tensor,                   # [k_max, n_max, n_sites] complex128
    cnk_rad_der: tf.Tensor,           # [k_max, n_max, P]
    cnk_azi_der: tf.Tensor,
    cnk_pol_der: tf.Tensor,
    pair_atom: tf.Tensor,             # [P] int32
    multiplicity_array: tf.Tensor,    # [n_active] float64
    kept_triples: list,               # Python list — captured at trace time
    compressed_idx: tf.Tensor,        # [P_nonzero] int32
    coeffs: tf.Tensor,                # [P_nonzero] float64
    n_compressed: int,
    n_sites,                          # int OR scalar int32 tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns (soap_norm [n_sites, n_compressed], soap_*_der each [n_compressed, P])."""
    cnk_at = tf.gather(cnk, pair_atom, axis=2)                       # [k_max, n_max, P]
    fwd_rows = []
    rad_rows = []
    azi_rows = []
    pol_rows = []
    for (n, nprime, l, c2_start, c2_count) in kept_triples:
        k_start = l * (l + 1) // 2
        k_end = k_start + (l + 1)
        mult_tf = multiplicity_array[c2_start:c2_start + c2_count]

        prod_fwd = (cnk[k_start:k_end, n, :]
                    * tf.math.conj(cnk[k_start:k_end, nprime, :]))
        fwd_rows.append(tf.reduce_sum(mult_tf[:, None] * tf.math.real(prod_fwd), axis=0))

        cnk_at_n = cnk_at[k_start:k_end, n, :]
        cnk_at_np = cnk_at[k_start:k_end, nprime, :]
        rn = cnk_rad_der[k_start:k_end, n, :]
        rnp = cnk_rad_der[k_start:k_end, nprime, :]
        an = cnk_azi_der[k_start:k_end, n, :]
        anp = cnk_azi_der[k_start:k_end, nprime, :]
        pn = cnk_pol_der[k_start:k_end, n, :]
        pnp = cnk_pol_der[k_start:k_end, nprime, :]
        rad_term = rn * tf.math.conj(cnk_at_np) + cnk_at_n * tf.math.conj(rnp)
        azi_term = an * tf.math.conj(cnk_at_np) + cnk_at_n * tf.math.conj(anp)
        pol_term = pn * tf.math.conj(cnk_at_np) + cnk_at_n * tf.math.conj(pnp)
        rad_rows.append(tf.reduce_sum(mult_tf[:, None] * tf.math.real(rad_term), axis=0))
        azi_rows.append(tf.reduce_sum(mult_tf[:, None] * tf.math.real(azi_term), axis=0))
        pol_rows.append(tf.reduce_sum(mult_tf[:, None] * tf.math.real(pol_term), axis=0))

    fwd_kept = tf.stack(fwd_rows, axis=0)                            # [n_kept, n_sites]
    rad_kept = tf.stack(rad_rows, axis=0)                            # [n_kept, P]
    azi_kept = tf.stack(azi_rows, axis=0)
    pol_kept = tf.stack(pol_rows, axis=0)

    P_dim = tf.shape(pair_atom)[0]
    n_compressed_t = tf.constant(n_compressed, tf.int32)
    n_sites_t = (n_sites if tf.is_tensor(n_sites)
                 else tf.constant(int(n_sites), tf.int32))
    soap_unnorm = tf.scatter_nd(
        compressed_idx[:, None],
        coeffs[:, None] * fwd_kept,
        shape=tf.stack([n_compressed_t, n_sites_t]),
    )
    soap_rad_der = tf.scatter_nd(
        compressed_idx[:, None],
        coeffs[:, None] * rad_kept,
        shape=tf.stack([n_compressed_t, P_dim]),
    )
    soap_azi_der = tf.scatter_nd(
        compressed_idx[:, None],
        coeffs[:, None] * azi_kept,
        shape=tf.stack([n_compressed_t, P_dim]),
    )
    soap_pol_der = tf.scatter_nd(
        compressed_idx[:, None],
        coeffs[:, None] * pol_kept,
        shape=tf.stack([n_compressed_t, P_dim]),
    )

    norms = tf.linalg.norm(soap_unnorm, axis=0)
    sqrt_dot_p = tf.where(norms < 1e-5, tf.ones_like(norms), norms)
    sdpp = tf.gather(sqrt_dot_p, pair_atom)                          # [P]
    soap_per_pair = tf.gather(soap_unnorm, pair_atom, axis=1)

    def _norm_der(der):
        dot = tf.reduce_sum(soap_per_pair * der, axis=0)
        return (der / sdpp[None, :]
                - soap_per_pair / sdpp[None, :] ** 3 * dot[None, :])

    soap_rad_der = _norm_der(soap_rad_der)
    soap_azi_der = _norm_der(soap_azi_der)
    soap_pol_der = _norm_der(soap_pol_der)
    soap_norm = tf.transpose(soap_unnorm / sqrt_dot_p[None, :])
    return soap_norm, soap_rad_der, soap_azi_der, soap_pol_der


def _spherical_to_cartesian_per_pair_tf(
    soap_rad_der: tf.Tensor,          # [n_compressed, P]
    soap_azi_der: tf.Tensor,
    soap_pol_der: tf.Tensor,
    rjs: tf.Tensor,                   # [P]
    thetas: tf.Tensor,
    phis: tf.Tensor,
    pair_active: tf.Tensor,           # [P] bool
) -> tf.Tensor:
    """Convert spherical → Cartesian gradients. Returns [P, 3, n_compressed]."""
    safe_rj = tf.where(pair_active, rjs, tf.ones_like(rjs))
    r_inv = 1.0 / safe_rj
    sin_t = tf.math.sin(thetas)
    cos_t = tf.math.cos(thetas)
    sin_p = tf.math.sin(phis)
    cos_p = tf.math.cos(phis)

    rad = tf.transpose(soap_rad_der)                                  # [P, n_compressed]
    azi = tf.transpose(soap_azi_der)
    pol = tf.transpose(soap_pol_der)

    coef_rad_x = (sin_t * cos_p)[:, None]
    coef_pol_x = (-cos_t * cos_p * r_inv)[:, None]
    coef_azi_x = (-sin_p * r_inv)[:, None]
    der_x = coef_rad_x * rad + coef_pol_x * pol + coef_azi_x * azi

    coef_rad_y = (sin_t * sin_p)[:, None]
    coef_pol_y = (-cos_t * sin_p * r_inv)[:, None]
    coef_azi_y = (cos_p * r_inv)[:, None]
    der_y = coef_rad_y * rad + coef_pol_y * pol + coef_azi_y * azi

    coef_rad_z = cos_t[:, None]
    coef_pol_z = (sin_t * r_inv)[:, None]
    der_z = coef_rad_z * rad + coef_pol_z * pol

    mask = tf.cast(pair_active, tf.float64)[:, None]
    der_x = der_x * mask
    der_y = der_y * mask
    der_z = der_z * mask
    return tf.stack([der_x, der_y, der_z], axis=1)                    # [P, 3, n_compressed]


def _aggregate_self_derivative_tf(
    soap_cart_der: tf.Tensor,         # [P, 3, n_compressed]
    pair_atom: tf.Tensor,             # [P]
    pair_is_central: tf.Tensor,       # [P] bool — strict rj=0 self-pair flag
    n_sites: int,
) -> tf.Tensor:
    """Translational invariance: -Σ_{p≠central} grad → central slot for centre i.

    Image self-pairs (same atom, non-zero displacement) are regular neighbours
    here — their gradient is subtracted from the rj=0 central slot, just like
    any other neighbour.
    """
    is_central_f = tf.cast(pair_is_central, tf.float64)[:, None, None]   # [P, 1, 1]
    non_central = soap_cart_der * (1.0 - is_central_f)
    centre_sum = tf.math.unsorted_segment_sum(non_central, pair_atom, n_sites)
    neg_centre = -centre_sum
    self_grad_to_add = tf.gather(neg_centre, pair_atom, axis=0)
    self_grad_to_add = self_grad_to_add * is_central_f
    return soap_cart_der + self_grad_to_add


# =========================================================================
# Top-level driver (TF, GPU)
# =========================================================================


@tf.function(reduce_retracing=True)
def _compute_soap_with_grad_inner_tf(
    rjs, thetas, phis, pair_atom, pair_gidx, pair_neighbour_species,
    pair_is_central, pair_active,
    W_single, multiplicity_array, compressed_idx, coeffs, preflm,
    *,
    kept_triples,
    n_atoms, n_species, alpha_max, l_max, rcut_hard, rcut_soft,
    atom_sigma_r, atom_sigma_r_scaling,
    atom_sigma_t, atom_sigma_t_scaling,
    amplitude_scaling, central_weight,
    radial_enhancement, nf, do_central, n_compressed, n_max,
):
    """Inner TF compute. Inputs are TF tensors; constants are Python/NumPy."""
    R, R_der = radial_expansion_coeff_poly3_with_der_tf(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard, rcut_soft=rcut_soft,
        atom_sigma_r=atom_sigma_r, atom_sigma_r_scaling=atom_sigma_r_scaling,
        amplitude_scaling=amplitude_scaling, central_weight=central_weight,
        radial_enhancement=radial_enhancement, nf=nf, do_central=do_central,
        W_single=W_single,
    )
    A, A_rad, A_azi, A_pol = angular_expansion_coeff_with_der_tf(
        rjs, thetas, phis, pair_active,
        l_max=l_max, atom_sigma_t=atom_sigma_t,
        atom_sigma_t_scaling=atom_sigma_t_scaling,
        rcut=rcut_hard, preflm=preflm,
    )
    cnk, cnk_rad, cnk_azi, cnk_pol = aggregate_cnk_with_der_tf(
        R, A, R_der, A_rad, A_azi, A_pol, pair_atom, n_atoms,
    )
    soap_norm, soap_rad_der, soap_azi_der, soap_pol_der = power_spectrum_with_grad_tf(
        cnk, cnk_rad, cnk_azi, cnk_pol, pair_atom,
        multiplicity_array, kept_triples, compressed_idx, coeffs,
        n_compressed=n_compressed, n_sites=n_atoms,
    )
    grad_cart = _spherical_to_cartesian_per_pair_tf(
        soap_rad_der, soap_azi_der, soap_pol_der,
        rjs, thetas, phis, ~pair_is_central & pair_active,
    )
    grad_cart = _aggregate_self_derivative_tf(grad_cart, pair_atom, pair_is_central, n_atoms)
    return soap_norm, grad_cart


# =========================================================================
# Forward-only TF helpers (skip derivative computation entirely)
# =========================================================================


def radial_expansion_coeff_poly3_tf(
    rjs: tf.Tensor, pair_neighbour_species: tf.Tensor, pair_is_central: tf.Tensor,
    n_species: int, alpha_max: int, rcut_hard: float, rcut_soft: float,
    atom_sigma_r: float, atom_sigma_r_scaling: float, amplitude_scaling: float,
    central_weight: float, radial_enhancement: int, nf: float,
    do_central: bool, W_single: tf.Tensor, global_scaling: float = 1.0,
) -> tf.Tensor:
    """Forward-only radial expansion. Returns [n_max, P]."""
    rj_n = rjs / rcut_hard
    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard
    atom_sigma_scaled = atom_sigma_n + atom_sigma_r_scaling * rj_n
    in_cutoff = rj_n < 1.0
    pair_active = in_cutoff
    if not do_central:
        pair_active = pair_active & ~pair_is_central
    # Reuse the with_der amplitude — discard amp_der (cheap to compute)
    amplitude, _ = _radial_amplitude_with_der_tf(
        rj_n, atom_sigma_scaled, atom_sigma_r_scaling, pair_is_central,
        central_weight, amplitude_scaling, radial_enhancement,
    )
    temp1 = _radial_first_integral_tf(rj_n, alpha_max, rcut_soft_n, atom_sigma_scaled)
    temp2, pref_f = _radial_second_integral_tf(
        rj_n, alpha_max, rcut_soft_n, atom_sigma_scaled, nf
    )
    near_cutoff = (rcut_soft_n - rj_n) < 4.0 * atom_sigma_scaled
    pref_f = tf.where(near_cutoff, pref_f, tf.zeros_like(pref_f))
    combined = temp1 + pref_f[None, :] * temp2
    transformed = tf.linalg.matmul(W_single, combined)
    raw = (amplitude[None, :] * transformed
           * tf.constant(global_scaling * np.sqrt(rcut_hard), dtype=tf.float64))
    pair_active_f = tf.cast(pair_active, tf.float64)
    raw = raw * pair_active_f[None, :]
    radial_blocks = []
    for s in range(n_species):
        species_mask = tf.cast(tf.equal(pair_neighbour_species, s), tf.float64)
        radial_blocks.append(raw * species_mask[None, :])
    return tf.concat(radial_blocks, axis=0)


def angular_expansion_coeff_tf(
    rjs: tf.Tensor, thetas: tf.Tensor, phis: tf.Tensor, pair_active: tf.Tensor,
    l_max: int, atom_sigma_t: float, atom_sigma_t_scaling: float,
    rcut: float, preflm: tf.Tensor,
) -> tf.Tensor:
    """Forward-only angular expansion. Returns [k_max, P] complex128."""
    k_max = (l_max + 1) * (l_max + 2) // 2
    x = tf.math.cos(thetas)
    plm = _get_plm_array_tf(x, l_max)
    atom_sigma = atom_sigma_t + atom_sigma_t_scaling * rjs
    rj_by_sigma = rjs / atom_sigma
    prefl = _get_ilexp_tf(rj_by_sigma, l_max)
    prefm = _get_eimphi_factor_tf(phis, l_max)
    eimphi_list = []
    for l in range(l_max + 1):
        for m in range(l + 1):
            prefl_l_c = tf.complex(prefl[l], tf.zeros_like(prefl[l]))
            eimphi_list.append(prefl_l_c * prefm[m])
    eimphi = tf.stack(eimphi_list, axis=0)
    amplitude = rcut ** 2 / atom_sigma ** 2
    real_factor = amplitude[None, :] * preflm[:, None] * plm
    real_factor_c = tf.complex(real_factor, tf.zeros_like(real_factor))
    exp_coeff = real_factor_c * eimphi
    pair_active_c = tf.complex(
        tf.cast(pair_active, tf.float64),
        tf.zeros((tf.shape(pair_active)[0],), dtype=tf.float64),
    )
    return exp_coeff * pair_active_c[None, :]


def aggregate_cnk_tf(
    R: tf.Tensor, A: tf.Tensor, pair_struct: tf.Tensor, n_sites: int,
) -> tf.Tensor:
    """Forward-only cnk scatter-sum. Returns [k_max, n_max, n_sites] complex128."""
    fac = tf.complex(tf.constant(4.0 * np.pi, dtype=tf.float64),
                     tf.constant(0.0, dtype=tf.float64))
    R_c = tf.complex(R, tf.zeros_like(R))
    prod = fac * A[:, None, :] * R_c[None, :, :]
    prod_perm = tf.transpose(prod, [2, 0, 1])
    cnk_perm = tf.math.unsorted_segment_sum(prod_perm, pair_struct, n_sites)
    return tf.transpose(cnk_perm, [1, 2, 0])


def power_spectrum_tf(
    cnk: tf.Tensor, multiplicity_array: tf.Tensor, kept_triples: list,
    compressed_idx: tf.Tensor, coeffs: tf.Tensor,
    n_compressed: int, n_sites,
) -> tf.Tensor:
    """Forward-only power spectrum. Returns [n_sites, n_compressed]."""
    fwd_rows = []
    for (n, nprime, l, c2_start, c2_count) in kept_triples:
        k_start = l * (l + 1) // 2
        k_end = k_start + (l + 1)
        mult_tf = multiplicity_array[c2_start:c2_start + c2_count]
        prod_fwd = (cnk[k_start:k_end, n, :]
                    * tf.math.conj(cnk[k_start:k_end, nprime, :]))
        fwd_rows.append(tf.reduce_sum(mult_tf[:, None] * tf.math.real(prod_fwd), axis=0))
    fwd_kept = tf.stack(fwd_rows, axis=0)
    n_compressed_t = tf.constant(n_compressed, tf.int32)
    n_sites_t = (n_sites if tf.is_tensor(n_sites)
                 else tf.constant(int(n_sites), tf.int32))
    soap_unnorm = tf.scatter_nd(
        compressed_idx[:, None],
        coeffs[:, None] * fwd_kept,
        shape=tf.stack([n_compressed_t, n_sites_t]),
    )
    norms = tf.linalg.norm(soap_unnorm, axis=0)
    sqrt_dot_p = tf.where(norms < 1e-5, tf.ones_like(norms), norms)
    return tf.transpose(soap_unnorm / sqrt_dot_p[None, :])


@tf.function(reduce_retracing=True)
def _compute_soap_inner_tf(
    rjs, thetas, phis, pair_atom, pair_neighbour_species,
    pair_is_central, pair_active,
    W_single, multiplicity_array, compressed_idx, coeffs, preflm,
    *,
    kept_triples,
    n_atoms, n_species, alpha_max, l_max, rcut_hard, rcut_soft,
    atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
    amplitude_scaling, central_weight,
    radial_enhancement, nf, do_central, n_compressed, n_max,
):
    """Forward-only inner pipeline (no derivatives)."""
    R = radial_expansion_coeff_poly3_tf(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard, rcut_soft=rcut_soft,
        atom_sigma_r=atom_sigma_r, atom_sigma_r_scaling=atom_sigma_r_scaling,
        amplitude_scaling=amplitude_scaling, central_weight=central_weight,
        radial_enhancement=radial_enhancement, nf=nf, do_central=do_central,
        W_single=W_single,
    )
    A = angular_expansion_coeff_tf(
        rjs, thetas, phis, pair_active,
        l_max=l_max, atom_sigma_t=atom_sigma_t,
        atom_sigma_t_scaling=atom_sigma_t_scaling,
        rcut=rcut_hard, preflm=preflm,
    )
    cnk = aggregate_cnk_tf(R, A, pair_atom, n_atoms)
    return power_spectrum_tf(
        cnk, multiplicity_array, kept_triples, compressed_idx, coeffs,
        n_compressed=n_compressed, n_sites=n_atoms,
    )


def _build_cfg_cache(species_Z: list[int], soap_params: dict) -> dict:
    """Compute the cfg-only-dependent constants used by the TF compute functions.

    These depend solely on (species_Z, alpha_max, l_max) and so can be cached
    across many frames with the same configuration. The dict is opaque — both
    forward and forward+grad paths consume it the same way.
    """
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    n_species = len(species_Z)
    alpha_per_species = [alpha_max] * n_species
    n_max = sum(alpha_per_species)
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    return dict(
        n_max=n_max,
        n_species=n_species,
        z_to_idx={int(z): idx for idx, z in enumerate(species_Z)},
        W_np=build_orthonormalization_matrix_poly3(alpha_max),
        mask_info=mask_info,
        multiplicity_np=build_multiplicity_array(n_max, l_max, mask_info["skip_mask"]),
        preflm_np=_get_preflm(l_max),
        kept_triples=_build_kept_triples(mask_info["skip_mask"], n_max, l_max),
    )


def compute_soap_from_positions_tf(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    numbers: np.ndarray,
    species_Z: list[int],
    soap_params: dict,
    *,
    nf: float = 4.0,
    cache: dict | None = None,
) -> tuple[tf.Tensor, np.ndarray, np.ndarray]:
    """Forward-only end-to-end SOAP on GPU.

    Returns (soap [n_atoms, n_compressed] float32, pair_atom, pair_gidx).

    Pass `cache=_build_cfg_cache(species_Z, soap_params)` to skip the
    per-call recompute of W, mask_info, multiplicity_array, etc. The class
    `DescriptorBuilderGPUTF` does this automatically.
    """
    n_atoms = positions.shape[0]
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    rcut_hard = float(soap_params["rcut_hard"])

    if cache is None:
        cache = _build_cfg_cache(species_Z, soap_params)
    n_species = cache["n_species"]
    n_max = cache["n_max"]
    z_to_idx = cache["z_to_idx"]
    W_np = cache["W_np"]
    mask_info = cache["mask_info"]
    multiplicity_np = cache["multiplicity_np"]
    preflm_np = cache["preflm_np"]
    kept_triples = cache["kept_triples"]

    pair_atom_np, pair_gidx_np, rjs_np, thetas_np, phis_np = build_neighbour_list_numpy(
        positions, cell, pbc, rcut_hard
    )
    pair_is_central_np = (pair_atom_np == pair_gidx_np) & (rjs_np < 1e-10)
    pair_active_np = rjs_np < rcut_hard
    pair_neighbour_species_np = np.asarray(
        [z_to_idx[int(numbers[int(j)])] for j in pair_gidx_np], dtype=np.int32
    )

    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        rjs = tf.constant(rjs_np, dtype=tf.float64)
        thetas = tf.constant(thetas_np, dtype=tf.float64)
        phis = tf.constant(phis_np, dtype=tf.float64)
        pair_atom = tf.constant(pair_atom_np, dtype=tf.int32)
        pair_neighbour_species = tf.constant(pair_neighbour_species_np, dtype=tf.int32)
        pair_is_central = tf.constant(pair_is_central_np)
        pair_active = tf.constant(pair_active_np)
        W_single = tf.constant(W_np, dtype=tf.float64)
        multiplicity_array = tf.constant(multiplicity_np, dtype=tf.float64)
        compressed_idx = tf.constant(mask_info["compressed_idx"], dtype=tf.int32)
        coeffs = tf.constant(mask_info["coeffs"], dtype=tf.float64)
        preflm = tf.constant(preflm_np, dtype=tf.float64)

        soap_norm = _compute_soap_inner_tf(
            rjs, thetas, phis, pair_atom, pair_neighbour_species,
            pair_is_central, pair_active,
            W_single, multiplicity_array, compressed_idx, coeffs, preflm,
            kept_triples=kept_triples,
            n_atoms=n_atoms, n_species=n_species, alpha_max=alpha_max, l_max=l_max,
            rcut_hard=rcut_hard, rcut_soft=float(soap_params["rcut_soft"]),
            atom_sigma_r=float(soap_params["atom_sigma_r"]),
            atom_sigma_r_scaling=float(soap_params["atom_sigma_r_scaling"]),
            atom_sigma_t=float(soap_params["atom_sigma_t"]),
            atom_sigma_t_scaling=float(soap_params["atom_sigma_t_scaling"]),
            amplitude_scaling=float(soap_params["amplitude_scaling"]),
            central_weight=float(soap_params["central_weight"]),
            radial_enhancement=int(soap_params["radial_enhancement"]),
            nf=nf,
            do_central=(float(soap_params["central_weight"]) != 0.0),
            n_compressed=int(mask_info["n_compressed"]), n_max=n_max,
        )
        soap_norm = tf.cast(soap_norm, tf.float32)
    return soap_norm, pair_atom_np, pair_gidx_np


def compute_soap_with_grad_from_positions_tf(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    numbers: np.ndarray,
    species_Z: list[int],
    soap_params: dict,
    *,
    nf: float = 4.0,
    cache: dict | None = None,
) -> tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray]:
    """End-to-end forward + Cartesian gradient on GPU when available.

    Returns (soap [n_atoms, n_compressed] float32,
             grad_values [P, 3, n_compressed] float32,
             pair_atom [P] int32,
             pair_gidx [P] int32).

    Pass `cache=_build_cfg_cache(species_Z, soap_params)` to skip the
    per-call recompute of W, mask_info, multiplicity, preflm, kept_triples.
    """
    n_atoms = positions.shape[0]
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    rcut_hard = float(soap_params["rcut_hard"])

    if cache is None:
        cache = _build_cfg_cache(species_Z, soap_params)
    n_species = cache["n_species"]
    n_max = cache["n_max"]
    z_to_idx = cache["z_to_idx"]
    W_np = cache["W_np"]
    mask_info = cache["mask_info"]
    multiplicity_np = cache["multiplicity_np"]
    preflm_np = cache["preflm_np"]
    kept_triples = cache["kept_triples"]

    pair_atom_np, pair_gidx_np, rjs_np, thetas_np, phis_np = build_neighbour_list_numpy(
        positions, cell, pbc, rcut_hard
    )
    # Strict self-pair: same atom AND zero displacement (Fortran's j==1 convention)
    pair_is_central_np = (pair_atom_np == pair_gidx_np) & (rjs_np < 1e-10)
    pair_active_np = rjs_np < rcut_hard
    pair_neighbour_species_np = np.asarray(
        [z_to_idx[int(numbers[int(j)])] for j in pair_gidx_np], dtype=np.int32
    )

    # Convert to TF tensors (float64/int32). Place on GPU if available.
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        rjs = tf.constant(rjs_np, dtype=tf.float64)
        thetas = tf.constant(thetas_np, dtype=tf.float64)
        phis = tf.constant(phis_np, dtype=tf.float64)
        pair_atom = tf.constant(pair_atom_np, dtype=tf.int32)
        pair_gidx = tf.constant(pair_gidx_np, dtype=tf.int32)
        pair_neighbour_species = tf.constant(pair_neighbour_species_np, dtype=tf.int32)
        pair_is_central = tf.constant(pair_is_central_np)
        pair_active = tf.constant(pair_active_np)
        W_single = tf.constant(W_np, dtype=tf.float64)
        multiplicity_array = tf.constant(multiplicity_np, dtype=tf.float64)
        compressed_idx = tf.constant(mask_info["compressed_idx"], dtype=tf.int32)
        uncompressed_idx = tf.constant(mask_info["uncompressed_idx"], dtype=tf.int32)
        coeffs = tf.constant(mask_info["coeffs"], dtype=tf.float64)
        preflm = tf.constant(preflm_np, dtype=tf.float64)

        soap_norm, grad_cart = _compute_soap_with_grad_inner_tf(
            rjs, thetas, phis, pair_atom, pair_gidx, pair_neighbour_species,
            pair_is_central, pair_active,
            W_single, multiplicity_array, compressed_idx, coeffs, preflm,
            kept_triples=kept_triples,
            n_atoms=n_atoms, n_species=n_species, alpha_max=alpha_max, l_max=l_max,
            rcut_hard=rcut_hard, rcut_soft=float(soap_params["rcut_soft"]),
            atom_sigma_r=float(soap_params["atom_sigma_r"]),
            atom_sigma_r_scaling=float(soap_params["atom_sigma_r_scaling"]),
            atom_sigma_t=float(soap_params["atom_sigma_t"]),
            atom_sigma_t_scaling=float(soap_params["atom_sigma_t_scaling"]),
            amplitude_scaling=float(soap_params["amplitude_scaling"]),
            central_weight=float(soap_params["central_weight"]),
            radial_enhancement=int(soap_params["radial_enhancement"]),
            nf=nf,
            do_central=(float(soap_params["central_weight"]) != 0.0),
            n_compressed=int(mask_info["n_compressed"]), n_max=n_max,
        )
        # Cast to float32 at output boundary
        soap_norm = tf.cast(soap_norm, tf.float32)
        grad_cart = tf.cast(grad_cart, tf.float32)
    return soap_norm, grad_cart, pair_atom_np, pair_gidx_np


# =========================================================================
# Class wrapper
# =========================================================================


class DescriptorBuilderGPUTF:
    """TF GPU-backed SOAP-turbo descriptor builder.

    Drop-in replacement for DescriptorBuilder with the same
    `build_descriptors_flat(dataset)` API. Each frame triggers one TF graph
    execution; with a GPU available, the SOAP compute runs on GPU and only
    the neighbour-list construction stays on CPU (Python).

    Restrictions: basis="poly3", compress_mode="trivial", uniform per-species
    hyperparameters (same as the NumPy DescriptorBuilderGPU).
    """

    def __init__(self, cfg) -> None:
        if getattr(cfg, "basis", "poly3") != "poly3":
            raise NotImplementedError(
                f"DescriptorBuilderGPUTF only supports basis='poly3', got '{cfg.basis}'"
            )
        if getattr(cfg, "compress_mode", "trivial") != "trivial":
            raise NotImplementedError(
                f"DescriptorBuilderGPUTF only supports compress_mode='trivial', got '{cfg.compress_mode}'"
            )
        self.cfg = cfg
        self._species_Z = (sorted(int(z) for z in cfg.types)
                           if getattr(cfg, "types", None) else None)
        self._soap_params = dict(
            alpha_max=int(cfg.alpha_max),
            l_max=int(cfg.l_max),
            rcut_hard=float(cfg.rcut_hard),
            rcut_soft=float(cfg.rcut_soft),
            atom_sigma_r=float(cfg.atom_sigma_r),
            atom_sigma_r_scaling=float(cfg.atom_sigma_r_scaling),
            atom_sigma_t=float(cfg.atom_sigma_t),
            atom_sigma_t_scaling=float(cfg.atom_sigma_t_scaling),
            amplitude_scaling=float(cfg.amplitude_scaling),
            central_weight=float(cfg.central_weight),
            radial_enhancement=int(cfg.radial_enhancement),
        )
        # Precompute cfg-only-dependent constants once. Built lazily to allow
        # construction before cfg.types is populated; rebuilt automatically if
        # cfg.types changes after construction (rare).
        self._cache = None
        self._cache_species_key = None
        self._with_grad_fn = None
        self._fwd_only_fn = None
        if self._species_Z is not None:
            self._cache = _build_cfg_cache(self._species_Z, self._soap_params)
            self._cache_species_key = tuple(self._species_Z)
            self._build_locked_compute_fns()

    def _ensure_cache(self):
        """Build (or rebuild if cfg.types changed) the constant cache.

        Also builds per-instance @tf.function wrappers with `input_signature`
        baked in. These take only TF tensors as args (cfg constants captured
        via closure) and trace exactly once per (cfg, do_grad) combination —
        eliminating per-call retracing on varying n_atoms / pair counts.
        """
        current_key = (tuple(sorted(int(z) for z in self.cfg.types))
                       if getattr(self.cfg, "types", None) else None)
        if current_key is None:
            raise RuntimeError(
                "DescriptorBuilderGPUTF: cfg.types is not set."
            )
        if self._cache is None or current_key != self._cache_species_key:
            self._species_Z = list(current_key)
            self._cache = _build_cfg_cache(self._species_Z, self._soap_params)
            self._cache_species_key = current_key
            self._build_locked_compute_fns()

    def _build_locked_compute_fns(self):
        """Construct @tf.function wrappers locked to a single trace per cfg."""
        cache = self._cache
        sp = self._soap_params
        kept_triples = cache["kept_triples"]
        n_species = cache["n_species"]
        n_max = cache["n_max"]
        n_compressed = int(cache["mask_info"]["n_compressed"])
        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

        # Pre-place cfg-constant tensors on the target device once
        with tf.device(device):
            self._W_t = tf.constant(cache["W_np"], dtype=tf.float64)
            self._mult_t = tf.constant(cache["multiplicity_np"], dtype=tf.float64)
            self._comp_idx_t = tf.constant(cache["mask_info"]["compressed_idx"], dtype=tf.int32)
            self._coeffs_t = tf.constant(cache["mask_info"]["coeffs"], dtype=tf.float64)
            self._preflm_t = tf.constant(cache["preflm_np"], dtype=tf.float64)

        # Common Python-int / float kwargs captured by closure (constants per cfg)
        common_static = dict(
            kept_triples=kept_triples,
            n_species=n_species, alpha_max=int(sp["alpha_max"]),
            l_max=int(sp["l_max"]),
            rcut_hard=float(sp["rcut_hard"]),
            rcut_soft=float(sp["rcut_soft"]),
            atom_sigma_r=float(sp["atom_sigma_r"]),
            atom_sigma_r_scaling=float(sp["atom_sigma_r_scaling"]),
            atom_sigma_t=float(sp["atom_sigma_t"]),
            atom_sigma_t_scaling=float(sp["atom_sigma_t_scaling"]),
            amplitude_scaling=float(sp["amplitude_scaling"]),
            central_weight=float(sp["central_weight"]),
            radial_enhancement=int(sp["radial_enhancement"]),
            nf=4.0,
            do_central=(float(sp["central_weight"]) != 0.0),
            n_compressed=n_compressed, n_max=n_max,
        )

        sig = [
            tf.TensorSpec(shape=[None], dtype=tf.float64, name="rjs"),
            tf.TensorSpec(shape=[None], dtype=tf.float64, name="thetas"),
            tf.TensorSpec(shape=[None], dtype=tf.float64, name="phis"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="pair_atom"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="pair_gidx"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="pair_neigh"),
            tf.TensorSpec(shape=[None], dtype=tf.bool, name="pair_central"),
            tf.TensorSpec(shape=[None], dtype=tf.bool, name="pair_active"),
            tf.TensorSpec(shape=[], dtype=tf.int32, name="n_atoms"),
        ]

        @tf.function(input_signature=sig)
        def with_grad_fn(rjs, thetas, phis, pair_atom, pair_gidx,
                         pair_neigh, pair_central, pair_active, n_atoms):
            return _compute_soap_with_grad_inner_tf(
                rjs, thetas, phis, pair_atom, pair_gidx, pair_neigh,
                pair_central, pair_active,
                self._W_t, self._mult_t, self._comp_idx_t, self._coeffs_t,
                self._preflm_t,
                n_atoms=n_atoms, **common_static,
            )

        sig_fwd = [s for s in sig if s.name != "pair_gidx"]

        @tf.function(input_signature=sig_fwd)
        def fwd_only_fn(rjs, thetas, phis, pair_atom, pair_neigh,
                        pair_central, pair_active, n_atoms):
            return _compute_soap_inner_tf(
                rjs, thetas, phis, pair_atom, pair_neigh,
                pair_central, pair_active,
                self._W_t, self._mult_t, self._comp_idx_t, self._coeffs_t,
                self._preflm_t,
                n_atoms=n_atoms, **common_static,
            )

        self._with_grad_fn = with_grad_fn
        self._fwd_only_fn = fwd_only_fn

    # Memory budget per @tf.function call (bytes). Tuned to leave headroom for
    # the model's own GPU activity in the trajectory inference pipeline.
    AUTO_BATCH_MEMORY_BUDGET_BYTES = 2 * (1 << 30)   # 2 GiB

    def build_descriptors_flat(
        self,
        dataset: list,
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
    ) -> list:
        """Per-frame flat COO output, with optional gradient computation.

        Args:
            dataset        : list of ase.Atoms
            calc_gradients : if False, gradient/pair arrays are empty (matches
                             quippy with grad=False).
            batch_frames   : number of frames per single TF graph call.
                             - 1 (default) : per-frame, lowest memory.
                             - int >= 2    : multi-frame batching, amortises
                                             kernel-launch overhead.
                             - None        : auto — choose the largest power-
                                             of-two batch that fits a 2 GiB
                                             memory budget for the SOAP graph.

        Multi-frame batches are concatenated into one TF call with global
        atom-index offsets, then unpacked back per frame after.
        """
        self._ensure_cache()
        if batch_frames is None:
            batch_frames = self._auto_batch_frames(dataset, calc_gradients)
        elif batch_frames < 1:
            raise ValueError(f"batch_frames must be >= 1 or None, got {batch_frames}")
        results = []
        for chunk_start in range(0, len(dataset), batch_frames):
            chunk = dataset[chunk_start:chunk_start + batch_frames]
            results.extend(self._build_flat_chunk(chunk, calc_gradients))
        return results

    def _estimate_mem_per_frame_bytes(self, atoms, calc_gradients: bool) -> int:
        """Rough estimate of peak GPU bytes for one frame's SOAP compute.

        Dominant terms (with gradients):
          - 3× cnk_*_der          [k_max, n_max, P]           complex128 (16 B)
          - intermediate prods    [P, k_max, n_max]           complex128 (16 B)
          - power-spec per-pair   3 × [n_compressed, P]       float64    (8 B)
          - Cartesian der          [P, 3, n_compressed]       float64    (8 B)
        Forward-only drops the gradient terms.
        """
        rcut = float(self._soap_params["rcut_hard"])
        pa, _, _, _, _ = build_neighbour_list_numpy(
            np.asarray(atoms.positions, dtype=np.float64),
            np.asarray(atoms.cell.array, dtype=np.float64),
            np.asarray(atoms.pbc, dtype=bool),
            rcut,
        )
        n_pairs = len(pa)
        n_atoms = len(atoms)
        cache = self._cache
        l_max = int(self._soap_params["l_max"])
        k_max = (l_max + 1) * (l_max + 2) // 2
        n_max = cache["n_max"]
        n_compressed = int(cache["mask_info"]["n_compressed"])
        # Forward (always present): cnk + soap output
        fwd = 16 * k_max * n_max * n_atoms + 8 * n_compressed * n_atoms
        if not calc_gradients:
            return fwd
        # Gradient: dominated by per-pair complex tensors
        grad = (
            4 * 16 * k_max * n_max * n_pairs        # cnk + 3 derivatives
            + 2 * 16 * n_pairs * k_max * n_max      # intermediate products
            + 4 * 8 * n_compressed * n_pairs        # rad/azi/pol/cart per-pair der
        )
        return fwd + grad

    def _auto_batch_frames(self, dataset: list, calc_gradients: bool) -> int:
        """Choose batch_frames so per-call memory fits AUTO_BATCH_MEMORY_BUDGET_BYTES.

        Estimates from the first frame; assumes other frames have similar size.
        Returns at least 1 and at most len(dataset).
        """
        if not dataset:
            return 1
        per_frame = self._estimate_mem_per_frame_bytes(dataset[0], calc_gradients)
        if per_frame <= 0:
            return 1
        budget = self.AUTO_BATCH_MEMORY_BUDGET_BYTES
        n = max(1, budget // per_frame)
        return int(min(len(dataset), n))

    def _build_flat_chunk(self, chunk: list, calc_gradients: bool) -> list:
        """Process a chunk of frames in one TF call.

        Single-frame chunks still go through the locked compute_fn (one trace
        ever per (cfg, do_grad) combination); multi-frame chunks concatenate
        pair lists with global atom offsets and unpack outputs after.
        """
        # ---- Build pair lists for all frames; concatenate with global offsets ----
        rcut_hard = float(self._soap_params["rcut_hard"])
        z_to_idx = self._cache["z_to_idx"]
        per_frame_n_atoms = []
        per_frame_n_pairs = []
        all_rjs, all_thetas, all_phis = [], [], []
        all_pa, all_pg, all_pneigh = [], [], []
        all_central, all_active = [], []
        atom_offset = 0
        for atoms in chunk:
            positions = np.asarray(atoms.positions, dtype=np.float64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            pbc = np.asarray(atoms.pbc, dtype=bool)
            numbers = np.asarray(atoms.numbers, dtype=np.int32)
            N = len(numbers)
            pa, pg, rjs, thetas, phis = build_neighbour_list_numpy(
                positions, cell, pbc, rcut_hard
            )
            pair_is_central = (pa == pg) & (rjs < 1e-10)
            pair_active = rjs < rcut_hard
            pair_neigh = np.asarray(
                [z_to_idx[int(numbers[int(j)])] for j in pg], dtype=np.int32
            )
            all_rjs.append(rjs)
            all_thetas.append(thetas)
            all_phis.append(phis)
            all_pa.append(pa.astype(np.int32) + atom_offset)
            all_pg.append(pg.astype(np.int32) + atom_offset)
            all_pneigh.append(pair_neigh)
            all_central.append(pair_is_central)
            all_active.append(pair_active)
            per_frame_n_atoms.append(N)
            per_frame_n_pairs.append(len(pa))
            atom_offset += N

        total_atoms = atom_offset
        rjs_c = np.concatenate(all_rjs)
        thetas_c = np.concatenate(all_thetas)
        phis_c = np.concatenate(all_phis)
        pa_c = np.concatenate(all_pa)
        pg_c = np.concatenate(all_pg)
        pneigh_c = np.concatenate(all_pneigh)
        central_c = np.concatenate(all_central)
        active_c = np.concatenate(all_active)

        # ---- Single TF call via the locked-signature wrapper ----
        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):
            rjs_t = tf.constant(rjs_c, dtype=tf.float64)
            thetas_t = tf.constant(thetas_c, dtype=tf.float64)
            phis_t = tf.constant(phis_c, dtype=tf.float64)
            pair_atom_t = tf.constant(pa_c, dtype=tf.int32)
            pair_gidx_t = tf.constant(pg_c, dtype=tf.int32)
            pair_neigh_t = tf.constant(pneigh_c, dtype=tf.int32)
            central_t = tf.constant(central_c)
            active_t = tf.constant(active_c)
            n_atoms_t = tf.constant(total_atoms, dtype=tf.int32)

            if calc_gradients:
                soap_t, grad_t = self._with_grad_fn(
                    rjs_t, thetas_t, phis_t, pair_atom_t, pair_gidx_t, pair_neigh_t,
                    central_t, active_t, n_atoms_t,
                )
                soap_np = tf.cast(soap_t, tf.float32).numpy()
                grad_np = tf.cast(grad_t, tf.float32).numpy()
            else:
                soap_t = self._fwd_only_fn(
                    rjs_t, thetas_t, phis_t, pair_atom_t, pair_neigh_t,
                    central_t, active_t, n_atoms_t,
                )
                soap_np = tf.cast(soap_t, tf.float32).numpy()
                grad_np = None

        # ---- Split outputs back per frame ----
        out = []
        atom_off = 0
        pair_off = 0
        for k, atoms in enumerate(chunk):
            N = per_frame_n_atoms[k]
            P = per_frame_n_pairs[k]
            soap_frame = soap_np[atom_off:atom_off + N]
            if calc_gradients:
                grad_frame = grad_np[pair_off:pair_off + P]
                pa_frame = (all_pa[k] - atom_off).astype(np.int32)
                pg_frame = (all_pg[k] - atom_off).astype(np.int32)
                out.append((soap_frame, grad_frame, pa_frame, pg_frame))
            else:
                out.append((
                    soap_frame,
                    np.zeros((0, 3, soap_frame.shape[1]), dtype=np.float32),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int32),
                ))
            atom_off += N
            pair_off += P
        return out


    def build_descriptors(
        self,
        dataset: list,
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
    ):
        """Bucketised-by-atom format used by the training pipeline.

        When calc_gradients=False, the second/third tuple elements are empty
        per-atom lists. The training pipeline assembles these into COO arrays
        of size 0 — the model's predict_batch path that consumes them must
        handle the no-gradient case (only target_mode=0/PES is meaningful
        without gradients).

        Returns:
            (dataset_descriptors, dataset_gradients, dataset_grad_index)
              calc_gradients=True : full per-pair gradients per atom
              calc_gradients=False: empty lists for gradients/grad_index
        """
        flat = self.build_descriptors_flat(
            dataset, calc_gradients=calc_gradients, batch_frames=batch_frames,
        )
        dataset_descriptors = []
        dataset_gradients = []
        dataset_grad_index = []
        for atoms, item in zip(dataset, flat):
            soap, grad, pa, pg = item
            N = soap.shape[0]
            dataset_descriptors.append(
                tf.convert_to_tensor(soap, dtype=tf.float32)
            )
            if calc_gradients:
                grads_per_atom = [[] for _ in range(N)]
                idx_per_atom = [[] for _ in range(N)]
                for p in range(len(pa)):
                    i = int(pa[p])
                    grads_per_atom[i].append(grad[p])
                    idx_per_atom[i].append(int(pg[p]))
                grads_tf = [
                    tf.convert_to_tensor(np.asarray(g, dtype=np.float32), dtype=tf.float32)
                    for g in grads_per_atom
                ]
                dataset_gradients.append(grads_tf)
                dataset_grad_index.append(idx_per_atom)
            else:
                # Empty per-atom lists: training pipeline will pad to zero pairs
                dataset_gradients.append([
                    tf.zeros((0, 3, soap.shape[1]), dtype=tf.float32)
                    for _ in range(N)
                ])
                dataset_grad_index.append([[] for _ in range(N)])
        return dataset_descriptors, dataset_gradients, dataset_grad_index


# =========================================================================
# Phase 11 validation: TF outputs match NumPy reference
# =========================================================================


def run_phase11_validation() -> None:
    """For all 7 fixtures, the TF backend must agree with the NumPy reference."""
    print("=== Phase 11 validation (TF GPU port vs NumPy reference) ===\n")
    fixtures = ["water_monomer", "water_dimer", "h2_close", "h2_far",
                "single_h", "si_bulk", "si_dimer"]
    max_soap = 0.0
    max_grad = 0.0
    for name in fixtures:
        f = load_fixture(name)
        positions = np.asarray(f["positions"], dtype=np.float64)
        cell = np.asarray(f["cell"], dtype=np.float64)
        pbc = np.asarray(f["pbc"], dtype=bool)
        numbers = np.asarray(f["numbers"], dtype=np.int32)
        species_Z = sorted(set(int(z) for z in numbers))
        soap_params = f["soap_params"]

        # NumPy reference
        soap_np, grad_np, pa_np, pg_np = compute_soap_with_grad_from_positions_numpy(
            positions, cell, pbc, numbers, species_Z, soap_params,
        )
        # TF backend
        soap_tf, grad_tf, pa_tf, pg_tf = compute_soap_with_grad_from_positions_tf(
            positions, cell, pbc, numbers, species_Z, soap_params,
        )
        soap_tf_np = soap_tf.numpy().astype(np.float64)
        grad_tf_np = grad_tf.numpy().astype(np.float64)

        # Pair lists match (same NL builder)
        if not np.array_equal(pa_np, pa_tf):
            raise AssertionError(f"{name}: pair_atom differs between NumPy and TF backends")

        s_err = float(np.abs(soap_np.astype(np.float32) - soap_tf_np.astype(np.float32)).max())
        g_err = float(np.abs(grad_np.astype(np.float32) - grad_tf_np.astype(np.float32)).max())
        max_soap = max(max_soap, s_err)
        max_grad = max(max_grad, g_err)
        if s_err > 1e-5 or g_err > 1e-4:
            raise AssertionError(
                f"{name}: TF vs NumPy mismatch. soap={s_err:.3e}, grad={g_err:.3e}"
            )
        print(f"   {name:<14} N={int(f['n_atoms']):>2}  P={len(pa_np):>4}  "
              f"|Δsoap|={s_err:.2e}  |Δgrad|={g_err:.2e}  PASS")
    print(f"\n   Overall max |Δsoap|={max_soap:.2e}, max |Δgrad|={max_grad:.2e}")
    print("\nAll Phase 11 (TF GPU) checks passed.")


if __name__ == "__main__":
    print(f"TF version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus if gpus else 'NONE — running on CPU'}\n")
    run_phase11_validation()
