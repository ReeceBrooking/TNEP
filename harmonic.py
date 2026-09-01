"""Harmonic (quadratic Taylor) approximation of a TNEP potential energy surface.

Around a structure r0 the energy is expanded to second order,

    E(r0 + u) ≈ E(r0) + g·u + ½ uᵀ H u,

and only the quadratic term is kept: H = ∂²E/∂r_aβ∂r_bγ. Diagonalising the
mass-weighted Hessian gives the harmonic vibrational eigenvalues.

The Hessian comes from the same central-difference stencil the dipole
derivatives use (spectroscopy._fd_stencil / _fd_assemble), evaluated through
predict_trajectory_batch — the descriptor backends supply first descriptor
derivatives only, so finite differences are the available route.
"""
from __future__ import annotations

import numpy as np
from ase import units

from spectroscopy import _fd_assemble, _fd_derivatives

# ω[rad/s] per sqrt(eV / (amu·Å²)), then to wavenumbers:
#   ν̃[cm⁻¹] = ω / (2π c) × 1e-2  →  521.47 cm⁻¹ per sqrt(eV/(amu·Å²))
_OMEGA_SI = np.sqrt(units._e / (units._amu * 1e-20))
CM1_PER_SQRT_EV_AMU_A2 = _OMEGA_SI / (2.0 * np.pi * units._c) * 1e-2


def energy_hessian_fd(model, atoms, atom_idx=None, h: float = 0.01,
                      builder=None, batch_size: int = 32, verbose: bool = True,
                      **traj_kw):
    """Central-difference energy, gradient and Hessian for one structure.

    Args:
        model    : trained PES model (cfg.target_mode == 0)
        atoms    : ase.Atoms
        atom_idx : atoms to differentiate w.r.t. (default: all). Cost is
                   ≈ (3·n_sel)² energy evaluations, so restrict this on
                   anything bigger than a small molecule.
        h        : displacement in Å.

    Returns:
        E   : float                  energy at r0 (eV)
        g   : [3·n_sel]              ∂E/∂r_aβ  (eV/Å), index order (atom, β)
        H   : [3·n_sel, 3·n_sel]     ∂²E/∂r_aβ∂r_bγ (eV/Å²)
    """
    if model.cfg.target_mode != 0:
        raise ValueError(
            f"energy_hessian_fd needs a PES model (target_mode=0), got "
            f"target_mode={model.cfg.target_mode}.")
    if builder is None:
        builder = model.builder

    if atom_idx is None:
        atom_idx = np.arange(len(atoms))
    atom_idx = np.asarray(atom_idx, dtype=int)
    dofs = [(int(i), c) for i in atom_idx for c in range(3)]
    D = len(dofs)

    if verbose:
        n_points = 1 + 2 * D + D * (D - 1)
        print(f"FD Hessian: {n_points} energy evaluations "
              f"({D} DOF, h={h} Å)")

    P = _fd_derivatives(model, [atoms], dofs, h, builder, batch_size, verbose,
                        **traj_kw)
    E, g, H = _fd_assemble(P[0], D, h)
    # predict_batch returns [B, 1] for target_mode 0; drop the component axis.
    return float(E[0]), g[:, 0], H[:, :, 0]


def vibrational_modes(H: np.ndarray, masses: np.ndarray,
                      energy_to_eV: float = 1.0):
    """Diagonalise the mass-weighted Hessian.

        H_mw[a,b] = H[a,b] / sqrt(m_a m_b),  eigenvalues λ = ω²

    Args:
        H      : [3n, 3n] Hessian, in energy-units/Å²
        masses : [n] atomic masses in amu (ase Atoms.get_masses())
        energy_to_eV : factor converting the model's energy unit to eV. The
            frequency scale goes as its square root, so getting this wrong is
            a silent ~5× error: datasets storing Hartree (e.g. water_monomer.xyz
            under the "potential" key, where H2O reads -76.35) need 27.211386,
            not the default 1.0.

    Returns:
        freqs : [3n] wavenumbers in cm⁻¹, ascending. Negative values denote
                imaginary frequencies (λ < 0), the usual convention.
        modes : [3n, 3n] eigenvectors of the mass-weighted Hessian, columns
                matching `freqs`.
    """
    m = np.repeat(np.asarray(masses, dtype=np.float64), 3)
    H_mw = np.asarray(H, dtype=np.float64) / np.sqrt(np.outer(m, m))
    # eigh needs exact symmetry; FD gives it to roundoff, so symmetrise.
    lam, modes = np.linalg.eigh(0.5 * (H_mw + H_mw.T) * energy_to_eV)
    freqs = np.sign(lam) * np.sqrt(np.abs(lam)) * CM1_PER_SQRT_EV_AMU_A2
    return freqs, modes


def harmonic_analysis(model, atoms, atom_idx=None, h: float = 0.01,
                      energy_to_eV: float = 1.0, **kw):
    """energy_hessian_fd + vibrational_modes for one structure.

    Returns a dict with keys: energy, gradient, hessian, freqs_cm, modes,
    atom_idx. With atom_idx given, masses are taken for those atoms only, so
    the modes are those of the selected fragment held in a frozen environment.
    """
    E, g, H = energy_hessian_fd(model, atoms, atom_idx=atom_idx, h=h, **kw)
    sel = np.arange(len(atoms)) if atom_idx is None else np.asarray(atom_idx, dtype=int)
    freqs, modes = vibrational_modes(H, atoms.get_masses()[sel],
                                     energy_to_eV=energy_to_eV)
    return dict(energy=E, gradient=g, hessian=H, freqs_cm=freqs, modes=modes,
                atom_idx=sel)
