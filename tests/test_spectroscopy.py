"""Raman pipeline (spectroscopy.compute_raman_spectrum): peak positions and the
Placzek depolarisation ratios on synthetic polarizability trajectories."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import spectroscopy as sp

C_CM_PER_FS = 2.99792458e-5


def _traj(T=40000, dt=1.0):
    t = np.arange(T) * dt
    g = np.cos(2 * np.pi * 1600 * C_CM_PER_FS * t)          # isotropic band at 1600 cm⁻¹
    b = np.cos(2 * np.pi * 3000 * C_CM_PER_FS * t)          # traceless band at 3000 cm⁻¹
    iso = np.stack([g, g, g, 0 * g, 0 * g, 0 * g], 1) + 40.0 * np.array([1, 1, 1, 0, 0, 0])
    an = np.stack([2 * b, -b, -b, 0 * b, 0 * b, 0 * b], 1)
    return iso, an


def test_raman_acfs_decompose_mean_free_tensor():
    iso, an = _traj(T=2000)
    C_iso, C_an = sp.compute_raman_acfs(iso)
    assert abs(C_iso[0] - 0.5) < 1e-3 and np.allclose(C_an, 0.0)   # ⟨γγ⟩(0) = ½ for cos; mean 40 removed
    C_iso, C_an = sp.compute_raman_acfs(an)
    assert np.allclose(C_iso, 0.0) and abs(C_an[0] - 3.0) < 1e-2      # Tr[ββ] = 6·cos², ⟨cos²⟩ = ½


def test_raman_peaks_and_depolarisation_ratios():
    iso, an = _traj()
    for pol, f0, rho in ((iso, 1600, 0.0), (an, 3000, 0.75)):
        f, VV, VH, I_iso, I_an = sp.compute_raman_spectrum(pol, dt_fs=1.0, smooth_k=0, quantum_correction="none")
        k = int(np.argmax(VV))
        assert abs(f[k] - f0) < 5
        assert abs(VH[k] / VV[k] - rho) < 1e-6                        # 3γ² / (45ᾱ² + 4γ²)
        assert VV[k] == 1.0                                           # normalised on VV
    f, VV, VH, I_iso, I_an = sp.compute_raman_spectrum(iso + an, dt_fs=1.0, smooth_k=0, quantum_correction="none")
    k1, k2 = int(np.argmin(abs(f - 1600))), int(np.argmin(abs(f - 3000)))
    assert VH[k1] / VV[k1] < 1e-3 and abs(VH[k2] / VV[k2] - 0.75) < 1e-3
    assert I_an[k2] == 1.0 and abs(I_iso[k1] / I_an[k2] - 0.5 / 3.0) < 2e-2   # joint normaliser, ACF amplitudes ½ : 3


def test_raman_harmonic_factor_and_smoothing_keep_shapes():
    iso, an = _traj(T=8000)
    f0, VV0, *_ = sp.compute_raman_spectrum(iso + an, smooth_k=0)
    f1, VV1, VH1, I_iso, I_an = sp.compute_raman_spectrum(iso + an, quantum_correction="harmonic", smooth_k=5)
    assert f1.shape == f0.shape == VV1.shape == VH1.shape == I_iso.shape == I_an.shape
    f2, VV2, *_ = sp.compute_raman_spectrum(iso + an, smooth_k=5, smooth_kind="box")
    assert len(f2) == len(f0) - 4 and len(VV2) == len(f2)


def test_quantum_prefactor_forms():
    """Harmonic QCF x/(1−e^−x) composed with the exact prefactor: IR ω², reduced Raman ω²/(1−e^−x);
    "bose" is the exact IR prefactor alone; "classical" is ω² for both."""
    f = np.array([200.0, 1000.0, 3400.0]); T = 298.15
    x = 1.23984e-4 * f / (8.617333e-5 * T)
    assert np.allclose(sp._quantum_prefactor(f, "harmonic", T), f ** 2)
    assert np.allclose(sp._quantum_prefactor(f, "harmonic", T, raman=True), f ** 2 / (1 - np.exp(-x)))
    assert np.allclose(sp._quantum_prefactor(f, "classical", T, raman=True), f ** 2)
    assert np.allclose(sp._quantum_prefactor(f, "bose", T), f * (1 - np.exp(-x)))
    r = sp._quantum_prefactor(f, "harmonic", T) / sp._quantum_prefactor(f, "bose", T) / (x / (1 - np.exp(-x)))
    assert np.allclose(r, r[0])                                        # harmonic/bose = the QCF up to kT/ℏ
