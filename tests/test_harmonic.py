import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

from harmonic import vibrational_modes, CM1_PER_SQRT_EV_AMU_A2


def test_vibrational_modes_diatomic_analytic():
    """Two masses on a spring along x: the one non-trivial mode must come out at
    ν = 521.47·sqrt(k/µ). Pins the mass-weighting and the cm⁻¹ conversion."""
    k, m1, m2 = 10.0, 1.0, 16.0           # eV/Å², amu
    H = np.zeros((6, 6))
    H[0, 0] = H[3, 3] = k
    H[0, 3] = H[3, 0] = -k

    freqs, modes = vibrational_modes(H, [m1, m2])

    mu = m1 * m2 / (m1 + m2)              # reduced mass
    expected = CM1_PER_SQRT_EV_AMU_A2 * np.sqrt(k / mu)
    assert np.allclose(np.sort(np.abs(freqs))[:5], 0.0, atol=1e-8)
    assert np.isclose(freqs[-1], expected, rtol=1e-10)
    assert modes.shape == (6, 6)


def test_cm1_constant():
    # Standard value: sqrt(eV/(amu·Å²)) → 521.47 cm⁻¹.
    assert np.isclose(CM1_PER_SQRT_EV_AMU_A2, 521.4709, atol=1e-3)


def test_hessian_translation_invariance():
    """A descriptor-based PES is exactly translation invariant, so the Hessian
    of ANY such model — trained or not — must have three zero eigenvalues and a
    gradient that sums to zero over atoms. Checks the FD Hessian end to end."""
    import ase.io
    from TNEPconfig import TNEPconfig
    from data import collect, split
    from DescriptorBuilderGPU import compute_dim_q
    from TNEP import TNEP
    from harmonic import harmonic_analysis

    cfg = TNEPconfig()
    cfg.data_path = 'datasets/water_monomer.xyz'; cfg.test_data_path = None
    cfg.allowed_species = [8, 1]; cfg.filter_mode = 'subset'
    cfg.target_mode = 0; cfg.target_key = 'potential'
    cfg.total_N = 8; cfg.test_ratio = 0.25
    cfg.num_neurons = 8; cfg.num_hidden_layers = 1
    cfg.descriptor_mode = 0; cfg.descriptor_mixing = False
    cfg.descriptor_preprocess_contract = "off"
    cfg.optimizer = "adam"; cfg.save_path = None; cfg.seed = 0
    cfg.batch_size = None; cfg.population_chunk_size = None
    cfg.batch_chunk_size = None; cfg.pin_data_to_cpu = True

    dataset, ti = collect(cfg); cfg.randomise(dataset)
    cfg.dim_q = compute_dim_q(cfg)
    split(dataset, ti, cfg)
    model = TNEP(cfg)

    atoms = ase.io.read(cfg.data_path, index='0')
    assert len(atoms) == 3

    res = harmonic_analysis(model, atoms, h=0.01, batch_size=32, verbose=False,
                            pin_to_cpu=True, descriptor_batch_frames=1)
    H, g, freqs = res['hessian'], res['gradient'], res['freqs_cm']

    assert H.shape == (9, 9)
    assert np.allclose(H, H.T, atol=1e-12)
    # Rigid translation: sum of forces zero, and H·t = 0 for t a translation.
    assert np.allclose(g.reshape(3, 3).sum(axis=0), 0.0, atol=1e-4)
    # Tolerance is set by the FD noise floor (float32 energies at h=0.01),
    # not by the invariance itself — a real violation would be O(|H|).
    tol = 0.05 * np.abs(H).max()
    for c in range(3):
        t = np.zeros(9); t[c::3] = 1.0
        assert np.abs(H @ t).max() < tol, "Hessian not translation invariant"
    # Three (near-)zero eigenvalues from the translations.
    lam = np.sort(np.abs(freqs))
    assert lam[2] < 20.0, f"expected 3 near-zero modes, got {lam[:4]}"
