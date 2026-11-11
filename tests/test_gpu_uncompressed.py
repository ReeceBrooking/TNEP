"""compress_mode='none' on the GPU builder matches quippy (descriptor_mode=0)."""
import numpy as np
import pytest
from ase.build import molecule

from TNEPconfig import TNEPconfig
from DescriptorBuilder import make_descriptor_builder
from descriptor_layout import compute_dim_q


@pytest.mark.parametrize("compress", ["trivial", "none"])
def test_gpu_uncompressed_matches_quippy(compress):
    # quippy always L2-normalises, so radial_basis_norm stays "l2" here.
    cfg = TNEPconfig()
    cfg.target_mode, cfg.dipole_readout = 1, "tnep"   # gradients needed
    cfg.nep_descriptor, cfg.radial_basis_size = "off", None
    cfg.basis, cfg.compress_mode, cfg.radial_basis_norm = "poly3", compress, "l2"
    cfg.l_max, cfg.alpha_max, cfg.rcut_hard, cfg.rcut_soft = 3, 3, 4.0, 3.5
    cfg.types, cfg.num_types, cfg.type_map = [8, 1], 2, {8: 0, 1: 1}
    cfg.gpu_descriptor_precision = "float64"
    cfg.dim_q = compute_dim_q(cfg)
    atoms = [molecule("H2O"), molecule("H2O2")]
    for a in atoms:
        a.center(vacuum=5.0)
        a.rattle(0.05, seed=1)
    out = {}
    for mode in (0, 1):
        cfg.descriptor_mode = mode
        out[mode] = make_descriptor_builder(cfg).build_descriptors(atoms)
    for s in range(len(atoms)):
        d0, d1 = (np.asarray(out[m][0][s]) for m in (0, 1))
        assert d0.shape == d1.shape == (len(atoms[s]), cfg.dim_q)
        np.testing.assert_allclose(d1, d0, atol=1e-8)
        for i in range(len(atoms[s])):
            # Builders order neighbours differently: align rows by index.
            g = {}
            for m in (0, 1):
                idx = np.asarray(out[m][2][s][i])
                g[m] = np.asarray(out[m][1][s][i])[np.argsort(idx, kind="stable")]
                g[m, "idx"] = np.sort(idx)
            np.testing.assert_array_equal(g[0, "idx"], g[1, "idx"])
            np.testing.assert_allclose(g[1], g[0], atol=1e-7)
