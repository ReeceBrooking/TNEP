"""gap_gpu.predict_gpu (batched float32 TF forward) == gap_dipole._predict_numpy (the fit's
float64 contraction) for the soap-only polarisability GAP with cross group and pair columns."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np
from ase.io import read

import gap_dipole as gd
import gap_gpu
from test_gap_dipole import _cfg

WATER = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def test_gpu_forward_matches_numpy_contraction():
    cfg = _cfg(data_path=WATER, total_N=12, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=4, l_max=4,
               compress_mode="trivial", basis="poly3", descriptor_mode=1, descriptor_batch_frames=4,
               dipole_channels=5, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
               dipole_species_pairs="symmetric", gap_sparse_per_species=20, gap_kernel_power=2, gap_sigma=0.1,
               gap_cross_channels=2)
    model = gd.fit_gap_dipole(cfg)
    frames = read(WATER, index="12:17")
    assert gap_gpu.gpu_applicable(model)
    ref = gd._predict_numpy(model, frames)
    out = gap_gpu.predict_gpu(model, frames, batch=2)                      # two batches, one ragged
    assert out.shape == ref.shape == (5, 6)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-4)                 # float64 forward on float32 descriptors
    assert np.abs(out - ref).max() < 0.05 * ref.std(0).mean()               # far below the fluctuation scale
    np.testing.assert_allclose(gd.predict(model, frames), out)                  # predict() routes through the GPU path
