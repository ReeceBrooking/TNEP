"""Two-atom sanity check on the TRAINED liquid-water dipole TNEP (no special fit): an
isolated O–H pair at separation d and at 2d. A polar bond's dipole grows with its
length (μ ≈ q·d for a fixed partial charge), so the trained model must predict a larger
dipole at 2d than at d, pointing O → H in both cases. Skipped when the checkpoint is absent."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
from ase import Atoms

MODEL = os.path.join(ROOT, "models", "n30_q216_pop100_20260711_193535_waterbulk_dipole_0.9998",
                     "train_waterbulk_O_H_dipole_best_val.h5")
L = 20.0                                              # box edge (Å): ≥ 2·rc, so the pair sees no image


def _pair(d, u):
    return Atoms("OH", positions=[[L / 2] * 3, [L / 2] * 3 + d * u], cell=L * np.eye(3), pbc=True)


@pytest.mark.skipif(not os.path.exists(MODEL), reason="trained water dipole checkpoint not present")
@pytest.mark.xfail(strict=True, reason="2026-10-27: bulk-trained models do not extrapolate to a stretched isolated O–H pair — "
                   "gradient-readout TNEPs flip sign beyond 1.7 Å (n30_q216) or decay to zero at the cutoff (n30_q75), and a bulk-fitted dipole GAP "
                   "falls to zero at its cutoff; both grow with d only up to ~1.5 Å. Remove this marker when a model passes.")
def test_trained_water_dipole_grows_with_oh_separation():
    from model_io import load_model
    from DescriptorBuilder import make_descriptor_builder
    from spectroscopy import predict_trajectory_batch
    import data as tnep_data
    model = load_model(MODEL)
    cfg = model.cfg
    d = 0.97; x = np.array([1.0, 0.0, 0.0])
    frames = [_pair(d, x), _pair(2 * d, x)]
    types = [np.asarray(t) for t in tnep_data.assign_type_indices(frames, cfg.types)]
    mu = predict_trajectory_batch(model, make_descriptor_builder(cfg), frames, types)
    mu_d, mu_2d = mu[0], mu[1]
    print(f"μ(d={d} Å) = {mu_d}, μ(2d) = {mu_2d}, |μ(2d)|/|μ(d)| = {np.linalg.norm(mu_2d) / np.linalg.norm(mu_d):.2f}")
    assert mu_d[0] > 0 and mu_2d[0] > 0                                # along the bond, O → H
    assert abs(mu_d[1]) + abs(mu_d[2]) < 0.05 * abs(mu_d[0])           # and nothing transverse
    assert np.linalg.norm(mu_2d) > np.linalg.norm(mu_d)                # longer bond, larger dipole
