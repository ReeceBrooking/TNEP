# TNEP — dipole and polarisability models for IR / Raman spectra

Machine-learning models of the dipole moment and polarisability of bulk water,
used to compute IR and Raman spectra from MD trajectories. Two model families
share one descriptor and readout pipeline:

- **GAP (production)** — a sparse-kernel (gap_fit-style) fit on a *neighbour-basis
  readout*: every atom gets kernel-predicted site scalars that weight its bond
  vectors, so the dipole is linear in the kernel coefficients and the fit is a
  single linear solve. `gap_dipole.py`.
- **TNEP ANN (reference)** — the Tensorial NEP of Xu et al. (JCTC 2024, 20, 3273):
  per-species single-hidden-layer networks on SOAP-turbo descriptors, trained with
  SNES. Kept for comparison; not the production path. `MasterTNEP.py`.

### Production models (bulk water, 500 training frames, 500 test frames)

| Target | Model | Test RRMSE |
|---|---|---|
| Dipole | SOAP-7 + 2-body site model + 2-channel cross group | 1.45–1.47 % |
| Polarisability | SOAP-7 (4.0 Å) + 3-body site model + second SOAP scale 6 Å + 4-channel cross group, λ_shear 2 | 13.0 % diag / 13.9 % off-diag |

RRMSE = RMSE / std of the test labels, per component. The TNEP ANN of the paper
scores 16.3 / 20.4 % on the same polarisability test set. The exact settings of
both rows live in `diagnostics/gap_experiments/gap_scan.py` (`MU`, `POL`, `ROWS`).

---

## Repository structure

```
TNEP/
├── gap_dipole.py        # GAP fit + prediction (fit_gap_dipole, predict); writes a run record to GAP/
├── gap_config.py        # GAPconfig: all knobs of the GAP fit (independent of TNEPconfig)
├── gap_models.py        # 2b / 3b site descriptors and the angular cross group
├── gap_gpu.py           # TF evaluation of a fitted GAP model (trajectories)
├── data.py              # Dataset loading, neighbour-basis readout geometry (V), COO packing
├── DescriptorBuilder.py # SOAP-turbo via quippy (CPU) …
├── DescriptorBuilderGPU_tf.py   # … and the native TF port (descriptor_mode=1)
├── descriptor_layout.py # Descriptor column layout shared by the ANN and the builders
├── spectroscopy.py      # IR / Raman spectra from dipole or polarisability time series
├── MasterTNEP.py        # ANN entry point: train_model, test_model, process_trajectory
├── TNEP.py, SNES.py, adam.py, nep_descriptor.py, TNEPconfig.py   # ANN model, optimisers, config
├── model_io.py, plotting.py
├── tests/               # pytest suite
├── scripts/             # smoke_train.py (ANN smoke), plot_thermo.py
├── diagnostics/         # gap_experiments/gap_scan.py (production rows + ladders), small probes
├── datasets/            # extxyz data (local, not tracked)
├── GAP/                 # fit run records (JSON, not tracked)
└── models/              # ANN run directories (not tracked)
```

---

## Install

```bash
pip install tensorflow numpy scipy h5py ase quippy-ase matplotlib tqdm pytest
```

`quippy-ase` needs a working [QUIP](https://github.com/libAtoms/QUIP) build. The
GPU descriptor builder (`descriptor_mode=1`) replaces quippy for `basis="poly3"`
and `compress_mode` "trivial" / "none".

---

## GAP fit

```python
from gap_config import GAPconfig
import gap_dipole as gd

cfg = GAPconfig()
cfg.data_path, cfg.test_data_path = "datasets/train_waterbulk.xyz", "datasets/test_waterbulk.xyz"
cfg.target_mode = 1                       # 1 = dipole, 2 = polarisability
cfg.rcut_hard, cfg.rcut_soft = 3.5, 3.0   # SOAP and readout cutoff
cfg.l_max = cfg.alpha_max = 7
cfg.dipole_channels, cfg.dipole_radial, cfg.dipole_cutoff = 10, "chebyshev_warped", "cosine2"
cfg.gap_sparse_per_species, cfg.gap_sigma = 250, 3e-3

model = gd.fit_gap_dipole(cfg)            # also writes GAP/<record>.json (config + metrics)
mu = gd.predict(model, frames)            # [F, 3]  (or [F, 6] Voigt for target_mode=2)
```

`GAPconfig` defaults are a small, fast configuration, not the production model;
copy the `MU` / `POL` dictionaries in `gap_scan.py` for those. Models are not
persisted (a fit takes minutes); the JSON record reproduces one.

**Readout.** For atom *i* with bond vectors **d**ᵢⱼ, radial channels *k* and
cutoff *f_c*:

```
v_i[:, k] = Σ_j R_k(r_ij) f_c(r_ij) d_ij            (species-resolved under dipole_species_pairs)
μ         = Σ_i Σ_k s_i^k v_i[:, k],   s_i^k = Σ_m α_m^k (q̂_i · q̂_m)^ξ
```

The polarisability uses the same fit with six Voigt rows and two kernel groups
per channel (isotropic |d|² I and traceless d⊗d − |d|² I/3). Optional column
groups: `gap_site_models` (2b / 3b kernels summed with SOAP), `gap_cross_channels`
(two-neighbour cross terms), `gap_extra_rcut` (a second SOAP scale for the
kernel), `gap_lambda_shear` (off-diagonal weight).

**Scans.** `gap_scan.py` runs one fit per process, driven by environment variables:

```bash
PS_PROP=mu  PS_ROW="soap+2b +cross2" PS_SIGMA=0.003 PS_FIT=500 python diagnostics/gap_experiments/gap_scan.py
PS_PROP=pol PS_LIST=1 python diagnostics/gap_experiments/gap_scan.py      # list row tags
```

---

## Spectra

`spectroscopy.py` turns a per-frame time series into a spectrum:

```python
from spectroscopy import ir_spectrum_from_file, raman_spectrum_from_file
ir_spectrum_from_file("plots/traj_dipoles.npy", dt_fs=1.0)              # ω²-weighted dipole ACF
raman_spectrum_from_file("plots/traj_polarizabilities.npy", dt_fs=1.0)  # VV / VH from iso / aniso ACFs
```

IR uses the cosine transform of the dipole autocorrelation; Raman the isotropic
(Tr α / 3) and anisotropic parts, with the classical ω² footing by default.
`scripts/raman_gap_gpumd_traj.py` (untracked) shows the full path from a GPUMD
trajectory through a GAP polarisability model to the `.npy` file.

---

## Dataset format

extxyz readable by ASE. Labels in the frame header:

| target_mode | key | shape |
|---|---|---|
| 1 dipole | `dipole` | 3 |
| 2 polarisability | `pol` | 6 Voigt (xx, yy, zz, xy, yz, zx) |
| 0 PES (ANN only) | `energy`, per-atom `force` | — |

Override with `cfg.target_key`. Dipoles default to e·bohr (`dipole_units`);
polarisability labels in `*_waterbulk_chi.xyz` are the susceptibility form
V(ε−1)/4π used for the production numbers.

---

## ANN path (reference)

```bash
python MasterTNEP.py        # trains with the defaults in TNEPconfig.py
```

`train_model` builds descriptors, trains with SNES (`cfg.optimizer="adam"` for an
autodiff trainer), scores, and saves to `models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/`
(config, history, HDF5 checkpoints, plots). `process_trajectory(model, path)`
streams a trajectory through the model and writes the spectrum. The committed
`TNEPconfig` defaults are the last experiment's values, not a tuned baseline.

```bash
python scripts/smoke_train.py --generations 20 --frames 8   # end-to-end ANN smoke
```

---

## Tests

```bash
python -m pytest tests          # ~1 min on CPU
```

Covers the neighbour-basis readout contracts in `data.py`, the GAP fit
(identities against hand-built columns, rotation covariance, GPU vs NumPy
prediction) and the spectrum routines.

---

## Citation

The ANN reference implementation follows:

```bibtex
@article{xu2024tnep,
  title={Tensorial Properties via the Neuroevolution Potential Framework: Fast Simulation of Infrared and Raman Spectra},
  author={Xu, Nan and Rosander, Petter and Sch{\"a}fer, Christian and Lindgren, Eric and {\"O}sterbacka, Nicklas and Fang, Mandi and Chen, Wei and He, Yi and Fan, Zheyong and Erhart, Paul},
  journal={Journal of Chemical Theory and Computation},
  volume={20}, pages={3273--3284}, year={2024},
  doi={10.1021/acs.jctc.3c01343}
}
```
