# TNEP — Tensorial Neuroevolution Potential

A TensorFlow implementation of the Tensorial NEP (TNEP) framework for machine learning models of atomic potential energy surfaces (PES), dipole moments, and polarizabilities. This work is based on the methodology introduced in:

> Xu et al., *Tensorial Properties via the Neuroevolution Potential Framework: Fast Simulation of Infrared and Raman Spectra*, J. Chem. Theory Comput. **2024**, 20, 3273–3284. https://doi.org/10.1021/acs.jctc.3c01343

---

## Overview

TNEP extends the NEP (Neuroevolution Potential) framework to predict not just scalar energies (rank-0 tensors) but also rank-1 tensors (dipole moments) and rank-2 tensors (polarizability). This enables the simulation of infrared (IR) and Raman spectra from molecular dynamics trajectories at a fraction of the cost of ab initio MD.

The model uses **SOAP-turbo** descriptors to encode local atomic environments, and is trained using a **Separable Natural Evolution Strategy (SNES)** — a gradient-free black-box optimiser well suited to neural network weight optimisation.

### Current status

| Target | Status |
|---|---|
| PES (energy + forces) | ✅ Implemented (`target_mode = 0`) |
| Dipole moment | ✅ Implemented (`target_mode = 1`) |
| Polarizability | ✅ Implemented (`target_mode = 2`) |

---

## Repository Structure

```
TNEP/
├── MasterTNEP.py        # Entry point — train_model() and process_trajectory()
├── TNEP.py              # Core model: per-type ANN, predict_batch (COO), score
├── TNEPconfig.py        # Configuration dataclass — all hyperparameters in one place
├── DescriptorBuilder.py # SOAP-turbo descriptor + gradient computation via quippy
├── SNES.py              # Separable Natural Evolution Strategy optimiser
├── data.py              # Loading, filtering, splitting, COO pad-and-stack
├── spectroscopy.py      # IR/Raman spectrum + trajectory inference (streaming batches)
├── plotting.py          # Fitness curves, correlation plots, spectra
├── model_io.py          # Model save/load (HDF5; legacy .npz still loadable)
├── datasets/            # Training and trajectory data (extxyz)
└── models/              # Auto-created run directories with config, history, plots
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Neural network, tensor operations |
| `numpy` | Numerical computing |
| `h5py` | HDF5 model checkpoints |
| `ase` | Reading extxyz datasets |
| `quippy-ase` | SOAP-turbo descriptors and descriptor gradients |
| `matplotlib` | Plotting |
| `tqdm` | Progress bars |

```bash
pip install tensorflow numpy h5py ase quippy-ase matplotlib tqdm
```

> **Note:** `quippy-ase` requires a working installation of [QUIP/GAP](https://github.com/libAtoms/QUIP). See the [quippy documentation](https://quippy.readthedocs.io) for platform-specific build instructions.

---

## Configuration

All hyperparameters are defined in `TNEPconfig.py`. Selected defaults shown below — see the source file for the full list of options.

```python
# Data
data_path        = "datasets/train_waterbulk.xyz"
test_data_path   = "datasets/test_waterbulk.xyz"   # None = split from data_path
allowed_species  = ["C", "H", "O"]                  # Filter by atomic species
filter_mode      = "subset"                          # "subset" or "exact"
target_mode      = 1                                 # 0=PES, 1=Dipole, 2=Polarizability
total_N          = 1000                              # Cap on training structures (None = all)
test_ratio       = 0.3

# Model
num_neurons      = 30
activation       = "tanh"
init_sigma       = 0.1

# SNES training
pop_size         = 80
num_generations  = 60000
batch_size       = None        # None = full train set per generation
eta_sigma        = None        # None = auto from problem dimension
patience         = None        # Early stopping (None = disabled)
loss_type        = "mse"       # "mse" or "mae"

# SOAP-turbo descriptor
l_max            = 4
alpha_max        = 4
rcut_hard        = 6.0         # Hard cutoff (Å)
rcut_soft        = 5.5         # Soft cutoff (Å)
basis            = "poly3"

# Regularisation
toggle_regularization    = True
lambda_1                 = 0.001    # L1
lambda_2                 = 0.001    # L2
per_type_regularization  = True     # Per-element ranking (GPUMD NEP4 style)

# Target scaling (dipole only)
scale_targets    = True        # Train against per-atom dipole; recovered at inference
dipole_units     = "e*bohr"    # "e*angstrom", "e*bohr", or "debye"

# Memory / parallelism
pin_data_to_cpu          = True   # Required when full dataset doesn't fit on GPU
num_descriptor_workers   = None   # None = auto from SLURM or cpu_count // 2

# Output
save_path        = "models/auto"   # Auto-creates run dir with timestamp
plot_interval    = 10000           # Periodic plots every N generations
```

---

## Dataset Format

Training data must be in **extxyz** format readable by ASE.

| Mode | Required field |
|---|---|
| PES (`target_mode=0`) | `energy=...` in header; `force` per-atom column (optional but recommended); optional `virial` for stress |
| Dipole (`target_mode=1`) | `dipole="..."` in header (3-vector) |
| Polarizability (`target_mode=2`) | `pol="..."` in header (6-component Voigt: xx,yy,zz,xy,yz,zx) |

Override the default key with `cfg.target_key = "mu"` for non-standard labels.

---

## Usage

### Training

Edit `TNEPconfig.py` (or pass a `TNEPconfig` instance), then run:

```bash
python MasterTNEP.py
```

This loads the dataset, builds SOAP descriptors, splits into train/test/val, trains via SNES, scores both the final-generation and best-validation models, and saves everything to `models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/`:

- `config.txt` — hyperparameters used for the run
- `history.csv` — per-generation losses, regularisation, σ stats
- `*_best_val.h5` and `*_final_gen.h5` — model checkpoints (HDF5)
- `plots/` — fitness curves, correlation plots, σ history, timing breakdown

### Inference / Spectroscopy

```python
from MasterTNEP import process_trajectory
from model_io import load_model

model = load_model("models/.../train_waterbulk_O_H_dipole_best_val.h5")
result = process_trajectory(
    model, "datasets/water_bulk_traj.xyz",
    dt_fs=1.0, batch_size=100, save_plots="plots", pin_to_cpu=True,
)
```

The trajectory is streamed in `batch_size`-frame chunks: descriptors are built, packed to COO, run through `predict_batch`, appended, and discarded before the next batch — peak memory is `O(batch_size)` rather than `O(all_frames)`. Outputs are saved alongside the spectrum:

- `<name>_dipoles.txt` (or `_polarizabilities.txt`) — predictions per frame
- `ir_spectrum.png`, `power_spectrum.png` (mode 1) or `raman_spectrum.png` (mode 2)

---

## Model Architecture

Each atom's local environment is encoded as a SOAP descriptor vector **q**. A single hidden-layer neural network maps **q** to a per-atom scalar output *Uᵢ*:

```
qᵢ → [W0, b0] → tanh → [W1] → Uᵢ
```

Per-element weight matrices give each species its own network parameters. The total property is assembled from atomic contributions:

- **Energy** — sum of site energies; forces from descriptor gradients via the chain rule
- **Dipole** (rank-1) — `μ = -Σᵢ Σⱼ |rᵢⱼ|² · (∂Uᵢ/∂rᵢⱼ)` (eq. 3 of the paper)
- **Polarizability** (rank-2) — dual-ANN: scalar branch contributes the diagonal isotropic part, tensor branch contributes the anisotropic virial-like part (eq. 4)

Internally, descriptor gradients are stored in **COO sparse format** (`grad_values [P,3,Q]` + `pair_struct [P]` + `pair_atom [P]` + `pair_gidx [P]`), avoiding the dense `[B, A_max, M_max, 3, Q]` allocation that would otherwise dominate memory.

---

## Training Algorithm (SNES)

Rather than gradient descent, the model is trained using a **Separable Natural Evolution Strategy** (Schaul, 2011). Per generation:

1. Sample a population of `pop_size` perturbed parameter vectors from a diagonal Gaussian
2. Evaluate each candidate on a batch of training structures (forward pass only)
3. Update the distribution mean and per-parameter σ based on fitness rankings

When `per_type_regularization=True`, fitness is ranked per element type — driving per-type natural-gradient updates in the GPUMD NEP4 style.

This avoids backpropagation through the descriptor computation and is naturally batchable across the population dimension.

---

## Vibrational Spectra

Once trained, TNEP models predict properties along MD trajectories to simulate vibrational spectra:

- **IR** — `σ(ω) ∝ ω² · M(ω)` where M(ω) is the cosine transform of the dipole autocorrelation function (Wiener–Khinchin via FFT, then DCT)
- **Raman** — separate cosine transforms of the isotropic γ(t) = Tr(α)/3 and anisotropic β(t) = α − γI ACFs, combined into VV (polarised) and VH (depolarised) intensities with a Bose–Einstein occupation factor

This enables nanosecond-scale sampling that would be prohibitive with ab initio MD.

---

## Citation

If you use this code, please cite the original TNEP paper:

```bibtex
@article{xu2024tnep,
  title={Tensorial Properties via the Neuroevolution Potential Framework: Fast Simulation of Infrared and Raman Spectra},
  author={Xu, Nan and Rosander, Petter and Sch{\"a}fer, Christian and Lindgren, Eric and {\"O}sterbacka, Nicklas and Fang, Mandi and Chen, Wei and He, Yi and Fan, Zheyong and Erhart, Paul},
  journal={Journal of Chemical Theory and Computation},
  volume={20},
  pages={3273--3284},
  year={2024},
  doi={10.1021/acs.jctc.3c01343}
}
```
