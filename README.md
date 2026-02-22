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
| Polarizability | 🚧 In progress (`target_mode = 2`) |

---

## Repository Structure

```
TNEP/
├── MasterTNEP.py        # Entry point — data loading, train/test split, model training
├── TNEP.py              # Core model: neural network, force calculation, dipole prediction
├── TNEPconfig.py        # Configuration dataclass — all hyperparameters in one place
├── DescriptorBuilder.py # SOAP-turbo descriptor and gradient computation via quippy-ase
├── SNES.py              # Separable Natural Evolution Strategy optimiser
├── train.xyz            # Example training dataset (extxyz format, dipole targets)
└── PEStrain.xyz         # Example training dataset (extxyz format, PES targets — Te bulk)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Neural network, tensor operations, autograd |
| `ase` | Reading extxyz datasets |
| `quippy-ase` | Interface to SOAP-turbo for descriptors and descriptor gradients |

Install Python dependencies:

```bash
pip install tensorflow ase quippy-ase
```

> **Note:** `quippy-ase` requires a working installation of [QUIP/GAP](https://github.com/libAtoms/QUIP). See the [quippy documentation](https://quippy.readthedocs.io) for platform-specific build instructions.

---

## Configuration

All hyperparameters are defined in `TNEPconfig.py`. Key settings:

```python
data_path       = "train.xyz"   # Path to training data
target_mode     = 1             # 0 = PES, 1 = Dipole, 2 = Polarizability (WIP)

num_neurons     = 64            # Hidden layer size
num_generations = 20            # Number of SNES training generations
batch_size      = 10            # Structures per training step
pop_size        = 16            # SNES population size per generation
total_N         = 100           # Max structures to use (None = all)
test_ratio      = 0.2           # Train/test split

n_radial        = 3             # Radial SOAP components
n_radial_ang    = 3             # Angular SOAP radial components
Lmax            = 2             # Maximum angular momentum
rc              = 6.0           # Cutoff radius (Å)

activation      = 'tanh'
init_sigma      = 0.1           # Initial SNES distribution std
seed            = None          # Set for reproducibility
```

---

## Dataset Format

Training data should be in **extxyz** format readable by ASE. The required properties depend on the target mode:

**PES training (`PEStrain.xyz`):**
```
energy=-937.191 ... Properties=species:S:1:pos:R:3:force:R:3
```

**Dipole training (`train.xyz`):**
Structures should include a `dipole` property in the extxyz header.

---

## Usage

Edit `TNEPconfig.py` to set your data path and target mode, then run:

```bash
python MasterTNEP.py
```

This will load the dataset, build SOAP descriptors, split into train/test sets, train the model using SNES, and report RMSE on the test set.

---

## Model Architecture

Each atom's local environment is encoded as a SOAP descriptor vector **q**. A single hidden-layer neural network maps **q** to a per-atom scalar output *Uᵢ*:

```
qᵢ → [W0, b0] → tanh → [W1] → Uᵢ
```

Per-type weight matrices are used so that each element has its own network parameters.

The total property is assembled from atomic contributions:

- **Energy** — sum of site energies; forces derived analytically from descriptor gradients
- **Dipole (rank-1)** — assembled by contracting the virial-like tensor with squared interatomic distances rᵢⱼ² (eq. 3 in the paper)
- **Polarizability (rank-2)** — combination of diagonal scalar and off-diagonal virial contributions (eq. 4 in the paper) — *in progress*

---

## Training Algorithm (SNES)

Rather than standard gradient descent, the model is trained using a **Separable Natural Evolution Strategy**. In each generation:

1. A population of perturbed parameter vectors is sampled from a Gaussian distribution
2. Each candidate is evaluated on a batch of training structures
3. The distribution mean and variance are updated based on fitness rankings

This avoids the need for backpropagation through the descriptor computation.

---

## Background: IR and Raman Spectra

Once trained, TNEP models can be used to predict properties along MD trajectories to simulate vibrational spectra:

- **IR spectrum** — Fourier transform of the dipole moment autocorrelation function (ACF)
- **Raman spectrum** — Fourier transform of the polarizability/susceptibility ACF
- Isotropic (polarized) and anisotropic (depolarized) Raman components can be separated from the full polarizability tensor

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

---

## License

See [LICENSE](LICENSE) for details.
