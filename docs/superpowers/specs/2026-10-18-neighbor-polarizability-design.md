# Neighbour-basis polarisability readout (target_mode = 2)

## Decision

Polarisability is the rank-2 analogue of the neighbour dipole readout. The
dual-ANN virial form (GPUMD approach: scalar ANN + descriptor-derivative
virial) is removed; mode 2 **always** uses the neighbour readout.

    α = Σ_i Σ_n Σ_t' s_i^n · c[n, p(t_i, t')] · W_i[:, n, t']
    W_i[αβ, n, t'] = Σ_{j ∈ t', |d| < r_c} R_n(|d|) · d_α d_β      d = r_i − r_j

Six Voigt components [xx, yy, zz, xy, yz, zx]. Because d⊗d is even under
i↔j, the per-site einsum equals the symmetric pair form
Σ_{i<j} R_n (s_i^n + s_j^n) d⊗d — a learned bond-polarisability model, and
also the field response of the dipole readout's charge-flow graph.

## What changes

- **Geometry kernel** gains a rank: `_neighbor_dipole_pairs/_V(..., rank)`
  return `[P|N, D, C, T]` with D = 3 (rank 1, d) or 6 (rank 2, Voigt(d⊗d)).
  Model mirror: `TNEP._bond_geometry(d)` used by `predict` and
  `_neighbor_V_coo`. Every readout einsum already uses a free axis label for
  D, so forward code is unchanged.
- **Single predicate** `neighbor_readout_on(cfg)` in TNEPconfig.py:
  mode 2, or mode 1 with dipole_readout="neighbor". data.py, TNEP, SNES,
  MasterTNEP and gap_dipole key on it instead of `target_mode == 1 and …`.
- **Removed**: W0_pol/b0_pol/W1_pol/b1_pol weights, `_polarizability_coo`,
  the mode-2 branches of `predict`/`predict_batch`, the second ANN in the
  SNES genome/regulariser/type map, Adam's pol variables, h5 pol datasets,
  the pol positional args of `predict_batch` (every caller updated).
  Old mode-2 saves fail loudly at load (W0_pol present).
- **Kept**: `lambda_shear` component weighting (loss, not model), unit
  label Å³, Raman pipeline (consumes [T, 6] as before).
- **GAP**: `_phi_frame` rows = D (read from v), targets [F, D], RMSE
  convention √(mean·D). `_validate` accepts target_mode ∈ {1, 2}; dipole unit
  conversion applies to mode 1 only.
- **Gradients**: mode 2 reads no descriptor gradients → self-pairs-only COO,
  the NEP concat/replace and learned-basis guards lift automatically
  (both key on `descriptor_gradients_needed`).

## Not done (YAGNI)

- No separate isotropic site head (Σ_i a_i I): the |d|² trace inside Voigt
  carries the isotropic part through the Chebyshev channels. Add one head
  slot with a constant [1,1,1,0,0,0] kernel if a per-atom constant is needed.
- No PSD constraint, no v⊗v (angle) basis, no joint μ+α model.
- scale_targets stays mode-1 only (unchanged behaviour).

## Tests

tests/test_neighbor_dipole.py gains a mode-2 block: pair-sum identity vs
numpy, four-path consistency (slot / in-graph / candidates / single),
rotation covariance α' = RαRᵀ + translation invariance, SNES tail size and
gen-0 identity, save/load round trip, Adam smoke. tests/test_gap_dipole.py
gains exact recovery of a known α* in mode 2 on the water `pol` labels.
