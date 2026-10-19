# GAP site models (2b / 3b / many-body decomposition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the GAP dipole/polarisability fit sum several *site models*, each a kernel on its own body-order descriptor (2b radial distribution, 3b angular power spectrum, many-body SOAP), with the 3b model carrying its own angular geometric kernel, so the NEP-style decomposition can be measured on the convex floor against the single-SOAP model.

**Architecture:** A site model is (descriptor builder, FPS sparse set, geometric kernel, amplitude δ). The existing multi-scale plumbing in gap_dipole.py already sums column blocks per (descriptor, sparse set) with their own K_MM prior; this plan generalises it so each block may also carry its own geometry vectors v and its own column width, and adds the three descriptor/kernel builders in a new module. The user's criterion is enforced structurally: a model's kernel is the dot-product kernel between two descriptors of *its own* body order, never mixed. Equivariance is guaranteed by construction: every geometric kernel keeps exactly one bond vector uncontracted (μ, odd) or a symmetric bond-vector pair (α, even); angular information enters only through invariant Legendre weights $P_l(\cos\theta_{jik})$. The long-range field channel (`dipole_field`) stays on the bond basis and is therefore inherited by the SOAP and 2b models.

**Tech Stack:** numpy, ASE neighbour lists, existing gap_dipole.py estimator (gap_fit whitened dual ridge), pytest. No new dependencies.

**Spec / evidence:** docs/gap_dipole_polarisability_models.md §5–7 (why 2b/3b *linear feature* blocks failed; why the cross group and the field column worked). Expectation to test, not assume: the decomposition cannot add information beyond the SOAP site function on the convex floor; its value, if any, is a smaller optimisation problem for the ANN. The plan therefore ends with a validation-selected scan whose result is the deliverable either way.

---

## File structure

- **Create `gap_models.py`** — body-order descriptors and the angular geometric kernel, numpy only:
  `two_body_descriptor`, `three_body_descriptor` (invariants, row-normalised),
  `angular_cross_tensor` (α kernel, Legendre-weighted traceless sym(d_j⊗d_k)),
  `angular_cross_vector` (μ kernel, Legendre-weighted odd three-body vector),
  `legendre` helper. One responsibility: turn a frame into site-model inputs.
- **Modify `gap_dipole.py`** — `_whiten` (per-block width from n//M), `_prior_factors`
  (per-scale δ and width), `_phi_frame` (per-scale v), `fit_gap_dipole`/`predict`
  (site-model list), `GapDipoleModel` (per-model sparse/alpha), `_validate`.
- **Modify `TNEPconfig.py`** — GAPconfig fields `gap_site_models`, `gap_model_delta`,
  `gap_site_channels`, `gap_site_lmax`, `gap_cross_lmax`.
- **Modify `tests/test_gap_dipole.py`** — whitelist new GAP-only fields; new tests.
- **Create `tests/test_gap_models.py`** — descriptor and kernel tests.
- **Create `diagnostics/gap_experiments/gap_site_models_scan.py`** — the scan.

Conventions used below: `_cfg(**kw)` is `tests/test_gap_dipole._cfg` (pins every readout field; set `cfg.types, cfg.num_types = [8, 1], 2` when no fit resolves them); `WATER_XYZ` is `datasets/train_waterbulk.xyz`; `_pairs(atoms, cfg)` means `data._neighbor_dipole_pairs(atoms, rc, C, radial, zeta, cutoff, gauss, 1)` → `(i, j, G[P,3,C])` with `G = R_n(r)·d`, `d = r_i − r_j`, image pairs, self-images dropped. Run tests with `CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 python -m pytest … -q -p no:warnings`.

---

### Task 1: Config fields and validation

**Files:**
- Modify: `TNEPconfig.py` (GAPconfig, after `gap_cross_channels`)
- Modify: `gap_dipole.py:_validate`
- Test: `tests/test_gap_dipole.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gap_dipole.py`:

```python
# ── site models (gap_site_models) ───────────────────────────────────────────

def test_site_model_fields_are_gap_only_and_validated():
    from TNEPconfig import GAPconfig, TNEPconfig
    for f in ("gap_site_models", "gap_model_delta", "gap_site_channels", "gap_site_lmax", "gap_cross_lmax"):
        assert hasattr(GAPconfig, f) and not hasattr(TNEPconfig, f)
    assert GAPconfig.gap_site_models == ["soap"]            # default = today's model
    with pytest.raises(ValueError, match="gap_site_models"):
        gap_dipole._validate(_cfg(gap_site_models=["soap", "4b"]))
    with pytest.raises(ValueError, match="gap_site_models"):
        gap_dipole._validate(_cfg(gap_site_models=[]))
    with pytest.raises(ValueError, match="gap_model_delta"):
        gap_dipole._validate(_cfg(gap_site_models=["soap", "2b"], gap_model_delta={"soap": 1.0}))
    with pytest.raises(ValueError, match="gap_cross_lmax"):
        gap_dipole._validate(_cfg(gap_site_models=["3b"], gap_cross_lmax=-1))
    with pytest.raises(ValueError, match="gap_cross_channels"):
        gap_dipole._validate(_cfg(gap_site_models=["3b"], dipole_radial="chebyshev", dipole_channels=3))
    with pytest.raises(ValueError, match="multi-channel radial"):
        gap_dipole._validate(_cfg(gap_site_models=["2b"]))          # _cfg default radial is "screened"
    gap_dipole._validate(_cfg(gap_site_models=["soap", "2b", "3b"], dipole_radial="chebyshev",
                              dipole_channels=3, gap_cross_channels=2,
                              gap_model_delta={"soap": 1.0, "2b": 0.5, "3b": 0.1}))
```

Also add the five names to the `gap_only` set in `test_configs_are_independent`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gap_dipole.py -k "site_model_fields or configs_are_independent" -q`
Expected: FAIL (`AttributeError: gap_site_models`).

- [ ] **Step 3: Add the fields**

In `TNEPconfig.py`, GAPconfig, directly after `gap_cross_channels: int = 0`:

```python
    # --- site models (GAP only): a kernel per body order, summed ------------
    # Each entry is a site function s_i = Σ_m α_m (D̂_i·D̂_m)^ξ on ITS OWN
    # descriptor with its own FPS sparse set and K_MM prior (never mixed):
    #   "soap" : the SOAP power spectrum (today's model)
    #   "2b"   : radial distribution q_n^{t'} = Σ_{j∈t'} R_n(r_ij)   (gap_models)
    #   "3b"   : angular power spectrum q_{nn'l}^{t't''} =
    #            Σ_{j∈t',k∈t''} R_n(r_ij) R_n'(r_ik) P_l(cos θ_jik)
    # "soap" and "2b" read out through the bond basis (+ field channel);
    # "3b" through its own angular kernel (gap_cross_lmax) — the only way a
    # three-body model keeps an uncontracted bond vector (equivariance).
    gap_site_models: list[str] = ["soap"]
    # Relative amplitude δ per site model (multiplies gap_delta_soap); the
    # prior std of that model's site function. Keys must cover gap_site_models.
    gap_model_delta: dict = {"soap": 1.0, "2b": 1.0, "3b": 1.0}
    # Radial channels C and Legendre order L of the 2b/3b DESCRIPTORS (the
    # readout channels stay dipole_channels). Width: 2b T·C, 3b T(T+1)/2·C²·(L+1).
    gap_site_channels: int = 6
    gap_site_lmax: int = 2
    # Legendre order of the 3b model's angular kernel: channels (a ≤ b, l),
    # l = 0..gap_cross_lmax; l = 0 is the existing sym(v⊗v) cross group.
    gap_cross_lmax: int = 1
```

- [ ] **Step 4: Validate**

In `gap_dipole._validate`, append to `checks`:

```python
        (bool(cfg.gap_site_models) and set(cfg.gap_site_models) <= {"soap", "2b", "3b"},
         f"gap_site_models must be a non-empty subset of soap/2b/3b, got {cfg.gap_site_models!r}"),
        (set(cfg.gap_site_models) <= set(cfg.gap_model_delta)
         and all(float(cfg.gap_model_delta[m]) > 0 for m in cfg.gap_site_models),
         "gap_model_delta needs a positive entry for every entry of gap_site_models"),
        (int(cfg.gap_site_channels) >= 1 and int(cfg.gap_site_lmax) >= 0,
         "gap_site_channels >= 1 and gap_site_lmax >= 0 required"),
        (int(cfg.gap_cross_lmax) >= 0, "gap_cross_lmax must be >= 0"),
        ("3b" not in cfg.gap_site_models or int(cfg.gap_cross_channels) >= 1,
         "the 3b site model reads out through the angular kernel: gap_cross_channels >= 1 required"),
        ("3b" not in cfg.gap_site_models or int(cfg.target_mode) == 2 or int(cfg.gap_cross_lmax) >= 1,
         "3b dipole kernel needs gap_cross_lmax >= 1 (l = 0 collapses to the one-hop basis)"),
        (set(cfg.gap_site_models) <= {"soap"} or cfg.dipole_radial != "screened",
         "2b/3b site models need a multi-channel radial (screened is single-channel)"),
        (not (cfg.gap_2b or cfg.gap_3b) or list(cfg.gap_site_models)[:1] == ["soap"],
         "the TurboGAP 2b/3b tails read out through the bond basis of the first (soap) site model"),
```

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add TNEPconfig.py gap_dipole.py tests/test_gap_dipole.py
git commit -m "feat(gap): site-model config fields (soap/2b/3b decomposition) with validation"
```

---

### Task 2: Body-order descriptors (`gap_models.py`)

**Files:**
- Create: `gap_models.py`
- Test: `tests/test_gap_models.py`

- [ ] **Step 1: Write the failing tests**

```python
"""gap_models.py — body-order site-model descriptors and angular kernels."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest
from ase.io import read
from ase.neighborlist import neighbor_list
import gap_models as gm
from test_gap_dipole import _cfg

WATER = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def _setup(**kw):
    cfg = _cfg(dipole_radial="chebyshev", dipole_cutoff="cosine2", dipole_channels=3,
               dipole_species_pairs="symmetric", gap_site_channels=3, gap_site_lmax=2, **kw)
    cfg.types, cfg.num_types = [8, 1], 2
    at = read(WATER, index="1")
    return cfg, at, np.array([0 if z == 8 else 1 for z in at.numbers])


def _rot(a, b, c):
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return Rz @ Ry @ Rx


def _moved(at, R=None, shift=None):
    b = at.copy()
    if R is not None:
        b.set_cell(at.cell[:] @ R.T, scale_atoms=False); b.positions = at.positions @ R.T
    if shift is not None:
        b.positions = b.positions + shift
    return b


def test_two_body_descriptor_matches_hand_sum_and_is_normalised():
    cfg, at, ti = _setup()
    q = gm.two_body_descriptor(at, ti, cfg)                   # [N, T·C]
    assert q.shape == (96, 2 * 3)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    rc, C = float(cfg.dipole_rc), 3
    i, j, d = neighbor_list("ijd", at, rc)
    keep = i != j; i, j, d = i[keep], j[keep], d[keep]
    x = 2.0 * d / rc - 1.0
    fc = ((1.0 + np.cos(np.pi * d / rc)) / 2.0) ** 2
    R = np.stack([np.ones_like(x), x, 2 * x * x - 1], 1) * fc[:, None]
    ref = np.zeros((96, 2, C)); np.add.at(ref, (i, ti[j]), R)
    ref = ref.reshape(96, -1); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.allclose(q, ref, atol=1e-6)


def test_three_body_descriptor_matches_triplet_loop_and_symmetries():
    cfg, at, ti = _setup()
    q = gm.three_body_descriptor(at, ti, cfg)                 # [N, T(T+1)/2 · C·C · (L+1)]
    assert q.shape == (96, 3 * 9 * 3)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    # explicit triplet loop for one atom
    rc, C, L = float(cfg.dipole_rc), 3, 2
    i, j, D = neighbor_list("ijD", at, rc); keep = i != j; i, j, D = i[keep], j[keep], -D[keep]
    a = 5; sel = i == a; js, ds = j[sel], D[sel]
    r = np.linalg.norm(ds, axis=1); x = 2 * r / rc - 1
    fc = ((1 + np.cos(np.pi * r / rc)) / 2) ** 2
    R = np.stack([np.ones_like(x), x, 2 * x * x - 1], 1) * fc[:, None]
    pidx = np.array([[0, 1], [1, 2]])
    ref = np.zeros((3, C, C, L + 1))
    for p in range(len(js)):
        for k in range(len(js)):
            if p == k: continue
            cos = ds[p] @ ds[k] / (r[p] * r[k])
            P = [1.0, cos, 0.5 * (3 * cos * cos - 1)]
            slot = pidx[ti[js[p]], ti[js[k]]]
            ref[slot] += R[p][:, None, None] * R[k][None, :, None] * np.array(P)[None, None, :]
    ref = ref.reshape(-1); ref /= np.linalg.norm(ref)
    assert np.allclose(q[a], ref, atol=1e-6)
    # invariances
    R3 = _rot(0.6, -0.3, 1.2)
    assert np.allclose(gm.three_body_descriptor(_moved(at, R3), ti, cfg), q, atol=1e-6)
    assert np.allclose(gm.three_body_descriptor(_moved(at, shift=[1.0, -2.0, 0.5]), ti, cfg), q, atol=1e-6)
    assert np.allclose(gm.two_body_descriptor(_moved(at, R3, [0.3, 0.3, 0.3]), ti, cfg),
                       gm.two_body_descriptor(at, ti, cfg), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_gap_models.py -q` → FAIL (`ModuleNotFoundError: gap_models`).

- [ ] **Step 3a: Factor the radial weights out of `data._neighbor_dipole_pairs`**

In `data.py`, move the `cutoff`/`radial` branches of `_neighbor_dipole_pairs`
(from `if cutoff == "cosine2":` through the `else: raise ValueError(f"unknown dipole_radial …")`)
into

```python
def _radial_weights(r, rc, C, radial, zeta, cutoff, gauss_sigma_scale) -> np.ndarray:
    """R_n(r)·f_c(r) → [P, C] (screened: [P, 1]); numpy twin of TNEP._radial."""
```

and call it from `_neighbor_dipole_pairs` (`Rw = _radial_weights(r, rc, C, radial, zeta, cutoff, gauss_sigma_scale)`).
Run `python -m pytest tests/test_neighbor_dipole.py -k "identity or four_path" -q` — must stay green (the numpy twin is unchanged in value).

- [ ] **Step 3b: Implement `gap_models.py`**

```python
"""Body-order site-model descriptors and angular geometric kernels for the GAP
dipole/polarisability fit (gap_site_models). Numpy only.

Descriptors (invariant, row-normalised; the GAP kernel is (q̂_i·q̂_m)^ξ on
descriptors of the SAME body order only):
    2b  q_n^{t'}(i)        = Σ_{j∈t'} R_n(r_ij)
    3b  q_{nn'l}^{t't''}(i) = Σ_{j∈t'} Σ_{k∈t'', k≠j} R_n(r_ij) R_n'(r_ik) P_l(cos θ_jik)
with R_n the readout's radial family on dipole_rc (data._neighbor_dipole_pairs).

Angular geometric kernels for the 3b model (equivariance by construction: the
angle enters as an invariant weight, the tensor character comes from the
uncontracted bond vectors):
    α : W_i^{(ab,l)} = Σ_{j,k≠j} R_a(r_ij) R_b(r_ik) P_l(cos θ) [sym(d_ij d_ikᵀ)]_traceless
    μ : w_i^{(ab,l)} = Σ_{j,k≠j} R_a(r_ij) R_b(r_ik) P_l(cos θ) d_ij           (odd)
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

import data as tnep_data
from TNEPconfig import GAPconfig, resolved_dipole_cutoff


def legendre(cos: np.ndarray, lmax: int) -> np.ndarray:
    """P_0..P_lmax(cos) by recurrence → [..., lmax+1]."""
    P = [np.ones_like(cos), cos]
    for l in range(1, lmax):
        P.append(((2 * l + 1) * cos * P[l] - l * P[l - 1]) / (l + 1))
    return np.stack(P[:lmax + 1], axis=-1)


def _site_pairs(atoms: Atoms, cfg: GAPconfig, C: int):
    """(i, j, d [P,3], R [P,C]) over image pairs within dipole_rc with the
    readout's radial family evaluated for C channels (data._radial_weights,
    the block factored out of _neighbor_dipole_pairs in Task 2 Step 3a)."""
    rc = float(cfg.dipole_rc or cfg.rcut_hard)
    i, j, D = tnep_data._neighbor_pairs(atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    R = tnep_data._radial_weights(np.linalg.norm(d, axis=1), rc, C, str(cfg.dipole_radial),
                                  float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg),
                                  float(cfg.dipole_gauss_sigma_scale))
    return i, j, d, R


def _normalise(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def two_body_descriptor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """[N, T·C] radial distribution per neighbour species, unit rows."""
    C, T = int(cfg.gap_site_channels), int(cfg.num_types)
    i, j, _, R = _site_pairs(atoms, cfg, C)
    q = np.zeros((len(atoms), T, C))
    np.add.at(q, (i, np.asarray(ti)[j]), R)
    return _normalise(q.reshape(len(atoms), -1))


def _triplets(i, j):
    """All ordered neighbour pairs (p, k) of the same centre, p ≠ k, as index
    arrays into the pair list (centres are sorted in ASE's output)."""
    order = np.argsort(i, kind="stable"); i, j = i[order], j[order]
    starts = np.flatnonzero(np.r_[True, i[1:] != i[:-1]]); ends = np.r_[starts[1:], len(i)]
    P, K = [], []
    for s, e in zip(starts, ends):
        idx = np.arange(s, e); pp, kk = np.meshgrid(idx, idx, indexing="ij")
        m = pp != kk; P.append(pp[m]); K.append(kk[m])
    return order, np.concatenate(P), np.concatenate(K)


def three_body_descriptor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """[N, T(T+1)/2·C·C·(L+1)] angular power spectrum, unit rows."""
    C, L, T = int(cfg.gap_site_channels), int(cfg.gap_site_lmax), int(cfg.num_types)
    i, j, d, R = _site_pairs(atoms, cfg, C)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)                           # [Q, L+1]
    from gap_descriptor import pair_index
    slot = pair_index(T)[np.asarray(ti)[j[P]], np.asarray(ti)[j[K]]]
    contrib = np.einsum("qa,qb,ql->qabl", R[P], R[K], Pl)                # [Q, C, C, L+1]
    q = np.zeros((len(atoms), T * (T + 1) // 2, C, C, L + 1))
    np.add.at(q, (i[P], slot), contrib)
    # The C×C block is symmetric in (n, n') under the ordered p≠k sum: redundant
    # but harmless columns of an INVARIANT descriptor (kept for a simple layout).
    return _normalise(q.reshape(len(atoms), -1))


def angular_cross_tensor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """α kernel [N, 6, K_x]: Legendre-weighted traceless sym(d_j ⊗ d_k) over
    channel pairs a ≤ b (species-resolved under pair columns) and l ≤ gap_cross_lmax."""
    Cx, L, T = int(cfg.gap_cross_channels), int(cfg.gap_cross_lmax), int(cfg.num_types)
    pair = str(cfg.dipole_species_pairs) != "off"
    i, j, d, R = _site_pairs(atoms, cfg, Cx)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)
    tj = np.asarray(ti)[j]
    # species-resolved radial weights: k = (a, t') with t' the partner species
    Tp = T if pair else 1
    Rp = np.zeros((len(R), Cx, Tp)); Rp[np.arange(len(R)), :, (tj if pair else 0)] = R
    Rp = Rp.reshape(len(R), -1)                                           # [P, K1]
    K1 = Rp.shape[1]; au, bu = np.triu_indices(K1)
    S = 0.5 * (d[P][:, :, None] * d[K][:, None, :] + d[K][:, :, None] * d[P][:, None, :])   # [Q, 3, 3]
    tr = np.trace(S, axis1=1, axis2=2) / 3.0
    S[:, 0, 0] -= tr; S[:, 1, 1] -= tr; S[:, 2, 2] -= tr
    voigt = np.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 1, 2], S[:, 2, 0]], 1)
    w = 0.5 * (Rp[P][:, au] * Rp[K][:, bu] + Rp[P][:, bu] * Rp[K][:, au])    # [Q, K_x/(L+1)] symmetric in a↔b
    contrib = np.einsum("qd,qc,ql->qdcl", voigt, w, Pl).reshape(len(P), 6, -1)
    W = np.zeros((len(atoms), 6, contrib.shape[2]))
    np.add.at(W, i[P], contrib)
    return W


def angular_cross_vector(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """μ kernel [N, 3, K_x]: Σ R_a(r_ij) R_b(r_ik) P_l(cos θ) d_ij (odd), l ≥ 1
    (l = 0 collapses to the one-hop basis and is skipped)."""
    Cx, L, T = int(cfg.gap_cross_channels), int(cfg.gap_cross_lmax), int(cfg.num_types)
    pair = str(cfg.dipole_species_pairs) != "off"
    i, j, d, R = _site_pairs(atoms, cfg, Cx)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)[:, 1:]                      # drop l = 0
    tj = np.asarray(ti)[j]; Tp = T if pair else 1
    Rp = np.zeros((len(R), Cx, Tp)); Rp[np.arange(len(R)), :, (tj if pair else 0)] = R
    Rp = Rp.reshape(len(R), -1)
    w = Rp[P][:, :, None] * Rp[K][:, None, :]                              # [Q, K1, K1] (a on j, b on k)
    contrib = np.einsum("qd,qab,ql->qdabl", d[P], w, Pl).reshape(len(P), 3, -1)
    out = np.zeros((len(atoms), 3, contrib.shape[2]))
    np.add.at(out, i[P], contrib)
    return out
```

- [ ] **Step 4: Run tests** — Expected: PASS. If the 3b hand loop differs only by the k≠j term, check `_triplets` excludes p == k (it must; the descriptor excludes self-pairing of a neighbour with itself).

- [ ] **Step 5: Commit**

```bash
git add gap_models.py tests/test_gap_models.py
git commit -m "feat(gap): 2b radial-distribution and 3b angular power-spectrum site descriptors"
```

---

### Task 3: Angular kernels — identity with the existing cross group, covariance

**Files:**
- Test: `tests/test_gap_models.py`
- (uses `gap_models.py` from Task 2; no production change)

- [ ] **Step 1: Write the failing tests**

```python
def test_angular_cross_tensor_l0_equals_existing_cross_group_and_is_covariant():
    import gap_dipole
    cfg, at, ti = _setup(target_mode=2, gap_cross_channels=2, gap_cross_lmax=0)
    W = gm.angular_cross_tensor(at, ti, cfg)                 # [N, 6, K1(K1+1)/2]
    X = gap_dipole._cross_group(at, ti, cfg)                 # sym(v⊗v), traceless
    # v⊗v = Σ_{j,k} d_j d_k (INCLUDING j = k); the l=0 angular kernel excludes
    # j = k, so add the bond-axis part back: Σ_j R_a R_b (d_j d_jᵀ)_traceless.
    i, j, d, R = gm._site_pairs(at, cfg, 2)
    tj = ti[j]; Rp = np.zeros((len(R), 2, 2)); Rp[np.arange(len(R)), :, tj] = R; Rp = Rp.reshape(len(R), -1)
    au, bu = np.triu_indices(Rp.shape[1])
    dd = d[:, :, None] * d[:, None, :]; tr = np.trace(dd, axis1=1, axis2=2) / 3
    for k in range(3): dd[:, k, k] -= tr
    voigt = np.stack([dd[:, 0, 0], dd[:, 1, 1], dd[:, 2, 2], dd[:, 0, 1], dd[:, 1, 2], dd[:, 2, 0]], 1)
    diag = np.zeros_like(W); np.add.at(diag, i, voigt[:, :, None] * (Rp[:, au] * Rp[:, bu])[:, None, :])
    assert np.allclose(W + diag, X, atol=1e-5 * np.abs(X).max())
    # covariance: W(Rx) = R W Rᵀ per site; translation invariance
    R3 = _rot(0.4, 0.9, -0.7)
    W2 = gm.angular_cross_tensor(_moved(at, R3), ti, cfg)
    def mat(v): return np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])
    for n in (0, 7, 50):
        for c in range(W.shape[2]):
            assert np.allclose(mat(W2[n, :, c]), R3 @ mat(W[n, :, c]) @ R3.T, atol=1e-5 * np.abs(W).max())
    assert np.allclose(gm.angular_cross_tensor(_moved(at, shift=[2, 1, -1]), ti, cfg), W, atol=1e-6 * np.abs(W).max())


def test_angular_cross_vector_is_odd_and_covariant():
    cfg, at, ti = _setup(target_mode=1, gap_cross_channels=2, gap_cross_lmax=2)
    w = gm.angular_cross_vector(at, ti, cfg)                 # [N, 3, K1²·2]
    assert w.shape[2] == 16 * 2
    R3 = _rot(0.2, 0.5, 0.8)
    w2 = gm.angular_cross_vector(_moved(at, R3), ti, cfg)
    assert np.allclose(w2, np.einsum("ab,nbk->nak", R3, w), atol=1e-5 * np.abs(w).max())
    assert np.allclose(gm.angular_cross_vector(_moved(at, -np.eye(3)), ti, cfg), -w, atol=1e-5 * np.abs(w).max())   # odd
```

- [ ] **Step 2: Run** — Expected: PASS straight away if Task 2 is right; a failure here means a missing `k ≠ j` exclusion or a Voigt/order mismatch — fix `gap_models.py`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gap_models.py
git commit -m "test(gap): angular kernels — l=0 identity with the cross group, covariance, parity"
```

---

### Task 4: Per-block widths and per-scale geometry in the estimator

**Files:**
- Modify: `gap_dipole.py:_whiten`, `_prior_factors`, `_phi_frame`
- Test: `tests/test_gap_dipole.py`

- [ ] **Step 1: Write the failing test**

```python
def test_phi_frame_accepts_per_scale_geometry_and_widths():
    """Scales may carry their own v (different K) and δ; the prior factors and
    whitening follow each block's width. Two scales with identical inputs must
    reproduce the single-scale columns twice."""
    import gap_dipole
    rng = np.random.default_rng(0)
    N, Q = 12, 5
    D = rng.normal(size=(N, Q)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    ti = np.array([0, 1] * 6); sparse = [D[ti == 0][:3], D[ti == 1][:3]]
    v3 = rng.normal(size=(N, 3, 4)); v7 = rng.normal(size=(N, 3, 7))
    one = gap_dipole._phi_frame([(D, sparse, v3, 1.0)], ti, (2,))
    two = gap_dipole._phi_frame([(D, sparse, v3, 1.0), (D, sparse, v3, 1.0)], ti, (2,))
    assert two.shape[1] == 2 * one.shape[1] and np.allclose(two[:, :one.shape[1]], one)
    mixed = gap_dipole._phi_frame([(D, sparse, v3, 1.0), (D, sparse, v7, 0.5)], ti, (2,))
    assert mixed.shape[1] == 6 * 4 + 6 * 7
    cfg = _cfg(); cfg.types, cfg.num_types = [8, 1], 2
    factors = gap_dipole._prior_factors(cfg, [(sparse, 1.0, 4), (sparse, 0.5, 7)], (2,))
    assert [n for n, _ in factors] == [12, 12, 21, 21]
    X = rng.normal(size=(5, mixed.shape[1]))
    Xw = gap_dipole._whiten(X, factors, False)                     # Φ → Φ·(L⁻ᵀ ⊗ I) per block
    back, off = np.empty_like(X), 0
    for n, L in factors:                                           # un-whiten by hand: ·(Lᵀ ⊗ I)
        M = len(L); k = n // M
        back[:, off:off + n] = np.einsum("rmk,mn->rnk", Xw[:, off:off + n].reshape(5, M, k), L.T).reshape(5, n)
        off += n
    assert np.allclose(back, X)
```

- [ ] **Step 2: Run to verify failure** — FAIL (`_phi_frame` signature).

- [ ] **Step 3: Generalise the three functions**

Replace `_phi_frame`'s signature and SOAP loop:

```python
def _phi_frame(scales: list, ti: np.ndarray, xis: tuple[int, ...],
               gap_q=None, deltas=(1.0, 1.0, 1.0), v: np.ndarray | None = None) -> np.ndarray:
    """[D, n_columns] design block for one frame — THE column contract for fit
    AND predict. scales = [(D, sparse_desc, v, delta)] per site model: each
    model's kernel couples ITS descriptors to ITS sparse points only, reads out
    through ITS geometry v [N, D, K] and carries δ² on its columns and prior.
    Block order: model → species → ξ → (m, k). The TurboGAP 2b/3b blocks
    (gap_q) read out through `v` (the bond basis) and follow all models."""
    d_soap, d_2b, d_3b = (float(d) for d in deltas)
    cols = []
    for D, sparse_desc, vm, dm in scales:
        n_rows = vm.shape[1]
        for t, Dm in enumerate(sparse_desc):
            sel = ti == t
            G = D[sel] @ Dm.T
            for xi in xis:
                cols.append(dm ** 2 * np.einsum("nm,nxk->xmk", G ** xi, vm[sel]).reshape(n_rows, -1))
    for q, d_q in zip(gap_q or (), (d_2b, d_3b)):
        if q is not None:
            for t in range(len(scales[0][1])):
                sel = ti == t
                cols.append(d_q ** 2 * np.einsum("ns,nxk->xsk", q[sel], v[sel]).reshape(v.shape[1], -1))
    return np.concatenate(cols, axis=1)
```

`_prior_factors(cfg, scale_sparse, xis)` where `scale_sparse = [(sparse_desc, delta, n_vcols)]`:

```python
    for sparse_desc, dm, k in scale_sparse:
        for Dm in sparse_desc:
            base = Dm @ Dm.T
            for xi in xis:
                out.append((len(Dm) * k, chol(dm ** 2 * base ** xi)))
```

and the TurboGAP 2b/3b tails take the bond-basis width as an explicit argument,
`_prior_factors(cfg, scale_sparse, xis, tail_vcols)` (= `_n_vcols(cfg)`, the width of the
bond basis `v` those tails read out through — NOT `scale_sparse[0][2]`, which is the
first site model's width and differs when "soap" is not first).

`_whiten(X, factors, solve_with_transpose)` — exactly three positional arguments,
width per block derived as `k = n // len(L)`. Do **not** add a fourth positional: the
legacy call `_whiten(Phi, factors, K1, False)` would otherwise run silently with
`solve_with_transpose=K1` (truthy) and give wrong numbers. With three arguments every
stale four-argument caller raises `TypeError` and gets fixed in Step 4. Keep the two
existing branches unchanged (forward: Φ·L⁻ᵀ per block; transpose: α = L⁻ᵀβ, single row).

- [ ] **Step 4: Update every call site** to the new shapes. `grep -n "_phi_frame(\|_prior_factors(\|_whiten(" gap_dipole.py gap_bond.py gap_ewald.py tests/*.py` lists them; at the time of writing:

| file | lines | change |
|---|---|---|
| `gap_dipole.py` `fit_gap_dipole`, `predict` | 576–579, 609, 619, 710–720 | `scales = [(descs[f], sparse_desc, vs[f], d_soap)]` (+ second SOAP scale with the same `vs[f]`); `scale_sparse = [(sparse_desc, d_soap, n_vcols), …]`; `_prior_factors(cfg, scale_sparse, xis, n_vcols)`; `_phi_frame(scales, tis[f], xis, gap_q, deltas, v=vs[f])`; `_whiten(…, False)` / `_whiten(…, True)` |
| `gap_bond.py` | 69 (`_prior_factors`), 80 (`_phi_frame`), 192, 196, 239, 240 (`_whiten`) | same shapes; `Setup.base_factors` becomes `_prior_factors(cfg, [(self.sparse, d_soap, self.n_vcols)], xis, self.n_vcols)` |
| `gap_ewald.py` | 87, 220, 281, 443 (`_prior_factors`); 93, 223, 282, 298, 446 (`_phi_frame`); 97, 98, 149, 150, 229, 230, 284, 285, 450, 451 (`_whiten`) | same; the extra-column fits pass their widened `K1` as the scale width |
| `tests/test_gap_dipole.py` | 447 | `_phi_frame([(descs, model.sparse_desc, v, 1.0)], ti, xi, …)` |
| `tests/test_gap_descriptor.py` | 617, 626–628, 641, 744–745, 765 | 2-tuple scales → 4-tuples with `v` and `deltas[0]`; `v=` keyword for the tails |

Run: `python -m pytest tests/test_gap_dipole.py tests/test_gap_descriptor.py tests/test_gap_models.py -q`
Expected: all pass (the existing α-hash bit-compat and column-count tests pin the legacy layout).
Also run `python -m pytest tests/test_gap_dipole.py -k "cross or field" -q` and the smoke lines of
`diagnostics/gap_experiments/gap_path_check.py` (production vs gap_bond path must still agree to ~1e-12).

- [ ] **Step 5: Commit**

```bash
git add gap_dipole.py tests/test_gap_dipole.py
git commit -m "refactor(gap): per-scale geometry, amplitude and width in the design, prior and whitening"
```

---

### Task 5: Site models in fit and predict

**Files:**
- Modify: `gap_dipole.py` (`GapDipoleModel`, `fit_gap_dipole`, `predict`, run record)
- Test: `tests/test_gap_dipole.py`

- [ ] **Step 1: Write the failing tests**

```python
POL_SITE = dict(target_mode=2, dipole_radial="chebyshev", dipole_channels=3, dipole_cutoff="cosine2",
                dipole_species_pairs="symmetric", gap_site_channels=3, gap_site_lmax=1, gap_cross_channels=2, gap_cross_lmax=1)


def test_site_models_column_count_and_exact_recovery(tmp_path):
    """soap+2b+3b: SOAP and 2b models read out through the bond basis (K_b
    columns each), 3b through its angular kernel (K_x columns); a known α*
    is recovered exactly from synthetic labels."""
    frames = read(WATER_XYZ, index="0:6")
    for f in frames: f.calc = None
    p0 = str(tmp_path / "pol0.xyz"); write(p0, frames)
    kw = dict(POL_SITE, total_N=None, gap_sparse_per_species=6, gap_sigma=1e-7,
              gap_site_models=["soap", "2b", "3b"], gap_model_delta={"soap": 1.0, "2b": 0.7, "3b": 0.3})
    with _chdir(tmp_path):
        base = gap_dipole.fit_gap_dipole(_cfg(data_path=p0, **kw))
    T, C, Cx, L = 2, 3, 2, 1
    K_b = 2 * C * T + (Cx * T) * (Cx * T + 1) // 2        # two bond groups + cross group (gap_cross_channels)
    K_x = (Cx * T) * (Cx * T + 1) // 2 * (L + 1)          # angular kernel channels of the 3b model
    assert [a.size for a in base.site_alpha["soap"]] == [6 * K_b] * T
    assert [a.size for a in base.site_alpha["2b"]] == [6 * K_b] * T
    assert [a.size for a in base.site_alpha["3b"]] == [6 * K_x] * T
    rng = np.random.default_rng(0)
    star = replace(base, site_alpha={m: [rng.normal(size=a.size) for a in al] for m, al in base.site_alpha.items()})
    a_star = gap_dipole.predict(star, frames)
    for f, a in zip(frames, a_star): f.info["pol"] = a
    p1 = str(tmp_path / "pol1.xyz"); write(p1, frames)
    with _chdir(tmp_path):
        refit = gap_dipole.fit_gap_dipole(_cfg(data_path=p1, **kw))
    assert np.allclose(gap_dipole.predict(refit, frames), a_star, atol=1e-8 * max(np.abs(a_star).max(), 1.0))


def test_site_models_3b_alone_dipole_runs_and_record(tmp_path):
    with _chdir(tmp_path):
        m = gap_dipole.fit_gap_dipole(_cfg(**CHEB_PAIRS, gap_site_models=["3b"], gap_cross_channels=2, gap_cross_lmax=2,
                                           gap_sparse_per_species=6))
        rec = json.load(open(sorted(__import__("glob").glob("GAP/*.json"))[-1]))
    assert rec["gap"]["site_models"] == ["3b"] and set(m.site_alpha) == {"3b"}
    assert gap_dipole.predict(m, [read(WATER_XYZ, index="3")]).shape == (1, 3)
```

- [ ] **Step 2: Run to verify failure** — FAIL (`site_alpha`).

- [ ] **Step 3: Implement**

In `GapDipoleModel` replace the fields `sparse_desc`/`alpha` by `site_sparse: dict` and
`site_alpha: dict` (model name → list per species), and expose `sparse_desc` and `alpha`
as **read-only properties** returning the `"soap"` entries so every existing reader
(gap_bond.py, gap_ewald.py, tests) keeps working. The three tests that rebuild a model
with `replace(base, alpha=[…])` (`tests/test_gap_dipole.py:179, 527`,
`tests/test_gap_descriptor.py:653`) change to `replace(base, site_alpha={**base.site_alpha, "soap": […]})`.
The `gap_extra_rcut` second SOAP scale keeps its own fields (`builder2`, `sparse_desc2`,
`alpha2`) and is appended to `scales` after the site models, exactly as today. Legacy
identity ("soap" alone == today's model) is pinned by the existing suites passing after
Task 4/5 (α-hash and explicit-loop tests); no separate test is needed. Add the descriptor/geometry dispatch:

```python
def _site_descriptor(name, builder, frames, tis, cfg):
    if name == "soap": return _descriptors(builder, frames)
    import gap_models as gm
    fn = gm.two_body_descriptor if name == "2b" else gm.three_body_descriptor
    return [fn(a, ti, cfg) for a, ti in zip(frames, tis)]


def _site_geometry(name, frames, tis, cfg, bond_v):
    """Bond basis (+ field channel) for soap/2b; the angular kernel for 3b."""
    if name != "3b": return bond_v
    import gap_models as gm
    fn = gm.angular_cross_tensor if int(cfg.target_mode) == 2 else gm.angular_cross_vector
    return [fn(a, ti, cfg) for a, ti in zip(frames, tis)]
```

In `fit_gap_dipole`: build `models = list(cfg.gap_site_models)`; for each model compute descriptors, FPS sparse set (`_fps_per_species` on that model's descriptors, same rng order as today so "soap" first keeps the legacy FPS bit-identical), geometry list and `delta = d_soap * cfg.gap_model_delta[name]`; `scales[f] = [(desc_m[f], sparse_m, geom_m[f], delta_m) for m]` (+ the `gap_extra_rcut` SOAP scale after, if set); `scale_sparse = [(sparse_m, delta_m, geom_m[0].shape[2]) for m]`; split `alpha_flat` by block sizes into `site_alpha[name]` (per species, ξ-major as today). Record `"site_models"`, `"model_delta"`, `"cross_lmax"`, `"site_channels"`, `"site_lmax"` in the run record. `predict` rebuilds the same per-model inputs via the two dispatchers and `_phi_frame`.

Guards live in `_validate` (Task 1): `"3b"` needs `gap_cross_channels >= 1`; under
`target_mode=1` it also needs `gap_cross_lmax >= 1` whenever it is present (an l = 0
dipole block has zero width and would silently produce dead columns); 2b/3b models need a
multi-channel radial; the TurboGAP tails need "soap" first.

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_gap_dipole.py tests/test_gap_models.py tests/test_gap_descriptor.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add gap_dipole.py tests/test_gap_dipole.py
git commit -m "feat(gap): summed site models — soap / 2b / 3b kernels, 3b with its own angular geometric kernel"
```

---

### Task 6: Validation-selected scan and write-up

**Files:**
- Create: `diagnostics/gap_experiments/gap_site_models_scan.py`
- Modify: `diagnostics/gap_experiments/README.md`, `docs/gap_dipole_polarisability_models.md` (§5/§6 rows)

- [ ] **Step 1: Write the scan** (fit on `train[:400]`, select σ ∈ {1e-3, 3e-3} (μ) / {0.1, 0.3} (α) and δ per model on `train[400:]`, refit the winner on 500, score test; one process per property, **separate training files and `GAP/` record directories per process** — see docs §8 lesson 6). Configurations:

| tag | models | notes |
|---|---|---|
| soap | ["soap"] | today's model (μ with `dipole_field=True`, α with `gap_cross_channels=4`) |
| 2b | ["2b"] | NEP-style 2b alone; μ with field |
| soap+2b | | δ_2b ∈ {0.3, 1} |
| 3b alone | ["3b"] | gap_cross_lmax ∈ {1, 2}; C_x = 4 |
| soap+3b | | α: the 3b model's angular kernel generalises the cross group; test with gap_cross_lmax ∈ {0, 1} (lmax 0 ⇒ the 3b model reads out through exactly the existing cross group, so this row isolates the value of a 3b *site function*) |
| soap+2b+3b | | the full decomposition |

Report RRMSE per the docs definition, plus n_columns and fit time.

- [ ] **Step 2: Run it** (`OMP_NUM_THREADS=8 python diagnostics/gap_experiments/gap_site_models_scan.py`), ~45 min.

- [ ] **Step 3: Record** the table in README.md and add rows to the report's §5/§6 (or a §4.4 if a model wins). Update the project memory note.

- [ ] **Step 4: Commit**

```bash
git add diagnostics/gap_experiments docs/gap_dipole_polarisability_models.md
git commit -m "exp(gap): site-model decomposition scan (soap / 2b / 3b) — results"
```

---

## Fixes to the problems raised in the assessment (where they live in this plan)

1. *Equivariance of the 3b model* — Task 2/3: the angle is a Legendre weight, the tensor character is the uncontracted bond vector(s); parity tested (odd for μ, even for α).
2. *"Power spectrum" readouts are invariant* — the 3b power spectrum is a **descriptor** (site-function input, Task 2); the readout kernel is `angular_cross_*` (Task 2/3). The two are never conflated.
3. *Kernels never mix body orders* — one (descriptor, sparse set, prior) per model, Task 5; the structural test is `test_site_models_column_count_and_exact_recovery`.
4. *Long-range field* — rides on the bond basis, so SOAP and 2b models inherit it (Task 5 `_site_geometry`).
5. *Parameter/column growth* — descriptors are narrow by default (C = 6, L = 2; kernel C_x = 4, l ≤ 1); σ and δ are re-selected per column set (Task 6).
6. *The premise is testable* — Task 6's "soap" row is the control; "2b"-alone shows how much a pure two-body site function recovers; the sum shows whether partitioning changes the convex floor.

## Candidate kernels to test with this machinery (brainstorm, ranked)

Polarisability:
1. **Legendre-weighted cross kernel** (this plan's `angular_cross_tensor`, l = 1, 2 on top of l = 0): the direct generalisation of the group that gave −10 %.
2. **Ewald field-gradient tensor** ∇E_i of the fixed charges (even, traceless, long-range): field-induced anisotropy; one column, Ewald cost already paid.
3. **Field × bond** sym(E_i ⊗ d_ij) R_n(r_ij): local anisotropy modulated by the total field; cheap, odd×odd = even.
4. **Plain long-range dipole tensor group** Σ_j T_ij (no Applequist solve): tests whether the traceless induced-coupling gain was the geometry or the response.

Dipole:
1. **Ewald potential φ_i as an appended descriptor feature** with a small weight (LODE-style concatenation; the *product* kernel failed because it narrowed similarity, concatenation does not).
2. **(∇E_i)·v_i** and **(E_i·d_ij) d_ij R_n**: field-gradient and field-modulated charge flux, odd vectors, cheap.
3. **Sum-of-kernels ξ = [1, 2, 4]** on the SOAP site function (−14 % in earlier dipole work; skipped so far).
4. **Species-split field** (−3 %, already measured) combined with the above once σ is re-selected.

Not worth re-testing: bond-level site functions, QEq channels, screened or smeared fields, iterative self-consistency, Bessel radials, wider cross groups, TurboGAP 2b/3b linear blocks.
