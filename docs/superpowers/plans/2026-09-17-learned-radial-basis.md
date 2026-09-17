# Learned SOAP-turbo Radial Basis — Implementation Plan (Phase 1: forward path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SOAP-turbo radial functions learnable SNES parameters (NEP's `c_nk` analogue) on the GPU descriptor path, with exact reproduction of the fixed descriptor at generation 0.

**Architecture:** The GPU builder emits raw centre coefficients over a K-sized orthonormal poly3 basis in the `descriptors` slot; `TNEP.soap_from_raw` contracts them with a learned `M_radial [T, S, α, K]`, forms the power spectrum, compresses and L2-normalises in-model (per candidate in SNES). Everything downstream of the descriptor (mixing, preprocess, W0, readouts) is unchanged. Spec: `docs/superpowers/specs/2026-09-17-learned-radial-basis-design.md`.

**Tech Stack:** Python 3.10, TensorFlow 2.20, numpy, h5py, pytest. No new dependencies. Env: `conda quipenv` (`/home/reece/miniconda3/envs/quipenv/bin/python`). Run tests with `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest ...` from `/home/reece/TNEP`.

**Scope note:** gradient-consuming modes (PES, `dipole_readout="tnep"`, polarizability) raise `NotImplementedError` under the knob; that is spec §5 / a separate plan. The neighbor dipole readout (the current default) needs no descriptor gradients and is fully supported.

**Commit hygiene (TNEPconfig.py):** the working tree carries the user's live experiment values. Never `git add TNEPconfig.py`. Stage only the new field with the helper in Task 1 Step 6.

---

## File map

| File | Change |
|---|---|
| `TNEPconfig.py` | `radial_basis_size: int \| None = None` (section 2) |
| `DescriptorBuilderGPU.py` | `radial_basis_init`, `radial_basis_channel_map`, `descriptor_input_width` (cfg-only helpers, next to `compute_dim_q`) |
| `DescriptorBuilderGPU_tf.py` | `_compute_cnk_inner_tf`; cache `W_K`; knob-aware locked fns; `build_descriptors_self_only` |
| `data.py` | `descriptor_gradients_needed(cfg)`; use it in `materialize_test_data`, `prepare_eval_data` |
| `TNEP.py` | `M_radial`, `soap_from_raw`, entry hooks in `predict` / `predict_batch` / `predict_batch_candidates`, per-candidate GEMM branch, guards |
| `SNES.py` | `n_radial` tail: layout, reconstruct, split, μ init, type labels, plumbing in `_evaluate_chunk_impl` / `validate`, `_set_model_params` |
| `model_io.py` | save/load `M_radial`; legacy default |
| `MasterTNEP.py` | width cross-check via `descriptor_input_width`; param breakdown row; info line |
| `spectroscopy.py` | width via `descriptor_input_width` (pack + fused TensorSpec) |
| `tests/test_radial_basis.py` | new |

---

### Task 0: GPU builder prerequisite — `build_descriptors_self_only`

**Files:**
- Modify: `DescriptorBuilderGPU_tf.py` (class `DescriptorBuilderGPUTF`, after `build_descriptors`, ~line 2240)
- Modify: `data.py:389-396` (add helper next to `dipole_self_pairs_only`)
- Test: `tests/test_radial_basis.py` (new file, first test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_radial_basis.py
"""Learned SOAP-turbo radial basis (cfg.radial_basis_size)."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import tensorflow as tf
from ase.io import read

from TNEPconfig import TNEPconfig
from DescriptorBuilderGPU import compute_dim_q

CHO_XYZ = os.path.join(ROOT, "datasets", "train.xyz")


def _cho_frames(n=3):
    """First n CHO-only frames of train.xyz (C=6, O=8, H=1)."""
    out = []
    for at in read(CHO_XYZ, index=":40"):
        if set(at.numbers).issubset({6, 8, 1}):
            out.append(at)
        if len(out) == n:
            break
    return out


def _cfg(**kw):
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.dipole_readout = "neighbor"
    cfg.descriptor_mode = 1
    cfg.l_max = 4
    cfg.alpha_max = 4
    cfg.rcut_hard = 4.0
    cfg.rcut_soft = 3.5
    cfg.dipole_rc = 3.5
    cfg.radial_enhancement = 0
    cfg.num_neurons = 8
    cfg.activation = "tanh"
    cfg.descriptor_mixing = False
    cfg.descriptor_preprocess_contract = "off"
    cfg.dipole_channels = 4
    cfg.dipole_species_pairs = "off"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_cutoff = None
    cfg.seed = 7
    cfg.types = [6, 8, 1]
    cfg.num_types = 3
    cfg.type_map = {6: 0, 8: 1, 1: 2}
    cfg.radial_basis_size = None
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.dim_q = compute_dim_q(cfg)
    return cfg


def test_gpu_builder_self_only_neighbor_readout_emits_no_gradients():
    from DescriptorBuilder import make_descriptor_builder
    cfg = _cfg()
    b = make_descriptor_builder(cfg)
    descs, grads, gidx = b.build_descriptors_self_only(_cho_frames(2), batch_size=None)
    assert len(descs) == 2 and descs[0].shape[1] == cfg.dim_q
    assert all(g.shape[0] == 0 for g in grads[0])   # neighbor readout reads no gradients
    assert gidx[0] == [[] for _ in range(len(descs[0]))]


def test_zero_pair_data_stages_on_both_paths():
    """P = 0 (neighbor readout, GPU builder) is a supported invariant through
    pad_and_stack and both chunk-staging paths."""
    from data import (prepare_eval_data, prefetched_chunks, move_data_to_gpu,
                      prestage_chunk_indices, slice_and_complete_chunk)
    cfg = _cfg()
    data = prepare_eval_data(_cho_frames(3), cfg)
    assert tuple(data["grad_values"].shape) == (0, 3, cfg.dim_q)
    ranges = [(0, 2), (2, 3)]
    prestage_chunk_indices(data, ranges)
    for pin in (True, False):
        d = dict(data)
        if not pin:
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)
        chunks = [c for _, _, c in prefetched_chunks(d, ranges, pin_to_cpu=pin)]
        assert [int(c["descriptors"].shape[0]) for c in chunks] == [2, 1]
        assert all(int(c["grad_values"].shape[0]) == 0 for c in chunks)
    c = slice_and_complete_chunk(data, [1, 2])
    assert int(c["pair_struct"].shape[0]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py::test_gpu_builder_self_only_neighbor_readout_emits_no_gradients -q 2>&1 | tail -5`
Expected: FAIL with `AttributeError: 'DescriptorBuilderGPUTF' object has no attribute 'build_descriptors_self_only'`

- [ ] **Step 3: Implement**

In `data.py`, directly after `dipole_self_pairs_only`:

```python
def descriptor_gradients_needed(cfg: TNEPconfig) -> bool:
    """False only for the neighbor dipole readout, which reads no descriptor
    gradients — every other mode contracts ∂q/∂r. NOTE: the neighbor readout
    still needs the PAIR INDEX stream (pair_atom/pair_gidx) on the trajectory
    path, where V is built in-graph — that path keeps calc_gradients=True and
    only the gradient VALUES are skipped (zero-width under the learned basis)."""
    return not (cfg.target_mode == 1 and cfg.dipole_readout == "neighbor")
```

Also in `data.py`, use it at the two eval-set build sites so the neighbor
readout never builds gradients it will not read (both backends):
`materialize_test_data`: `builder.build_descriptors(test_dataset, calc_gradients=descriptor_gradients_needed(cfg), **_kw)`;
`prepare_eval_data`: `builder.build_descriptors(dataset, calc_gradients=descriptor_gradients_needed(cfg))`.

In `DescriptorBuilderGPU_tf.py`, add to `DescriptorBuilderGPUTF` after `build_descriptors`:

```python
    def build_descriptors_self_only(self, dataset: list, batch_size=None,
                                    **build_kwargs):
        """Same triple as `build_descriptors`, reduced to what mode-1 self-only
        callers read: nothing for the neighbor readout (no descriptor
        gradients at all), the self-pair row ∂q_i/∂r_i otherwise.
        `batch_size` is accepted for API parity with the quippy builder (this
        backend already batches by cfg.descriptor_batch_frames)."""
        from data import descriptor_gradients_needed
        if not descriptor_gradients_needed(self.cfg):
            return self.build_descriptors(dataset, calc_gradients=False,
                                          **build_kwargs)
        from DescriptorBuilder import _filter_to_self_pairs
        return _filter_to_self_pairs(
            *self.build_descriptors(dataset, **build_kwargs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: same command. Expected: PASS.

- [ ] **Step 5: Verify end-to-end on the GPU path (fixed basis) and the existing suite**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_neighbor_dipole.py -q 2>&1 | tail -3`
Expected: all pass (record the count; it is the regression baseline for later tasks).

Run:
```bash
python - <<'EOF' 2>&1 | grep -v "^WARNING\|^I0000\|cuda\|oneDNN" | tail -15
from MasterTNEP import train_model
from TNEPconfig import TNEPconfig
cfg = TNEPconfig(); cfg.descriptor_mode = 1; cfg.total_N = 60; cfg.num_generations = 5
cfg.save_path = None; cfg.checkpoint_interval = None; cfg.plot_interval = None; cfg.show_plots = False
train_model(cfg)
EOF
```
Expected: reaches `Run complete!` and prints test-set RMSE lines (record them).

- [ ] **Step 6: Commit**

```bash
git add DescriptorBuilderGPU_tf.py data.py tests/test_radial_basis.py
git commit -m "fix: GPU descriptor builder gains build_descriptors_self_only (neighbor readout crashed with descriptor_mode=1)"
```

---

### Task 1: Config knob + cfg-only helpers (`radial_basis_init`, `radial_basis_channel_map`, `descriptor_input_width`)

**Files:**
- Modify: `TNEPconfig.py` (section 2, after `descriptor_memory_budget_bytes`)
- Modify: `DescriptorBuilderGPU.py` (after `compute_dim_q`, ~line 338)
- Test: `tests/test_radial_basis.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_radial_basis_init_rows_orthonormal_and_identity_at_K_eq_alpha():
    from DescriptorBuilderGPU import radial_basis_init
    M0 = radial_basis_init(4, 8)
    assert M0.shape == (4, 8)
    np.testing.assert_allclose(M0 @ M0.T, np.eye(4), atol=1e-8)
    np.testing.assert_allclose(radial_basis_init(4, 4), np.eye(4), atol=1e-8)
    with pytest.raises(ValueError):
        radial_basis_init(4, 3)
    with pytest.raises(ValueError):
        radial_basis_init(4, 11)


def test_radial_basis_init_maps_K_basis_onto_alpha_basis():
    """M0 @ R_K == R_alpha for the TF radial expansion (single species)."""
    from DescriptorBuilderGPU import (radial_basis_init,
                                      build_orthonormalization_matrix_poly3)
    from DescriptorBuilderGPU_tf import radial_expansion_coeff_poly3_tf
    a, K = 4, 8
    M0 = radial_basis_init(a, K)
    rjs = tf.constant(np.linspace(0.05, 3.99, 200), tf.float64)
    sp = tf.zeros([200], tf.int32); cen = tf.zeros([200], tf.bool)
    kw = dict(n_species=1, rcut_hard=4.0, rcut_soft=3.5, atom_sigma_r=0.5,
              atom_sigma_r_scaling=0.0, amplitude_scaling=1.0, central_weight=1.0,
              radial_enhancement=0, nf=4.0, do_central=True)
    Ra = radial_expansion_coeff_poly3_tf(
        rjs, sp, cen, alpha_max=a,
        W_single=tf.constant(build_orthonormalization_matrix_poly3(a)), **kw).numpy()
    RK = radial_expansion_coeff_poly3_tf(
        rjs, sp, cen, alpha_max=K,
        W_single=tf.constant(build_orthonormalization_matrix_poly3(K)), **kw).numpy()
    np.testing.assert_allclose(M0 @ RK, Ra, atol=1e-7)


def test_radial_basis_channel_map_reproduces_power_spectrum():
    """Pivot-row Gram + gather == the NumPy reference power spectrum, channel
    for channel, on random centre coefficients."""
    from DescriptorBuilderGPU import (radial_basis_channel_map,
                                      make_compress_mask_trivial,
                                      build_multiplicity_array,
                                      power_spectrum_numpy)
    alpha, l_max, S = 4, 4, 3
    n_max = alpha * S; L = l_max + 1; k_max = L * (L + 1) // 2
    cm = radial_basis_channel_map(alpha, l_max, S)
    mi = make_compress_mask_trivial([alpha] * S, l_max)
    assert cm["gather_idx"].shape == (mi["n_compressed"],)
    rng = np.random.default_rng(0)
    cnk = rng.normal(size=(k_max, n_max, 5)) + 1j * rng.normal(size=(k_max, n_max, 5))
    ref = power_spectrum_numpy(
        cnk, build_multiplicity_array(n_max, l_max, mi["skip_mask"]),
        mi["skip_mask"], mi["compressed_idx"], mi["uncompressed_idx"],
        mi["coeffs"], mi["n_compressed"], n_max, l_max)          # [5, Q] normalised
    # Candidate: G[i, p, n', l] = Σ_k Wm[k,l]·Re(c[piv_p,k]·conj c[n',k])
    c = np.transpose(cnk, (2, 1, 0))                             # [N, n_max, k_max]
    piv = c[:, cm["pivot_rows"], :]                              # [N, S, k_max]
    G = np.einsum('ipk,ink,kl->ipnl', piv, np.conj(c), cm["Wm"]).real
    q = G.reshape(5, -1)[:, cm["gather_idx"]] * cm["scale"]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    np.testing.assert_allclose(q, ref, atol=1e-10)


def test_descriptor_input_width():
    from DescriptorBuilderGPU import descriptor_input_width
    cfg = _cfg()
    assert descriptor_input_width(cfg) == cfg.dim_q
    cfg = _cfg(radial_basis_size=8)
    assert descriptor_input_width(cfg) == 3 * 8 * 15 * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q -k "radial_basis_init or channel_map or input_width" 2>&1 | tail -5`
Expected: FAIL with `ImportError: cannot import name 'radial_basis_init'`

- [ ] **Step 3: Add the config field**

In `TNEPconfig.py` section 2, after `descriptor_memory_budget_bytes`:

```python
    # Learnable radial basis (GPU builder only, descriptor_mode=1):
    #   None : fixed SOAP-turbo basis (current behaviour)
    #   K    : int ≥ alpha_max. The builder emits raw centre coefficients over
    #          the K orthonormal poly3 functions; the alpha_max radial
    #          functions per (centre type, neighbour species) become learned
    #          combinations M_radial[T, S, alpha_max, K] in the SNES vector
    #          (NEP's c_nk analogue; init reproduces the fixed basis exactly).
    #          Supported with the neighbor dipole readout (no descriptor
    #          gradients); gradient-consuming modes raise NotImplementedError.
    radial_basis_size: int | None = None
```

- [ ] **Step 4: Add the helpers to `DescriptorBuilderGPU.py`** (after `compute_dim_q`)

```python
def radial_basis_init(alpha_max: int, n_basis: int) -> np.ndarray:
    """M0 [alpha_max, n_basis]: the alpha_max Löwdin-orthonormal poly3
    functions written in the n_basis-orthonormal set, M0 = [W_α | 0] · W_K⁻¹,
    so M0 @ φ_K == φ_α exactly (both sets span I_1..I_α). Rows are
    orthonormal; n_basis == alpha_max gives the identity. W_K⁻¹ = S_K^{1/2} is
    formed from the SVD of the overlap (cond(W_K) ~ 1e6 at K = 8)."""
    if not alpha_max <= n_basis <= 10:
        raise ValueError(
            f"radial_basis_size={n_basis} must satisfy alpha_max={alpha_max} "
            f"<= K <= 10 (poly3 overlap is SPD only up to ~10).")
    W_a = build_orthonormalization_matrix_poly3(alpha_max)
    U, svd, Vt = np.linalg.svd(build_overlap_matrix_poly3(n_basis))
    W_K_inv = U @ np.diag(np.sqrt(svd)) @ Vt
    pad = np.zeros((alpha_max, n_basis - alpha_max), dtype=np.float64)
    return np.concatenate([W_a, pad], axis=1) @ W_K_inv


def radial_basis_channel_map(alpha_max: int, l_max: int, n_species: int) -> dict:
    """Index map from the pivot-row Gram G[s_piv, n', l] to the compressed
    channel order of make_compress_mask_trivial (Fortran (n, n', l) order,
    kept when n or n' is a species' first radial index).

    Returns dict: pivot_rows [S] int32 (0-based global n of each species'
    first function), gather_idx [dim_q] int32 into G flattened as
    (S, n_max, L), scale [dim_q] float32 (√2 for n != n', SOAP-turbo's
    multiplicity), Wm [k_max, L] float32 (2 for m > 0 else 1, block-diagonal
    in l; k = l(l+1)/2 + m)."""
    n_max = alpha_max * n_species
    L = l_max + 1
    k_max = L * (L + 1) // 2
    pivots = [s * alpha_max for s in range(n_species)]
    piv_of = {n: s for s, n in enumerate(pivots)}
    gather: list[int] = []
    scale: list[float] = []
    for n in range(n_max):
        for nprime in range(n, n_max):
            if n not in piv_of and nprime not in piv_of:
                continue
            s, other = ((piv_of[n], nprime) if n in piv_of
                        else (piv_of[nprime], n))
            for l in range(L):
                gather.append((s * n_max + other) * L + l)
                scale.append(np.sqrt(2.0) if n != nprime else 1.0)
    Wm = np.zeros((k_max, L), dtype=np.float32)
    k = 0
    for l in range(L):
        for m in range(l + 1):
            Wm[k, l] = 2.0 if m > 0 else 1.0
            k += 1
    return dict(
        pivot_rows=np.asarray(pivots, dtype=np.int32),
        gather_idx=np.asarray(gather, dtype=np.int32),
        scale=np.asarray(scale, dtype=np.float32),
        Wm=Wm,
    )


def descriptor_input_width(cfg) -> int:
    """Width of the per-atom vector the builder emits in the `descriptors`
    slot: the raw compressed SOAP dim, or under cfg.radial_basis_size the
    flattened raw centre coefficients S·K·k_max·2 (species, κ, (l,m), re/im)."""
    K = getattr(cfg, "radial_basis_size", None)
    if K is None:
        return compute_dim_q(cfg)
    L = int(cfg.l_max) + 1
    return int(cfg.num_types) * int(K) * (L * (L + 1) // 2) * 2
```

- [ ] **Step 5: Run tests to verify they pass**

Run: same command. Expected: 4 passed.

- [ ] **Step 6: Commit (stage only the new config field)**

```bash
python - <<'EOF'
import subprocess, re
base = subprocess.check_output(["git", "show", "HEAD:TNEPconfig.py"], text=True)
work = open("TNEPconfig.py").read()
# Lift exactly the new radial_basis_size block (comment + field) out of the working tree.
m = re.search(r"\n    # Learnable radial basis.*?radial_basis_size: int \| None = None\n", work, re.S)
assert m, "field block not found"
anchor = "    descriptor_memory_budget_bytes: int | None = None\n"
assert anchor in base
staged = base.replace(anchor, anchor + m.group(0).lstrip("\n"), 1)
h = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=staged, text=True).strip()
subprocess.check_call(["git", "update-index", "--cacheinfo", f"100644,{h},TNEPconfig.py"])
print(subprocess.check_output(["git", "diff", "--cached", "--stat"], text=True))
EOF
git diff --cached TNEPconfig.py   # must show ONLY the radial_basis_size block
git add DescriptorBuilderGPU.py tests/test_radial_basis.py
git commit -m "feat: radial_basis_size knob + M0 init and channel-map helpers for the learned radial basis"
```

---

### Task 2: Builder emits raw centre coefficients under the knob

**Files:**
- Modify: `DescriptorBuilderGPU_tf.py`: `_build_cfg_cache` (~1214), new `_compute_cnk_inner_tf` after `_compute_soap_inner_tf` (~1211), `DescriptorBuilderGPUTF.__init__` (`_soap_params`, ~1667) and `_build_locked_compute_fns` (~1756-1840)
- Test: `tests/test_radial_basis.py`

- [ ] **Step 1: Write the failing test**

```python
def _soap_from_raw_np(raw, Z, M, alpha, l_max, S):
    """Independent NumPy reference of TNEP.soap_from_raw (one structure).
    raw [A, S·K·k_max·2]; Z [A]; M [T, S, alpha, K]."""
    from DescriptorBuilderGPU import radial_basis_channel_map
    L = l_max + 1; k_max = L * (L + 1) // 2; K = M.shape[-1]
    A = raw.shape[0]
    r = raw.reshape(A, S, K, k_max, 2).astype(np.float64)
    c_raw = r[..., 0] + 1j * r[..., 1]                                   # [A, S, K, k]
    cc = np.einsum('asnk,askm->asnm', M[Z].astype(np.float64), c_raw)   # [A, S, alpha, k]
    cc = cc.reshape(A, S * alpha, k_max)
    cm = radial_basis_channel_map(alpha, l_max, S)
    G = np.einsum('apk,ank,kl->apnl', cc[:, cm["pivot_rows"]], np.conj(cc), cm["Wm"]).real
    q = G.reshape(A, -1)[:, cm["gather_idx"]] * cm["scale"]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n < 1e-5, 1.0, n)


@pytest.mark.parametrize("K", [4, 8])
def test_builder_raw_coefficients_reproduce_fixed_descriptor(K):
    from DescriptorBuilder import make_descriptor_builder
    from DescriptorBuilderGPU import radial_basis_init, descriptor_input_width
    frames = _cho_frames(3)
    fixed = make_descriptor_builder(_cfg()).build_descriptors_flat(frames, calc_gradients=False)
    cfg = _cfg(radial_basis_size=K)
    b = make_descriptor_builder(cfg)
    raw = b.build_descriptors_flat(frames, calc_gradients=False)
    assert raw[0][0].shape[1] == descriptor_input_width(cfg)
    M0 = np.broadcast_to(radial_basis_init(4, K), (3, 3, 4, K))
    for fr, (desc, _, _, _), (r, _, _, _) in zip(frames, fixed, raw):
        Z = np.array([cfg.type_map[int(z)] for z in fr.numbers])
        q = _soap_from_raw_np(r, Z, M0, 4, 4, 3)
        np.testing.assert_allclose(q, desc, atol=2e-5)
    # Neighbor readout + gradients requested (trajectory path): real pair
    # stream, zero-width gradient values.
    out = b.build_descriptors_flat(frames, calc_gradients=True)
    assert out[0][1].shape == (out[0][2].shape[0], 3, 0) and out[0][2].shape[0] > 0
    # Gradient-consuming readout: not implemented under the knob.
    b2 = make_descriptor_builder(_cfg(radial_basis_size=K, dipole_readout="tnep"))
    with pytest.raises(NotImplementedError):
        b2.build_descriptors_flat(frames, calc_gradients=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q -k reproduce_fixed 2>&1 | tail -5`
Expected: FAIL — `raw[0][0].shape[1]` is 165, not 360/720 (the knob is ignored).

- [ ] **Step 3: Implement**

(a) `_build_cfg_cache`: add after `kept_triples=...`:
```python
        n_basis=(int(soap_params["radial_basis_size"])
                 if soap_params.get("radial_basis_size") is not None else None),
        W_K_np=(build_orthonormalization_matrix_poly3(int(soap_params["radial_basis_size"]))
                if soap_params.get("radial_basis_size") is not None else None),
```

(b) New inner function after `_compute_soap_inner_tf`:
```python
@tf.function(reduce_retracing=True)
def _compute_cnk_inner_tf(
    rjs, thetas, phis, pair_atom, pair_neighbour_species,
    pair_is_central, pair_active, W_K, preflm,
    *,
    n_atoms, n_species, n_basis, l_max, rcut_hard, rcut_soft,
    atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
    amplitude_scaling, central_weight, radial_enhancement, nf, do_central,
):
    """Learned-radial-basis forward: raw centre coefficients over the n_basis
    orthonormal poly3 functions, stopping before the power spectrum (which
    TNEP.soap_from_raw forms per candidate). Returns float32
    [n_atoms, S·K·k_max·2] laid out (species, κ, k=(l,m), re/im)."""
    is_fp32 = rjs.dtype == tf.float32
    rjs_r = tf.cast(rjs, tf.float64) if is_fp32 else rjs
    W_r = tf.cast(W_K, tf.float64) if is_fp32 else W_K
    R = radial_expansion_coeff_poly3_tf(
        rjs_r, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=n_basis,
        rcut_hard=rcut_hard, rcut_soft=rcut_soft,
        atom_sigma_r=atom_sigma_r, atom_sigma_r_scaling=atom_sigma_r_scaling,
        amplitude_scaling=amplitude_scaling, central_weight=central_weight,
        radial_enhancement=radial_enhancement, nf=nf, do_central=do_central,
        W_single=W_r,
    )
    if is_fp32:
        R = tf.cast(R, rjs.dtype)
    A = angular_expansion_coeff_tf(
        rjs, thetas, phis, pair_active,
        l_max=l_max, atom_sigma_t=atom_sigma_t,
        atom_sigma_t_scaling=atom_sigma_t_scaling,
        rcut=rcut_hard, preflm=preflm,
    )
    cnk = aggregate_cnk_tf(R, A, pair_atom, n_atoms)          # [k_max, S·K, N]
    cnk = tf.transpose(cnk, [2, 1, 0])                        # [N, S·K, k_max]
    out = tf.stack([tf.math.real(cnk), tf.math.imag(cnk)], axis=-1)
    k_max = (l_max + 1) * (l_max + 2) // 2
    return tf.reshape(tf.cast(out, tf.float32),
                      [-1, n_species * n_basis * k_max * 2])
```

(c) `DescriptorBuilderGPUTF.__init__`: add `radial_basis_size=getattr(cfg, "radial_basis_size", None),` to `self._soap_params`. Add `self._learn_radial = self._soap_params["radial_basis_size"] is not None`.

(d) `_build_locked_compute_fns`: after the existing `fwd_only_fn` definition and before the cache store, add:
```python
        if self._learn_radial:
            # Learned radial basis: the forward emits raw centre coefficients
            # (TNEP.soap_from_raw finishes the descriptor); gradients through
            # the learned basis are not implemented yet (spec §5).
            with tf.device(device):
                W_K_t = tf.constant(cache["W_K_np"], dtype=dtype)
            self._W_K_t = W_K_t
            cnk_static = dict(
                n_species=n_species, n_basis=int(cache["n_basis"]),
                l_max=int(sp["l_max"]), rcut_hard=float(sp["rcut_hard"]),
                rcut_soft=float(sp["rcut_soft"]),
                atom_sigma_r=float(sp["atom_sigma_r"]),
                atom_sigma_r_scaling=float(sp["atom_sigma_r_scaling"]),
                atom_sigma_t=float(sp["atom_sigma_t"]),
                atom_sigma_t_scaling=float(sp["atom_sigma_t_scaling"]),
                amplitude_scaling=float(sp["amplitude_scaling"]),
                central_weight=float(sp["central_weight"]),
                radial_enhancement=int(sp["radial_enhancement"]),
                nf=4.0, do_central=(float(sp["central_weight"]) != 0.0),
            )

            @tf.function(input_signature=sig_fwd, jit_compile=jit_compile)
            def fwd_only_fn(rjs, thetas, phis, pair_atom, pair_neigh,
                            pair_central, pair_active, n_atoms):
                return _compute_cnk_inner_tf(
                    rjs, thetas, phis, pair_atom, pair_neigh,
                    pair_central, pair_active, W_K_t, preflm_t,
                    n_atoms=n_atoms, **cnk_static)

            from data import descriptor_gradients_needed
            _grads_needed = descriptor_gradients_needed(self.cfg)

            def with_grad_fn(rjs, thetas, phis, pair_atom, pair_gidx,
                             pair_neigh, pair_central, pair_active, n_atoms):
                if _grads_needed:
                    raise NotImplementedError(
                        "cfg.radial_basis_size: descriptor gradients through "
                        "the learned radial basis are not implemented (PES / "
                        "tnep readout / polarizability). Use the neighbor "
                        "dipole readout, or set radial_basis_size=None.")
                # Neighbor readout on the trajectory path: it needs the pair
                # index stream (V is built in-graph) but no gradient values.
                soap = fwd_only_fn(rjs, thetas, phis, pair_atom, pair_neigh,
                                   pair_central, pair_active, n_atoms)
                return soap, tf.zeros([tf.shape(rjs)[0], 3, 0], soap.dtype)
```
(`sig_fwd`, `preflm_t`, `device`, `cache`, `sp`, `n_species`, `dtype` are already in scope there.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q 2>&1 | tail -5`
Expected: all passed (the K=4 and K=8 parametrisations both agree with the fixed builder to 2e-5).

- [ ] **Step 5: Commit**

```bash
git add DescriptorBuilderGPU_tf.py tests/test_radial_basis.py
git commit -m "feat: GPU builder emits raw centre coefficients over a K-basis under radial_basis_size"
```

---

### Task 3: Model — `M_radial`, `soap_from_raw`, forward hooks

**Files:**
- Modify: `TNEP.py`: `__init__` (after `self.dim_q_forward` is set, ~line 85, and guards), new methods after `_W0_preprocess_eff_nep4` (~line 685), `predict` (~706), `predict_batch` (~1258), `predict_batch_candidates` (~1348-1440)
- Test: `tests/test_radial_basis.py`

- [ ] **Step 1: Write the failing tests**

```python
def _model(cfg):
    from TNEP import TNEP
    tf.keras.utils.set_random_seed(0)
    return TNEP(cfg)


def _eval_data(cfg, frames):
    from data import prepare_eval_data
    return prepare_eval_data(frames, cfg)


def _predict(model, data, M=None):
    return model.predict_batch(
        data["descriptors"], data["grad_values"],
        data["pair_atom"], data["pair_gidx"], data["pair_struct"],
        data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        W_atom=data.get("V_atom"), M_radial=M).numpy()


@pytest.mark.parametrize("K", [4, 8])
def test_model_soap_from_raw_matches_numpy_reference(K):
    cfg = _cfg(radial_basis_size=K)
    model = _model(cfg)
    frames = _cho_frames(2)
    data = _eval_data(cfg, frames)
    rng = np.random.default_rng(3)
    M = (model.M_radial.numpy() + 0.1 * rng.normal(size=model.M_radial.shape)).astype(np.float32)
    q_tf = model.soap_from_raw(data["descriptors"], data["Z_int"], tf.constant(M)).numpy()
    for s, fr in enumerate(frames):
        A = len(fr)
        Z = data["Z_int"][s].numpy()[:A]
        ref = _soap_from_raw_np(data["descriptors"][s].numpy()[:A], Z, M, 4, 4, 3)
        np.testing.assert_allclose(q_tf[s, :A], ref, atol=1e-5)
    # padded atoms must be exactly zero (norm guard on an empty row)
    A_max = data["descriptors"].shape[1]
    for s, fr in enumerate(frames):
        assert np.all(q_tf[s, len(fr):] == 0.0)


def test_learned_model_at_init_equals_fixed_model():
    """Same ANN weights, fixed vs learned(K=8) at M0 → identical dipoles."""
    frames = _cho_frames(3)
    cfg_f = _cfg(); m_f = _model(cfg_f); d_f = _eval_data(cfg_f, frames)
    cfg_l = _cfg(radial_basis_size=8); m_l = _model(cfg_l); d_l = _eval_data(cfg_l, frames)
    for w in ("W0", "b0", "W1", "b1"):
        getattr(m_l, w).assign(getattr(m_f, w))
    np.testing.assert_allclose(_predict(m_l, d_l), _predict(m_f, d_f), rtol=1e-4, atol=1e-6)


def test_candidates_match_single_candidate_predictions():
    cfg = _cfg(radial_basis_size=6, dipole_species_pairs="symmetric")
    model = _model(cfg)
    data = _eval_data(cfg, _cho_frames(3))
    rng = np.random.default_rng(5)
    C = 3
    M0 = model.M_radial.numpy()
    Ms = np.stack([M0 + 0.05 * c * rng.normal(size=M0.shape) for c in range(C)]).astype(np.float32)
    W0 = np.stack([model.W0.numpy()] * C); b0 = np.stack([model.b0.numpy()] * C)
    W1 = np.stack([model.W1.numpy()] * C); b1 = np.zeros(C, np.float32)
    cp = np.stack([model.c_pair.numpy()] * C)
    preds = model.predict_batch_candidates(
        data["descriptors"], data["V_atom"], data["Z_int"], data["atom_mask"],
        tf.constant(W0), tf.constant(b0), tf.constant(W1), tf.constant(b1),
        c_pair=tf.constant(cp), M_radial=tf.constant(Ms)).numpy()         # [C, B, 3]
    for c in range(C):
        np.testing.assert_allclose(preds[c], _predict(model, data, tf.constant(Ms[c])),
                                   rtol=1e-4, atol=1e-6)


def test_learned_radial_rejects_gradient_modes_and_cpu_builder():
    with pytest.raises(NotImplementedError):
        _model(_cfg(radial_basis_size=8, dipole_readout="tnep"))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=8, descriptor_mode=0))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=3))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_size=11))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q -k "soap_from_raw_matches or equals_fixed or candidates_match or rejects" 2>&1 | tail -5`
Expected: FAIL (`AttributeError: ... has no attribute 'soap_from_raw'` / unexpected kwarg `M_radial` / no error raised).

- [ ] **Step 3: Implement in `TNEP.py`**

(a) In `__init__`, immediately BEFORE `self.builder = make_descriptor_builder(cfg)` (so a bad cfg fails before quippy / the GPU builder are constructed):
```python
        # Learned radial basis (cfg.radial_basis_size = K): the builder emits
        # raw centre coefficients [.., A, S·K·k_max·2] in the descriptors slot
        # and soap_from_raw finishes the SOAP-turbo descriptor with M_radial.
        _K = getattr(cfg, "radial_basis_size", None)
        self._learn_radial = _K is not None
        if self._learn_radial:
            from DescriptorBuilderGPU import (radial_basis_init,
                                              radial_basis_channel_map)
            K = int(_K)
            alpha = int(cfg.alpha_max)
            if int(cfg.descriptor_mode) != 1:
                raise ValueError(
                    "cfg.radial_basis_size requires descriptor_mode=1 (the "
                    "TF/GPU builder); the quippy backend cannot emit raw "
                    "centre coefficients.")
            if not alpha <= K <= 10:
                raise ValueError(
                    f"cfg.radial_basis_size={K} must satisfy alpha_max="
                    f"{alpha} <= K <= 10 (the poly3 overlap matrix is SPD "
                    f"only up to ~10).")
            if not (cfg.target_mode == 1
                    and str(cfg.dipole_readout) == "neighbor"):
                raise NotImplementedError(
                    "cfg.radial_basis_size: descriptor gradients through the "
                    "learned radial basis are not implemented — only the "
                    "neighbor dipole readout (target_mode=1) is supported.")
            cm = radial_basis_channel_map(alpha, int(cfg.l_max), cfg.num_types)
            if int(cm["gather_idx"].size) != int(self.dim_q_forward):
                raise RuntimeError(
                    f"radial_basis_channel_map gives {cm['gather_idx'].size} "
                    f"channels, dim_q={self.dim_q_forward}.")
            L = int(cfg.l_max) + 1
            self._rb_alpha, self._rb_K = alpha, K
            self._rb_L, self._rb_kmax = L, L * (L + 1) // 2
            self._rb_pivots = tf.constant(cm["pivot_rows"])
            self._rb_gather = tf.constant(cm["gather_idx"])
            self._rb_scale = tf.constant(cm["scale"])
            # m-multiplicity per k within each l block (2 for m > 0, else 1).
            self._rb_mw = [tf.constant(cm["Wm"][l * (l + 1) // 2:
                                              l * (l + 1) // 2 + l + 1, l]
                                       [:, tf.newaxis])
                           for l in range(L)]
            M0 = radial_basis_init(alpha, K)
            self.M_radial = tf.Variable(
                np.broadcast_to(M0, (cfg.num_types, cfg.num_types, alpha, K))
                .astype(np.float32).copy(),
                trainable=False, name="M_radial")
        else:
            self.M_radial = None
```
NOTE: this block must sit AFTER `self.dim_q_forward` is assigned (it is, just above `self.builder`).

(b) New methods after `_W0_preprocess_eff_nep4`:
```python
    # ---- learned radial basis ---------------------------------------------

    @property
    def descriptor_input_width(self) -> int:
        """Width of the per-atom vector in the `descriptors` slot (see
        DescriptorBuilderGPU.descriptor_input_width). Distinct from
        dim_q_forward, the width of the W0 matmul after the preprocess fold."""
        from DescriptorBuilderGPU import descriptor_input_width
        return descriptor_input_width(self.cfg)

    def soap_from_raw(self, raw: tf.Tensor, Z: tf.Tensor,
                      M: tf.Tensor | None = None) -> tf.Tensor:
        """SOAP-turbo descriptor from raw centre coefficients with the learned
        radial basis (cfg.radial_basis_size):
            c[a, (s,n), k] = Σ_κ M[t_a, s, n, κ] · c_raw[a, s, κ, k]
            q[a, (n,n',l)] = Σ_m mult · Re(c_{n l m} conj c_{n' l m})   (trivial
            compression: n or n' a species pivot), then L2-normalised per atom
            with SOAP-turbo's ‖q‖ < 1e-5 guard (padded atoms stay 0).
        Args:
            raw : [..., A, S·K·k_max·2]  builder output (species, κ, k, re/im)
            Z   : [..., A]               type indices
            M   : [T, S, α, K] (None → self.M_radial) or per-candidate
                  [C, T, S, α, K]
        Returns [..., A, Q], or [C, ..., A, Q] for per-candidate M.
        """
        M = self.M_radial if M is None else M
        S, K, kmax, alpha, L = (self.num_types, self._rb_K, self._rb_kmax,
                                self._rb_alpha, self._rb_L)
        per_cand = len(M.shape) == 5
        lead = tf.shape(raw)[:-1]
        N = tf.reduce_prod(lead)
        r = tf.reshape(raw, [N, S, K, kmax * 2])                 # [N, S, K, 2k]
        Zf = tf.reshape(Z, [N])
        M_at = tf.gather(M, Zf, axis=-4)                          # [(C,) N, S, α, K]
        if per_cand:
            cc = tf.einsum('cnsak,nskj->cnsaj', M_at, r)          # [C, N, S, α, 2k]
            cc = tf.reshape(cc, [-1, N, S * alpha, kmax, 2])
        else:
            cc = tf.einsum('nsak,nskj->nsaj', M_at, r)
            cc = tf.reshape(cc, [1, N, S * alpha, kmax, 2])
        piv = tf.gather(cc, self._rb_pivots, axis=2)              # [C, N, S, k, 2]
        G_l = []
        for l in range(L):
            k0 = l * (l + 1) // 2
            p_l = piv[:, :, :, k0:k0 + l + 1, :] * self._rb_mw[l]
            c_l = cc[:, :, :, k0:k0 + l + 1, :]
            G_l.append(tf.einsum('cnpkr,cnmkr->cnpm', p_l, c_l))   # [C, N, S, S·α]
        G = tf.stack(G_l, axis=-1)                                # [C, N, S, S·α, L]
        G = tf.reshape(G, [-1, N, S * S * alpha * L])
        q = tf.gather(G, self._rb_gather, axis=-1) * self._rb_scale
        norm = tf.norm(q, axis=-1, keepdims=True)
        q = q / tf.where(norm < 1e-5, tf.ones_like(norm), norm)
        out_shape = tf.concat([lead, [self.dim_q_forward]], axis=0)
        if per_cand:
            return tf.reshape(q, tf.concat([[-1], out_shape], axis=0))
        return tf.reshape(q[0], out_shape)
```

(c) `predict`: add parameter `M_radial: tf.Tensor | None = None` and, as the first statement of the body:
```python
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
```
(d) `predict_batch`: add `M_radial: tf.Tensor | None = None` after `c_pair`; docstring line `M_radial : [T, S, α, K] learned radial basis (None → self.M_radial)`; first statement of the body:
```python
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
```
(e) `predict_batch_candidates`: add `M_radial: tf.Tensor | None = None` after `c_pair` (docstring: `M_radial : [C, T, S, α, K]`). Replace the block from `B = tf.shape(descriptors)[0]` through `pre_h = tf.add_n(pre_h_terms)` with:
```python
        B = tf.shape(descriptors)[0]
        A = tf.shape(descriptors)[1]
        C = tf.shape(W0)[0]

        # Type masks [B, A, 1] — C-independent, reused for both matmul directions.
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(T)
        ]

        # Per-candidate mixing: absorb U_pairᵀ into W0 so forward/backward stay
        # in raw-desc space (de_dq combines with raw grad_values). See _W0_eff.
        if self.descriptor_mixing and U_pair is not None:
            W0 = self._W0_eff(W0, U_pair)

        # Preprocess fold: W0 Q_new → Q_raw so the matmul below is uniform at
        # Q_raw and de_dq comes out at Q_raw for the raw-W_atom sum. Mutually
        # exclusive with mixing.
        if self.descriptor_preprocess_contract != "off":
            W0 = self._W0_preprocess_eff(W0, W_pre_override=W_pre_angular)

        # ── Forward: input→hidden ─────────────────────────────────────────────
        pre_h_terms = []
        if self._learn_radial:
            # Learned radial basis: descriptors are per candidate [C, B, A, Q],
            # so the input→hidden product is a batched GEMM over C per type.
            assert M_radial is not None, "candidates need per-candidate M_radial [C, T, S, α, K]"
            desc_c = tf.reshape(
                self.soap_from_raw(descriptors, Z, M_radial), [C, B * A, Q])
            for t in range(T):
                ph = tf.reshape(tf.matmul(desc_c, W0[:, t, :, :]), [C, B, A, H])
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        else:
            # Per type: [B*A, Q] @ [Q, C*H] → [C, B, A, H]. One GEMM/type, not C.
            desc_flat = tf.reshape(descriptors, [B * A, Q])
            for t in range(T):
                W0_t     = W0[:, t, :, :]                                              # [C, Q, H]
                W0_t_mat = tf.reshape(tf.transpose(W0_t, [1, 0, 2]), [Q, C * H])      # [Q, C*H]
                ph_flat  = tf.matmul(desc_flat, W0_t_mat)                             # [B*A, C*H]
                ph       = tf.transpose(tf.reshape(ph_flat, [B, A, C, H]), [2, 0, 1, 3])  # [C,B,A,H]
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        pre_h = tf.add_n(pre_h_terms)  # [C, B, A, H]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 5: Run the existing neighbor-readout suite for regressions**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_neighbor_dipole.py -q 2>&1 | tail -3`
Expected: same pass count as before the change (record it).

- [ ] **Step 6: Commit**

```bash
git add TNEP.py tests/test_radial_basis.py
git commit -m "feat: TNEP.soap_from_raw + M_radial — learned radial basis finishes the SOAP descriptor in-model"
```

---

### Task 4: SNES — `M_radial` in the parameter vector

**Files:**
- Modify: `SNES.py`: `_set_model_params` (31-58), `__init__` tail sizing (~309-337), `_build_mu_init` (~458-513), `_build_type_of_variable` (~627-687), `_split_reconstructed` (~1401-1425), `reconstruct_params_tf` (~1500-1523), `validate` (~1268-1300, ~1370-1380), `_evaluate_chunk_impl` (~1990-2000)
- Modify: `MasterTNEP.py:_print_param_breakdown` (~372)
- Test: `tests/test_radial_basis.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_snes_tail_holds_M0_and_roundtrips():
    from SNES import _set_model_params
    cfg = _cfg(radial_basis_size=8)
    model = _model(cfg)
    snes = model.optimizer
    T, S, a, K = 3, 3, 4, 8
    assert snes.n_radial == T * S * a * K
    assert snes.dim == snes.n_anns_total + snes.n_U_pair + snes.n_preprocess + snes.n_radial
    named = snes._split_reconstructed(snes.reconstruct_params_tf(snes.mu))
    np.testing.assert_allclose(named["M_radial"].numpy(), model.M_radial.numpy(), atol=1e-7)
    tov = snes._build_type_of_variable()
    assert tov.shape == (snes.dim,)
    np.testing.assert_array_equal(tov[-snes.n_radial:], np.repeat(np.arange(T), S * a * K))
    # batched reconstruct + assign into a fresh model
    pv = tf.stack([snes.mu, snes.mu + 0.01])
    named_b = snes._split_reconstructed(snes.reconstruct_params_tf(pv))
    assert named_b["M_radial"].shape == (2, T, S, a, K)
    m2 = _model(cfg)
    _set_model_params(m2, *snes.reconstruct_params_tf(snes.mu + 0.01))
    np.testing.assert_allclose(m2.M_radial.numpy(), named_b["M_radial"][1].numpy(), atol=1e-7)


def test_snes_gen0_fitness_matches_fixed_basis():
    """Same seed ⇒ identical ANN μ; learned tail at M0 ⇒ identical fitness."""
    frames = _cho_frames(6)
    out = {}
    for K in (None, 8):
        cfg = _cfg(radial_basis_size=K, pop_size=2)
        model = _model(cfg)
        data = _eval_data(cfg, frames)
        data["types_contained"] = tf.ones((len(frames), 3))
        snes = model.optimizer
        fit = snes.evaluate_population(tf.stack([snes.mu, snes.mu]), data).numpy()
        out[K] = (fit, snes.validate(data, snes.mu))
    np.testing.assert_allclose(out[8][0], out[None][0], rtol=1e-4)
    assert abs(out[8][1] - out[None][1]) < 1e-5 * max(1.0, out[None][1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q -k snes 2>&1 | tail -5`
Expected: FAIL (`AttributeError: 'SNES' object has no attribute 'n_radial'`).

- [ ] **Step 3: Implement in `SNES.py`**

(a) `__init__`, replace `self.dim = self.n_anns_total + self.n_U_pair + self.n_preprocess` with:
```python
        # Learned radial basis tail: M_radial [T, S, α, K], centre-type-major.
        self.n_radial = (int(np.prod(model.M_radial.shape))
                         if getattr(model, "M_radial", None) is not None else 0)
        self.dim = (self.n_anns_total + self.n_U_pair + self.n_preprocess
                    + self.n_radial)
```
(b) `_build_mu_init`, before `return mu`:
```python
        # Learned radial basis: M0 (reproduces the fixed descriptor at gen 0).
        if self.n_radial > 0:
            rb_start = self.dim - self.n_radial
            mu[rb_start:] = self.model.M_radial.numpy().reshape(-1)
```
(c) `_build_type_of_variable`, before `if tail_labels_parts:`:
```python
        # M_radial [T, S, α, K] is T-major: each centre type's radial functions
        # are ranked by that type's fitness.
        if self.n_radial > 0:
            tail_labels_parts.append(
                np.repeat(np.arange(T, dtype=np.int32), self.n_radial // T))
```
(d) `_split_reconstructed`: add `"M_radial": None` to the initial dict and, at the end before `return out`:
```python
        if self.n_radial > 0 and idx < len(params):
            out["M_radial"] = params[idx]; idx += 1
```
(e) `reconstruct_params_tf`, before `return tail`:
```python
        if self.n_radial > 0:
            shape = list(self.model.M_radial.shape)
            M = tf.reshape(param_vectors[..., offset:offset + self.n_radial],
                           ([-1] + shape) if is_batched else shape)
            offset += self.n_radial
            tail = tail + (M,)
```
(f) `_set_model_params` (module function), append:
```python
    if (idx < len(params)
            and getattr(model, "M_radial", None) is not None):
        model.M_radial.assign(params[idx]); idx += 1
```
(g) `_evaluate_chunk_impl`, in the `predict_batch_candidates` call add `M_radial=named.get("M_radial")`.
(h) `validate`: in the `mu_tf is not None and not _cache_hit` branch add `M_val = named.get("M_radial")` and store it as a 10th element of the `"fold"` tuple; in the `_cache_hit` branch unpack it; in the model-weights branch set `M_val = None`. Pass `M_radial=M_val` to `predict_batch` in `_consume`.
(i) `MasterTNEP._print_param_breakdown`, after the preprocess row:
```python
    if int(getattr(opt, "n_radial", 0)) > 0:
        rows.append(("radial basis  M_radial [T·S·α·K]", int(opt.n_radial)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add SNES.py MasterTNEP.py tests/test_radial_basis.py
git commit -m "feat: M_radial joins the SNES parameter vector (tail after W_pre; per-centre-type ranking)"
```

---

### Task 5: Save / load

**Files:**
- Modify: `model_io.py`: `_LEGACY_FIELD_DEFAULTS` (~21-56), `save_model` weights group (~216), `_load_weights` signature+body (427-483), `_load_model_h5` weights dict (~510-523), `_load_model_npz` call (~590)
- Test: `tests/test_radial_basis.py`

- [ ] **Step 1: Write the failing test**

```python
def test_save_load_roundtrip_preserves_M_radial_and_predictions(tmp_path):
    from model_io import save_model, load_model
    cfg = _cfg(radial_basis_size=6)
    model = _model(cfg)
    rng = np.random.default_rng(9)
    model.M_radial.assign((model.M_radial.numpy()
                           + 0.2 * rng.normal(size=model.M_radial.shape)).astype(np.float32))
    data = _eval_data(cfg, _cho_frames(2))
    before = _predict(model, data)
    path = str(tmp_path / "m.h5")
    save_model(model, cfg, path)
    loaded = load_model(path)
    assert loaded.cfg.radial_basis_size == 6
    np.testing.assert_allclose(loaded.M_radial.numpy(), model.M_radial.numpy())
    np.testing.assert_allclose(_predict(loaded, data), before, rtol=1e-5, atol=1e-7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q -k save_load 2>&1 | tail -5`
Expected: FAIL (loaded `M_radial` is M0, not the perturbed one; or the load raises).

- [ ] **Step 3: Implement**

- `_LEGACY_FIELD_DEFAULTS`: add `"radial_basis_size": None,` with the comment `# Pre-existence: fixed radial basis.`
- `save_model`, after the `W_pre_angular` block:
```python
        # Learned radial basis coefficients (cfg.radial_basis_size).
        if getattr(model, "M_radial", None) is not None:
            wg.create_dataset("M_radial", data=model.M_radial.numpy())
```
- `_load_weights`: add `M_radial=None` to the signature; at the end:
```python
    if getattr(model, "M_radial", None) is not None:
        if M_radial is None:
            raise ValueError(
                "cfg.radial_basis_size is set but the save has no M_radial "
                "dataset — the learned radial basis would silently reset to "
                "its init. Re-train, or set radial_basis_size=None.")
        if tuple(M_radial.shape) != tuple(model.M_radial.shape):
            raise ValueError(
                f"saved M_radial shape {tuple(M_radial.shape)} != model "
                f"{tuple(model.M_radial.shape)} (alpha_max / "
                f"radial_basis_size / num_types changed).")
        model.M_radial.assign(M_radial)
```
- `_load_model_h5`: add `"M_radial": wg["M_radial"][:] if "M_radial" in wg else None,` to `weights`.
- `_load_model_npz`: add `M_radial=data.get("M_radial")` to the `_load_weights` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py -q 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add model_io.py tests/test_radial_basis.py
git commit -m "feat: persist M_radial in model saves; legacy default for radial_basis_size"
```

---

### Task 6: Driver + inference width sites

**Files:**
- Modify: `MasterTNEP.py`: width cross-check (~459-465), info print after the neighbor-readout print (~503)
- Modify: `spectroscopy.py`: `_get_fused_predict` dim_q (~632-636), `predict_trajectory_batch` `_dim_q_pack` (~947-950)

- [ ] **Step 1: Implement**

(a) `MasterTNEP._train_model_inner`, replace the `built_dim_q` check with:
```python
    # Cross-check built descriptor width vs cfg (catches cfg/builder drift).
    from DescriptorBuilderGPU import descriptor_input_width
    built_width = int(train_data["descriptors"][0].shape[-1])
    if built_width != descriptor_input_width(cfg):
        raise RuntimeError(
            f"descriptor_input_width={descriptor_input_width(cfg)} disagrees "
            f"with built descriptor width {built_width}; cfg / builder mismatch.")
    print("Dimension of q: " + str(cfg.dim_q))
    if cfg.radial_basis_size is not None:
        print(f"Learned radial basis: {cfg.alpha_max} functions per (centre "
              f"type, neighbour species) as combinations of K="
              f"{cfg.radial_basis_size} orthonormal poly3 functions "
              f"(M_radial [{cfg.num_types}, {cfg.num_types}, {cfg.alpha_max}, "
              f"{cfg.radial_basis_size}] in the SNES vector; init = fixed basis)")
```
(b) (done in Task 0.)
(c) `spectroscopy._get_fused_predict`: replace the `dim_q = (...)` expression with
```python
    from DescriptorBuilderGPU import descriptor_input_width
    dim_q = descriptor_input_width(cfg)   # raw SOAP width, or S·K·k_max·2 (learned basis)
```
In `predict_trajectory_batch` leave both `build_descriptors_flat(...)` calls at their default `calc_gradients=True` (the neighbor readout builds V in-graph from the pair stream; under the knob the builder returns zero-width gradient values with real pair indices). Replace the `_dim_q_pack = (...)` expression with `_dim_q_pack = descriptor_input_width(cfg)`.

- [ ] **Step 2: Verify: full test file + end-to-end training with the knob**

Run: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python -m pytest tests/test_radial_basis.py tests/test_neighbor_dipole.py -q 2>&1 | tail -3`
Expected: all passed.

Run (GPU):
```bash
python - <<'EOF' 2>&1 | grep -v "^WARNING\|^I0000\|cuda\|oneDNN" | tail -30
from MasterTNEP import train_model
from TNEPconfig import TNEPconfig
cfg = TNEPconfig(); cfg.descriptor_mode = 1; cfg.radial_basis_size = 8
cfg.total_N = 60; cfg.num_generations = 5
cfg.save_path = None; cfg.checkpoint_interval = None; cfg.plot_interval = None; cfg.show_plots = False
train_model(cfg)
EOF
```
Expected: `Learned radial basis: ...` line, `radial basis  M_radial` row in the breakdown (288 params), `Run complete!`.

Trajectory-inference smoke (fused GPU path) under the knob: with the model from a `save_path` run (or build one in-process), call `spectroscopy.predict_trajectory_batch`-backed `MasterTNEP.process_trajectory(model, "datasets/test.xyz", batch_size=50, descriptor_mode=1, pin_to_cpu=False, save_plots=None)` on the first frames and check that the returned dipoles are finite and not all zero, and equal (rtol 1e-4) to `model.score`-path predictions on the same frames.

- [ ] **Step 3: Commit**

```bash
git add MasterTNEP.py spectroscopy.py
git commit -m "feat: learned radial basis through the training driver and trajectory inference width sites"
```

---

### Task 7: A/B validation on CHO (record verbatim)

**Files:**
- Create: `scripts/ab_radial_basis.py`
- Append results to: `docs/superpowers/specs/2026-09-17-learned-radial-basis-design.md` (new section "7. Results")

- [ ] **Step 1: Write the script**

```python
"""A/B: fixed SOAP basis vs learned radial basis on the CHO dipole set.
Usage: python scripts/ab_radial_basis.py <K or none> <num_generations> [total_N]
Prints the run's own terminal output; nothing is post-processed."""
import sys
from MasterTNEP import train_model
from TNEPconfig import TNEPconfig

K = None if sys.argv[1].lower() == "none" else int(sys.argv[1])
cfg = TNEPconfig()
cfg.data_path = "datasets/train.xyz"
cfg.test_data_path = "datasets/test.xyz"
cfg.allowed_species = [6, 1, 8]
cfg.descriptor_mode = 1
cfg.radial_basis_size = K
cfg.num_generations = int(sys.argv[2])
cfg.total_N = int(sys.argv[3]) if len(sys.argv) > 3 else None
cfg.seed = 928375439201
cfg.save_path = f"models/ab_radial_K{K}"
cfg.checkpoint_interval = None
cfg.plot_interval = None
cfg.show_plots = False
train_model(cfg)
```

- [ ] **Step 2: Run the three arms** (same seed, same budget; run sequentially on the GPU)

```bash
for K in none 4 8; do
  python scripts/ab_radial_basis.py $K 2000 2>&1 | grep -v "^WARNING\|^I0000\|cuda\|oneDNN" \
    | grep -E "Model Parameters|radial|Final-gen|Best-val|Cosine|evaluate|validate|best val RMSE: [0-9.]+\s+elapsed" \
    > /tmp/claude-1000/-home-reece-TNEP/83293e9c-0843-4e42-a836-d0bcf984a77c/scratchpad/ab_K$K.log
done
```
Then read each log and copy the printed lines (final progress line, Final-gen / Best-val test blocks, timing lines) into the spec's "7. Results" section verbatim. Do not compute or infer anything not printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/ab_radial_basis.py docs/superpowers/specs/2026-09-17-learned-radial-basis-design.md
git commit -m "docs: learned radial basis A/B results on CHO"
```
