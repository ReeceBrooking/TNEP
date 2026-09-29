# Bilinear Pair Head for the Neighbor Dipole Readout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional antisymmetric bilinear pair term to the neighbor-basis dipole readout so the per-bond charge flow depends on the environments at *both* ends of the bond, while keeping the existing per-channel site scalar, the learned species-pair table `c_pair`, and every K=0 result bit-identical.

**Architecture:** The ANN head widens from C outputs per atom to C·(1+2K): the site scalar s [C] (today's term), plus two rank-K factor vectors a [C, K] and b [C, K]. The dipole gains a second term summed over ordered neighbor pairs, μ_bl = Σ_i Σ_j Σ_n c[n, t_i t_j] G_ij^n · (a_i^n · b_j^n), with G_ij^n = R_n(r_ij)(r_i − r_j) precomputed per pair in data.py as a padded neighbor table. Ordered double summation antisymmetrises automatically (swap i↔j flips the sign of G and the roles of a, b). SNES only sees a wider W1; the extra geometry rides two new dense per-structure keys through the existing `SMALL_KEYS` plumbing, the same way `V_atom` does. The single-weight full-pair path (spectroscopy, score) builds the pair term in-graph from the COO stream, mirroring `_neighbor_V_coo`.

**Tech Stack:** TensorFlow 2 eager/tf.function, NumPy, ase `neighbor_list`, pytest (`tests/test_neighbor_dipole.py`, CPU-only via `CUDA_VISIBLE_DEVICES=''`).

---

## 0. Design summary (read before any task)

### Formulation

Today (`dipole_readout="neighbor"`, cfg `dipole_channels=C`, `dipole_species_pairs="symmetric"`):

    μ = Σ_i Σ_n s_i^n Σ_t' c[n, pair(t_i,t')] V_i[:, n, t'],   V_i[:, n, t'] = Σ_{j∈t', r_ij<r_c} R_n(r_ij) (r_i − r_j)

New, with `dipole_pair_rank = K ≥ 1`:

    μ = (term above, unchanged)
      + Σ_i Σ_{j≠i, r_ij<r_c} Σ_n c[n, pair(t_i,t_j)] · G_ij^n · Σ_k a_i^{n,k} b_j^{n,k}

    G_ij^n = R_n(r_ij) (r_i − r_j)      (3-vector; same R_n and f_c as V)

Properties that the tests pin:
- The pair sum is over ORDERED pairs. Since G_ji = −G_ij, the double sum equals Σ_{i<j} G_ij^n (a_i·b_j − a_j·b_i): antisymmetric by construction, no explicit antisymmetrisation.
- The site-scalar term is the special case b ≡ 1, a = s. We keep it as a separate slice so K=0 is bit-identical to today and gen 0 of a K>0 run can start exactly at the difference model (b-slice of W1 zeroed in μ-init).
- **Species-pair learned terms: yes.** The same `c_pair[n, pair]` multiplies both terms per channel, so the bilinear flow inherits the learned per-species-pair radial profile. The radial channels R_n stay shared (no new radial parameters).
- Any per-frame constant shift of s is still exact gauge for the s-term (Σ_i V_i ≡ 0). The bilinear term has NO such gauge, which is intended — that is what breaks the difference-form locality limit.

### Head layout (single source of truth: `TNEP._head_width` and the slice helper `TNEP._split_head`)

    out = h · W1[t]  ∈ R^{C·(1+2K)}
    s = out[..., 0:C]                       → [.., C]
    a = out[..., C : C + C·K]               → reshape [.., C, K]
    b = out[..., C + C·K : C + 2·C·K]       → reshape [.., C, K]

`W1` shape becomes `[T, H, C·(1+2K)]`. With K=0 that is `[T, H, C]` — today's shape, today's genome.

### Data tensors (only built when K>0)

Built in `data.py` from the same ase `neighbor_list("ijD", atoms, rc)` call that builds `V_atom`:

    nbr_index [S, A_max, M_max] int32    neighbor atom index j (frame-local); padding → 0
    nbr_G     [S, A_max, M_max, 3, C] f32  G_ij^n; padding → 0  (so no separate mask is needed)

M_max = max over the batch of the per-centre neighbor count within r_c. Species of j is looked up in-graph via `tf.gather(Z, nbr_index, batch_dims=1)`; the packed pair slot via `model._pair_idx[t_i, t_j]`.

Size check (bulk water, 500 frames, 192 atoms, ~12 neighbors within 3.0 Å, C=20): 500·192·12·3·20·4 B ≈ 276 MB — ~6× `V_atom`, fine in VRAM. The candidate forward materialises `[C_cand, B, A, M, C]` tensors; `population_chunk_size` is the memory knob (ponytail comment in the code names this ceiling).

### Two implementations of the pair term, cross-checked by the four-path test

1. `TNEP._bilinear_padded(a, b, c_full, Z, nbr_index, nbr_G)` — leading candidate axis, used by `predict_batch_candidates` (SNES fit) and by `predict_batch` when the padded tensors are passed (SNES validate, score with chunks). `predict_batch` calls it with `a[tf.newaxis]` and takes `[0]`.
2. `TNEP._bilinear_coo(a, b, c_full, Z, pair_struct, pair_atom, pair_gidx, positions, boxes, box_inv, B)` — for full-pair callers that pass `nbr_index=None` (legacy pack; the four-path test with `self_pairs_only=False`). Mirrors `_neighbor_V_coo`. NOTE: the fused spectroscopy predict passes `self_only=True` for the neighbor readout (`spectroscopy.py:908/932`), so its COO stream has no neighbor pairs and it does NOT exercise this helper (pre-existing: `_neighbor_V_coo` gets the same stream). The four-path test is the only automated coverage of the COO leg.
3. `TNEP.predict` (single padded structure) computes the term inline from the `[A, M]` table it already has — it already has `Rw`, `keep`, `dr`, `grad_index`.

### Out of scope (explicitly)

- GAP mode (`gap_dipole.py`): `_validate` raises on `dipole_pair_rank > 0`. The convex fit relies on μ linear in the site scalars; the bilinear term is quadratic in ANN outputs.
- Analytic ∂μ/∂r for the pair term (nothing in production consumes it).
- The depth knob (`num_hidden_layers`) — separate plan; orthogonal (hidden block vs head).

### Files

| File | Change |
|---|---|
| `TNEPconfig.py` | `dipole_pair_rank: int = 0` + comment |
| `TNEP.py` | `_head_width`, `_split_head`, W1 shape; bilinear term in `predict`, `predict_batch` (+ `nbr_index`, `nbr_G` kwargs), `predict_batch_candidates` (+ same kwargs); helpers `_bilinear_padded`, `_bilinear_coo` |
| `data.py` | `_neighbor_dipole_pairs` (shared pair helper), `_neighbor_dipole_nbr` (padded table), assembly, `pad_and_stack`, four `SMALL_KEYS` tuples |
| `SNES.py` | `_Ch` (channels, for `n_c`) vs `_Hw` (head width, for W1); μ-init zero b-slice; thread `nbr_index`/`nbr_G` in `struct_keys`, `validate`, fit |
| `MasterTNEP.py` | W1 breakdown label, readout info line |
| `gap_dipole.py` | `_validate` raise |
| `tests/test_neighbor_dipole.py` | 7 new tests, 1 parametrisation extended |

---

## Task 1: Config knob + head width (K=0 stays bit-identical)

**Files:**
- Modify: `TNEPconfig.py` (after `dipole_species_pairs`, ~line 125)
- Modify: `TNEP.py:153-160` (W1 shape), `TNEP.py:112-120` (readout flags)
- Test: `tests/test_neighbor_dipole.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pair_rank_head_width():
    # K=0: head width == C (today's W1 shape, bit-compatible genome).
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=0)
    m = _model(cfg)
    assert m._head_width == 4 and tuple(m.W1.shape) == (2, 8, 4)
    # K=2: C·(1+2K) = 4·5 = 20.
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=2)
    m = _model(cfg)
    assert m._head_width == 20 and tuple(m.W1.shape) == (2, 8, 20)
    # _split_head slices: s [.., C], a/b [.., C, K].
    out = tf.reshape(tf.range(20, dtype=tf.float32), [1, 20])
    s, a, b = m._split_head(out)
    assert tuple(s.shape) == (1, 4)
    assert tuple(a.shape) == (1, 4, 2) and tuple(b.shape) == (1, 4, 2)
    np.testing.assert_array_equal(s.numpy()[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(a.numpy()[0, 0], [4, 5])
    np.testing.assert_array_equal(b.numpy()[0, 0], [12, 13])


def test_pair_rank_rejected_outside_neighbor_readout():
    cfg = _cfg(dipole_readout="tnep", dipole_pair_rank=1)
    with pytest.raises(ValueError, match="dipole_pair_rank"):
        _model(cfg)
```

Also add `cfg.dipole_pair_rank = 0` to the pinned fields in `_cfg` (the suite must not inherit working-tree drift).

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -k pair_rank -q`
Expected: FAIL — `AttributeError: 'TNEP' object has no attribute '_head_width'`

- [ ] **Step 3: Implement**

`TNEPconfig.py`, after `dipole_species_pairs`:

```python
    # Bilinear pair head of the neighbor readout (neighbor_dipole_bilinear_plan.md):
    # K ≥ 1 adds Σ_ij Σ_n c[n,pair] R_n(r_ij)(r_i−r_j)·(a_i^n·b_j^n) with
    # a, b ∈ R^{C×K} extra ANN outputs (head width C·(1+2K)); the flow on a
    # bond then depends on BOTH end environments. 0 = off (bit-identical).
    dipole_pair_rank: int = 0
```

`TNEP.py` — in `__init__`, next to `self._dipole_channels = ...`:

```python
        self._pair_rank = int(cfg.dipole_pair_rank)
        if self._pair_rank < 0:
            raise ValueError(f"cfg.dipole_pair_rank must be ≥ 0, got {self._pair_rank}.")
        if self._pair_rank > 0 and not self._neighbor_readout:
            raise ValueError(
                "cfg.dipole_pair_rank > 0 requires target_mode=1 and "
                "dipole_readout='neighbor' (the bilinear pair term is a "
                "neighbor-readout head).")
        # Head width C·(1+2K): s [C] | a [C·K] | b [C·K]. THE single source for
        # W1's last axis; SNES/model_io read it back from here.
        self._head_width = (self._dipole_channels * (1 + 2 * self._pair_rank)
                            if self._neighbor_readout else 1)
```

W1 shape (replace the `self._dipole_channels` branch):

```python
            shape=((cfg.num_types, self._H_final, self._head_width)
                   if self._neighbor_readout
                   else (cfg.num_types, self._H_final)),
```

Slice helper (add near `_c_full`):

```python
    def _split_head(self, out: tf.Tensor):
        """Split the head output [.., C·(1+2K)] → s [.., C], a [.., C, K], b [.., C, K].
        K = 0 ⇒ a, b have a zero-width last axis (never contracted)."""
        C, K = self._dipole_channels, self._pair_rank
        lead = tf.shape(out)[:-1]
        s = out[..., :C]
        a = tf.reshape(out[..., C:C + C * K], tf.concat([lead, [C, K]], 0))
        b = tf.reshape(out[..., C + C * K:], tf.concat([lead, [C, K]], 0))
        return s, a, b
```

- [ ] **Step 4: Run the two tests + the whole file**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -q`
Expected: 57 passed (55 + 2). Nothing else changes because K=0 keeps `_head_width == C`.

- [ ] **Step 5: Commit**

```bash
git add TNEPconfig.py TNEP.py tests/test_neighbor_dipole.py
git commit -m "feat: dipole_pair_rank knob + head width for the bilinear pair term (K=0 bit-identical)"
```

---

## Task 2: Padded pair table in data.py

**Files:**
- Modify: `data.py:398-444` (`_neighbor_dipole_V` → shared pair helper), `data.py:363-373` (assembly), `data.py:720-800, 845-852` (`pad_and_stack`), the four `SMALL_KEYS` tuples (`data.py:966, 1081, 1107, 1169`)
- Test: `tests/test_neighbor_dipole.py`

- [ ] **Step 1: Write the failing test**

```python
def test_nbr_table_matches_ase_pairs():
    """nbr_index / nbr_G reproduce the ase pair list (padding zero, G = R_n·d)."""
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=1)
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    assert "nbr_index" in d and "nbr_G" in d
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=True)
    idx, G = full["nbr_index"][0].numpy(), full["nbr_G"][0].numpy()   # [A,M], [A,M,3,C]
    # Reference: V_i = Σ_j G_ij (V's definition) — ties the table to the V kernel.
    V_ref = full["V_atom"][0].numpy().sum(-1)                           # [A,3,C]
    np.testing.assert_allclose(G.sum(1), V_ref, rtol=1e-5, atol=1e-6)
    # Padding rows carry zero G and index 0; real rows have |G| > 0 somewhere.
    i, j, D = neighbor_list("ijD", atoms, cfg.dipole_rc)
    counts = np.bincount(i[i != j], minlength=len(atoms))
    for a_ in range(len(atoms)):
        assert np.all(G[a_, counts[a_]:] == 0.0)
        assert np.all(idx[a_, counts[a_]:] == 0)
        assert set(idx[a_, :counts[a_]]) == set(j[(i == a_) & (i != j)])


def test_nbr_table_absent_when_rank_zero():
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=0)
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    assert "nbr_index" not in d and "nbr_G" not in d
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -k nbr_table -q`
Expected: first test FAILS on `assert "nbr_index" in d`; second PASSES already (keep it — it guards the K=0 memory footprint).

- [ ] **Step 3: Refactor `_neighbor_dipole_V` into a pair helper + two consumers**

Replace `_neighbor_dipole_V` with:

```python
def _neighbor_dipole_pairs(atoms: Atoms, rc: float, C: int, radial: str,
                           zeta: float, cutoff: str, gauss_sigma_scale: float):
    """Ordered neighbor pairs of the neighbor-basis readout: (i, j, G) with
    G[p] = R_n(|d|)·d [P, 3, C], d = r_i − r_j (image pairs, self-images
    dropped). R_n / f_c per `radial` / `cutoff` (numpy twin of
    TNEP._radial/_fc; pairs are filtered before the power, so no r=0 guard).
    float64 throughout; callers cast."""
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)  # ase D = r_j − r_i
    r = np.linalg.norm(d, axis=1)
    ... (the existing fc / Rw code, unchanged) ...
    return i, j, d[:, :, None] * Rw[:, None, :]              # [P, 3, C]


def _neighbor_dipole_V(atoms, types_int, rc, C, T, radial="chebyshev",
                       zeta=-2.0, cutoff="cosine2", gauss_sigma_scale=1.0):
    """(docstring as today)"""
    i, j, G = _neighbor_dipole_pairs(atoms, rc, C, radial, zeta, cutoff,
                                     gauss_sigma_scale)
    V = np.zeros((len(atoms), T, 3, C))
    np.add.at(V, (i, np.asarray(types_int)[j]), G)
    return V.transpose(0, 2, 3, 1).astype(np.float32)      # [N, 3, C, T]


def _neighbor_dipole_nbr(atoms, rc, C, radial, zeta, cutoff, gauss_sigma_scale):
    """Padded per-centre pair table for the bilinear pair head:
    nbr_index [N, M] int32 (padding 0), nbr_G [N, M, 3, C] float32 (padding 0),
    M = max neighbors of any centre in THIS frame (re-padded across frames in
    pad_and_stack)."""
    i, j, G = _neighbor_dipole_pairs(atoms, rc, C, radial, zeta, cutoff,
                                     gauss_sigma_scale)
    N = len(atoms)
    order = np.argsort(i, kind="stable")
    i, j, G = i[order], j[order], G[order]
    counts = np.bincount(i, minlength=N)
    M = int(counts.max()) if N else 0
    slot = np.arange(len(i)) - np.repeat(np.cumsum(counts) - counts, counts)
    nbr_index = np.zeros((N, M), dtype=np.int32)
    nbr_G = np.zeros((N, M, 3, C), dtype=np.float32)
    nbr_index[i, slot] = j
    nbr_G[i, slot] = G
    return nbr_index, nbr_G
```

Assembly (`assemble_data_dict`, right after the `V_atom` block):

```python
        if int(cfg.dipole_pair_rank) > 0:
            _nbr = [_neighbor_dipole_nbr(
                        s, float(cfg.dipole_rc or cfg.rcut_hard),
                        int(cfg.dipole_channels), str(cfg.dipole_radial),
                        float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg),
                        float(cfg.dipole_gauss_sigma_scale))
                    for s in dataset]
            data["nbr_index"] = [n for n, _ in _nbr]
            data["nbr_G"] = [g for _, g in _nbr]
```

`pad_and_stack` — mirror the `has_V` pattern exactly (allocate `[S, max_atoms, M_max, ...]` with `M_max = max(t.shape[1] for t in data["nbr_index"])`, fill `[s, :N_s, :M_s]`, emit `result["nbr_index"]`, `result["nbr_G"]`). Add `"nbr_index", "nbr_G"` to all four `SMALL_KEYS` / `_RESIDENT_SMALL_KEYS` tuples.

- [ ] **Step 4: Run the test file**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -q`
Expected: 59 passed. In particular every existing V-consuming test still passes — the refactor must leave `_neighbor_dipole_V` numerically identical (it is the same arithmetic, only hoisted).

- [ ] **Step 5: Commit**

```bash
git add data.py tests/test_neighbor_dipole.py
git commit -m "feat: padded neighbor pair table (nbr_index/nbr_G) for the bilinear pair head; V built from the shared pair helper"
```

---

## Task 3: Bilinear term in the forwards (padded helper + COO helper + predict)

**Files:**
- Modify: `TNEP.py:733-756` (`predict` neighbor branch), `TNEP.py:1225-1312` (`predict_batch`), `TNEP.py:1348-1460` (`predict_batch_candidates`), new helpers next to `_neighbor_V_coo` (~line 1715)
- Test: `tests/test_neighbor_dipole.py`

- [ ] **Step 1: Write the failing tests**

```python
def _bilinear_reference(model, atoms, cfg, h):
    """Brute-force μ_bl from the ase pair list and the ANN head outputs h·W1[t]."""
    types_int = np.asarray(assign_type_indices([atoms], cfg.types)[0])
    out = np.einsum('ah,ahk->ak', h, model.W1.numpy()[types_int])
    C, K = cfg.dipole_channels, cfg.dipole_pair_rank
    a = out[:, C:C + C * K].reshape(-1, C, K)
    b = out[:, C + C * K:].reshape(-1, C, K)
    c_full = model._c_full(model.c_pair).numpy()            # [C, T, T]
    i, j, G = tnep_data._neighbor_dipole_pairs(
        atoms, cfg.dipole_rc, C, cfg.dipole_radial, cfg.dipole_zeta,
        resolved_dipole_cutoff(cfg), cfg.dipole_gauss_sigma_scale)   # [P,3,C]
    mu = np.zeros(3)
    for p in range(len(i)):
        ab = (a[i[p]] * b[j[p]]).sum(-1)                     # [C]
        mu += G[p] @ (c_full[:, types_int[i[p]], types_int[j[p]]] * ab)
    return mu


def test_bilinear_matches_bruteforce_and_is_antisymmetric():
    cfg = _cfg(dipole_channels=3, dipole_pair_rank=2, dipole_species_pairs="symmetric")
    model = _model(cfg)
    # Randomise c so the species-pair coupling is actually exercised.
    model.c_pair.assign(tf.random.stateless_uniform(model.c_pair.shape, [1, 2], 0.5, 1.5))
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=True)
    args = (full["descriptors"], full["grad_values"], full["pair_atom"],
            full["pair_gidx"], full["pair_struct"], full["positions"],
            full["Z_int"], full["boxes"], full["atom_mask"],
            model.W0, model.b0, model.W1, model.b1)
    kw = dict(W_atom=full["V_atom"], nbr_index=full["nbr_index"], nbr_G=full["nbr_G"])
    mu_full = model.predict_batch(*args, **kw)[0].numpy()
    # s-term alone: zero the a/b slices of W1.
    W1_s = model.W1.numpy(); W1_s[:, :, cfg.dipole_channels:] = 0.0
    mu_s = model.predict_batch(*args[:-2], tf.constant(W1_s), model.b1, **kw)[0].numpy()
    # Hidden activations for the reference (same arithmetic as predict_batch).
    z = np.einsum('aq,aqh->ah', full["descriptors"][0].numpy(),
                  model._W0_eff(model.W0).numpy()[full["Z_int"][0].numpy()]) \
        + model.b0.numpy()[full["Z_int"][0].numpy()]
    h = model.activation(tf.constant(z)).numpy()
    mu_ref = _bilinear_reference(model, atoms, cfg, h)
    np.testing.assert_allclose(mu_full - mu_s, mu_ref, rtol=1e-4, atol=1e-5)
    # Antisymmetry: swapping the a and b slices negates the pair term.
    W1_sw = model.W1.numpy().copy()
    C, K = cfg.dipole_channels, cfg.dipole_pair_rank
    W1_sw[:, :, C:C + C * K], W1_sw[:, :, C + C * K:] = (
        model.W1.numpy()[:, :, C + C * K:], model.W1.numpy()[:, :, C:C + C * K])
    mu_sw = model.predict_batch(*args[:-2], tf.constant(W1_sw), model.b1, **kw)[0].numpy()
    np.testing.assert_allclose(mu_sw - mu_s, -(mu_full - mu_s), rtol=1e-4, atol=1e-5)
```

And extend the existing four-path test so the K>0 legs are covered:

```python
@pytest.mark.parametrize("mode", ["chebyshev", "screened", "chebyshev_warped",
                                  "gaussian", "cheb_tanh3"])
@pytest.mark.parametrize("rank", [0, 2])
def test_four_path_consistency(mode, rank):
    cfg, model, atoms, _ = _bulk_setup(mode, dipole_pair_rank=rank)
    ...
    mu_slot = model.predict_batch(*args, W_atom=full["V_atom"],
                                  nbr_index=full.get("nbr_index"), nbr_G=full.get("nbr_G"))[0].numpy()
    mu_ingraph = model.predict_batch(*args, W_atom=None)[0].numpy()          # COO path
    mu_cand = model.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        model.W0[tf.newaxis], model.b0[tf.newaxis], model.W1[tf.newaxis], model.b1[tf.newaxis],
        nbr_index=full.get("nbr_index"), nbr_G=full.get("nbr_G"))[0, 0].numpy()
    ... (predict() leg unchanged) ...
```

(`_bulk_setup(mode="chebyshev")` at `tests/test_neighbor_dipole.py:96` builds `_cfg(**_MODE_KW[mode])`; change it to `_bulk_setup(mode="chebyshev", **kw)` → `_cfg(**_MODE_KW[mode], **kw)`. `screened` forces C=1 — rank 2 with C=1 is fine, head width 5. Note `_bulk_setup` uses `prepare_eval_data`, so the padded `nbr_*` keys must also flow through that path — check it calls `assemble_data_dict` + `pad_and_stack`; if it has its own key list, add the two keys there too.)

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -k "bilinear or four_path" -q`
Expected: FAIL — `TypeError: predict_batch() got an unexpected keyword argument 'nbr_index'`.

- [ ] **Step 3: Implement the two helpers**

Next to `_neighbor_V_coo`:

```python
    def _bilinear_padded(self, a, b, c_full, Z, nbr_index, nbr_G):
        """Bilinear pair term from the padded neighbor table.
            μ_bl[c,b] = Σ_{a,m,n} c_full[c,n,t_a,t_j] · nbr_G[b,a,m,:,n] · (a[c,b,a,n,:]·b[c,b,j,n,:])
        a, b     : [Cc, B, A, C, K]   (Cc = candidates; predict_batch passes Cc=1)
        c_full   : [Cc, C, T, T]
        Z        : [B, A] int; nbr_index [B, A, M] int; nbr_G [B, A, M, 3, C]
        Returns [Cc, B, 3]. Padding slots have nbr_G = 0, so no mask.
        # ponytail: materialises [Cc,B,A,M,C] twice (ab product, c gather);
        # population_chunk_size is the memory knob — loop over K if it bites."""
        # One gather over the flattened (B·A) atom axis — no batch_dims bookkeeping.
        Cc = tf.shape(a)[0]; B = tf.shape(Z)[0]; A = tf.shape(Z)[1]; M = tf.shape(nbr_index)[2]
        T = self.num_types; C = self._dipole_channels; K = self._pair_rank
        flat_j = tf.reshape(nbr_index + (tf.range(B) * A)[:, tf.newaxis, tf.newaxis], [-1])   # [B·A·M] into B·A
        b_flat = tf.reshape(b, [Cc, B * A, C, K])
        b_j = tf.reshape(tf.gather(b_flat, flat_j, axis=1), [Cc, B, A, M, C, K])
        ab = tf.einsum('cbank,cbamnk->cbamn', a, b_j)                 # [Cc, B, A, M, C]
        t_j = tf.reshape(tf.gather(tf.reshape(Z, [-1]), flat_j), [B, A, M])
        slot = Z[:, :, tf.newaxis] * T + t_j                          # [B, A, M] into T·T
        c_flat = tf.reshape(tf.transpose(c_full, [0, 2, 3, 1]), [Cc, T * T, C])
        c_ij = tf.gather(c_flat, slot, axis=1)                        # [Cc, B, A, M, C]
        return tf.einsum('cbamn,cbamn,bamdn->cbd', ab, c_ij, tf.cast(nbr_G, ab.dtype))
```

(`_c_full` already accepts a leading candidate axis — `TNEP.py:1710` gathers on `axis=-1`.)

```python
    def _bilinear_coo(self, a, b, c_full, Z, pair_struct, pair_atom, pair_gidx,
                      positions, boxes, box_inv, B):
        """Bilinear pair term from the full r_d COO pair stream (single weight
        set; spectroscopy / legacy callers). a, b : [B, A, C, K]; c_full [C, T, T].
        Returns [B, 3]. Mirrors _neighbor_V_coo's pair filtering."""
        dr, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        r = tf.sqrt(rij2)
        keep = tf.logical_and(r < float(self.cfg.dipole_rc),
                              tf.not_equal(pair_atom, pair_gidx))
        Rw = self._radial(r) * tf.cast(keep, r.dtype)[:, tf.newaxis]   # [P, C]
        G = (-dr)[:, :, tf.newaxis] * Rw[:, tf.newaxis, :]              # [P, 3, C]
        ii = tf.stack([pair_struct, pair_atom], axis=1)
        jj = tf.stack([pair_struct, pair_gidx], axis=1)
        a_i = tf.gather_nd(a, ii); b_j = tf.gather_nd(b, jj)            # [P, C, K]
        ab = tf.reduce_sum(a_i * b_j, axis=-1)                          # [P, C]
        t_i = tf.gather_nd(Z, ii); t_j = tf.gather_nd(Z, jj)
        c_ij = tf.gather(tf.reshape(tf.transpose(c_full, [1, 2, 0]), [-1, tf.shape(c_full)[0]]),
                         t_i * self.num_types + t_j)                    # [P, C]
        per_pair = tf.einsum('pn,pn,pdn->pd', ab, c_ij, tf.cast(G, ab.dtype))   # [P, 3]
        return tf.math.unsorted_segment_sum(per_pair, pair_struct, num_segments=B)
```

- [ ] **Step 4: Wire the three forwards**

`predict_batch`: signature gains `nbr_index: tf.Tensor | None = None, nbr_G: tf.Tensor | None = None`. In the neighbor branch replace `s = tf.einsum(...)` / final return with:

```python
            out = tf.einsum('bah,bahk->bak', h1, W1_t)                # [B, A, C(1+2K)]
            s, a, b = self._split_head(out)
            ...(V as today)...
            c_full = self._c_full(c_pair)                              # [C, T, T]
            c_at = tf.gather(c_full, Z, axis=1)                        # [C, B, A, T]
            mu = tf.einsum('ban,nbat,badnt->bd', s, c_at, V)           # [B, 3]
            if self._pair_rank > 0:
                if nbr_index is not None:
                    mu += self._bilinear_padded(a[tf.newaxis], b[tf.newaxis],
                                                c_full[tf.newaxis], Z,
                                                nbr_index, nbr_G)[0]
                else:
                    mu += self._bilinear_coo(a, b, c_full, Z, pair_struct,
                                             pair_atom, pair_gidx, positions,
                                             boxes, box_inv, tf.shape(descriptors)[0])
            return mu
```

`predict_batch_candidates`: signature gains the same two kwargs; `_w1_shape` uses `self._head_width` instead of `self._dipole_channels`; neighbor branch:

```python
            out = tf.einsum('cbah,cbahk->cbak', h1, W1_t_all)          # [Cc,B,A,C(1+2K)]
            s, a, b = self._split_head(out)
            if c_pair is None:
                # Candidate-independent c: tile once so ONE einsum serves both cases.
                # (Symbolic C — never `list + TensorShape`, it does not trace.)
                cf = self._c_full(self.c_pair)
                c_full = tf.tile(cf[tf.newaxis], [C, 1, 1, 1])        # [Cc, C, T, T]; C here = candidates
            else:
                c_full = self._c_full(c_pair)                          # [Cc, C, T, T]
            c_at = tf.gather(c_full, Z, axis=2)                        # [Cc,C,B,A,T]
            mu = tf.einsum('cban,cnbat,badnt->cbd', s, c_at, W_atom)
            if self._pair_rank > 0:
                mu += self._bilinear_padded(a, b, c_full, Z, nbr_index, nbr_G)
            return mu
```

(`_c_full` must accept a leading candidate axis — check `TNEP.py:1710`; today the candidate path already calls it on `[Cc, C, n_pairs]`, so it does. The `c_pair is None` branch today skips the candidate axis; broadcasting it once is simpler than keeping two einsums — keep ONE einsum.)

`predict` (single padded structure, `TNEP.py:733-756`): after `s`, `Rw`, `tp`, `V`:

```python
            out = tf.einsum('ah,ahk->ak', h, W1_t)
            s, a, b = self._split_head(out)
            ...
            mu = tf.einsum('an,nat,adnt->d', s, c_at, tf.cast(V, s.dtype))
            if self._pair_rank > 0:
                # Same [A, M] table as V: G = Rw·(−dr) per slot, padding masked by w.
                G = tf.cast(Rw[:, :, tf.newaxis, :] * (-dr)[:, :, :, tf.newaxis], s.dtype)  # [A,M,3,C]
                mu += self._bilinear_padded(
                    a[tf.newaxis, tf.newaxis], b[tf.newaxis, tf.newaxis],
                    self._c_full(self.c_pair)[tf.newaxis], Z[tf.newaxis],
                    grad_index[tf.newaxis], G[tf.newaxis])[0, 0]
            return mu
```

- [ ] **Step 5: Run the file**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -q`
Expected: 65 passed (59 + bilinear test + 5 extra four-path params). If the padded/COO legs disagree only at K>0, the bug is in the pair-slot index (`t_i*T + t_j`) or the sign of `G` — the brute-force test isolates which.

- [ ] **Step 6: Commit**

```bash
git add TNEP.py tests/test_neighbor_dipole.py
git commit -m "feat: bilinear pair term in predict / predict_batch / predict_batch_candidates (padded + COO paths, four-path tested)"
```

---

## Task 4: SNES — head width vs channel count, zero-init b-slice, thread the pair table

**Files:**
- Modify: `SNES.py:89-113` (counts), `458-515` (`_build_mu_init`), `583-627` + `1707-1740` (per-type extract), `627-660` (`_build_type_of_variable`), `1427-1480` (`reconstruct_params_tf`), `952-964` (`struct_keys`), `1342-1370` (validate call), `1990-2015` (fit call)
- Modify: `MasterTNEP.py:348-350` (label), `~507` (info line)
- Test: `tests/test_neighbor_dipole.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_pair_rank_snes_tail_and_gen0_identity():
    """W1 block = T·H·C(1+2K); c_pair block unchanged (C·pairs); μ-init zeroes the
    b-slice so gen 0 equals the K=0 model with the same s-slice weights."""
    from SNES import SNES, _set_model_params
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=2, dipole_species_pairs="symmetric")
    model = _model(cfg); opt = SNES(model)
    T, H, C, K = 2, 8, 4, 2
    assert opt._n_W1 == T * H * C * (1 + 2 * K)
    assert opt.n_c == C * (T * (T + 1) // 2)
    named = opt._split_reconstructed(opt.reconstruct_params_tf(opt.mu))
    W1 = named["W1"].numpy()
    assert tuple(W1.shape) == (T, H, C * (1 + 2 * K))
    assert np.all(W1[:, :, C + C * K:] == 0.0)            # b-slice zero at gen 0
    assert np.any(W1[:, :, C:C + C * K] != 0.0)           # a-slice Glorot
    # Round trip: reconstruct → assign → predict_batch == predict_batch_candidates(mu).
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=True)
    mu_vec = opt.mu + 0.05 * tf.random.stateless_normal(opt.mu.shape, [3, 4])  # leave gen 0
    params = opt.reconstruct_params_tf(mu_vec)
    _set_model_params(model, *params)
    mu_model = model.predict_batch(
        full["descriptors"], full["grad_values"], full["pair_atom"], full["pair_gidx"],
        full["pair_struct"], full["positions"], full["Z_int"], full["boxes"],
        full["atom_mask"], model.W0, model.b0, model.W1, model.b1,
        W_atom=full["V_atom"], nbr_index=full["nbr_index"], nbr_G=full["nbr_G"])[0].numpy()
    n = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec[tf.newaxis]))
    mu_cand = model.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        n["W0"], n["b0"], n["W1"], n["b1"], c_pair=n["c_pair"],
        nbr_index=full["nbr_index"], nbr_G=full["nbr_G"])[0, 0].numpy()
    np.testing.assert_allclose(mu_cand, mu_model, rtol=1e-5, atol=1e-6)
    assert np.any(mu_cand != 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -k snes_tail_and_gen0 -q`
Expected: FAIL at `opt._n_W1 == ...` (SNES still sizes W1 by `_Ch`).

- [ ] **Step 3: Implement**

`SNES.__init__` (`SNES.py:89-113`):

```python
        self._Ch = (int(model._dipole_channels) if model._neighbor_readout else 1)   # channels → c_pair
        self._Hw = (int(model._head_width) if model._neighbor_readout else 1)        # W1 last axis
        ...
        n_W1 = self.cfg.num_types * self.H * self._Hw
        ...
        self._n_per_type = ((self.cfg.dim_q * self.H + self.H + self.H * self._Hw)
                            * (2 if self.cfg.target_mode == 2 else 1))
```

Then replace `_Ch` by `_Hw` at every W1-sizing site — `_extract_type_params`, `_extract_type_params_batched`, `_build_type_of_variable` (the `W1: [T, H·Ch]` loop), `reconstruct_params_tf` (`n_W1`, `w1_shape`), `_build_mu_init` (`c_W1`, `n_W1`). `n_c`, `n_pairs = self.n_c // self._Ch` and the c reshape KEEP `_Ch`. Grep after: `grep -n "_Ch" SNES.py` must show only the c_pair lines (init, n_c, n_pairs, c reshape).

`_build_mu_init` — after filling W1 in `_fill_ann`, zero the b-slice:

```python
            n_W1 = T * H * self._Hw
            w1 = rng.uniform(-c_W1, c_W1, size=(T, H, self._Hw)).astype(np.float32)
            if self.model._neighbor_readout and self.model._pair_rank > 0:
                # Gen 0 == the difference model: b ≡ 0 kills the pair term.
                C, K = self._Ch, self.model._pair_rank
                w1[:, :, C + C * K:] = 0.0
            mu[off:off + n_W1] = w1.reshape(-1)
            off += n_W1
```

Threading (three sites, copy the `V_atom` lines):

```python
                for _k in ("V_atom", "nbr_index", "nbr_G"):          # SNES.py:959
                    if _k in train_data:
                        struct_keys.append(_k)
```

validate (`SNES.py:1359`): add `nbr_index=chunk.get("nbr_index"), nbr_G=chunk.get("nbr_G")` to the `predict_batch` call.
fit (`SNES.py:2010`): add `nbr_index=batch_data.get("nbr_index"), nbr_G=batch_data.get("nbr_G")` to the `predict_batch_candidates` call. Also `TNEP.py:944` (score chunks): pass `chunk.get("nbr_index")`, `chunk.get("nbr_G")`.

`MasterTNEP.py:348`: label `"ANN  W1  [T·H·C(1+2K)]"` when `model._pair_rank > 0`; info line at ~507 appends `f", bilinear pair rank K = {model._pair_rank}"` when > 0.

- [ ] **Step 4: Run the file**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py -q`
Expected: 66 passed. `test_ordered_snes_tail_size` (existing) must still pass — it pins the K=0 layout.

- [ ] **Step 5: Commit**

```bash
git add SNES.py TNEP.py MasterTNEP.py tests/test_neighbor_dipole.py
git commit -m "feat: SNES sizes W1 by head width, zero-inits the bilinear b-slice, threads nbr_index/nbr_G through validate and fit"
```

---

## Task 5: Save/load round trip + GAP guard

**Files:**
- Modify: `gap_dipole.py` (`_validate`, ~line 100)
- Test: `tests/test_neighbor_dipole.py`, `tests/test_gap_dipole.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_load_roundtrip_pair_rank(tmp_path):
    # Same pattern as test_save_load_roundtrip_ordered_omega (line ~591).
    import model_io
    cfg = _cfg(dipole_channels=4, dipole_pair_rank=2, dipole_species_pairs="symmetric")
    model = _model(cfg)
    model.W1.assign(tf.random.stateless_normal(model.W1.shape, [5, 6]))
    atoms = _bulk_frame()
    mu0 = _mu(model, atoms, cfg)          # _mu must pass nbr_index/nbr_G — see Task 3 note
    p = str(tmp_path / "rank2.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded._pair_rank == 2 and tuple(loaded.W1.shape) == tuple(model.W1.shape)
    np.testing.assert_array_equal(loaded.W1.numpy(), model.W1.numpy())
    np.testing.assert_array_equal(loaded.c_pair.numpy(), model.c_pair.numpy())
    assert np.allclose(mu0, _mu(loaded, atoms, loaded.cfg), atol=1e-6)
```

(`_mu` at `tests/test_neighbor_dipole.py:~72` calls `predict_batch(..., W_atom=data.get("V_atom"))`; add `nbr_index=data.get("nbr_index"), nbr_G=data.get("nbr_G")` there in Task 3 so every μ-comparing test exercises the padded pair path when K>0.)

In `tests/test_gap_dipole.py` (its cfg helper is `_cfg` at line 24):

```python
def test_gap_rejects_pair_rank():
    cfg = _cfg(); cfg.dipole_pair_rank = 1
    with pytest.raises(ValueError, match="dipole_pair_rank"):
        gap_dipole._validate(cfg)
```

- [ ] **Step 2: Run to verify**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py tests/test_gap_dipole.py -k pair_rank -q`
Expected: round trip may already PASS (cfg is serialised with the model and W1's shape is derived from it) — keep the test as the regression pin; the GAP test FAILS until the guard exists.

- [ ] **Step 3: Implement the GAP guard**

In `gap_dipole._validate`'s checks list add:

```python
        (int(cfg.dipole_pair_rank) == 0,
         "GAP mode needs dipole_pair_rank=0: the bilinear pair term is "
         "quadratic in the ANN outputs, so μ is no longer linear in the site "
         "scalars and the convex sparse-kernel fit does not apply."),
```

- [ ] **Step 4: Run both files**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_neighbor_dipole.py tests/test_gap_dipole.py -q`
Expected: all pass (67 + 18).

- [ ] **Step 5: Commit**

```bash
git add gap_dipole.py tests/test_neighbor_dipole.py tests/test_gap_dipole.py
git commit -m "test: save/load round trip at pair rank 2; GAP mode rejects dipole_pair_rank > 0"
```

---

## Task 6: End-to-end smoke (SNES, real data, GPU) + memory note

**Files:** none modified (scratch script only)

- [ ] **Step 1: Run an 8-generation fit at K=2 with the default config**

```python
# scratch: smoke_bilinear.py
import sys, os; sys.path.insert(0, "/home/reece/TNEP"); os.chdir("/home/reece/TNEP")
from TNEPconfig import TNEPconfig; from MasterTNEP import train_model
cfg = TNEPconfig(); cfg.total_N = 60; cfg.num_generations = 8; cfg.pop_size = 20
cfg.dipole_pair_rank = 2; cfg.save_path = sys.argv[1] + "/models/auto"; cfg.plot_interval = None
m = train_model(cfg); print("SMOKE OK", tuple(m.W1.shape))
```

Run: `python smoke_bilinear.py <scratch dir> 2>&1 | grep -v "^WARNING\|^I0000\|^W0000" | tail -30`
Expected: `SMOKE OK (2, 70, 100)` (C=20, K=2 → 20·5 = 100); the parameter breakdown prints `ANN  W1  [T·H·C(1+2K)]`; peak VRAM printed by the timing plot stays under the card's limit at `population_chunk_size=10`. If it OOMs, halve `population_chunk_size` — do NOT restructure the einsum in this task.

- [ ] **Step 2: Record the smoke in the method doc**

Append a short "Bilinear pair head" subsection to `neighbor_dipole_method.md` §6 (formulation, K semantics, gen-0 identity, memory knob) — no results claims beyond "runs; K=0 bit-identical (tests)".

- [ ] **Step 3: Commit**

```bash
git add neighbor_dipole_method.md
git commit -m "docs: bilinear pair head — formulation, gen-0 identity, memory knob"
```

---

## Validation the plan does NOT do (user decides)

Per the standing preference (prototype features are proven by tests + short smokes; no long SNES A/B runs unless asked): the plan stops at the smoke. The A/B that would answer "does locality buy anything" is: same config, K=0 vs K=2 vs K=4, ≥2k generations, on the test split plus the density-split OOD arm. The gen-0 identity means the K>0 arms start exactly where K=0 starts, so any early divergence is the pair term, not the init.
