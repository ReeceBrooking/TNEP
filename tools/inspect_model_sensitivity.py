"""Inspect a trained TNEP model's sensitivity to each SOAP descriptor channel.

Complements `inspect_descriptors.py`:
  inspect_descriptors        : descriptor-side variance (channel ranges,
                                per-l aggregates, q_scaler estimates).
  inspect_model_sensitivity  : MODEL-side attribution — for each channel d,
                                reports (a) how much the model's per-atom
                                output U_i changes when q_{i,d} is perturbed
                                and (b) how much overall R² drops when q_d
                                is ablated (set to its dataset mean) across
                                the whole evaluation set.

Three complementary signals are reported per channel:

    var_d        = Var(q[:, d])                       descriptor stat
    sens_d       = mean_i  |∂U_i / ∂q_{i,d}|          model stat (analytical)
    importance_d = sens_d² · var_d                    ∝ contribution to Var(U)

    delta_r2_d   = r2(baseline) − r2(ablation_d)      contribution to R²
                   (ablation = replace q[:, d] by its per-species mean
                   for every atom in the eval set, then re-score)

Per-channel ablation runs one full `model.score()` per channel — typically
a few seconds per channel on a small (<200 structure) eval set. Set
RUN_CHANNEL_ABLATION = False to skip it if Q is large; the (pair, l)
group-ablation is much cheaper and still gives an interpretable rollup.

Configure the run by editing the CONFIG block below — there is NO CLI
argument parser, so the file is meant to be opened and executed in the
IDE (right-click → Run File).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from ase.io import read

# Ensure the project root is importable even when the file is launched
# from `tools/` directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import (                                                  # noqa: E402
    assign_type_indices,
    filter_bad_data,
    filter_by_species,
    prepare_eval_data,
)
from DescriptorBuilderGPU import descriptor_block_layout            # noqa: E402
from model_io import load_model                                    # noqa: E402


def _apply_model_filters(structures: list, cfg) -> list:
    """Apply the SAME filters that `data.collect()` applies during training:
      1. species filter (cfg.allowed_species / cfg.filter_mode)
      2. unknown-species drop (anything not in cfg.types)
      3. missing-target / bad-data drop (cfg.filter_bad_data)

    This is what makes the eval dataset comparable to the model's
    training distribution. Without it (a) structures with species the
    model never saw would crash assign_type_indices, and (b) R² /
    sensitivity stats would be contaminated by out-of-distribution
    structures the model has no path to predict correctly.
    """
    n0 = len(structures)
    # 1. allowed-species filter (matches the cfg used at training time).
    if getattr(cfg, "allowed_species", None) is not None:
        # filter_by_species keeps `dataset_types_int` parallel but
        # doesn't read it — pass placeholders.
        placeholder = [np.zeros(len(s.numbers), dtype=np.int32)
                       for s in structures]
        structures, _ = filter_by_species(
            structures, placeholder,
            allowed_Z=cfg.allowed_species,
            mode=cfg.filter_mode)
        print(f"  after allowed_species filter "
              f"({cfg.filter_mode}, {cfg.allowed_species}): "
              f"{len(structures)} / {n0} structures")

    # 2. Hard-drop any structure carrying a species not in cfg.types
    #    (assign_type_indices would crash on them otherwise — e.g.
    #    when allowed_species is None but the eval file is broader
    #    than the model's training composition).
    known_Z = set(int(z) for z in cfg.types)
    before = len(structures)
    structures = [s for s in structures
                  if set(int(z) for z in s.numbers).issubset(known_Z)]
    if len(structures) < before:
        print(f"  dropped {before - len(structures)} structures with "
              f"species outside cfg.types ({sorted(known_Z)}): "
              f"{len(structures)} remaining")

    # 3. missing-target / bad-data filter. filter_bad_data needs the
    #    per-atom type-index lists; build them now that the species set
    #    is guaranteed to be a subset of cfg.types.
    types_int = assign_type_indices(structures, cfg.types)
    before = len(structures)
    structures, _ = filter_bad_data(structures, types_int, cfg)
    if len(structures) < before:
        print(f"  filter_bad_data dropped {before - len(structures)}: "
              f"{len(structures)} remaining")

    if len(structures) == 0:
        raise RuntimeError(
            "Eval dataset is empty after applying the model's filters. "
            "Check that cfg.allowed_species / cfg.types match the eval "
            "data, and that the target key is present in the XYZ.")
    return structures


# ═══════════════════════════════════════════════════════════════════
# CONFIG — edit values here, then run the file.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Path to a trained model file (.h5 or .npz) saved by model_io.save_model.
    model_path: str = "models/n30_q165_pop100_20260523_000816_New_best_r2/train_C_O_H_dipole_final_gen.h5"

    # XYZ file with structures to evaluate sensitivity on. Targets
    # must be present (dipole / pol / energy as appropriate for the
    # model's cfg.target_mode) so R² ablation is meaningful.
    data_path: str = "datasets/test.xyz"

    # Number of structures to use from `data_path` (None = all).
    n_structures: int | None = None

    # How many channels to show in the headline tables.
    top_n: int = 25

    # If True, run the per-channel R² ablation. Cost: ~Q full
    # forward passes; with Q≈100 and a 50-structure eval set,
    # typically 1–3 minutes on CPU. Set False for very large Q
    # (the per-(pair, l) and per-pair group ablations still run).
    run_channel_ablation: bool = True

    # Ablation strategy:
    #   "mean"    : replace q[:, d] with its per-species mean.
    #               Preserves the per-channel offset that downstream
    #               biases were trained against. Least destructive —
    #               approximates "what if this channel carried no
    #               structure-dependent information?".
    #   "zero"    : replace q[:, d] with 0. Harsher; conflates
    #               removing the channel with shifting it. Use only
    #               when the model was trained on already-centered
    #               descriptors.
    ablation_strategy: str = "mean"

    # Optional CSV path for the full per-channel table (None = skip).
    save_csv: str | None = "channel_sensitivity.csv"


CFG = Config()


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════


def _q_to_meta(q: int, layout: dict, L: int) -> tuple:
    """Reverse-lookup: q_idx → (pair_key, l, n_in_pair).

    Within a pair's q-index list, position i has l = i % L and n = i // L
    (mirrors the Fortran emit order in `descriptor_block_layout`).
    """
    for pk in layout["pair_keys"]:
        qidx = layout["pair_q_index"][pk]
        if q in qidx:
            pos = int(np.where(qidx == q)[0][0])
            return pk, pos % L, pos // L
    return (-1, -1), -1, -1


def _ann_sensitivity(model, descr_list: list[np.ndarray],
                     types_list: list[np.ndarray]) -> tuple:
    """Compute analytical |∂U_i / ∂q_{i,d}| per atom, then aggregate.

    For the current per-type ANN with mixing absorbed into W0_eff:

        a_i = q_i @ W0_eff[t_i] + b0[t_i]              # [H]
        h_i = tanh(a_i)                                 # [H]
        U_i = h_i · W1[t_i] + b1                        # scalar

        dU_i / dq_{i,d} = Σ_h (1 − h_i[h]²) · W1[t_i, h] · W0_eff[t_i, d, h]

    Returns (sens_mean_abs[Q], sens_rms[Q], sens_per_species[T, Q],
             species_atom_count[T]).
    """
    cfg = model.cfg
    Q = int(cfg.dim_q)
    T = int(cfg.num_types)

    # Pre-fold U_pair^T into W0 once. _W0_eff is a no-op when descriptor
    # mixing is disabled, so this is safe in all configurations.
    W0_eff = model._W0_eff(model.W0).numpy()                  # [T, Q, H]
    b0     = model.b0.numpy()                                  # [T, H]
    W1     = model.W1.numpy()                                  # [T, H]

    # If the cfg carries a frozen q_scaler, the model was trained on
    # scaled descriptors. We must apply it here so the analytical
    # gradient is computed at the descriptor values the model actually
    # consumes — and so the sensitivity is comparable to the q_scaler-
    # space variance (also computed below from the scaled descriptors).
    q_scaler = getattr(cfg, "_q_scaler", None)

    sens_abs_sum = np.zeros(Q, dtype=np.float64)
    sens_sq_sum  = np.zeros(Q, dtype=np.float64)
    sens_per_sp_abs = np.zeros((T, Q), dtype=np.float64)
    sp_count = np.zeros(T, dtype=np.int64)
    N_atoms_total = 0

    for descr, types_i in zip(descr_list, types_list):
        q_np = np.asarray(descr, dtype=np.float32).reshape(-1, Q)
        z_np = np.asarray(types_i, dtype=np.int32).reshape(-1)
        if q_scaler is not None:
            q_np = q_np * np.asarray(q_scaler, dtype=np.float32)[None, :]
        A = q_np.shape[0]
        N_atoms_total += A

        # Vectorise across atoms using per-atom-type gathered weights.
        W0_a = W0_eff[z_np]                                   # [A, Q, H]
        b0_a = b0[z_np]                                        # [A, H]
        W1_a = W1[z_np]                                        # [A, H]

        a_i = np.einsum('ad,adh->ah', q_np, W0_a) + b0_a       # [A, H]
        h_i = np.tanh(a_i)                                     # [A, H]
        dtanh = 1.0 - h_i * h_i                                 # [A, H]
        # dU/dq[a, d] = Σ_h dtanh[a, h] · W1[a, h] · W0_eff[a, d, h]
        de_dh = dtanh * W1_a                                   # [A, H]
        de_dq = np.einsum('ah,adh->ad', de_dh, W0_a)           # [A, Q]

        abs_de = np.abs(de_dq)
        sens_abs_sum += abs_de.sum(axis=0)
        sens_sq_sum  += (de_dq * de_dq).sum(axis=0)

        for t in range(T):
            sel = z_np == t
            if sel.any():
                sens_per_sp_abs[t] += abs_de[sel].sum(axis=0)
                sp_count[t] += int(sel.sum())

    sens_mean_abs = sens_abs_sum / max(N_atoms_total, 1)
    sens_rms      = np.sqrt(sens_sq_sum / max(N_atoms_total, 1))
    sens_per_sp   = sens_per_sp_abs / np.maximum(sp_count[:, None], 1)
    return sens_mean_abs, sens_rms, sens_per_sp, sp_count


def _baseline_eval(model, eval_data: dict) -> tuple[float, dict]:
    """Run the full eval pipeline once and return (baseline_r2, metrics)."""
    metrics, _ = model.score(eval_data)
    r2 = float(metrics["r2"].numpy() if hasattr(metrics["r2"], "numpy")
               else metrics["r2"])
    return r2, metrics


def _ablate_descriptor_dim(eval_data: dict, dims: list[int] | int,
                            mode: str,
                            per_species_mean: np.ndarray | None) -> dict:
    """Return a copy of `eval_data` with `dims` ablated in the
    `descriptors` tensor. `dims` may be a single int or a list.

    mode="zero" : zero the column(s) outright.
    mode="mean" : replace with per-(species, q) means. The per-species
                  mean is gathered atom-wise via the eval dict's Z_int.
    """
    dims = [int(dims)] if isinstance(dims, int) else [int(d) for d in dims]
    desc = eval_data["descriptors"].numpy().copy()             # [S, A, Q]
    z = eval_data["Z_int"].numpy()                              # [S, A]
    atom_mask = eval_data["atom_mask"].numpy()                  # [S, A]

    if mode == "zero":
        for d in dims:
            desc[:, :, d] = 0.0
    elif mode == "mean":
        assert per_species_mean is not None, (
            "ablation_strategy='mean' requires per_species_mean")
        # per_species_mean[t, d] : mean of channel d for atoms of species t
        # Broadcast atom-wise: z[s, a] indexes into species, then we
        # write into desc[s, a, d].
        for d in dims:
            # Build [S, A] replacement values via gather along species
            repl = per_species_mean[z, d] * atom_mask          # zero padding
            desc[:, :, d] = repl
    else:
        raise ValueError(
            f"ablation_strategy={mode!r}; expected 'zero' or 'mean'")

    out = dict(eval_data)
    out["descriptors"] = tf.constant(desc, dtype=tf.float32)
    return out


def _per_species_mean(eval_data: dict, num_types: int) -> np.ndarray:
    """Compute per-(species, channel) mean of the *padded* descriptors,
    using atom_mask so padding atoms don't bias the mean.

    Returns [num_types, Q] float64.
    """
    desc = eval_data["descriptors"].numpy().astype(np.float64)  # [S, A, Q]
    z = eval_data["Z_int"].numpy()                               # [S, A]
    mask = eval_data["atom_mask"].numpy().astype(np.float64)     # [S, A]
    Q = desc.shape[-1]
    out = np.zeros((num_types, Q), dtype=np.float64)
    for t in range(num_types):
        sel = (z == t) & (mask > 0)
        n = int(sel.sum())
        if n == 0:
            continue
        # Mean over the (S, A) atoms with species t.
        out[t] = desc[sel].sum(axis=0) / n
    return out


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════


def main() -> int:
    cfg_run = CFG
    print(f"Loading model: {cfg_run.model_path}")
    model = load_model(cfg_run.model_path)
    cfg = model.cfg
    print(f"  target_mode={cfg.target_mode}  dim_q={cfg.dim_q}  "
          f"num_types={cfg.num_types}  num_neurons={cfg.num_neurons}  "
          f"types={cfg.types}")

    # ───────────────────── load + prep eval data
    # Read the full file first, apply the model's training-time filters,
    # then truncate to n_structures. Doing it in this order means
    # `n_structures` counts STRUCTURES THE MODEL WILL ACTUALLY SCORE,
    # not file-position rows that may all be filtered out.
    structures = read(cfg_run.data_path, index=":")
    print(f"Loaded {len(structures)} structures from {cfg_run.data_path}")
    structures = _apply_model_filters(structures, cfg)
    if cfg_run.n_structures is not None:
        n = int(cfg_run.n_structures)
        if len(structures) > n:
            structures = structures[:n]
            print(f"  truncated to first {n} post-filter structures")

    t0 = time.perf_counter()
    eval_data = prepare_eval_data(structures, cfg)
    print(f"  built padded eval dict in {time.perf_counter() - t0:.1f}s")

    # Per-channel descriptor stats from the SAME (q_scaler-applied)
    # padded descriptors the model consumes — so var ↔ sens are in
    # the same space.
    desc = eval_data["descriptors"].numpy()
    mask = eval_data["atom_mask"].numpy().astype(bool)
    real = desc[mask]                                          # [N_real, Q]
    Q = int(real.shape[1])
    chan_mean = real.mean(axis=0)
    chan_var  = real.var(axis=0)
    print(f"  {real.shape[0]} real atoms × Q={Q} descriptor channels")

    # ───────────────────── model sensitivity (analytical)
    # Recompute against the raw per-structure descriptor lists so we
    # don't accidentally include padding in the average. Note: the
    # structures still need their per-atom type indices.
    types_int_list = assign_type_indices(structures, cfg.types)
    # Build a raw (unpadded) per-structure descriptor list from the
    # padded tensor by stripping each frame's padding tail.
    descr_unpadded = []
    n_atoms_arr = eval_data["num_atoms"].numpy()
    desc_all    = eval_data["descriptors"].numpy()
    for s, n in enumerate(n_atoms_arr):
        descr_unpadded.append(desc_all[s, :int(n), :])

    print("  computing analytical model sensitivity (∂U/∂q per atom)…")
    sens_mean_abs, sens_rms, sens_per_sp, _ = _ann_sensitivity(
        model, descr_unpadded, types_int_list)

    # Combine into the importance proxy (variance × squared sensitivity).
    chan_importance = (sens_rms ** 2) * chan_var
    total_imp = chan_importance.sum()
    imp_frac = (chan_importance / total_imp
                if total_imp > 0 else np.zeros_like(chan_importance))

    # ───────────────────── R² baseline + (optional) per-channel ablation
    print("\nBaseline scoring (no ablation)…")
    baseline_r2, baseline_metrics = _baseline_eval(model, eval_data)
    baseline_rmse = float(baseline_metrics["rmse"].numpy())
    print(f"  baseline R² = {baseline_r2:.6f}   RMSE = {baseline_rmse:.6f}")

    per_species_mean = (_per_species_mean(eval_data, cfg.num_types)
                        if cfg_run.ablation_strategy == "mean" else None)

    chan_delta_r2 = np.full(Q, np.nan, dtype=np.float64)
    if cfg_run.run_channel_ablation:
        print(f"\nPer-channel R² ablation "
              f"(strategy={cfg_run.ablation_strategy!r}, Q={Q})…")
        t_abl = time.perf_counter()
        for d in range(Q):
            ablated = _ablate_descriptor_dim(
                eval_data, d, cfg_run.ablation_strategy,
                per_species_mean)
            r2_d, _ = _baseline_eval(model, ablated)
            chan_delta_r2[d] = baseline_r2 - r2_d
            if (d + 1) % max(1, Q // 20) == 0:
                pct_done = 100 * (d + 1) / Q
                elapsed = time.perf_counter() - t_abl
                eta = elapsed * (Q - d - 1) / max(d + 1, 1)
                print(f"  channel {d + 1:>4} / {Q}  "
                      f"({pct_done:5.1f}%)  elapsed {elapsed:5.1f}s  "
                      f"ETA {eta:5.1f}s")
        print(f"  done in {time.perf_counter() - t_abl:.1f}s")
    else:
        print("\nPer-channel R² ablation: SKIPPED (run_channel_ablation=False)")

    # ───────────────────── (pair, l) group ablation
    layout = descriptor_block_layout(cfg)
    L = int(cfg.l_max) + 1

    print(f"\nPer-(pair, l) group R² ablation "
          f"(strategy={cfg_run.ablation_strategy!r})…")
    group_delta_r2: dict[tuple, float] = {}
    for pk in layout["pair_keys"]:
        for l in range(L):
            qidx = layout["pair_ln_index"][pk][l]
            if len(qidx) == 0:
                continue
            ablated = _ablate_descriptor_dim(
                eval_data, list(qidx), cfg_run.ablation_strategy,
                per_species_mean)
            r2_g, _ = _baseline_eval(model, ablated)
            group_delta_r2[(pk, l)] = baseline_r2 - r2_g

    # ───────────────────── reports
    print("\n" + "=" * 110)
    print(f"BASELINE   R² = {baseline_r2:.6f}    RMSE = {baseline_rmse:.6f}")
    print("=" * 110)

    print("\n" + "=" * 110)
    print(f"TOP {cfg_run.top_n} CHANNELS BY IMPORTANCE  ((∂U/∂q)²·Var(q))")
    print("=" * 110)
    print(f"{'q':>4} {'pair':>10} {'l':>3} {'n':>3} "
          f"{'Var(q)':>10} {'mean|dU/dq|':>12} {'rms dU/dq':>12} "
          f"{'importance':>12} {'%imp':>7} {'ΔR²':>10}")
    order_imp = np.argsort(-chan_importance)[: cfg_run.top_n]
    for q in order_imp:
        pk, l_val, n_in_pair = _q_to_meta(int(q), layout, L)
        dr2 = (f"{chan_delta_r2[q]:>10.4e}"
               if not np.isnan(chan_delta_r2[q]) else f"{'—':>10}")
        print(f"{q:>4} {str(pk):>10} {l_val:>3} {n_in_pair:>3} "
              f"{chan_var[q]:>10.4f} {sens_mean_abs[q]:>12.4f} "
              f"{sens_rms[q]:>12.4f} "
              f"{chan_importance[q]:>12.4e} {imp_frac[q] * 100:>6.2f}% "
              f"{dr2}")

    if cfg_run.run_channel_ablation:
        print("\n" + "=" * 110)
        print(f"TOP {cfg_run.top_n} CHANNELS BY ΔR² "
              f"(largest drop when channel ablated)")
        print("=" * 110)
        print(f"{'q':>4} {'pair':>10} {'l':>3} {'n':>3} "
              f"{'ΔR²':>10} {'rms dU/dq':>12} {'Var(q)':>10} "
              f"{'importance':>12} {'%imp':>7}")
        order_dr2 = np.argsort(-chan_delta_r2)[: cfg_run.top_n]
        for q in order_dr2:
            pk, l_val, n_in_pair = _q_to_meta(int(q), layout, L)
            print(f"{q:>4} {str(pk):>10} {l_val:>3} {n_in_pair:>3} "
                  f"{chan_delta_r2[q]:>10.4e} {sens_rms[q]:>12.4f} "
                  f"{chan_var[q]:>10.4f} "
                  f"{chan_importance[q]:>12.4e} {imp_frac[q] * 100:>6.2f}%")

    # Per-l rollup
    print("\n" + "=" * 110)
    print("PER-l IMPORTANCE BREAKDOWN")
    print("=" * 110)
    print(f"{'l':>3} {'N_l':>5} {'Σ imp':>14} {'%imp':>7} "
          f"{'mean Var(q)':>12} {'mean rms|dU/dq|':>17} "
          f"{'Σ |Δ R²|':>12}")
    l_index = layout["l_index"]
    for l in range(L):
        qidx = l_index[l]
        imp_sum = chan_importance[qidx].sum()
        var_mean = chan_var[qidx].mean()
        sens_mean = sens_rms[qidx].mean()
        dr2_sum = (np.nansum(np.abs(chan_delta_r2[qidx]))
                   if cfg_run.run_channel_ablation else float("nan"))
        pct = 100 * imp_sum / max(total_imp, 1e-30)
        dr2_str = (f"{dr2_sum:>12.4e}" if cfg_run.run_channel_ablation
                   else f"{'—':>12}")
        print(f"{l:>3} {qidx.size:>5} {imp_sum:>14.4e} {pct:>6.2f}% "
              f"{var_mean:>12.4f} {sens_mean:>17.4f} {dr2_str}")

    # Per-pair rollup (using GROUP ablation R² so the numbers reflect
    # what actually happens when the whole pair block is removed, not
    # just a sum of independent single-channel ablations).
    print("\n" + "=" * 110)
    print("PER-(species-pair) IMPORTANCE BREAKDOWN")
    print("=" * 110)
    type_to_sym = {idx: z for z, idx in cfg.type_map.items()}

    def _pretty_pair(pk):
        a = type_to_sym.get(pk[0], pk[0])
        b = type_to_sym.get(pk[1], pk[1])
        return f"({a},{b})"

    print(f"{'pair':>10} {'bs':>4} {'Σ imp':>14} {'%imp':>7} "
          f"{'mean Var(q)':>12} {'mean rms|dU/dq|':>17} "
          f"{'Σ Δ R² (per-l groups)':>22}")
    for pk in layout["pair_keys"]:
        qidx = layout["pair_q_index"][pk]
        imp_sum = chan_importance[qidx].sum()
        var_mean = chan_var[qidx].mean()
        sens_mean = sens_rms[qidx].mean()
        # Sum the per-(pair, l) ablation drops for this pair
        dr2_pair = sum(group_delta_r2.get((pk, l), 0.0) for l in range(L))
        pct = 100 * imp_sum / max(total_imp, 1e-30)
        print(f"{_pretty_pair(pk):>10} {qidx.size:>4} "
              f"{imp_sum:>14.4e} {pct:>6.2f}% "
              f"{var_mean:>12.4f} {sens_mean:>17.4f} "
              f"{dr2_pair:>22.4e}")

    # Per-(pair, l) rollup — the cell where the model is doing real work
    print("\n" + "=" * 110)
    print("PER-(species-pair, l) GROUP ABLATION  (each cell shows ΔR² "
          "when that whole (pair, l) block is replaced by per-species mean)")
    print("=" * 110)
    header = "pair".rjust(10) + "".join(
        f"  l={l:<2}    " for l in range(L))
    print(header)
    for pk in layout["pair_keys"]:
        row = f"{_pretty_pair(pk):>10}"
        for l in range(L):
            v = group_delta_r2.get((pk, l), None)
            row += "  " + (f"{v:>+8.2e}" if v is not None else "    —    ")
        print(row)

    # Dominant-species view: for each channel, which species is most
    # sensitive to it on average?
    print("\n" + "=" * 110)
    print(f"TOP {cfg_run.top_n} CHANNELS BY MAX PER-SPECIES SENSITIVITY")
    print("=" * 110)
    max_sens_per_sp = sens_per_sp.max(axis=0)                    # [Q]
    argmax_species  = sens_per_sp.argmax(axis=0)
    order_sp = np.argsort(-max_sens_per_sp)[: cfg_run.top_n]
    print(f"{'q':>4} {'pair':>10} {'l':>3} {'n':>3} "
          f"{'dominant species':>18} {'mean |dU/dq| (that sp)':>24}")
    for q in order_sp:
        pk, l_val, n_in_pair = _q_to_meta(int(q), layout, L)
        t = int(argmax_species[q])
        sym = f"Z={type_to_sym.get(t, t)}"
        print(f"{q:>4} {str(pk):>10} {l_val:>3} {n_in_pair:>3} "
              f"{sym:>18} {max_sens_per_sp[q]:>24.4f}")

    # ───────────────────── CSV dump
    if cfg_run.save_csv:
        with open(cfg_run.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["q_idx", "pair_a", "pair_b", "l", "n_in_pair",
                      "var_q", "mean_q",
                      "mean_abs_dU_dq", "rms_dU_dq",
                      "importance", "importance_frac",
                      "delta_r2"]
            for t in range(cfg.num_types):
                header.append(f"sens_species_Z={type_to_sym.get(t, t)}")
            w.writerow(header)
            for q in range(Q):
                pk, l_val, n_in_pair = _q_to_meta(int(q), layout, L)
                row = [q, pk[0], pk[1], l_val, n_in_pair,
                       float(chan_var[q]), float(chan_mean[q]),
                       float(sens_mean_abs[q]), float(sens_rms[q]),
                       float(chan_importance[q]), float(imp_frac[q]),
                       (float(chan_delta_r2[q])
                        if not np.isnan(chan_delta_r2[q]) else "")]
                row.extend(sens_per_sp[:, q].astype(float).tolist())
                w.writerow(row)
        print(f"\nPer-channel CSV → {cfg_run.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
