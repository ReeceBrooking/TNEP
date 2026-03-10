from __future__ import annotations

import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from data import (collect, split, pad_and_stack, filter_by_species,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data_raw,
                  component_labels, print_score_summary,
                  estimate_padded_bytes, get_available_memory,
                  compute_max_padded_size, chunk_raw_data)
from plotting import plot_snes_history, plot_log_val_fitness, plot_sigma_history, plot_timing, plot_correlation
from model_io import save_model, load_model, convert_z_to_type_indices
from spectroscopy import (predict_dipole_trajectory, predict_polarizability_trajectory,
                           compute_ir_spectrum, plot_ir_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from ase.io import read


def _concat_raw(a: dict, b: dict) -> dict:
    """Concatenate two raw (un-padded) data dicts along the structure axis."""
    return {key: a[key] + b[key] for key in a}


def merge_histories(histories: list[dict]) -> dict:
    """Concatenate history dicts from multiple chunked fit() calls.

    Offsets generation numbers and sigma_resets indices so they form a
    continuous sequence across chunks.
    """
    merged = {
        "generation": [],
        "train_loss": [],
        "val_loss": [],
        "L1": [],
        "L2": [],
        "best_rmse": [],
        "worst_rmse": [],
        "sigma_min": [],
        "sigma_max": [],
        "sigma_mean": [],
        "sigma_median": [],
        "sigma_resets": [],
        "vram_mb": [],
        "timing": {
            "sample_batch": [],
            "evaluate": [],
            "rank_update": [],
            "validate": [],
            "overhead": [],
        },
    }

    gen_offset = 0
    for h in histories:
        n = len(h["generation"])
        merged["generation"].extend([g + gen_offset for g in h["generation"]])
        for key in ["train_loss", "val_loss", "L1", "L2", "best_rmse", "worst_rmse",
                     "sigma_min", "sigma_max", "sigma_mean", "sigma_median", "vram_mb"]:
            merged[key].extend(h[key])
        merged["sigma_resets"].extend([r + gen_offset for r in h["sigma_resets"]])
        for phase in merged["timing"]:
            merged["timing"][phase].extend(h["timing"][phase])
        gen_offset += n

    return merged


def score_chunked(model: TNEP, raw_chunks: list[dict], cfg: TNEPconfig) -> tuple[dict, tf.Tensor]:
    """Score model across raw data chunks, concatenating results.

    For each chunk: pad_and_stack → predict_batch → collect predictions → free memory.
    Metrics (RMSE, R², cosine sim) computed on the full concatenated set.

    Args:
        model      : trained TNEP model
        raw_chunks : list of raw data dicts from chunk_raw_data()
        cfg        : TNEPconfig

    Returns:
        metrics : dict with rmse, r2, r2_components, cos_sim_mean, cos_sim_all
        preds   : [S_total, T] tensor of all predictions
    """
    all_preds = []
    all_targets = []

    for i, chunk in enumerate(raw_chunks):
        print(f"  Scoring chunk {i + 1}/{len(raw_chunks)} "
              f"({len(chunk['descriptors'])} structures)...")
        padded = pad_and_stack(chunk)
        preds = model.predict_batch(
            padded["descriptors"], padded["gradients"],
            padded["grad_index"], padded["positions"],
            padded["Z_int"], padded["boxes"],
            padded["atom_mask"], padded["neighbor_mask"],
            model.W0, model.b0, model.W1, model.b1,
            getattr(model, 'W0_pol', None),
            getattr(model, 'b0_pol', None),
            getattr(model, 'W1_pol', None),
            getattr(model, 'b1_pol', None),
        )
        all_preds.append(preds)
        all_targets.append(padded["targets"])
        del padded  # free memory

    preds = tf.concat(all_preds, axis=0)
    targets = tf.concat(all_targets, axis=0)

    diff = preds - targets
    rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))

    ss_res = tf.reduce_sum(tf.square(diff))
    ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
    r2 = 1.0 - ss_res / ss_tot

    ss_res_comp = tf.reduce_sum(tf.square(diff), axis=0)
    ss_tot_comp = tf.reduce_sum(
        tf.square(targets - tf.reduce_mean(targets, axis=0)), axis=0)
    r2_components = 1.0 - ss_res_comp / tf.maximum(ss_tot_comp, 1e-12)

    metrics = {"rmse": rmse, "r2": r2, "r2_components": r2_components}

    if cfg.target_mode >= 1:
        dot = tf.reduce_sum(preds * targets, axis=1)
        norm_p = tf.linalg.norm(preds, axis=1)
        norm_t = tf.linalg.norm(targets, axis=1)
        cos_sim = dot / tf.maximum(norm_p * norm_t, 1e-12)
        metrics["cos_sim_mean"] = tf.reduce_mean(cos_sim)
        metrics["cos_sim_all"] = cos_sim

    return metrics, preds


def train_model(cfg: TNEPconfig | None = None) -> tuple[TNEP, TNEPconfig]:
    """Run full TNEP training pipeline: load, split, train, test, plot, save.

    Args:
        cfg : TNEPconfig or None (uses defaults)

    Returns:
        model  : trained TNEP model
        cfg    : TNEPconfig (potentially modified with dim_q, types, etc.)
    """
    if cfg is None:
        cfg = TNEPconfig()

    # Load raw dataset and assign initial type indices
    dataset, dataset_types_int = collect(cfg)
    print("Number of species in raw dataset: " + str(cfg.num_types))
    print("Number of structures in raw dataset: " + str(len(dataset)))

    # Filter by species if configured
    if cfg.allowed_species is not None:
        dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=cfg.allowed_species)
        print("After species filter: " + str(len(dataset)) + " structures")

    # Recompute type list and indices (needed after filtering, and to normalise
    # the inconsistent indexing from collect())
    cfg.types = []
    for struct in dataset:
        for z in struct.numbers:
            if z not in cfg.types:
                cfg.types.append(z)
    cfg.num_types = len(cfg.types)
    dataset_types_int = assign_type_indices(dataset, cfg.types)
    print("Species: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

    if cfg.target_mode == 1:
        print_dipole_statistics(dataset)
    elif cfg.target_mode == 2:
        print_polarizability_statistics(dataset)

    cfg.randomise(dataset)

    # Split into train/test/val and build SOAP descriptors (slow)
    train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

    # dim_q is determined by the SOAP descriptor size
    cfg.dim_q = train_data["descriptors"][0].shape[-1]
    print("Dimension of q: " + str(cfg.dim_q))

    # Memory budget: compute max structures that fit in padded tensors
    combined_raw = _concat_raw(_concat_raw(train_data, val_data), test_data)
    max_structures = compute_max_padded_size(combined_raw, cfg)
    S_train = len(train_data["descriptors"])
    S_val = len(val_data["descriptors"])
    S_test = len(test_data["descriptors"])
    S_total_fit = S_train + S_val

    per_struct = estimate_padded_bytes(combined_raw, 1)
    ram_bytes, vram_bytes = get_available_memory(cfg)
    print(f"\n=== Memory Budget ===")
    print(f"  Per-structure estimate: {per_struct / 1024 / 1024:.1f} MB")
    print(f"  Available RAM: {ram_bytes / 1024**3:.1f} GB  "
          f"(budget: {ram_bytes * cfg.ram_threshold / 1024**3:.1f} GB)")
    if vram_bytes != float('inf'):
        print(f"  GPU VRAM: {vram_bytes / 1024**3:.1f} GB  "
              f"(budget: {vram_bytes * cfg.vram_threshold / 1024**3:.1f} GB)")
    print(f"  Max padded structures: {max_structures}")
    print(f"  Train+Val: {S_total_fit}  Test: {S_test}")

    needs_chunking = max_structures < S_total_fit

    model = TNEP(cfg)
    print("Model Parameters: " + str(model.optimizer.dim))
    print("Population Size: " + str(model.optimizer.pop_size))
    print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
    print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))

    if not needs_chunking:
        # --- NON-CHUNKED PATH (existing behavior) ---
        print("\nAll data fits in memory — using single-pass training.")
        train_padded = pad_and_stack(train_data)
        test_padded = pad_and_stack(test_data)
        val_padded = pad_and_stack(val_data)

        def periodic_plot_callback(history, gen):
            """Called during training at plot_interval to show progress."""
            print(f"\n--- Periodic plots at generation {gen} ---")
            m, preds = model.score(test_padded)
            print(f"  Test RMSE: {float(m['rmse']):.4f}  R²: {float(m['r2']):.4f}")
            plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
            plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
            plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
            plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
            plot_correlation(test_padded["targets"].numpy(), preds.numpy(), m, cfg,
                             cfg.save_plots, cfg.show_plots)

        history = model.fit(train_padded, val_padded,
                            plot_callback=periodic_plot_callback if cfg.plot_interval else None)

        metrics, test_preds = model.score(test_padded)

    else:
        # --- CHUNKED PATH ---
        train_frac = 1.0 - 2.0 * cfg.test_ratio
        val_frac = cfg.test_ratio
        fit_frac = train_frac + val_frac

        train_chunk_size = max(int(max_structures * train_frac / fit_frac), 1)
        val_chunk_size = max(max_structures - train_chunk_size, 1)

        train_chunks = chunk_raw_data(train_data, train_chunk_size)
        val_chunks = chunk_raw_data(val_data, val_chunk_size)
        n_chunks = len(train_chunks)
        chunk_gens = cfg.num_generations // n_chunks
        remainder = cfg.num_generations % n_chunks

        print(f"\nChunked training: {n_chunks} chunks "
              f"(train: {train_chunk_size}, val: {val_chunk_size} structures/chunk)")
        print(f"  Generations per chunk: {chunk_gens}"
              + (f" (+{remainder} for first chunk)" if remainder else ""))

        if train_chunk_size < 10:
            print(f"  WARNING: Very small train chunks ({train_chunk_size}) — "
                  f"training may be noisy. Consider reducing dataset or increasing memory.")

        all_histories = []
        for i in range(n_chunks):
            gens_this_chunk = chunk_gens + (remainder if i == 0 else 0)
            if gens_this_chunk == 0:
                continue
            auto_patience = min(cfg.patience, gens_this_chunk // 2) if cfg.patience is not None else None

            print(f"\n--- Chunk {i + 1}/{n_chunks} "
                  f"({len(train_chunks[i]['descriptors'])} train, "
                  f"{len(val_chunks[i % len(val_chunks)]['descriptors'])} val, "
                  f"{gens_this_chunk} generations, patience={auto_patience}) ---")

            if cfg.chunk_sigma_reset and i > 0:
                model.optimizer.sigma.assign(
                    tf.fill([model.optimizer.dim], cfg.init_sigma))

            train_padded = pad_and_stack(train_chunks[i])
            val_padded = pad_and_stack(val_chunks[i % len(val_chunks)])

            h = model.fit(train_padded, val_padded,
                          plot_callback=None,
                          num_generations=gens_this_chunk,
                          patience=auto_patience)
            all_histories.append(h)
            del train_padded, val_padded  # free memory

        history = merge_histories(all_histories)

        # Score test data
        print("\nScoring test set...")
        if max_structures >= S_test:
            test_padded = pad_and_stack(test_data)
            metrics, test_preds = model.score(test_padded)
        else:
            test_chunks = chunk_raw_data(test_data, max_structures)
            metrics, test_preds = score_chunked(model, test_chunks, cfg)

    print_score_summary(metrics, cfg, prefix="Model test set")

    # Save model
    if cfg.save_path is not None:
        save_model(model, cfg, cfg.save_path)

    # Timing summary
    timing = history.get("timing", {})
    if timing:
        phases = ["sample_batch", "evaluate", "rank_update", "validate", "overhead"]
        grand = sum(sum(timing[p]) for p in phases)
        n_gens_actual = len(timing["evaluate"])
        print(f"\n=== Timing Breakdown ({grand:.2f}s over {n_gens_actual} generations) ===")
        for p in phases:
            t = sum(timing[p])
            avg = t / max(n_gens_actual, 1)
            pct = 100 * t / max(grand, 1e-9)
            print(f"  {p:15s}: {t:.3f}s total ({pct:5.1f}%) | {avg*1000:.1f}ms/gen")

    # VRAM summary
    vram = history.get("vram_mb", [])
    if vram:
        print(f"\n=== VRAM Usage ===")
        print(f"  Peak: {max(vram):.1f} MB / 12288 MB ({100*max(vram)/12288:.1f}%)")
        print(f"  Last: {vram[-1]:.1f} MB")

    # Plots — use padded test data if available, otherwise re-pad for correlation plot
    if not needs_chunking:
        test_targets = test_padded["targets"].numpy()
    else:
        # Collect targets from raw test data for plotting
        test_targets = tf.concat([
            pad_and_stack(c)["targets"] for c in chunk_raw_data(test_data, max_structures)
        ], axis=0).numpy()

    plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_correlation(test_targets, test_preds.numpy(), metrics, cfg,
                     cfg.save_plots, cfg.show_plots)

    print("Run complete!")
    return model, cfg


def test_model(
    model: TNEP,
    cfg: TNEPconfig,
    data_path: str,
    save_plots: str | None = None,
    show_plots: bool = True,
) -> tuple[dict, tf.Tensor]:
    """Test a trained model on an external dataset.

    Loads structures from data_path, builds descriptors, and scores.

    Args:
        model      : trained TNEP model
        cfg        : TNEPconfig from training (carries types, descriptor params)
        data_path  : str — path to .xyz file with test structures
        save_plots : str or None — directory to save plot into (None = don't save)
        show_plots : bool — True to display plot interactively (default True)

    Returns:
        metrics    : dict with rmse, r2, r2_components, etc.
        predictions : [S, T] tensor of predictions
    """
    dataset = read(data_path, index=":")
    print(f"Loaded {len(dataset)} structures from {data_path}")

    raw_data = prepare_eval_data_raw(dataset, cfg)
    S = len(raw_data["descriptors"])
    max_structures = compute_max_padded_size(raw_data, cfg)

    if max_structures >= S:
        data = pad_and_stack(raw_data)
        metrics, predictions = model.score(data)
        targets_np = data["targets"].numpy()
    else:
        print(f"  Chunked scoring: {S} structures in chunks of {max_structures}")
        chunks = chunk_raw_data(raw_data, max_structures)
        metrics, predictions = score_chunked(model, chunks, cfg)
        targets_np = tf.concat([
            pad_and_stack(c)["targets"] for c in chunks
        ], axis=0).numpy()

    print_score_summary(metrics, cfg, prefix="External test")

    # Plot correlation
    plot_correlation(targets_np, predictions.numpy(), metrics, cfg,
                     save_plots, show_plots)

    return metrics, predictions


def process_trajectory(
    model: TNEP,
    cfg: TNEPconfig,
    trajectory_path: str,
    dt_fs: float = 1.0,
    save_plots: str | None = None,
    show_plots: bool = True,
) -> dict:
    """Predict properties along an MD trajectory and compute spectra.

    For dipole models (mode 1): predicts dipole trajectory and computes IR spectrum.
    For polarizability models (mode 2): predicts polarizability trajectory and
    computes Raman spectrum.

    Args:
        model           : trained TNEP model
        cfg             : TNEPconfig from training
        trajectory_path : str — path to .xyz trajectory file
        dt_fs           : float — MD timestep in femtoseconds
        save_plots      : str or None — directory to save plot into (None = don't save)
        show_plots      : bool — True to display plot interactively (default True)

    Returns:
        For mode 1 (dipole):
            dict with keys: dipoles, freq_cm, intensity, acf
        For mode 2 (polarizability):
            dict with keys: polarizabilities, freq_cm, I_VV, I_VH, I_total,
                            acf_iso, acf_aniso
    """
    trajectory = read(trajectory_path, index=":")
    print(f"Loaded {len(trajectory)} frames from {trajectory_path}")

    dataset_types_int = assign_type_indices(trajectory, cfg.types)

    if cfg.target_mode == 1:
        dipoles = predict_dipole_trajectory(model, trajectory, dataset_types_int, cfg)
        freq_cm, intensity, acf = compute_ir_spectrum(dipoles, dt_fs=dt_fs)
        plot_ir_spectrum(freq_cm, intensity, cfg, save_plots, show_plots)
        return {"dipoles": dipoles, "freq_cm": freq_cm, "intensity": intensity, "acf": acf}

    elif cfg.target_mode == 2:
        pols = predict_polarizability_trajectory(model, trajectory, dataset_types_int, cfg)
        freq_cm, I_VV, I_VH, I_total, acf_iso, acf_aniso = compute_raman_spectrum(
            pols, dt_fs=dt_fs)
        plot_raman_spectrum(freq_cm, I_VV, I_VH, I_total, cfg, save_plots, show_plots)
        return {"polarizabilities": pols, "freq_cm": freq_cm,
                "I_VV": I_VV, "I_VH": I_VH, "I_total": I_total,
                "acf_iso": acf_iso, "acf_aniso": acf_aniso}

    else:
        raise ValueError(f"Spectroscopy not supported for target_mode={cfg.target_mode} (PES). "
                         f"Use mode 1 (dipole) or mode 2 (polarizability).")


if __name__ == '__main__':
    model, cfg = train_model()
