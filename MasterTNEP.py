from __future__ import annotations

import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from data import (collect, split, pad_and_stack, filter_by_species,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data,
                  component_labels, print_score_summary)
from plotting import plot_snes_history, plot_log_val_fitness, plot_sigma_history, plot_timing, plot_correlation
from model_io import save_model, load_model, convert_z_to_type_indices
from spectroscopy import (predict_dipole_trajectory, predict_polarizability_trajectory,
                           compute_ir_spectrum, plot_ir_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from ase.io import read


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

    # Convert to padded dense tensors for GPU-batched evaluation
    train_data = pad_and_stack(train_data)
    test_data = pad_and_stack(test_data)
    val_data = pad_and_stack(val_data)

    # dim_q is determined by the SOAP descriptor size
    cfg.dim_q = train_data["descriptors"][0].shape[-1]
    print("Dimension of q: " + str(cfg.dim_q))

    # Memory check: estimate padded tensor sizes vs available RAM/VRAM
    def _estimate_tensor_mb(data: dict) -> float:
        total_bytes = 0
        for key, val in data.items():
            if isinstance(val, tf.Tensor):
                total_bytes += val.dtype.size * int(tf.size(val))
        return total_bytes / (1024 * 1024)

    tensor_mb = max(_estimate_tensor_mb(train_data),
                    _estimate_tensor_mb(test_data),
                    _estimate_tensor_mb(val_data))

    use_gpu = bool(tf.config.list_physical_devices('GPU'))
    if use_gpu:
        # Assume 12 GB VRAM (common GPU); adjust if your GPU differs
        total_vram_mb = 12288
        usage_pct = 100 * tensor_mb / total_vram_mb
        print(f"\n=== Memory Check (GPU) ===")
        print(f"  Largest padded tensor set: {tensor_mb:.1f} MB")
        print(f"  Assumed VRAM:              {total_vram_mb:.0f} MB")
        print(f"  Estimated usage:           {usage_pct:.1f}%")
        if usage_pct > 90:
            print(f"  WARNING: Tensor data alone uses >{usage_pct:.0f}% of VRAM! "
                  f"Risk of OOM. Reduce batch_chunk_size or population_chunk_size.")
        elif usage_pct > 70:
            print(f"  WARNING: Tensor data uses {usage_pct:.0f}% of VRAM. "
                  f"May run tight during training.")
    else:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_ram_mb = int(line.split()[1]) / 1024
                    break
            else:
                total_ram_mb = None
        print(f"\n=== Memory Check (CPU-only — no GPU detected) ===")
        print(f"  Largest padded tensor set: {tensor_mb:.1f} MB")
        if total_ram_mb is not None:
            usage_pct = 100 * tensor_mb / total_ram_mb
            print(f"  Total system RAM:          {total_ram_mb:.0f} MB")
            print(f"  Estimated usage:           {usage_pct:.1f}%")
            if usage_pct > 70:
                print(f"  WARNING: Tensor data uses {usage_pct:.0f}% of RAM! "
                      f"Risk of OOM. Reduce dataset size or total_N.")
            elif usage_pct > 50:
                print(f"  WARNING: Tensor data uses {usage_pct:.0f}% of RAM. "
                      f"May run tight during training.")
        else:
            print(f"  WARNING: Could not determine total system RAM.")

    model = TNEP(cfg)
    print("Model Parameters: " + str(model.optimizer.dim))
    print("Population Size: " + str(model.optimizer.pop_size))
    print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
    print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))

    def periodic_plot_callback(history, gen):
        """Called during training at plot_interval to show progress."""
        print(f"\n--- Periodic plots at generation {gen} ---")
        m, preds = model.score(test_data)
        print(f"  Test RMSE: {float(m['rmse']):.4f}  R²: {float(m['r2']):.4f}")
        plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_correlation(test_data["targets"].numpy(), preds.numpy(), m, cfg,
                         cfg.save_plots, cfg.show_plots)

    # Train
    history = model.fit(train_data, val_data,
                        plot_callback=periodic_plot_callback if cfg.plot_interval else None)

    # Test
    metrics, test_preds = model.score(test_data)
    print_score_summary(metrics, cfg, prefix="Model test set")

    # Save model
    if cfg.save_path is not None:
        save_model(model, cfg, cfg.save_path)

    # Timing summary
    timing = history.get("timing", {})
    if timing:
        phases = ["sample_batch", "evaluate", "rank_update", "validate", "overhead"]
        grand = sum(sum(timing[p]) for p in phases)
        n_gens = len(timing["evaluate"])
        print(f"\n=== Timing Breakdown ({grand:.2f}s over {n_gens} generations) ===")
        for p in phases:
            t = sum(timing[p])
            avg = t / max(n_gens, 1)
            pct = 100 * t / max(grand, 1e-9)
            print(f"  {p:15s}: {t:.3f}s total ({pct:5.1f}%) | {avg*1000:.1f}ms/gen")

    # VRAM summary
    vram = history.get("vram_mb", [])
    if vram:
        print(f"\n=== VRAM Usage ===")
        print(f"  Peak: {max(vram):.1f} MB / 12288 MB ({100*max(vram)/12288:.1f}%)")
        print(f"  Last: {vram[-1]:.1f} MB")

    # Plots
    plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_correlation(test_data["targets"].numpy(), test_preds.numpy(), metrics, cfg,
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

    data = prepare_eval_data(dataset, cfg)

    # Score
    metrics, predictions = model.score(data)
    print_score_summary(metrics, cfg, prefix="External test")

    # Plot correlation
    plot_correlation(data["targets"].numpy(), predictions.numpy(), metrics, cfg,
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
