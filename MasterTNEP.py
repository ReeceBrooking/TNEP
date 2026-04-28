from __future__ import annotations

import numpy as np
import os

_slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
_cpu_threads = int(_slurm_cpus) if _slurm_cpus else max(os.cpu_count() // 2, 1)

_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
_has_gpu = (_cuda_visible not in ('', '-1')) or os.path.exists('/dev/nvidiactl')

os.environ['OMP_NUM_THREADS'] = str(_cpu_threads)
os.environ['MKL_NUM_THREADS'] = str(_cpu_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_cpu_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
if _has_gpu:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from data import (collect, split, pad_and_stack,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data, print_score_summary,
                  _resolve_target_key)
from plotting import (plot_snes_history, plot_log_val_fitness, plot_sigma_history,
                      plot_timing, plot_correlation, plot_cosine_similarity,
                      plot_loss_breakdown, plot_error_vs_magnitude)
from model_io import save_model, save_history, setup_run_directory, load_model
from spectroscopy import (predict_dipole_trajectory, predict_polarizability_trajectory,
                           compute_ir_spectrum, plot_ir_spectrum, plot_power_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from ase.io import read
from tqdm import tqdm


def train_model(cfg: TNEPconfig | None = None) -> TNEP:
    """Run full TNEP training pipeline: load, split, train, test, plot, save.

    Args:
        cfg : TNEPconfig or None (uses defaults)

    Returns:
        model  : trained TNEP model (access config via model.cfg)
    """
    if cfg is None:
        cfg = TNEPconfig()

    # Load dataset, filter by species, then filter bad data
    dataset, dataset_types_int = collect(cfg)
    cfg.type_map = {z: idx for idx, z in enumerate(cfg.types)}

    if cfg.target_mode == 1:
        print_dipole_statistics(dataset, cfg, target_key=_resolve_target_key(cfg))
    elif cfg.target_mode == 2:
        print_polarizability_statistics(dataset, target_key=_resolve_target_key(cfg))

    cfg.randomise(dataset)

    # Split into train/test/val and build SOAP descriptors (slow)
    train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

    # Convert to padded dense tensors for GPU-batched evaluation
    train_data = pad_and_stack(train_data, num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu)
    test_data  = pad_and_stack(test_data,  num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu)
    val_data   = pad_and_stack(val_data,   num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu)

    # dim_q is determined by the SOAP descriptor size
    cfg.dim_q = train_data["descriptors"][0].shape[-1]
    print("Dimension of q: " + str(cfg.dim_q))

    # Set up run directory: models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/
    if cfg.save_path is not None:
        setup_run_directory(cfg)

    model = TNEP(cfg)
    print(f"Model Parameters: {model.optimizer.dim}  |  Population Size: {model.optimizer.pop_size}")
    print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
    print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))

    def periodic_plot_callback(history, gen):
        """Called during training at plot_interval to show progress."""
        print(f"\n--- Periodic plots at generation {gen} ---")
        m, preds = model.score(test_data)
        print(f"  Test RMSE: {float(m['rmse']):.4f}  R²: {float(m['r2']):.4f}")
        if "total_rmse" in m:
            print(f"  Test total RMSE: {float(m['total_rmse']):.4f}  "
                  f"total R²: {float(m['total_r2']):.4f}")
        # Save periodic plots into a generation-specific subfolder
        gen_save = os.path.join(cfg.save_plots, f"{gen}_plots") if cfg.save_plots else None
        plot_snes_history(history, cfg, gen_save, cfg.show_plots)
        plot_log_val_fitness(history, cfg, gen_save, cfg.show_plots)
        plot_sigma_history(history, cfg, gen_save, cfg.show_plots)
        plot_loss_breakdown(history, cfg, gen_save, cfg.show_plots)
        plot_timing(history, cfg, gen_save, cfg.show_plots)
        plot_correlation(test_data["targets"].numpy(), preds.numpy(), m, cfg,
                         gen_save, cfg.show_plots, suffix="per_atom")
        plot_cosine_similarity(m, cfg, gen_save, cfg.show_plots,
                               suffix="per_atom")
        plot_error_vs_magnitude(test_data["targets"].numpy(), preds.numpy(), cfg,
                                gen_save, cfg.show_plots, suffix="per_atom")
        if "total_rmse" in m:
            na = test_data["num_atoms"].numpy().astype(np.float32)[:, np.newaxis]
            plot_correlation(test_data["targets"].numpy() * na, preds.numpy() * na,
                             {"rmse": m["total_rmse"], "r2": m["total_r2"],
                              "r2_components": m["total_r2_components"],
                              **({k: m[k] for k in ("cos_sim_mean", "cos_sim_all") if k in m})},
                             cfg, gen_save, cfg.show_plots, suffix="total")
            plot_error_vs_magnitude(test_data["targets"].numpy() * na, preds.numpy() * na, cfg,
                                    gen_save, cfg.show_plots, suffix="total")

    # Train
    history, final_model, best_val_model = model.fit(
        train_data, val_data,
        plot_callback=periodic_plot_callback if cfg.plot_interval else None)

    # Score final-generation model
    final_metrics, final_preds = final_model.score(test_data)
    print_score_summary(final_metrics, cfg, prefix="Final-gen test set")

    # Score best-val model
    metrics, test_preds = best_val_model.score(test_data)
    print_score_summary(metrics, cfg, prefix="Best-val test set")

    # Save models and history
    if cfg.save_path is not None:
        save_model(best_val_model, cfg, cfg.save_path, label="best_val")
        save_model(final_model, cfg, cfg.save_path, label="final_gen")
        save_history(history, cfg)

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

    # Plots
    plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_loss_breakdown(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
    # Best-val model: test set plots (per-atom)
    plot_correlation(test_data["targets"].numpy(), test_preds.numpy(), metrics, cfg,
                     cfg.save_plots, cfg.show_plots, suffix="best_val_per_atom")
    plot_cosine_similarity(metrics, cfg, cfg.save_plots, cfg.show_plots,
                           suffix="best_val_per_atom")
    plot_error_vs_magnitude(test_data["targets"].numpy(), test_preds.numpy(), cfg,
                            cfg.save_plots, cfg.show_plots, suffix="best_val_per_atom")
    # Best-val model: test set plots (total) — only when target scaling is active
    # Cosine similarity omitted for total: scale-invariant, already plotted at per-atom scale
    if "total_rmse" in metrics:
        num_atoms = test_data["num_atoms"].numpy().astype(np.float32)
        scale = num_atoms[:, np.newaxis]
        total_targets = test_data["targets"].numpy() * scale
        total_preds = test_preds.numpy() * scale
        total_metrics = {
            "rmse": metrics["total_rmse"],
            "r2": metrics["total_r2"],
            "r2_components": metrics["total_r2_components"],
        }
        if "cos_sim_all" in metrics:
            total_metrics["cos_sim_mean"] = metrics["cos_sim_mean"]
            total_metrics["cos_sim_all"] = metrics["cos_sim_all"]
        plot_correlation(total_targets, total_preds, total_metrics, cfg,
                         cfg.save_plots, cfg.show_plots, suffix="best_val_total")
        plot_error_vs_magnitude(total_targets, total_preds, cfg,
                                cfg.save_plots, cfg.show_plots, suffix="best_val_total")

    # Best-val model: validation set plots (per-atom)
    val_metrics, val_preds = best_val_model.score(val_data)
    plot_correlation(val_data["targets"].numpy(), val_preds.numpy(), val_metrics, cfg,
                     cfg.save_plots, cfg.show_plots, suffix="best_val_val_per_atom")
    plot_cosine_similarity(val_metrics, cfg, cfg.save_plots, cfg.show_plots,
                           suffix="best_val_val_per_atom")
    plot_error_vs_magnitude(val_data["targets"].numpy(), val_preds.numpy(), cfg,
                            cfg.save_plots, cfg.show_plots, suffix="best_val_val_per_atom")
    # Best-val model: validation set plots (total)
    # Cosine similarity omitted for total: scale-invariant, already plotted at per-atom scale
    if "total_rmse" in val_metrics:
        num_atoms_val = val_data["num_atoms"].numpy().astype(np.float32)
        scale_val = num_atoms_val[:, np.newaxis]
        val_total_targets = val_data["targets"].numpy() * scale_val
        val_total_preds = val_preds.numpy() * scale_val
        val_total_metrics = {
            "rmse": val_metrics["total_rmse"],
            "r2": val_metrics["total_r2"],
            "r2_components": val_metrics["total_r2_components"],
        }
        if "cos_sim_all" in val_metrics:
            val_total_metrics["cos_sim_mean"] = val_metrics["cos_sim_mean"]
            val_total_metrics["cos_sim_all"] = val_metrics["cos_sim_all"]
        plot_correlation(val_total_targets, val_total_preds, val_total_metrics, cfg,
                         cfg.save_plots, cfg.show_plots, suffix="best_val_val_total")
        plot_error_vs_magnitude(val_total_targets, val_total_preds, cfg,
                                cfg.save_plots, cfg.show_plots, suffix="best_val_val_total")

    # Final-gen model: test set plots (per-atom)
    plot_correlation(test_data["targets"].numpy(), final_preds.numpy(), final_metrics, cfg,
                     cfg.save_plots, cfg.show_plots, suffix="final_gen_per_atom")
    plot_cosine_similarity(final_metrics, cfg, cfg.save_plots, cfg.show_plots,
                           suffix="final_gen_per_atom")
    plot_error_vs_magnitude(test_data["targets"].numpy(), final_preds.numpy(), cfg,
                            cfg.save_plots, cfg.show_plots, suffix="final_gen_per_atom")
    # Final-gen model: test set plots (total)
    # Cosine similarity omitted for total: scale-invariant, already plotted at per-atom scale
    if "total_rmse" in final_metrics:
        na = test_data["num_atoms"].numpy().astype(np.float32)[:, np.newaxis]
        fg_total_targets = test_data["targets"].numpy() * na
        fg_total_preds = final_preds.numpy() * na
        fg_total_metrics = {
            "rmse": final_metrics["total_rmse"],
            "r2": final_metrics["total_r2"],
            "r2_components": final_metrics["total_r2_components"],
        }
        if "cos_sim_all" in final_metrics:
            fg_total_metrics["cos_sim_mean"] = final_metrics["cos_sim_mean"]
            fg_total_metrics["cos_sim_all"] = final_metrics["cos_sim_all"]
        plot_correlation(fg_total_targets, fg_total_preds, fg_total_metrics, cfg,
                         cfg.save_plots, cfg.show_plots, suffix="final_gen_total")
        plot_error_vs_magnitude(fg_total_targets, fg_total_preds, cfg,
                                cfg.save_plots, cfg.show_plots, suffix="final_gen_total")

    print("Run complete!")
    return best_val_model


def test_model(
    model: TNEP,
    data_path: str,
    save_plots: str | None = None,
    show_plots: bool = True,
) -> tuple[dict, tf.Tensor]:
    """Test a trained model on an external dataset.

    Loads structures from data_path, builds descriptors, and scores.

    Args:
        model      : trained TNEP model (config accessed via model.cfg)
        data_path  : str — path to .xyz file with test structures
        save_plots : str or None — directory to save plot into (None = don't save)
        show_plots : bool — True to display plot interactively (default True)

    Returns:
        metrics    : dict with rmse, r2, r2_components, etc.
        predictions : [S, T] tensor of predictions
    """
    cfg = model.cfg
    dataset = read(data_path, index=":")
    print(f"Loaded {len(dataset)} structures from {data_path}")

    data = prepare_eval_data(dataset, cfg)

    # Score
    metrics, predictions = model.score(data)
    print_score_summary(metrics, cfg, prefix="External test")

    # Plot per-atom
    plot_correlation(data["targets"].numpy(), predictions.numpy(), metrics, cfg,
                     save_plots, show_plots, suffix="per_atom")
    plot_cosine_similarity(metrics, cfg, save_plots, show_plots, suffix="per_atom")
    plot_error_vs_magnitude(data["targets"].numpy(), predictions.numpy(), cfg,
                            save_plots, show_plots, suffix="per_atom")

    # Plot total
    if "total_rmse" in metrics:
        num_atoms = data["num_atoms"].numpy().astype(np.float32)
        scale = num_atoms[:, np.newaxis]
        total_targets = data["targets"].numpy() * scale
        total_preds = predictions.numpy() * scale
        total_metrics = {
            "rmse": metrics["total_rmse"],
            "r2": metrics["total_r2"],
            "r2_components": metrics["total_r2_components"],
        }
        if "cos_sim_all" in metrics:
            total_metrics["cos_sim_mean"] = metrics["cos_sim_mean"]
            total_metrics["cos_sim_all"] = metrics["cos_sim_all"]
        plot_correlation(total_targets, total_preds, total_metrics, cfg,
                         save_plots, show_plots, suffix="total")
        plot_error_vs_magnitude(total_targets, total_preds, cfg,
                                save_plots, show_plots, suffix="total")

    return metrics, predictions


def process_trajectory(
    model: TNEP,
    trajectory_path: str,
    dt_fs: float = 1.0,
    save_plots: str | None = "plots",
    show_plots: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Predict properties along an MD trajectory and compute spectra.

    For dipole models (mode 1): predicts dipole trajectory and computes IR spectrum.
    For polarizability models (mode 2): predicts polarizability trajectory and
    computes Raman spectrum.

    Args:
        model           : trained TNEP model (config accessed via model.cfg)
        trajectory_path : str — path to .xyz trajectory file
        dt_fs           : float — MD timestep in femtoseconds
        save_plots      : str or None — directory to save plot into (default "plots")
        show_plots      : bool — True to display plot interactively (default False)
        batch_size      : int or None — if set, process the trajectory in batches of
                          this many frames to avoid computing all descriptors at once.
                          If None, process the entire trajectory in one go.

    Returns:
        For mode 1 (dipole):
            dict with keys: dipoles, freq_cm, intensity, acf
        For mode 2 (polarizability):
            dict with keys: polarizabilities, freq_cm, I_VV, I_VH, I_total,
                            acf_iso, acf_aniso
    """
    cfg = model.cfg
    trajectory = read(trajectory_path, index=":")
    print(f"Loaded {len(trajectory)} frames from {trajectory_path}")

    dataset_types_int = assign_type_indices(trajectory, cfg.types)

    if cfg.target_mode == 1:
        if batch_size is None:
            dipoles = predict_dipole_trajectory(model, trajectory, dataset_types_int)
        else:
            n_frames = len(trajectory)
            batches = []
            starts = list(range(0, n_frames, batch_size))
            for start in tqdm(starts, desc="Dipole batches", unit="batch"):
                end = min(start + batch_size, n_frames)
                batch_dipoles = predict_dipole_trajectory(
                    model, trajectory[start:end], dataset_types_int[start:end])
                batches.append(batch_dipoles)
            dipoles = np.concatenate(batches, axis=0)
        freq_cm, intensity, power, acf = compute_ir_spectrum(dipoles, dt_fs=dt_fs)
        plot_ir_spectrum(freq_cm, intensity, cfg, save_plots, show_plots)
        plot_power_spectrum(freq_cm, power, cfg, save_plots, show_plots)
        return {"dipoles": dipoles, "freq_cm": freq_cm, "intensity": intensity, "power": power, "acf": acf}

    elif cfg.target_mode == 2:
        if batch_size is None:
            pols = predict_polarizability_trajectory(model, trajectory, dataset_types_int)
        else:
            n_frames = len(trajectory)
            batches = []
            starts = list(range(0, n_frames, batch_size))
            for start in tqdm(starts, desc="Polarizability batches", unit="batch"):
                end = min(start + batch_size, n_frames)
                batch_pols = predict_polarizability_trajectory(
                    model, trajectory[start:end], dataset_types_int[start:end])
                batches.append(batch_pols)
            pols = np.concatenate(batches, axis=0)
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
    model = train_model()
    #model = load_model("models/n30_q75_pop80_20260422_142744_water_monomer_dipole/water_monomer_O_H_dipole_best_val.npz")
    #dipoles = process_trajectory(model, "datasets/water_monomer_traj.xyz", batch_size=3000)
    