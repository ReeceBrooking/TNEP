from __future__ import annotations

import numpy as np
import os

# CPU parallelisation: detect GPU vs CPU-only and set thread counts accordingly
# Physical cores = logical cores / 2 (excludes hyperthreads)
_logical = os.cpu_count() or 1
_physical = max(_logical // 2, 1)
_has_gpu = os.path.isdir('/proc/driver/nvidia') or os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''

if _has_gpu:
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
else:
    os.environ['OMP_NUM_THREADS'] = str(_physical)
    os.environ['MKL_NUM_THREADS'] = str(_physical)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(_physical)
    os.environ['TF_NUM_INTEROP_THREADS'] = '4'
import tensorflow as tf

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from data import (collect, split, pad_and_stack,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data, print_score_summary)
from plotting import (plot_snes_history, plot_log_val_fitness, plot_sigma_history,
                      plot_timing, plot_correlation, plot_loss_breakdown,
                      plot_error_vs_magnitude)
from model_io import save_model, setup_run_directory
from spectroscopy import (predict_dipole_trajectory, predict_polarizability_trajectory,
                           compute_ir_spectrum, plot_ir_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from debug_signs import (diagnose_sign_flips, correct_sign_flips, check_cells,
                         characterize_flipped, test_target_negation)
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

    # Load dataset, filter by species, then filter bad data
    dataset, dataset_types_int = collect(cfg)

    if cfg.target_mode == 1:
        print_dipole_statistics(dataset)
    elif cfg.target_mode == 2:
        print_polarizability_statistics(dataset)

    cfg.randomise(dataset)

    # Split into train/test/val and build SOAP descriptors (slow)
    train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

    # Convert to padded dense tensors for GPU-batched evaluation
    train_data = pad_and_stack(train_data, num_types=cfg.num_types)
    test_data = pad_and_stack(test_data, num_types=cfg.num_types)
    val_data = pad_and_stack(val_data, num_types=cfg.num_types)

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
        plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_loss_breakdown(history, cfg, cfg.save_plots, cfg.show_plots)
        plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)
        if "total_rmse" in m:
            na = test_data["num_atoms"].numpy().astype(np.float32)[:, np.newaxis]
            plot_correlation(test_data["targets"].numpy() * na, preds.numpy() * na,
                             {"rmse": m["total_rmse"], "r2": m["total_r2"],
                              "r2_components": m["total_r2_components"],
                              **({k: m[k] for k in ("cos_sim_mean", "cos_sim_all") if k in m})},
                             cfg, cfg.save_plots, cfg.show_plots, suffix=f"total_dipole_gen{gen}")
        else:
            plot_correlation(test_data["targets"].numpy(), preds.numpy(), m, cfg,
                             cfg.save_plots, cfg.show_plots, suffix=f"gen{gen}")

    # Train
    history, final_model, best_val_model = model.fit(
        train_data, val_data,
        plot_callback=periodic_plot_callback if cfg.plot_interval else None)

    # Score final-generation model
    final_metrics, final_preds = final_model.score(test_data)
    print_score_summary(final_metrics, cfg, prefix="Final-gen test set")
    if "total_rmse" in final_metrics:
        na = test_data["num_atoms"].numpy().astype(np.float32)[:, np.newaxis]
        plot_correlation(test_data["targets"].numpy() * na, final_preds.numpy() * na,
                         {"rmse": final_metrics["total_rmse"], "r2": final_metrics["total_r2"],
                          "r2_components": final_metrics["total_r2_components"],
                          **({k: final_metrics[k] for k in ("cos_sim_mean", "cos_sim_all") if k in final_metrics})},
                         cfg, cfg.save_plots, cfg.show_plots, suffix="total_dipole_final_gen")
    else:
        plot_correlation(test_data["targets"].numpy(), final_preds.numpy(), final_metrics, cfg,
                         cfg.save_plots, cfg.show_plots, suffix="final_gen")

    # Score best-val model
    metrics, test_preds = best_val_model.score(test_data)
    print_score_summary(metrics, cfg, prefix="Best-val test set")

    # Debug sign-flipped dipole predictions
    if cfg.target_mode == 1:
        diag = diagnose_sign_flips(best_val_model, test_data)
        if cfg.test_data_path is not None:
            test_structures = read(cfg.test_data_path, index=":")
            if cfg.allowed_species:
                from ase.data import atomic_numbers
                allowed = set(atomic_numbers[z] if isinstance(z, str) else z
                              for z in cfg.allowed_species)
                test_structures = [s for s in test_structures
                                   if set(s.numbers).issubset(allowed)]
        else:
            n_test = int(cfg.test_ratio * len(cfg.indices))
            test_idx = cfg.indices[:n_test]
            test_structures = [dataset[i] for i in test_idx]
        check_cells(test_structures)
        if diag["flipped_idx"]:
            characterize_flipped(test_structures, diag["flipped_idx"], diag["good_idx"])
            test_target_negation(diag["preds"], diag["targets"], diag["flipped_idx"])
        # NOTE: verify_gradient_sign() is not called here because it invokes
        # quippy's C descriptor code again, which can cause heap corruption
        # (malloc_consolidate) after training. Run it standalone if needed:
        #   verify_gradient_sign(model, structures, cfg, structure_idx=0)

        # Correct flipped predictions and recompute metrics
        if cfg.fix_sign_flips and diag["flipped_idx"]:
            corrected_preds = correct_sign_flips(diag["preds"], diag["cos_sim"])
            test_preds = tf.constant(corrected_preds, dtype=tf.float32)
            targets_np = diag["targets"]
            diff = corrected_preds - targets_np
            rmse = np.sqrt(np.mean(diff ** 2))
            dot = np.sum(corrected_preds * targets_np, axis=1)
            norm_p = np.linalg.norm(corrected_preds, axis=1)
            norm_t = np.linalg.norm(targets_np, axis=1)
            cos_sim = dot / np.maximum(norm_p * norm_t, 1e-12)
            print("\n=== After sign-flip correction ===")
            print(f"  RMSE:          {rmse:.4f}")
            print(f"  Mean cos_sim:  {cos_sim.mean():.4f}")

    # Save model
    if cfg.save_path is not None:
        save_model(best_val_model, cfg, cfg.save_path)

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
    # Plot total (not per-atom) dipole correlations for test and val data
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
                         cfg.save_plots, cfg.show_plots, suffix="total_dipole")
        plot_error_vs_magnitude(total_targets, total_preds, cfg,
                                cfg.save_plots, cfg.show_plots, suffix="total_dipole")
    else:
        plot_correlation(test_data["targets"].numpy(), test_preds.numpy(), metrics, cfg,
                         cfg.save_plots, cfg.show_plots)
        plot_error_vs_magnitude(test_data["targets"].numpy(), test_preds.numpy(), cfg,
                                cfg.save_plots, cfg.show_plots)

    # Validation set correlation plot
    val_metrics, val_preds = best_val_model.score(val_data)
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
                         cfg.save_plots, cfg.show_plots, suffix="total_dipole_val")
        plot_error_vs_magnitude(val_total_targets, val_total_preds, cfg,
                                cfg.save_plots, cfg.show_plots, suffix="total_dipole_val")
    else:
        plot_correlation(val_data["targets"].numpy(), val_preds.numpy(), val_metrics, cfg,
                         cfg.save_plots, cfg.show_plots, suffix="val")
        plot_error_vs_magnitude(val_data["targets"].numpy(), val_preds.numpy(), cfg,
                                cfg.save_plots, cfg.show_plots, suffix="val")

    print("Run complete!")
    return best_val_model, cfg


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
