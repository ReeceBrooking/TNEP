from __future__ import annotations

import numpy as np
import os

_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
_has_gpu = (_cuda_visible not in ('', '-1')) or os.path.exists('/dev/nvidiactl')


def _allocated_cpu_count() -> int:
    """CPU cores this process may use.

    Under Slurm, use ALL allocated cores so the CPU forward pass saturates the
    node: SLURM_CPUS_PER_TASK, then SLURM_CPUS_ON_NODE, then the cpuset affinity
    mask — never the ``//2`` heuristic (which would halve the node when
    ``--cpus-per-task`` was omitted). Off Slurm, use ``cpu_count() // 2`` to
    avoid hogging a shared workstation.
    """
    for _var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        _val = os.environ.get(_var)
        if _val:
            try:
                # SLURM_CPUS_ON_NODE is a plain int, but guard "128(x2)".
                return max(int(_val.split('(')[0]), 1)
            except ValueError:
                pass
    # Slurm allocation without an explicit CPU count: use the affinity mask.
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID'):
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except AttributeError:                           # non-Linux
            pass
    return max((os.cpu_count() or 2) // 2, 1)


_cpu_threads = _allocated_cpu_count()

# Threading budget. GPU run: >4 main-process threads cost more scheduler
# overhead than they save (heavy work is on-device). CPU-only run: the
# matmul-heavy forward pass needs the full Slurm allocation. NUMEXPR /
# OPENBLAS pinned so NumPy paths in data.py don't oversubscribe.
# NOTE: these env vars MUST be set before TensorFlow is imported below.
_main_threads = 4 if _has_gpu else _cpu_threads
os.environ['OMP_NUM_THREADS'] = str(_main_threads)
os.environ['MKL_NUM_THREADS'] = str(_main_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(_main_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = '2' if _has_gpu else '4'
if _has_gpu:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

# Memory growth: stops TF grabbing the whole GPU at startup so shared queue
# nodes can coexist with other tenants. No-op on CPU. try guards the case
# where TF has already initialised the device.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from data import (collect, split, pad_and_stack,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data, print_score_summary,
                  _resolve_target_key, materialize_test_data)
from plotting import (plot_snes_history, plot_log_val_fitness, plot_sigma_history,
                      plot_timing, plot_correlation, plot_cosine_similarity,
                      plot_loss_breakdown, plot_error_vs_magnitude)
from model_io import save_model, save_history, setup_run_directory, load_model
from spectroscopy import (predict_trajectory_batch,
                           compute_ir_spectrum, plot_ir_spectrum, plot_power_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from DescriptorBuilder import make_descriptor_builder
from tqdm import tqdm
from ase.io import read, iread, write


def _apply_csc_overrides(cfg: TNEPconfig) -> None:
    """When `cfg.csc_enable=True`, retune chunk sizing to the Mahti hardware.

    GPU profile: A100 fits the whole population on-device, so drop chunking to
    the minimum. CPU profile: pin data to host (no GPU to upload to) and keep
    chunk sizes modest since larger chunks just inflate per-op CPU latency.
    No-op outside csc_enable mode.
    """
    if not getattr(cfg, "csc_enable", False):
        return

    # Profile: explicit cfg.csc_profile, else "auto" hardware detection.
    requested = str(getattr(cfg, "csc_profile", "auto")).lower()
    if requested not in ("auto", "cpu", "gpu"):
        raise ValueError(
            f"cfg.csc_profile={requested!r} not recognised; expected "
            f"one of 'auto', 'cpu', 'gpu'.")
    if requested == "gpu":
        use_gpu_profile = True
    elif requested == "cpu":
        use_gpu_profile = False
    else:
        use_gpu_profile = _has_gpu

    overrides: dict[str, object] = {}
    if use_gpu_profile:
        # GPU tuning per project_mahti_speedup memory (empirical).
        overrides.update({
            "population_chunk_size": None,
            "batch_chunk_size": 2000,
            "pin_data_to_cpu": False,
        })
        profile = "GPU"
    else:
        # CPU partition: pin data to host (no GPU to copy to); smaller
        # batch_chunk_size keeps per-op CPU latency reasonable.
        overrides.update({
            "population_chunk_size": 10,
            "batch_chunk_size": 500,
            "pin_data_to_cpu": True,
        })
        profile = "CPU"

    if requested != "auto":
        profile = f"{profile} (forced via cfg.csc_profile={requested!r})"
    changed = []
    for k, v in overrides.items():
        if getattr(cfg, k, None) != v:
            changed.append(f"{k}={getattr(cfg, k, None)!r}→{v!r}")
            setattr(cfg, k, v)
    print(f"  csc_enable=True ({profile} profile, OMP={_main_threads})"
          + (f" — {', '.join(changed)}" if changed else " — no changes needed"))


def train_model(cfg: TNEPconfig | None = None,
                checkpoint: str | None = None,
                extract_model: bool = False,
                cfg_overrides: dict | None = None) -> TNEP:
    """Run full TNEP training pipeline: load, split, train, test, plot, save.

    Args:
        cfg          : TNEPconfig or None (defaults). Ignored when
                       `checkpoint` is set (the checkpoint embeds its own cfg).
        checkpoint   : optional path to a `checkpoint.h5`. When provided, cfg is
                       loaded from it and training resumes from `last_gen + 1`.
        extract_model: when True, skip the SNES loop and build final_model /
                       best_val_model directly from the checkpoint's μ / best_μ,
                       then score/save/plot as a completed run. Requires
                       `checkpoint`.
        cfg_overrides: optional dict of cfg field → value applied after
                       load_checkpoint, before the model is built. Repairs old
                       checkpoints whose saved JSON is missing architecture
                       fields (else class defaults leak in → μ-shape mismatch).

    Returns:
        model  : trained TNEP model (access config via model.cfg)
    """
    if extract_model and checkpoint is None:
        raise ValueError(
            "extract_model=True requires a `checkpoint` path — there is "
            "nothing to extract without a stored μ / best_μ.")
    if cfg_overrides is not None and checkpoint is None:
        raise ValueError(
            "cfg_overrides is only meaningful with `checkpoint` — pass "
            "the values directly via `cfg` when starting from scratch.")
    resume_state = None
    if checkpoint is not None:
        from model_io import load_checkpoint
        cfg, resume_state = load_checkpoint(checkpoint)
        if cfg_overrides:
            print(f"  applying cfg_overrides: {cfg_overrides}")
            # Validate keys: a typo would otherwise create a new attribute
            # via setattr and silently never apply. Hard fail on unknowns.
            valid_keys = set(getattr(TNEPconfig, "__annotations__", {}).keys())
            unknown = [k for k in cfg_overrides if k not in valid_keys]
            if unknown:
                # Suggest near matches.
                import difflib
                hints = []
                for bad in unknown:
                    suggestions = difflib.get_close_matches(bad, valid_keys, n=3)
                    if suggestions:
                        hints.append(f"  {bad!r} → did you mean {suggestions}?")
                    else:
                        hints.append(f"  {bad!r} (no close match)")
                raise ValueError(
                    f"cfg_overrides contains unknown TNEPconfig field(s):\n"
                    + "\n".join(hints))
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)
        if extract_model:
            print(f"Extracting models from {checkpoint} "
                  f"(treating gen {resume_state['last_gen'] + 1} as final; "
                  f"no further SNES generations will run).")
        else:
            print(f"Resuming training from {checkpoint} "
                  f"(continuing at gen {resume_state['last_gen'] + 1} "
                  f"of {cfg.num_generations})")
    elif cfg is None:
        cfg = TNEPconfig()

    # CSC / Slurm mode: retune chunk sizing before downstream code runs.
    _apply_csc_overrides(cfg)

    return _train_model_inner(cfg, resume_state=resume_state,
                              extract_model=extract_model)


def _plot_eval_set(cfg: TNEPconfig, data: dict, preds, metrics: dict,
                   suffix_per_atom: str, suffix_total: str,
                   save_dir: str | None = None,
                   show: bool | None = None) -> None:
    """Emit correlation / cos-sim / error-vs-magnitude plots for one
    (data, predictions, metrics) triple.

    Always emits the per-atom variant; emits the total variant (per-atom ×
    num_atoms) only when metrics contain `total_*` keys (scale_targets=True).
    Cos-sim is omitted from total plots (scale-invariant). `save_dir` / `show`
    override cfg.save_plots / cfg.show_plots when non-None.
    """
    targets = data["targets"].numpy()
    preds_np = preds.numpy()
    save = save_dir if save_dir is not None else cfg.save_plots
    show = show if show is not None else cfg.show_plots

    # RRMSE = RMSE / std(ref) (≡ sqrt(1 - R²)), computed once from total-scale
    # values and shown identically on per-atom and total plots. Per-component
    # uses per-component std; overall uses the flattened target std.
    has_total = "total_rmse" in metrics
    if has_total:
        scale = data["num_atoms"].numpy().astype(np.float32)[:, np.newaxis]
        total_targets = targets * scale
        total_preds = preds_np * scale
        total_rmse_scalar = float(metrics["total_rmse"])
    else:
        # cfg.scale_targets=False or PES — `targets` is already total.
        total_targets = targets
        total_preds = preds_np
        total_rmse_scalar = float(metrics["rmse"])

    total_diff = total_targets - total_preds
    target_std_overall = max(float(total_targets.std()), 1e-12)
    target_std_comp = np.maximum(total_targets.std(axis=0), 1e-12)
    rmse_comp_total = np.sqrt(np.mean(total_diff ** 2, axis=0))
    rrmse_payload = {
        "rrmse": total_rmse_scalar / target_std_overall,
        "rrmse_components": rmse_comp_total / target_std_comp,
    }

    plot_correlation(targets, preds_np, {**metrics, **rrmse_payload},
                     cfg, save, show, suffix=suffix_per_atom)
    plot_cosine_similarity(metrics, cfg, save, show, suffix=suffix_per_atom)
    plot_error_vs_magnitude(targets, preds_np, cfg, save, show, suffix=suffix_per_atom)

    if not has_total:
        return
    total_metrics = {
        "rmse": metrics["total_rmse"],
        "r2": metrics["total_r2"],
        "r2_components": metrics["total_r2_components"],
        **rrmse_payload,
    }
    # Carry cos-sim annotations into plot_correlation; standalone cos-sim
    # plot omitted at total scale (scale-invariant).
    if "cos_sim_all" in metrics:
        total_metrics["cos_sim_mean"] = metrics["cos_sim_mean"]
        total_metrics["cos_sim_all"] = metrics["cos_sim_all"]
    plot_correlation(total_targets, total_preds, total_metrics, cfg, save, show, suffix=suffix_total)
    plot_error_vs_magnitude(total_targets, total_preds, cfg, save, show, suffix=suffix_total)


def _setup_grad_staging(cfg: TNEPconfig, train_data: dict, val_data: dict) -> None:
    """Pre-stage per-chunk pair indices and pick the staging mode per
    `cfg.pin_data_to_cpu`.

    pin=False: move static fields to GPU; eval/score use the resident fast
    path. pin=True: tensors stay on host and each chunk is sliced on demand.
    """
    from data import prestage_chunk_indices, move_data_to_gpu

    chunk = cfg.batch_chunk_size
    pin = bool(getattr(cfg, "pin_data_to_cpu", True))

    # Pre-stage pair indices per chunk range. Tiny GPU-resident tensors.
    for d in (train_data, val_data):
        S = int(d["num_atoms"].shape[0])
        c = chunk if chunk is not None else S
        ranges = [(s, min(s + c, S)) for s in range(0, S, c)]
        prestage_chunk_indices(d, ranges)

    # GPU-resident path: move static fields on-device and set the flag that
    # triggers `_stage_chunk_resident` in `prefetched_chunks`.
    if not pin:
        for d in (train_data, val_data):
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)


def _print_param_breakdown(model) -> None:
    """Print a per-layer breakdown (component → count → % of total) of the
    SNES parameter budget. Polarisability mirrors the primary ANN when
    target_mode == 2.
    """
    opt = model.optimizer
    cfg = model.cfg
    total = int(opt.dim)
    if total <= 0:
        return

    rows: list[tuple[str, int]] = []

    # Primary ANN — per-layer breakdown.
    rows.append(("ANN  W0  [T·Q·H]", int(opt._n_W0)))
    rows.append(("ANN  b0  [T·H]",   int(opt._n_b0)))
    rows.append(("ANN  W1  [T·H]",   int(opt._n_W1)))
    rows.append(("ANN  b1",          int(opt._n_b1)))

    # Polarisability ANN — same per-layer shape, mirrored doubling.
    if cfg.target_mode == 2:
        rows.append(("ANN_pol  W0  [T·Q·H]", int(opt._n_W0)))
        rows.append(("ANN_pol  b0  [T·H]",   int(opt._n_b0)))
        rows.append(("ANN_pol  W1  [T·H]",   int(opt._n_W1)))
        rows.append(("ANN_pol  b1",          int(opt._n_b1)))

    # Optional add-ons.
    if int(opt.n_U_pair) > 0:
        rows.append(("input-side mixing  U_pair", int(opt.n_U_pair)))
    if int(opt.n_preprocess) > 0:
        rows.append(("preprocess  W_pre (summed only)", int(opt.n_preprocess)))

    accounted = sum(n for _, n in rows)
    if accounted != total:
        rows.append(("other / overhead", total - accounted))

    label_w = max(len(lbl) for lbl, _ in rows)
    print("Parameter breakdown:")
    for label, n in rows:
        pct = 100.0 * n / total
        print(f"  {label.ljust(label_w)}  {n:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL'.ljust(label_w)}  {total:>8d}")


def _train_model_inner(cfg: TNEPconfig,
                        resume_state: dict | None = None,
                        extract_model: bool = False) -> TNEP:
    """Body of train_model.

    On resume, the train/val split comes from the checkpoint's cfg.indices
    (not re-shuffled) and run-directory setup is skipped. When
    extract_model=True (resume only), the SNES loop is skipped and
    final_model / best_val_model are rebuilt from the checkpoint's μ / best_μ.
    """
    # Load dataset, filter by species, then filter bad data
    dataset, dataset_types_int = collect(cfg)
    cfg.type_map = {z: idx for idx, z in enumerate(cfg.types)}

    if cfg.target_mode == 1:
        print_dipole_statistics(dataset, cfg, target_key=_resolve_target_key(cfg))
    elif cfg.target_mode == 2:
        print_polarizability_statistics(dataset, target_key=_resolve_target_key(cfg))

    # Resolve cfg.seed=None to a concrete, serialisable seed BEFORE any RNG is
    # constructed, else the consumed entropy is lost and can't be reproduced.
    if cfg.seed is None:
        cfg.seed = int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        print(f"  cfg.seed was None — generated and stored "
              f"reproducible seed: {cfg.seed}")

    if resume_state is not None and isinstance(getattr(cfg, "indices", None), np.ndarray):
        # Reuse the checkpoint's split rather than re-shuffling.
        print(f"  resume: using checkpoint train/val split "
              f"({len(cfg.indices)} indices)")
    else:
        cfg.randomise(dataset)

    # Split into train/val (built now) and a deferred test placeholder; test
    # descriptors are built lazily on first scoring (materialize_test_data).
    train_data, test_pending, val_data = split(dataset, dataset_types_int, cfg)

    # Resolve dim_q before any consumer needs it; cross-checked below.
    from DescriptorBuilderGPU import compute_dim_q
    cfg.dim_q = compute_dim_q(cfg)

    # Pad to dense tensors for GPU-batched eval (test_data padded lazily).
    # When dipole_rij_power=0 (target_mode=1) only self-pairs contribute to
    # the dipole sum, so drop neighbour pairs (grad_values O(N·M) → O(N)).
    _self_only = (cfg.target_mode == 1
                  and int(getattr(cfg, "dipole_rij_power", 2)) == 0)
    train_data = pad_and_stack(
        train_data, num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        self_pairs_only=_self_only)
    val_data   = pad_and_stack(
        val_data,   num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        self_pairs_only=_self_only)
    if _self_only:
        n_train_pairs = int(train_data["grad_values"].shape[0])
        n_val_pairs   = int(val_data["grad_values"].shape[0])
        print(f"  dipole_rij_power=0: COO restricted to self-pairs only "
              f"(train P={n_train_pairs}, val P={n_val_pairs})")

    _setup_grad_staging(cfg, train_data, val_data)

    # Lazy test-data build; cached after the first call.
    def get_test_data():
        return materialize_test_data(test_pending, cfg,
                                     num_types=cfg.num_types,
                                     pin_to_cpu=cfg.pin_data_to_cpu)

    # Cross-check built descriptor shape vs cfg.dim_q (catches cfg/builder drift).
    built_dim_q = int(train_data["descriptors"][0].shape[-1])
    if built_dim_q != cfg.dim_q:
        raise RuntimeError(
            f"compute_dim_q={cfg.dim_q} disagrees with built descriptor "
            f"shape {built_dim_q}; cfg / builder mismatch.")
    print("Dimension of q: " + str(cfg.dim_q))

    # Set up run directory: models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/.
    # On resume, reuse the checkpoint's existing run dir.
    if cfg.save_path is not None and resume_state is None:
        setup_run_directory(cfg)
    elif resume_state is not None and cfg.save_path is not None:
        run_dir = os.path.dirname(cfg.save_path) or "."
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        cfg.save_plots = os.path.join(run_dir, "plots")
        print(f"  resume: writing outputs to existing run dir {run_dir}")

    model = TNEP(cfg)
    # With preprocess contraction on, cfg.dim_q is the contracted Q_new
    # (raw at cfg.dim_q_raw); report the compression ratio.
    if getattr(cfg, "descriptor_preprocess_contract", "off") != "off":
        q_raw  = int(getattr(cfg, "dim_q_raw", cfg.dim_q))
        q_new  = int(cfg.dim_q)
        ratio  = q_raw / q_new if q_new else float("inf")
        reduce = (1.0 - q_new / q_raw) * 100.0 if q_raw else 0.0
        print(f"Preprocess contraction ({cfg.descriptor_preprocess_contract}): "
              f"Q_raw={q_raw} → Q_new={q_new}  "
              f"(×{ratio:.2f} compression, {reduce:.1f}% reduction)")
    print(f"Model Parameters: {model.optimizer.dim}  |  Population Size: {model.optimizer.pop_size}")
    print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
    print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))
    _print_param_breakdown(model)

    def periodic_plot_callback(history, gen):
        """Called at plot_interval during training to show progress."""
        print(f"\n--- Periodic plots at generation {gen} ---")
        # First call triggers the deferred test descriptor build.
        test_data = get_test_data()
        m, preds = model.score(test_data)
        print(f"  Test RMSE: {float(m['rmse']):.4f}  R²: {float(m['r2']):.4f}")
        if "total_rmse" in m:
            print(f"  Test total RMSE: {float(m['total_rmse']):.4f}  "
                  f"total R²: {float(m['total_r2']):.4f}")
        gen_save = os.path.join(cfg.save_plots, f"{gen}_plots") if cfg.save_plots else None
        plot_snes_history(history, cfg, gen_save, cfg.show_plots)
        plot_log_val_fitness(history, cfg, gen_save, cfg.show_plots)
        plot_sigma_history(history, cfg, gen_save, cfg.show_plots)
        plot_loss_breakdown(history, cfg, gen_save, cfg.show_plots)
        plot_timing(history, cfg, gen_save, cfg.show_plots)
        _plot_eval_set(cfg, test_data, preds, m,
                       "per_atom", "total", save_dir=gen_save)

    # Train — unless extract_model=True, in which case rebuild final_model
    # and best_val_model from the checkpoint's μ / best_μ and reuse its history.
    if extract_model:
        from SNES import _set_model_params
        if resume_state is None:
            # Guard the internal contract (outer train_model also enforces this).
            raise ValueError(
                "_train_model_inner: extract_model=True requires "
                "resume_state (μ / best_μ must come from a checkpoint).")
        print(f"  extract_model: skipping SNES training loop; rebuilding "
              f"models from checkpoint state at gen "
              f"{resume_state['last_gen'] + 1}.")
        snes = model.optimizer
        ckpt_dim = int(np.asarray(resume_state["mu"]).size)
        if ckpt_dim != int(snes.dim):
            # μ dim mismatch: the checkpoint's cfg disagrees with the cfg used
            # at save time on an architecture field (usually missing JSON fields).
            raise ValueError(
                f"Checkpoint μ has dim {ckpt_dim} but the current model "
                f"builds dim={snes.dim}. The cfg loaded from the "
                f"checkpoint must disagree with the cfg used at save "
                f"time on at least one architectural field "
                f"(num_neurons, descriptor_mixing, "
                f"descriptor_mixing_per_type, "
                f"target_mode, num_types, alpha_max, l_max). The most "
                f"likely cause is legacy checkpoints that pre-date the "
                f"_serialize_config fix — class-default fields weren't "
                f"saved, so the current class defaults are leaking in.\n"
                f"Recovery: pass cfg_overrides={{...}} to train_model "
                f"with the architectural fields restored to their "
                f"original values. The run directory name "
                f"({getattr(cfg, 'save_path', None)}) usually encodes "
                f"num_neurons (n<H>_) and dim_q (q<Q>_)."
            )
        snes.mu.assign(tf.constant(resume_state["mu"], dtype=tf.float32))
        snes.sigma.assign(tf.constant(resume_state["sigma"], dtype=tf.float32))
        best_mu = tf.constant(resume_state["best_mu"], dtype=tf.float32)

        final_params = snes.reconstruct_params_tf(snes.mu)
        final_model = TNEP(cfg)
        _set_model_params(final_model, *final_params)

        best_val_params = snes.reconstruct_params_tf(best_mu)
        best_val_model = TNEP(cfg)
        _set_model_params(best_val_model, *best_val_params)

        # Mirror SNES.fit's final restoration: align in-place `model` and its
        # optimizer with best-val so downstream reuse sees best run-end state.
        snes.mu.assign(best_mu)
        # best_sigma may be absent in older snapshots; skip rather than error.
        if resume_state.get("best_sigma") is not None:
            snes.sigma.assign(tf.constant(
                resume_state["best_sigma"], dtype=tf.float32))
        _set_model_params(model, *best_val_params)

        history = resume_state["history"]
    else:
        history, final_model, best_val_model = model.fit(
            train_data, val_data,
            plot_callback=periodic_plot_callback if cfg.plot_interval else None,
            resume_state=resume_state)

    # Build test descriptors (if not already built) and score both models.
    test_data = get_test_data()

    final_metrics, final_preds = final_model.score(test_data)
    print_score_summary(final_metrics, cfg, prefix="Final-gen test set")

    metrics, test_preds = best_val_model.score(test_data)
    print_score_summary(metrics, cfg, prefix="Best-val test set")

    if cfg.save_path is not None:
        save_model(best_val_model, cfg, cfg.save_path, label="best_val")
        save_model(final_model, cfg, cfg.save_path, label="final_gen")
        save_history(history, cfg)

    # Timing summary
    timing = history.get("timing", {})
    if timing:
        phases = ["sample_batch", "evaluate", "rank_update", "validate", "overhead"]
        grand = sum(sum(timing[p]) for p in phases)
        # Sampled once per val_interval, so n_recorded is the val-tick count,
        # not the generation count (each tick is one gen's timing).
        n_recorded = len(timing["evaluate"])
        print(f"\n=== Timing Breakdown ({grand:.2f}s sampled across "
              f"{n_recorded} val ticks, val_interval={cfg.val_interval}) ===")
        for p in phases:
            t = sum(timing[p])
            avg = t / max(n_recorded, 1)
            pct = 100 * t / max(grand, 1e-9)
            print(f"  {p:15s}: {t:.3f}s total ({pct:5.1f}%) | {avg*1000:.1f}ms/gen")

    # Training-history plots — independent of any test set.
    plot_snes_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_log_val_fitness(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_sigma_history(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_loss_breakdown(history, cfg, cfg.save_plots, cfg.show_plots)
    plot_timing(history, cfg, cfg.save_plots, cfg.show_plots)

    # Per-model × per-dataset correlation / error / cos-sim plots.
    val_metrics, val_preds = best_val_model.score(val_data)
    _plot_eval_set(cfg, test_data, test_preds,  metrics,       "best_val_per_atom",     "best_val_total")
    _plot_eval_set(cfg, val_data,  val_preds,   val_metrics,   "best_val_val_per_atom", "best_val_val_total")
    _plot_eval_set(cfg, test_data, final_preds, final_metrics, "final_gen_per_atom",    "final_gen_total")

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

    _plot_eval_set(cfg, data, predictions, metrics,
                   "per_atom", "total",
                   save_dir=save_plots, show=show_plots)

    return metrics, predictions


def _count_xyz_frames(path: str) -> int:
    """Count frames in a trajectory file. Format-aware:
    XYZ/extXYZ walk atom-count headers (no parsing); .traj uses len(); anything
    else falls back to ase.io.iread (slower).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".traj":
        from ase.io.trajectory import Trajectory
        with Trajectory(path, "r") as traj:
            return len(traj)

    if ext in (".xyz", ".extxyz"):
        n_frames = 0
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    n_atoms = int(line.strip())
                except ValueError:
                    break
                for _ in range(n_atoms + 1):
                    if not f.readline():
                        return n_frames
                n_frames += 1
        return n_frames

    # Unknown extension — fall back to ASE's streaming iterator.
    from ase.io import iread
    return sum(1 for _ in iread(path, index=":"))


def process_trajectory(
    model: TNEP,
    trajectory_path: str,
    dt_fs: float = 1.0,
    save_plots: str | None = "plots",
    show_plots: bool = False,
    batch_size: int | None = None,
    pin_to_cpu: bool = True,
    descriptor_mode: int | None = None,
    descriptor_batch_frames: int | None = 1,
    descriptor_memory_budget_bytes: int | None = None,
    descriptor_precision: str | None = None,
    descriptor_pair_tile_size: int | None = None,
    ir_split_at_cm: float | None = None,
    ir_smooth: int = 10,
    ir_smooth_kind: str = "gaussian",
    ir_power_dc_cutoff_cm: float = 100.0,
    ir_transmittance_mode: str = "beer_lambert",
    ir_transmittance_scale: float = 1.0,
) -> dict:
    """Predict properties along an MD trajectory and compute spectra.

    Mode 1 (dipole): dipole trajectory → IR spectrum. Mode 2 (polarizability):
    polarizability trajectory → Raman spectrum. Frames are processed in batches
    so peak memory is O(batch_size), not O(all_frames).

    Args:
        model           : trained TNEP model (config via model.cfg)
        trajectory_path : path to .xyz trajectory file
        dt_fs           : MD timestep in femtoseconds
        save_plots      : directory to save plots into (default "plots")
        show_plots      : display plots interactively (default False)
        batch_size      : frames per inference batch; None = whole trajectory.
        pin_to_cpu      : place batch tensors on CPU. Required when a batch's
                          COO tensors exceed VRAM. Default True.
        descriptor_mode : override cfg.descriptor_mode (0 = quippy CPU,
                          1 = native TF/GPU). None uses model.cfg.
        descriptor_batch_frames : frames per descriptor TF graph call (mode 1).
                          1 = per-frame (lowest memory); >=2 batches for
                          throughput; None auto-sizes to the memory budget.
        descriptor_memory_budget_bytes : GPU budget for the auto-sizer when
                          descriptor_batch_frames is None (default 6 GiB).
        descriptor_precision : GPU kernel precision (mode 1): "float64"
                          (default, mirrors Fortran) or "float32" (~2×
                          throughput, ~½ VRAM). None uses cfg. Outputs are
                          always cast to float32 at the trajectory boundary.

    Returns:
        Mode 1: dict(dipoles, freq_cm, intensity, power, acf)
        Mode 2: dict(polarizabilities, freq_cm, I_VV, I_VH, I_total,
                     acf_iso, acf_aniso)
    """
    cfg = model.cfg

    if cfg.target_mode not in (1, 2):
        raise ValueError(f"Spectroscopy not supported for target_mode={cfg.target_mode} (PES). "
                         f"Use mode 1 (dipole) or mode 2 (polarizability).")

    if save_plots:
        os.makedirs(save_plots, exist_ok=True)
    stem = os.path.splitext(os.path.basename(trajectory_path))[0]

    # Fast frame count so the progress bar can show a total.
    n_total = _count_xyz_frames(trajectory_path)
    print(f"Loaded {n_total} frames from {trajectory_path}")
    total_batches = (((n_total + batch_size - 1) // batch_size)
                     if batch_size else 1)

    # One descriptor builder reused across all batches (quippy is expensive to
    # construct). Backend selected by descriptor_mode / cfg.descriptor_mode.
    builder = make_descriptor_builder(cfg, mode=descriptor_mode)
    # Precision override used by mode-1 builder only (quippy ignores it).
    _resolved_precision = (descriptor_precision
                           if descriptor_precision is not None
                           else getattr(cfg, "descriptor_precision", "float64"))
    print(f"Descriptor backend: {type(builder).__name__}  "
          f"(precision: {_resolved_precision})")

    # Stream frames in fixed-size batches (build → predict → drop). A 2-deep
    # prefetch queue on a background thread overlaps iread parsing (+ host-side
    # assign_type_indices) with GPU compute; memory stays ~3 × batch_size.
    import queue, threading

    _PREFETCH_DEPTH = 2
    q: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)
    _SENTINEL = object()
    _producer_err: list = []

    def _producer():
        try:
            buf = []
            for frame in iread(trajectory_path, index=":"):
                buf.append(frame)
                if batch_size is not None and len(buf) == batch_size:
                    q.put((buf, assign_type_indices(buf, cfg.types)))
                    buf = []
            if buf:
                q.put((buf, assign_type_indices(buf, cfg.types)))
        except Exception as e:
            _producer_err.append(e)
        finally:
            q.put(_SENTINEL)

    prod_thread = threading.Thread(target=_producer, name="traj-prefetch", daemon=True)
    prod_thread.start()

    result_batches = []
    n_frames = 0
    pbar = tqdm(total=total_batches, desc="Trajectory batches", unit="batch")
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            batch_frames, batch_types = item
            result_batches.append(
                predict_trajectory_batch(model, builder, batch_frames, batch_types,
                                         pin_to_cpu=pin_to_cpu,
                                         descriptor_batch_frames=descriptor_batch_frames,
                                         descriptor_memory_budget_bytes=descriptor_memory_budget_bytes,
                                         descriptor_precision=_resolved_precision,
                                         descriptor_pair_tile_size=descriptor_pair_tile_size))
            n_frames += len(batch_frames)
            pbar.update(1)
            del batch_frames, batch_types, item
    finally:
        pbar.close()
        prod_thread.join()
    if _producer_err:
        raise _producer_err[0]

    print(f"Processed {n_frames} frames from {trajectory_path}")
    results = np.concatenate(result_batches, axis=0)
    del result_batches

    # Outputs always saved (.npy + .txt) so long MD inference isn't lost when
    # plotting is off. Default location: save_plots, else next to the trajectory.
    if save_plots:
        out_dir = save_plots
    else:
        out_dir = os.path.dirname(trajectory_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if cfg.target_mode == 1:
        dipoles = results
        npy_path = os.path.join(out_dir, f"{stem}_dipoles.npy")
        txt_path = os.path.join(out_dir, f"{stem}_dipoles.txt")
        np.save(npy_path, dipoles)
        np.savetxt(txt_path, dipoles, fmt="%.8e",
                   header="dipole_x  dipole_y  dipole_z  (e*Angstrom)")
        print(f"Dipoles saved to {npy_path} (binary) and {txt_path} (text)")
        freq_cm, intensity, power, acf = compute_ir_spectrum(
            dipoles, dt_fs=dt_fs,
            smooth_k=ir_smooth, smooth_kind=ir_smooth_kind,
            power_dc_cutoff_cm=ir_power_dc_cutoff_cm)
        # Plot label <trajectory>_<model> so runs don't overwrite each other.
        model_label = os.path.splitext(os.path.basename(
            getattr(cfg, "save_path", "") or "model"))[0]
        plot_ir_spectrum(freq_cm, intensity, cfg, save_plots, show_plots,
                         trajectory_path=trajectory_path,
                         model_label=model_label,
                         split_at_cm=ir_split_at_cm,
                         transmittance_mode=ir_transmittance_mode,
                         transmittance_scale=ir_transmittance_scale)
        plot_power_spectrum(freq_cm, power, cfg, save_plots, show_plots,
                            trajectory_path=trajectory_path,
                            model_label=model_label,
                            low_cm_cutoff=ir_power_dc_cutoff_cm)
        return {"dipoles": dipoles, "freq_cm": freq_cm, "intensity": intensity,
                "power": power, "acf": acf}

    else:
        pols = results
        npy_path = os.path.join(out_dir, f"{stem}_polarizabilities.npy")
        txt_path = os.path.join(out_dir, f"{stem}_polarizabilities.txt")
        np.save(npy_path, pols)
        np.savetxt(txt_path, pols, fmt="%.8e",
                   header="alpha_xx  alpha_yy  alpha_zz  alpha_xy  alpha_yz  alpha_zx")
        print(f"Polarizabilities saved to {npy_path} (binary) and {txt_path} (text)")
        freq_cm, I_VV, I_VH, I_total, acf_iso, acf_aniso = compute_raman_spectrum(
            pols, dt_fs=dt_fs)
        plot_raman_spectrum(freq_cm, I_VV, I_VH, I_total, cfg, save_plots, show_plots)
        return {"polarizabilities": pols, "freq_cm": freq_cm,
                "I_VV": I_VV, "I_VH": I_VH, "I_total": I_total,
                "acf_iso": acf_iso, "acf_aniso": acf_aniso}

def filter_dataset_by_species(input_xyz: str,
                                output_xyz: str,
                                allowed_species: list[int | str],
                                mode: str = "subset") -> int:
    """Filter an .xyz dataset to structures whose species satisfy `allowed_species`.

    Args:
        input_xyz       : path to input .xyz file
        output_xyz      : path to write the filtered .xyz
        allowed_species : list of atomic numbers (e.g. [6, 1, 8]) or
                          chemical symbols (e.g. ["C", "H", "O"])
        mode            : "subset" — keep structures whose species are
                                     a subset of `allowed_species` (default).
                          "exact"  — keep structures containing exactly
                                     the same set as `allowed_species`.

    Returns:
        n_kept : int — number of structures written to `output_xyz`.
    """
    from ase.data import atomic_numbers, chemical_symbols
    allowed_Z = set(atomic_numbers[z] if isinstance(z, str) else int(z)
                    for z in allowed_species)
    allowed_str = ", ".join(f"{chemical_symbols[z]}(Z={z})" for z in sorted(allowed_Z))

    dataset = read(input_xyz, index=":")
    n_in = len(dataset)
    kept = []
    seen_outside = set()
    for s in dataset:
        species = set(int(z) for z in s.numbers)
        if mode == "exact":
            keep = species == allowed_Z
        else:
            keep = species.issubset(allowed_Z)
        if keep:
            kept.append(s)
        else:
            seen_outside.update(species - allowed_Z)

    n_out = len(kept)
    print(f"Filter ({mode}) → {{ {allowed_str} }}")
    print(f"  read    {n_in:6d} structures from {input_xyz}")
    print(f"  kept    {n_out:6d}")
    print(f"  dropped {n_in - n_out:6d}", end="")
    if seen_outside:
        sym = ", ".join(f"{chemical_symbols[z]}(Z={z})" for z in sorted(seen_outside))
        print(f"  (offending species: {sym})")
    else:
        print()

    if n_out == 0:
        raise ValueError(
            f"No structures match filter — refusing to write empty {output_xyz}.")

    os.makedirs(os.path.dirname(output_xyz) or ".", exist_ok=True)
    write(output_xyz, kept)
    print(f"  wrote   {n_out:6d} structures → {output_xyz}")
    return n_out


def dump_dipole_predictions(model_path: str,
                             test_xyz: str,
                             out_path: str = "datasets/dipole_test.out") -> None:
    """Score `model_path` on `test_xyz` and write per-atom dipoles to disk.

    Output columns: pred_xyz | ref_xyz | N_atoms. Structures with species
    outside the model's cfg.types are dropped (with a summary line).
    """
    model = load_model(model_path)
    cfg = model.cfg

    dataset = read(test_xyz, index=":")
    print(f"Loaded {len(dataset)} structures from {test_xyz}")

    # Drop frames containing species the model doesn't know.
    known_Z = set(int(z) for z in cfg.types)
    kept = [s for s in dataset
            if set(int(z) for z in s.numbers).issubset(known_Z)]
    dropped = len(dataset) - len(kept)
    if dropped:
        from ase.data import chemical_symbols
        offending = set()
        for s in dataset:
            extra = set(int(z) for z in s.numbers) - known_Z
            offending.update(extra)
        sym = ", ".join(f"{chemical_symbols[z]}(Z={z})" for z in sorted(offending))
        print(f"  Dropped {dropped}/{len(dataset)} structures containing species "
              f"outside model types {sorted(known_Z)}: {sym}")
    if not kept:
        raise ValueError(
            f"No structures in {test_xyz} match model species "
            f"{sorted(known_Z)} — all {dropped} were dropped.")
    dataset = kept

    data = prepare_eval_data(dataset, cfg)
    metrics, preds = model.score(data)            # [S, 3] per-atom dipoles
    print_score_summary(metrics, cfg, prefix=f"Test ({test_xyz})")

    preds_np  = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
    refs_np   = (data["targets"].numpy() if hasattr(data["targets"], "numpy")
                 else np.asarray(data["targets"]))
    natoms_np = (data["num_atoms"].numpy() if hasattr(data["num_atoms"], "numpy")
                 else np.asarray(data["num_atoms"]))

    table = np.column_stack([preds_np, refs_np, natoms_np.astype(np.int32)])
    header = ("pred_x pred_y pred_z   ref_x ref_y ref_z   N_atoms  "
              f"(units: {model.cfg.dipole_units if not getattr(cfg, 'convert_dipole_to_eangstrom', True) else 'e*angstrom'}, per-atom)")
    fmt = ["%.8e"] * 6 + ["%d"]
    np.savetxt(out_path, table, fmt=fmt, header=header)
    print(f"Wrote {out_path}  ({len(dataset)} rows)")


if __name__ == '__main__':
    model = train_model()
    #model = load_model("models/n30_q75_pop100_20260530_181726_water_bulk_dipole_NEW/train_waterbulk_O_H_dipole_best_val.h5")
    #dipoles = process_trajectory(model, "plots/BulkWater/GPUMD traj/water_bulk_traj.xyz", dt_fs = 1.0, batch_size=40, descriptor_mode=1, descriptor_batch_frames=40, pin_to_cpu=False, descriptor_precision="float32", descriptor_pair_tile_size=2000)
    #dump_dipole_predictions(model_path="models/n50_q165_pop100_20260513_161930_CHO_best_r2/train_C_O_H_dipole_best_val.h5", test_xyz="datasets/test.xyz", out_path="datasets/dipole_test_tnep.out")
    #filter_dataset_by_species(input_xyz="datasets/test.xyz", output_xyz="datasets/cho_filter_test.xyz", allowed_species=[6, 1, 8])
    #ir_spectrum_from_file(dipole_path="plots/nve_dipoles.txt", dt_fs = 1, save_dir=None, acf_ratio=0.1, smooth_k=10, quantum_correction="quadratic")
    #model.cfg.plot_units = "e*angstrom"
    #model.score_from_file("datasets/test_waterbulk.xyz", plot=True, presentation=True, shared_axis_scale=True)
    """
    from spectroscopy import plot_ir_overlay

    plot_ir_overlay(dipoles={
        "Adapted TNEP": ["plots/BulkWater/GPUMD traj/water_bulk_traj_dipoles.txt",
                                1.0],
        "TNEP (Xu et al)": ["plots/BulkWater/GPUMD traj/dipole_gpumd_out.out",
                            1.0]
    },
    spectra= {
        "Experimental (Max et al)": ["max_water_ir.csv", "wavenumber_cm1", "k_H2O"]
    },
    temperature=298.15,
    quantum_correction="harmonic",
    window_cm=(100, 4000),
    smooth_k=20,
    presentation=True
    )
    """