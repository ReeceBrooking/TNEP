from __future__ import annotations

import numpy as np
import os
import shutil
import tempfile

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
from ase.io import read, iread


def _resolve_scratch_dir(cfg: TNEPconfig) -> str:
    """Pick a node-local fast-storage directory for the grad_values
    cache. **Slurm env-vars are only consulted when
    `cfg.csc_enable=True`** — outside CSC mode this method always
    returns the current working directory, so a local dev environment
    that happens to set `$TMPDIR` doesn't have its scratch redirected.

    With `cfg.csc_enable=True` the resolver tries (in priority order):

        SLURM_TMPDIR     (Slurm node-local scratch)
        TMPDIR           (POSIX standard; Mahti / Puhti set this)
        LOCAL_SCRATCH    (some sites; e.g. PBS environments)

    falling back to cwd when none are set. On HPC compute nodes cwd
    is usually a network filesystem (Lustre/GPFS) which is much
    slower than node-local NVMe — so the env-vars matter there.
    """
    if getattr(cfg, "csc_enable", False):
        for var in ("SLURM_TMPDIR", "TMPDIR", "LOCAL_SCRATCH"):
            path = os.environ.get(var)
            if path and os.path.isdir(path):
                return path
    return os.getcwd()


def _apply_csc_overrides(cfg: TNEPconfig) -> None:
    """When `cfg.csc_enable=True`, force every gradient-caching
    option off. Grad_values then stays in host RAM as a normal
    tf.constant; the chunk-staging path uses the in-RAM passthrough
    branch with no NVMe scratch, no pinned pool, no cuFile.
    """
    if not getattr(cfg, "csc_enable", False):
        return
    overrides = {
        "cache_gradients_to_disk": False,
        "chunk_prefetch": False,
        "use_pinned_buffers": False,
        "use_cufile": False,
    }
    changed = []
    for k, v in overrides.items():
        if getattr(cfg, k, None) != v:
            changed.append(f"{k}={getattr(cfg, k, None)!r}→{v!r}")
            setattr(cfg, k, v)
    print(f"  csc_enable=True — caching options disabled"
          + (f" ({', '.join(changed)})" if changed else ""))


def train_model(cfg: TNEPconfig | None = None,
                checkpoint: str | None = None,
                extract_model: bool = False,
                cfg_overrides: dict | None = None) -> TNEP:
    """Run full TNEP training pipeline: load, split, train, test, plot, save.

    Args:
        cfg          : TNEPconfig or None (uses defaults). Ignored when
                       `checkpoint` is set — the checkpoint embeds the
                       full cfg used for the original run, including the
                       train/val split (cfg.indices) and architecture
                       fields. Continuing with a different cfg would
                       break determinism or shape-match.
        checkpoint   : optional path to a `checkpoint.h5` written by a
                       previous run via `cfg.checkpoint_interval`. When
                       provided, the cfg is loaded from the file and
                       training resumes from `last_gen + 1`. Default
                       None starts a fresh run.
        extract_model: when True, skip the SNES training loop entirely
                       and treat the checkpoint as if it were the final
                       generation — build final_model and best_val_model
                       directly from the checkpoint's μ and best_μ,
                       then run scoring / saving / plotting exactly as
                       a completed training run would. Requires
                       `checkpoint` to be set (raises otherwise).
                       Default False.
        cfg_overrides: optional dict of cfg field → value to apply
                       after `load_checkpoint`, BEFORE the model is
                       built. Use this to repair old checkpoints whose
                       saved JSON is missing fields (e.g. legacy
                       checkpoints saved before the
                       `_serialize_config` fix, where class-default
                       fields like `num_neurons` and
                       `descriptor_mixing_arch` were not persisted).
                       Without overrides, the current class defaults
                       are used — which may not match the checkpoint's
                       architecture, producing a μ-shape mismatch.

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
            # Validate keys against the TNEPconfig annotations. A
            # typo would otherwise create a brand-new instance
            # attribute via setattr() and the user's intended
            # override would silently never apply — visible only at
            # the eventual dim-mismatch crash. Hard fail with a list
            # of the known fields to make the typo obvious.
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

    # CSC / Slurm-supercomputer mode: force every cache off before any
    # downstream code reads those flags. Must run before scratch-dir
    # resolution and pad_and_stack, both of which branch on
    # cfg.cache_gradients_to_disk.
    _apply_csc_overrides(cfg)

    # Allocate a per-run scratch directory for the disk-backed
    # grad_values cache. Prefers node-local fast storage on HPC nodes
    # when csc_enable=True (see _resolve_scratch_dir). Removed in the
    # finally block at the end of the function.
    cfg._gradient_cache_path = None
    if getattr(cfg, "cache_gradients_to_disk", False):
        scratch_root = _resolve_scratch_dir(cfg)
        cfg._gradient_cache_path = tempfile.mkdtemp(
            prefix=".grad_cache_", dir=scratch_root)
        print(f"  cache_gradients_to_disk=True → scratch dir "
              f"{cfg._gradient_cache_path}")

    try:
        return _train_model_inner(cfg, resume_state=resume_state,
                                  extract_model=extract_model)
    finally:
        # Always remove the scratch directory, even on exceptions, so
        # repeated runs don't leak ~10 GB per attempt onto the disk.
        cache_dir = getattr(cfg, "_gradient_cache_path", None)
        if cache_dir is not None and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  cleaned up gradient scratch dir {cache_dir}")
            cfg._gradient_cache_path = None


def _plot_eval_set(cfg: TNEPconfig, data: dict, preds, metrics: dict,
                   suffix_per_atom: str, suffix_total: str,
                   save_dir: str | None = None,
                   show: bool | None = None) -> None:
    """Emit the standard correlation / cos-sim / error-vs-magnitude
    plots for one (data, predictions, metrics) triple. Always emits
    the per-atom variant; emits the total variant (per-atom × num_atoms)
    only when target-scaling produced `total_*` keys in the metrics
    dict (i.e. dipole/polarizability with cfg.scale_targets=True).
    Cosine similarity is omitted from the total plots because it's
    scale-invariant and already shown at per-atom scale.

    `save_dir` and `show` override cfg.save_plots / cfg.show_plots
    when non-None — used by the periodic-plot callback (per-gen
    subfolder) and by test_model (custom destination per call).
    """
    targets = data["targets"].numpy()
    preds_np = preds.numpy()
    save = save_dir if save_dir is not None else cfg.save_plots
    show = show if show is not None else cfg.show_plots

    # RRMSE is computed once from total-scale targets / total-scale RMSE
    # and shown identically on both per-atom and total plots — RRMSE is
    # a model-vs-dataset property that shouldn't depend on which scale
    # the correlation panel happens to be drawn in.
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
    target_abs_mean = max(float(np.mean(np.abs(total_targets))), 1e-12)
    target_abs_mean_comp = np.maximum(np.mean(np.abs(total_targets), axis=0), 1e-12)
    rmse_comp_total = np.sqrt(np.mean(total_diff ** 2, axis=0))
    rrmse_payload = {
        "rrmse": total_rmse_scalar / target_abs_mean,
        "rrmse_components": rmse_comp_total / target_abs_mean_comp,
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
    # Carry forward cos-sim annotations for plot_correlation; the
    # standalone plot_cosine_similarity is intentionally omitted at
    # total scale because cosine similarity is scale-invariant.
    if "cos_sim_all" in metrics:
        total_metrics["cos_sim_mean"] = metrics["cos_sim_mean"]
        total_metrics["cos_sim_all"] = metrics["cos_sim_all"]
    plot_correlation(total_targets, total_preds, total_metrics, cfg, save, show, suffix=suffix_total)
    plot_error_vs_magnitude(total_targets, total_preds, cfg, save, show, suffix=suffix_total)


def _setup_grad_staging(cfg: TNEPconfig, train_data: dict, val_data: dict) -> None:
    """Pre-stage per-chunk pair indices, then pick the chunk-staging
    mode for each data dict based on `cfg.pin_data_to_cpu`:

      pin_data_to_cpu=False : everything lives on /GPU:0. If grad
        was disk-backed, read it fully into a GPU tf.constant and
        drop the memmap. Move every other static field to GPU too.
        The SNES eval / score loops then use the pure-GPU
        `_stage_chunk_resident` fast path.

      pin_data_to_cpu=True  : tensors stay on host. For disk-backed
        grad_values, attach pinned-host + cuFile pools so per-chunk
        slices DMA straight to GPU. For in-RAM grad_values the
        passthrough branch in `_stage_finalize_tf` handles slicing.

    With cfg.eval_jit_compile, pair indices are padded to the
    per-data global max so the XLA-compiled eval kernel sees a
    single shape and compiles once.
    """
    from data import (prestage_chunk_indices, compute_max_chunk_pairs,
                      move_data_to_gpu, make_pinned_pool_for)

    chunk = cfg.batch_chunk_size
    pin = bool(getattr(cfg, "pin_data_to_cpu", True))

    # Pre-stage pair indices for every chunk range. Tiny tensors,
    # GPU-resident, reused every generation.
    for d in (train_data, val_data):
        S = int(d["num_atoms"].shape[0])
        c = chunk if chunk is not None else S
        ranges = [(s, min(s + c, S)) for s in range(0, S, c)]
        pad_to = (compute_max_chunk_pairs(d, ranges)
                  if getattr(cfg, "eval_jit_compile", False) else None)
        prestage_chunk_indices(d, ranges, pad_to=pad_to)

    # GPU-resident path. Loads disk-backed grad into a GPU tf.constant
    # (if applicable), moves all other fields on-device, then sets the
    # `_gv_resident_gpu` flag that triggers `_stage_chunk_resident` in
    # `prefetched_chunks`.
    if not pin:
        for d, tag in ((train_data, "train"), (val_data, "val")):
            gv = d["grad_values"]
            if d.get("_gv_disk_backed", False):
                shape = tuple(int(x) for x in gv.shape)
                nbytes = int(np.prod(shape)) * int(np.dtype(gv.dtype).itemsize)
                with tf.device("/GPU:0"):
                    gv_gpu = tf.constant(np.asarray(gv))
                d["grad_values"] = gv_gpu
                d["_gv_disk_backed"] = False
                print(f"  GPU-resident: {tag} grad_values loaded from disk "
                      f"{shape} {gv.dtype} = {nbytes/1e9:.2f} GB on /GPU:0")
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)
        return

    # Pinned / cuFile pool path. Only meaningful when grad is disk-backed.
    n_buffers = max(int(getattr(cfg, "pinned_pool_size", 4)),
                    int(getattr(cfg, "prefetch_depth", 1)) + 1)
    if getattr(cfg, "use_pinned_buffers", True):
        for d, tag in ((train_data, "train"), (val_data, "val")):
            if not d.get("_gv_disk_backed", False):
                continue
            pool = make_pinned_pool_for(d, batch_chunk_size=chunk,
                                         n_buffers=n_buffers)
            if pool is not None:
                d["_pinned_pool"] = pool
                print(f"  pinned-buffer pool ({len(pool._all)} × "
                      f"{pool.buffer_nbytes/1e6:.0f} MB) attached to {tag}_data")

    if getattr(cfg, "use_cufile", True):
        try:
            from cufile_io import (cuFile_available, CuFileHandle,
                                   make_cufile_pool_for)
        except Exception as e:
            print(f"  cuFile import failed ({e}) — falling back to pinned path")
            return
        if not cuFile_available():
            return
        n_cf = max(int(getattr(cfg, "cufile_pool_size", 4)),
                   int(getattr(cfg, "prefetch_depth", 1)) + 1)
        for d, tag in ((train_data, "train"), (val_data, "val")):
            if not d.get("_gv_disk_backed", False):
                continue
            gv = d.get("grad_values")
            if not hasattr(gv, "filename"):
                continue
            pool = make_cufile_pool_for(d, batch_chunk_size=chunk,
                                         n_buffers=n_cf)
            if pool is None:
                continue
            try:
                handle = CuFileHandle(gv.filename)
            except Exception as e:
                print(f"  cuFile open failed for {tag}: {e}")
                continue
            d["_cufile_ctx"] = {"handle": handle, "pool": pool}
            print(f"  cuFile pool ({len(pool._all)} × "
                  f"{pool.nbytes/1e6:.0f} MB) attached to {tag}_data")


def _train_model_inner(cfg: TNEPconfig,
                        resume_state: dict | None = None,
                        extract_model: bool = False) -> TNEP:
    """Body of train_model, factored out so the outer try/finally can
    guarantee cleanup of the disk-backed gradient scratch directory.

    On resume (`resume_state` provided), the train/val split is taken
    from the checkpoint's `cfg.indices` rather than re-shuffled, and
    the run directory setup is skipped so outputs land in the existing
    model dir.

    When `extract_model=True` (only valid with `resume_state`), the
    SNES training loop is skipped entirely. `final_model` and
    `best_val_model` are reconstructed from the checkpoint's μ and
    best_μ, and the existing history (loaded from the checkpoint) is
    passed through to the post-fit scoring / saving / plotting path
    unchanged. The function then returns as if training had just
    finished naturally.
    """
    # Load dataset, filter by species, then filter bad data
    dataset, dataset_types_int = collect(cfg)
    cfg.type_map = {z: idx for idx, z in enumerate(cfg.types)}

    if cfg.target_mode == 1:
        print_dipole_statistics(dataset, cfg, target_key=_resolve_target_key(cfg))
    elif cfg.target_mode == 2:
        print_polarizability_statistics(dataset, target_key=_resolve_target_key(cfg))

    if resume_state is not None and isinstance(getattr(cfg, "indices", None), np.ndarray):
        # Indices already restored from checkpoint — would re-shuffle to
        # the same values anyway (deterministic via cfg.seed), but
        # skipping makes the resume self-explanatory and avoids any
        # surprise if the dataset was extended between runs.
        print(f"  resume: using checkpoint train/val split "
              f"({len(cfg.indices)} indices)")
    else:
        cfg.randomise(dataset)

    # Split into train/val (built now) and a deferred test placeholder. The
    # test descriptors are built on first scoring (materialize_test_data),
    # not before training — saves time when the user aborts mid-run and
    # avoids a large test-set descriptor build delaying generation 0.
    train_data, test_pending, val_data = split(dataset, dataset_types_int, cfg)

    # Resolve dim_q from cfg before any consumer (q_scaler, pad_and_stack)
    # needs it. Cross-checked against built descriptor shape further down.
    from DescriptorBuilderGPU import compute_dim_q
    cfg.dim_q = compute_dim_q(cfg)

    # Per-channel descriptor scaling. Computed ONCE over the training-
    # set per-atom descriptors (before padding) and applied identically
    # to train/val/test/trajectory inputs so the scaler is a frozen
    # property of the trained model, persisted in /weights/q_scaler.
    # On resume, cfg._q_scaler is restored by load_checkpoint BEFORE
    # this point, so the guard below keeps it intact.
    if str(getattr(cfg, "descriptor_scaling", "none")) == "q_scaler":
        if getattr(cfg, "_q_scaler", None) is None:
            granularity = str(getattr(
                cfg, "q_scaler_granularity", "per_component")).lower()
            n_atoms_total = sum(
                int(d.shape[0]) for d in train_data["descriptors"])
            if granularity == "per_component":
                from data import _compute_q_scaler
                cfg._q_scaler = _compute_q_scaler(
                    train_data["descriptors"], cfg.dim_q)
            elif granularity == "l_block":
                from data import _compute_q_scaler_l_block
                from DescriptorBuilderGPU import descriptor_block_layout
                layout = descriptor_block_layout(cfg)
                cfg._q_scaler = _compute_q_scaler_l_block(
                    train_data["descriptors"], layout)
            else:
                raise ValueError(
                    f"cfg.q_scaler_granularity={granularity!r} not "
                    "recognised (expected 'per_component' or 'l_block').")
            qs = cfg._q_scaler
            n_unique = int(np.unique(qs).size)
            print(f"  Computed q_scaler ({granularity}) over "
                  f"{n_atoms_total} training atoms: {qs.size} q-channels, "
                  f"{n_unique} unique multipliers:")
            print(f"    multiplier distribution: "
                  f"min={qs.min():.4f}  max={qs.max():.4f}  "
                  f"mean={qs.mean():.4f}  std={qs.std():.4f}")
            mid = qs.size // 2
            print(f"    sample channels: s[0]={qs[0]:.4f}  "
                  f"s[{mid}]={qs[mid]:.4f}  s[{qs.size - 1}]={qs[-1]:.4f}")
        else:
            print(f"  Reusing q_scaler from checkpoint (shape="
                  f"{cfg._q_scaler.shape}, no recompute on resume).")
    elif str(getattr(cfg, "descriptor_scaling", "none")) != "none":
        raise ValueError(
            f"cfg.descriptor_scaling={cfg.descriptor_scaling!r} "
            "not recognised (expected 'none' or 'q_scaler').")

    # Convert to padded dense tensors for GPU-batched evaluation. test_data
    # is intentionally NOT padded here; it gets padded by materialize_test_data
    # the first time it's actually consumed.
    train_data = pad_and_stack(
        train_data, num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        gradient_cache_path=getattr(cfg, "_gradient_cache_path", None),
        cache_tag="train",
        q_scaler=getattr(cfg, "_q_scaler", None))
    val_data   = pad_and_stack(
        val_data,   num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        gradient_cache_path=getattr(cfg, "_gradient_cache_path", None),
        cache_tag="val",
        q_scaler=getattr(cfg, "_q_scaler", None))

    _setup_grad_staging(cfg, train_data, val_data)

    # Lazy-build helper. First call performs descriptor build + pad_and_stack;
    # subsequent calls return the cached dict (idempotent on test_pending).
    def get_test_data():
        return materialize_test_data(test_pending, cfg,
                                     num_types=cfg.num_types,
                                     pin_to_cpu=cfg.pin_data_to_cpu)

    # Cross-check: built descriptor shape must match cfg.dim_q resolved
    # above. Catches cfg / builder drift before the SNES loop starts.
    built_dim_q = int(train_data["descriptors"][0].shape[-1])
    if built_dim_q != cfg.dim_q:
        raise RuntimeError(
            f"compute_dim_q={cfg.dim_q} disagrees with built descriptor "
            f"shape {built_dim_q}; cfg / builder mismatch.")
    print("Dimension of q: " + str(cfg.dim_q))

    # Set up run directory: models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/
    # On resume, save_path is already an existing run dir from the
    # checkpoint — skip directory creation so outputs continue to land
    # there and the original timestamped name is preserved.
    if cfg.save_path is not None and resume_state is None:
        setup_run_directory(cfg)
    elif resume_state is not None and cfg.save_path is not None:
        run_dir = os.path.dirname(cfg.save_path) or "."
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        cfg.save_plots = os.path.join(run_dir, "plots")
        print(f"  resume: writing outputs to existing run dir {run_dir}")

    model = TNEP(cfg)
    print(f"Model Parameters: {model.optimizer.dim}  |  Population Size: {model.optimizer.pop_size}")
    print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
    print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))

    def periodic_plot_callback(history, gen):
        """Called during training at plot_interval to show progress."""
        print(f"\n--- Periodic plots at generation {gen} ---")
        # First periodic plot triggers the deferred test descriptor build;
        # subsequent calls return the cached padded dict instantly.
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

    # Train — unless extract_model=True, in which case rebuild
    # `final_model` and `best_val_model` straight from the
    # checkpoint's μ / best_μ and reuse its history dict. The
    # post-fit scoring + plotting code below runs unchanged.
    if extract_model:
        from SNES import _set_model_params
        if resume_state is None:
            # Outer train_model already enforced this, but guard the
            # internal contract too so a future caller of
            # _train_model_inner can't trip the same wire silently.
            raise ValueError(
                "_train_model_inner: extract_model=True requires "
                "resume_state (μ / best_μ must come from a checkpoint).")
        print(f"  extract_model: skipping SNES training loop; rebuilding "
              f"models from checkpoint state at gen "
              f"{resume_state['last_gen'] + 1}.")
        snes = model.optimizer
        ckpt_dim = int(np.asarray(resume_state["mu"]).size)
        if ckpt_dim != int(snes.dim):
            # Diagnose the most common cause: the saved cfg JSON is
            # missing fields (legacy serializer bug or new fields
            # introduced since save), so the model rebuilt from cfg
            # has a different architecture than what produced the
            # stored μ. Tell the user exactly what to do.
            raise ValueError(
                f"Checkpoint μ has dim {ckpt_dim} but the current model "
                f"builds dim={snes.dim}. The cfg loaded from the "
                f"checkpoint must disagree with the cfg used at save "
                f"time on at least one architectural field "
                f"(num_neurons, descriptor_mixing, "
                f"descriptor_mixing_arch, descriptor_mixing_per_type, "
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

        # Mirror SNES.fit's final restoration: keep the in-place
        # `model` (and its optimizer) aligned with the best-val state
        # so any downstream caller that re-uses `model` directly sees
        # the best run-end configuration, matching post-fit behaviour.
        snes.mu.assign(best_mu)
        snes.sigma.assign(tf.constant(
            resume_state["best_sigma"], dtype=tf.float32))
        _set_model_params(model, *best_val_params)

        history = resume_state["history"]
    else:
        history, final_model, best_val_model = model.fit(
            train_data, val_data,
            plot_callback=periodic_plot_callback if cfg.plot_interval else None,
            resume_state=resume_state)

    # Build the test descriptors now (if not already built by a periodic
    # plot during training), then score final + best-val on it.
    test_data = get_test_data()

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
        # History is sampled once per val_interval, so n_recorded is the
        # number of val ticks — not the total generation count. The
        # per-tick averages are still representative of typical per-gen
        # cost since each recorded tick is itself a single gen's timing.
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

    # Per-model × per-dataset correlation / error / cos-sim plots. Each
    # combination produces a per-atom plot plus a "total" plot (per-atom
    # values × num_atoms) when scale_targets is active.
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
    """Count frames in an XYZ trajectory by walking only the atom-count headers.

    Each frame is: <N> line, comment line, then N atom lines. We read the N
    integer and skip N+1 lines per frame — no parsing, just counting.
    """
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
) -> dict:
    """Predict properties along an MD trajectory and compute spectra.

    For dipole models (mode 1): predicts dipole trajectory and computes IR spectrum.
    For polarizability models (mode 2): predicts polarizability trajectory and
    computes Raman spectrum.

    Frames are processed in batches: descriptors are built, inferred, and discarded
    per batch so peak memory is O(batch_size) not O(all_frames).

    Args:
        model           : trained TNEP model (config accessed via model.cfg)
        trajectory_path : str — path to .xyz trajectory file
        dt_fs           : float — MD timestep in femtoseconds
        save_plots      : str or None — directory to save plot into (default "plots")
        show_plots      : bool — True to display plot interactively (default False)
        batch_size      : int or None — frames per inference batch; None processes
                          the whole trajectory as one batch.
        pin_to_cpu      : bool — place batch tensors on CPU instead of GPU.
                          Required when a single batch's COO tensors exceed VRAM
                          (large systems × large batch_size). Default True.
        descriptor_mode : int or None — overrides cfg.descriptor_mode for this
                          run only. 0 = quippy (CPU), 1 = native TF/GPU.
                          None = use the value baked into model.cfg.
        descriptor_batch_frames : int or None — frames per descriptor-builder
                          TF graph call (mode 1 only). 1 = per-frame (default,
                          lowest memory). int >= 2 = multi-frame batching for
                          throughput. None = auto-size to
                          descriptor_memory_budget_bytes (default 6 GiB).
                          Quippy mode ignores this field.
        descriptor_memory_budget_bytes : int or None — GPU memory budget
                          (bytes) used by the auto-sizer when
                          descriptor_batch_frames is None. None falls back to
                          the builder's default (6 GiB). Quippy mode and
                          explicit-int batch sizes ignore this field.
        descriptor_precision : str or None — internal compute precision for
                          the GPU descriptor kernels (mode 1 only):
                          "float64" (default, mirrors Fortran reference),
                          "float32" (~2× throughput, ~½ VRAM, slight loss of
                          agreement vs quippy). None falls back to
                          cfg.descriptor_precision. Outputs are always cast
                          to float32 at the trajectory boundary regardless.

    Returns:
        For mode 1 (dipole):
            dict with keys: dipoles, freq_cm, intensity, power, acf
        For mode 2 (polarizability):
            dict with keys: polarizabilities, freq_cm, I_VV, I_VH, I_total,
                            acf_iso, acf_aniso
    """
    cfg = model.cfg

    if cfg.target_mode not in (1, 2):
        raise ValueError(f"Spectroscopy not supported for target_mode={cfg.target_mode} (PES). "
                         f"Use mode 1 (dipole) or mode 2 (polarizability).")

    if save_plots:
        os.makedirs(save_plots, exist_ok=True)
    stem = os.path.splitext(os.path.basename(trajectory_path))[0]

    # Fast frame count (line-skip, no parsing) so the progress bar can show a total.
    n_total = _count_xyz_frames(trajectory_path)
    print(f"Loaded {n_total} frames from {trajectory_path}")
    total_batches = (((n_total + batch_size - 1) // batch_size)
                     if batch_size else 1)

    # One descriptor builder reused across all batches — quippy descriptors are
    # expensive to construct, so we build once. The backend is selected by
    # cfg.descriptor_mode (0 = quippy, 1 = native TF/GPU); the per-call
    # `descriptor_mode` argument above overrides for this trajectory only.
    builder = make_descriptor_builder(cfg, mode=descriptor_mode)
    # Trajectory-time precision override: used by mode-1 builder only. Quippy
    # backend simply ignores the kwarg via its existing build_descriptors_flat
    # signature (memory_budget_bytes / precision are no-ops there).
    _resolved_precision = (descriptor_precision
                           if descriptor_precision is not None
                           else getattr(cfg, "descriptor_precision", "float64"))
    print(f"Descriptor backend: {type(builder).__name__}  "
          f"(precision: {_resolved_precision})")

    # Stream frames in fixed-size batches: build → pack → predict → append → drop.
    # Only batch_size ASE Atoms exist in memory at any moment.
    #
    # Phase 6: a 2-deep prefetch ring buffer on a background thread overlaps
    # ase.io.iread parsing with GPU compute. The producer also pre-runs
    # assign_type_indices on the host so the consumer thread (which holds the
    # GPU) doesn't pay that cost. _PREFETCH_DEPTH=2 keeps memory bounded to
    # ~3 × batch_size ASE Atoms (current GPU batch + queued + producer's
    # half-built batch).
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

    # Resolve where to write trajectory outputs. dipoles / polarizabilities
    # are always saved (binary .npy + human-readable .txt) so a long-running
    # MD inference is never lost just because plotting was disabled. Default
    # location: save_plots dir if set, else next to the trajectory file.
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
        freq_cm, intensity, power, acf = compute_ir_spectrum(dipoles, dt_fs=dt_fs)
        plot_ir_spectrum(freq_cm, intensity, cfg, save_plots, show_plots)
        plot_power_spectrum(freq_cm, power, cfg, save_plots, show_plots)
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


if __name__ == '__main__':
    model = train_model()
    #model = load_model("models/n30_q75_pop80_20260428_120216/train_waterbulk_O_H_dipole_best_val.npz")
    #dipoles = process_trajectory(model, "datasets/water_bulk_traj.xyz", batch_size=20, descriptor_mode=1, descriptor_batch_frames=5, pin_to_cpu=False, descriptor_precision="float64")
    