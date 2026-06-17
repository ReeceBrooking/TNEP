from __future__ import annotations

import numpy as np
import os
import shutil
import signal
import tempfile

_slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
_cpu_threads = int(_slurm_cpus) if _slurm_cpus else max(os.cpu_count() // 2, 1)

_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
_has_gpu = (_cuda_visible not in ('', '-1')) or os.path.exists('/dev/nvidiactl')

# Threading budget. On a GPU run the heavy work happens on-device — the
# main process's TF/OpenMP threads only handle data prep + small CPU
# fallbacks, where >4 threads costs more in scheduler overhead than it
# saves. On a CPU-only run (Mahti CPU partition: 128 EPYC cores) the
# matmul-heavy forward pass executes on CPU and needs the full SLURM
# allocation. NUMEXPR / OPENBLAS pinned so NumPy paths in data.py and
# the q_scaler don't quietly oversubscribe on Mahti's shared CPU node.
_main_threads = 4 if _has_gpu else _cpu_threads
os.environ['OMP_NUM_THREADS'] = str(_main_threads)
os.environ['MKL_NUM_THREADS'] = str(_main_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(_main_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
if _has_gpu:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

# Memory growth: stops TF from grabbing the whole GPU at startup, so
# shared queue nodes (Mahti gpusmall/gpumedium) can coexist with other
# tenants. No-op on CPU-only runs. Wrapped in try because the call
# fails if TF has already initialised the device.
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
                           compute_raman_spectrum, plot_raman_spectrum, ir_spectrum_from_file)
from DescriptorBuilder import make_descriptor_builder
from tqdm import tqdm
from ase.io import read, iread, write


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
    """When `cfg.csc_enable=True`, retune the cfg to the actual Mahti
    hardware the job is running on. Two profiles:

    GPU partition (A100 40 GB, ~32 cores per GPU):
        - kill the disk-cache / pinned-pool / cuFile pipeline (grad_values
          is small enough to live in host RAM as a tf.constant)
        - drop population/batch chunking down to the natural minimum since
          the A100 trivially fits the entire population on-device
        - keep tf.constant grad path

    CPU partition (128 EPYC cores, no GPU):
        - same caching kills (no GPU → no point pinning host buffers or
          using cuFile)
        - pin data to host explicitly (`pin_data_to_cpu=True`) so the
          chunk-staging branch doesn't try to upload to a phantom GPU
        - keep chunk sizes modest because forward passes execute on CPU,
          where larger chunks just inflate per-op latency without the
          GPU's batching amortisation

    All overrides are no-ops outside csc_enable mode (local development
    config is untouched).
    """
    if not getattr(cfg, "csc_enable", False):
        return

    # Pick profile: explicit cfg.csc_profile override, else hardware
    # auto-detection. "auto" picks GPU when /dev/nvidiactl exists or
    # CUDA_VISIBLE_DEVICES is set (computed at module import time).
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

    # Common (both partitions): kill the disk + pinned + cuFile pipeline.
    overrides: dict[str, object] = {
        "cache_gradients_to_disk": False,
        "chunk_prefetch": False,
        "use_pinned_buffers": False,
        "use_cufile": False,
    }
    if use_gpu_profile:
        # Mahti GPU partition tuning. These match the user's empirical
        # finding (population_chunk_size=None, batch_chunk_size=2000)
        # captured in the project_mahti_speedup memory.
        overrides.update({
            "population_chunk_size": None,
            "batch_chunk_size": 2000,
            "pin_data_to_cpu": False,
        })
        profile = "GPU"
    else:
        # CPU partition: no device-side batching benefit, and the chunk-
        # staging code path assumes a GPU. Force the data to live in
        # host RAM (pin_data_to_cpu=True) so the forward path doesn't
        # try to copy to a non-existent device. Smaller batch_chunk_size
        # keeps per-op latency reasonable on the CPU executor.
        overrides.update({
            "population_chunk_size": 10,
            "batch_chunk_size": 500,
            "pin_data_to_cpu": True,
        })
        profile = "CPU"

    # Annotate the profile string when forced via cfg.csc_profile so the
    # log line distinguishes auto-detected vs explicit overrides.
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


# ═══════════════════════════════════════════════════════════════════════
# A/B testing pipeline
# ═══════════════════════════════════════════════════════════════════════

# Config fields the A/B variant FORBIDS in per-arm overrides. These determine
# the shared data/descriptor/architecture pipeline that runs once for both
# arms; changing any of them between arms would invalidate the shared
# descriptors/grad_values or change the model shape, so they MUST be set
# identically in the base cfg.
_AB_SHARED_FIELDS: frozenset[str] = frozenset({
    # Dataset / split / seed
    "data_path", "test_data_path", "allowed_species", "filter_mode",
    "filter_bad_data", "test_ratio", "total_N", "seed",
    # Targets / units (affect descriptor build and pad_and_stack)
    "target_mode", "target_key", "scale_targets", "dipole_units",
    "convert_dipole_to_eangstrom", "dipole_rij_power", "skip_h_centers",
    # SOAP descriptor geometry
    "l_max", "alpha_max", "rcut_hard", "rcut_soft", "basis",
    "compress_mode", "atom_sigma_r", "atom_sigma_t",
    "atom_sigma_r_scaling", "atom_sigma_t_scaling", "central_weight",
    "amplitude_scaling", "scaling_mode", "radial_enhancement",
    "descriptor_mode", "descriptor_precision",
    # Descriptor preprocessing
    "descriptor_scaling", "q_scaler_granularity", "target_centering",
    # Model architecture (shapes the parameter vector)
    "num_neurons", "activation", "descriptor_mixing",
    "descriptor_mixing_arch", "descriptor_mixing_per_type",
    # Loop budget and validation cadence (must be lockstep)
    "num_generations", "val_interval", "val_size",
    # Population sampling (same μ-step structure → shared mirrored noise scheme)
    "pop_size", "init_sigma",
})


def train_model_ab(
    base_cfg: TNEPconfig,
    overrides_a: dict,
    overrides_b: dict,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[TNEP, TNEP]:
    """Run two optimizer configurations in lockstep on the SAME dataset.

    Shared (taken from `base_cfg`, fixed across both arms):
      - Data / split / seed
      - SOAP descriptor geometry + preprocessing
      - Target units / construction
      - Model architecture (so μ is the same shape; sampling RNG matches)
      - num_generations / val_interval / val_size / pop_size / init_sigma

    Variable (specified independently in `overrides_a` and `overrides_b`):
      - Anything NOT in `_AB_SHARED_FIELDS` — most commonly the optimiser
        flags: `snes_active_utilities`, `snes_cov_mode`, `snes_mean_optimizer`,
        `snes_sigma_cumulation`, etc.

    Each arm gets the SAME train/val/test descriptors and grad_values
    (built ONCE), the SAME seed (so RNG draws are identical for the
    sampling-pre-rank phase), and the SAME validation set. They differ
    only in the per-arm optimiser-side cfg. Progress bar shows both
    arms' train/val RMSE per val tick plus the per-arm σ stats plus a
    Δ column (B − A).

    Args:
        base_cfg     : TNEPconfig — fields in `_AB_SHARED_FIELDS` are
                       used as-is. All other fields are overwritten by
                       per-arm overrides; if absent from BOTH overrides
                       the base value applies to both.
        overrides_a  : dict — per-arm cfg overrides for arm A. Must NOT
                       touch any field in `_AB_SHARED_FIELDS`.
        overrides_b  : dict — same for arm B.
        label_a, _b  : short tags shown on the progress bar / save dirs.

    Returns:
        (best_val_model_a, best_val_model_b)
    """
    # --- Validate overrides ---
    valid_keys = set(getattr(TNEPconfig, "__annotations__", {}).keys())
    for arm_name, ovr in (("overrides_a", overrides_a), ("overrides_b", overrides_b)):
        unknown = set(ovr) - valid_keys
        if unknown:
            raise ValueError(f"{arm_name} has unknown TNEPconfig field(s): "
                             f"{sorted(unknown)}")
        forbidden = set(ovr) & _AB_SHARED_FIELDS
        if forbidden:
            raise ValueError(
                f"{arm_name} overrides shared field(s) that determine the "
                f"data/architecture pipeline: {sorted(forbidden)}. "
                f"Set these on `base_cfg` instead — they must match between arms.")

    # --- Build cfgs for each arm (start from base, apply overrides) ---
    import copy as _copy
    cfg_a = _copy.deepcopy(base_cfg)
    cfg_b = _copy.deepcopy(base_cfg)
    for k, v in overrides_a.items():
        setattr(cfg_a, k, v)
    for k, v in overrides_b.items():
        setattr(cfg_b, k, v)
    # Drop disk caches in BOTH arms — they'd race for the same scratch dir.
    cfg_a.cache_gradients_to_disk = False
    cfg_b.cache_gradients_to_disk = False
    cfg_a._gradient_cache_path = None
    cfg_b._gradient_cache_path = None

    # --- Shared data prep: build descriptors + grad_values ONCE ---
    # We do it on cfg_a (data + descriptor fields are identical across the
    # two cfgs by construction — see _AB_SHARED_FIELDS).
    print(f"\n=== A/B run: {label_a} vs {label_b} ===")
    print(f"  shared base cfg → building data + descriptors once")
    dataset, dataset_types_int = collect(cfg_a)
    cfg_a.type_map = {z: idx for idx, z in enumerate(cfg_a.types)}
    cfg_b.type_map = dict(cfg_a.type_map)
    cfg_b.types = list(cfg_a.types)
    cfg_b.num_types = cfg_a.num_types
    if cfg_a.target_mode == 1:
        print_dipole_statistics(dataset, cfg_a, target_key=_resolve_target_key(cfg_a))

    # Resolve shared seed (both arms use the same value)
    if cfg_a.seed is None:
        cfg_a.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        print(f"  cfg.seed was None — generated shared seed: {cfg_a.seed}")
    cfg_b.seed = cfg_a.seed

    cfg_a.randomise(dataset)
    cfg_b.indices = cfg_a.indices   # share train/val split exactly
    train_data, test_pending, val_data = split(dataset, dataset_types_int, cfg_a)

    from DescriptorBuilderGPU import compute_dim_q
    cfg_a.dim_q = compute_dim_q(cfg_a)
    cfg_b.dim_q = cfg_a.dim_q

    _self_only = (cfg_a.target_mode == 1
                  and int(getattr(cfg_a, "dipole_rij_power", 2)) == 0)
    train_data = pad_and_stack(
        train_data, num_types=cfg_a.num_types,
        pin_to_cpu=cfg_a.pin_data_to_cpu,
        q_scaler=getattr(cfg_a, "_q_scaler", None),
        target_mean=getattr(cfg_a, "_target_mean", None),
        self_pairs_only=_self_only)
    val_data = pad_and_stack(
        val_data, num_types=cfg_a.num_types,
        pin_to_cpu=cfg_a.pin_data_to_cpu,
        q_scaler=getattr(cfg_a, "_q_scaler", None),
        target_mean=getattr(cfg_a, "_target_mean", None),
        self_pairs_only=_self_only)

    # --- Build two models — separate run directories ---
    if cfg_a.save_path is not None:
        # Use a parent ab/ directory with per-arm sub-directories.
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent = os.path.join("models", f"ab_{label_a}_vs_{label_b}_{timestamp}")
        os.makedirs(parent, exist_ok=True)
        for cfg_arm, lab in ((cfg_a, label_a), (cfg_b, label_b)):
            arm_dir = os.path.join(parent, lab)
            os.makedirs(os.path.join(arm_dir, "plots"), exist_ok=True)
            cfg_arm.save_path = os.path.join(arm_dir, "auto")
            cfg_arm.save_plots = os.path.join(arm_dir, "plots")
        print(f"  output → {parent}/{{{label_a},{label_b}}}/")

    model_a = TNEP(cfg_a)
    model_b = TNEP(cfg_b)
    print(f"  {label_a}: dim={model_a.optimizer.dim} "
          f"pop={model_a.optimizer.pop_size}")
    print(f"  {label_b}: dim={model_b.optimizer.dim} "
          f"pop={model_b.optimizer.pop_size}")
    if model_a.optimizer.dim != model_b.optimizer.dim:
        raise RuntimeError(
            f"Architecture mismatch: {label_a} dim={model_a.optimizer.dim} "
            f"vs {label_b} dim={model_b.optimizer.dim}. Make sure overrides "
            f"don't touch architecture fields (see _AB_SHARED_FIELDS).")

    # --- Custom interleaved training loop ---
    history_a, history_b, final_a, final_b, best_a, best_b = _run_ab_fit(
        model_a, model_b, train_data, val_data, label_a, label_b)

    # --- Score + save both arms ---
    test_data = materialize_test_data(test_pending, cfg_a,
                                       num_types=cfg_a.num_types,
                                       pin_to_cpu=cfg_a.pin_data_to_cpu)
    for arm_label, arm_model, arm_history in (
            (label_a, best_a, history_a), (label_b, best_b, history_b)):
        m, _ = arm_model.score(test_data)
        print_score_summary(m, arm_model.cfg, prefix=f"[{arm_label}] best-val test set")
        if arm_model.cfg.save_path is not None:
            save_model(arm_model, arm_model.cfg, arm_model.cfg.save_path, label="best_val")
            save_history(arm_history, arm_model.cfg)
            plot_snes_history(arm_history, arm_model.cfg, arm_model.cfg.save_plots, arm_model.cfg.show_plots)
            plot_log_val_fitness(arm_history, arm_model.cfg, arm_model.cfg.save_plots, arm_model.cfg.show_plots)
            plot_sigma_history(arm_history, arm_model.cfg, arm_model.cfg.save_plots, arm_model.cfg.show_plots)

    # --- A/B summary ---
    final_best_a = float(np.min(history_a["val_loss"])) if history_a["val_loss"] else float("nan")
    final_best_b = float(np.min(history_b["val_loss"])) if history_b["val_loss"] else float("nan")
    print(f"\n=== A/B summary ===")
    print(f"  {label_a}: best val_RMSE = {final_best_a:.6e}")
    print(f"  {label_b}: best val_RMSE = {final_best_b:.6e}")
    delta = final_best_b - final_best_a
    pct = (delta / final_best_a * 100.0) if final_best_a > 0 else 0.0
    print(f"  Δ (B − A) = {delta:+.6e}  ({pct:+.2f}%)")
    if abs(pct) < 0.5:
        print(f"  → arms statistically indistinguishable at this granularity")
    elif delta < 0:
        print(f"  → {label_b} wins by {-pct:.2f}%")
    else:
        print(f"  → {label_a} wins by {pct:.2f}%")
    return best_a, best_b


def _run_ab_fit(
    model_a: TNEP, model_b: TNEP,
    train_data: dict, val_data: dict,
    label_a: str, label_b: str,
):
    """Interleaved per-generation training of two TNEPs sharing the same data.

    Mirrors the core mechanics of SNES.fit() — ask, evaluate (per-type or
    global), rank, update, validate — but runs both optimisers in lockstep
    inside one Python loop so we can:
      1. share the training/val tensors (built once, by the caller);
      2. print a single combined progress bar with both arms' stats + Δ;
      3. keep their RNG streams independent (each has its own tf_rng
         generator seeded from cfg.seed).

    Intentionally simplified vs SNES.fit():
      - No plateau-driven σ reset.
      - No checkpointing — A/B is a comparison utility, not a long-running
        production run. Use train_model() for production resumability.
    """
    import time, sys
    cfg_a = model_a.cfg
    cfg_b = model_b.cfg
    snes_a = model_a.optimizer
    snes_b = model_b.optimizer

    def _init_history():
        return {
            "generation": [], "train_loss": [], "train_rmse": [], "val_loss": [],
            "best_rmse": [], "worst_rmse": [],
            "sigma_min": [], "sigma_max": [], "sigma_mean": [], "sigma_median": [],
            "L1": [], "L2": [], "best_rrmse": [], "avg_rrmse": [],
            "timing": {"sample_batch": [], "evaluate": [], "rank_update": [],
                       "validate": [], "overhead": []},
        }
    history_a = _init_history()
    history_b = _init_history()

    num_gen = int(cfg_a.num_generations)
    val_interval = int(cfg_a.val_interval)
    best_val_a, best_val_b = float("inf"), float("inf")
    best_mu_a = tf.identity(snes_a.mu)
    best_mu_b = tf.identity(snes_b.mu)

    train_start = time.perf_counter()
    print(f"\n  ── A/B training: {num_gen} gens, val every {val_interval} ──")

    def _one_gen(snes, batch_data) -> tuple[float, float, np.ndarray]:
        """One SNES generation for a single arm. Returns (avg_fitness,
        best_rmse, fitness_per_cand). Mirrors SNES.fit's SNES branch."""
        samples, aux = snes.ask()
        if snes._per_type:
            fitness_per_type_rmse = snes.evaluate_population(
                samples, batch_data, return_per_type=True)
            fitness = fitness_per_type_rmse[:, -1]
        else:
            fitness = snes.evaluate_population(samples, batch_data)
            fitness_per_type_rmse = None
        rmse_pc = snes._last_rmse_per_cand
        avg_f = float(tf.reduce_mean(fitness).numpy())
        best_f = float(tf.reduce_min(rmse_pc).numpy())
        if snes._per_type:
            s_iso_sorted = snes._build_per_type_gradients(
                aux["s_iso"], fitness_per_type_rmse, samples)
            delta_sorted = snes._build_per_type_gradients(
                aux["delta"], fitness_per_type_rmse, samples)
        else:
            ranks = tf.argsort(fitness)
            s_iso_sorted = tf.gather(aux["s_iso"], ranks)
            delta_sorted = tf.gather(aux["delta"], ranks)
        global_ranks = tf.argsort(fitness)
        s_eff_global = tf.gather(aux["delta"], global_ranks) / snes.sigma
        update_aux = {"s_iso": s_iso_sorted, "delta": delta_sorted,
                      "s_eff_global": s_eff_global}
        if "y" in aux:
            update_aux["y_global"] = tf.gather(aux["y"], global_ranks)
            update_aux["s_iso_global"] = tf.gather(aux["s_iso"], global_ranks)
        snes.update(snes.utilities, update_aux)
        return avg_f, best_f, fitness.numpy()

    for gen in range(num_gen):
        avg_a, best_rmse_a, _ = _one_gen(snes_a, train_data)
        avg_b, best_rmse_b, _ = _one_gen(snes_b, train_data)

        do_val = (gen % val_interval == 0) or (gen == num_gen - 1)
        if do_val:
            val_a = float(snes_a.validate(val_data, snes_a.mu))
            val_b = float(snes_b.validate(val_data, snes_b.mu))
            train_a = float(snes_a.validate(train_data, snes_a.mu))
            train_b = float(snes_b.validate(train_data, snes_b.mu))
            if val_a < best_val_a:
                best_val_a = val_a
                best_mu_a = tf.identity(snes_a.mu)
            if val_b < best_val_b:
                best_val_b = val_b
                best_mu_b = tf.identity(snes_b.mu)
            for hist, lab, train_l, val_l, best_v, snes in (
                    (history_a, label_a, train_a, val_a, best_val_a, snes_a),
                    (history_b, label_b, train_b, val_b, best_val_b, snes_b)):
                sig = snes.sigma.numpy()
                hist["generation"].append(gen)
                hist["train_loss"].append(train_l)
                hist["train_rmse"].append(train_l)
                hist["val_loss"].append(val_l)
                hist["best_rmse"].append(best_rmse_a if lab == label_a else best_rmse_b)
                hist["worst_rmse"].append(0.0)
                hist["sigma_min"].append(float(sig.min()))
                hist["sigma_max"].append(float(sig.max()))
                hist["sigma_mean"].append(float(sig.mean()))
                hist["sigma_median"].append(float(np.median(sig)))
                hist["L1"].append(0.0); hist["L2"].append(0.0)
                hist["best_rrmse"].append(0.0); hist["avg_rrmse"].append(0.0)
                for k in hist["timing"]:
                    hist["timing"][k].append(0.0)

            elapsed = time.perf_counter() - train_start
            eta_s = (elapsed / max(gen + 1, 1)) * (num_gen - gen - 1)
            d_train = train_b - train_a
            d_val = val_b - val_a
            d_best = best_val_b - best_val_a
            sigma_a_max = float(snes_a.sigma.numpy().max())
            sigma_b_max = float(snes_b.sigma.numpy().max())
            # Single compact line; \r-rewriting style like SNES.fit
            line = (
                f"\r{gen+1:>6}/{num_gen}  "
                f"[{label_a}] tr={train_a:.4e} v={val_a:.4e} bv={best_val_a:.4e} σmax={sigma_a_max:.2e}  "
                f"[{label_b}] tr={train_b:.4e} v={val_b:.4e} bv={best_val_b:.4e} σmax={sigma_b_max:.2e}  "
                f"Δv={d_val:+.3e} Δbv={d_best:+.3e}  "
                f"⏱ {int(elapsed//60):02d}:{int(elapsed%60):02d} "
                f"ETA {int(eta_s//60):02d}:{int(eta_s%60):02d}"
            )
            sys.stdout.write(line[:240])   # truncate if narrower terminal
            sys.stdout.flush()
            if (gen + 1) % (val_interval * 100) == 0:
                sys.stdout.write("\n")   # periodic line-flush so log files stay readable

    sys.stdout.write("\n")

    # Build final and best-val models for each arm
    from SNES import _set_model_params
    final_a = TNEP(cfg_a); _set_model_params(final_a, *snes_a.reconstruct_params_tf(snes_a.mu))
    best_a  = TNEP(cfg_a); _set_model_params(best_a,  *snes_a.reconstruct_params_tf(best_mu_a))
    final_b = TNEP(cfg_b); _set_model_params(final_b, *snes_b.reconstruct_params_tf(snes_b.mu))
    best_b  = TNEP(cfg_b); _set_model_params(best_b,  *snes_b.reconstruct_params_tf(best_mu_b))
    return history_a, history_b, final_a, final_b, best_a, best_b


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
    #
    # Definition: RRMSE = RMSE / std(ref). Algebraically equivalent to
    # sqrt(1 - R²) and consistent with the SNES per-candidate form
    # (SNES.py:2046). Per-component uses per-component std; overall
    # uses std of the flattened target array.
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


def _print_param_breakdown(model) -> None:
    """Print a per-layer breakdown of the SNES parameter budget.

    Pulls the cached per-component counts from `model.optimizer` and
    prints a table of (component → count → % of total). Polarisability
    mirrors the primary ANN when target_mode == 2 (n_anns_total =
    2 · n_primary).
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

    # Resolve cfg.seed=None to a concrete, serialisable seed BEFORE any RNG
    # is constructed (whether this is a fresh run or a resume from an old
    # checkpoint that didn't persist its seed). Without this the actual
    # entropy consumed by np.random.default_rng(None) / TF's
    # non-deterministic generator is lost — and a resumed run from a
    # pre-fix checkpoint would carry the unreproducibility forward.
    if cfg.seed is None:
        cfg.seed = int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        print(f"  cfg.seed was None — generated and stored "
              f"reproducible seed: {cfg.seed}")

    # Optional bitwise reproducibility: cfg.seed fixes every RNG stream, but
    # GPU reductions are still non-deterministic across runs unless TF op
    # determinism is enabled. Done here (before data/model/training ops are
    # built) so the whole training computation runs deterministically.
    if getattr(cfg, "deterministic", False):
        try:
            tf.config.experimental.enable_op_determinism()
            print("  cfg.deterministic=True — TF op determinism enabled "
                  "(same cfg.seed => bitwise-identical runs; GPU ops slower, "
                  "and an op lacking a deterministic GPU kernel will raise).")
        except Exception as e:                       # pragma: no cover
            print(f"  WARNING: could not enable op determinism: {e}")

    if resume_state is not None and isinstance(getattr(cfg, "indices", None), np.ndarray):
        # Indices already restored from checkpoint — would re-shuffle to
        # the same values anyway (deterministic via cfg.seed for runs
        # that persisted the resolved seed), but skipping makes the
        # resume self-explanatory and avoids any surprise if the
        # dataset was extended between runs.
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

    # Optional pretrained encoder front-end. Loads the encoder bundle
    # (model + analytic Jacobian) and either:
    #   - static preprocess (train_encoder=False): rewrites train/val
    #     descriptor + gradient lists in-place so downstream consumers
    #     see a Z-dim descriptor. test_data is handled lazily inside
    #     `materialize_test_data`, which reads the same cfg bundle.
    #   - iterative (train_encoder=True): only loads + stashes; the
    #     TNEP forward path applies the encoder on the fly.
    # In both modes, cfg.dim_q is overridden to the encoder's latent
    # dim so q_scaler, target_mean, pad_and_stack and TNEP's own
    # parameter shapes are sized correctly.
    if getattr(cfg, "encoder_path", None) is not None:
        from encoder_frontend import (
            setup_encoder_frontend, apply_encoder_to_lists)
        # Mutual-exclusion: the encoder rewrites the descriptor axis at
        # data-prep time, so any feature that operates on the raw SOAP
        # block layout (descriptor_preprocess_contract, descriptor_mixing)
        # would size its weights against the wrong axis. These features
        # were not designed to compose; flag the combination loudly.
        if str(getattr(cfg, "descriptor_preprocess_contract", "off")) != "off":
            raise NotImplementedError(
                f"encoder_path is incompatible with "
                f"descriptor_preprocess_contract="
                f"{cfg.descriptor_preprocess_contract!r}: the preprocess "
                f"fold (W_pre_angular) operates on raw SOAP block "
                f"layout, but the encoder rewrites the descriptor axis "
                f"to a {cfg._encoder_latent_dim if hasattr(cfg, '_encoder_latent_dim') else 'Z'}-dim latent. "
                f"Set descriptor_preprocess_contract='off' to use the "
                f"encoder.")
        if bool(getattr(cfg, "descriptor_mixing", False)):
            raise NotImplementedError(
                "encoder_path is incompatible with descriptor_mixing="
                "True: U_pair is sized to raw SOAP per-pair blocks, but "
                "the encoder rewrites the descriptor axis. Set "
                "descriptor_mixing=False to use the encoder.")
        if (str(getattr(cfg, "q_scaler_granularity", "per_component")).lower()
                == "l_block"):
            raise NotImplementedError(
                "encoder_path is incompatible with "
                "q_scaler_granularity='l_block': the l-block layout "
                "is defined against raw SOAP, but the encoder rewrites "
                "the descriptor axis. Use 'per_component' instead.")
        setup_encoder_frontend(cfg)
        # Tightened layout match — Q_raw alone can collide between
        # different (T, αmax, l_max, compress_mode) combinations.
        # Verify each axis-shaping cfg field against the encoder run's
        # saved cfg so the per-channel ordering matches by construction.
        from pathlib import Path as _Path
        import encoder_io as _enc_io
        _enc_cfg = _enc_io.load_config(_Path(cfg.encoder_path))
        _checks = [
            ("alpha_max", int(_enc_cfg.alpha_max), int(cfg.alpha_max)),
            ("l_max", int(_enc_cfg.l_max), int(cfg.l_max)),
            ("compress_mode", str(_enc_cfg.compress_mode),
             str(cfg.compress_mode)),
        ]
        # T (num species) check via the encoder's stored T attribute.
        _enc_T = (int(getattr(cfg._encoder, "T", 0))
                  or int(getattr(_enc_cfg, "T", 0))
                  or 0)
        if _enc_T:
            _checks.append(("num_types (T)", _enc_T, int(cfg.num_types)))
        _mismatches = [(name, enc_v, cur_v) for name, enc_v, cur_v in _checks
                       if enc_v != cur_v]
        if _mismatches:
            raise ValueError(
                "Encoder SOAP layout does not match current cfg "
                "(matching Q_raw is not sufficient — per-channel "
                "ordering also depends on αmax, l_max, T, "
                "compress_mode):\n" +
                "\n".join(
                    f"  {name}: encoder={enc_v}  cfg={cur_v}"
                    for name, enc_v, cur_v in _mismatches))
        if int(cfg._encoder_q_raw) != int(cfg.dim_q):
            raise ValueError(
                f"Encoder expects Q_raw={cfg._encoder_q_raw} but the "
                f"current cfg resolves to dim_q={cfg.dim_q}. Re-train "
                f"the encoder with the matching SOAP layout, or update "
                f"cfg.alpha_max / l_max / compress_mode to match the "
                f"encoder's training-time layout.")
        # SNES + iterative is not yet wired (would need encoder
        # application inside the SNES candidate-eval / validate path,
        # plus exposing Willatt u to the μ vector). The user must
        # either switch to Adam (which supports full encoder backprop)
        # or set train_encoder=False to use the static preprocess.
        if (str(getattr(cfg, "optimizer", "snes")).lower() == "snes"
                and bool(getattr(cfg, "train_encoder", False))):
            raise NotImplementedError(
                "train_encoder=True is only supported under "
                "cfg.optimizer='adam' (the Adam path backprops through "
                "the encoder). For SNES, set train_encoder=False and "
                "use the static preprocess.")
        if not bool(getattr(cfg, "train_encoder", False)):
            # Disk-backed gradient streaming is incompatible with the
            # static rewrite (gradients live on disk as raw bytes, not
            # in `train_data['gradients']`). Force in-memory mode for
            # the static encoder pipeline; the iterative path keeps the
            # raw stream and applies the encoder online instead.
            if (getattr(cfg, "cache_gradients_to_disk", False)
                    and train_data.get("_prebuilt_gv") is not None):
                raise NotImplementedError(
                    "cache_gradients_to_disk=True is incompatible with "
                    "static encoder preprocess (train_encoder=False). "
                    "Either disable disk-streaming or set "
                    "train_encoder=True so the encoder is applied "
                    "per-batch over the on-disk raw gradients.")
            train_data["descriptors"], train_data["gradients"] = (
                apply_encoder_to_lists(
                    train_data["descriptors"], train_data["gradients"],
                    cfg._encoder_J, cfg._encoder))
            val_data["descriptors"], val_data["gradients"] = (
                apply_encoder_to_lists(
                    val_data["descriptors"], val_data["gradients"],
                    cfg._encoder_J, cfg._encoder))
            cfg.dim_q = int(cfg._encoder_latent_dim)
            print(f"[encoder] static preprocess: dim_q overridden to "
                  f"{cfg.dim_q} (Q_raw={cfg._encoder_q_raw}).")
        else:
            # Iterative path also overrides dim_q since TNEP sees Z, not
            # Q_raw, when its inputs flow through the encoder.
            cfg.dim_q = int(cfg._encoder_latent_dim)
            print(f"[encoder] iterative mode: dim_q overridden to "
                  f"{cfg.dim_q}; encoder applied per-batch by TNEP.")

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

    # Per-component target centering. Computed ONCE over the training
    # targets, applied to train/val/test, persisted in /weights/
    # target_mean. Restored from checkpoint on resume (preserves cfg
    # value if already set).
    if bool(getattr(cfg, "target_centering", False)):
        if getattr(cfg, "_target_mean", None) is None:
            from data import _compute_target_mean
            target_dim = (1 if cfg.target_mode == 0
                          else (3 if cfg.target_mode == 1 else 6))
            cfg._target_mean = _compute_target_mean(
                train_data["targets"], target_dim)
            tm = cfg._target_mean
            print(f"  Computed target_mean over {len(train_data['targets'])} "
                  f"training structures (target_dim={tm.size}):")
            print(f"    mean = {np.array2string(tm, precision=4, suppress_small=True)}")
            print(f"    Targets will be shifted to zero-mean for training; "
                  f"mean added back at inference.")
        else:
            print(f"  Reusing target_mean from checkpoint "
                  f"(shape={cfg._target_mean.shape}, no recompute on resume).")

    # Convert to padded dense tensors for GPU-batched evaluation. test_data
    # is intentionally NOT padded here; it gets padded by materialize_test_data
    # the first time it's actually consumed.
    # When dipole_rij_power=0 (target_mode=1), only self-pair gradients
    # contribute to the dipole sum — neighbour pairs would be multiplied
    # by zero. Tell pad_and_stack to drop them at data-build time so
    # grad_values shrinks from O(N·M) to O(N) per structure. Actual
    # savings depend on the average neighbour count — the train_P /
    # val_P counts logged below report the true post-filter pair count.
    _self_only = (cfg.target_mode == 1
                  and int(getattr(cfg, "dipole_rij_power", 2)) == 0)
    train_data = pad_and_stack(
        train_data, num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        gradient_cache_path=getattr(cfg, "_gradient_cache_path", None),
        cache_tag="train",
        q_scaler=getattr(cfg, "_q_scaler", None),
        target_mean=getattr(cfg, "_target_mean", None),
        self_pairs_only=_self_only)
    val_data   = pad_and_stack(
        val_data,   num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        gradient_cache_path=getattr(cfg, "_gradient_cache_path", None),
        cache_tag="val",
        q_scaler=getattr(cfg, "_q_scaler", None),
        target_mean=getattr(cfg, "_target_mean", None),
        self_pairs_only=_self_only)
    if _self_only:
        n_train_pairs = int(train_data["grad_values"].shape[0])
        n_val_pairs   = int(val_data["grad_values"].shape[0])
        print(f"  dipole_rij_power=0: COO restricted to self-pairs only "
              f"(train P={n_train_pairs}, val P={n_val_pairs})")

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
    # When preprocess contraction is on, cfg.dim_q has been overridden to
    # the contracted Q_new (raw dim lives at cfg.dim_q_raw). Report the
    # compression ratio so the effect of mode / l_keep / per_type is visible.
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
        # best_sigma is None under cov_mode="crfmnes" (snapshot skipped on
        # save; the active scale lives in _cr_sig). Same lifecycle pattern
        # as SNES.fit's end-of-run restore at SNES.py:2248 — skip the assign
        # rather than KeyError'ing on None.
        if (str(getattr(cfg, "snes_cov_mode", "none")).lower() != "crfmnes"
                and resume_state.get("best_sigma") is not None):
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
    """Count frames in a trajectory file. Format-aware:

    - XYZ / extXYZ : walk only the atom-count headers (no parsing).
        Each frame is: <N> line, comment line, then N atom lines.
        Read N, skip N+1 lines per frame.
    - ASE binary .traj : use ase.io.trajectory.Trajectory which
        supports len() directly without loading frames.
    - Anything else : fall back to ase.io.iread (slower; streams the
        file but discards each frame after counting).

    Function name kept for back-compat.
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

    # Unknown extension — fall back to ASE's iterator. Streams the file
    # without keeping frames; slower than the xyz/traj fast paths but
    # works for any format ASE understands (e.g. .lammpstrj, .pdb).
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
        freq_cm, intensity, power, acf = compute_ir_spectrum(
            dipoles, dt_fs=dt_fs,
            smooth_k=ir_smooth, smooth_kind=ir_smooth_kind,
            power_dc_cutoff_cm=ir_power_dc_cutoff_cm)
        # Build a descriptive plot label: <trajectory_stem>_<model_stem>
        # so multiple runs against different models or different
        # trajectories don't overwrite each other.
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

    Output columns: pred_xyz | ref_xyz | N_atoms (whitespace-separated).
    Targets and predictions are in per-atom space when cfg.scale_targets=True
    (the default for dipole models).

    Structures whose species are not a subset of the model's `cfg.types`
    are silently dropped (with a summary line) — otherwise the model
    would crash on the first unknown Z in `assign_type_indices`.
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