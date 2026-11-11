from __future__ import annotations

import numpy as np
import os

_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
_has_gpu = (_cuda_visible not in ('', '-1')) or os.path.exists('/dev/nvidiactl')


def _allocated_cpu_count() -> int:
    """CPU cores this process may use: under Slurm the full allocation
    (SLURM_CPUS_PER_TASK, SLURM_CPUS_ON_NODE, then the affinity mask — never //2,
    which would halve the node); off Slurm cpu_count()//2 to spare a shared box.
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


def _csc_use_all_cpus() -> bool:
    """Use the full CPU allocation even on a GPU node on CSC/Mahti (billing is
    per node, so idle cores are free). Decided before TensorFlow imports because
    the thread env vars freeze at import: TNEP_CSC_ENABLE env first, then the
    TNEPconfig.csc_enable class default. A per-instance cfg.csc_enable set later
    is too late to grow TF's thread pools.
    """
    env = os.environ.get('TNEP_CSC_ENABLE')
    if env is not None:
        return env.strip().lower() not in ('', '0', 'false', 'no', 'off')
    try:
        from TNEPconfig import TNEPconfig
        return bool(getattr(TNEPconfig, 'csc_enable', False))
    except Exception:
        return False


# Threading budget, set BEFORE TensorFlow is imported. GPU runs cap the main
# process at 4 threads (work is on-device); CPU-only and csc_enable runs use
# the full allocation. NUMEXPR/OPENBLAS pinned so data.py's NumPy paths don't
# oversubscribe.
_use_all_cpus = (not _has_gpu) or _csc_use_all_cpus()
_main_threads = _cpu_threads if _use_all_cpus else 4
os.environ['OMP_NUM_THREADS'] = str(_main_threads)
os.environ['MKL_NUM_THREADS'] = str(_main_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(_main_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_main_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = '4' if _use_all_cpus else '2'
if _has_gpu:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

# Memory growth so TF doesn't grab the whole GPU on shared nodes; try guards an
# already-initialised device.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from TNEP import TNEP, gen0_ann_inputs
from TNEPconfig import TNEPconfig, neighbor_readout_on
from data import (collect, split, pad_and_stack,
                  print_dipole_statistics, print_polarizability_statistics,
                  assign_type_indices, prepare_eval_data, print_score_summary,
                  _resolve_target_key, materialize_test_data,
                  dipole_self_pairs_only)
from plotting import (plot_snes_history, plot_log_val_fitness, plot_sigma_history,
                      plot_timing, plot_correlation, plot_cosine_similarity,
                      plot_loss_breakdown, plot_error_vs_magnitude,
                      plot_error_decomposition)
from model_io import save_model, save_history, setup_run_directory
from spectroscopy import (predict_trajectory_batch,
                           compute_ir_spectrum, plot_ir_spectrum, plot_power_spectrum,
                           compute_raman_spectrum, plot_raman_spectrum)
from DescriptorBuilder import make_descriptor_builder
from tqdm import tqdm
from ase.io import read, iread


def _apply_csc_overrides(cfg: TNEPconfig) -> None:
    """csc_enable=True: retune chunk sizing to Mahti. GPU profile: A100 holds
    the whole population, so chunking drops to the minimum. CPU profile: pin
    data to host and keep chunks modest (larger ones inflate per-op latency).
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
        # CPU partition: pin to host; smaller chunks keep per-op latency reasonable.
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
    # Thread env vars froze at import; if csc_enable was set only on this cfg
    # instance (not the class default / env), a GPU run is still capped at 4.
    if _has_gpu and _main_threads < _cpu_threads:
        print(f"  NOTE: main-process threads are {_main_threads}, not the full "
              f"{_cpu_threads} allocated cores. Export TNEP_CSC_ENABLE=1 (or set "
              f"the TNEPconfig.csc_enable class default) before launch to use all "
              f"cores on a GPU node — TF thread pools cannot grow after import.")


def train_model(cfg: TNEPconfig | None = None,
                checkpoint: str | None = None,
                extract_model: bool = False,
                cfg_overrides: dict | None = None) -> TNEP:
    """Full TNEP pipeline: load, split, train, test, plot, save.

    checkpoint: resume from a checkpoint.h5 at last_gen + 1 (its embedded cfg
    replaces `cfg`). extract_model: skip SNES and rebuild final_model /
    best_val_model from the checkpoint's μ / best_μ, then score/save/plot.
    cfg_overrides: cfg fields applied after load_checkpoint — repairs old
    checkpoints whose JSON lacks architecture fields (else class defaults leak
    in → μ-shape mismatch). Returns the best-val model (cfg via model.cfg).
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
            # Validate keys: a typo would setattr a new attribute and silently never apply.
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
    """Correlation / cos-sim / error plots for one (data, preds, metrics) triple:
    per-atom always; total (× num_atoms) only when metrics has total_* keys.
    Cos-sim is scale-invariant so it is omitted at total scale. save_dir / show
    override cfg when non-None.
    """
    targets = data["targets"].numpy()
    preds_np = preds.numpy()
    save = save_dir if save_dir is not None else cfg.save_plots
    show = show if show is not None else cfg.show_plots

    # RRMSE = RMSE / std(ref) (≡ sqrt(1 − R²)) from total-scale values; shown
    # identically on per-atom and total plots.
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
    plot_error_decomposition(targets, preds_np, cfg, save, show, suffix=suffix_per_atom)

    if not has_total:
        return
    total_metrics = {
        "rmse": metrics["total_rmse"],
        "r2": metrics["total_r2"],
        "r2_components": metrics["total_r2_components"],
        **rrmse_payload,
    }
    # Cos-sim annotations only (no standalone plot at total scale).
    if "cos_sim_all" in metrics:
        total_metrics["cos_sim_mean"] = metrics["cos_sim_mean"]
        total_metrics["cos_sim_all"] = metrics["cos_sim_all"]
    plot_correlation(total_targets, total_preds, total_metrics, cfg, save, show, suffix=suffix_total)
    plot_error_vs_magnitude(total_targets, total_preds, cfg, save, show, suffix=suffix_total)
    plot_error_decomposition(total_targets, total_preds, cfg, save, show, suffix=suffix_total)


def _setup_grad_staging(cfg: TNEPconfig, train_data: dict, val_data: dict) -> None:
    """Pre-stage per-chunk pair indices; pin_data_to_cpu=False also moves static
    fields to GPU for the resident fast path, else chunks are sliced on demand."""
    from data import prestage_chunk_indices, move_data_to_gpu

    chunk = cfg.batch_chunk_size
    pin = bool(getattr(cfg, "pin_data_to_cpu", True))

    # Pre-stage pair indices per chunk range. Tiny GPU-resident tensors.
    for d in (train_data, val_data):
        S = int(d["num_atoms"].shape[0])
        c = chunk if chunk is not None else S
        ranges = [(s, min(s + c, S)) for s in range(0, S, c)]
        prestage_chunk_indices(d, ranges)

    # GPU-resident path; the flag triggers _stage_chunk_resident in prefetched_chunks.
    if not pin:
        for d in (train_data, val_data):
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)


def _print_param_breakdown(model) -> None:
    """Print a per-layer breakdown (component → count → % of total) of the
    SNES parameter budget."""
    opt = model.optimizer
    cfg = model.cfg
    total = int(opt.dim)
    if total <= 0:
        return

    rows: list[tuple[str, int]] = []

    # Primary ANN — per-layer breakdown.
    _G = model._head_groups
    _H = f"{_G}·H" if _G > 1 else "H"          # one hidden block per kernel group
    rows.append((f"ANN  W0  [T·Q·{_H}]", int(opt._n_W0)))
    rows.append((f"ANN  b0  [T·{_H}]",   int(opt._n_b0)))
    _S = str(model._head_slots)
    _w1_label = (f"ANN  W1  [T·{_H}·{_S}] block-diag" if _G > 1
                 else f"ANN  W1  [T·H·{_S}]" if model._neighbor_readout
                 else "ANN  W1  [T·H]")
    rows.append((_w1_label,          int(opt._n_W1)))
    # b1 stays in the SNES tail for layout stability but cancels identically
    # in the neighbor-basis dipole readout (Σ_i v_i ≡ 0).
    _b1_label = ("ANN  b1  (unused: neighbor readout)"
                 if model._neighbor_readout else "ANN  b1")
    rows.append((_b1_label,          int(opt._n_b1)))
    if opt.n_c > 0:
        _m = len(model._models)
        rows.append((("c_pair  [C·T²" if model._species_pairs == "ordered"
                      else "c_pair  [C·T(T+1)/2") + (f"·{_m} models]" if _m > 1 else "]"),
                     int(opt.n_c)))
    # nep_descriptor="sum": extra models' ANNs, one per model after the main one.
    for name, n in zip(model._models[1:], opt._n_expert):
        rows.append((f"ANN_{name}  [T·(d·H_e + H_e + H_e·Hw)]", int(n)))

    # Optional add-ons.
    if int(opt.n_U_pair) > 0:
        rows.append(("input-side mixing  U_pair", int(opt.n_U_pair)))
    if int(getattr(opt, "n_radial", 0)) > 0:
        rows.append(("radial basis  M_radial [T·S·α·K]", int(opt.n_radial)))
    if int(getattr(opt, "n_nep", 0)) > 0:
        rows.append(("NEP  c_r  [T·T·(n+1)·(K+1)]", int(opt.n_c_r)))
        rows.append(("NEP  c_a  [T·T·(n+1)·(K+1)]", int(opt.n_c_a)))
    if int(getattr(opt, "n_sum_delta", 0)) > 0:
        rows.append(("sum delta  [3]", int(opt.n_sum_delta)))

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
    """Body of train_model. On resume the split comes from the checkpoint's
    cfg.indices and run-directory setup is skipped; extract_model skips SNES
    and rebuilds the models from the checkpoint's μ / best_μ.
    """
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

    # Test descriptors are built lazily on first scoring (materialize_test_data).
    train_data, test_pending, val_data = split(dataset, dataset_types_int, cfg)

    # Resolve dim_q before any consumer needs it; cross-checked below.
    from descriptor_layout import compute_dim_q, staged_descriptor_width
    cfg.dim_q = compute_dim_q(cfg)

    # dipole_rij_power=0 / neighbor readout read only self-pair gradients, so
    # drop neighbour pairs (grad_values O(N·M) → O(N)).
    _self_only = dipole_self_pairs_only(cfg)
    train_data = pad_and_stack(
        train_data, num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        self_pairs_only=_self_only)
    val_data   = pad_and_stack(
        val_data,   num_types=cfg.num_types, pin_to_cpu=cfg.pin_data_to_cpu,
        self_pairs_only=_self_only)
    if _self_only:
        n_train_pairs = int(train_data["grad_values"].shape[0])
        n_val_pairs   = int(val_data["grad_values"].shape[0])
        _reason = ("neighbor readout"
                   if neighbor_readout_on(cfg)
                   else "dipole_rij_power=0")
        print(f"  {_reason}: COO restricted to self-pairs only "
              f"(train P={n_train_pairs}, val P={n_val_pairs})")

    _setup_grad_staging(cfg, train_data, val_data)

    # Lazy test-data build; cached after the first call.
    def get_test_data():
        return materialize_test_data(test_pending, cfg,
                                     num_types=cfg.num_types,
                                     pin_to_cpu=cfg.pin_data_to_cpu)

    # Cross-check built descriptor width vs cfg (zero under nep_descriptor="replace",
    # where the SOAP spectrum is skipped).
    built_width = int(train_data["descriptors"][0].shape[-1])
    if built_width != staged_descriptor_width(cfg):
        raise RuntimeError(
            f"staged_descriptor_width={staged_descriptor_width(cfg)} disagrees "
            f"with built descriptor width {built_width}; cfg / builder mismatch.")
    print("Dimension of q: " + str(cfg.dim_q))
    if cfg.radial_basis_size is not None:
        _per = ("shared across neighbour species (species-summed, dim_q = "
                "α(α+1)/2·L)" if cfg.radial_basis_species_sum
                else "per (centre type, neighbour species) (species-blocked, "
                "SOAP-turbo dimensions)")
        _shape = [cfg.num_types, cfg.num_types, cfg.alpha_max, cfg.radial_basis_size]
        if cfg.radial_basis_per_l:
            _shape.append(cfg.l_max + 1)
        print(f"Learned radial basis: {cfg.alpha_max} functions {_per} as "
              f"combinations of K={cfg.radial_basis_size} orthonormal poly3 "
              f"functions"
              + (", separate per l block" if cfg.radial_basis_per_l else "")
              + {"l2": "", "none": "; per-atom L2 normalisation OFF (raw spectrum × scale²)",
                 "sqrt": "; per-atom L2 normalisation OFF (sign·√|q| × scale, degree 1)"}[cfg.radial_basis_norm]
              + f" (M_radial {_shape} in the SNES vector; init = fixed basis)")
    if cfg.nep_descriptor != "off":
        from nep_descriptor import nep_dim
        print(f"NEP descriptor ({cfg.nep_descriptor}): dim_nep={nep_dim(cfg)}, "
              f"rc_radial={cfg.nep_rc_radial}, rc_angular={cfg.nep_rc_angular}, "
              f"n_max=({cfg.nep_n_max_radial}, {cfg.nep_n_max_angular}), "
              f"basis_size=({cfg.nep_basis_size_radial}, {cfg.nep_basis_size_angular}), "
              f"four_body={cfg.nep_four_body}, five_body={cfg.nep_five_body}; "
              f"c init U(−1,1) per GPUMD; c excluded from L1/L2 (GPUMD includes it)"
              + (f"; sum: ANN_2b + ANN_3b (H_e = {cfg.nep_num_neurons}) + SOAP ANN, "
                 f"expert W1 = 0 at gen 0, delta (soap, 2b, 3b) = "
                 f"{list(cfg.sum_delta)}"
                 + (" (learned, excluded from L1/L2)" if cfg.sum_delta_learn else " (fixed)")
                 if cfg.nep_descriptor == "sum" else "")
              + ("; tnep readout: NEP pair forces per candidate (image-exact |r_ij|^"
                 f"{cfg.dipole_rij_power} weights)"
                 if cfg.target_mode == 1 and cfg.dipole_readout == "tnep" else ""))

    # Run dir models/n{neurons}_q{dim_q}_pop{pop}_{timestamp}/; resume reuses the checkpoint's.
    if cfg.save_path is not None and resume_state is None:
        setup_run_directory(cfg)
    elif resume_state is not None and cfg.save_path is not None:
        run_dir = os.path.dirname(cfg.save_path) or "."
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        cfg.save_plots = os.path.join(run_dir, "plots")
        print(f"  resume: writing outputs to existing run dir {run_dir}")

    if cfg.radial_basis_norm != "l2" and str(cfg.descriptor_scale) == "off":
        print("WARNING: radial_basis_norm != 'l2' without descriptor_scale — "
              "the raw spectrum's per-channel spread is uncorrected; set "
              "descriptor_scale='block', 'channel' or 'minmax'.")
    # Descriptor scale m from the gen-0 training descriptor, once (a checkpoint's
    # cfg already has it).
    if str(cfg.descriptor_scale) != "off" and cfg.q_scale is None:
        from descriptor_layout import compute_q_scale, q_scale_groups
        q0 = gen0_ann_inputs(cfg, train_data)
        amask = train_data["atom_mask"]
        m, mu = compute_q_scale(q0, np.asarray(train_data["Z_int"]),
                                amask.numpy() if hasattr(amask, "numpy") else np.asarray(amask), cfg)
        cfg.q_scale = m.tolist()
        cfg.q_shift = mu.tolist()
        for t, z in enumerate(cfg.types):
            print(f"Descriptor scale ({cfg.descriptor_scale}), centre Z={z}: m per group "
                  f"[min / median / max] = "
                  + ", ".join(f"{m[t, g].min():.2f}/{np.median(m[t, g]):.2f}/{m[t, g].max():.2f}"
                              for g in q_scale_groups(cfg))
                  + ("; no centring (minmax)" if str(cfg.descriptor_scale) == "minmax" else
                     "; centred (mean|μ| per group = "
                     + ", ".join(f"{np.abs(mu[t, g]).mean():.2e}" for g in q_scale_groups(cfg)) + ")"))
        del q0

    # The NEP 3-body Gram path gets the angular invariants from one GEMM per
    # centre type; cuBLAS would run it in TF32 and shift the descriptor by ~2e-4
    # (GPU ≠ CPU ≠ s path). TF32 buys nothing here (bandwidth-bound), so off.
    from nep_descriptor import nep_gram_on
    if nep_gram_on(cfg):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("NEP 3-body Gram path active (no per-candidate s tensor); "
              "TensorFloat-32 matmul disabled to keep the descriptor exact.")
    model = TNEP(cfg)
    if model._neighbor_readout:
        _c_desc = {
            "off": "c ≡ 1",
            "symmetric": "learned symmetric c[C, T(T+1)/2] (unordered pairs)",
            "ordered": "learned ordered Ω[C, T²] (asymmetric — breaks gauge "
                       "invariance across species)",
        }[model._species_pairs]
        _r_desc = {
            "screened": f"R = r^(ζ−1)·f_c (screened, ζ = {model._dipole_zeta})",
            "chebyshev": "R_n = T_n(2r/r_c − 1)·f_c (Chebyshev)",
            "chebyshev_warped": "R_n = T_n(2(r/r_c − 1)² − 1)·f_c "
                                "(warped Chebyshev)",
            "gaussian": f"R_n = exp(−(r − μ_n)²/2σ²)·f_c (Gaussian, "
                        f"σ-scale = {cfg.dipole_gauss_sigma_scale})",
        }[model._dipole_radial]
        _fc_desc = {
            "cosine2": "f_c = ((1+cos(πr/r_c))/2)²",
            "cosine": "f_c = (1+cos(πr/r_c))/2",
        }[model._dipole_cutoff]
        print(f"Dipole readout: neighbor basis — "
              f"b_ij = Σ_n c_n[t_i,t_j]·R_n(r_ij)·s_i^n (+ mirrored j term), "
              f"μ = Σ_pairs b_ij·(r_i − r_j); {_r_desc}, {_fc_desc}, "
              f"C = {model._dipole_channels} radial channels"
              + (f" × {model._bond_rank} kernel groups (isotropic, traceless)"
                 if model._bond_rank == 2 else "")
              + f", {_c_desc}, "
              f"r_c = {model._rc} Å"
              + f"; head = {model._dipole_head}")
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

    # Train, or (extract_model) rebuild the models from the checkpoint's μ / best_μ.
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
            # Checkpoint cfg disagrees with the save-time cfg on an architecture field.
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

        # Mirror SNES.fit's final restoration: in-place model/optimizer = best-val.
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
        # Filename = {dataset}_{elements}_{mode}_{label}.h5 with the test
        # metrics in the label, e.g. ..._best_val_r2_0.9987_rmse_0.0021.h5
        def _label(tag, m):
            return f"{tag}_r2_{m['r2']:.4f}_rmse_{m['rmse']:.4g}"
        save_model(best_val_model, cfg, cfg.save_path, label=_label("best_val", metrics))
        save_model(final_model, cfg, cfg.save_path, label=_label("final_gen", final_metrics))
        save_history(history, cfg)

    # Timing summary
    timing = history.get("timing", {})
    if timing:
        phases = ["sample_batch", "evaluate", "rank_update", "validate", "overhead"]
        grand = sum(sum(timing[p]) for p in phases)
        # Sampled once per val_interval: n_recorded is the val-tick count.
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
    """Score a trained model on an external .xyz dataset and plot.
    Returns (metrics dict, predictions [S, T])."""
    cfg = model.cfg
    dataset = read(data_path, index=":")
    print(f"Loaded {len(dataset)} structures from {data_path}")

    data = prepare_eval_data(dataset, cfg)

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
    """Predict along an MD trajectory in batches (peak memory O(batch_size))
    and compute spectra: mode 1 dipoles → IR, mode 2 polarizabilities → Raman.

    pin_to_cpu: batch tensors on CPU (needed when a batch's COO exceeds VRAM).
    descriptor_mode: 0 = quippy CPU, 1 = native TF/GPU (None = cfg).
    descriptor_batch_frames: frames per TF graph call (1 = lowest memory; None
    auto-sizes to descriptor_memory_budget_bytes, default 6 GiB).
    descriptor_precision: "float64" (mirrors Fortran) or "float32" (~2× faster,
    ~½ VRAM); outputs are always float32.

    Returns:
        Mode 1: dict(dipoles, freq_cm, intensity, power, acf)
        Mode 2: dict(polarizabilities, freq_cm, I_VV, I_VH, I_iso, I_aniso)
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

    # One builder reused across batches (quippy is expensive to construct).
    builder = make_descriptor_builder(cfg, mode=descriptor_mode)
    # Precision override used by mode-1 builder only (quippy ignores it).
    _resolved_precision = (descriptor_precision
                           if descriptor_precision is not None
                           else getattr(cfg, "descriptor_precision", "float64"))
    print(f"Descriptor backend: {type(builder).__name__}  "
          f"(precision: {_resolved_precision})")

    # Stream batches (build → predict → drop); a 2-deep prefetch thread overlaps
    # iread parsing with GPU compute, memory ~3 × batch_size.
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

    # Always save outputs (.npy + .txt) so long inference isn't lost when plotting is off.
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
        freq_cm, I_VV, I_VH, I_iso, I_aniso = compute_raman_spectrum(
            pols, dt_fs=dt_fs, smooth_k=ir_smooth, smooth_kind=ir_smooth_kind)
        model_label = os.path.splitext(os.path.basename(
            getattr(cfg, "save_path", "") or "model"))[0]
        plot_raman_spectrum(freq_cm, I_VV, I_VH, I_iso, I_aniso, cfg, save_plots, show_plots,
                            trajectory_path=trajectory_path, model_label=model_label)
        return {"polarizabilities": pols, "freq_cm": freq_cm,
                "I_VV": I_VV, "I_VH": I_VH, "I_iso": I_iso, "I_aniso": I_aniso}


if __name__ == '__main__':
    model = train_model()
