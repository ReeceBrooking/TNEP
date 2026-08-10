from __future__ import annotations

import json
import numpy as np
import os
import h5py
from TNEPconfig import TNEPconfig
from TNEP import TNEP


# Backward-compat: if a saved model predates a config field, the live
# TNEPconfig class default may have drifted since it was trained, silently
# changing behaviour on load. For each field here, when the saved config
# lacks it, restore the value it implicitly had before the field existed.
# Fields whose default already matches are kept listed so a future
# default-flip can't silently regress old loads.
# (The `_cr_*` CR-FM-NES keys are deliberately NOT here — they're `snes`
# group datasets, not config fields; their absence just means the model
# wasn't trained with that feature.)
_LEGACY_FIELD_DEFAULTS: dict[str, object] = {
    # Pre-existence: only the N=2 dipole path existed (Xu et al. JCTC 2024);
    # class default is now 0.
    "dipole_rij_power": 2,
    # Architecture-affecting fields (silent-mismatch danger zone): a
    # pre-mixing model has no U_pair, so letting the True default through
    # would build a mixing layer with random weights and mis-predict.
    "descriptor_mixing": False,
    "descriptor_mixing_per_type": False,
    "descriptor_mixing_separate_pol": False,
    "descriptor_mixing_regularizer": "off",
    # Pre-existence: no radial enhancement (class default now 1).
    "radial_enhancement": 0,
    # Pre-existence: no preprocess contraction — W0 lives at raw Q.
    "descriptor_preprocess_contract": "off",
    "descriptor_preprocess_angular_l_keep": 1,
    "descriptor_preprocess_per_type": True,
    "descriptor_preprocess_init": "mean",
    "descriptor_preprocess_lambda_1": 0.0,
    "descriptor_preprocess_lambda_2": 0.0,
    "preprocess_sigma_scale": 1.0,
    # Pre-existence: not present; None preserves Q_raw (nep4_radial mode only).
    "descriptor_nep4_n_max_out": None,
    # Pre-existence: no search preconditioning — uniform σ, μ holds W0 itself.
    "descriptor_sigma_scaling": "off",
    "descriptor_weight_reparam": "off",
    "descriptor_scaling_exponent": 0.5,
    "descriptor_scaling_clamp": 64.0,
}


def _apply_legacy_field_defaults(cfg: TNEPconfig,
                                 saved_keys) -> None:
    """Set every `_LEGACY_FIELD_DEFAULTS` field absent from `saved_keys`
    to its pre-existence value on `cfg`; genuine saved values are untouched.
    """
    saved = set(saved_keys)
    for field, legacy in _LEGACY_FIELD_DEFAULTS.items():
        if field not in saved:
            setattr(cfg, field, legacy)


def setup_run_directory(cfg: TNEPconfig) -> str:
    """Create a timestamped run directory (with plots/ and a human-readable
    config.txt) and set cfg.save_path/save_plots in place.

    The base directory is taken from cfg.save_path — the timestamped run dir is
    created under `dirname(cfg.save_path)`, so save_path='myruns/auto' saves under
    myruns/, and the default 'models/auto' saves under models/. Requires cfg.dim_q
    (call after descriptor building). Returns the run dir path.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pop = cfg.pop_size if cfg.pop_size is not None else "auto"
    dir_name = f"n{cfg.num_neurons}_q{cfg.dim_q}_pop{pop}_{timestamp}"
    # Honor the user's chosen base directory (the part before the trailing
    # 'auto'); fall back to models/ when save_path has no directory component.
    base_dir = os.path.dirname(cfg.save_path) if cfg.save_path else ""
    run_dir = os.path.join(base_dir or "models", dir_name)
    plots_dir = os.path.join(run_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    # Write config as human-readable text
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"# TNEPconfig — {timestamp}\n")
        f.write(f"# Run directory: {run_dir}\n\n")
        for k in sorted(vars(TNEPconfig)):
            if k.startswith('_'):
                continue
            default = getattr(TNEPconfig, k, None)
            if callable(default):
                continue
            actual = getattr(cfg, k, default)
            if isinstance(actual, np.ndarray):
                f.write(f"{k} = ndarray shape={actual.shape} dtype={actual.dtype}\n")
            else:
                f.write(f"{k} = {actual!r}\n")

    # Update cfg so save_model and plotting use this directory
    cfg.save_path = os.path.join(run_dir, "auto")
    cfg.save_plots = plots_dir

    print(f"Run directory: {run_dir}")
    return run_dir


def _z_to_symbol(z: int) -> str:
    from ase.data import chemical_symbols
    return chemical_symbols[z]


def _generate_model_filename(cfg: TNEPconfig) -> str:
    """Generate a model filename: {dataset}_{elements}_{mode}.h5"""
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    dataset_name = os.path.splitext(os.path.basename(cfg.data_path))[0]
    elements = "_".join(_z_to_symbol(z) for z in cfg.types)
    return f"{dataset_name}_{elements}_{mode}.h5"


def _serialize_config(cfg: TNEPconfig) -> dict:
    """Convert TNEPconfig to a JSON-serialisable dict.

    Walks `__annotations__` (not just `vars(cfg)`) so fields left at their
    class default are captured too — otherwise a restore would pick up the
    current class default and mismatch the saved architecture/μ.
    """
    # Annotated class fields + any runtime extras stashed on the instance
    # (type_map, indices, dim_q, ... — set at data-load time, no class default).
    field_names = set(getattr(type(cfg), "__annotations__", {}).keys())
    field_names.update(k for k in vars(cfg).keys() if not k.startswith("_"))

    config_dict = {}
    for k in sorted(field_names):
        if k.startswith('_'):
            continue
        if not hasattr(cfg, k):
            continue
        v = getattr(cfg, k)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.integer, np.bool_)):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, dict):
            v = {(int(dk) if isinstance(dk, np.integer) else dk):
                 (int(dv) if isinstance(dv, np.integer) else
                  float(dv) if isinstance(dv, np.floating) else dv)
                 for dk, dv in v.items()}
        elif isinstance(v, list):
            v = [int(x) if isinstance(x, np.integer) else
                 float(x) if isinstance(x, np.floating) else x
                 for x in v]
        config_dict[k] = v
    return config_dict


def save_model(model: TNEP, cfg: TNEPconfig, path: str | None = None,
               label: str | None = None) -> None:
    """Save trained TNEP weights + config to an HDF5 (.h5) file.

    Layout: top-level attrs (target_mode, num_types, num_neurons, dim_q,
    elements); /weights (W0,b0,W1,b1, + pol variants for mode 2); /descriptor
    (z_to_type_index); /config (TNEPconfig as JSON string).

    Cayley note: under descriptor_mixing_regularizer=="cayley" the saved
    U_pair holds the dense reconstructed V, not the skew-symmetric A SNES
    searched — fine for inference, but Cayley fine-tuning must resume from the
    .h5_checkpoint (which persists A in μ), since A is unrecoverable from V.

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : output file path. None or ending "auto" = auto-generate.
        label : optional suffix before .h5 (e.g. "best_val", "final_gen")
    """
    if path is None or path.endswith("auto") or os.path.isdir(path):
        if path and os.path.isdir(path):
            directory = path
        else:
            directory = (os.path.dirname(path)
                         if path and os.path.dirname(path) else ".")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _generate_model_filename(cfg))

    if label:
        base, ext = os.path.splitext(path)
        path = f"{base}_{label}{ext}"

    config_dict = _serialize_config(cfg)
    z_to_type_index = np.array(
        [[z, idx] for idx, z in enumerate(cfg.types)], dtype=np.int32)

    with h5py.File(path, "w") as f:
        # Top-level metadata — visible via `h5ls -v model.h5` without loading weights
        f.attrs["target_mode"] = cfg.target_mode
        f.attrs["num_types"] = cfg.num_types
        f.attrs["num_neurons"] = cfg.num_neurons
        f.attrs["dim_q"] = cfg.dim_q
        f.attrs["elements"] = np.asarray(cfg.types, dtype=np.int32)

        # Weights
        wg = f.create_group("weights")
        wg.create_dataset("W0", data=model.W0.numpy())
        wg.create_dataset("b0", data=model.b0.numpy())
        wg.create_dataset("W1", data=model.W1.numpy())
        wg.create_dataset("b1", data=model.b1.numpy())
        if cfg.target_mode == 2:
            wg.create_dataset("W0_pol", data=model.W0_pol.numpy())
            wg.create_dataset("b0_pol", data=model.b0_pol.numpy())
            wg.create_dataset("W1_pol", data=model.W1_pol.numpy())
            wg.create_dataset("b1_pol", data=model.b1_pol.numpy())
        # Optional descriptor-mixing layer, stored only when trained with
        # cfg.descriptor_mixing=True. Dataset "U_pair" holds the residual
        # V = U - I; loaders fall back to V=0 (U_full=I, no-op) when absent.
        if getattr(model, "descriptor_mixing", False) and model.U_pair is not None:
            wg.create_dataset("U_pair", data=model.U_pair.numpy())
            # Second rotation for the pol scalar ANN
            # (descriptor_mixing_separate_pol); absent otherwise.
            if getattr(model, "U_pair_pol", None) is not None:
                wg.create_dataset("U_pair_pol",
                                  data=model.U_pair_pol.numpy())

        # Optional preprocess contraction tail, stored only when trained with
        # cfg.descriptor_preprocess_contract != "off". Loaders fall back to
        # None (W_pre kept at init values) when absent.
        if (getattr(model, "descriptor_preprocess_contract", "off") != "off"
                and getattr(model, "W_pre_angular", None) is not None):
            wg.create_dataset(
                "W_pre_angular", data=model.W_pre_angular.numpy())
            wg.attrs["preprocess_contract"] = str(
                model.descriptor_preprocess_contract)

        # Descriptor metadata
        dg = f.create_group("descriptor")
        dg.create_dataset("z_to_type_index", data=z_to_type_index)
        # Channel multiplier for either preconditioning mode. The channel
        # statistics it was derived from are not serialised, so without this
        # a preconditioned model could not be reconstructed at all.
        if getattr(model.optimizer, "_chan_mult", None) is not None:
            dg.create_dataset("channel_scale",
                              data=model.optimizer._chan_mult)

        # Full config as JSON string
        f.create_dataset("config", data=json.dumps(config_dict))

    print(f"Model saved to {path}")


def save_history(history: dict, cfg: TNEPconfig) -> None:
    """Write training history to history.csv in the run directory.

    Always-on cols: generation, train_loss, val_loss, L1, L2, best_rmse,
    worst_rmse, sigma_{min,max,mean,median}. Optional (if present):
    best_rrmse, avg_rrmse, L_orth. Short columns are NaN-padded so every
    row has a value (a resume from an older checkpoint may lack new keys).
    """
    run_dir = os.path.dirname(cfg.save_path) if cfg.save_path else "."
    path = os.path.join(run_dir, "history.csv")

    base_cols = [
        "generation", "train_loss", "val_loss",
        "L1", "L2", "best_rmse", "worst_rmse",
        "sigma_min", "sigma_max", "sigma_mean", "sigma_median",
    ]
    optional_cols = ["best_rrmse", "avg_rrmse", "L_orth"]
    cols = base_cols + [c for c in optional_cols if c in history]
    int_cols = {"generation"}

    n_rows = len(history["generation"])
    # Pad short columns with NaN (resumed histories may have new keys
    # that only started accumulating partway through).
    padded = {}
    for c in cols:
        col = history.get(c, [])
        if len(col) < n_rows:
            col = [float("nan")] * (n_rows - len(col)) + list(col)
        padded[c] = col

    def _fmt(c: str, val) -> str:
        if c in int_cols:
            return str(int(val))
        try:
            f = float(val)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if f != f else f"{f:.6g}"  # NaN-safe

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(n_rows):
            f.write(",".join(_fmt(c, padded[c][i]) for c in cols) + "\n")
    print(f"History saved to {path}")


def save_checkpoint(path: str, cfg: TNEPconfig, state: dict,
                    history: dict, last_gen: int) -> None:
    """Write a rolling training checkpoint (config, descriptor map, SNES
    state, and history) at `path`, replacing any existing file atomically.

    `state` keys: mu, sigma (current SNES distribution); best_mu, best_sigma
    (best-val params); best_val_loss (float); gens_without_improvement (int);
    tf_rng_state (optional Generator state).
    """
    config_dict = _serialize_config(cfg)
    z_to_type_index = np.array(
        [[z, idx] for idx, z in enumerate(cfg.types)], dtype=np.int32)

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        # Inspection attrs (h5ls-friendly)
        f.attrs["target_mode"] = cfg.target_mode
        f.attrs["num_types"] = cfg.num_types
        f.attrs["num_neurons"] = cfg.num_neurons
        f.attrs["dim_q"] = cfg.dim_q
        f.attrs["elements"] = np.asarray(cfg.types, dtype=np.int32)
        f.attrs["last_gen"] = int(last_gen)
        f.attrs["num_generations"] = int(cfg.num_generations)
        f.create_dataset("config", data=json.dumps(config_dict))
        f.create_dataset("descriptor/z_to_type_index", data=z_to_type_index)
        # SNES state
        sg = f.create_group("snes")
        sg.create_dataset("mu",         data=_np(state["mu"]))
        sg.create_dataset("sigma",      data=_np(state["sigma"]))
        sg.create_dataset("best_mu",    data=_np(state["best_mu"]))
        # best_sigma omitted under cov_mode="crfmnes" (scale lives in _cr_sig,
        # no snapshot); vanilla SNES always includes it.
        if state.get("best_sigma") is not None:
            sg.create_dataset("best_sigma", data=_np(state["best_sigma"]))
        sg.attrs["best_val_loss"] = float(state["best_val_loss"])
        sg.attrs["gens_without_improvement"] = int(state["gens_without_improvement"])
        # Descriptor channel multiplier. Mandatory under
        # descriptor_weight_reparam (μ holds W0_hat, so the effective W0 is
        # unrecoverable without it); also saved under sigma-scaling so the
        # resumed run reuses the exact multiplier instead of re-deriving it.
        if state.get("descriptor_scale") is not None:
            sg.create_dataset("descriptor_scale",
                              data=np.asarray(state["descriptor_scale"],
                                              dtype=np.float32))
        rng = state.get("tf_rng_state")
        if rng is not None:
            sg.create_dataset("rng_state", data=_np(rng))
        # CR-FM-NES learned state (guarded — absent for pure-SNES checkpoints).
        if state.get("cr_v") is not None:
            sg.create_dataset("cr_v",   data=_np(state["cr_v"]))
            sg.create_dataset("cr_D",   data=_np(state["cr_D"]))
            sg.create_dataset("cr_psg", data=_np(state["cr_psg"]))
            sg.create_dataset("cr_pc",  data=_np(state["cr_pc"]))
            sg.attrs["cr_sig"] = float(state["cr_sig"])
        # History (so plots / early-stop continuity carry over)
        hg = f.create_group("history")
        for k, v in history.items():
            if k == "timing":
                tg = hg.create_group("timing")
                for tk, tv in v.items():
                    tg.create_dataset(tk, data=np.asarray(tv, dtype=np.float64))
            else:
                hg.create_dataset(k, data=np.asarray(v))
    os.replace(tmp, path)


def load_checkpoint(path: str) -> tuple[TNEPconfig, dict]:
    """Load a training checkpoint. Returns `(cfg, resume_state)`: `cfg` is
    the fully-restored config (architecture, indices, run params) exactly as
    when written, and `resume_state` carries the SNES + history fields to
    continue from `last_gen + 1`. Any caller-passed cfg is ignored.
    """
    cfg = TNEPconfig()
    with h5py.File(path, "r") as f:
        config_dict = json.loads(f["config"][()])
        for k, v in config_dict.items():
            if k in ("descriptor_mean", "type_map"):
                # Legacy / re-derived below.
                continue
            setattr(cfg, k, v)
        _apply_legacy_field_defaults(cfg, config_dict.keys())
        # Enforce Python int types (belt-and-braces for old checkpoints).
        if hasattr(cfg, "types"):
            cfg.types = [int(z) for z in cfg.types]
        # indices come back from json as a list — coerce to ndarray.
        if isinstance(getattr(cfg, "indices", None), list):
            cfg.indices = np.asarray(cfg.indices, dtype=np.int64)
        cfg.type_map = {int(row[0]): int(row[1])
                         for row in f["descriptor/z_to_type_index"][:]}

        last_gen = int(f.attrs["last_gen"])
        sg = f["snes"]
        resume_state = {
            "mu":         sg["mu"][:],
            "sigma":      sg["sigma"][:],
            "best_mu":    sg["best_mu"][:],
            # best_sigma absent under cov_mode="crfmnes"; resume falls back to
            # tf.identity(self.sigma).
            "best_sigma": sg["best_sigma"][:] if "best_sigma" in sg else None,
            "best_val_loss":            float(sg.attrs["best_val_loss"]),
            "gens_without_improvement": int(sg.attrs["gens_without_improvement"]),
            "rng_state":  sg["rng_state"][:] if "rng_state" in sg else None,
            "last_gen":   last_gen,
        }
        if "cr_v" in sg:
            resume_state["cr_v"]   = sg["cr_v"][:]
            resume_state["cr_D"]   = sg["cr_D"][:]
            resume_state["cr_psg"] = sg["cr_psg"][:]
            resume_state["cr_pc"]  = sg["cr_pc"][:]
            resume_state["cr_sig"] = float(sg.attrs["cr_sig"])
        if "descriptor_scale" in sg:
            cfg.descriptor_scale = sg["descriptor_scale"][:].astype(np.float32)
        elif str(getattr(cfg, "descriptor_weight_reparam", "off")) != "off":
            raise ValueError(
                f"{path!r} was trained with descriptor_weight_reparam="
                f"{cfg.descriptor_weight_reparam!r} but has no "
                f"/snes/descriptor_scale. mu holds the reparameterised W0, so "
                f"resuming without the multiplier would train a different "
                f"model.")
        hg = f["history"]
        history = {}
        for k in hg:
            if k == "timing":
                history["timing"] = {tk: list(hg["timing"][tk][:])
                                      for tk in hg["timing"]}
            else:
                history[k] = list(hg[k][:])
    # NaN-back-pad metric keys added since the checkpoint was written
    # (best_rrmse, avg_rrmse, L_orth) to the generation length, else
    # downstream zip() against generation silently truncates.
    n_rows = len(history.get("generation", []))
    for k in ("best_rrmse", "avg_rrmse", "L_orth"):
        if k not in history:
            history[k] = [float("nan")] * n_rows
        elif len(history[k]) < n_rows:
            history[k] = ([float("nan")] * (n_rows - len(history[k]))
                          + list(history[k]))
    resume_state["history"] = history
    return cfg, resume_state


def _load_weights(model: TNEP, cfg: TNEPconfig, W0, b0, W1, b1,
                  W0_pol=None, b0_pol=None, W1_pol=None, b1_pol=None,
                  U_pair=None, U_pair_pol=None,
                  W_pre_angular=None) -> None:
    model.W0.assign(W0)
    model.b0.assign(b0)
    model.W1.assign(W1)
    model.b1.assign(b1)
    if cfg.target_mode == 2:
        model.W0_pol.assign(W0_pol)
        model.b0_pol.assign(b0_pol)
        model.W1_pol.assign(W1_pol)
        model.b1_pol.assign(b1_pol)
    # Optional V_pair restore (h5 dataset "U_pair" holds V = U - I). When
    # absent (pre-mixing or mixing-disabled runs), keep TNEP.__init__'s
    # zero-init V (U_full = I, no-mixing path).
    if (U_pair is not None
            and getattr(model, "descriptor_mixing", False)
            and model.U_pair is not None):
        if tuple(U_pair.shape) != tuple(model.U_pair.shape):
            # Hard fail rather than silently dropping learned U_pair (which
            # would masquerade as training from scratch).
            raise ValueError(
                f"saved U_pair shape {tuple(U_pair.shape)} != "
                f"model.U_pair shape {tuple(model.U_pair.shape)}. "
                f"The descriptor-mixing layout (alpha_max / l_max / "
                f"per_type) likely changed between save and load. "
                f"Re-train from scratch, or rebuild the cfg to match "
                f"the saved model.")
        model.U_pair.assign(U_pair)
    # Optional second rotation (descriptor_mixing_separate_pol). Absent in
    # shared-rotation checkpoints → model.U_pair_pol is None too.
    if (U_pair_pol is not None
            and getattr(model, "U_pair_pol", None) is not None):
        if tuple(U_pair_pol.shape) != tuple(model.U_pair_pol.shape):
            raise ValueError(
                f"saved U_pair_pol shape {tuple(U_pair_pol.shape)} != "
                f"model.U_pair_pol shape {tuple(model.U_pair_pol.shape)}. "
                f"The descriptor-mixing layout (alpha_max / l_max / "
                f"per_type) likely changed between save and load. "
                f"Re-train from scratch, or rebuild the cfg to match "
                f"the saved model.")
        model.U_pair_pol.assign(U_pair_pol)
    # Optional W_pre_angular restore. Absent in pre-preprocess checkpoints
    # → keep init-time values.
    if (W_pre_angular is not None
            and getattr(model, "descriptor_preprocess_contract", "off") != "off"
            and getattr(model, "W_pre_angular", None) is not None):
        if tuple(W_pre_angular.shape) != tuple(model.W_pre_angular.shape):
            raise ValueError(
                f"saved W_pre_angular shape {tuple(W_pre_angular.shape)} != "
                f"model.W_pre_angular shape {tuple(model.W_pre_angular.shape)}. "
                f"descriptor_preprocess_contract mode or alpha_max likely "
                f"changed between save and load.")
        model.W_pre_angular.assign(W_pre_angular)


def _print_load_summary(path: str, cfg: TNEPconfig) -> None:
    from ase.data import chemical_symbols
    type_str = ", ".join(f"{chemical_symbols[z]}(Z={z})→{idx}"
                         for z, idx in cfg.type_map.items())
    print(f"Model loaded from {path}")
    print(f"  target_mode={cfg.target_mode}, dim_q={cfg.dim_q}, "
          f"num_types={cfg.num_types}")
    print(f"  Type mapping: {type_str}")
    if getattr(cfg, "descriptor_mixing", False):
        print(f"  Descriptor mixing: arch=l_aware, "
              f"per_type={getattr(cfg, 'descriptor_mixing_per_type', False)}")


def _load_model_h5(path: str) -> TNEP:
    cfg = TNEPconfig()

    # Read everything before constructing TNEP (which initialises quippy
    # descriptors) so the file handle closes early.
    with h5py.File(path, "r") as f:
        config_dict = json.loads(f["config"][()])

        cfg.type_map = {int(row[0]): int(row[1])
                        for row in f["descriptor/z_to_type_index"][:]}

        # Search-preconditioning channel multiplier (absent unless the model
        # was trained with one). Read here because the handle closes below.
        dg = f["descriptor"]
        channel_scale = (dg["channel_scale"][:].astype(np.float32)
                         if "channel_scale" in dg else None)

        wg = f["weights"]
        weights = {
            "W0": wg["W0"][:], "b0": wg["b0"][:],
            "W1": wg["W1"][:], "b1": wg["b1"][()],
            "W0_pol": wg["W0_pol"][:] if "W0_pol" in wg else None,
            "b0_pol": wg["b0_pol"][:] if "b0_pol" in wg else None,
            "W1_pol": wg["W1_pol"][:] if "W1_pol" in wg else None,
            # b1_pol is a scalar dataset like b1, so it needs [()]; [:] raises
            # "Illegal slicing argument for scalar dataspace" and made every
            # target_mode=2 model unloadable.
            "b1_pol": wg["b1_pol"][()] if "b1_pol" in wg else None,
            "U_pair": wg["U_pair"][:] if "U_pair" in wg else None,
            "U_pair_pol": (wg["U_pair_pol"][:]
                           if "U_pair_pol" in wg else None),
            "W_pre_angular": (wg["W_pre_angular"][:]
                              if "W_pre_angular" in wg else None),
        }

    for k, v in config_dict.items():
        if k == "descriptor_mean":
            # Legacy field — descriptor scaling was removed; ignore.
            continue
        if k == "type_map":
            # JSON gives str keys; the authoritative int-keyed type_map was
            # already built from descriptor/z_to_type_index above.
            continue
        setattr(cfg, k, v)
    _apply_legacy_field_defaults(cfg, config_dict.keys())
    # Must precede TNEP(cfg): the channel statistics are not serialised, so
    # SNES can only rebuild the multiplier from this restored copy.
    if channel_scale is not None:
        cfg.descriptor_scale = channel_scale

    model = TNEP(cfg)
    _load_weights(model, cfg, **weights)

    _print_load_summary(path, cfg)
    return model


def _load_model_npz(path: str) -> TNEP:
    """Legacy loader for .npz checkpoints."""
    data = np.load(path, allow_pickle=True)
    cfg = TNEPconfig()

    if "config_json" in data:
        config_dict = json.loads(str(data["config_json"]))
        for k, v in config_dict.items():
            if k == "descriptor_mean":
                continue  # legacy: ignore
            setattr(cfg, k, v)
        _apply_legacy_field_defaults(cfg, config_dict.keys())
    else:
        cfg.num_types = int(data["num_types"])
        cfg.num_neurons = int(data["num_neurons"])
        cfg.dim_q = int(data["dim_q"])
        cfg.types = data["types"].tolist()
        cfg.target_mode = int(data["target_mode"])
        cfg.l_max = int(data["l_max"])
        cfg.alpha_max = int(data["alpha_max"])
        cfg.activation = str(data["activation"])
        cfg.data_path = str(data["data_path"])
        if "rc" in data:
            rc = float(data["rc"])
            cfg.rcut_hard = rc
            cfg.rcut_soft = rc - 0.5
        # No config_json → no field saved, so every legacy default applies.
        _apply_legacy_field_defaults(cfg, [])

    cfg.type_map = {int(row[0]): int(row[1]) for row in data["z_to_type_index"]}

    # Cross-check descriptor_mixing vs saved weights: mismatch (mixing on
    # but no U_pair, or vice versa) fails loud instead of silently
    # mis-predicting with random weights.
    has_U_pair = "U_pair" in data.files
    if bool(getattr(cfg, "descriptor_mixing", False)) and not has_U_pair:
        raise ValueError(
            f"Checkpoint at {path!r} has cfg.descriptor_mixing=True "
            f"but the npz has no U_pair tensor. The mixing layer's "
            f"weights are missing — loading would build it with random "
            f"Glorot/zeros and silently mis-predict. Either this save "
            f"predates the mixing layer (set cfg.descriptor_mixing="
            f"False before loading) or the save is corrupt.")
    if has_U_pair and not bool(getattr(cfg, "descriptor_mixing", False)):
        raise ValueError(
            f"Checkpoint at {path!r} has a U_pair tensor but "
            f"cfg.descriptor_mixing is False. The trained mixing layer "
            f"would be discarded silently. Set cfg.descriptor_mixing="
            f"True (and the matching arch) before loading, or use the "
            f"h5 loader which records the arch alongside U_pair.")

    model = TNEP(cfg)
    _load_weights(
        model, cfg,
        data["W0"], data["b0"], data["W1"], data["b1"],
        data.get("W0_pol"), data.get("b0_pol"),
        data.get("W1_pol"), data.get("b1_pol"),
        U_pair=(data["U_pair"] if has_U_pair else None),
        W_pre_angular=data.get("W_pre_angular"),
    )

    _print_load_summary(path, cfg)
    return model


def load_model(path: str) -> TNEP:
    """Load a trained TNEP model from an HDF5 (.h5) or legacy NumPy (.npz) file.

    Args:
        path : path to saved model file

    Returns:
        model : TNEP model with loaded weights and reconstructed TNEPconfig
    """
    if path.endswith(".npz"):
        return _load_model_npz(path)
    return _load_model_h5(path)
