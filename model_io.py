from __future__ import annotations

import json
import numpy as np
import os
import h5py
from TNEPconfig import TNEPconfig
from TNEP import TNEP


def setup_run_directory(cfg: TNEPconfig) -> str:
    """Create a run directory under models/ and configure cfg paths.

    Directory structure:
        models/
            n{neurons}_q{dim_q}_pop{pop_size}_{YYYYMMDD_HHMMSS}/
                plots/
                config.txt
                (model .h5 saved here after training)

    Requires cfg.dim_q to be set (call after descriptor building).
    Updates cfg.save_path and cfg.save_plots in place.

    Returns:
        run_dir : str — path to the created run directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pop = cfg.pop_size if cfg.pop_size is not None else "auto"
    dir_name = f"n{cfg.num_neurons}_q{cfg.dim_q}_pop{pop}_{timestamp}"
    run_dir = os.path.join("models", dir_name)
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

    Walks `TNEPconfig.__annotations__` rather than `vars(cfg)` so that
    fields whose value matches the class-level default are *also*
    captured. Using `vars(cfg)` alone would miss any cfg field that
    was never explicitly written to the instance — which is the common
    case for fields that just use their default. Without this, a
    checkpoint saved when the class default for e.g. `num_neurons`
    was 30 would silently restore as whatever the current class
    default is (e.g. 50), giving an architectural mismatch with the
    saved μ.
    """
    # Annotated fields from TNEPconfig (class scope) + any extras the
    # caller has stashed on the instance (runtime fields like
    # `type_map`, `indices`, `dim_q`, etc., which are populated at
    # data-load time and have no class-level default).
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
    """Save trained TNEP model weights and config to an HDF5 (.h5) file.

    File layout:
        /                       — top-level attributes: target_mode, num_types,
                                  num_neurons, dim_q, elements (quick inspection)
        /weights/               — W0, b0, W1, b1 (+ pol variants for mode 2)
        /descriptor/            — z_to_type_index
        /config                 — full TNEPconfig serialised as JSON string

    Load with:
        import h5py
        with h5py.File('model.h5', 'r') as f:
            W0 = f['weights/W0'][:]
            cfg_dict = json.loads(f['config'][()])

    Note on Cayley parameterisation: when `cfg.descriptor_mixing_regularizer
    == "cayley"`, the saved `/weights/U_pair` dataset holds the DENSE
    reconstructed V (= U_cayley − I), NOT the upper-triangle skew-
    symmetric A that SNES was searching over. Loading the model for
    inference reuses the dense V directly — no Cayley re-derivation
    occurs. Consequence: a Cayley-trained model file is essentially
    a `linear`/`l_aware`/`cross_pair_l` model at inference time, and
    further fine-tuning under Cayley is NOT possible from this file
    alone (the upper-triangle A is unrecoverable from the dense V
    without inverting the Cayley map). To continue Cayley training,
    resume from the `.h5_checkpoint` instead, where the SNES μ vector
    (which holds A) is persisted.

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : output file path. None or ending "auto" = auto-generate.
        label : optional suffix before .h5 (e.g. "best_val", "final_gen")
    """
    if path is None or path.endswith("auto"):
        directory = os.path.dirname(path) if path and os.path.dirname(path) else "."
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
        # Store as a native int array attribute — h5py handles this
        # directly (visible via `h5ls -v model.h5`), avoiding the JSON
        # round-trip that the full `config` dataset below needs for
        # heterogeneous dict serialisation.
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
        # Optional descriptor-mixing layer. Stored only when the model
        # was trained with cfg.descriptor_mixing=True. The dataset
        # named "U_pair" holds the residual V = U - I (the internal
        # parameterisation); loaders fall back to V=0 (so U_full=I,
        # a no-op mixing) when absent. The `mixing_arch` attribute
        # disambiguates the shape:
        #   "linear"  : [num_pairs, max_bs, max_bs] (or +T leading)
        #   "l_aware" : [num_pairs, L, max_α, max_α] (or +T leading)
        # NOTE: checkpoints from before the V-residual switch stored
        # identity-init U_pair; reloading those will be interpreted
        # as V=I → U_full=2I and produce wrong predictions.
        if getattr(model, "descriptor_mixing", False) and model.U_pair is not None:
            wg.create_dataset("U_pair", data=model.U_pair.numpy())
            wg.attrs["mixing_arch"] = getattr(
                model, "descriptor_mixing_arch", "linear")

        # Per-channel descriptor scaler (cfg.descriptor_scaling="q_scaler").
        # Persisted alongside W0 etc. so inference scripts can replay
        # the same scaling at descriptor-build time without recomputing.
        # The scaler array is too long for the JSON config; store it as
        # a dedicated float32 dataset.
        if (str(getattr(cfg, "descriptor_scaling", "none")) != "none"
                and getattr(cfg, "_q_scaler", None) is not None):
            wg.create_dataset(
                "q_scaler",
                data=np.asarray(cfg._q_scaler, dtype=np.float32))
            wg.attrs["descriptor_scaling"] = str(cfg.descriptor_scaling)

        # Per-component target mean (cfg.target_centering=True). Stored
        # alongside the model so inference adds it back to predictions
        # to restore original-unit values.
        if (bool(getattr(cfg, "target_centering", False))
                and getattr(cfg, "_target_mean", None) is not None):
            wg.create_dataset(
                "target_mean",
                data=np.asarray(cfg._target_mean, dtype=np.float32))
            wg.attrs["target_centering"] = True

        # Descriptor metadata
        dg = f.create_group("descriptor")
        dg.create_dataset("z_to_type_index", data=z_to_type_index)

        # Full config as JSON string
        f.create_dataset("config", data=json.dumps(config_dict))

    print(f"Model saved to {path}")


def save_history(history: dict, cfg: TNEPconfig) -> None:
    """Write training history to history.csv in the run directory.

    Always-on columns:
        generation, train_loss, val_loss, L1, L2,
        best_rmse, worst_rmse, sigma_min, sigma_max, sigma_mean, sigma_median.

    Optional columns (written when present in history):
        best_rrmse, avg_rrmse, L_orth.

    Columns shorter than `len(history["generation"])` are padded with
    NaN so all rows have a value for every column (this happens after
    a resume that loads an older checkpoint missing the new keys).
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
    """Write a rolling training checkpoint at `path`. Atomically
    overwrites any existing checkpoint at the same path so a
    half-written file can never confuse the loader.

    `state` keys:
        mu, sigma                 : tf.Variable / np.ndarray — current SNES distribution
        best_mu, best_sigma       : tf.Tensor / np.ndarray — best-val params seen
        best_val_loss             : float
        gens_without_improvement  : int
        tf_rng_state              : tf.Tensor / np.ndarray (optional) — Generator state
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
        # Full cfg + descriptor type map for sturdy reload
        f.create_dataset("config", data=json.dumps(config_dict))
        f.create_dataset("descriptor/z_to_type_index", data=z_to_type_index)
        # SNES state
        sg = f.create_group("snes")
        sg.create_dataset("mu",         data=_np(state["mu"]))
        sg.create_dataset("sigma",      data=_np(state["sigma"]))
        sg.create_dataset("best_mu",    data=_np(state["best_mu"]))
        sg.create_dataset("best_sigma", data=_np(state["best_sigma"]))
        sg.attrs["best_val_loss"] = float(state["best_val_loss"])
        sg.attrs["gens_without_improvement"] = int(state["gens_without_improvement"])
        rng = state.get("tf_rng_state")
        if rng is not None:
            sg.create_dataset("rng_state", data=_np(rng))
        # Per-channel descriptor scaler (frozen at training-set creation
        # time). Persist into the SNES group so load_checkpoint can
        # restore it BEFORE pad_and_stack runs again — preventing a
        # silent recompute on resume that would shift the scaler.
        if getattr(cfg, "_q_scaler", None) is not None:
            sg.create_dataset(
                "q_scaler",
                data=np.asarray(cfg._q_scaler, dtype=np.float32))
        # Per-component target mean (same restore-before-pad_and_stack
        # rationale as q_scaler — keeps the resumed run consistent).
        if getattr(cfg, "_target_mean", None) is not None:
            sg.create_dataset(
                "target_mean",
                data=np.asarray(cfg._target_mean, dtype=np.float32))
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
    """Load a training checkpoint. Returns `(cfg, resume_state)` where
    `cfg` is the fully-restored config (architecture + indices + run
    params) and `resume_state` carries the SNES + history fields needed
    to continue training from `last_gen + 1`.

    The cfg returned is identical to the one that was running when the
    checkpoint was written — the architecture-defining fields (dim_q,
    num_types, types, type_map, etc.) and the train/val split (via
    cfg.indices) come straight from the file. Any cfg passed by the
    caller of `train_model` is ignored when `checkpoint=` is set.
    """
    cfg = TNEPconfig()
    with h5py.File(path, "r") as f:
        config_dict = json.loads(f["config"][()])
        for k, v in config_dict.items():
            if k in ("descriptor_mean", "type_map"):
                # Legacy / re-derived below.
                continue
            setattr(cfg, k, v)
        # Restore types as Python ints (json may have int64 → int already
        # via _serialize_config, but enforce here for old checkpoints).
        if hasattr(cfg, "types"):
            cfg.types = [int(z) for z in cfg.types]
        # Indices come back as a list from json — coerce to ndarray so
        # downstream code that indexes with cfg.indices keeps working.
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
            "best_sigma": sg["best_sigma"][:],
            "best_val_loss":            float(sg.attrs["best_val_loss"]),
            "gens_without_improvement": int(sg.attrs["gens_without_improvement"]),
            "rng_state":  sg["rng_state"][:] if "rng_state" in sg else None,
            "last_gen":   last_gen,
        }
        # Restore the per-channel descriptor scaler if the checkpoint
        # carries one — sets cfg._q_scaler BEFORE pad_and_stack runs
        # on resume, ensuring the scaler is reused (not recomputed)
        # so the resumed run remains numerically consistent with the
        # original.
        if "q_scaler" in sg:
            cfg._q_scaler = np.asarray(sg["q_scaler"][:], dtype=np.float32)
        if "target_mean" in sg:
            cfg._target_mean = np.asarray(
                sg["target_mean"][:], dtype=np.float32)
        # Consistency check: if the saved cfg has target_centering=True
        # but no target_mean dataset was written, the resumed run would
        # silently recompute the mean from a (potentially different)
        # train split. That's a correctness hazard — refuse to load.
        if (bool(getattr(cfg, "target_centering", False))
                and getattr(cfg, "_target_mean", None) is None):
            raise ValueError(
                f"Checkpoint at {path!r} has cfg.target_centering=True "
                f"but no /snes/target_mean dataset. Cannot safely resume "
                f"— the mean would be recomputed from a different train "
                f"split. Either restore the checkpoint that has the mean "
                f"saved, or set cfg.target_centering=False to opt out.")
        hg = f["history"]
        history = {}
        for k in hg:
            if k == "timing":
                history["timing"] = {tk: list(hg["timing"][tk][:])
                                      for tk in hg["timing"]}
            else:
                history[k] = list(hg[k][:])
    # Back-pad metric keys added since the checkpoint was written
    # (best_rrmse, avg_rrmse, L_orth) with NaN to match the length of
    # the generation column. Without this, downstream consumers that
    # `zip(history["generation"], history["best_rrmse"])` silently
    # truncate to the shorter list.
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
                  U_pair=None) -> None:
    model.W0.assign(W0)
    model.b0.assign(b0)
    model.W1.assign(W1)
    model.b1.assign(b1)
    if cfg.target_mode == 2:
        model.W0_pol.assign(W0_pol)
        model.b0_pol.assign(b0_pol)
        model.W1_pol.assign(W1_pol)
        model.b1_pol.assign(b1_pol)
    # Optional V_pair restore (h5 dataset still named "U_pair" but
    # holds V = U - I internally). When absent in the checkpoint
    # (e.g. pre-mixing models, or mixing-disabled runs), the model
    # keeps whatever V_pair its TNEP.__init__ produced (zero init)
    # so U_full = I, reproducing the no-mixing path.
    if (U_pair is not None
            and getattr(model, "descriptor_mixing", False)
            and model.U_pair is not None):
        if tuple(U_pair.shape) != tuple(model.U_pair.shape):
            # Hard fail rather than silently falling back to V=0
            # (no mixing): a long restart that silently dropped the
            # learned U_pair would look like training from scratch
            # without warning anyone.
            raise ValueError(
                f"saved U_pair shape {tuple(U_pair.shape)} != "
                f"model.U_pair shape {tuple(model.U_pair.shape)}. "
                f"This usually means cfg.descriptor_mixing_arch was "
                f"changed between save and load (e.g. linear → "
                f"l_aware or vice versa). Re-train from scratch with "
                f"the new arch, or rebuild the cfg to match the "
                f"saved model's arch.")
        model.U_pair.assign(U_pair)


def _print_load_summary(path: str, cfg: TNEPconfig) -> None:
    from ase.data import chemical_symbols
    type_str = ", ".join(f"{chemical_symbols[z]}(Z={z})→{idx}"
                         for z, idx in cfg.type_map.items())
    print(f"Model loaded from {path}")
    print(f"  target_mode={cfg.target_mode}, dim_q={cfg.dim_q}, "
          f"num_types={cfg.num_types}")
    print(f"  Type mapping: {type_str}")
    if getattr(cfg, "descriptor_mixing", False):
        print(f"  Descriptor mixing: arch={cfg.descriptor_mixing_arch}, "
              f"per_type={getattr(cfg, 'descriptor_mixing_per_type', False)}")


def _load_model_h5(path: str) -> TNEP:
    cfg = TNEPconfig()

    # Read everything into memory before constructing TNEP (which initialises
    # quippy descriptors) so the file handle is closed as early as possible.
    with h5py.File(path, "r") as f:
        config_dict = json.loads(f["config"][()])

        cfg.type_map = {int(row[0]): int(row[1])
                        for row in f["descriptor/z_to_type_index"][:]}

        wg = f["weights"]
        weights = {
            "W0": wg["W0"][:], "b0": wg["b0"][:],
            "W1": wg["W1"][:], "b1": wg["b1"][()],
            "W0_pol": wg["W0_pol"][:] if "W0_pol" in wg else None,
            "b0_pol": wg["b0_pol"][:] if "b0_pol" in wg else None,
            "W1_pol": wg["W1_pol"][:] if "W1_pol" in wg else None,
            "b1_pol": wg["b1_pol"][:] if "b1_pol" in wg else None,
            "U_pair": wg["U_pair"][:] if "U_pair" in wg else None,
        }
        # mixing_arch attribute is the authoritative record of which
        # arch produced the U_pair tensor. Used below to validate
        # against the cfg-restored arch — a mismatch here is the
        # earliest reliable signal that the user changed the cfg
        # mid-restart.
        saved_mixing_arch = (str(wg.attrs["mixing_arch"])
                             if "mixing_arch" in wg.attrs else None)
        # Per-channel descriptor scaler: restore alongside weights so
        # inference scripts (process_trajectory, etc.) can apply it
        # consistently with training. cfg.descriptor_scaling carries
        # the scheme name; cfg._q_scaler carries the array.
        saved_q_scaler = (np.asarray(wg["q_scaler"][:], dtype=np.float32)
                          if "q_scaler" in wg else None)
        saved_descriptor_scaling = (
            str(wg.attrs["descriptor_scaling"])
            if "descriptor_scaling" in wg.attrs else None)
        saved_target_mean = (np.asarray(wg["target_mean"][:], dtype=np.float32)
                             if "target_mean" in wg else None)
        saved_target_centering = (
            bool(wg.attrs["target_centering"])
            if "target_centering" in wg.attrs else False)

    for k, v in config_dict.items():
        if k == "descriptor_mean":
            # Legacy field — silently ignore on load (descriptor scaling
            # has been removed from the runtime).
            continue
        if k == "type_map":
            # JSON stringifies dict keys, so the round-tripped value
            # has str keys ("6": 0) instead of int. The authoritative
            # int-keyed type_map was already built from the
            # `descriptor/z_to_type_index` dataset above — don't
            # overwrite it.
            continue
        setattr(cfg, k, v)

    # Cross-check that the saved arch matches the cfg-restored arch.
    # If they disagree, the TNEP constructor below would silently build
    # a different shape and `_load_weights` would raise on shape
    # mismatch — but the diagnostic message is more useful at this
    # point where we can name the actual mismatch.
    cfg_arch = str(getattr(cfg, "descriptor_mixing_arch", "linear"))
    if (saved_mixing_arch is not None
            and getattr(cfg, "descriptor_mixing", False)
            and saved_mixing_arch != cfg_arch):
        raise ValueError(
            f"Checkpoint at {path!r} was saved with "
            f"descriptor_mixing_arch={saved_mixing_arch!r} but the "
            f"restored cfg specifies {cfg_arch!r}. The two must match "
            f"to load weights correctly.")

    # Restore the descriptor scaler. Authoritative source is the
    # /weights/q_scaler dataset + descriptor_scaling attribute,
    # falling back to the cfg JSON field if those weren't written.
    if saved_descriptor_scaling is not None:
        cfg.descriptor_scaling = saved_descriptor_scaling
    if saved_q_scaler is not None:
        cfg._q_scaler = saved_q_scaler
    elif str(getattr(cfg, "descriptor_scaling", "none")) != "none":
        raise ValueError(
            f"Model at {path!r} has descriptor_scaling="
            f"{cfg.descriptor_scaling!r} but no /weights/q_scaler "
            f"dataset was saved. The scaler is required for consistent "
            f"inference. Retrain the model or set "
            f"cfg.descriptor_scaling='none'.")

    # Restore target centering. The mean is added back to predictions
    # at the inference boundary; without it, predictions would emerge
    # in the centered space the network was trained on.
    if saved_target_centering:
        cfg.target_centering = True
    if saved_target_mean is not None:
        cfg._target_mean = saved_target_mean
    elif bool(getattr(cfg, "target_centering", False)):
        raise ValueError(
            f"Model at {path!r} has target_centering=True but no "
            f"/weights/target_mean dataset was saved. The mean is "
            f"required to map predictions back to original units. "
            f"Retrain the model or set cfg.target_centering=False.")

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

    cfg.type_map = {int(row[0]): int(row[1]) for row in data["z_to_type_index"]}

    model = TNEP(cfg)
    _load_weights(
        model, cfg,
        data["W0"], data["b0"], data["W1"], data["b1"],
        data.get("W0_pol"), data.get("b0_pol"),
        data.get("W1_pol"), data.get("b1_pol"),
        U_pair=(data["U_pair"] if "U_pair" in data.files else None),
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
