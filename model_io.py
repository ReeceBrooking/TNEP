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
    """Convert TNEPconfig to a JSON-serialisable dict."""
    config_dict = {}
    for k, v in vars(cfg).items():
        if k.startswith('_'):
            continue
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
        # named "U_pair" actually holds the residual V = U - I (the
        # internal parameterisation); loaders fall back to V=0 (so
        # U_full=I, a no-op mixing) when absent.
        # NOTE: checkpoints from before the V-residual switch stored
        # identity-init U_pair; reloading those will be interpreted as
        # V=I → U_full=2I and produce wrong predictions.
        if getattr(model, "descriptor_mixing", False) and model.U_pair is not None:
            wg.create_dataset("U_pair", data=model.U_pair.numpy())

        # Descriptor metadata
        dg = f.create_group("descriptor")
        dg.create_dataset("z_to_type_index", data=z_to_type_index)

        # Full config as JSON string
        f.create_dataset("config", data=json.dumps(config_dict))

    print(f"Model saved to {path}")


def save_history(history: dict, cfg: TNEPconfig) -> None:
    """Write training history to history.csv in the run directory.

    CSV format: load with pd.read_csv('history.csv') or
    np.loadtxt('history.csv', delimiter=',', skiprows=1).

    Columns: generation, train_loss (RMSE + reg), val_loss, L1, L2,
             best_rmse, worst_rmse, sigma_min, sigma_max, sigma_mean, sigma_median.
    """
    run_dir = os.path.dirname(cfg.save_path) if cfg.save_path else "."
    path = os.path.join(run_dir, "history.csv")

    cols = [
        "generation", "train_loss", "val_loss",
        "L1", "L2", "best_rmse", "worst_rmse",
        "sigma_min", "sigma_max", "sigma_mean", "sigma_median",
    ]
    int_cols = {"generation"}

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(history["generation"])):
            row = ",".join(
                str(int(history[c][i])) if c in int_cols else f"{history[c][i]:.6g}"
                for c in cols
            )
            f.write(row + "\n")
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
        hg = f["history"]
        history = {}
        for k in hg:
            if k == "timing":
                history["timing"] = {tk: list(hg["timing"][tk][:])
                                      for tk in hg["timing"]}
            else:
                history[k] = list(hg[k][:])
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
            print(f"  warning: saved U_pair shape {tuple(U_pair.shape)} "
                  f"differs from model.U_pair shape {tuple(model.U_pair.shape)}; "
                  f"keeping fresh V=0 init")
        else:
            model.U_pair.assign(U_pair)


def _print_load_summary(path: str, cfg: TNEPconfig) -> None:
    from ase.data import chemical_symbols
    type_str = ", ".join(f"{chemical_symbols[z]}(Z={z})→{idx}"
                         for z, idx in cfg.type_map.items())
    print(f"Model loaded from {path}")
    print(f"  target_mode={cfg.target_mode}, dim_q={cfg.dim_q}, "
          f"num_types={cfg.num_types}")
    print(f"  Type mapping: {type_str}")


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
