from __future__ import annotations

import json
import numpy as np
import os
import h5py
from TNEPconfig import TNEPconfig
from TNEP import TNEP


def setup_run_directory(cfg: TNEPconfig) -> str:
    """Create a run directory under models/ and configure cfg paths.

    Directory structure (GAP convention):
        models/
            gap_M{n_sparse}_q{dim_q}_z{zeta}_{YYYYMMDD_HHMMSS}/
                plots/
                config.txt
                (model .h5 saved here after training)

    Naming reflects the active GAP hyperparameters:
        M : sparse-point count (gap_n_sparse)
        q : descriptor dimension
        z : polynomial kernel exponent (gap_zeta)

    Requires cfg.dim_q to be set (call after descriptor building).
    Updates cfg.save_path and cfg.save_plots in place.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = (f"gap_M{cfg.gap_n_sparse}_q{cfg.dim_q}_"
                f"z{cfg.gap_zeta}_{timestamp}")
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
    fields whose value matches the class-level default are also
    captured (vars(cfg) misses any field never explicitly written to
    the instance, which is the common case for defaults).
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


def save_model(model: TNEP, cfg: TNEPconfig, path: str | None = None) -> None:
    """Save trained GAP model weights and config to an HDF5 (.h5) file.

    File layout:
        /                — top-level attributes: target_mode, num_types,
                           dim_q, elements (quick inspection)
        /weights/        — sparse_q, sparse_q_norm, sparse_species,
                           alpha, delta_per_species, plus optional
                           q_scaler / target_mean
        /descriptor/     — z_to_type_index
        /config          — full TNEPconfig serialised as JSON string

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : output file path. None or ending "auto" = auto-generate.
    """
    if path is None or path.endswith("auto"):
        directory = os.path.dirname(path) if path and os.path.dirname(path) else "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _generate_model_filename(cfg))

    config_dict = _serialize_config(cfg)
    z_to_type_index = np.array(
        [[z, idx] for idx, z in enumerate(cfg.types)], dtype=np.int32)

    with h5py.File(path, "w") as f:
        # Top-level metadata — visible via `h5ls -v model.h5` without loading weights
        f.attrs["target_mode"] = cfg.target_mode
        f.attrs["num_types"] = cfg.num_types
        f.attrs["dim_q"] = cfg.dim_q
        # Store as a native int array attribute — h5py handles this
        # directly (visible via `h5ls -v model.h5`), avoiding the JSON
        # round-trip that the full `config` dataset below needs for
        # heterogeneous dict serialisation.
        f.attrs["elements"] = np.asarray(cfg.types, dtype=np.int32)

        # Weights — GAP model state (sparse_q, alpha, etc.).
        # `model_schema_version` attr tags the file as GAP-era; legacy
        # NN-era files have `/weights/W0` instead and trip the loader
        # detection.
        wg = f.create_group("weights")
        f.attrs["model_schema_version"] = "gap-1"
        wg.create_dataset("sparse_q", data=model.sparse_q.numpy())
        wg.create_dataset("sparse_q_norm", data=model.sparse_q_norm.numpy())
        wg.create_dataset("sparse_species", data=model.sparse_species.numpy())
        wg.create_dataset("alpha", data=model.alpha.numpy())
        wg.create_dataset(
            "delta_per_species", data=model.delta_per_species.numpy())

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
    """Write a one-row history.csv summarising the GAP fit.

    Columns: generation, train_loss, val_loss, best_rmse, worst_rmse,
    best_rrmse, avg_rrmse. Closed-form GAP produces a single row.
    """
    run_dir = os.path.dirname(cfg.save_path) if cfg.save_path else "."
    path = os.path.join(run_dir, "history.csv")

    cols = ["generation", "train_loss", "val_loss",
            "best_rmse", "worst_rmse", "best_rrmse", "avg_rrmse"]
    int_cols = {"generation"}
    n_rows = len(history.get("generation", []))

    def _fmt(c: str, val) -> str:
        if c in int_cols:
            return str(int(val))
        try:
            f = float(val)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if f != f else f"{f:.6g}"

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(n_rows):
            f.write(",".join(_fmt(c, history.get(c, [float("nan")] * n_rows)[i])
                              for c in cols) + "\n")
    print(f"History saved to {path}")


def _load_weights(model: TNEP, cfg: TNEPconfig,
                   sparse_q, sparse_q_norm, sparse_species, alpha,
                   delta_per_species) -> None:
    """Restore GAP model state from saved arrays."""
    model.sparse_q.assign(sparse_q)
    model.sparse_q_norm.assign(sparse_q_norm)
    model.sparse_species.assign(sparse_species)
    model.alpha.assign(alpha)
    model.delta_per_species.assign(delta_per_species)


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
        # Detect legacy NN-era files via the presence of /weights/W0.
        # New GAP files have /weights/sparse_q instead.
        if "W0" in wg and "sparse_q" not in wg:
            raise ValueError(
                f"Model at {path!r} was saved by the NN-era TNEP "
                f"(presence of /weights/W0). The current codebase is GAP "
                f"and cannot load NN-era checkpoints. Check out the "
                f"`pre-gap-baseline` git tag to load it, or retrain.")
        if "sparse_q" not in wg:
            raise ValueError(
                f"Model at {path!r} lacks /weights/sparse_q — not a GAP "
                f"checkpoint.")
        weights = {
            "sparse_q":          wg["sparse_q"][:],
            "sparse_q_norm":     wg["sparse_q_norm"][:],
            "sparse_species":    wg["sparse_species"][:],
            "alpha":             wg["alpha"][:],
            "delta_per_species": wg["delta_per_species"][:],
        }
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
    """Legacy .npz loader. Not supported under GAP — raises."""
    raise NotImplementedError(
        f"_load_model_npz({path!r}): legacy NumPy checkpoints are no "
        f"longer supported. Re-export the model via the current h5 API "
        f"from the `pre-gap-baseline` git tag if needed.")




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
