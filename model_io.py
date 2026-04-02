from __future__ import annotations

import json
import numpy as np
import os
from TNEPconfig import TNEPconfig
from TNEP import TNEP


def setup_run_directory(cfg: TNEPconfig) -> str:
    """Create a run directory under models/ and configure cfg paths.

    Directory structure:
        models/
            n{neurons}_q{dim_q}_pop{pop_size}_{YYYYMMDD_HHMMSS}/
                plots/
                config.txt
                (model .npz saved here after training)

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
    """Convert atomic number to element symbol."""
    from ase.data import chemical_symbols
    return chemical_symbols[z]


def _generate_model_filename(cfg: TNEPconfig) -> str:
    """Generate a model filename from config: {dataset}_{elements}_{mode}.npz

    Examples:
        train_C_H_O_dipole.npz
        water_O_H_pes.npz
    """
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    dataset_name = os.path.splitext(os.path.basename(cfg.data_path))[0]
    elements = "_".join(_z_to_symbol(z) for z in cfg.types)
    return f"{dataset_name}_{elements}_{mode}.npz"


def save_model(model: TNEP, cfg: TNEPconfig, path: str | None = None,
               label: str | None = None) -> None:
    """Save trained TNEP model weights and config to a .npz file.

    Saves all config attributes as a JSON string for full reproducibility.
    Also saves model weights and Z-to-type-index mapping.

    If path is True or "auto", generates filename from dataset name,
    element types, and target mode.

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : str or None — output file path. If None, auto-generates.
        label : optional suffix inserted before .npz (e.g. "best_val", "final_gen")
    """
    if path is None or path.endswith("auto"):
        directory = os.path.dirname(path) if path and os.path.dirname(path) else "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _generate_model_filename(cfg))

    if label:
        base, ext = os.path.splitext(path)
        path = f"{base}_{label}{ext}"

    # Build Z -> type index mapping: {atomic_number: layer_index}
    z_to_type_index = np.array(
        [[z, idx] for idx, z in enumerate(cfg.types)], dtype=np.int32)

    # Serialize full config as JSON — convert numpy types to native Python
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

    data = {
        # --- Model weights ---
        "W0": model.W0.numpy(),
        "b0": model.b0.numpy(),
        "W1": model.W1.numpy(),
        "b1": model.b1.numpy(),
        # --- Species mapping ---
        "z_to_type_index": z_to_type_index,
        # --- Full config ---
        "config_json": np.array(json.dumps(config_dict)),
    }
    if cfg.target_mode == 2:
        data["W0_pol"] = model.W0_pol.numpy()
        data["b0_pol"] = model.b0_pol.numpy()
        data["W1_pol"] = model.W1_pol.numpy()
        data["b1_pol"] = model.b1_pol.numpy()

    if cfg.descriptor_mean is not None:
        data["descriptor_mean"] = np.asarray(cfg.descriptor_mean)

    # Save PCA projection if used
    if hasattr(cfg, '_descriptor_pca') and cfg._descriptor_pca is not None:
        data.update(cfg._descriptor_pca.to_dict())

    np.savez(path, **data)
    print(f"Model saved to {path}")


def load_model(path: str = "tnep_model.npz") -> tuple[TNEP, TNEPconfig, dict[int, int]]:
    """Load a trained TNEP model from a .npz file.

    Reconstructs a TNEPconfig and TNEP model with the saved weights.
    Supports both new format (config_json) and legacy format (individual fields).

    Args:
        path : str — path to saved .npz file

    Returns:
        model          : TNEP model with loaded weights
        cfg            : TNEPconfig reconstructed from saved parameters
        type_map       : dict {atomic_number: layer_index} for converting Z arrays
    """
    data = np.load(path, allow_pickle=True)
    cfg = TNEPconfig()

    if "config_json" in data:
        # New format: restore full config from JSON
        config_dict = json.loads(str(data["config_json"]))
        for k, v in config_dict.items():
            # descriptor_mean is serialized as list in JSON; convert back to ndarray
            if k == "descriptor_mean" and v is not None:
                v = np.array(v, dtype=np.float32)
            setattr(cfg, k, v)
    else:
        # Legacy fallback: load individual fields
        cfg.num_types = int(data["num_types"])
        cfg.num_neurons = int(data["num_neurons"])
        cfg.dim_q = int(data["dim_q"])
        cfg.types = data["types"].tolist()
        cfg.target_mode = int(data["target_mode"])
        cfg.l_max = int(data["l_max"])
        cfg.alpha_max = int(data["alpha_max"])
        cfg.activation = str(data["activation"])
        cfg.data_path = str(data["data_path"])
        # Map old rc -> rcut_hard/rcut_soft
        if "rc" in data:
            rc = float(data["rc"])
            cfg.rcut_hard = rc
            cfg.rcut_soft = rc - 0.5

    # Restore descriptor_mean from dedicated array key (takes precedence over JSON)
    if "descriptor_mean" in data:
        cfg.descriptor_mean = data["descriptor_mean"].astype(np.float32)

    # Restore PCA projection if saved
    if "pca_components" in data:
        from descriptor_pca import DescriptorPCA
        cfg._descriptor_pca = DescriptorPCA.from_dict({
            "pca_components": data["pca_components"],
            "pca_mean": data["pca_mean"],
            "pca_explained_variance_ratio": data["pca_explained_variance_ratio"],
            "pca_n_components": data["pca_n_components"],
        })

    # Reconstruct Z -> type index mapping and store in cfg
    cfg.type_map = {int(row[0]): int(row[1]) for row in data["z_to_type_index"]}

    model = TNEP(cfg)
    model.W0.assign(data["W0"])
    model.b0.assign(data["b0"])
    model.W1.assign(data["W1"])
    model.b1.assign(data["b1"])

    if cfg.target_mode == 2:
        model.W0_pol.assign(data["W0_pol"])
        model.b0_pol.assign(data["b0_pol"])
        model.W1_pol.assign(data["W1_pol"])
        model.b1_pol.assign(data["b1_pol"])

    from ase.data import chemical_symbols
    type_str = ", ".join(f"{chemical_symbols[z]}(Z={z})→{idx}"
                         for z, idx in cfg.type_map.items())
    print(f"Model loaded from {path}")
    print(f"  target_mode={cfg.target_mode}, dim_q={cfg.dim_q}, "
          f"num_types={cfg.num_types}")
    print(f"  Type mapping: {type_str}")
    return model


def convert_z_to_type_indices(z_array: np.ndarray, type_map: dict[int, int]) -> np.ndarray:
    """Convert an array of atomic numbers to model-compatible type indices.

    Args:
        z_array  : ndarray of atomic numbers (e.g. structure.numbers)
        type_map : dict {atomic_number: layer_index} from load_model()

    Returns:
        type_indices : ndarray of int — model layer indices
    """
    return np.array([type_map[z] for z in z_array], dtype=np.int32)
