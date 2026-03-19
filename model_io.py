from __future__ import annotations

import json
import numpy as np
import os
import torch
from TNEPconfig import TNEPconfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TNEP import TNEP


def _z_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    from ase.data import chemical_symbols
    return chemical_symbols[z]


def _generate_model_filename(cfg: TNEPconfig) -> str:
    """Generate a model filename from config: {dataset}_{elements}_{mode}.pt

    Examples:
        train_C_H_O_dipole.pt
        water_O_H_pes.pt
    """
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    dataset_name = os.path.splitext(os.path.basename(cfg.data_path))[0]
    elements = "_".join(_z_to_symbol(z) for z in cfg.types)
    return f"{dataset_name}_{elements}_{mode}.pt"


def save_model(model: TNEP, cfg: TNEPconfig, path: str | None = None) -> None:
    """Save trained TNEP model weights and config to a .pt file.

    Saves all config attributes as a JSON string for full reproducibility.
    Also saves model weights and Z-to-type-index mapping.

    If path is True or "auto", generates filename from dataset name,
    element types, and target mode.

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : str or None — output file path. If None, auto-generates.
    """
    if path is None or path.endswith("auto"):
        directory = os.path.dirname(path) if path and os.path.dirname(path) else "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _generate_model_filename(cfg))

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
        elif isinstance(v, list):
            v = [int(x) if isinstance(x, np.integer) else
                 float(x) if isinstance(x, np.floating) else x
                 for x in v]
        config_dict[k] = v

    data = {
        # --- Model weights ---
        "W0": model.W0.cpu().numpy(),
        "b0": model.b0.cpu().numpy(),
        "W1": model.W1.cpu().numpy(),
        "b1": model.b1.cpu().numpy(),
        # --- Species mapping ---
        "z_to_type_index": z_to_type_index,
        # --- Full config ---
        "config_json": json.dumps(config_dict),
    }
    if cfg.target_mode == 2:
        data["W0_pol"] = model.W0_pol.cpu().numpy()
        data["b0_pol"] = model.b0_pol.cpu().numpy()
        data["W1_pol"] = model.W1_pol.cpu().numpy()
        data["b1_pol"] = model.b1_pol.cpu().numpy()

    if cfg.descriptor_mean is not None:
        data["descriptor_mean"] = np.asarray(cfg.descriptor_mean)

    # Save PCA projection if used
    if hasattr(cfg, '_descriptor_pca') and cfg._descriptor_pca is not None:
        data.update(cfg._descriptor_pca.to_dict())

    torch.save(data, path)
    print(f"Model saved to {path}")


def load_model(path: str = "tnep_model.pt") -> tuple[TNEP, TNEPconfig, dict[int, int]]:
    """Load a trained TNEP model from a .pt file (or legacy .npz file).

    Reconstructs a TNEPconfig and TNEP model with the saved weights.
    Supports both new .pt format (config_json as plain string),
    legacy .npz format (config_json as np.array or individual fields).

    Args:
        path : str — path to saved .pt or .npz file

    Returns:
        model          : TNEP model with loaded weights
        cfg            : TNEPconfig reconstructed from saved parameters
        type_map       : dict {atomic_number: layer_index} for converting Z arrays
    """
    from TNEP import TNEP

    # Load data — detect legacy .npz vs new .pt format
    if path.endswith(".npz"):
        data = dict(np.load(path, allow_pickle=True))
    else:
        data = torch.load(path, weights_only=False)

    cfg = TNEPconfig()

    if "config_json" in data:
        # New format: restore full config from JSON
        raw = data["config_json"]
        # Handle both np.array (legacy .npz) and plain string (.pt)
        config_str = str(raw) if not isinstance(raw, str) else raw
        config_dict = json.loads(config_str)
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
        # cfg.activation loaded but unused — tanh is hardcoded
        cfg.data_path = str(data["data_path"])
        # Map old rc -> rcut_hard/rcut_soft
        if "rc" in data:
            rc = float(data["rc"])
            cfg.rcut_hard = rc
            cfg.rcut_soft = rc - 0.5

    # Restore descriptor_mean from dedicated array key (takes precedence over JSON)
    if "descriptor_mean" in data:
        cfg.descriptor_mean = np.asarray(data["descriptor_mean"], dtype=np.float32)

    # Restore PCA projection if saved
    if "pca_components" in data:
        from descriptor_pca import DescriptorPCA
        cfg._descriptor_pca = DescriptorPCA.from_dict({
            "pca_components": data["pca_components"],
            "pca_mean": data["pca_mean"],
            "pca_explained_variance_ratio": data["pca_explained_variance_ratio"],
            "pca_n_components": data["pca_n_components"],
        })

    # Reconstruct Z -> type index mapping
    type_map = {int(row[0]): int(row[1]) for row in data["z_to_type_index"]}

    model = TNEP(cfg)
    model.W0 = torch.tensor(np.asarray(data["W0"]), dtype=torch.float32, device=model.device)
    model.b0 = torch.tensor(np.asarray(data["b0"]), dtype=torch.float32, device=model.device)
    model.W1 = torch.tensor(np.asarray(data["W1"]), dtype=torch.float32, device=model.device)
    model.b1 = torch.tensor(np.asarray(data["b1"]), dtype=torch.float32, device=model.device)

    if cfg.target_mode == 2:
        model.W0_pol = torch.tensor(np.asarray(data["W0_pol"]), dtype=torch.float32, device=model.device)
        model.b0_pol = torch.tensor(np.asarray(data["b0_pol"]), dtype=torch.float32, device=model.device)
        model.W1_pol = torch.tensor(np.asarray(data["W1_pol"]), dtype=torch.float32, device=model.device)
        model.b1_pol = torch.tensor(np.asarray(data["b1_pol"]), dtype=torch.float32, device=model.device)

    from ase.data import chemical_symbols
    type_str = ", ".join(f"{chemical_symbols[z]}(Z={z})→{idx}"
                         for z, idx in type_map.items())
    print(f"Model loaded from {path}")
    print(f"  target_mode={cfg.target_mode}, dim_q={cfg.dim_q}, "
          f"num_types={cfg.num_types}")
    print(f"  Type mapping: {type_str}")
    return model, cfg, type_map


def convert_z_to_type_indices(z_array: np.ndarray, type_map: dict[int, int]) -> np.ndarray:
    """Convert an array of atomic numbers to model-compatible type indices.

    Args:
        z_array  : ndarray of atomic numbers (e.g. structure.numbers)
        type_map : dict {atomic_number: layer_index} from load_model()

    Returns:
        type_indices : ndarray of int — model layer indices
    """
    return np.array([type_map[z] for z in z_array], dtype=np.int32)
