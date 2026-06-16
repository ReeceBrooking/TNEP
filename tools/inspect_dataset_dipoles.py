"""Quick stats on a dataset XYZ: mean atoms/structure and mean per-atom |μ|.

Usage:
    python inspect_dataset_dipoles.py [path]
    (default path: datasets/test.xyz)
"""
from __future__ import annotations

import sys

import numpy as np
from ase.io import read


def main(path: str = "datasets/test.xyz", key: str = "dipole",
         allowed_species: list[int] | None = None) -> None:
    structures = read(path, index=":")
    if allowed_species is not None:
        before = len(structures)
        allowed = set(allowed_species)
        structures = [s for s in structures
                      if set(int(z) for z in s.get_atomic_numbers()).issubset(allowed)]
        print(f"Species filter {sorted(allowed)}: "
              f"kept {len(structures)}/{before} structures")
    n_total = len(structures)
    n_with_dipole = 0
    atom_counts: list[int] = []
    per_atom_mags: list[float] = []
    total_mags: list[float] = []
    dipoles: list[np.ndarray] = []
    for s in structures:
        atom_counts.append(len(s))
        if key in s.info:
            mu = np.asarray(s.info[key], dtype=np.float64)
        elif s.calc is not None and key in getattr(s.calc, "results", {}):
            mu = np.asarray(s.calc.results[key], dtype=np.float64)
        else:
            continue
        n_with_dipole += 1
        mag = float(np.linalg.norm(mu))
        total_mags.append(mag)
        per_atom_mags.append(mag / max(len(s), 1))
        dipoles.append(mu)

    print(f"File: {path}")
    print(f"  Structures             : {n_total}")
    print(f"  Mean atoms / structure : {np.mean(atom_counts):.3f}")
    print(f"  Min/max atoms          : {min(atom_counts)} / {max(atom_counts)}")
    print(f"  Structures with '{key}' : {n_with_dipole}")
    if per_atom_mags:
        t = np.asarray(total_mags)
        print(f"  Mean total |μ|        : {t.mean():.6f}")
        print(f"    median              : {np.median(t):.6f}")
        print(f"    std                 : {t.std():.6f}")
        print(f"    min / max           : {t.min():.6f} / {t.max():.6f}")
        a = np.asarray(per_atom_mags)
        print(f"  Mean per-atom |μ|     : {a.mean():.6f}")
        print(f"    median              : {np.median(a):.6f}")
        print(f"    std                 : {a.std():.6f}")
        print(f"    min / max           : {a.min():.6f} / {a.max():.6f}")
    else:
        print(f"  No '{key}' field found in any structure (info or calc).")
        return

    # RRMSE denominators (per-component scalar interpretation: SS_tot
    # runs over all 3N (structure, component) entries). RMSE assumed
    # to be per-component as well. Two conventions reported.
    arr = np.asarray(dipoles)                       # [N, 3] total dipole
    nat = np.asarray(atom_counts[:n_with_dipole], dtype=np.float64).reshape(-1, 1)
    arr_per_atom = arr / nat                          # [N, 3] per-atom dipole

    print()
    print("  TOTAL-DIPOLE space (matches TNEP score()'s 'total_rmse'):")
    _print_rrmse_block(arr)
    print()
    print("  PER-ATOM-DIPOLE space (matches GPUMD's loss.out + TNEP score()'s 'rmse'):")
    _print_rrmse_block(arr_per_atom)


def _print_rrmse_block(arr: np.ndarray) -> None:
    cmean = arr.mean(axis=0)
    ms_target = float((arr ** 2).mean())             # mean(y²) across all 3N
    ms_centered = float(((arr - cmean) ** 2).mean())
    print(f"    mean (per-component) = {np.array2string(cmean, precision=4)}")
    print(f"    √mean(y²)            = {ms_target ** 0.5:.6f}")
    print(f"    √mean((y-ȳ)²)        = {ms_centered ** 0.5:.6f}")
    for rmse in (0.0241, 0.0181, 0.0050):
        rr_unc = rmse / (ms_target ** 0.5)
        print(f"    RMSE = {rmse:.4f}  →  un-centered RRMSE = {rr_unc:.4%}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/test.xyz"
    species = None
    if len(sys.argv) > 2:
        species = [int(s) for s in sys.argv[2].split(",")]
    main(path, allowed_species=species)
