"""CPU-sized end-to-end smoke of train_model for the multi-descriptor modes.

Runs load → descriptors → SNES/Adam → test scoring → save, then re-loads the
saved best-val model and scores the test slice again (save/load path).
No accuracy is claimed; the smoke proves "runs, validates, saves, loads".
The 5-frame test slice is frames 0-4 of the training file and may overlap
the (shuffled) training frames, so its RMSE is not a held-out number.

    CUDA_VISIBLE_DEVICES='' python scripts/smoke_train.py \
        --readout neighbor --nep sum --optimizer adam --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readout", choices=["neighbor", "tnep"], default="neighbor")
    p.add_argument("--rij-power", type=int, default=2, help="tnep readout: |r_ij|^N weight")
    p.add_argument("--nep", choices=["off", "concat", "replace", "sum", "gap_sum"], default="off")
    p.add_argument("--head", choices=["vector", "scalar"], default="vector")
    p.add_argument("--scale", choices=["off", "block", "channel", "minmax"], default="off")
    p.add_argument("--norm", choices=["l2", "none"], default="l2")
    p.add_argument("--optimizer", choices=["snes", "adam"], default="snes")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=None, help="SNES mini-batch (None = full batch)")
    p.add_argument("--data", default=os.path.join(ROOT, "datasets", "train_waterbulk.xyz"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from ase.io import read, write
    from MasterTNEP import train_model
    from TNEPconfig import TNEPconfig
    from model_io import load_model
    import nep_descriptor as nd

    os.makedirs(a.out, exist_ok=True)
    test_path = os.path.join(a.out, "test_slice.xyz")
    write(test_path, read(a.data, index=":5"))

    cfg = TNEPconfig()
    cfg.data_path = a.data
    cfg.test_data_path = test_path
    cfg.total_N = a.frames
    cfg.descriptor_mode = 1                 # TF builder; radial_basis_norm="none" requires it
    cfg.dipole_readout = a.readout
    cfg.dipole_head = a.head
    cfg.nep_descriptor = a.nep
    cfg.descriptor_scale = a.scale
    cfg.radial_basis_norm = a.norm
    cfg.optimizer = a.optimizer
    cfg.num_generations = a.generations     # adam.py: steps = num_generations
    cfg.pop_size = a.pop
    cfg.batch_size = a.batch_size
    if a.readout == "tnep":
        cfg.dipole_rij_power = a.rij_power
        cfg.activation = "tanh"              # GPUMD TNEP
    if a.nep == "replace":
        cfg.descriptor_mixing = False       # replace has no SOAP blocks to mix
    # CPU sizing.
    cfg.adam_lr = 1e-3
    cfg.num_neurons = 8
    cfg.nep_num_neurons = 8
    cfg.dipole_channels = 4
    cfg.population_chunk_size = 4
    cfg.batch_chunk_size = 4
    cfg.val_interval = 1
    cfg.plot_interval = None
    cfg.checkpoint_interval = None
    cfg.show_plots = False
    cfg.save_path = os.path.join(a.out, "models", "auto")

    t0 = time.time()
    model = train_model(cfg)
    n_params = int(model.optimizer.dim)

    saved = glob.glob(os.path.join(a.out, "models", "*", "*best_val*.h5"))
    assert len(saved) == 1, saved
    loaded = load_model(saved[0])
    metrics, _ = loaded.score_from_file(test_path, show_plots=False)
    assert "rmse" in metrics, metrics

    if a.readout == "tnep" and a.nep in ("concat", "replace"):
        # GPUMD's number_of_variables for the same header; under "replace"
        # (no mixing / preprocess / learned SOAP basis; descriptor_scale adds
        # no parameters, so the identity holds under --scale minmax too)
        # our typed slab is exactly (dim+2)·H per type + b1 + c_r/c_a.
        n_gpumd = nd.gpumd_num_params(cfg, soap_dim=cfg.dim_q - nd.nep_dim(cfg))
        print(f"n_params={n_params} n_params_gpumd={n_gpumd}")
        if a.nep == "replace":
            assert n_params == n_gpumd, (n_params, n_gpumd)
    print(f"SMOKE OK {' '.join(sys.argv[1:] if argv is None else argv)} "
          f"W1={tuple(int(d) for d in loaded.W1.shape)} dim_q={cfg.dim_q} "
          f"n_params={n_params} wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
