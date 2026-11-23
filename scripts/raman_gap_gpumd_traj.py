"""Polarisability along the GPUMD bulk-water trajectory with the recommended α GAP
(docs/gap_dipole_polarisability_models.md §4.2 "base": two bond groups + 4-channel
cross group, no long-range column, fitted on the 500 training frames), saved as a
[T, 6] Voigt file — the Raman twin of plots/<traj>_dipoles.npy. The spectrum is
then one call: spectroscopy.raman_spectrum_from_file(<npy>, dt_fs=1.0).

    python scripts/raman_gap_gpumd_traj.py [n_frames] [stride]     (defaults: all, 1)

SOAP on the GPU (descriptor_mode=1, basis poly3 — same test RMSE as poly3gauss, which
the GPU builder lacks; predictions agree with quippy to 3e-6). Frames stream from the
4.8 GB xyz into RAMAN_WORKERS spawned processes that share the GPU; each fits its own
copy of the model at start (~15 s) because a TF/CUDA context cannot be forked.
Per-frame work is ~27 ms single-process (17 ms descriptors, 10 ms readout + kernels),
and the GPU is mostly idle, so 2 processes overlap to ~14 ms/frame wall (a third adds little and 4 saturate the 12 GB card).
GPUMD wrote one frame per fs (Time= header). Env: RAMAN_DATA / RAMAN_TEST (label
files, e.g. the *_waterbulk_chi.xyz susceptibility labels), RAMAN_TAG (output suffix)."""
import os, sys, json, time, multiprocessing as mp
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
import numpy as np

TRAJ = "datasets/water_bulk_GPUMD_traj.xyz"
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else None
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CHUNK, WORKERS = 8, int(os.environ.get("RAMAN_WORKERS", 2))

# RAMAN_DATA / RAMAN_TEST: label files (default the CM-form pol; the susceptibility labels are datasets/*_waterbulk_chi.xyz); RAMAN_TAG: output suffix.
ALPHA_BEST = dict(data_path=os.environ.get("RAMAN_DATA", "datasets/train_waterbulk.xyz"),
                  test_data_path=os.environ.get("RAMAN_TEST", "datasets/test_waterbulk.xyz"), total_N=None,
                  target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7, compress_mode="trivial",
                  basis="poly3", descriptor_mode=1, descriptor_batch_frames=4,
                  dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
                  dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_sigma=0.1,
                  gap_cross_channels=4)

MODEL = None


def _init():
    """Per-process: own TF/CUDA context (memory growth so WORKERS processes share the card), own model."""
    global MODEL
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        # a hard per-process cap: with memory growth, 4 processes filled the 12 GB card and thrashed (160 ms/frame)
        tf.config.set_logical_device_configuration(
            g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get("RAMAN_VRAM_MB", 2500)))])
    import gap_dipole as gd
    from test_gap_dipole import _cfg                 # pins every readout/descriptor field
    MODEL = gd.fit_gap_dipole(_cfg(**ALPHA_BEST))


def _predict(frames):
    import gap_dipole as gd
    return gd.predict(MODEL, frames)


def _chunks():
    from ase.io import iread
    buf, kept = [], 0
    for k, at in enumerate(iread(TRAJ, index=":")):
        if k % STRIDE:
            continue
        at.calc = None; buf.append(at); kept += 1
        if len(buf) == CHUNK:
            yield buf; buf = []
        if N_FRAMES is not None and kept >= N_FRAMES:
            break
    if buf:
        yield buf


if __name__ == "__main__":
    t0, out, n = time.time(), [], 0
    with mp.get_context("spawn").Pool(WORKERS, initializer=_init) as pool:
        for P in pool.imap(_predict, _chunks()):
            out.append(P); n += len(P)
            if n % (CHUNK * WORKERS * 10) == 0:
                rate = (time.time() - t0) / n; total = N_FRAMES if N_FRAMES else 200000 // STRIDE
                print(f"  {n} frames, {rate * 1e3:.1f} ms/frame wall (incl. start-up); ETA {rate * (total - n) / 60:.1f} min", flush=True)
    pols = np.concatenate(out)
    dt = time.time() - t0
    print(f"predicted {len(pols)} frames in {dt:.0f} s ({dt / len(pols) * 1e3:.1f} ms/frame wall)")
    os.makedirs("plots", exist_ok=True)
    stem = "plots/water_bulk_GPUMD_traj_polarizabilities" + os.environ.get("RAMAN_TAG", "")
    np.save(stem + ".npy", pols)
    np.savetxt(stem + ".txt", pols, fmt="%.8e", header="alpha_xx  alpha_yy  alpha_zz  alpha_xy  alpha_yz  alpha_zx  (A^3 per cell)")
    json.dump({"n_frames": len(pols), "stride": STRIDE, "dt_fs": 1.0 * STRIDE, "model": ALPHA_BEST},
              open(stem + ".json", "w"), indent=1, default=str)
    print(f"saved {stem}.npy / .txt / .json")
