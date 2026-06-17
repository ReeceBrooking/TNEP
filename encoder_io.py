"""Persistence helpers for the SOAP autoencoder.

Mirrors `model_io.py` in style and conventions, but for `SoapAutoencoder`:
the encoder, decoder, and per-feature standardiser are written as three
independent artifacts so any one can be loaded in isolation.

Layout per run:

    models/autoencoder/
      {YYYYMMDD_HHMMSS}_z{latent}_h{hidden_signature}/
        config.json          # machine-readable AutoencoderConfig snapshot
        config.txt           # human-readable mirror
        standardizer.npz     # mean, std arrays (Q_raw floats each)
        encoder.weights.h5   # encoder Sequential weights (Keras 3 suffix)
        decoder.weights.h5   # decoder Sequential weights
        history.json
        history.csv
        test_metrics.json    # (when cfg.test_data_path/cache supplied)

The encoder/decoder files contain ONLY their respective Sequential's
trainable layer kernels and biases. The standardiser is stored separately
so a downstream consumer can load just the encoder (for compression) or
just the decoder (for reconstruction from a saved latent) without pulling
in the other half's weights file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from SoapAutoencoder import AutoencoderConfig, SoapAutoencoder, Standardizer


# ════════════════════════════════════════════════════════════════════════
# Run-directory setup + config serialisation
# ════════════════════════════════════════════════════════════════════════

def _hidden_signature(hidden_dims) -> str:
    """Hyphen-joined hidden-layer dim signature for the run name.

    `()` → 'linear' (single linear projection, no hidden layers).
    """
    hd = tuple(hidden_dims) if hidden_dims else ()
    if not hd:
        return "linear"
    return "-".join(str(int(h)) for h in hd)


def setup_encoder_run_directory(cfg: AutoencoderConfig) -> Path:
    """Create models/autoencoder/{ts}_z{Z}_h{H1-H2-…}/ and write config.txt.

    Mirrors `model_io.setup_run_directory`. Returns the run directory as
    a `Path`. Updates `cfg.output_dir` to point at the parent so a
    downstream consumer reading the cfg back knows where to look.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(cfg.output_dir)
    arch = str(getattr(cfg, "architecture", "mlp")).lower()
    if arch == "willatt":
        K = cfg.willatt_K if cfg.willatt_K is not None else "auto"
        tied = "_tied" if getattr(cfg, "tied_weights", False) else ""
        run_name = f"{timestamp}_willatt_K{K}{tied}"
    elif arch == "l_block_pca":
        K = cfg.l_block_K if cfg.l_block_K is not None else f"z{int(cfg.latent_dim)}"
        run_name = f"{timestamp}_lblockpca_K{K}"
    elif arch == "willatt_l_block":
        Ks = cfg.willatt_K if cfg.willatt_K is not None else "auto"
        Kl = cfg.l_block_K if cfg.l_block_K is not None else f"z{int(cfg.latent_dim)}"
        tied = "_tied" if getattr(cfg, "tied_weights", False) else ""
        run_name = f"{timestamp}_willattlblock_Ks{Ks}_Kl{Kl}{tied}"
    else:
        run_name = (f"{timestamp}_z{int(cfg.latent_dim)}"
                    f"_h{_hidden_signature(cfg.hidden_dims)}")
    run_dir = parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable config snapshot. Walks both the annotated class
    # fields and any instance-level attributes the caller has stashed.
    annotated = set(getattr(AutoencoderConfig, "__annotations__", {}).keys())
    instance_extras = {k for k in vars(cfg).keys() if not k.startswith("_")}
    fields = sorted(annotated | instance_extras)
    config_txt = run_dir / "config.txt"
    with open(config_txt, "w") as f:
        f.write(f"# AutoencoderConfig — {timestamp}\n")
        f.write(f"# Run directory: {run_dir}\n\n")
        for k in fields:
            if k.startswith("_"):
                continue
            if not hasattr(cfg, k):
                continue
            v = getattr(cfg, k)
            if callable(v):
                continue
            if isinstance(v, np.ndarray):
                f.write(f"{k} = ndarray shape={v.shape} dtype={v.dtype}\n")
            else:
                f.write(f"{k} = {v!r}\n")

    print(f"Run directory: {run_dir}")
    return run_dir


def _serialize_config(cfg: AutoencoderConfig) -> dict:
    """Convert AutoencoderConfig to a JSON-safe dict.

    Walks `AutoencoderConfig.__annotations__` so fields whose value
    matches the class default are still captured (same reasoning as
    `model_io._serialize_config` — protects against silent default
    drift between save and reload).
    """
    field_names = set(getattr(AutoencoderConfig, "__annotations__", {}).keys())
    field_names.update(k for k in vars(cfg).keys() if not k.startswith("_"))

    out: dict = {}
    for k in sorted(field_names):
        if k.startswith("_"):
            continue
        if not hasattr(cfg, k):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.integer, np.bool_)):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, Path):
            v = str(v)
        elif isinstance(v, dict):
            # JSON forces dict keys to strings on serialise, which would
            # corrupt int-keyed maps like `cfg.type_map = {6: 0, 1: 1}`.
            # Store such dicts as a list of [key, value] pairs so the
            # loader can round-trip the original int types.
            v = [[int(kk) if isinstance(kk, (int, np.integer)) else kk,
                  int(vv) if isinstance(vv, (int, np.integer)) else vv]
                 for kk, vv in v.items()]
        out[k] = v
    return out


def _deserialize_config(blob: dict) -> AutoencoderConfig:
    """Rebuild an AutoencoderConfig from a JSON dict.

    Sets every field present in the blob onto a fresh AutoencoderConfig
    instance. Fields absent from the blob keep their class-level defaults
    — there's no `_LEGACY_FIELD_DEFAULTS` table here (yet) because the
    AE doesn't have model_io's deeper backward-compat history. Add one
    here if/when an AE cfg field's default ever changes meaning.
    """
    cfg = AutoencoderConfig()
    # Fields known to hold int-keyed dicts persisted as list-of-pairs.
    _DICT_FIELDS = {"type_map"}
    for k, v in blob.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list) and k in ("hidden_dims", "l_block_hidden_dims"):
            v = tuple(int(h) for h in v)
        elif k in _DICT_FIELDS and isinstance(v, list):
            # Round-trip from `_serialize_config`'s [[k, v], ...] form.
            v = {int(kk): int(vv) for kk, vv in v}
        elif k in _DICT_FIELDS and isinstance(v, dict):
            # Defensive: handle dicts whose keys were stringified by an
            # older serialiser. Cast both key and value back to int.
            v = {int(kk): int(vv) for kk, vv in v.items()}
        setattr(cfg, k, v)
    return cfg


def save_config(cfg: AutoencoderConfig, out_dir: Path) -> None:
    """Write both config.json (machine-readable) and config.txt (already
    written by setup_encoder_run_directory; re-emitted here is a no-op).
    """
    out_dir = Path(out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump(_serialize_config(cfg), f, indent=2)


def load_config(out_dir: Path) -> AutoencoderConfig:
    """Read config.json from a run directory."""
    out_dir = Path(out_dir)
    with open(out_dir / "config.json", "r") as f:
        blob = json.load(f)
    return _deserialize_config(blob)


# ════════════════════════════════════════════════════════════════════════
# Encoder / decoder / standardiser save
# ════════════════════════════════════════════════════════════════════════
#
# Each half of the autoencoder is wrapped in a thin tf.keras.Model whose
# sole purpose is to scope save_weights to just the layers we want. This
# keeps both .h5 files purely "weights for this Sequential" with no
# leakage of the standardiser or the other half's parameters.

class _EncoderWrapper(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Sequential):
        super().__init__(name="encoder_wrapper")
        self.encoder = encoder

    def call(self, x, training=None):
        return self.encoder(x, training=training)


class _DecoderWrapper(tf.keras.Model):
    def __init__(self, decoder: tf.keras.Sequential):
        super().__init__(name="decoder_wrapper")
        self.decoder = decoder

    def call(self, x, training=None):
        return self.decoder(x, training=training)


def _is_willatt_l_block(model) -> bool:
    """The composed Willatt + l_block AE has BOTH the species projection
    `u` and the per-l Dense list. Detected by its K-species attribute,
    which neither of the standalone backbones expose.
    """
    return (getattr(model, "u", None) is not None
            and getattr(model, "per_l_encoders", None) is not None
            and getattr(model, "K_species", None) is not None)


def _is_willatt(model) -> bool:
    """The Willatt AE is identified by its `u` species-projection matrix.

    Using attribute presence rather than isinstance to avoid a circular
    import (encoder_io is imported by the SoapAutoencoder __main__ block).
    Must run AFTER the willatt_l_block check, which also exposes `u`.
    """
    return (getattr(model, "u", None) is not None
            and getattr(model, "per_l_encoders", None) is None)


def _is_l_block_pca(model) -> bool:
    """The l-block PCA AE is identified by its per-l Dense encoder list.

    Must run AFTER the willatt_l_block check, which also exposes
    per_l_encoders.
    """
    return (getattr(model, "per_l_encoders", None) is not None
            and getattr(model, "u", None) is None)


def save_encoder(model, out_dir: Path) -> Path:
    """Write the encoder weights.

    MLP backbone → `encoder.weights.h5` (Keras Sequential).
    Willatt backbone → `encoder.npz` containing the species-projection
        matrix `u[T, K]` and the integers (T, K, alpha_max, l_max). The
        bilinear scatter/gather plumbing is rebuilt from the cfg at
        load time, so only the trained matrix needs to live on disk.
    """
    out_dir = Path(out_dir)
    if _is_willatt_l_block(model):
        path = out_dir / "encoder.npz"
        hidden_dims = tuple(int(h) for h in
                            getattr(model, "l_block_hidden_dims", ()))
        save: dict = {
            "T": int(model.T),
            "K_species": int(model.K_species),
            "alpha_max": int(model.alpha_max),
            "l_max": int(model.l_max),
            "L": int(model.L),
            "q_raw_T": int(model.q_raw_T),
            "q_raw_K": int(model.q_raw_K),
            "latent_dim": int(model.latent_dim),
            "per_l_Q_K": np.asarray(model._per_l_Q_K, dtype=np.int32),
            "per_l_K": np.asarray(model._per_l_K, dtype=np.int32),
            "compress_mode": str(model.compress_mode),
            "l_block_hidden_dims": np.asarray(list(hidden_dims), dtype=np.int32),
            "tied": bool(model.tied_weights),
            "u": model.u.numpy().astype(np.float32),
        }
        for l in range(model.L):
            branch = model.per_l_encoders[l]
            for li, layer in enumerate(branch.layers):
                save[f"enc_l{l}_layer{li}_W"] = layer.kernel.numpy().astype(np.float32)
                save[f"enc_l{l}_layer{li}_b"] = layer.bias.numpy().astype(np.float32)
        np.savez(path, **save)
        return path
    if _is_willatt(model):
        path = out_dir / "encoder.npz"
        np.savez(
            path,
            u=model.u.numpy().astype(np.float32),
            T=int(model.T), K=int(model.K),
            alpha_max=int(model.alpha_max), l_max=int(model.l_max),
            q_raw_T=int(model.q_raw_T), q_raw_K=int(model.q_raw_K),
            compress_mode=str(model.compress_mode),
            tied=bool(model.tied_weights),
        )
        return path
    if _is_l_block_pca(model):
        path = out_dir / "encoder.npz"
        hidden_dims = tuple(int(h) for h in
                            getattr(model, "l_block_hidden_dims", ()))
        save: dict = {
            "T": int(model.T),
            "alpha_max": int(model.alpha_max),
            "l_max": int(model.l_max),
            "L": int(model.L),
            "q_raw": int(model.q_raw),
            "latent_dim": int(model.latent_dim),
            "per_l_Q": np.asarray(model._per_l_Q, dtype=np.int32),
            "per_l_K": np.asarray(model._per_l_K, dtype=np.int32),
            "compress_mode": str(model.compress_mode),
            "l_block_hidden_dims": np.asarray(list(hidden_dims), dtype=np.int32),
        }
        # Per-l branches are Sequentials — iterate their internal Dense
        # layers and dump kernel + bias for each. n_layers = len(hidden) + 1
        # always (hidden chain plus the output layer).
        for l in range(model.L):
            branch = model.per_l_encoders[l]
            for li, layer in enumerate(branch.layers):
                save[f"enc_l{l}_layer{li}_W"] = layer.kernel.numpy().astype(np.float32)
                save[f"enc_l{l}_layer{li}_b"] = layer.bias.numpy().astype(np.float32)
        np.savez(path, **save)
        return path
    wrapper = _EncoderWrapper(model.encoder)
    # Trigger build so save_weights has variables to serialise.
    wrapper(tf.zeros([1, model.q_raw], dtype=tf.float32))
    path = out_dir / "encoder.weights.h5"
    wrapper.save_weights(path)
    return path


def save_decoder(model, out_dir: Path) -> Path:
    """Write the decoder weights.

    MLP backbone → `decoder.weights.h5` (Keras Sequential).
    Willatt backbone → `decoder.npz` containing `v[K, T]` (or a marker
        when tied so v = uᵀ).
    """
    out_dir = Path(out_dir)
    if _is_willatt_l_block(model):
        path = out_dir / "decoder.npz"
        hidden_dims = tuple(int(h) for h in
                            getattr(model, "l_block_hidden_dims", ()))
        save: dict = {
            "T": int(model.T),
            "K_species": int(model.K_species),
            "alpha_max": int(model.alpha_max),
            "l_max": int(model.l_max),
            "L": int(model.L),
            "q_raw_T": int(model.q_raw_T),
            "q_raw_K": int(model.q_raw_K),
            "latent_dim": int(model.latent_dim),
            "per_l_Q_K": np.asarray(model._per_l_Q_K, dtype=np.int32),
            "per_l_K": np.asarray(model._per_l_K, dtype=np.int32),
            "compress_mode": str(model.compress_mode),
            "l_block_hidden_dims": np.asarray(list(hidden_dims), dtype=np.int32),
            "tied": bool(model.tied_weights),
        }
        if not model.tied_weights:
            save["v"] = model.v.numpy().astype(np.float32)
        for l in range(model.L):
            branch = model.per_l_decoders[l]
            for li, layer in enumerate(branch.layers):
                save[f"dec_l{l}_layer{li}_W"] = layer.kernel.numpy().astype(np.float32)
                save[f"dec_l{l}_layer{li}_b"] = layer.bias.numpy().astype(np.float32)
        np.savez(path, **save)
        return path
    if _is_willatt(model):
        path = out_dir / "decoder.npz"
        common = dict(T=int(model.T), K=int(model.K),
                      alpha_max=int(model.alpha_max),
                      l_max=int(model.l_max),
                      compress_mode=str(model.compress_mode))
        if model.tied_weights:
            np.savez(path, tied=True, **common)
        else:
            np.savez(path, tied=False,
                     v=model.v.numpy().astype(np.float32), **common)
        return path
    if _is_l_block_pca(model):
        path = out_dir / "decoder.npz"
        hidden_dims = tuple(int(h) for h in
                            getattr(model, "l_block_hidden_dims", ()))
        save: dict = {
            "T": int(model.T),
            "alpha_max": int(model.alpha_max),
            "l_max": int(model.l_max),
            "L": int(model.L),
            "q_raw": int(model.q_raw),
            "latent_dim": int(model.latent_dim),
            "per_l_Q": np.asarray(model._per_l_Q, dtype=np.int32),
            "per_l_K": np.asarray(model._per_l_K, dtype=np.int32),
            "compress_mode": str(model.compress_mode),
            "l_block_hidden_dims": np.asarray(list(hidden_dims), dtype=np.int32),
        }
        for l in range(model.L):
            branch = model.per_l_decoders[l]
            for li, layer in enumerate(branch.layers):
                save[f"dec_l{l}_layer{li}_W"] = layer.kernel.numpy().astype(np.float32)
                save[f"dec_l{l}_layer{li}_b"] = layer.bias.numpy().astype(np.float32)
        np.savez(path, **save)
        return path
    wrapper = _DecoderWrapper(model.decoder)
    wrapper(tf.zeros([1, model.latent_dim], dtype=tf.float32))
    path = out_dir / "decoder.weights.h5"
    wrapper.save_weights(path)
    return path


def save_standardizer(model: SoapAutoencoder, out_dir: Path) -> Path:
    """Write standardizer.npz holding the per-feature mean / std arrays.

    Saved as plain numpy so a downstream consumer can read it without
    importing TensorFlow.
    """
    out_dir = Path(out_dir)
    path = out_dir / "standardizer.npz"
    np.savez(
        path,
        mean=model.standardizer.mean.numpy().astype(np.float32),
        std=model.standardizer.std.numpy().astype(np.float32),
    )
    return path


def save_test_metrics(metrics: dict, out_dir: Path) -> None:
    """Dump held-out-test metrics from `SoapAutoencoder.evaluate` as JSON.

    Schema: `{n_atoms, loss, rmse_raw, rmse_standardised, r2}`. Written
    as a flat dict (no nesting) so a downstream consumer can read it
    with stdlib json without importing TF / numpy.
    """
    out_dir = Path(out_dir)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def save_history(history: dict, out_dir: Path) -> None:
    """Dump the training history dict as both JSON and CSV.

    Columns: epoch, train_loss, val_loss, val_rmse_raw.
    """
    out_dir = Path(out_dir)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "history.csv", "w") as f:
        f.write("epoch,train_loss,val_loss,val_rmse_raw\n")
        n = len(history.get("epoch", []))
        for i in range(n):
            f.write(
                f"{history['epoch'][i]},"
                f"{history['train_loss'][i]},"
                f"{history['val_loss'][i]},"
                f"{history['val_rmse_raw'][i]}\n"
            )


# ════════════════════════════════════════════════════════════════════════
# Encoder / decoder load — three flavours
# ════════════════════════════════════════════════════════════════════════

def _build_encoder_only(cfg: AutoencoderConfig, q_raw: int
                         ) -> tuple[tf.keras.Model, Standardizer]:
    """Build a fresh (random-weight) encoder + standardiser pair matching
    the cfg's architecture. Used internally by load_encoder / load_full.
    """
    act = tf.keras.activations.get(cfg.activation)
    bn_act = tf.keras.activations.get(cfg.bottleneck_activation)
    hidden = tuple(int(h) for h in cfg.hidden_dims)
    enc_layers: list[tf.keras.layers.Layer] = []
    for k, h in enumerate(hidden):
        enc_layers.append(tf.keras.layers.Dense(
            h, activation=act, name=f"enc_h{k}",
            kernel_initializer="glorot_uniform"))
    enc_layers.append(tf.keras.layers.Dense(
        int(cfg.latent_dim), activation=bn_act, name="enc_latent",
        kernel_initializer="glorot_uniform"))
    encoder = tf.keras.Sequential(enc_layers, name="encoder")
    standardizer = Standardizer(q_raw=q_raw)
    standardizer.build((None, q_raw))
    return encoder, standardizer


def _build_decoder_only(cfg: AutoencoderConfig, q_raw: int) -> tf.keras.Model:
    """Build a fresh decoder Sequential matching the cfg's architecture."""
    act = tf.keras.activations.get(cfg.activation)
    hidden = tuple(int(h) for h in cfg.hidden_dims)
    dec_layers: list[tf.keras.layers.Layer] = []
    for k, h in enumerate(reversed(hidden)):
        dec_layers.append(tf.keras.layers.Dense(
            h, activation=act, name=f"dec_h{k}",
            kernel_initializer="glorot_uniform"))
    dec_layers.append(tf.keras.layers.Dense(
        int(q_raw), activation="linear", name="dec_out",
        kernel_initializer="glorot_uniform"))
    decoder = tf.keras.Sequential(dec_layers, name="decoder")
    return decoder


def _load_standardizer_into(standardizer: Standardizer, out_dir: Path) -> None:
    """Populate a Standardizer layer from standardizer.npz on disk."""
    data = np.load(Path(out_dir) / "standardizer.npz")
    standardizer.mean.assign(data["mean"].astype(np.float32))
    standardizer.std.assign(data["std"].astype(np.float32))


def _arch_of(cfg) -> str:
    return str(getattr(cfg, "architecture", "mlp")).lower()


def _load_full_willatt(out_dir: Path, cfg: AutoencoderConfig
                        ) -> tf.keras.Model:
    """Rebuild a `WillattAutoencoder` from disk and load u / v / standardiser.

    Reads encoder.npz (u + integer layout fields), decoder.npz (v if
    untied, or just the tied marker), and standardizer.npz. The
    bilinear scatter/gather plumbing is rebuilt fresh from the npz's
    (T, K, alpha_max, l_max, compress_mode) fields, so the npz is the
    sole source of layout truth.
    """
    from SoapAutoencoder import WillattAutoencoder
    enc_npz = np.load(out_dir / "encoder.npz", allow_pickle=False)
    dec_npz = np.load(out_dir / "decoder.npz", allow_pickle=False)
    T = int(enc_npz["T"])
    K = int(enc_npz["K"])
    alpha_max = int(enc_npz["alpha_max"])
    l_max = int(enc_npz["l_max"])
    q_raw_T = int(enc_npz["q_raw_T"])
    compress_mode = str(enc_npz["compress_mode"])
    tied = bool(enc_npz["tied"]) if "tied" in enc_npz.files else bool(dec_npz["tied"])
    # Reflect the on-disk layout into cfg so __init__ rebuilds matching
    # scatter/gather indices. Caller's cfg may differ from the saved
    # one; the npz wins.
    cfg.architecture = "willatt"
    cfg.willatt_K = K
    cfg.tied_weights = tied
    cfg.alpha_max = alpha_max
    cfg.l_max = l_max
    cfg.compress_mode = compress_mode
    model = WillattAutoencoder(
        q_raw_T=q_raw_T, T=T,
        alpha_max=alpha_max, l_max=l_max, cfg=cfg)
    model(tf.zeros([1, q_raw_T], dtype=tf.float32))
    _load_standardizer_into(model.standardizer, out_dir)
    model.u.assign(enc_npz["u"].astype(np.float32))
    if not tied:
        model.v.assign(dec_npz["v"].astype(np.float32))
    return model


def _load_full_l_block_pca(out_dir: Path, cfg: AutoencoderConfig
                            ) -> tf.keras.Model:
    """Rebuild an `LBlockPCAAutoencoder` and load every per-l Dense + standardiser.

    Layout knobs (T, α, l_max, L, compress_mode, per_l_Q, per_l_K) come
    from the encoder.npz; the per-l kernels and biases are populated
    from both npz files.
    """
    from SoapAutoencoder import LBlockPCAAutoencoder
    enc_npz = np.load(out_dir / "encoder.npz", allow_pickle=False)
    dec_npz = np.load(out_dir / "decoder.npz", allow_pickle=False)
    T = int(enc_npz["T"])
    alpha_max = int(enc_npz["alpha_max"])
    l_max = int(enc_npz["l_max"])
    L = int(enc_npz["L"])
    q_raw = int(enc_npz["q_raw"])
    per_l_K = enc_npz["per_l_K"].tolist()
    compress_mode = str(enc_npz["compress_mode"])
    # Per-l hidden signature — present in npz files written after
    # nonlinear-l_block support landed; missing in older saves which
    # implicitly meant `()` (linear PCA-equivalent per-l branch).
    if "l_block_hidden_dims" in enc_npz.files:
        hidden_dims = tuple(int(h) for h in enc_npz["l_block_hidden_dims"])
    else:
        hidden_dims = ()
    cfg.architecture = "l_block_pca"
    cfg.alpha_max = alpha_max
    cfg.l_max = l_max
    cfg.compress_mode = compress_mode
    cfg.l_block_hidden_dims = hidden_dims
    # Force `_per_l_K` to match what was saved (the K-cap may have
    # silently fired during training; we honour the saved values).
    cfg.l_block_K = None
    cfg.latent_dim = int(sum(per_l_K))
    model = LBlockPCAAutoencoder(
        q_raw=q_raw, T=T, alpha_max=alpha_max,
        l_max=l_max, cfg=cfg)
    # Overwrite per_l_K with the saved values in case our fallback split
    # produced different per-block sizes (unlikely but possible if the
    # saved cfg used l_block_K with the cap firing).
    saved_per_l_K = [int(k) for k in per_l_K]
    if model._per_l_K != saved_per_l_K:
        raise ValueError(
            f"Loaded per_l_K={saved_per_l_K} does not match the rebuilt "
            f"model's per_l_K={model._per_l_K}. The saved config likely "
            "used l_block_K with a different cap; explicitly set "
            "cfg.l_block_K before calling the loader.")
    model(tf.zeros([1, q_raw], dtype=tf.float32))
    _load_standardizer_into(model.standardizer, out_dir)
    # Per-l branches are Sequentials with len(hidden_dims) + 1 internal
    # Dense layers (hidden chain + output). Iterate and assign by index.
    for l in range(L):
        enc_branch = model.per_l_encoders[l]
        dec_branch = model.per_l_decoders[l]
        for li, layer in enumerate(enc_branch.layers):
            layer.kernel.assign(
                enc_npz[f"enc_l{l}_layer{li}_W"].astype(np.float32))
            layer.bias.assign(
                enc_npz[f"enc_l{l}_layer{li}_b"].astype(np.float32))
        for li, layer in enumerate(dec_branch.layers):
            layer.kernel.assign(
                dec_npz[f"dec_l{l}_layer{li}_W"].astype(np.float32))
            layer.bias.assign(
                dec_npz[f"dec_l{l}_layer{li}_b"].astype(np.float32))
    return model


def _load_full_willatt_l_block(out_dir: Path, cfg: AutoencoderConfig
                                ) -> tf.keras.Model:
    """Rebuild a `WillattLBlockAutoencoder` and load u, v (if untied),
    every per-l Dense, and the standardiser.

    Layout knobs come from the encoder.npz; the per-l kernels and biases
    are populated from both npz files.
    """
    from SoapAutoencoder import WillattLBlockAutoencoder
    enc_npz = np.load(out_dir / "encoder.npz", allow_pickle=False)
    dec_npz = np.load(out_dir / "decoder.npz", allow_pickle=False)
    T = int(enc_npz["T"])
    K_species = int(enc_npz["K_species"])
    alpha_max = int(enc_npz["alpha_max"])
    l_max = int(enc_npz["l_max"])
    L = int(enc_npz["L"])
    q_raw_T = int(enc_npz["q_raw_T"])
    per_l_K = enc_npz["per_l_K"].tolist()
    compress_mode = str(enc_npz["compress_mode"])
    tied = (bool(enc_npz["tied"]) if "tied" in enc_npz.files
            else bool(dec_npz["tied"]))
    if "l_block_hidden_dims" in enc_npz.files:
        hidden_dims = tuple(int(h) for h in enc_npz["l_block_hidden_dims"])
    else:
        hidden_dims = ()
    cfg.architecture = "willatt_l_block"
    cfg.willatt_K = K_species
    cfg.tied_weights = tied
    cfg.alpha_max = alpha_max
    cfg.l_max = l_max
    cfg.compress_mode = compress_mode
    cfg.l_block_hidden_dims = hidden_dims
    cfg.l_block_K = None
    cfg.latent_dim = int(sum(per_l_K))
    model = WillattLBlockAutoencoder(
        q_raw_T=q_raw_T, T=T, alpha_max=alpha_max,
        l_max=l_max, cfg=cfg)
    saved_per_l_K = [int(k) for k in per_l_K]
    if model._per_l_K != saved_per_l_K:
        raise ValueError(
            f"Loaded per_l_K={saved_per_l_K} does not match the rebuilt "
            f"model's per_l_K={model._per_l_K}. The saved config likely "
            "used l_block_K with a different cap; explicitly set "
            "cfg.l_block_K before calling the loader.")
    model(tf.zeros([1, q_raw_T], dtype=tf.float32))
    _load_standardizer_into(model.standardizer, out_dir)
    model.u.assign(enc_npz["u"].astype(np.float32))
    if not tied:
        model.v.assign(dec_npz["v"].astype(np.float32))
    for l in range(L):
        enc_branch = model.per_l_encoders[l]
        dec_branch = model.per_l_decoders[l]
        for li, layer in enumerate(enc_branch.layers):
            layer.kernel.assign(
                enc_npz[f"enc_l{l}_layer{li}_W"].astype(np.float32))
            layer.bias.assign(
                enc_npz[f"enc_l{l}_layer{li}_b"].astype(np.float32))
        for li, layer in enumerate(dec_branch.layers):
            layer.kernel.assign(
                dec_npz[f"dec_l{l}_layer{li}_W"].astype(np.float32))
            layer.bias.assign(
                dec_npz[f"dec_l{l}_layer{li}_b"].astype(np.float32))
    return model


def load_encoder(out_dir: Path, cfg: AutoencoderConfig | None = None
                  ) -> tf.keras.Model:
    """Rebuild a callable encoder model from disk.

    Dispatches on `cfg.architecture` (read from config.json if `cfg` is
    None):
      - "willatt" / "l_block_pca" → returns the full structured model;
        call `model.encode(x)` or `model(x)`.
      - "mlp" (default) → returns a thin wrapper exposing only the
        encoder Sequential (standardiser + Dense stack), matching the
        legacy single-half load API.
    """
    out_dir = Path(out_dir)
    if cfg is None:
        cfg = load_config(out_dir)
    arch = _arch_of(cfg)
    if arch == "willatt":
        return _load_full_willatt(out_dir, cfg)
    if arch == "l_block_pca":
        return _load_full_l_block_pca(out_dir, cfg)
    if arch == "willatt_l_block":
        return _load_full_willatt_l_block(out_dir, cfg)
    # MLP path — encoder-only thin wrapper, as before.
    q_raw = int(np.load(out_dir / "standardizer.npz")["mean"].shape[0])
    encoder, standardizer = _build_encoder_only(cfg, q_raw)
    _load_standardizer_into(standardizer, out_dir)

    class _Encoder(tf.keras.Model):
        def __init__(self_):
            super().__init__(name="loaded_encoder")
            self_.standardizer = standardizer
            self_.encoder = encoder
            self_.q_raw = q_raw
            self_.latent_dim = int(cfg.latent_dim)

        def call(self_, x, training=None):
            return self_.encoder(self_.standardizer(x))

    model = _Encoder()
    model(tf.zeros([1, q_raw], dtype=tf.float32))
    wrapper = _EncoderWrapper(encoder)
    wrapper(tf.zeros([1, q_raw], dtype=tf.float32))
    wrapper.load_weights(out_dir / "encoder.weights.h5")
    return model


def load_decoder(out_dir: Path, cfg: AutoencoderConfig | None = None
                  ) -> tf.keras.Model:
    """Rebuild a callable decoder model from disk.

    Dispatches on `cfg.architecture`:
      - "willatt" / "l_block_pca" → returns the full structured model;
        call `model.decode(z)`.
      - "mlp" → thin decoder Sequential wrapper. Honours `tied_weights`:
        when True the loaded decoder kernels are unused (decode walks
        the encoder's transposed kernels), so the load just preserves
        the saved biases.
    """
    out_dir = Path(out_dir)
    if cfg is None:
        cfg = load_config(out_dir)
    arch = _arch_of(cfg)
    if arch == "willatt":
        return _load_full_willatt(out_dir, cfg)
    if arch == "l_block_pca":
        return _load_full_l_block_pca(out_dir, cfg)
    if arch == "willatt_l_block":
        return _load_full_willatt_l_block(out_dir, cfg)
    q_raw = int(np.load(out_dir / "standardizer.npz")["mean"].shape[0])
    decoder = _build_decoder_only(cfg, q_raw)
    standardizer = Standardizer(q_raw=q_raw)
    standardizer.build((None, q_raw))
    _load_standardizer_into(standardizer, out_dir)

    class _Decoder(tf.keras.Model):
        def __init__(self_):
            super().__init__(name="loaded_decoder")
            self_.standardizer = standardizer
            self_.decoder = decoder
            self_.q_raw = q_raw
            self_.latent_dim = int(cfg.latent_dim)

        def call(self_, z, training=None):
            return self_.standardizer(self_.decoder(z), inverse=True)

    model = _Decoder()
    model(tf.zeros([1, int(cfg.latent_dim)], dtype=tf.float32))
    wrapper = _DecoderWrapper(decoder)
    wrapper(tf.zeros([1, int(cfg.latent_dim)], dtype=tf.float32))
    wrapper.load_weights(out_dir / "decoder.weights.h5")
    return model


def load_full_autoencoder(out_dir: Path) -> tf.keras.Model:
    """Rebuild a fully-wired AE model (`SoapAutoencoder` /
    `WillattAutoencoder` / `LBlockPCAAutoencoder`).

    Dispatches on the saved cfg.architecture. The returned model
    exposes the same `encode` / `decode` / `call` API regardless of
    backbone.
    """
    out_dir = Path(out_dir)
    cfg = load_config(out_dir)
    arch = _arch_of(cfg)
    if arch == "willatt":
        return _load_full_willatt(out_dir, cfg)
    if arch == "l_block_pca":
        return _load_full_l_block_pca(out_dir, cfg)
    if arch == "willatt_l_block":
        return _load_full_willatt_l_block(out_dir, cfg)
    q_raw = int(np.load(out_dir / "standardizer.npz")["mean"].shape[0])
    model = SoapAutoencoder(q_raw=q_raw, cfg=cfg)
    model(tf.zeros([1, q_raw], dtype=tf.float32))
    _load_standardizer_into(model.standardizer, out_dir)
    enc_wrapper = _EncoderWrapper(model.encoder)
    enc_wrapper(tf.zeros([1, q_raw], dtype=tf.float32))
    enc_wrapper.load_weights(out_dir / "encoder.weights.h5")
    # Decoder kernels are loaded for parity with the saved file, but
    # under `tied_weights=True` the model's decode() path walks the
    # encoder's transposed kernels instead, leaving the loaded decoder
    # kernels untouched. We still load them so the trainable-variable
    # tree round-trips.
    dec_wrapper = _DecoderWrapper(model.decoder)
    dec_wrapper(tf.zeros([1, int(cfg.latent_dim)], dtype=tf.float32))
    dec_wrapper.load_weights(out_dir / "decoder.weights.h5")
    return model
