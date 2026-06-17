"""Pretrained-encoder front-end for TNEP.

Provides:

  load_linear_encoder(path)
      → returns (encoder_model, jacobian, q_raw, latent_dim)

      Loads a SOAP autoencoder run directory written by `encoder_io`,
      verifies the architecture is linear (no hidden layers in the
      encoder branches), and extracts the analytic affine Jacobian
      `J[Z, Q]` such that `encoder.encode(x) = J·x + b`. The Jacobian
      is the LINEAR part only; the per-channel mean μ is absorbed into
      the affine offset b and does NOT show up in J.

      The Jacobian is what lets the static preprocess path transform
      `grad_values[P, 3, Q] → grad_values[P, 3, Z]` exactly via
      `einsum('zq, p3q -> p3z', J, gv)`, since for any linear f:
          z = J·x + b   ⇒   ∂z/∂R = J · ∂x/∂R.

  apply_encoder_static(descriptors, grad_values, jacobian, encoder)
      → returns (descriptors_enc, grad_values_enc)

      One-shot transformation of a split's descriptor + gradient
      tensors. Used by the data pipeline when train_encoder=False.

The iterative path (train_encoder=True) does NOT use the static
helper — it keeps the encoder model in memory and is wired directly
into the TNEP forward pass.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from pathlib import Path


def _assert_linear_encoder(cfg, arch: str) -> None:
    """Verify the loaded encoder is linear so its Jacobian is well-defined.

    Raised at TNEP-side load time, not at autoencoder save time — a
    nonlinear AE is still a valid AE for reconstruction; it just can't
    be used as a TNEP front-end without per-step autodiff (which we
    are not supporting yet).
    """
    if arch == "mlp":
        # The MLP backbone has hidden_dims by default — only treat as
        # linear when there are none.
        hd = tuple(getattr(cfg, "hidden_dims", ()) or ())
        if hd:
            raise ValueError(
                f"encoder_path points to an 'mlp' encoder with "
                f"hidden_dims={hd}; TNEP front-end requires a linear "
                f"encoder. Re-train with hidden_dims=() or pick a "
                f"linear architecture.")
        return
    if arch in ("l_block_pca", "willatt_l_block"):
        hd = tuple(getattr(cfg, "l_block_hidden_dims", ()) or ())
        if hd:
            raise ValueError(
                f"encoder_path points to a {arch!r} encoder with "
                f"l_block_hidden_dims={hd}; TNEP front-end requires a "
                f"linear encoder. Re-train with l_block_hidden_dims=() "
                f"or pick a different encoder.")
        return
    if arch == "willatt":
        # Willatt is always linear by construction.
        return
    raise ValueError(
        f"encoder architecture {arch!r} not recognised. "
        f"Expected 'mlp' / 'willatt' / 'l_block_pca' / 'willatt_l_block'.")


def _extract_jacobian(encoder, q_raw: int, latent_dim: int) -> np.ndarray:
    """Recover J[Z, Q] from a linear/affine encoder by passing the
    canonical basis through it. Works for any affine f: ℝ^Q → ℝ^Z.

    For x in ℝ^Q, f(x) = J·x + b. Then:
        f(e_q) = J·e_q + b = J[:, q] + b
        f(0)   = b
    So J[:, q] = f(e_q) − f(0) and J as a whole = f(I) − f(0).

    Computed in batches of `chunk` columns so the intermediate
    [Q_chunk, Q_raw] identity slice and its [Q_chunk, latent_dim]
    output don't blow up VRAM for large Q.
    """
    chunk = min(1024, q_raw)
    zero_out = encoder.encode(tf.zeros([1, q_raw], dtype=tf.float32)).numpy()
    b_offset = zero_out[0]                                  # [Z]
    J = np.zeros((latent_dim, q_raw), dtype=np.float32)
    for start in range(0, q_raw, chunk):
        end = min(start + chunk, q_raw)
        # Identity columns [end-start, q_raw] — each row is e_q.
        block = np.zeros((end - start, q_raw), dtype=np.float32)
        block[np.arange(end - start), np.arange(start, end)] = 1.0
        z = encoder.encode(tf.constant(block)).numpy()      # [end-start, Z]
        # J column q is (f(e_q) − b).T
        J[:, start:end] = (z - b_offset[None, :]).T
    return J


def load_linear_encoder(path: str | Path) -> tuple:
    """Load a SOAP autoencoder run directory and return a usable bundle.

    Returns
    -------
    encoder : tf.keras.Model
        The reconstructed encoder (architecture-specific subclass).
        Exposes `encode(x) → z` and the live trainable variables.
    jacobian : np.ndarray, shape [Z, Q_raw]
        The analytic Jacobian. Linear-only encoders are guaranteed to
        have `encode(x) = J·x + b` with `b = encode(0)`.
    q_raw : int
        Input descriptor dimension expected by the encoder.
    latent_dim : int
        Output (latent) dimension after the encoder.
    """
    # Local import keeps a TNEP-only path from pulling in TF at import
    # time when no encoder is configured.
    import encoder_io
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"encoder_path {path!s} does not exist.")
    if not path.is_dir():
        raise NotADirectoryError(
            f"encoder_path {path!s} must point at a run directory "
            f"(containing config.json + encoder.* + standardizer.npz).")
    cfg = encoder_io.load_config(path)
    arch = str(getattr(cfg, "architecture", "mlp")).lower()
    _assert_linear_encoder(cfg, arch)
    encoder = encoder_io.load_full_autoencoder(path)
    # Resolve dims from the loaded model — these are the truth, not
    # the cfg (the cfg may have been edited).
    q_raw = int(getattr(encoder, "q_raw_T", None)
                or getattr(encoder, "q_raw", None))
    latent_dim = int(encoder.latent_dim)
    J = _extract_jacobian(encoder, q_raw, latent_dim)
    return encoder, J, q_raw, latent_dim


def setup_encoder_frontend(cfg) -> None:
    """Load the encoder bundle described by `cfg.encoder_path` once and
    stash it on cfg as private attributes for later consumers.

    A no-op when `cfg.encoder_path is None`. Sets:
        cfg._encoder            : the loaded encoder model
        cfg._encoder_J          : [Z, Q_raw] linear Jacobian
        cfg._encoder_q_raw      : input dim expected by the encoder
        cfg._encoder_latent_dim : output (Z) dim

    The data pipeline (`data.split` / `data.materialize_test_data`)
    reads `cfg._encoder_J` to decide whether to run the static
    preprocess. TNEP reads the bundle for the iterative path.
    """
    if getattr(cfg, "encoder_path", None) is None:
        # Guard: train_encoder=True without encoder_path is meaningless
        # — there's no encoder to co-train. Caught here rather than
        # silently treated as no-op so the user doesn't think iterative
        # encoder co-training is happening when it isn't.
        if bool(getattr(cfg, "train_encoder", False)):
            raise ValueError(
                "cfg.train_encoder=True requires cfg.encoder_path to "
                "point at a SoapAutoencoder run directory. With no "
                "encoder, there's nothing to iteratively co-train.")
        # Make sure no stale bundle persists if cfg was re-used.
        for attr in ("_encoder", "_encoder_J",
                     "_encoder_q_raw", "_encoder_latent_dim",
                     "_encoder_loaded_path"):
            if hasattr(cfg, attr):
                delattr(cfg, attr)
        return
    # Honour cfg re-use: if the user swapped encoder_path between
    # successive setup calls, re-load instead of silently returning the
    # stale bundle. Compare the stashed path to the current cfg field.
    if (hasattr(cfg, "_encoder_J")
            and getattr(cfg, "_encoder_loaded_path", None) == cfg.encoder_path):
        return
    if hasattr(cfg, "_encoder_J"):
        # Path changed — clear stale state before reloading.
        for attr in ("_encoder", "_encoder_J",
                     "_encoder_q_raw", "_encoder_latent_dim",
                     "_encoder_loaded_path"):
            if hasattr(cfg, attr):
                delattr(cfg, attr)
    encoder, J, q_raw, Z = load_linear_encoder(cfg.encoder_path)
    cfg._encoder = encoder
    cfg._encoder_J = J
    cfg._encoder_q_raw = int(q_raw)
    cfg._encoder_latent_dim = int(Z)
    cfg._encoder_loaded_path = cfg.encoder_path
    mode = ("iterative (live per-batch)" if getattr(cfg, "train_encoder", False)
            else "static preprocess (one-shot)")
    print(f"[encoder] loaded from {cfg.encoder_path}: "
          f"Q_raw={q_raw} → Z={Z}, mode = {mode}")


def apply_encoder_to_lists(descriptors_list, gradients_list,
                            jacobian, encoder, progress: bool = True
                            ) -> tuple:
    """Transform a split's per-structure (descriptors, gradients) lists.

    Used by the data pipeline to apply the encoder once after the
    descriptor builder runs and before pad_and_stack converts the
    lists into batched tensors.

    Parameters
    ----------
    descriptors_list : list of tf.Tensor / np.ndarray
        Per-structure descriptors, each shape [N_i, Q_raw].
    gradients_list : list of (list of tf.Tensor / np.ndarray)
        Per-structure, per-atom gradients. Each inner element has
        shape [M, 3, Q_raw] for the standard COO layout. Empty inner
        lists are honoured (e.g. when streaming-to-disk has stashed
        the gradient tensor elsewhere — but see the caller-side
        guard below; the disk-backed path is not supported here).
    jacobian : np.ndarray
        [Z, Q_raw] linear-encoder Jacobian.
    encoder : tf.keras.Model
        The loaded encoder. Used for the affine descriptor encode.
    progress : bool
        Whether to log periodic progress (every 500 structures).

    Returns
    -------
    descriptors_out : list of tf.Tensor [N_i, Z]
    gradients_out :  list of (list of tf.Tensor [M, 3, Z])
    """
    J_tf = tf.constant(jacobian, dtype=tf.float32)
    n = len(descriptors_list)
    descriptors_out: list = []
    gradients_out: list = []
    for i, d_i in enumerate(descriptors_list):
        d_tf = (tf.constant(d_i, dtype=tf.float32)
                if isinstance(d_i, np.ndarray) else tf.cast(d_i, tf.float32))
        d_enc = encoder.encode(d_tf)                 # [N_i, Z]
        descriptors_out.append(d_enc)
        gi_in = gradients_list[i] if i < len(gradients_list) else []
        gi_out: list = []
        for g_a in gi_in:
            if g_a is None or (hasattr(g_a, "shape") and 0 in g_a.shape):
                gi_out.append(g_a)
                continue
            gv_tf = (tf.constant(g_a, dtype=tf.float32)
                     if isinstance(g_a, np.ndarray) else tf.cast(g_a, tf.float32))
            gi_out.append(tf.einsum('zq,mcq->mcz', J_tf, gv_tf))
        gradients_out.append(gi_out)
        if progress and (i + 1) % 500 == 0:
            print(f"  encoder front-end: transformed {i + 1}/{n} structures")
    if progress:
        print(f"  encoder front-end: transformed {n}/{n} structures.")
    return descriptors_out, gradients_out


def apply_encoder_static(descriptors, grad_values, jacobian,
                          encoder) -> tuple:
    """Transform a split's (descriptors, grad_values) once and return
    the encoded tensors. Caller is responsible for discarding the raw
    versions and updating any dim references downstream.

    descriptors:  [B, A, Q_raw] (tf.Tensor or np.ndarray)
    grad_values:  [P, 3, Q_raw]
    jacobian:     [Z, Q_raw] (np.ndarray or tf.Tensor)
    encoder:      the loaded encoder model. Used to apply the AFFINE
                  encode (which subtracts mean) to descriptors. For
                  grad_values, the per-channel mean has zero
                  derivative, so the Jacobian alone is exact.

    Returns
    -------
    descriptors_enc : [B, A, Z]
    grad_values_enc : [P, 3, Z]
    """
    J = tf.convert_to_tensor(jacobian, dtype=tf.float32)
    # ── Descriptors: use the encoder's affine encode (centres + projects). ──
    if isinstance(descriptors, np.ndarray):
        desc_tf = tf.constant(descriptors, dtype=tf.float32)
    else:
        desc_tf = tf.cast(descriptors, tf.float32)
    B, A = desc_tf.shape[0], desc_tf.shape[1]
    desc_flat = tf.reshape(desc_tf, [B * A, -1])
    z_flat = encoder.encode(desc_flat)
    Z = z_flat.shape[-1]
    descriptors_enc = tf.reshape(z_flat, [B, A, Z])

    # ── Gradients: Jacobian-only contraction (mean has zero derivative). ──
    if isinstance(grad_values, np.ndarray):
        gv_tf = tf.constant(grad_values, dtype=tf.float32)
    else:
        gv_tf = tf.cast(grad_values, tf.float32)
    grad_values_enc = tf.einsum('zq,p3q->p3z'.replace('3', 'c'),
                                 J, gv_tf)
    return descriptors_enc, grad_values_enc
