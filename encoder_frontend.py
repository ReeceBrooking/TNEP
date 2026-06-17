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
                            jacobian, encoder, progress: bool = True,
                            chunk_rows: int = 4096
                            ) -> tuple:
    """Transform a split's per-structure (descriptors, gradients) lists.

    Used by the data pipeline to apply the encoder once after the
    descriptor builder runs and before pad_and_stack converts the
    lists into batched tensors.

    **Batched implementation** — concatenates all per-structure
    descriptors into a single [N_total, Q_raw] array (and all per-atom
    gradients into a single [P_total, 3, Q_raw] array), encodes them in
    chunks of `chunk_rows` rows, then splits back into per-structure
    lists. The previous per-structure loop launched ~N_struct kernel
    calls for descriptors and ~N_struct·A_avg calls for gradients;
    on a 5768-structure × 30-atom dataset that was ~170k GPU op
    launches, which churns the allocator and stalls VS Code's IO.
    Batched, the same workload reduces to ~80 chunked encodes total.

    Parameters
    ----------
    descriptors_list : list of tf.Tensor / np.ndarray
        Per-structure descriptors, each shape [N_i, Q_raw].
    gradients_list : list of (list of tf.Tensor / np.ndarray)
        Per-structure, per-atom gradients. Each inner element has
        shape [M, 3, Q_raw] for the standard COO layout. Empty inner
        elements are passed through unchanged.
    jacobian : np.ndarray
        [Z, Q_raw] linear-encoder Jacobian.
    encoder : tf.keras.Model
        The loaded encoder. Used for the affine descriptor encode.
    progress : bool
        Whether to log periodic progress.
    chunk_rows : int
        Encoder row-chunk size. Bounds peak VRAM in the bilinear
        scatter to ~N · G_T² · L floats per chunk. 4096 ≈ 470 MB for
        T=6, αmax=10, L=8.

    Returns
    -------
    descriptors_out : list of tf.Tensor [N_i, Z]
    gradients_out :  list of (list of tf.Tensor [M, 3, Z])
    """
    n = len(descriptors_list)
    if n == 0:
        return [], []

    # ── Descriptors: concat → encode in chunks → split per structure ──
    desc_arrays: list = []
    desc_atom_counts: list = []
    for d_i in descriptors_list:
        if isinstance(d_i, np.ndarray):
            d_np = d_i.astype(np.float32, copy=False)
        else:
            d_np = d_i.numpy().astype(np.float32, copy=False)
        desc_arrays.append(d_np)
        desc_atom_counts.append(int(d_np.shape[0]))
    desc_concat = np.concatenate(desc_arrays, axis=0) if desc_arrays else \
        np.zeros((0, jacobian.shape[1]), dtype=np.float32)
    del desc_arrays
    N_total = int(desc_concat.shape[0])
    Z = int(jacobian.shape[0])

    if progress:
        print(f"  encoder front-end: encoding {N_total} atom descriptors "
              f"in chunks of {chunk_rows} …")

    # Chunked encode. The encoder's bilinear scatter is the OOM hot
    # path; bounded chunks keep peak VRAM under ~500 MB regardless of
    # dataset size.
    enc_chunks: list = []
    for start in range(0, N_total, chunk_rows):
        end = min(start + chunk_rows, N_total)
        z = encoder.encode(tf.constant(desc_concat[start:end])).numpy()
        enc_chunks.append(z)
    desc_enc_flat = (np.concatenate(enc_chunks, axis=0)
                      if enc_chunks else
                      np.zeros((0, Z), dtype=np.float32))
    del enc_chunks, desc_concat

    # Split per structure (numpy view splits — no copy).
    desc_offsets = np.cumsum([0] + desc_atom_counts).astype(np.int64)
    descriptors_out: list = [
        tf.constant(desc_enc_flat[desc_offsets[i]:desc_offsets[i + 1]])
        for i in range(n)]
    del desc_enc_flat

    # ── Gradients: concat ALL per-atom pair blocks → matmul-chunked ──
    # The gradient is a Jacobian-vector product (Jacobian only, no
    # affine offset because the constant mean's derivative vanishes).
    # `J · gv` is a single GEMM along the Q axis — no bilinear scatter,
    # so the per-chunk memory ceiling is much lower than for
    # descriptors and we can use a larger chunk.
    J_tf = tf.constant(jacobian, dtype=tf.float32)
    grad_chunks_per_struct: list = []
    flat_grad_arrays: list = []
    flat_grad_atom_lens: list = []   # number of (m·3) rows contributed per atom
    structure_atom_counts: list = []   # how many atoms per structure
    structure_pair_offsets: list = []  # per-structure flat-row offsets
    cur_offset = 0
    for i in range(n):
        gi = gradients_list[i] if i < len(gradients_list) else []
        n_atoms_i = len(gi)
        structure_atom_counts.append(n_atoms_i)
        structure_pair_offsets.append(cur_offset)
        for g_a in gi:
            if (g_a is None
                    or (hasattr(g_a, "shape") and 0 in tuple(g_a.shape))):
                flat_grad_atom_lens.append(0)
                continue
            g_np = (g_a if isinstance(g_a, np.ndarray)
                    else g_a.numpy())
            g_np = g_np.astype(np.float32, copy=False)
            M, C, Q = int(g_np.shape[0]), int(g_np.shape[1]), int(g_np.shape[2])
            # Flatten (M·C) so we can stack everything into one [P_total·3, Q]
            # tensor and matmul it as a single GEMM.
            flat_grad_arrays.append(g_np.reshape(M * C, Q))
            flat_grad_atom_lens.append(M * C)
            cur_offset += M * C
    structure_pair_offsets.append(cur_offset)  # sentinel for last split

    if flat_grad_arrays:
        gv_concat = np.concatenate(flat_grad_arrays, axis=0)
        del flat_grad_arrays
        P_flat_total = int(gv_concat.shape[0])
        if progress:
            print(f"  encoder front-end: projecting {P_flat_total} "
                  f"gradient rows through Jacobian …")
        # Single chunked matmul: out = gv @ J.T  (shape [P_flat_total, Z]).
        # Chunked along the pair axis so peak transient is bounded.
        gv_z_chunks: list = []
        gv_chunk_rows = max(chunk_rows * 8, 32_768)
        for start in range(0, P_flat_total, gv_chunk_rows):
            end = min(start + gv_chunk_rows, P_flat_total)
            gv_z_chunks.append(
                tf.linalg.matmul(
                    tf.constant(gv_concat[start:end]), J_tf,
                    transpose_b=True).numpy())
        gv_z_flat = np.concatenate(gv_z_chunks, axis=0)
        del gv_z_chunks, gv_concat
    else:
        gv_z_flat = np.zeros((0, Z), dtype=np.float32)

    # Split back per-(structure, atom).
    gradients_out: list = []
    gv_row_cursor = 0
    atom_cursor = 0
    for i in range(n):
        n_atoms_i = structure_atom_counts[i]
        gi_out: list = []
        for _ in range(n_atoms_i):
            n_rows = flat_grad_atom_lens[atom_cursor]
            atom_cursor += 1
            if n_rows == 0:
                gi_out.append(tf.zeros([0, 3, Z], dtype=tf.float32))
                continue
            sl = gv_z_flat[gv_row_cursor:gv_row_cursor + n_rows]
            gv_row_cursor += n_rows
            gi_out.append(
                tf.constant(sl.reshape(n_rows // 3, 3, Z)))
        gradients_out.append(gi_out)

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
