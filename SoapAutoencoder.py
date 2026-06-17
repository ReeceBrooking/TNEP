"""SOAP autoencoder for alchemical descriptor compression.

Trains a symmetric MLP autoencoder over per-atom SOAP vectors to learn a
low-dimensional latent that is shared across all atomic species. The
compression is alchemical in the sense that a *single* encoder operates on
the raw descriptor regardless of central atom type — the species identity is
already folded into the SOAP itself via the species-pair channels, so the
latent only has to capture chemistry-agnostic structure.

The trained encoder can then preprocess SOAP vectors into a compressed
representation for downstream tasks (e.g. as a drop-in replacement for the
raw descriptor in TNEP).

Run:
    # Edit AutoencoderConfig's class-level defaults below to control the
    # run (data paths, architecture, training schedule), then:
    python SoapAutoencoder.py

I/O (save/load, run-directory layout, config and history persistence)
lives in `encoder_io.py` so this module stays focused on the model and
training loop.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════

class AutoencoderConfig:
    """Hyperparameters for the SOAP autoencoder training run.

    Plain class with type-annotated defaults — matches TNEPconfig style so
    the two configs can coexist in a single training script. CLI flags in
    `main()` overwrite the relevant fields.
    """

    # ── Dataset ────────────────────────────────────────────────────────
    # Path to .xyz with per-atom positions (used when --soap-cache is unset).
    data_path: str = "datasets/train.xyz"
    test_data_path: str | None = "datasets/test.xyz"
    # Precomputed per-atom SOAP arrays [N_atoms, Q_raw] in float32. When
    # set, the corresponding .xyz path is skipped entirely and no TNEP
    # imports are needed for that split.
    soap_cache: str | None = None
    soap_val_cache: str | None = None
    soap_test_cache: str | None = None
    # Species filter applied when reading the .xyz (mirrors TNEPconfig).
    allowed_species: list[int | str] | None = None #[6, 1, 8]
    filter_mode: str = "subset"
    test_ratio: float = 0.2
    total_N: int | None = None
    seed: int = 928375439201
    # Canonical species → integer-index map, mirroring TNEPconfig.types /
    # TNEPconfig.type_map. Resolved ONCE on the first SOAP build (or set
    # explicitly by the user) and reused for every subsequent .xyz read,
    # so train and test descriptors always land at the same channel
    # layout. None = auto-resolve on first read; pre-set as either a
    # list of Z values OR a dict {Z: index} to lock in a custom order.
    #
    # Resolution priority at first SOAP build:
    #   1. cfg.type_map  is set (dict) → use as-is.
    #   2. cfg.types     is set (list) → derive `type_map = {Z: i}`.
    #   3. cfg.allowed_species is set  → Z-sort its integer entries.
    #   4. Otherwise                   → Z-sort the union of species
    #                                    present in the .xyz being read.
    # Subsequent reads (e.g. test set) reuse the cached mapping; any
    # species in a later file that isn't in the map raises a clear
    # error before the SOAP build.
    types: list[int] | None = None
    type_map: dict[int, int] | None = None

    # ── SOAP descriptor parameters (must match the dataset) ────────────
    # Used only when building SOAP from .xyz; ignored under soap_cache.
    alpha_max: int = 7
    l_max: int = 7
    rcut_hard: float = 6.0
    rcut_soft: float = 5.5
    nf: float = 4.0
    skip_h_centers: bool = False
    # Descriptor backend: 0 = quippy (CPU, supports all compress_modes),
    # 1 = GPU TF (faster but `trivial`-only). Use mode 0 if you want
    # uncompressed SOAP — the TF backend will raise on anything else.
    descriptor_mode: int = 0
    # SOAP-turbo compression scheme. Determines how the (n, n', l) power
    # spectrum is reduced before the AE sees it:
    #   "trivial"  — keep only pairs touching a per-species radial pivot.
    #                Default; matches TNEP. O(T·α·L) channels.
    #   "none"     — full upper-triangle power spectrum, no compression.
    #                O((T·α)²·L) channels (e.g. 1848 for T=3, α=7, L=8).
    #                Requires descriptor_mode = 0 (quippy).
    # The Willatt backbone's layout walk dispatches on this — the
    # bilinear scatter/gather indices are recomputed for both Q_T and
    # Q_K under whichever mode is active.
    compress_mode: str = "trivial"

    # ── Autoencoder architecture ───────────────────────────────────────
    # Backbone selector:
    #   "mlp"          — symmetric MLP encoder/decoder (free linear projection).
    #                    Latent is an unstructured R^Z vector of size `latent_dim`.
    #   "willatt"      — bilinear species-projection AE. Latent IS a K-pseudo-
    #                    species SOAP vector that preserves the (n, n', l) tensor
    #                    structure; only the species axis is compressed via a
    #                    learned matrix u[T, K]. Willatt 2018's rank-K alchemical
    #                    compression specialised to an atom-centred AE.
    #   "l_block_pca"  — L independent per-l autoencoders. Channels at l
    #                    mix only with the same-l output (no cross-l
    #                    leakage). With `cfg.l_block_hidden_dims = ()`
    #                    each per-l branch is a single linear projection
    #                    (true block-diagonal PCA — L× fewer params than
    #                    full linear PCA). With non-empty hidden dims,
    #                    each per-l branch becomes an MLP with within-
    #                    block nonlinearity but still no cross-l mixing
    #                    (block-diagonal MLP — preserves the angular
    #                    structural prior while adding nonlinear capacity
    #                    inside each l-block). Same activation /
    #                    bottleneck_activation conventions as the "mlp"
    #                    backbone.
    #   "willatt_l_block" — composed two-stage compression. First applies
    #                    Willatt species projection T → K (preserving the
    #                    SOAP tensor structure with K pseudo-species), then
    #                    applies l_block_pca's per-l SVD on the K-species
    #                    SOAP. With `pca_init=True` AND linear branches,
    #                    both stages are computed analytically in one
    #                    chained closed-form solve (HOSVD then per-l SVD)
    #                    — no training needed. Otherwise both `u` and the
    #                    per-l weights are gradient-trained jointly.
    architecture: str = "willatt_l_block"
    # Latent vector dimension Z (mlp architecture only). The compressed
    # descriptor sits here.
    latent_dim: int = 256
    # Encoder hidden dims for the mlp backbone (decoder mirrors in reverse).
    # Set to () for a single-layer linear projection Q_raw → Z → Q_raw —
    # the PCA-equivalent baseline at MSE loss. Ignored for "willatt".
    hidden_dims: tuple[int, ...] = ()
    activation: str = "tanh"
    # Latent activation. Default linear lets the latent take any real
    # value (unbounded codes); use 'tanh' for bounded codes in [-1, 1].
    bottleneck_activation: str = "linear"
    # Tie encoder/decoder weights. For mlp: decoder W_l = encoder W_l^T,
    # halving param count. For willatt: decoder species matrix v = u^T so
    # the bilinear projection is symmetric (matches the strict Willatt 2018
    # formulation where one matrix mediates both encode and decode).
    tied_weights: bool = False
    # Willatt: number of pseudo-species K. None ⇒ auto = max(1, T-1) where
    # T is inferred from cfg.allowed_species. Smaller K = stronger species-
    # axis compression; K = T is no compression (only basis rotation).
    willatt_K: int | None = 4
    # Willatt: initialise u[T, K] from HOSVD on the per-atom power
    # spectrum's species-mode covariance. For the (current) linear
    # bilinear backbone this is the closed-form MSE optimum — training
    # is skipped entirely. SOAP power spectrum's symmetry in (α, β)
    # makes the mode-α covariance equal the mode-β covariance, so a
    # single SVD optimally seeds both species legs. Untied weights are
    # seeded with v = uᵀ (also the symmetric optimum).
    willatt_pca_init: bool = True
    # l_block_pca: per-l latent dimension K. Total latent dim = K · L.
    # Mirrors willatt_K's "per-axis output dim" semantics: set K small
    # for aggressive angular-block compression, K = Q_l (~60 for the
    # trivially-compressed CHO case) for no compression (basis rotation).
    # When None, falls back to splitting cfg.latent_dim across L blocks.
    l_block_K: int | None = 64
    # l_block_pca: per-l hidden layer sizes. Empty tuple () = linear per-l
    # branch (PCA-equivalent within each angular block). Non-empty =
    # per-l MLP encoder Q_l → h_1 → h_2 → ... → K (with `activation`
    # between layers, decoder mirrors). Adds within-block nonlinearity
    # without violating the cross-l block-diagonality. The same hidden
    # signature is applied independently per l. Output (latent) layer
    # uses `bottleneck_activation`; decoder output is linear.
    l_block_hidden_dims: tuple[int, ...] = ()
    # l_block_pca: initialise weights from per-l SVD of the standardised
    # training data instead of Glorot random. For LINEAR per-l branches
    # (`l_block_hidden_dims = ()`) this yields the closed-form PCA optimum
    # — training is then skipped entirely and only the analytic solution
    # is saved + evaluated. For NONLINEAR per-l branches it warm-starts
    # the first encoder and last decoder Dense layers from PCA components
    # (middle layers stay Glorot), then gradient descent continues from
    # the PCA-equivalent baseline. Saxe et al. 2014 show this converges
    # in O(1) epochs vs O(100) for random init on linear AEs; for shallow
    # nonlinear it cuts the warm-up phase by ~half. Ignored for the mlp
    # and willatt backbones.
    pca_init: bool = True

    # When True, override the "skip training" short-circuit that the
    # linear PCA / HOSVD paths normally take after analytic init. The
    # model is warm-started from the closed-form SVD / HOSVD optimum,
    # then continues gradient training for `cfg.epochs` from that
    # initialisation. This closes the gap between the analytic
    # ONE-axis optimum and the joint bilinear optimum (HOSVD is exact
    # only one mode at a time; the (u uᵀ) ⊗ (u uᵀ) projection couples
    # both species axes). Applies to `willatt`, `l_block_pca`, and
    # `willatt_l_block`. Ignored by the MLP backbone (no PCA path).
    pca_init_finetune: bool = True

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 6000
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0   # AdamW decoupled L2 on all dense weights
    grad_clip: float | None = 1.0
    # Optional warmup + cosine decay. When False, constant LR for the run.
    lr_schedule: str = "constant"   # 'constant' | 'cosine'
    warmup_frac: float = 0.05

    # ── Loss ───────────────────────────────────────────────────────────
    # mse   — mean squared error (default). Units (descriptor)².
    # rmse  — sqrt(MSE). Same optimum as MSE; gradient is amplified by
    #         1/(2·RMSE) near convergence (effectively a self-scheduling
    #         learning rate). Reported on the same numerical scale as
    #         `val_rmse_raw`, so the tqdm bar's `train` / `val` / `rmse`
    #         all line up in magnitude.
    # mae   — mean absolute error. Robust to outlier channels.
    # huber — Huber loss with `huber_delta` corner. MSE-like near zero,
    #         MAE-like in the tails.
    loss_type: str = "rmse"   # mse | rmse | mae | huber
    huber_delta: float = 1e-3
    # Per-feature z-score the SOAP inputs before fitting. Recommended ON:
    # raw SOAP channels span orders of magnitude (radial-low vs angular-high)
    # and an unnormalised MSE collapses onto whichever channels are loudest.
    normalize_inputs: bool = False
    # Mean-centre only (do not divide by std). Useful when the per-channel
    # variance is itself the signal you want to preserve.
    center_only: bool = False
    # Denoising autoencoder (Vincent et al. 2008): inject Gaussian noise
    # into the encoder input during training, but compute the loss
    # against the CLEAN target. Forces robustness to small input
    # perturbations — particularly useful when the encoder will be
    # deployed on noisy MD frames (atom positions with thermal jitter).
    # Value is the noise standard deviation in standardised units (i.e.
    # per-channel std multiplies the noise when `normalize_inputs=True`).
    # 0.0 = disabled (no input noise, standard reconstruction). Typical
    # value: 0.05 – 0.2. The noise is sampled fresh per minibatch.
    denoising_noise_stddev: float = 0.0

    # ── Output ─────────────────────────────────────────────────────────
    # Parent directory under which `encoder_io.setup_encoder_run_directory`
    # creates {timestamp}_z{latent}_h{hidden_signature}/. Final encoder,
    # decoder, standardiser, config, and history are written there at end
    # of training — there are no intermediate per-epoch checkpoints.
    output_dir: str = "models/autoencoder"
    log_every: int = 1
    val_every: int = 1


# ════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════

class Standardizer(tf.keras.layers.Layer):
    """Per-feature z-score layer with persistable mean/std buffers.

    The buffers are held as `tf.Variable`s on a Layer (not directly on the
    enclosing Model) so Keras's `save_weights`/`load_weights` round-trips
    them alongside the trainable kernels. Without this indirection,
    Model-level non-trainable Variables silently disappear from the
    `.weights.h5` file.
    """

    def __init__(self, q_raw: int, name: str = "standardizer"):
        super().__init__(name=name)
        self.q_raw = int(q_raw)

    def build(self, input_shape):
        self.mean = self.add_weight(
            name="mean", shape=(self.q_raw,),
            initializer="zeros", trainable=False)
        self.std = self.add_weight(
            name="std", shape=(self.q_raw,),
            initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, x, inverse: bool = False):
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / self.std


class SoapAutoencoder(tf.keras.Model):
    """Symmetric MLP autoencoder over per-atom SOAP vectors.

    Layout (latent_dim = Z, encoder hidden dims = [H1, H2, ..., Hk]):
        encoder: Q_raw → H1 → H2 → ... → Hk → Z
        decoder: Z → Hk → ... → H2 → H1 → Q_raw

    Hidden layers use `cfg.activation`; the latent layer uses
    `cfg.bottleneck_activation` (default linear) so the code can take any
    real value; the output layer is linear so reconstructed SOAPs land in
    the same range as the (normalised) input — no squashing.

    With `cfg.tied_weights=True`, the decoder reuses the encoder weights
    transposed (W_dec_l = W_enc_{k-l}^T). Biases stay independent. This
    halves the trainable parameter count and is the standard
    PCA-equivalent baseline at small latent_dim.
    """

    def __init__(self, q_raw: int, cfg: AutoencoderConfig):
        super().__init__(name="soap_autoencoder")
        self.q_raw = int(q_raw)
        self.cfg = cfg
        self.latent_dim = int(cfg.latent_dim)
        act = tf.keras.activations.get(cfg.activation)
        bn_act = tf.keras.activations.get(cfg.bottleneck_activation)
        hidden = tuple(int(h) for h in cfg.hidden_dims)

        # Per-feature standardisation. Held inside a Layer so its mean/std
        # tf.Variables get tracked by `save_weights`/`load_weights`;
        # populated by `fit_normalizer()` from the training set.
        self.normalize_inputs = bool(cfg.normalize_inputs)
        self.center_only = bool(cfg.center_only)
        self.standardizer = Standardizer(q_raw=self.q_raw)
        self.standardizer.build((None, self.q_raw))

        # Encoder. Each Dense layer: linear → activation.
        enc_layers: list[tf.keras.layers.Layer] = []
        in_dim = self.q_raw
        for k, h in enumerate(hidden):
            enc_layers.append(tf.keras.layers.Dense(
                h, activation=act, name=f"enc_h{k}",
                kernel_initializer="glorot_uniform"))
            in_dim = h
        enc_layers.append(tf.keras.layers.Dense(
            self.latent_dim, activation=bn_act, name="enc_latent",
            kernel_initializer="glorot_uniform"))
        self.encoder = tf.keras.Sequential(enc_layers, name="encoder")

        # Decoder (mirror). When tied, kernel matrices are taken from the
        # encoder in reverse order (transposed) inside `decode()`; the
        # Sequential below still holds the bias-only layers.
        self.tied_weights = bool(cfg.tied_weights)
        dec_layers: list[tf.keras.layers.Layer] = []
        in_dim = self.latent_dim
        rev_hidden = tuple(reversed(hidden))
        for k, h in enumerate(rev_hidden):
            dec_layers.append(tf.keras.layers.Dense(
                h, activation=act, name=f"dec_h{k}",
                use_bias=True,
                kernel_initializer="glorot_uniform"))
            in_dim = h
        dec_layers.append(tf.keras.layers.Dense(
            self.q_raw, activation="linear", name="dec_out",
            use_bias=True,
            kernel_initializer="glorot_uniform"))
        self.decoder = tf.keras.Sequential(dec_layers, name="decoder")

    # ── Normalisation ──────────────────────────────────────────────────

    def fit_normalizer(self, soap_train: np.ndarray) -> None:
        """Populate `mean` / `std` from the training SOAP array.

        Per-feature z-score keeps the MSE loss commensurate across the
        Q_raw channels — without it, the radial-low slots (variance ~ 1)
        dominate and the angular-high slots (variance ~ 1e-3) are
        effectively ignored.
        """
        # Centering is unconditional. `normalize_inputs` controls only
        # whether per-channel std rescaling happens; centering is free
        # (the decoder adds the mean back through the standardiser's
        # inverse path) and strictly improves any linear/bilinear
        # downstream stage (PCA, HOSVD, MLP first-layer bias).
        mean = np.mean(soap_train, axis=0).astype(np.float32)
        if self.normalize_inputs and not self.center_only:
            std = np.std(soap_train, axis=0).astype(np.float32)
            # Floor against degenerate (constant) channels — they carry no
            # information; the encoder will learn to ignore them anyway.
            std = np.maximum(std, 1e-6)
        else:
            std = np.ones_like(mean)
        self.standardizer.mean.assign(mean)
        self.standardizer.std.assign(std)

    @property
    def mean(self) -> tf.Variable:
        return self.standardizer.mean

    @property
    def std(self) -> tf.Variable:
        return self.standardizer.std

    def _standardize(self, x: tf.Tensor) -> tf.Tensor:
        return self.standardizer(x, inverse=False)

    def _unstandardize(self, x: tf.Tensor) -> tf.Tensor:
        return self.standardizer(x, inverse=True)

    # ── Forward ────────────────────────────────────────────────────────

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Map raw SOAP [B, Q_raw] → latent [B, Z]."""
        return self.encoder(self._standardize(x))

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Map latent [B, Z] → reconstructed raw SOAP [B, Q_raw]."""
        if not self.tied_weights:
            x_std = self.decoder(z)
        else:
            # Walk encoder layers in reverse, sharing kernels (transposed)
            # with the encoder. Biases come from the decoder sub-modules
            # so the optimiser still has those degrees of freedom.
            enc_layers = list(self.encoder.layers)
            dec_layers = list(self.decoder.layers)
            x_std = z
            for k, (enc_layer, dec_layer) in enumerate(
                    zip(reversed(enc_layers), dec_layers)):
                W_t = tf.transpose(enc_layer.kernel)
                b = dec_layer.bias
                x_std = tf.matmul(x_std, W_t) + b
                # All but the final layer use the activation; the output
                # layer stays linear (matches the untied decoder above).
                if k < len(dec_layers) - 1:
                    x_std = dec_layer.activation(x_std)
        return self._unstandardize(x_std)

    def call(self, x: tf.Tensor, training: bool | None = None):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ════════════════════════════════════════════════════════════════════════
# Willatt-style bilinear AE
# ════════════════════════════════════════════════════════════════════════
#
# Compresses the SOAP power spectrum by projecting the species axis from
# T → K pseudo-species through a learned matrix u[T, K], applied
# bilinearly on both species indices. Output is again a (trivially-
# compressed) SOAP vector at K pseudo-species, so the (n, n', l) tensor
# structure is preserved end-to-end.
#
# Math (trivially-compressed power spectrum, kept entries only):
#     p_K[J_a, k_a; J_b, k_b; l] = Σ_{α, β} u[α, J_a] · u[β, J_b]
#                                          · p_T[α, k_a; β, k_b; l]
# and the decoder mirrors with v[K, T] (or v = u^T when tied).
#
# The fold goes through a dense intermediate `[T·α, T·α, L]` tensor so
# tf.einsum can do the (α → J_a) × (β → J_b) bilinear contraction in one
# call. Memory peaks at (T·α)²·L per atom × batch — manageable for
# T·α ≤ 30 (e.g. T=3, α=7 → 21·21·8 = 3528 entries per atom).

def _trivial_compress_indices(T: int, alpha_max: int, l_max: int,
                               compress_mode: str = "trivial"
                               ) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, int]:
    """Enumerate the SOAP-turbo (n, n', l) layout for `T` species and
    `alpha_max` radial functions per species.

    Two modes:
      - `"trivial"` : keep only kept (n, n') upper-triangle pairs that
        touch a per-species pivot. Matches `descriptor_preprocess_layout`
        in the nep4_radial branch and what TNEP uses internally.
      - `"none"`    : keep every upper-triangle (n, n') pair — the full
        SOAP power spectrum, no compression.

    Returns:
        n_global  : [Q_raw] int   — global radial index of the first leg
                                    (α · alpha_max + k_a), 0-based
        np_global : [Q_raw] int   — global radial index of the second leg
        l_of_q    : [Q_raw] int   — angular momentum at each q
        dense_q   : [Q_raw] int   — flat index into a dense
                                    `[T·alpha_max · T·alpha_max · L]`
                                    tensor for the kept (n, n', l) entry
        Q_raw     : int           — number of kept entries
    """
    if compress_mode not in ("trivial", "none"):
        raise ValueError(
            f"compress_mode={compress_mode!r} not supported by the Willatt "
            "layout walk; expected 'trivial' or 'none'. (Linear / Darby "
            "modes need a different walk and aren't wired here yet.)")
    L = l_max + 1
    n_max_fortran = T * alpha_max
    # 1-based Fortran pivots — only consulted in "trivial" mode.
    pivots = {i * alpha_max + 1 for i in range(T)}
    n_global: list[int] = []
    np_global: list[int] = []
    l_of_q: list[int] = []
    dense_q: list[int] = []
    G = T * alpha_max
    for n in range(1, n_max_fortran + 1):
        for nprime in range(n, n_max_fortran + 1):
            if compress_mode == "trivial":
                kept = (n in pivots) or (nprime in pivots)
            else:   # "none"
                kept = True
            for l in range(L):
                if kept:
                    n0 = n - 1
                    n0p = nprime - 1
                    n_global.append(n0)
                    np_global.append(n0p)
                    l_of_q.append(l)
                    # Row-major flat index into [G, G, L]:
                    #   dense_q = n0 · G · L + n0p · L + l
                    dense_q.append(n0 * G * L + n0p * L + l)
    return (np.asarray(n_global, dtype=np.int32),
            np.asarray(np_global, dtype=np.int32),
            np.asarray(l_of_q, dtype=np.int32),
            np.asarray(dense_q, dtype=np.int32),
            len(n_global))


class _SpeciesProjector(tf.keras.layers.Layer):
    """Container for the bilinear species-projection matrix `u[T, K]` (or
    `v[K, T]`), held as a trainable tf.Variable.

    The matrix lives inside a Layer rather than directly on the enclosing
    Model so Keras's auto-tracking includes it in `trainable_variables`
    and `save_weights`/`load_weights` round-trips it. Model-level
    Variables are silently skipped — same issue we solved earlier for
    the Standardiser's mean/std buffers.
    """

    def __init__(self, shape: tuple[int, int], seed: int,
                 name: str = "species_projector"):
        super().__init__(name=name)
        self._shape = tuple(int(s) for s in shape)
        self._seed = int(seed)

    def build(self, input_shape):
        rows, cols = self._shape
        limit = float(np.sqrt(6.0 / (rows + cols)))
        rng = np.random.default_rng(self._seed)
        init = rng.uniform(
            -limit, limit, size=self._shape).astype(np.float32)
        self.W = self.add_weight(
            name="W", shape=self._shape, dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(init),
            trainable=True)
        super().build(input_shape)


class WillattAutoencoder(tf.keras.Model):
    """Bilinear species-projection AE matching Willatt (2018, [arXiv:1807.00236]).

    Encode: project the T-species trivially-compressed SOAP power spectrum
    onto K pseudo-species via `u[T, K]`, applied bilinearly on both species
    legs. The result is a K-species trivially-compressed SOAP — same tensor
    layout, fewer species channels.

    Decode: project K → T via `v[K, T]` (or `u^T` when `tied_weights=True`)
    in the same bilinear form.

    All bilinear ops route through a dense `[T·α, T·α, L]` intermediate
    so a single `einsum` can do the (α → J) projection. The dense
    representation is sparse in practice (only the trivially-kept entries
    are non-zero), so memory use is proportional to `(T·α)² · L · batch`
    — fine for typical SOAP knobs.
    """

    def __init__(self, q_raw_T: int, T: int, alpha_max: int, l_max: int,
                 cfg: AutoencoderConfig):
        super().__init__(name="willatt_autoencoder")
        K_cfg = cfg.willatt_K
        if K_cfg is None:
            K = max(1, T - 1)
        else:
            K = int(K_cfg)
        if not 1 <= K <= T:
            raise ValueError(
                f"willatt_K={K} out of range [1, T={T}]; either pick a "
                "smaller K or check cfg.allowed_species.")

        self.cfg = cfg
        self.T = int(T)
        self.K = int(K)
        self.alpha_max = int(alpha_max)
        self.l_max = int(l_max)
        self.L = int(l_max) + 1
        self.q_raw = int(q_raw_T)
        self.q_raw_T = int(q_raw_T)
        self.tied_weights = bool(cfg.tied_weights)
        self.normalize_inputs = bool(cfg.normalize_inputs)
        self.center_only = bool(cfg.center_only)
        self.compress_mode = str(getattr(cfg, "compress_mode", "trivial"))

        # Layouts for both species counts under the cfg's compress mode.
        # We only need the dense-tensor index map and the total Q count;
        # the (n_global, n'_global, l_of_q) decomposition is implicit in
        # dq. The same mode is used for both T and K so the bilinear
        # scatter/gather lines up.
        (_, _, _, dq_T, q_T) = _trivial_compress_indices(
            self.T, self.alpha_max, self.l_max, self.compress_mode)
        (_, _, _, dq_K, q_K) = _trivial_compress_indices(
            self.K, self.alpha_max, self.l_max, self.compress_mode)
        if q_T != self.q_raw_T:
            raise ValueError(
                f"Layout walk produced Q_raw={q_T} for compress_mode="
                f"{self.compress_mode!r}, but the data has Q_raw="
                f"{self.q_raw_T}. Check cfg.allowed_species / alpha_max / "
                "l_max / compress_mode consistency with the SOAP build.")
        self.q_raw_K = int(q_K)
        self.latent_dim = int(q_K)   # for downstream reporting symmetry

        # Constant index tensors for the scatter/gather hops between
        # trivially-compressed and dense layouts. Built as np arrays then
        # frozen as tf.constants — the layout walk is python-only.
        # Dense flat dim: G_T·G_T·L for the T side, G_K·G_K·L for K side.
        self._G_T = self.T * self.alpha_max
        self._G_K = self.K * self.alpha_max
        self._dense_T_flat = int(self._G_T * self._G_T * self.L)
        self._dense_K_flat = int(self._G_K * self._G_K * self.L)
        # Scatter targets (q_T → dense_T flat index) and (q_K → dense_K).
        self._dense_q_T = tf.constant(dq_T[:, None], dtype=tf.int32)
        self._dense_q_K = tf.constant(dq_K[:, None], dtype=tf.int32)
        # Gather sources (dense_T flat → q_T) and (dense_K → q_K) — used
        # to pull the kept entries back out of the dense intermediate.
        self._gather_q_T = tf.constant(dq_T, dtype=tf.int32)
        self._gather_q_K = tf.constant(dq_K, dtype=tf.int32)

        # Per-feature standardiser in T-SOAP space, same as the MLP path.
        self.standardizer = Standardizer(q_raw=self.q_raw_T)
        self.standardizer.build((None, self.q_raw_T))

        # Encoder species matrix u[T, K], held inside a Layer so Keras
        # tracks it for save_weights / trainable_variables. Initialiser
        # is Glorot uniform with fan_in + fan_out = T + K.
        seed = int(getattr(cfg, "seed", 0))
        self._u_layer = _SpeciesProjector(
            shape=(self.T, self.K), seed=seed, name="encoder_u")
        self._u_layer.build((None,))
        # Decoder species matrix v[K, T]. Untied: independent Glorot init.
        # Tied (v ≡ uᵀ) is handled lazily inside decode() — no storage.
        if not self.tied_weights:
            # `seed ^ 0xC0DE`: any nonzero bit-pattern would do; 0xC0DE
            # (decimal 49374) is arbitrary but cheap to recognise in
            # logs. The XOR guarantees u and v get independent Glorot
            # draws while remaining deterministic at fixed cfg.seed.
            self._v_layer = _SpeciesProjector(
                shape=(self.K, self.T), seed=seed ^ 0xC0DE,
                name="decoder_v")
            self._v_layer.build((None,))
        else:
            self._v_layer = None

    # ── Normalisation ──────────────────────────────────────────────────

    def fit_normalizer(self, soap_train: np.ndarray) -> None:
        # Always centre per channel — even when `normalize_inputs=False`.
        # Centering is essentially free (the decoder adds the mean back
        # through the standardiser's inverse path) and strictly improves
        # the bilinear projection: an un-centered HOSVD wastes one
        # species direction on the DC offset (the mean is a rank-1
        # component of XᵀX) so only K−1 directions remain for actual
        # species-to-species variation. With centering all K directions
        # carry variation around the mean — directly comparable to R².
        # `normalize_inputs` now controls only the per-channel std
        # rescaling; centering is unconditional.
        mean = np.mean(soap_train, axis=0).astype(np.float32)
        if self.normalize_inputs and not self.center_only:
            std = np.std(soap_train, axis=0).astype(np.float32)
            std = np.maximum(std, 1e-6)
        else:
            std = np.ones_like(mean)
        self.standardizer.mean.assign(mean)
        self.standardizer.std.assign(std)

    @property
    def mean(self) -> tf.Variable:
        return self.standardizer.mean

    @property
    def std(self) -> tf.Variable:
        return self.standardizer.std

    @property
    def u(self) -> tf.Variable:
        return self._u_layer.W

    @property
    def v(self) -> tf.Variable | None:
        return None if self._v_layer is None else self._v_layer.W

    def _standardize(self, x):
        return self.standardizer(x, inverse=False)

    def _unstandardize(self, x):
        return self.standardizer(x, inverse=True)

    # ── HOSVD initialisation ──────────────────────────────────────────

    def pca_initialize_weights(self, soap_train: np.ndarray) -> bool:
        """HOSVD init of u[T, K] via the species-mode covariance.

        Builds the [T, T] covariance of the standardised training
        power spectrum along the first species axis (= the second
        species axis by SOAP symmetry), then takes the top-K
        eigenvectors as the species embedding. This IS the closed-form
        Tucker-K Willatt optimum:

            recon = (u uᵀ) · p · (u uᵀ)
            argmin_u ‖p - recon‖²  ⇔  span(u) = top-K eigvec of mode-α cov

        For the SOAP power spectrum (symmetric in (α, β)), the optimal
        u for the unconstrained Tucker decomposition coincides with
        Willatt's symmetric-bilinear constraint — no accuracy loss for
        imposing v = uᵀ (the tied form). Untied weights are seeded
        with v = uᵀ as well, by the same symmetry argument.

        Chunked over atoms to keep peak memory at O(B · (T·α)² · L)
        rather than O(N · (T·α)² · L) — fine for any realistic SOAP
        knobs even on the GPU's ~10 GB budget.

        Returns:
            True — Willatt's current backbone is purely linear bilinear,
                   so the HOSVD solution is the global MSE optimum and
                   no training can improve on it. Caller skips train().
        """
        soap_train = np.asarray(soap_train, dtype=np.float32)
        mean = self.standardizer.mean.numpy()
        std = self.standardizer.std.numpy()
        X_std = (soap_train - mean) / std                  # [N, Q_raw_T]
        gather = self._gather_q_T.numpy()                  # [Q_raw_T]
        L = self.L
        T_ = self.T
        alpha = self.alpha_max
        dense_flat_dim = int(self._dense_T_flat)

        # Accumulate the species-mode covariance C[α, α'] over chunks.
        # Per-chunk dense tensor has shape [B, T, α, T, α, L]; the
        # einsum 'bACBDl,bECBDl->AE' contracts every axis except the
        # first species axis on each operand. C is the [T, T] Gram
        # matrix of the row-space of M_α (the mode-α matricisation).
        C = np.zeros((T_, T_), dtype=np.float64)
        chunk = 512
        N = X_std.shape[0]
        for i in range(0, N, chunk):
            xs = X_std[i:i + chunk]                        # [B, Q_raw]
            B_ = xs.shape[0]
            # Scatter into dense [B, dense_T_flat] then reshape. Use
            # advanced indexing for cache-friendly placement.
            dense = np.zeros((B_, dense_flat_dim), dtype=np.float32)
            dense[:, gather] = xs
            d6 = dense.reshape(B_, T_, alpha, T_, alpha, L)
            C += np.einsum('bACBDl,bECBDl->AE', d6, d6,
                           optimize=True).astype(np.float64)

        # Symmetrise C (it is symmetric in exact arithmetic; tiny
        # asymmetry can arise from float32 accumulation order).
        C = 0.5 * (C + C.T)
        # eigh returns eigenvalues ASCENDING. Pull the top K columns
        # and reverse so column 0 is the most important component.
        eigvals, eigvecs = np.linalg.eigh(C)
        u_init = eigvecs[:, -self.K:][:, ::-1].astype(np.float32)
        self._u_layer.W.assign(u_init)
        if not self.tied_weights:
            # By SOAP symmetry the optimal v also satisfies span(v) =
            # span(u); the simplest fit is v = uᵀ (identical projection
            # on the second leg). Training would only drift them apart
            # if nonlinear capacity were present — which the current
            # bilinear backbone doesn't have.
            self._v_layer.W.assign(u_init.T)

        # Report the captured-variance fraction so the user can sanity-
        # check how much information K species retains.
        var_top = float(np.sum(eigvals[-self.K:]))
        var_total = float(np.sum(eigvals))
        if var_total > 0:
            ratio = var_top / var_total
            print(f"[willatt] HOSVD init: top-{self.K} species components "
                  f"capture {ratio:.4%} of total species-mode variance "
                  f"(out of T={T_} species).")
        return True

    # ── Bilinear scatter/gather hops ───────────────────────────────────

    def _to_dense_T(self, x_std):
        """Scatter [B, q_raw_T] → [B, G_T, G_T, L] (zeros at non-kept slots)."""
        B = tf.shape(x_std)[0]
        # tf.scatter_nd writes the SOURCE_AXES leading; we need a per-batch
        # scatter. Transpose to put Q first, scatter, transpose back.
        flat = tf.transpose(x_std)                                # [Q_T, B]
        dense_flat = tf.scatter_nd(
            self._dense_q_T, flat,
            shape=[self._dense_T_flat, B])                        # [dense_T, B]
        dense_flat = tf.transpose(dense_flat)                     # [B, dense_T]
        return tf.reshape(
            dense_flat, [B, self._G_T, self._G_T, self.L])

    def _from_dense_K(self, dense_K):
        """Gather [B, G_K, G_K, L] → [B, q_raw_K] (drops non-kept slots)."""
        B = tf.shape(dense_K)[0]
        dense_flat = tf.reshape(dense_K, [B, self._dense_K_flat])
        return tf.gather(dense_flat, self._gather_q_K, axis=1)

    def _to_dense_K(self, z):
        B = tf.shape(z)[0]
        flat = tf.transpose(z)                                    # [Q_K, B]
        dense_flat = tf.scatter_nd(
            self._dense_q_K, flat,
            shape=[self._dense_K_flat, B])                        # [dense_K, B]
        dense_flat = tf.transpose(dense_flat)                     # [B, dense_K]
        return tf.reshape(
            dense_flat, [B, self._G_K, self._G_K, self.L])

    def _from_dense_T(self, dense_T):
        B = tf.shape(dense_T)[0]
        dense_flat = tf.reshape(dense_T, [B, self._dense_T_flat])
        return tf.gather(dense_flat, self._gather_q_T, axis=1)

    # ── Forward ────────────────────────────────────────────────────────

    def encode(self, x):
        """SOAP-T [B, q_raw_T] → SOAP-K [B, q_raw_K] via bilinear u.

        Reshape splits the flat dense `[G_T, G_T, L]` axis into species
        and within-species-radial pieces `[T, α, T, α, L]`; the einsum
        contracts species index A → J on leg 1 and B → M on leg 2 through
        `u[T, K]`, yielding `[K, α, K, α, L]`. Flatten back to dense_K and
        gather the kept K-slot entries.
        """
        x_std = self._standardize(x)
        dense_T = self._to_dense_T(x_std)               # [B, G_T, G_T, L]
        b = tf.shape(dense_T)[0]
        d_T = tf.reshape(
            dense_T,
            [b, self.T, self.alpha_max, self.T, self.alpha_max, self.L])
        d_K = tf.einsum('bACBDl,AJ,BM->bJCMDl', d_T, self.u, self.u)
        dense_K = tf.reshape(d_K, [b, self._G_K, self._G_K, self.L])
        return self._from_dense_K(dense_K)

    def decode(self, z):
        """SOAP-K [B, q_raw_K] → reconstructed SOAP-T [B, q_raw_T] via v."""
        dense_K = self._to_dense_K(z)
        b = tf.shape(dense_K)[0]
        v = tf.transpose(self.u) if self.tied_weights else self.v
        d_K = tf.reshape(
            dense_K,
            [b, self.K, self.alpha_max, self.K, self.alpha_max, self.L])
        d_T = tf.einsum('bJCMDl,JA,MB->bACBDl', d_K, v, v)
        dense_T = tf.reshape(d_T, [b, self._G_T, self._G_T, self.L])
        x_std = self._from_dense_T(dense_T)
        return self._unstandardize(x_std)

    def call(self, x, training=None):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ════════════════════════════════════════════════════════════════════════
# L-block linear PCA (angular-preserving)
# ════════════════════════════════════════════════════════════════════════
#
# An ablation between the unstructured MLP PCA (mixes everything) and the
# Willatt species-projection AE (mixes species only). The structural prior
# here is BLOCK-DIAGONAL IN ANGULAR MOMENTUM l: channels at l mix only
# with channels at the same l. Each l ∈ [0, L) gets its own independent
# linear AE; the global latent is the concatenation of per-l latents.
#
# Why: the angular-momentum-block prior is the same one that TNEP's
# `descriptor_mixing_arch = 'l_aware'` validated empirically — l ≠ l'
# channels carry distinct physical content (radial, dipolar, angular
# detail) and shouldn't be linearly mixed.
#
# Free parameters at total latent dim Z: 2 · Σ_l Q_l · K_l. With K_l =
# Z/L uniform and Q_l = Q/L uniform (typical for trivially-compressed
# SOAP-turbo with l as inner loop), that's 2 · L · (Q/L) · (Z/L) =
# 2 · Q · Z / L — **L× fewer params** than a full linear PCA at the
# same latent size.

class LBlockPCAAutoencoder(tf.keras.Model):
    """L-block-diagonal linear AE — one independent linear PCA per l.

    Encode: for each angular momentum l, take the Q_l SOAP channels at
    that l and project linearly to K_l latent dimensions. The global
    latent is `concat([z_l=0, z_l=1, …, z_{L-1}])` of total size Z.

    Decode: per-l linear projection back to Q_l channels, scattered into
    their original positions in the flat Q vector.

    `cfg.latent_dim` is split across L by integer division; the
    remainder Z mod L is distributed to the lowest-l blocks (most
    physically informative). With Z=256, L=8 → K_l = 32 for all l.
    """

    def __init__(self, q_raw: int, T: int, alpha_max: int, l_max: int,
                 cfg: AutoencoderConfig):
        super().__init__(name="l_block_pca_autoencoder")
        self.cfg = cfg
        self.T = int(T)
        self.alpha_max = int(alpha_max)
        self.l_max = int(l_max)
        self.L = int(l_max) + 1
        self.q_raw = int(q_raw)
        self.normalize_inputs = bool(cfg.normalize_inputs)
        self.center_only = bool(cfg.center_only)
        self.compress_mode = str(getattr(cfg, "compress_mode", "trivial"))

        # Layout walk to discover which flat-q indices belong to each l.
        # The (l_of_q) array maps every kept q to its angular momentum.
        _, _, l_of_q, _, q_check = _trivial_compress_indices(
            self.T, self.alpha_max, self.l_max, self.compress_mode)
        if q_check != self.q_raw:
            raise ValueError(
                f"Layout walk produced Q={q_check} under compress_mode="
                f"{self.compress_mode!r}, but data has Q={self.q_raw}. "
                "Check cfg.allowed_species / alpha_max / l_max / "
                "compress_mode consistency with the SOAP build.")

        # Per-l index lists. `gather_l[l]` collects the q-indices at l.
        per_l_indices = [np.where(l_of_q == l)[0].astype(np.int32)
                         for l in range(self.L)]
        self._per_l_Q = [int(idx.size) for idx in per_l_indices]
        self._per_l_gather = [tf.constant(idx) for idx in per_l_indices]

        # Build a scatter-index tensor for the inverse hop (per-l latent
        # blocks → flat q) used at the end of decode(). For each l, the
        # per-l reconstructions land in positions per_l_indices[l].
        # tf.scatter_nd takes a single [Q_l, 1] index array per l, but
        # the flat scatter is cleaner: concat all indices, then one
        # scatter_nd writes the assembled per-l vectors into a [Q]
        # tensor with no overlaps (each q is assigned exactly one l).
        flat_scatter_idx = np.concatenate(per_l_indices, axis=0)[:, None]
        self._flat_scatter_idx = tf.constant(flat_scatter_idx, dtype=tf.int32)

        # Per-l latent sizes.
        # Priority:
        #   1. cfg.l_block_K (preferred) — same K for every l, total Z = K·L.
        #      Mirrors willatt_K's per-axis-output-dim semantics.
        #   2. fall back to splitting cfg.latent_dim across L (with the
        #      remainder distributed to the lowest-l blocks, which carry
        #      the densest information).
        l_block_K = getattr(cfg, "l_block_K", None)
        if l_block_K is not None:
            K_per_l = int(l_block_K)
            if K_per_l < 1:
                raise ValueError(
                    f"l_block_K={K_per_l} must be ≥ 1 (one latent unit "
                    "minimum per angular momentum block).")
            # Per-l Q lower-bounds K_per_l — can't extract more directions
            # than channels exist. Cap and log per-block when it fires so
            # the user sees that effective Z < K·L.
            self._per_l_K = [min(K_per_l, Q_l) for Q_l in self._per_l_Q]
            capped = [(l, Q_l) for l, Q_l in enumerate(self._per_l_Q)
                      if K_per_l > Q_l]
            if capped:
                clipped = ", ".join(f"l={l} → K={Q_l}" for l, Q_l in capped)
                # If hidden_dims is non-empty and every per-l block has
                # K_capped == Q_l, the network is `Q_l → h → Q_l` — pure
                # nonlinearity with no dimensional bottleneck. Flag it
                # so the user knows compression is architectural only.
                hd_set = bool(getattr(cfg, "l_block_hidden_dims", ()))
                fully_uncapped = all(
                    self._per_l_K[l] == Q_l
                    for l, Q_l in enumerate(self._per_l_Q))
                note = ""
                if hd_set and fully_uncapped:
                    note = (" (MLP mode: K = Q_l for every l means no "
                            "dimensional compression — only the per-l "
                            "MLP's hidden layers do any work).")
                print(
                    f"[l_block_pca] WARNING: l_block_K={K_per_l} exceeds "
                    f"per-l channel count for some l; capped to Q_l "
                    f"({clipped}). Effective latent_dim = "
                    f"{int(sum(self._per_l_K))} (not K·L = "
                    f"{K_per_l * self.L}).{note}")
        else:
            total_K = int(cfg.latent_dim)
            if total_K < self.L:
                raise ValueError(
                    f"latent_dim={total_K} < L={self.L} — at least one "
                    "latent unit per angular momentum is required.")
            base = total_K // self.L
            rem = total_K % self.L
            self._per_l_K = [base + (1 if l < rem else 0)
                              for l in range(self.L)]
        self.latent_dim = int(sum(self._per_l_K))
        # Sync the realised total back onto cfg so downstream readers
        # (logging, run-dir naming, save_config) see the actual value
        # rather than the requested-but-possibly-capped one.
        cfg.latent_dim = int(self.latent_dim)

        # tied_weights is not yet supported for this architecture. Honour
        # the cfg as far as logging it — the per-l Dense layers always
        # have independent kernels, so the encoder/decoder remain
        # untied in practice. A future implementation would override
        # decode() to walk encoder kernels transposed (mirroring the MLP
        # tied path).
        if bool(getattr(cfg, "tied_weights", False)):
            print(
                "[l_block_pca] WARNING: tied_weights=True ignored for "
                "this architecture; encoder and decoder per-l kernels "
                "are independent. Set cfg.tied_weights = False to "
                "silence this notice.")

        # Per-l hidden-layer signature. Empty tuple ⇒ each branch is a
        # single linear Dense (the original "block-diagonal PCA" mode).
        # Non-empty ⇒ each branch is an MLP: Q_l → h_1 → … → K_l with
        # `cfg.activation` between hidden layers, `bottleneck_activation`
        # on the latent layer, linear on the decoder output.
        # Accept None / scalar int / iterable forms — a bare `64` in
        # the cfg is treated as `(64,)`.
        _hd = getattr(cfg, "l_block_hidden_dims", ())
        if _hd is None:
            _hd = ()
        elif isinstance(_hd, (int, np.integer)):
            _hd = (int(_hd),)
        try:
            _hd = tuple(int(h) for h in _hd)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"cfg.l_block_hidden_dims={_hd!r} could not be coerced "
                f"to a tuple of ints: {e}") from None
        # Reject non-positive widths up-front — Dense(0) builds but
        # produces empty activations; Dense(-N) raises an opaque Keras
        # error during branch construction.
        bad = [h for h in _hd if h < 1]
        if bad:
            raise ValueError(
                f"cfg.l_block_hidden_dims contains non-positive entries "
                f"{bad}; every hidden width must be ≥ 1.")
        self.l_block_hidden_dims = _hd
        # `activation = "linear"` with non-empty hidden_dims is an
        # algebraic collapse — the linear stack reduces to a single
        # linear projection, wasting the extra parameters. Warn so the
        # user knows their MLP isn't actually nonlinear.
        if self.l_block_hidden_dims and str(cfg.activation).lower() in (
                "linear", "identity", "none"):
            print(
                f"[l_block_pca] WARNING: cfg.activation="
                f"{cfg.activation!r} with non-empty l_block_hidden_dims="
                f"{self.l_block_hidden_dims} collapses algebraically to "
                "a single linear projection. Set cfg.activation to "
                "silu / tanh / gelu / relu to get the nonlinearity, or "
                "set l_block_hidden_dims=() for the pure linear PCA "
                "baseline (fewer params, same result).")
        act = tf.keras.activations.get(cfg.activation)
        bn_act = tf.keras.activations.get(cfg.bottleneck_activation)

        # Per-l encoder / decoder branches are stored as Sequentials —
        # Keras tracks them as Model attributes. Single-Dense branches
        # are the linear PCA mode; multi-Dense branches give nonlinear
        # per-l capacity without violating the block-diagonal-in-l prior.
        def _enc_branch(l):
            layers: list[tf.keras.layers.Layer] = []
            for k, h in enumerate(self.l_block_hidden_dims):
                layers.append(tf.keras.layers.Dense(
                    int(h), activation=act,
                    kernel_initializer="glorot_uniform",
                    name=f"enc_l{l}_h{k}"))
            layers.append(tf.keras.layers.Dense(
                self._per_l_K[l], activation=bn_act,
                kernel_initializer="glorot_uniform",
                name=f"enc_l{l}_out"))
            return tf.keras.Sequential(layers, name=f"enc_l{l}")

        def _dec_branch(l):
            layers: list[tf.keras.layers.Layer] = []
            for k, h in enumerate(reversed(self.l_block_hidden_dims)):
                layers.append(tf.keras.layers.Dense(
                    int(h), activation=act,
                    kernel_initializer="glorot_uniform",
                    name=f"dec_l{l}_h{k}"))
            layers.append(tf.keras.layers.Dense(
                self._per_l_Q[l], activation="linear",
                kernel_initializer="glorot_uniform",
                name=f"dec_l{l}_out"))
            return tf.keras.Sequential(layers, name=f"dec_l{l}")

        self.per_l_encoders = [_enc_branch(l) for l in range(self.L)]
        self.per_l_decoders = [_dec_branch(l) for l in range(self.L)]

        # Standardiser at the full Q level (same as the MLP path).
        self.standardizer = Standardizer(q_raw=self.q_raw)
        self.standardizer.build((None, self.q_raw))

    # ── Normalisation ──────────────────────────────────────────────────

    def fit_normalizer(self, soap_train: np.ndarray) -> None:
        # Centering is unconditional (see Willatt's fit_normalizer above
        # for the full reasoning). `normalize_inputs` controls only the
        # per-channel std rescaling. The per-l SVD inside
        # pca_initialize_weights re-centers each l-block anyway and
        # absorbs the residual mean into the Dense biases, so the
        # standardiser's mean and the per-l means compose harmlessly.
        mean = np.mean(soap_train, axis=0).astype(np.float32)
        if self.normalize_inputs and not self.center_only:
            std = np.std(soap_train, axis=0).astype(np.float32)
            std = np.maximum(std, 1e-6)
        else:
            std = np.ones_like(mean)
        self.standardizer.mean.assign(mean)
        self.standardizer.std.assign(std)

    @property
    def mean(self) -> tf.Variable:
        return self.standardizer.mean

    @property
    def std(self) -> tf.Variable:
        return self.standardizer.std

    def _standardize(self, x):
        return self.standardizer(x, inverse=False)

    def _unstandardize(self, x):
        return self.standardizer(x, inverse=True)

    # ── PCA initialisation ────────────────────────────────────────────

    def pca_initialize_weights(self, soap_train: np.ndarray) -> bool:
        """Initialise per-l encoder + decoder kernels from per-l SVD.

        Closed-form PCA: for each angular momentum l, take the per-l
        slice of the standardised training data `X_l[N, Q_l]`, centre
        it (the standardiser already subtracts the mean — but we do it
        again on the slice for numerical safety), compute the truncated
        SVD `X_l = U Σ Vᵀ`, and set:
            encoder_l.kernel  ← Vᵀ[:K_l].T          shape [Q_l, K_l]
            encoder_l.bias    ← 0
            decoder_l.kernel  ← Vᵀ[:K_l]            shape [K_l, Q_l]
            decoder_l.bias    ← 0

        With these weights and identity-activation, encoder maps a
        standardised SOAP slice to its top-K projection along the
        per-l principal axes, and the decoder reconstructs by the
        transpose — exactly classical PCA.

        Returns:
            True   when all branches are pure-linear (`l_block_hidden_dims
                   = ()`). The model is at the closed-form MSE optimum
                   and gradient descent will not improve it. Caller
                   should skip training in this case.
            False  when branches contain hidden layers. The first
                   encoder layer and last decoder layer are PCA-
                   initialised; intermediate layers stay Glorot.
                   Caller should continue with gradient descent —
                   this is a warm-start, not the final answer.
        """
        soap_train = np.asarray(soap_train, dtype=np.float32)
        # Apply the standardiser by hand so we don't run any TF graph
        # ops here. The standardiser's mean/std are already populated
        # by `fit_normalizer` (called before this method).
        mean = self.standardizer.mean.numpy()
        std = self.standardizer.std.numpy()
        X_std = (soap_train - mean) / std

        is_linear = len(self.l_block_hidden_dims) == 0
        for l in range(self.L):
            idx = self._per_l_gather[l].numpy()
            X_l = X_std[:, idx]                        # [N, Q_l]
            K_l = int(self._per_l_K[l])
            # Per-l mean of WHAT THE ENCODER SEES. When normalize_inputs
            # = True the Standardiser already subtracts the global mean
            # so this is ≈ 0; when False this is the raw per-l mean,
            # which must be absorbed into the biases for the model's
            # forward pass to compute the genuine PCA reconstruction
            # `(x − μ) V Vᵀ + μ`. Without these biases the linear AE
            # is off the optimum by exactly that translation offset —
            # which is exactly why a few gradient steps used to beat
            # the bias-free PCA init.
            X_l_mean = X_l.mean(axis=0).astype(np.float32)
            X_l_c = X_l - X_l_mean                     # centred for SVD
            # Economy SVD: V columns are right singular vectors.
            # numpy returns Vt = Vᵀ with shape [min(N,Q_l), Q_l].
            _, _, Vt = np.linalg.svd(X_l_c, full_matrices=False)
            top = Vt[:K_l].astype(np.float32)         # [K_l, Q_l]
            enc_branch = self.per_l_encoders[l]
            dec_branch = self.per_l_decoders[l]
            if is_linear:
                # Encoder: z = (X_l − μ_l) Vᵀ = X_l Vᵀ + b_enc
                #   ⇒ b_enc = −μ_l Vᵀ
                # Decoder: recon = z V + μ_l
                #   ⇒ W_dec = V (i.e. `top`),  b_dec = μ_l
                enc_branch.layers[0].kernel.assign(top.T)   # [Q_l, K_l]
                enc_branch.layers[0].bias.assign(
                    (-X_l_mean @ top.T).astype(np.float32))
                dec_branch.layers[0].kernel.assign(top)     # [K_l, Q_l]
                dec_branch.layers[0].bias.assign(X_l_mean)
            else:
                # Nonlinear warm-start: the per-l branch is
                #   Q_l → H_1 → … → K_l   (encoder)
                #   K_l → … → H_1 → Q_l   (decoder)
                # We can only place the first encoder layer and the
                # last decoder layer at the PCA solution; the hidden
                # middle layers stay at random Glorot init. The
                # placement is "first K_l units of H_1 do PCA, the
                # remaining H_1 − K_l units stay random features" —
                # the optimiser then learns how the random features
                # should be combined with the PCA features.
                #
                # In the LINEAR REGIME (small inputs / activations near
                # zero where silu ≈ x), the model approximately computes
                #   recon = (X_l − μ_l) V Vᵀ + μ_l
                # — i.e. the genuine PCA reconstruction. As activations
                # grow into silu's nonlinear region the bend gives
                # additional capacity that training exploits.
                #
                # Encoder bias on the PCA channels = −μ_l · top.T,
                # exactly mirroring the linear case so silu sees
                # mean-zero inputs from the start.
                enc_first = enc_branch.layers[0]
                first_out = int(enc_first.kernel.shape[1])
                K_emb = min(K_l, first_out)
                W_enc = enc_first.kernel.numpy()
                b_enc = enc_first.bias.numpy()
                W_enc[:, :K_emb] = top[:K_emb].T
                b_enc[:K_emb] = (-X_l_mean @ top[:K_emb].T).astype(np.float32)
                enc_first.kernel.assign(W_enc)
                enc_first.bias.assign(b_enc)
                # Decoder output kernel: first K_l ROWS = PCA top.
                # Output bias = per-l mean (the un-centring term).
                dec_last = dec_branch.layers[-1]
                last_in = int(dec_last.kernel.shape[0])
                K_emb2 = min(K_l, last_in)
                W_dec = dec_last.kernel.numpy()
                W_dec[:K_emb2, :] = top[:K_emb2]
                dec_last.kernel.assign(W_dec)
                dec_last.bias.assign(X_l_mean)
        return is_linear

    # ── Forward ────────────────────────────────────────────────────────

    def encode(self, x):
        """Per-l linear projection. Returns latent [B, Z=Σ K_l]."""
        x_std = self._standardize(x)
        per_l_z = []
        for l in range(self.L):
            x_l = tf.gather(x_std, self._per_l_gather[l], axis=1)
            per_l_z.append(self.per_l_encoders[l](x_l))
        return tf.concat(per_l_z, axis=1)

    def decode(self, z):
        """Per-l linear reconstruction + scatter back to flat Q."""
        per_l_x = []
        offset = 0
        for l in range(self.L):
            K_l = self._per_l_K[l]
            z_l = z[:, offset:offset + K_l]
            per_l_x.append(self.per_l_decoders[l](z_l))
            offset += K_l
        # Concatenate per-l reconstructions and scatter into a flat
        # [B, Q] tensor at the layout-walk positions.
        x_concat = tf.concat(per_l_x, axis=1)            # [B, Σ Q_l = Q]
        # tf.scatter_nd writes the SOURCE axes leading; transpose so Q
        # leads the batch, scatter, transpose back. Mirrors the Willatt
        # _to_dense_T pattern.
        B = tf.shape(x_concat)[0]
        flat_T = tf.transpose(x_concat)                  # [Q, B]
        x_std = tf.scatter_nd(
            self._flat_scatter_idx, flat_T,
            shape=[self.q_raw, B])                       # [Q, B]
        x_std = tf.transpose(x_std)                      # [B, Q]
        return self._unstandardize(x_std)

    def call(self, x, training=None):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ════════════════════════════════════════════════════════════════════════
# Composed Willatt + l_block_pca (alchemical + dimensional)
# ════════════════════════════════════════════════════════════════════════
#
# Two-stage hierarchical compression:
#   1. Willatt species projection T → K (preserves SOAP tensor structure)
#   2. Per-l SVD / MLP on the K-species SOAP (collapses each l-block to K_l)
#
# Order matters: Willatt is applied first because l_block_pca's per-l flat
# axis MIXES species and radial — it can't preserve the species-block
# structure that Willatt's alphabet relies on. By doing alchemical
# compression first, the per-l blocks then operate on a clean K-species
# SOAP whose channel layout is already alchemically reduced.
#
# When `pca_init = True` AND `l_block_hidden_dims = ()`, BOTH stages are
# solved analytically in one chained closed-form:
#   stage 1: HOSVD on species-mode covariance → u[T, K]
#   stage 2: project all training data via u, then per-l SVD on the
#            resulting K-species SOAP → per-l V_l matrices
# Training is then skipped — both u and per-l weights are at the MSE
# optimum. With nonlinear branches or pca_init=False, both stages are
# trained jointly via gradient descent.

class WillattLBlockAutoencoder(tf.keras.Model):
    """Composed alchemical + dimensional autoencoder.

    Encoder pipeline:
        SOAP_T  →  standardise  →  Willatt(u) → SOAP_K  →
        per-l projection (V_l)  →  latent

    Decoder pipeline (reverse):
        latent  →  per-l expansion → SOAP_K  →
        Willatt(v or uᵀ) → SOAP_T  →  un-standardise

    The `pca_init` flag toggles between:
      • One-solve analytic mode: HOSVD on raw species axis followed by
        per-l SVD on the K-species data. No gradient training.
      • Joint gradient mode: both u and the per-l weights are trained
        simultaneously via Adam on the reconstruction loss.
    """

    def __init__(self, q_raw_T: int, T: int, alpha_max: int, l_max: int,
                 cfg: AutoencoderConfig):
        super().__init__(name="willatt_l_block_autoencoder")
        # ── Species side (Willatt) ──────────────────────────────────────
        K_cfg = cfg.willatt_K if cfg.willatt_K is not None else max(1, T - 1)
        if not 1 <= int(K_cfg) <= T:
            raise ValueError(
                f"willatt_K={K_cfg} out of range [1, T={T}].")
        self.K_species = int(K_cfg)
        self.T = int(T)
        self.alpha_max = int(alpha_max)
        self.l_max = int(l_max)
        self.L = int(l_max) + 1
        self.q_raw_T = int(q_raw_T)
        self.q_raw = int(q_raw_T)     # alias used by save / encoder_io
        self.tied_weights = bool(cfg.tied_weights)
        self.normalize_inputs = bool(cfg.normalize_inputs)
        self.center_only = bool(cfg.center_only)
        self.compress_mode = str(getattr(cfg, "compress_mode", "trivial"))

        # Layouts for T-species (input) and K-species (intermediate).
        _, _, l_of_q_T_np, dq_T, q_T_check = _trivial_compress_indices(
            self.T, self.alpha_max, self.l_max, self.compress_mode)
        if q_T_check != self.q_raw_T:
            raise ValueError(
                f"Layout walk produced Q_T={q_T_check}, data has "
                f"Q_T={self.q_raw_T}.")
        _, _, l_of_q_K_np, dq_K, q_K_count = _trivial_compress_indices(
            self.K_species, self.alpha_max, self.l_max, self.compress_mode)
        self.q_raw_K = int(q_K_count)
        # Per-l gather on the T-species INPUT layout. Used by `evaluate`
        # to report input-side per-l R² — the bilinear projection
        # preserves the l axis, so the input-side l-blocks are
        # well-defined and directly comparable to l_block_pca's per-l
        # R² breakdown. (The K-species intermediate's per-l gather lives
        # on `_per_l_gather_K`, used by the model's own encode/decode.)
        per_l_indices_T = [
            np.where(l_of_q_T_np == l)[0].astype(np.int32)
            for l in range(self.L)]
        self._per_l_gather = [tf.constant(idx) for idx in per_l_indices_T]
        self._per_l_Q_T = [int(idx.size) for idx in per_l_indices_T]

        self._G_T = self.T * self.alpha_max
        self._G_K = self.K_species * self.alpha_max
        self._dense_T_flat = int(self._G_T * self._G_T * self.L)
        self._dense_K_flat = int(self._G_K * self._G_K * self.L)
        self._dense_q_T = tf.constant(dq_T[:, None], dtype=tf.int32)
        self._dense_q_K = tf.constant(dq_K[:, None], dtype=tf.int32)
        self._gather_q_T = tf.constant(dq_T, dtype=tf.int32)
        self._gather_q_K = tf.constant(dq_K, dtype=tf.int32)

        # Standardiser at the T-species level (input boundary).
        self.standardizer = Standardizer(q_raw=self.q_raw_T)
        self.standardizer.build((None, self.q_raw_T))

        # Willatt u[T, K] (and v[K, T] if untied).
        seed = int(getattr(cfg, "seed", 0))
        self._u_layer = _SpeciesProjector(
            shape=(self.T, self.K_species), seed=seed, name="encoder_u")
        self._u_layer.build((None,))
        if not self.tied_weights:
            self._v_layer = _SpeciesProjector(
                shape=(self.K_species, self.T), seed=seed ^ 0xC0DE,
                name="decoder_v")
            self._v_layer.build((None,))
        else:
            self._v_layer = None

        # ── Per-l side (l_block_pca on K-species SOAP) ──────────────────
        # Per-l index lists computed on the K-species layout. The l_aware
        # operations work on the K-species SOAP's flat channels, not T's.
        per_l_indices_K = [
            np.where(l_of_q_K_np == l)[0].astype(np.int32)
            for l in range(self.L)]
        self._per_l_Q_K = [int(idx.size) for idx in per_l_indices_K]
        self._per_l_gather_K = [tf.constant(idx) for idx in per_l_indices_K]
        flat_scatter_idx_K = np.concatenate(per_l_indices_K, axis=0)[:, None]
        self._flat_scatter_idx_K = tf.constant(
            flat_scatter_idx_K, dtype=tf.int32)

        # Per-l latent sizes (same logic as l_block_pca's: l_block_K
        # preferred, fallback to latent_dim split).
        l_block_K = getattr(cfg, "l_block_K", None)
        if l_block_K is not None:
            K_per_l = int(l_block_K)
            if K_per_l < 1:
                raise ValueError(
                    f"l_block_K={K_per_l} must be ≥ 1.")
            self._per_l_K = [
                min(K_per_l, Q_l) for Q_l in self._per_l_Q_K]
        else:
            total_K = int(cfg.latent_dim)
            if total_K < self.L:
                raise ValueError(
                    f"latent_dim={total_K} < L={self.L}.")
            base = total_K // self.L
            rem = total_K % self.L
            self._per_l_K = [
                base + (1 if l < rem else 0) for l in range(self.L)]
        self.latent_dim = int(sum(self._per_l_K))
        cfg.latent_dim = int(self.latent_dim)

        # Per-l hidden_dims (accept None / int / iterable like l_block_pca).
        _hd = getattr(cfg, "l_block_hidden_dims", ())
        if _hd is None:
            _hd = ()
        elif isinstance(_hd, (int, np.integer)):
            _hd = (int(_hd),)
        _hd = tuple(int(h) for h in _hd)
        if any(h < 1 for h in _hd):
            raise ValueError(
                f"l_block_hidden_dims={_hd} has non-positive entries.")
        self.l_block_hidden_dims = _hd

        act = tf.keras.activations.get(cfg.activation)
        bn_act = tf.keras.activations.get(cfg.bottleneck_activation)

        def _enc_branch(l):
            layers: list[tf.keras.layers.Layer] = []
            for k, h in enumerate(self.l_block_hidden_dims):
                layers.append(tf.keras.layers.Dense(
                    int(h), activation=act,
                    kernel_initializer="glorot_uniform",
                    name=f"enc_l{l}_h{k}"))
            layers.append(tf.keras.layers.Dense(
                self._per_l_K[l], activation=bn_act,
                kernel_initializer="glorot_uniform",
                name=f"enc_l{l}_out"))
            return tf.keras.Sequential(layers, name=f"enc_l{l}")

        def _dec_branch(l):
            layers: list[tf.keras.layers.Layer] = []
            for k, h in enumerate(reversed(self.l_block_hidden_dims)):
                layers.append(tf.keras.layers.Dense(
                    int(h), activation=act,
                    kernel_initializer="glorot_uniform",
                    name=f"dec_l{l}_h{k}"))
            layers.append(tf.keras.layers.Dense(
                self._per_l_Q_K[l], activation="linear",
                kernel_initializer="glorot_uniform",
                name=f"dec_l{l}_out"))
            return tf.keras.Sequential(layers, name=f"dec_l{l}")

        self.per_l_encoders = [_enc_branch(l) for l in range(self.L)]
        self.per_l_decoders = [_dec_branch(l) for l in range(self.L)]

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def u(self) -> tf.Variable:
        return self._u_layer.W

    @property
    def v(self) -> tf.Variable | None:
        return None if self._v_layer is None else self._v_layer.W

    @property
    def mean(self) -> tf.Variable:
        return self.standardizer.mean

    @property
    def std(self) -> tf.Variable:
        return self.standardizer.std

    def fit_normalizer(self, soap_train: np.ndarray) -> None:
        # Centering is unconditional (see Willatt's fit_normalizer for
        # the full reasoning). `normalize_inputs` controls only the
        # per-channel std rescaling. The downstream per-l SVD inside
        # the composed pca_initialize_weights also centers, but on the
        # K-species intermediate — different data, different mean.
        mean = np.mean(soap_train, axis=0).astype(np.float32)
        if self.normalize_inputs and not self.center_only:
            std = np.std(soap_train, axis=0).astype(np.float32)
            std = np.maximum(std, 1e-6)
        else:
            std = np.ones_like(mean)
        self.standardizer.mean.assign(mean)
        self.standardizer.std.assign(std)

    # ── Willatt species hop ────────────────────────────────────────────

    def _willatt_compress(self, x_std_T):
        """SOAP_T [B, Q_T] → SOAP_K [B, Q_K] via bilinear u contraction."""
        B = tf.shape(x_std_T)[0]
        flat = tf.transpose(x_std_T)
        dense_flat = tf.scatter_nd(
            self._dense_q_T, flat, shape=[self._dense_T_flat, B])
        dense_flat = tf.transpose(dense_flat)
        dense_T = tf.reshape(
            dense_flat, [B, self._G_T, self._G_T, self.L])
        d_T = tf.reshape(
            dense_T,
            [B, self.T, self.alpha_max, self.T, self.alpha_max, self.L])
        d_K = tf.einsum('bACBDl,AJ,BM->bJCMDl', d_T, self.u, self.u)
        dense_K_flat = tf.reshape(d_K, [B, self._dense_K_flat])
        return tf.gather(dense_K_flat, self._gather_q_K, axis=1)

    def _willatt_expand(self, x_K):
        """SOAP_K [B, Q_K] → SOAP_T [B, Q_T] via bilinear v contraction."""
        B = tf.shape(x_K)[0]
        flat = tf.transpose(x_K)
        dense_flat = tf.scatter_nd(
            self._dense_q_K, flat, shape=[self._dense_K_flat, B])
        dense_flat = tf.transpose(dense_flat)
        dense_K = tf.reshape(
            dense_flat, [B, self._G_K, self._G_K, self.L])
        d_K = tf.reshape(
            dense_K,
            [B, self.K_species, self.alpha_max,
             self.K_species, self.alpha_max, self.L])
        v = tf.transpose(self.u) if self.tied_weights else self.v
        d_T = tf.einsum('bJCMDl,JA,MB->bACBDl', d_K, v, v)
        dense_T_flat = tf.reshape(d_T, [B, self._dense_T_flat])
        return tf.gather(dense_T_flat, self._gather_q_T, axis=1)

    # ── Forward / inverse ──────────────────────────────────────────────

    def encode(self, x):
        x_std = self.standardizer(x, inverse=False)
        x_K = self._willatt_compress(x_std)
        per_l_z = []
        for l in range(self.L):
            x_l = tf.gather(x_K, self._per_l_gather_K[l], axis=1)
            per_l_z.append(self.per_l_encoders[l](x_l))
        return tf.concat(per_l_z, axis=1)

    def decode(self, z):
        per_l_x = []
        offset = 0
        for l in range(self.L):
            K_l = self._per_l_K[l]
            z_l = z[:, offset:offset + K_l]
            per_l_x.append(self.per_l_decoders[l](z_l))
            offset += K_l
        x_K_concat = tf.concat(per_l_x, axis=1)
        B = tf.shape(x_K_concat)[0]
        flat = tf.transpose(x_K_concat)
        x_K = tf.scatter_nd(
            self._flat_scatter_idx_K, flat, shape=[self.q_raw_K, B])
        x_K = tf.transpose(x_K)
        x_recon_std = self._willatt_expand(x_K)
        return self.standardizer(x_recon_std, inverse=True)

    def call(self, x, training=None):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    # ── Composed PCA init (HOSVD → per-l SVD on K-species data) ────────

    def pca_initialize_weights(self, soap_train: np.ndarray) -> bool:
        """Chained two-stage closed-form solve.

        Stage 1: HOSVD on the T-species mode-α covariance → u[T, K].
        Stage 2: forward-pass training data through Willatt to get the
                 K-species SOAP, then per-l SVD on that to seed V_l.

        Returns True iff both stages are linear (no hidden layers) — in
        which case the result is the closed-form MSE optimum and training
        can be skipped.
        """
        soap_train = np.asarray(soap_train, dtype=np.float32)
        mean = self.standardizer.mean.numpy()
        std = self.standardizer.std.numpy()
        X_std = (soap_train - mean) / std

        # ── Stage 1: HOSVD species init ────────────────────────────────
        gather_T = self._gather_q_T.numpy()
        C = np.zeros((self.T, self.T), dtype=np.float64)
        chunk = 512
        N = X_std.shape[0]
        for i in range(0, N, chunk):
            xs = X_std[i:i + chunk]
            B_ = xs.shape[0]
            dense = np.zeros((B_, self._dense_T_flat), dtype=np.float32)
            dense[:, gather_T] = xs
            d6 = dense.reshape(B_, self.T, self.alpha_max,
                                self.T, self.alpha_max, self.L)
            C += np.einsum('bACBDl,bECBDl->AE', d6, d6,
                           optimize=True).astype(np.float64)
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        u_init = eigvecs[:, -self.K_species:][:, ::-1].astype(np.float32)
        self._u_layer.W.assign(u_init)
        if not self.tied_weights:
            self._v_layer.W.assign(u_init.T)
        var_top = float(np.sum(eigvals[-self.K_species:]))
        var_total = float(np.sum(eigvals))
        if var_total > 0:
            print(f"[willatt_l_block] HOSVD: top-{self.K_species} species "
                  f"components capture {var_top / var_total:.4%} of total "
                  f"species-mode variance.")

        # ── Stage 2: project T → K, per-l SVD on K-species data ─────────
        # Forward training data through Willatt to get the K-species SOAP
        # that the per-l side actually sees. Chunked because
        # `_willatt_compress` materialises an O(N · G_T² · L) dense
        # tensor inside its scatter — the whole training set would OOM
        # a 10 GB GPU at typical N (e.g. 71k atoms × 6² × 10² × 8 floats
        # ≈ 11 GB in one batch).
        chunk_compress = 4096
        N_train = X_std.shape[0]
        x_K_parts: list[np.ndarray] = []
        for i in range(0, N_train, chunk_compress):
            xb = tf.constant(X_std[i:i + chunk_compress], dtype=tf.float32)
            x_K_parts.append(self._willatt_compress(xb).numpy())
        x_K_train = np.concatenate(x_K_parts, axis=0)
        del x_K_parts

        is_linear = len(self.l_block_hidden_dims) == 0
        for l in range(self.L):
            idx = self._per_l_gather_K[l].numpy()
            X_l = x_K_train[:, idx]
            X_l_mean = X_l.mean(axis=0).astype(np.float32)
            X_l_c = X_l - X_l_mean
            _, _, Vt = np.linalg.svd(X_l_c, full_matrices=False)
            K_l = int(self._per_l_K[l])
            top = Vt[:K_l].astype(np.float32)
            enc_branch = self.per_l_encoders[l]
            dec_branch = self.per_l_decoders[l]
            if is_linear:
                enc_branch.layers[0].kernel.assign(top.T)
                enc_branch.layers[0].bias.assign(
                    (-X_l_mean @ top.T).astype(np.float32))
                dec_branch.layers[0].kernel.assign(top)
                dec_branch.layers[0].bias.assign(X_l_mean)
            else:
                enc_first = enc_branch.layers[0]
                K_emb = min(K_l, int(enc_first.kernel.shape[1]))
                W_enc = enc_first.kernel.numpy()
                b_enc = enc_first.bias.numpy()
                W_enc[:, :K_emb] = top[:K_emb].T
                b_enc[:K_emb] = (-X_l_mean @ top[:K_emb].T).astype(np.float32)
                enc_first.kernel.assign(W_enc)
                enc_first.bias.assign(b_enc)
                dec_last = dec_branch.layers[-1]
                K_emb2 = min(K_l, int(dec_last.kernel.shape[0]))
                W_dec = dec_last.kernel.numpy()
                W_dec[:K_emb2, :] = top[:K_emb2]
                dec_last.kernel.assign(W_dec)
                dec_last.bias.assign(X_l_mean)
        return is_linear


# ════════════════════════════════════════════════════════════════════════
# Loss + training
# ════════════════════════════════════════════════════════════════════════

def reconstruction_loss(x_true: tf.Tensor, x_pred: tf.Tensor,
                        loss_type: str, huber_delta: float,
                        std: tf.Tensor | None) -> tf.Tensor:
    """Reconstruction loss on STANDARDISED inputs.

    Working in the standardised space (mean 0, std 1 per channel) makes
    the MSE invariant to the per-channel scale of the raw SOAP, so a fit
    isn't dominated by a handful of high-variance radial channels. Pass
    `std=None` to compute the loss in raw units.
    """
    if std is not None:
        x_true = x_true / std
        x_pred = x_pred / std
    diff = x_pred - x_true
    if loss_type == "mse":
        return tf.reduce_mean(tf.square(diff))
    if loss_type == "rmse":
        # Same optimum as MSE (sqrt is monotonic) but with a 1/(2·RMSE)
        # gradient scale — small early, amplified near convergence. Add
        # a tiny epsilon under the sqrt so the gradient stays finite if
        # the model achieves bit-exact reconstruction on a minibatch.
        # Reported in the same units as `val_rmse_raw`, so the tqdm bar
        # shows `train`, `val`, and `rmse` on the same numerical scale.
        return tf.sqrt(tf.reduce_mean(tf.square(diff)) + 1e-12)
    if loss_type == "mae":
        return tf.reduce_mean(tf.abs(diff))
    if loss_type == "huber":
        return tf.reduce_mean(
            tf.keras.losses.huber(x_true, x_pred, delta=huber_delta))
    raise ValueError(
        f"loss_type={loss_type!r} not in ('mse', 'rmse', 'mae', 'huber')")


def _lr_schedule(cfg: AutoencoderConfig, steps_per_epoch: int):
    """Return either a constant float or a Keras LR schedule."""
    if cfg.lr_schedule == "constant":
        return float(cfg.learning_rate)
    if cfg.lr_schedule == "cosine":
        total = steps_per_epoch * cfg.epochs
        warmup = int(total * cfg.warmup_frac)
        # Linear warmup → cosine decay to 1% of base LR.
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=max(1, total - warmup),
            alpha=0.01,
            warmup_target=cfg.learning_rate,
            warmup_steps=warmup)
    raise ValueError(
        f"lr_schedule={cfg.lr_schedule!r} not in ('constant', 'cosine')")


def train(model: SoapAutoencoder, cfg: AutoencoderConfig,
          soap_train: np.ndarray,
          soap_val: np.ndarray | None) -> dict:
    """Train the autoencoder. Returns the loss-history dict.

    Persistence is handled outside this function — the caller (typically
    the `__main__` block at the bottom of this module) invokes
    `encoder_io.save_*` on the trained model after `train()` returns.
    Keeping I/O outside the loop avoids tying the training code to a
    specific on-disk layout.
    """
    soap_train = np.asarray(soap_train, dtype=np.float32)
    if soap_val is not None:
        soap_val = np.asarray(soap_val, dtype=np.float32)
    N = soap_train.shape[0]
    # Round UP so the trailing partial batch is also seen each epoch.
    # The last batch may be smaller than cfg.batch_size; @tf.function's
    # `reduce_retracing=True` keeps the retrace cost bounded (one trace
    # at full size, one at the partial size).
    steps_per_epoch = max(1, (N + cfg.batch_size - 1) // cfg.batch_size)

    # Fit the normaliser BEFORE building the optimiser/schedule so the
    # model's mean/std variables are populated and the loss in the very
    # first epoch is already in standardised space.
    model.fit_normalizer(soap_train)

    lr_obj = _lr_schedule(cfg, steps_per_epoch)
    if cfg.weight_decay > 0.0:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_obj, weight_decay=cfg.weight_decay,
            clipnorm=cfg.grad_clip)
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_obj, clipnorm=cfg.grad_clip)

    rng = np.random.default_rng(cfg.seed)
    history: dict[str, list[float]] = {"epoch": [], "train_loss": [],
                                       "val_loss": [], "val_rmse_raw": []}
    # Last computed val numbers, carried forward on epochs where val is
    # skipped (val_every > 1). Prevents nan holes in history.csv and
    # keeps the tqdm bar's `val` / `rmse` fields populated.
    last_val_loss = float("nan")
    last_val_rmse_raw = float("nan")

    # Both step functions return three scalars: the optimiser `loss`
    # (used for backprop / display) + `mse_metric` (mean(diff²) in
    # standardised units, accumulated linearly for a true population
    # MSE) + `sq_raw_sum` (sum(diff²) in raw units for population RMSE).
    # The dual accumulation lets us report a TRUE epoch RMSE when
    # `loss_type='rmse'` instead of the Jensen-biased mean-of-batch-RMSE.
    # Denoising autoencoder regulariser (Vincent et al. 2008): add
    # Gaussian noise to the encoder input each minibatch, but compute
    # the loss against the CLEAN target. The noise scale is in
    # standardised units; multiply by per-channel std so noise is
    # comparable across radial-low and angular-high channels.
    denoise_std = float(getattr(cfg, "denoising_noise_stddev", 0.0) or 0.0)
    denoise_const = tf.constant(denoise_std, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def _train_step(x_batch):
        if denoise_std > 0.0:
            noise = tf.random.normal(tf.shape(x_batch),
                                      stddev=denoise_const)
            if model.normalize_inputs:
                # Noise specified per-channel-std-relative; rescale into
                # raw units before adding to the raw input.
                noise = noise * model.std
            x_in = x_batch + noise
        else:
            x_in = x_batch
        with tf.GradientTape() as tape:
            x_recon, _ = model(x_in, training=True)
            loss = reconstruction_loss(
                x_batch, x_recon,           # reconstruct CLEAN target
                cfg.loss_type, cfg.huber_delta,
                std=model.std if model.normalize_inputs else None)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        diff = x_recon - x_batch
        std_use = model.std if model.normalize_inputs else None
        if std_use is not None:
            mse_metric = tf.reduce_mean(tf.square(diff / std_use))
        else:
            mse_metric = tf.reduce_mean(tf.square(diff))
        return loss, mse_metric

    @tf.function(reduce_retracing=True)
    def _val_step(x_batch):
        x_recon, _ = model(x_batch, training=False)
        loss = reconstruction_loss(
            x_batch, x_recon,
            cfg.loss_type, cfg.huber_delta,
            std=model.std if model.normalize_inputs else None)
        diff = x_recon - x_batch
        std_use = model.std if model.normalize_inputs else None
        if std_use is not None:
            mse_metric = tf.reduce_mean(tf.square(diff / std_use))
        else:
            mse_metric = tf.reduce_mean(tf.square(diff))
        sq_raw_mean = tf.reduce_mean(tf.square(diff))
        return loss, mse_metric, sq_raw_mean

    # Single-line tqdm progress bar across all epochs. set_postfix keeps
    # the live train/val metrics next to the bar without scrolling the
    # terminal — useful for long runs (cfg.epochs = 2000+).
    bar = tqdm(range(1, cfg.epochs + 1), desc="train", unit="ep",
               dynamic_ncols=True)
    for epoch in bar:
        # Shuffle each epoch; sample without replacement. Step boundary
        # `min(end, N)` clamps the final partial batch so every atom is
        # covered exactly once per epoch (steps_per_epoch is ceil(N/bs)).
        # Loss accumulator weights by batch size so true epoch-mean is
        # reported rather than mean-of-batch-means.
        perm = rng.permutation(N)
        loss_acc = 0.0
        mse_acc = 0.0    # population MSE in standardised space
        for step in range(steps_per_epoch):
            start = step * cfg.batch_size
            end = min(start + cfg.batch_size, N)
            idx = perm[start:end]
            x_batch = tf.constant(soap_train[idx], dtype=tf.float32)
            loss, mse_m = _train_step(x_batch)
            bs = end - start
            loss_acc += float(loss.numpy()) * bs
            mse_acc += float(mse_m.numpy()) * bs
        # For RMSE we report sqrt(population MSE) — the linearly-
        # accumulated MSE then a single sqrt avoids the Jensen bias of
        # averaging per-batch RMSE values. For MSE/MAE/Huber the loss
        # itself is a per-sample mean so the weighted average is the
        # true population value.
        if cfg.loss_type == "rmse":
            epoch_loss = float(np.sqrt(max(0.0, mse_acc / float(N))))
        else:
            epoch_loss = loss_acc / float(N)

        # Default to the last computed values when val is skipped this
        # epoch — keeps history rows complete and the tqdm bar populated.
        val_loss = last_val_loss
        val_rmse_raw = last_val_rmse_raw
        if soap_val is not None and (epoch % cfg.val_every == 0
                                     or epoch == cfg.epochs):
            v_loss_acc = 0.0
            v_mse_acc = 0.0
            v_sq_raw_acc = 0.0
            n_val = int(soap_val.shape[0])
            for i in range(0, n_val, cfg.batch_size):
                x_v = tf.constant(
                    soap_val[i:i + cfg.batch_size], dtype=tf.float32)
                v_l, v_mse, v_sq = _val_step(x_v)
                bs = int(x_v.shape[0])
                v_loss_acc += float(v_l.numpy()) * bs
                v_mse_acc += float(v_mse.numpy()) * bs
                v_sq_raw_acc += float(v_sq.numpy()) * bs
            if cfg.loss_type == "rmse":
                val_loss = float(np.sqrt(max(0.0, v_mse_acc / max(1, n_val))))
            else:
                val_loss = v_loss_acc / max(1, n_val)
            val_rmse_raw = float(np.sqrt(v_sq_raw_acc / max(1, n_val)))
            last_val_loss = val_loss
            last_val_rmse_raw = val_rmse_raw

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            bar.set_postfix(
                train=f"{epoch_loss:.3e}",
                val=f"{val_loss:.3e}",
                rmse=f"{val_rmse_raw:.3e}",
            )
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse_raw"].append(val_rmse_raw)

    bar.close()
    return history


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════

def _resolve_types(cfg: AutoencoderConfig, dataset) -> list[int]:
    """Resolve the canonical species → type-index ordering.

    Mirrors TNEPconfig's `types` / `type_map` pattern: derived ONCE on
    the first SOAP build and cached on the cfg so every subsequent
    `.xyz` read (val / test / fine-tune dataset) lands at the same
    channel layout. Priority order:

      1. `cfg.type_map`       (dict)        — used as-is.
      2. `cfg.types`          (list)        — derives `type_map = {Z: i}`.
      3. `cfg.allowed_species` (list)        — Z-sorts the int entries.
      4. `dataset` species   (this file)    — Z-sorts the union of
                                              atoms in this dataset.

    For cases 3 and 4 the resolved (Z-sorted) types are written back
    onto `cfg` so the next call hits case 2 (a no-op).

    Any atom in `dataset` whose Z is not present in the resolved
    `type_map` raises with a list of the offending species. This
    catches the test-set-introduces-new-species failure mode.
    """
    # Priority 1: explicit type_map dict.
    if cfg.type_map is not None:
        # Coerce both keys and values to int — config.json round-trips
        # may have stringified keys (JSON convention).
        tm = {int(k): int(v) for k, v in cfg.type_map.items()}
        cfg.type_map = tm
        # Validate values form a contiguous permutation [0, N-1].
        # Without this, gaps (`{6: 0, 1: 2, 8: 5}`) or duplicates
        # silently produce wrong type indices downstream because
        # `assign_type_indices` uses positional `types.index(z)`.
        n = len(tm)
        if sorted(tm.values()) != list(range(n)):
            raise ValueError(
                f"cfg.type_map values {sorted(tm.values())} do not form "
                f"a contiguous permutation [0, {n - 1}]. Each species "
                "must map to a unique index in that range.")
        # Cross-check: if cfg.types is also set, both must agree.
        types = sorted(tm.keys(), key=lambda z: tm[z])
        if cfg.types is not None and list(cfg.types) != types:
            raise ValueError(
                f"cfg.types={list(cfg.types)} disagrees with the "
                f"ordering implied by cfg.type_map ({types}). Pick "
                "one source of truth, or set them consistently.")
        cfg.types = list(types)
    # Priority 2: explicit types list.
    elif cfg.types is not None:
        types = list(cfg.types)
        cfg.type_map = {int(z): i for i, z in enumerate(types)}
    # Priority 3: derive from allowed_species (int entries only).
    elif cfg.allowed_species is not None:
        types = sorted({int(s) for s in cfg.allowed_species
                        if isinstance(s, (int, np.integer))})
        if not types:
            raise ValueError(
                "cfg.allowed_species had no integer Z entries; can't "
                "resolve a canonical type ordering.")
        cfg.types = types
        cfg.type_map = {z: i for i, z in enumerate(types)}
    # Priority 4: derive from the dataset itself.
    else:
        types = sorted({int(z) for s in dataset for z in s.numbers})
        cfg.types = types
        cfg.type_map = {z: i for i, z in enumerate(types)}
        print(f"[data] resolved cfg.type_map = {cfg.type_map} (from "
              f"first .xyz read; will be reused for subsequent files)")

    # Validate that every species in this dataset is known to the map.
    file_species = {int(z) for s in dataset for z in s.numbers}
    unknown = sorted(file_species - set(cfg.type_map.keys()))
    if unknown:
        raise ValueError(
            f"File contains species {unknown} not present in the "
            f"cached cfg.type_map ({sorted(cfg.type_map.keys())}). "
            "Either set cfg.allowed_species to cover them, or pre-set "
            "cfg.types to include them before the first SOAP build.")
    return list(cfg.types)


def _soaps_from_xyz(path: str, cfg: AutoencoderConfig) -> np.ndarray:
    """Read a single .xyz file, apply the species filter, build SOAP.

    Returns a flat `[N_atoms, Q_raw]` array of per-atom SOAPs. Shared
    by the train+val loader and the held-out test loader so the
    descriptor build paths are byte-identical across splits.

    Skips `data.collect` — its unconditional target-key filter drops
    every structure missing the target, and an unsupervised AE has no
    target. Reuses `filter_by_species` / `assign_type_indices` so the
    species ordering matches a normal TNEP run.
    """
    try:
        from ase.io import read as ase_read
        from data import filter_by_species, assign_type_indices
        from DescriptorBuilder import make_descriptor_builder
        from TNEPconfig import TNEPconfig
        from DescriptorBuilderGPU import compute_dim_q
    except ImportError as e:
        raise RuntimeError(
            "Cannot import TNEP modules; supply a .npy cache to bypass "
            f"the on-the-fly SOAP build. Original error: {e}")

    print(f"[data] reading {path} …")
    dataset = ase_read(path, index=":")
    n_raw = len(dataset)
    file_species = sorted({int(z) for s in dataset for z in s.numbers})
    print(f"[data] raw structures: {n_raw}, raw species (Z-sorted): "
          f"{file_species}")

    if cfg.allowed_species is not None:
        # Need a provisional type-index assignment to drive
        # filter_by_species (it uses indices internally).
        prov_types = file_species
        prov_idx = assign_type_indices(dataset, prov_types)
        dataset, _ = filter_by_species(
            dataset, prov_idx,
            allowed_Z=cfg.allowed_species, mode=cfg.filter_mode)
        print(f"[data] after species filter ({cfg.filter_mode}): "
              f"{len(dataset)} structures")

    # Canonical species ordering. _resolve_types caches on cfg so the
    # FIRST file (typically train.xyz) sets the global layout and every
    # subsequent file (val/test) reuses it. Without this, train and
    # test can land on different SOAP channel layouts and the test R²
    # collapses silently — see the type_map cfg docstring.
    types = _resolve_types(cfg, dataset)
    assign_type_indices(dataset, types)

    tcfg = TNEPconfig()
    for key in ("alpha_max", "l_max", "rcut_hard", "rcut_soft", "nf",
                "skip_h_centers", "descriptor_mode", "compress_mode"):
        if hasattr(cfg, key):
            setattr(tcfg, key, getattr(cfg, key))
    tcfg.descriptor_mixing = False
    tcfg.descriptor_preprocess_contract = "off"
    tcfg.types = types
    tcfg.num_types = len(types)
    # The GPU TF backend hard-rejects anything other than "trivial".
    # quippy accepts "trivial" / "none" / "linear" / Darby modes —
    # raise a clear error before submitting an incompatible combo.
    cm = str(getattr(cfg, "compress_mode", "trivial"))
    if int(getattr(cfg, "descriptor_mode", 0)) == 1 and cm != "trivial":
        raise NotImplementedError(
            f"compress_mode={cm!r} requires descriptor_mode=0 (quippy); "
            "the GPU TF builder supports 'trivial' only.")
    # quippy backend (descriptor_mode=0) reads cfg.dim_q directly.
    # compute_dim_q handles 'trivial' and 'linear' in closed form; for
    # 'none' we derive Q from the full upper-triangle count locally.
    if cm == "none":
        Talpha = int(tcfg.num_types) * int(tcfg.alpha_max)
        tcfg.dim_q = int(Talpha * (Talpha + 1) // 2 * (int(tcfg.l_max) + 1))
    else:
        tcfg.dim_q = compute_dim_q(tcfg)

    print(f"[data] building SOAP for {len(dataset)} structures "
          f"(Q_raw={tcfg.dim_q}) …")
    builder = make_descriptor_builder(tcfg)
    # The AE is unsupervised — it only ever reads the per-atom SOAP, never
    # the ∂q/∂r Cartesian gradients. quippy's build_descriptors_flat takes
    # a `calc_gradients` kwarg whose False branch skips the dominant
    # cost; the GPU TF backend has no such switch. Use signature
    # introspection to pass the kwarg only when supported.
    import inspect
    sig = inspect.signature(builder.build_descriptors_flat)
    if "calc_gradients" in sig.parameters:
        frames = builder.build_descriptors_flat(dataset, calc_gradients=False)
    else:
        # GPU TF backend: always computes gradients; can't avoid it here.
        frames = builder.build_descriptors_flat(dataset)
    all_soaps = []
    for (soap, _grad, _pa, _pg), s in zip(frames, dataset):
        A_real = int(len(s))
        all_soaps.append(np.asarray(soap[:A_real], dtype=np.float32))
    return np.concatenate(all_soaps, axis=0)


def _build_soap_from_xyz(cfg: AutoencoderConfig
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Read `cfg.data_path` and random-split into (train, val).

    The val fraction is `cfg.test_ratio` of the loaded atoms. Returned
    arrays are float32 `[N_atoms, Q_raw]`. The held-out test set lives
    in `cfg.test_data_path` and is loaded separately by `_build_test_soap`.
    """
    soaps = _soaps_from_xyz(cfg.data_path, cfg)
    rng = np.random.default_rng(int(cfg.seed))
    perm = rng.permutation(soaps.shape[0])
    n_train = int(soaps.shape[0] * (1.0 - cfg.test_ratio))
    train = soaps[perm[:n_train]]
    val = soaps[perm[n_train:]]
    return train, val


def _build_test_soap(cfg: AutoencoderConfig) -> np.ndarray | None:
    """Return held-out test SOAPs, or None if neither cache nor path is set.

    Prefers `cfg.soap_test_cache` when available, else reads
    `cfg.test_data_path`. The test set is NEVER used during training —
    it's evaluated once at the end by `evaluate()` and the metrics are
    written to `test_metrics.json` in the run directory.
    """
    if cfg.soap_test_cache is not None:
        return np.load(cfg.soap_test_cache).astype(np.float32)
    if cfg.test_data_path is None:
        return None
    return _soaps_from_xyz(cfg.test_data_path, cfg)


def _load_or_build_soap(cfg: AutoencoderConfig
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, val) SOAP arrays, prefering .npy caches when set."""
    if cfg.soap_cache is not None:
        train = np.load(cfg.soap_cache).astype(np.float32)
        if cfg.soap_val_cache is not None:
            val = np.load(cfg.soap_val_cache).astype(np.float32)
        else:
            # Reserve a random test_ratio slice of the cached train as val.
            # Shuffling first matters: cached arrays are often produced
            # by walking train.xyz in order, so the tail is systematically
            # different chemistry (e.g. later trajectory frames). A
            # positional split would bias val toward those samples and
            # silently inflate val_loss.
            rng = np.random.default_rng(int(cfg.seed))
            perm = rng.permutation(train.shape[0])
            n_train = int(train.shape[0] * (1.0 - cfg.test_ratio))
            train_idx, val_idx = perm[:n_train], perm[n_train:]
            train, val = train[train_idx], train[val_idx]
        return train, val
    return _build_soap_from_xyz(cfg)


# ════════════════════════════════════════════════════════════════════════
# Evaluation on the held-out test set
# ════════════════════════════════════════════════════════════════════════

def evaluate(model: SoapAutoencoder, cfg: AutoencoderConfig,
             soap_test: np.ndarray) -> dict:
    """Run the trained AE on the held-out test set and return metrics.

    Computes four scalars characterising the reconstruction quality:
      - `loss` (standardised MSE/MAE/Huber, same as training) — directly
        comparable to the final `val_loss` recorded in history.
      - `rmse_raw` — RMSE in the SOAP's native units. Easy to read but
        dominated by the loudest channels.
      - `rmse_standardised` — per-feature-normalised RMSE; commensurate
        across channels and equals √loss when loss_type='mse'.
      - `r2` — fraction of total variance the AE captures across every
        (atom, channel) entry, computed against the standardiser's
        stored training mean as the zero-knowledge baseline.
        `r2 = 1` is perfect reconstruction; `r2 = 0` means the AE is
        no better than predicting the per-channel mean.

    Batched in chunks of `cfg.batch_size` so memory stays bounded for
    large test sets. Test atoms are never shuffled and never enter the
    training loop.
    """
    soap_test = np.asarray(soap_test, dtype=np.float32)
    N = int(soap_test.shape[0])
    if N == 0:
        return {"n_atoms": 0, "loss": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_standardised": float("nan"),
                "r2": float("nan")}

    recons = np.empty_like(soap_test)
    loss_acc = 0.0
    n_chunks = 0
    for i in range(0, N, cfg.batch_size):
        x = tf.constant(soap_test[i:i + cfg.batch_size], dtype=tf.float32)
        x_recon, _z = model(x, training=False)
        loss = reconstruction_loss(
            x, x_recon,
            cfg.loss_type, cfg.huber_delta,
            std=model.std if model.normalize_inputs else None)
        loss_acc += float(loss.numpy())
        n_chunks += 1
        recons[i:i + cfg.batch_size] = x_recon.numpy()

    diff = recons - soap_test
    rmse_raw = float(np.sqrt(np.mean(diff ** 2)))
    std_np = (model.std.numpy() if model.normalize_inputs
              else np.ones_like(diff[0]))
    rmse_std = float(np.sqrt(np.mean((diff / std_np) ** 2)))

    # R² against the training mean (the standardiser's stored mean) —
    # the honest zero-knowledge baseline a reconstructor must beat.
    mean_np = (model.mean.numpy() if model.normalize_inputs
               else soap_test.mean(axis=0))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((soap_test - mean_np) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    out = {
        "n_atoms": N,
        "loss": loss_acc / max(1, n_chunks),
        "rmse_raw": rmse_raw,
        "rmse_standardised": rmse_std,
        "r2": r2,
    }

    # Per-l R² — surfaces which angular-momentum block is the
    # reconstruction bottleneck. Channels at l contribute different
    # amounts to the global R² (large-variance radial channels dominate),
    # so the global number can mask poor fits on the small-variance
    # angular channels.
    #
    # Available for both `l_block_pca` (per-l SVD literally on input
    # l-blocks) and `willatt_l_block` (Willatt preserves the l axis, so
    # the input-side l-blocks are still well-defined even though the
    # intermediate per-l SVD operates on the K-species SOAP). Both
    # models expose `_per_l_gather` on the INPUT layout — distinct from
    # `_per_l_gather_K` (K-species intermediate) used internally by
    # willatt_l_block's encode/decode.
    per_l_gather = getattr(model, "_per_l_gather", None)
    if (getattr(model, "per_l_encoders", None) is not None
            and per_l_gather is not None):
        per_l_r2: list[float] = []
        per_l_rmse_raw: list[float] = []
        for l in range(int(model.L)):
            idx = per_l_gather[l].numpy()
            x_l = soap_test[:, idx]
            r_l = recons[:, idx]
            d_l = r_l - x_l
            ss_res_l = float(np.sum(d_l ** 2))
            ss_tot_l = float(np.sum((x_l - mean_np[idx]) ** 2))
            r2_l = (1.0 - ss_res_l / ss_tot_l) if ss_tot_l > 0 else float("nan")
            per_l_r2.append(r2_l)
            per_l_rmse_raw.append(float(np.sqrt(np.mean(d_l ** 2))))
        out["per_l_r2"] = per_l_r2
        out["per_l_rmse_raw"] = per_l_rmse_raw

    return out


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════
#
# Edit `AutoencoderConfig`'s class-level defaults above to control the
# run. The block below loads / builds SOAP, constructs the model, runs
# `train`, and hands every artifact off to `encoder_io` for persistence.

if __name__ == "__main__":
    import encoder_io

    cfg = AutoencoderConfig()
    # Keras's set_random_seed forwards to np.random.seed, which only
    # accepts uint32. cfg.seed may be a larger long for TNEP compatibility;
    # mask to 32 bits here without touching the cfg field.
    tf.keras.utils.set_random_seed(int(cfg.seed) & 0xFFFFFFFF)

    print("[ae] loading / building SOAP …")
    soap_train, soap_val = _load_or_build_soap(cfg)
    q_raw = int(soap_train.shape[1])
    print(f"[ae] train atoms = {soap_train.shape[0]}, "
          f"val atoms = {soap_val.shape[0]}, Q_raw = {q_raw}")

    out_dir = encoder_io.setup_encoder_run_directory(cfg)
    encoder_io.save_config(cfg, out_dir)

    architecture = str(getattr(cfg, "architecture", "mlp")).lower()
    # Whitelist check: silent fallthrough to MLP on a typo
    # (e.g. cfg.architecture = "willat") would train the wrong model
    # without any error, so reject any unknown string up front.
    _VALID_ARCHITECTURES = {"mlp", "willatt", "l_block_pca", "willatt_l_block"}
    if architecture not in _VALID_ARCHITECTURES:
        raise ValueError(
            f"cfg.architecture={architecture!r} not in "
            f"{sorted(_VALID_ARCHITECTURES)}. Check for typos.")

    def _resolve_T(label: str) -> int:
        """Resolve T from cfg or by inverse-Q lookup. Shared by both
        structured architectures (willatt / l_block_pca).

        Priority: cfg.types (the canonical resolved list, populated by
        `_resolve_types` during the SOAP build) → cfg.allowed_species →
        inverse-Q lookup. The cfg.types path catches the case where the
        user left allowed_species=None but the build resolved types
        from the data — we should use the resolved value, not re-derive."""
        if cfg.types is not None:
            return int(len(cfg.types))
        if cfg.allowed_species is not None:
            return int(len(cfg.allowed_species))
        for T_cand in range(1, 30):
            _, _, _, _, q_cand = _trivial_compress_indices(
                T_cand, int(cfg.alpha_max), int(cfg.l_max),
                str(cfg.compress_mode))
            if q_cand == q_raw:
                print(f"[ae] {label}: T={T_cand} derived from Q_raw "
                      "(cfg.allowed_species is None)")
                return int(T_cand)
        raise ValueError(
            f"Could not derive T from Q_raw={q_raw} under "
            f"alpha_max={cfg.alpha_max}, l_max={cfg.l_max}, "
            f"compress_mode={cfg.compress_mode!r}. Set "
            "cfg.allowed_species explicitly or check the SOAP "
            "knobs match how the cache was built.")

    if architecture == "willatt":
        T_real = _resolve_T("willatt")
        model = WillattAutoencoder(
            q_raw_T=q_raw, T=T_real,
            alpha_max=int(cfg.alpha_max), l_max=int(cfg.l_max),
            cfg=cfg)
        model(tf.zeros([1, q_raw], dtype=tf.float32))
        n_params = int(np.sum(
            [np.prod(v.shape) for v in model.trainable_variables]))
        print(f"[ae] willatt: T={model.T} → K={model.K} pseudo-species, "
              f"Q_T={q_raw} → Q_K={model.q_raw_K} (latent dim), "
              f"tied={model.tied_weights}")
        print(f"[ae] trainable params: {n_params:,d}")
    elif architecture == "l_block_pca":
        T_real = _resolve_T("l_block_pca")
        model = LBlockPCAAutoencoder(
            q_raw=q_raw, T=T_real,
            alpha_max=int(cfg.alpha_max), l_max=int(cfg.l_max),
            cfg=cfg)
        model(tf.zeros([1, q_raw], dtype=tf.float32))
        n_params = int(np.sum(
            [np.prod(v.shape) for v in model.trainable_variables]))
        per_l_Q = model._per_l_Q
        per_l_K = model._per_l_K
        hidden = tuple(model.l_block_hidden_dims)
        kind = "block-diag MLP" if hidden else "block-diag PCA (linear)"
        print(f"[ae] l_block_pca ({kind}): T={model.T}, L={model.L}, "
              f"per-l Q={per_l_Q}, per-l K={per_l_K}, "
              f"total Z={model.latent_dim}, "
              f"per-l hidden={hidden}")
        print(f"[ae] trainable params: {n_params:,d}")
    elif architecture == "willatt_l_block":
        T_real = _resolve_T("willatt_l_block")
        model = WillattLBlockAutoencoder(
            q_raw_T=q_raw, T=T_real,
            alpha_max=int(cfg.alpha_max), l_max=int(cfg.l_max),
            cfg=cfg)
        model(tf.zeros([1, q_raw], dtype=tf.float32))
        n_params = int(np.sum(
            [np.prod(v.shape) for v in model.trainable_variables]))
        per_l_Q = model._per_l_Q_K
        per_l_K = model._per_l_K
        hidden = tuple(model.l_block_hidden_dims)
        kind = "linear" if not hidden else "MLP"
        print(f"[ae] willatt_l_block ({kind}): T={model.T} → "
              f"K_species={model.K_species}, L={model.L}, "
              f"Q_T={q_raw} → Q_K={model.q_raw_K}, "
              f"per-l Q_K={per_l_Q}, per-l K={per_l_K}, "
              f"total Z={model.latent_dim}, "
              f"per-l hidden={hidden}, tied={model.tied_weights}")
        print(f"[ae] trainable params: {n_params:,d}")
    else:
        model = SoapAutoencoder(q_raw=q_raw, cfg=cfg)
        # Build sub-modules eagerly so trainable_variables and weight-saving
        # have concrete shapes; Sequential layers defer creation until first
        # call.
        model(tf.zeros([1, q_raw], dtype=tf.float32))
        n_params = int(np.sum(
            [np.prod(v.shape) for v in model.trainable_variables]))
        print(f"[ae] mlp: Z={cfg.latent_dim}, hidden={tuple(cfg.hidden_dims)},"
              f" tied={cfg.tied_weights}")
        print(f"[ae] trainable params: {n_params:,d}")

    # PCA-initialisation hook (only for l_block_pca). Saxe et al. 2014:
    # a linear AE's MSE optimum is the closed-form per-l SVD of the
    # training data. Pure-linear branches go straight to it — gradient
    # descent can't improve on a global optimum, so training is skipped
    # entirely. Nonlinear branches warm-start the first encoder layer
    # and last decoder layer from PCA components, then training
    # continues normally from a much-better-than-random starting point.
    pca_init_used = (architecture == "l_block_pca"
                     and bool(getattr(cfg, "pca_init", False)))
    willatt_pca_init_used = (architecture == "willatt"
                             and bool(getattr(cfg, "willatt_pca_init", False)))
    # Composed init flag: any of the two parent flags being on enables the
    # chained HOSVD + per-l SVD solve.
    willatt_l_block_pca_init_used = (
        architecture == "willatt_l_block"
        and (bool(getattr(cfg, "pca_init", False))
             or bool(getattr(cfg, "willatt_pca_init", False))))
    # User-controlled override: continue gradient training from the
    # analytic init instead of short-circuiting. Closes the gap between
    # the one-axis HOSVD optimum and the joint bilinear optimum, at the
    # cost of cfg.epochs of training.
    finetune_after_pca = bool(getattr(cfg, "pca_init_finetune", False))
    skip_training = False
    if pca_init_used:
        model.fit_normalizer(soap_train)
        is_linear = model.pca_initialize_weights(soap_train)
        if is_linear and not finetune_after_pca:
            print("[ae] pca_init + linear l_block_pca: training skipped "
                  "(analytic PCA solution IS the MSE optimum).")
            skip_training = True
        elif is_linear and finetune_after_pca:
            print(f"[ae] pca_init + linear l_block_pca + finetune: "
                  f"warm-started from analytic PCA; continuing gradient "
                  f"training for {cfg.epochs} epochs.")
        else:
            print(f"[ae] pca_init + nonlinear l_block_pca: warm-started "
                  f"first/last layers from per-l SVD; continuing training "
                  f"for {cfg.epochs} epochs.")
    elif willatt_pca_init_used:
        # Willatt's current backbone is purely linear bilinear (no
        # hidden layers between encoder and decoder bilinear hops), so
        # the HOSVD species-projection is the closed-form ONE-AXIS MSE
        # optimum. The JOINT bilinear (u uᵀ) ⊗ (u uᵀ) optimum couples
        # both species axes and is generally tighter; with
        # `pca_init_finetune=True`, Adam closes that gap.
        model.fit_normalizer(soap_train)
        model.pca_initialize_weights(soap_train)
        if finetune_after_pca:
            print(f"[ae] willatt_pca_init + finetune: HOSVD seeded u; "
                  f"continuing gradient training for {cfg.epochs} epochs "
                  f"to refine the joint bilinear projection.")
        else:
            print("[ae] willatt_pca_init: training skipped (HOSVD species "
                  "projection IS the one-axis MSE optimum; enable "
                  "`pca_init_finetune` to refine).")
            skip_training = True
    elif willatt_l_block_pca_init_used:
        # Chained two-stage analytic solve: HOSVD on species axis then
        # per-l SVD on the K-species SOAP. Closed-form MSE optimum iff
        # the per-l branches are linear (no hidden layers); otherwise
        # warm-start and continue gradient training.
        model.fit_normalizer(soap_train)
        is_linear = model.pca_initialize_weights(soap_train)
        if is_linear and not finetune_after_pca:
            print("[ae] pca_init + linear willatt_l_block: training "
                  "skipped (chained HOSVD + per-l SVD IS the MSE "
                  "optimum).")
            skip_training = True
        elif is_linear and finetune_after_pca:
            print(f"[ae] pca_init + linear willatt_l_block + finetune: "
                  f"chained HOSVD + per-l SVD seeded; continuing "
                  f"gradient training for {cfg.epochs} epochs to refine "
                  f"the joint bilinear / per-l coupling.")
        else:
            print(f"[ae] pca_init + nonlinear willatt_l_block: HOSVD "
                  f"seeded u, warm-started per-l first/last layers; "
                  f"continuing joint training for {cfg.epochs} epochs.")

    if skip_training:
        # Produce a one-row history with the analytic loss on train+val
        # so save_history / downstream consumers see a populated file.
        # Stream through the model in `cfg.batch_size` chunks — the
        # bilinear Willatt backbones materialise an O(N·G²·L) dense
        # tensor inside .call(), so a single full-tensor evaluation OOMs
        # the GPU on realistic training-set sizes.
        bs = int(cfg.batch_size)

        def _streamed_mse(arr, divide_by_std):
            std_use = model.std.numpy() if model.normalize_inputs else 1.0
            sq_sum = 0.0
            n_elem = 0
            for i in range(0, arr.shape[0], bs):
                xb = tf.constant(arr[i:i + bs], dtype=tf.float32)
                xrb, _ = model(xb)
                diff = (xrb - xb) / std_use if divide_by_std else (xrb - xb)
                sq_sum += float(tf.reduce_sum(tf.square(diff)))
                n_elem += int(tf.size(xb))
            return sq_sum / max(1, n_elem)

        history = {"epoch": [0], "train_loss": [], "val_loss": [],
                   "val_rmse_raw": []}
        for arr, key in ((soap_train, "train_loss"),
                         (soap_val, "val_loss")):
            if arr is None:
                history[key].append(float("nan"))
                continue
            mse = _streamed_mse(arr, divide_by_std=True)
            history[key].append(float(np.sqrt(mse))
                                if cfg.loss_type == "rmse" else mse)
        if soap_val is not None:
            mse_raw = _streamed_mse(soap_val, divide_by_std=False)
            history["val_rmse_raw"].append(float(np.sqrt(mse_raw)))
        else:
            history["val_rmse_raw"].append(float("nan"))
    else:
        history = train(model, cfg, soap_train, soap_val)

    encoder_io.save_encoder(model, out_dir)
    encoder_io.save_decoder(model, out_dir)
    encoder_io.save_standardizer(model, out_dir)
    encoder_io.save_history(history, out_dir)
    print(f"[ae] done. final train_loss={history['train_loss'][-1]:.5e},"
          f" val_loss={history['val_loss'][-1]:.5e},"
          f" val_rmse_raw={history['val_rmse_raw'][-1]:.5e}")

    # Held-out test evaluation: runs the trained model on the never-seen
    # test set and writes test_metrics.json. Skipped only when neither
    # cfg.test_data_path nor cfg.soap_test_cache is set.
    print("\n[ae] evaluating on held-out test set …")
    soap_test = _build_test_soap(cfg)
    if soap_test is None:
        print("[ae] no test set configured (test_data_path and "
              "soap_test_cache both None); skipping evaluation.")
    else:
        metrics = evaluate(model, cfg, soap_test)
        encoder_io.save_test_metrics(metrics, out_dir)
        print(f"[ae] test atoms     = {metrics['n_atoms']:,d}")
        print(f"[ae] test loss      = {metrics['loss']:.5e}")
        print(f"[ae] test rmse_raw  = {metrics['rmse_raw']:.5e}")
        print(f"[ae] test rmse_std  = {metrics['rmse_standardised']:.5e}")
        print(f"[ae] test R²        = {metrics['r2']:.4f}")
        if "per_l_r2" in metrics:
            r2s = metrics["per_l_r2"]
            rmses = metrics["per_l_rmse_raw"]
            print(f"[ae] per-l R²       = "
                  + "  ".join(f"l={l}:{r:.4f}" for l, r in enumerate(r2s)))
            print(f"[ae] per-l RMSE_raw = "
                  + "  ".join(f"l={l}:{r:.2e}" for l, r in enumerate(rmses)))
            worst_l = int(np.argmin(r2s))
            print(f"[ae] worst l = {worst_l} (R²={r2s[worst_l]:.4f})")
    print(f"[ae] artifacts written to {out_dir}")
