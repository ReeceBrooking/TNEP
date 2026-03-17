from __future__ import annotations

import os
import resource
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Callable
from TNEPconfig import TNEPconfig

if TYPE_CHECKING:
    from TNEP import TNEP

def _set_model_params(model: TNEP, *params: tf.Tensor) -> None:
    """Assign weight arrays directly into the TNEP model's tf.Variables.

    For modes 0/1: (W0, b0, W1, b1)
    For mode 2:    (W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol)
    """
    model.W0.assign(params[0])
    model.b0.assign(params[1])
    model.W1.assign(params[2])
    model.b1.assign(params[3])
    if len(params) == 8:
        model.W0_pol.assign(params[4])
        model.b0_pol.assign(params[5])
        model.W1_pol.assign(params[6])
        model.b1_pol.assign(params[7])

class SNES:
    """Separable Natural Evolution Strategy optimizer for TNEP.

    Maintains a diagonal Gaussian search distribution N(mu, diag(sigma^2)) over
    the flattened parameter vector of the TNEP model.  Each generation:
      1. Sample pop_size candidates: z_p = mu + sigma * s_p,  s_p ~ N(0,1)
      2. Evaluate fitness (RMSE) for each candidate on a random batch
      3. Rank candidates by fitness, pair with log-shaped utilities
      4. Update:  mu    <- mu + sigma * sum_p u_p * s_p
                  sigma <- sigma * exp(eta_sigma * sum_p u_p * (s_p^2 - 1))

    All sampling, ranking, and update operations use TensorFlow ops to
    stay on GPU.  Only scalar history values are transferred to CPU.

    Total parameter count = num_types * dim_q * num_neurons   (W0)
                          + num_types * num_neurons            (b0)
                          + num_types * num_neurons            (W1)
                          + 1                                  (b1)
    """

    def __init__(self, model: TNEP) -> None:
        self.model = model
        self.cfg = model.cfg
        self.dim_q = self.cfg.dim_q
        self.batch_size = self.cfg.batch_size

        # TF random generator for all stochastic ops in training loop
        if self.cfg.seed is not None:
            self.tf_rng = tf.random.Generator.from_seed(self.cfg.seed)
        else:
            self.tf_rng = tf.random.Generator.from_non_deterministic_state()

        # Total number of trainable parameters
        n_W0 = self.cfg.num_types * self.cfg.dim_q * self.cfg.num_neurons
        n_b0 = self.cfg.num_types * self.cfg.num_neurons
        n_W1 = self.cfg.num_types * self.cfg.num_neurons
        n_b1 = 1
        self.n_primary = n_W0 + n_b0 + n_W1 + n_b1
        # Mode 2 (polarizability) adds a second ANN with identical shape
        if self.cfg.target_mode == 2:
            self.dim = 2 * self.n_primary
        else:
            self.dim = self.n_primary

        # Search distribution parameters as tf.Variables (stay on GPU)
        # GPUMD initialises mu in [-1, 1] (see snes.cu line 6709)
        rng = np.random.default_rng(self.cfg.seed)
        mu_init = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        self.mu = tf.Variable(mu_init, trainable=False, name="snes_mu")
        self.sigma = tf.Variable(
            tf.fill([self.dim], self.cfg.init_sigma),
            trainable=False, name="snes_sigma")

        auto_pop = int(4 + (3 * np.log(self.dim)))
        self.pop_size = self.cfg.pop_size if self.cfg.pop_size is not None else auto_pop

        # Resolve auto-default regularization: sqrt(dim * 1e-6 / num_types)
        # GPUMD divides by num_types for type-specific ANN (version != 3)
        auto_lambda = np.sqrt(self.dim * 1e-6 / self.cfg.num_types)
        self.lambda_1 = self.cfg.lambda_1 if self.cfg.lambda_1 is not None else auto_lambda
        self.lambda_2 = self.cfg.lambda_2 if self.cfg.lambda_2 is not None else auto_lambda

        self.eta_sigma = self.compute_eta_sigma()
        self.utilities = tf.constant(self.compute_utilities(), dtype=tf.float32)

    def compute_regularization(self, param_vector: tf.Tensor | np.ndarray) -> tuple[float, float]:
        """Compute L1 and L2 regularization penalties (GPUMD formula).

        Works with both numpy arrays and TF tensors.

        L1 = lambda_1 * sum(|params|) / dim
        L2 = lambda_2 * sqrt(sum(params^2) / dim)

        Args:
            param_vector : [dim] tensor or ndarray — flat parameter vector

        Returns:
            l1 : float — L1 penalty
            l2 : float — L2 penalty
        """
        pv = tf.cast(param_vector, tf.float32)
        l1 = self.lambda_1 * tf.reduce_sum(tf.abs(pv)) / self.dim
        l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(pv)) / self.dim)
        return float(l1), float(l2)

    def compute_eta_sigma(self) -> float:
        """Compute the sigma learning rate from per-type parameter dimensionality.

        GPUMD (version != 3) divides by num_types for type-specific ANNs,
        giving a larger step size that accounts for per-type independence.

        Returns:
            eta_sigma : float — controls how fast sigma adapts.
                  eta_sigma = (3 + ln(num)) / (5 * sqrt(num)) / 2
                  where num = dim / num_types
        """
        num = float(self.dim) / self.cfg.num_types
        num = max(num, 1.0)
        eta_sigma = ((3.0 + np.log(num)) / (5.0 * np.sqrt(num))) / 2.0
        return float(eta_sigma)

    def compute_utilities(self) -> np.ndarray:
        """Precompute rank-based utility weights for the population.

        Utilities are log-shaped and zero-centred so that top-ranked
        individuals contribute positive gradient and bottom-ranked
        contribute negative.  Computed once at init, stored as tf.constant.

        Returns:
            utilities : ndarray [pop_size] — weights indexed by rank (0 = best).
        """

        lam = self.pop_size

        ranks = np.arange(lam) + 1

        raw = np.log((lam * 0.5) + 1.0) - np.log(ranks)
        raw = np.maximum(0.0, raw)

        # Normalise to sum=1, then shift to zero-mean
        total = np.sum(raw)
        if total > 0:
            raw /= total
        else:
            if self.cfg.debug:
                print("Utility calc failed due to negative total")
        utilities = raw - 1.0 / lam
        if self.cfg.debug:
            print("utilities = ", utilities)
        return utilities

    def ask(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample pop_size candidate parameter vectors from N(mu, diag(sigma^2)).

        All operations run on GPU via TensorFlow.

        Returns:
            samples : [pop_size, dim] float32 tensor — candidate parameter vectors
            s       : [pop_size, dim] float32 tensor — standard normal noise used
        """
        s = self.tf_rng.normal(shape=(self.pop_size, self.dim))
        samples = self.mu + s * self.sigma
        return samples, s

    def update(self, utilities: tf.Tensor, s: tf.Tensor) -> None:
        """Update mu and sigma using fitness-ranked noise vectors.

        All operations run on GPU via TensorFlow.

        Args:
            utilities : [pop_size] float32 tensor — rank-based weights (best first)
            s         : [pop_size, dim] float32 tensor — noise vectors sorted by fitness
                        (s[0] = noise of best individual, s[-1] = worst)

        Mutates self.mu and self.sigma tf.Variables in place.
        """
        grad_mu = tf.einsum('p,pd->d', utilities, s)
        grad_sigma = tf.einsum('p,pd->d', utilities, s ** 2 - 1.0)

        self.mu.assign_add(self.sigma * grad_mu)
        self.sigma.assign(self.sigma * tf.exp(self.eta_sigma * grad_sigma))

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor], plot_callback: Callable | None = None) -> dict:
        """Run the SNES training loop using GPU-batched population evaluation.

        All sampling, ranking, and update operations run on GPU.
        Only scalar metrics are transferred to CPU for history/reporting.

        Args:
            train_data    : dict with padded tensors from pad_and_stack()
            val_data      : same structure
            plot_callback : optional callable(history, gen) — called every
                            cfg.plot_interval generations for periodic plotting

        Returns:
            history : dict with training metrics per generation
        """
        print("Fitting model...")
        cfg = self.cfg
        S_train = train_data["descriptors"].shape[0]

        history = {
            "generation": [],
            "train_loss": [],
            "val_loss": [],
            "L1": [],
            "L2": [],
            "best_rmse": [],
            "worst_rmse": [],
            "sigma_min": [],
            "sigma_max": [],
            "sigma_mean": [],
            "sigma_median": [],
            "sigma_resets": [],
            "vram_mb": [],
            "ram_mb": [],
            "cpu_load": [],
            "timing": {
                "sample_batch": [],
                "evaluate": [],
                "rank_update": [],
                "validate": [],
                "overhead": [],
            },
        }

        best_val_loss = float('inf')
        best_mu = tf.identity(self.mu)
        best_sigma = tf.identity(self.sigma)
        gens_without_improvement = 0
        train_start = time.perf_counter()

        for gen in range(cfg.num_generations):
            t0 = time.perf_counter()

            samples, s = self.ask()

            # Select batch: None = full train set, int = random subset
            if cfg.batch_size is None:
                batch_data = train_data
            else:
                batch_idx_tf = tf.argsort(
                    self.tf_rng.uniform(shape=[S_train]))[:cfg.batch_size]
                batch_data = {
                    key: tf.gather(train_data[key], batch_idx_tf)
                    for key in ["descriptors", "gradients", "grad_index",
                                "positions", "Z_int", "boxes", "targets",
                                "atom_mask", "neighbor_mask"]
                }

            t1 = time.perf_counter()

            # Evaluate entire population on GPU
            fitness = self.evaluate_population(samples, batch_data)

            # GPU sync: extract scalar metrics
            avg_fitness = float(tf.reduce_mean(fitness))
            best_rmse = float(tf.reduce_min(fitness))
            worst_rmse = float(tf.reduce_max(fitness))

            # VRAM snapshot (peak after evaluate — the heaviest phase)
            if tf.config.list_physical_devices('GPU'):
                vram_info = tf.config.experimental.get_memory_info('GPU:0')
                history["vram_mb"].append(vram_info["peak"] / 1024 / 1024)
            else:
                history["vram_mb"].append(0.0)

            # RAM and CPU usage
            ram_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            history["ram_mb"].append(ram_peak_kb / 1024)
            history["cpu_load"].append(os.getloadavg()[0])

            t2 = time.perf_counter()

            # Regularization at current mean for reporting
            if cfg.toggle_regularization:
                gen_l1, gen_l2 = self.compute_regularization(self.mu)
            else:
                gen_l1, gen_l2 = 0, 0

            # Rank and update (GPU)
            ranks = tf.argsort(fitness)
            s_sorted = tf.gather(s, ranks)
            self.update(self.utilities, s_sorted)
            # GPU sync: read mu to ensure update completed before timing validate
            self.mu.numpy()

            t3 = time.perf_counter()

            # Validate with updated mean (GPU — mu is tf.Variable)
            val_fitness = self.validate(val_data, self.mu)

            t4 = time.perf_counter()

            history["generation"].append(gen)
            history["train_loss"].append(avg_fitness)
            history["val_loss"].append(float(val_fitness))
            history["L1"].append(gen_l1)
            history["L2"].append(gen_l2)
            history["best_rmse"].append(best_rmse)
            history["worst_rmse"].append(worst_rmse)
            history["sigma_min"].append(float(tf.reduce_min(self.sigma)))
            history["sigma_max"].append(float(tf.reduce_max(self.sigma)))
            history["sigma_mean"].append(float(tf.reduce_mean(self.sigma)))
            history["sigma_median"].append(float(np.median(self.sigma.numpy())))

            # Progress bar
            frac = (gen + 1) / cfg.num_generations
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.perf_counter() - train_start
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            line = (f"\r{bar} {gen + 1}/{cfg.num_generations} "
                    f"train RMSE: {avg_fitness:.4f}  "
                    f"val RMSE: {float(val_fitness):.4f}  "
                    f"best val RMSE: {best_val_loss:.4f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping
            val_loss_scalar = float(val_fitness)
            if val_loss_scalar < best_val_loss:
                best_val_loss = val_loss_scalar
                best_mu = tf.identity(self.mu)
                best_sigma = tf.identity(self.sigma)
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"\nEarly stopping at generation {gen + 1} "
                      f"(no improvement for {cfg.patience} generations)")
                self.mu.assign(best_mu)
                self.sigma.assign(best_sigma)
                break

            # Sigma reset on stagnation
            if cfg.sigma_reset_patience is not None:
                mean_sigma = float(tf.reduce_mean(self.sigma))
                sigma_collapsed = mean_sigma < (cfg.sigma_reset_threshold * cfg.init_sigma)
                if (gens_without_improvement >= cfg.sigma_reset_patience
                        and sigma_collapsed):
                    print(f"\nSigma reset at generation {gen + 1}: "
                          f"mean sigma {mean_sigma:.6f} < threshold, "
                          f"stagnant for {gens_without_improvement} gens")
                    self.sigma.assign(tf.fill([self.dim], cfg.init_sigma))
                    gens_without_improvement = 0
                    history["sigma_resets"].append(gen)

            t5 = time.perf_counter()

            history["timing"]["sample_batch"].append(t1 - t0)
            history["timing"]["evaluate"].append(t2 - t1)
            history["timing"]["rank_update"].append(t3 - t2)
            history["timing"]["validate"].append(t4 - t3)
            history["timing"]["overhead"].append(t5 - t4)

            # Periodic plotting callback
            if gen + 1 < cfg.num_generations:
                if (plot_callback is not None
                        and cfg.plot_interval is not None
                        and (gen + 1) % cfg.plot_interval == 0):
                    # Temporarily restore best params for score()
                    params = self.reconstruct_params_tf(best_mu)
                    _set_model_params(self.model, *params)
                    plot_callback(history, gen + 1)
                    # Restore current mu back into model (training continues)
                    params = self.reconstruct_params_tf(self.mu)
                    _set_model_params(self.model, *params)

        print()  # newline after progress bar
        # Always restore best parameters into model for downstream score() calls
        self.mu.assign(best_mu)
        self.sigma.assign(best_sigma)
        params = self.reconstruct_params_tf(self.mu)
        _set_model_params(self.model, *params)

        return history

    def validate(self, val_data: dict[str, tf.Tensor], mu_tf: tf.Tensor | None = None) -> float:
        """Compute mean RMSE on a subset of validation structures using batched predict.

        Args:
            val_data : dict with padded tensors from pad_and_stack()
            mu_tf    : optional [dim] float32 tensor/Variable — parameter vector.
                       If provided, weights are reconstructed on GPU.
                       If None, uses current model weights.

        Returns:
            fitness : float — mean RMSE
        """
        if self.cfg.val_size is None:
            batch_data = val_data
        else:
            S_val = val_data["descriptors"].shape[0]
            val_idx_tf = tf.argsort(
                self.tf_rng.uniform(shape=[S_val]))[:self.cfg.val_size]

            batch_data = {
                key: tf.gather(val_data[key], val_idx_tf)
                for key in ["descriptors", "gradients", "grad_index",
                            "positions", "Z_int", "boxes", "targets",
                            "atom_mask", "neighbor_mask"]
            }

        if mu_tf is not None:
            params = self.reconstruct_params_tf(mu_tf)
            if self.cfg.target_mode == 2:
                W0, b0, W1, b1, W0p, b0p, W1p, b1p = params
            else:
                W0, b0, W1, b1 = params
                W0p = b0p = W1p = b1p = None
        else:
            W0, b0, W1, b1 = self.model.W0, self.model.b0, self.model.W1, self.model.b1
            W0p = getattr(self.model, 'W0_pol', None)
            b0p = getattr(self.model, 'b0_pol', None)
            W1p = getattr(self.model, 'W1_pol', None)
            b1p = getattr(self.model, 'b1_pol', None)

        preds = self.model.predict_batch(
            batch_data["descriptors"], batch_data["gradients"],
            batch_data["grad_index"], batch_data["positions"],
            batch_data["Z_int"], batch_data["boxes"],
            batch_data["atom_mask"], batch_data["neighbor_mask"],
            W0, b0, W1, b1, W0p, b0p, W1p, b1p,
        )
        diff = preds - batch_data["targets"]
        rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))
        return float(rmse)

    def reconstruct_params(self, param_vector: np.ndarray) -> tuple:
        """Reconstruct TNEP parameters from a flat numpy vector.

        For modes 0/1: returns (W0, b0, W1, b1)
        For mode 2:    returns (W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol)
        """
        pv = np.asarray(param_vector, dtype=float)
        assert pv.shape[0] == self.dim, (
            f"param_vector has length {pv.shape[0]}, expected {self.dim}"
        )

        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        n_W0 = T * Q * H
        n_b0 = T * H
        n_W1 = T * H
        n_b1 = 1

        def _extract_ann(pv, offset):
            W0 = pv[offset: offset + n_W0].reshape((T, Q, H))
            offset += n_W0
            b0 = pv[offset: offset + n_b0].reshape((T, H))
            offset += n_b0
            W1 = pv[offset: offset + n_W1].reshape((T, H))
            offset += n_W1
            b1 = float(pv[offset])
            offset += n_b1
            return W0, b0, W1, b1, offset

        W0, b0, W1, b1, offset = _extract_ann(pv, 0)

        if self.cfg.target_mode == 2:
            W0_pol, b0_pol, W1_pol, b1_pol, _ = _extract_ann(pv, offset)
            return W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol

        return W0, b0, W1, b1

    def reconstruct_params_tf(self, param_vectors: tf.Tensor) -> tuple:
        """Reconstruct TNEP weight tensors from flat vectors using TF ops.

        Works inside @tf.function. Handles single [dim] or batched [P, dim] vectors.

        Args:
            param_vectors : [P, dim] or [dim] float32 tensor

        Returns:
            tuple of (W0, b0, W1, b1) and optionally (W0_pol, b0_pol, W1_pol, b1_pol)
            Each has shape [P, ...] for batched input or [...] for single.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        n_W0 = T * Q * H
        n_b0 = T * H
        n_W1 = T * H
        n_b1 = 1

        is_batched = len(param_vectors.shape) == 2

        def _extract(pv, offset):
            W0 = tf.reshape(pv[..., offset:offset + n_W0],
                            [-1, T, Q, H] if is_batched else [T, Q, H])
            offset += n_W0
            b0 = tf.reshape(pv[..., offset:offset + n_b0],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_b0
            W1 = tf.reshape(pv[..., offset:offset + n_W1],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_W1
            b1 = pv[..., offset]  # [P] or scalar
            offset += n_b1
            return W0, b0, W1, b1, offset

        W0, b0, W1, b1, offset = _extract(param_vectors, 0)

        if self.cfg.target_mode == 2:
            W0p, b0p, W1p, b1p, _ = _extract(param_vectors, offset)
            return W0, b0, W1, b1, W0p, b0p, W1p, b1p

        return W0, b0, W1, b1

    def compute_regularization_tf(self, param_vectors: tf.Tensor) -> tf.Tensor:
        """Compute L1+L2 regularization for batched parameter vectors.

        Args:
            param_vectors : [P, dim] float32

        Returns:
            reg : [P] float32 — L1 + L2 penalty per candidate
        """
        l1 = self.lambda_1 * tf.reduce_sum(tf.abs(param_vectors), axis=1) / self.dim
        l2 = self.lambda_2 * tf.sqrt(
            tf.reduce_sum(tf.square(param_vectors), axis=1) / self.dim)
        return l1 + l2

    def evaluate_population(self, samples_tf: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate all SNES candidates on a batch of structures on GPU.

        Chunks along both population (population_chunk_size) and structure
        (batch_chunk_size) dimensions to limit VRAM usage. Accumulates sum
        of squared errors across structure chunks for correct RMSE.

        Args:
            samples_tf : [P, dim] float32 — all candidate parameter vectors
            batch_data : dict with padded batch tensors:
                descriptors   : [B, A, Q]
                gradients     : [B, A, M, 3, Q]
                grad_index    : [B, A, M]
                positions     : [B, A, 3]
                Z_int         : [B, A]
                boxes         : [B, 3, 3]
                targets       : [B, T_dim]
                atom_mask     : [B, A]
                neighbor_mask : [B, A, M]

        Returns:
            fitness : [P] float32 — RMSE (+ regularization) per candidate
        """
        P = self.pop_size
        B = batch_data["descriptors"].shape[0]
        T_dim = batch_data["targets"].shape[1]
        pop_chunk = self.cfg.population_chunk_size if self.cfg.population_chunk_size is not None else P
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else B

        batch_keys = ["descriptors", "gradients", "grad_index",
                      "positions", "Z_int", "boxes", "targets",
                      "atom_mask", "neighbor_mask"]

        all_fitness = []

        for p_start in range(0, P, pop_chunk):
            p_end = min(p_start + pop_chunk, P)
            candidates = samples_tf[p_start:p_end]  # [C, dim]
            C = p_end - p_start

            # Accumulate SSE across structure chunks
            total_sse = tf.zeros([C])
            total_elements = 0

            for s_start in range(0, B, struct_chunk):
                s_end = min(s_start + struct_chunk, B)
                struct_batch = {key: batch_data[key][s_start:s_end] for key in batch_keys}
                chunk_sse = self._evaluate_chunk(candidates, struct_batch)  # [C]
                total_sse = total_sse + chunk_sse
                total_elements += (s_end - s_start) * T_dim

            chunk_fitness = tf.sqrt(total_sse / tf.cast(total_elements, tf.float32))
            all_fitness.append(chunk_fitness)

        fitness = tf.concat(all_fitness, axis=0)  # [P]

        if self.cfg.toggle_regularization:
            reg = self.compute_regularization_tf(samples_tf)
            fitness = fitness + reg

        return fitness

    @tf.function
    def _evaluate_chunk(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate a chunk of C candidates on B structures.

        Returns sum of squared errors (not RMSE) so that structure chunks
        can be aggregated correctly in evaluate_population.

        Args:
            chunk_samples : [C, dim]
            batch_data    : dict of [B, ...] tensors

        Returns:
            sse : [C] — sum of squared errors per candidate
        """
        desc = batch_data["descriptors"]       # [B, A, Q]
        grads = batch_data["gradients"]        # [B, A, M, 3, Q]
        gidx = batch_data["grad_index"]        # [B, A, M]
        pos = batch_data["positions"]          # [B, A, 3]
        Z = batch_data["Z_int"]               # [B, A]
        boxes = batch_data["boxes"]            # [B, 3, 3]
        targets = batch_data["targets"]        # [B, T_dim]
        amask = batch_data["atom_mask"]        # [B, A]
        nmask = batch_data["neighbor_mask"]    # [B, A, M]

        # Reconstruct weights for all candidates in chunk
        params = self.reconstruct_params_tf(chunk_samples)

        if self.cfg.target_mode == 2:
            W0, b0, W1, b1, W0p, b0p, W1p, b1p = params

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p = args
                preds = self.model.predict_batch(
                    desc, grads, gidx, pos, Z, boxes, amask, nmask,
                    w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p,
                )
                diff = preds - targets
                return tf.reduce_sum(tf.square(diff))

            stacked = (W0, b0, W1, b1, W0p, b0p, W1p, b1p)
        else:
            W0, b0, W1, b1 = params

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1 = args
                preds = self.model.predict_batch(
                    desc, grads, gidx, pos, Z, boxes, amask, nmask,
                    w0, bb0, w1, bb1, None, None, None, None,
                )
                diff = preds - targets
                return tf.reduce_sum(tf.square(diff))

            stacked = (W0, b0, W1, b1)

        sse = tf.vectorized_map(_forward_one_candidate, stacked)  # [C]
        return sse
