from __future__ import annotations

import os
import resource
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Callable

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
        cfg = model.cfg
        self.cfg = cfg
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
        self.n_typed = n_W0 + n_b0 + n_W1  # per-type params (excludes b1)
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

        if self.cfg.stagnation_response is not None:
            assert self.cfg.stagnation_response in ('interpolate', 'noise'), (
                f"stagnation_response must be 'interpolate', 'noise', or None, "
                f"got {self.cfg.stagnation_response!r}")

        # Loss function flags (Python bools so tf.function traces correct branch)
        self._use_mae = (cfg.loss_type == "mae")
        self._use_inv_weight = (cfg.inverse_weight_eps is not None)
        self._inv_weight_eps = cfg.inverse_weight_eps if self._use_inv_weight else 1.0

        # Polarizability shear weight: scale off-diagonal components [xy, yz, zx]
        # Targets are [xx, yy, zz, xy, yz, zx] — indices 3,4,5 are off-diagonal
        if cfg.target_mode == 2:
            shear_sq = cfg.lambda_shear ** 2
            self._pol_weights = tf.constant(
                [1.0, 1.0, 1.0, shear_sq, shear_sq, shear_sq], dtype=tf.float32)
        else:
            self._pol_weights = None

        # Per-type ranking: build type_of_variable map [dim] -> type index
        # type 0..T-1 for typed params (W0, b0, W1), type T for b1 (global)
        self._per_type = (cfg.per_type_regularization
                          and cfg.toggle_regularization
                          and cfg.num_types > 1)
        if self._per_type:
            self._type_of_variable = self._build_type_of_variable()
            self._type_of_variable_tf = tf.constant(self._type_of_variable, dtype=tf.int32)

        self.eta_sigma = self.cfg.eta_sigma if self.cfg.eta_sigma is not None else self.compute_eta_sigma()
        self.utilities = tf.constant(self.compute_utilities(), dtype=tf.float32)

    def compute_regularization(self, param_vector: tf.Tensor | np.ndarray) -> tuple[float, float]:
        """Compute L1 and L2 regularization penalties (GPUMD NEP4 formula).

        For multi-element systems, computes per-type regularization: each
        atom type's parameters are penalized separately using
        num_vars/num_types as the denominator, then averaged across types
        and added to a global regularization term over all parameters.

        Works with both numpy arrays and TF tensors.

        Args:
            param_vector : [dim] tensor or ndarray — flat parameter vector

        Returns:
            l1 : float — L1 penalty
            l2 : float — L2 penalty
        """
        pv = tf.cast(param_vector, tf.float32)
        T = self.cfg.num_types
        n_per_type = self.n_primary // T  # params per type (W0_t + b0_t + W1_t)

        if T > 1:
            # Per-type regularization: average L1/L2 across types + global term
            total_l1 = tf.constant(0.0)
            total_l2 = tf.constant(0.0)

            for t in range(T):
                type_params = self._extract_type_params(pv, t)
                total_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(type_params)) / n_per_type
                total_l2 += self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(type_params)) / n_per_type)

            # Average per-type + global term over typed params only (excludes b1)
            typed = tf.concat([pv[:self.n_typed]], axis=0)
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                # Second ANN's typed params (skip b1 of primary ANN)
                typed = tf.concat([typed, pv[self.n_primary:self.n_primary + self.n_typed]], axis=0)
                n_typed_total = 2 * self.n_typed
            l1 = total_l1 / T + self.lambda_1 * tf.reduce_sum(tf.abs(typed)) / n_typed_total
            l2 = total_l2 / T + self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(typed)) / n_typed_total)
        else:
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(pv)) / self.dim
            l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(pv)) / self.dim)

        return float(l1), float(l2)

    def _extract_type_params(self, pv: tf.Tensor, t: int) -> tf.Tensor:
        """Extract parameters belonging to atom type t from flat vector.

        Parameter layout: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)]
        Per-type slice: W0[t,:,:] + b0[t,:] + W1[t,:]

        Args:
            pv : [dim] flat parameter vector
            t  : type index (0 to num_types-1)

        Returns:
            [Q*H + H + H] tensor — concatenated type-t parameters
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        # W0 block: [T, Q, H] flattened → stride T*Q*H, type t starts at t*Q*H
        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        # b0 block: after all W0, [T, H] → type t at offset T*Q*H + t*H
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        # W1 block: after b0, [T, H] → type t at offset T*Q*H + T*H + t*H
        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H

        return tf.concat([pv[w0_start:w0_end], pv[b0_start:b0_end], pv[w1_start:w1_end]], axis=0)

    def _build_type_of_variable(self) -> np.ndarray:
        """Build array mapping each parameter index to its atom type.

        Layout per ANN: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)]
        W0/b0/W1 params for type t get label t. b1 gets label T (global).

        Returns:
            [dim] int array — type label per variable
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        def _ann_types() -> np.ndarray:
            tov = np.empty(self.n_primary, dtype=np.int32)
            offset = 0
            # W0: [T, Q, H] — type t owns contiguous block of Q*H
            for t in range(T):
                tov[offset:offset + Q * H] = t
                offset += Q * H
            # b0: [T, H]
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            # W1: [T, H]
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            # b1: global (label = T)
            tov[offset] = T
            return tov

        if self.cfg.target_mode == 2:
            return np.concatenate([_ann_types(), _ann_types()])
        return _ann_types()

    def _build_per_type_gradients(
        self,
        s: tf.Tensor,
        fitness_per_type_rmse: tf.Tensor,
        samples: tf.Tensor,
    ) -> tf.Tensor:
        """Build composite noise matrix with per-type rankings.

        For each variable v, permutes the P noise vectors according to the
        ranking of type_of_variable[v]'s fitness. The result can be passed
        directly to update() with the standard utilities.

        Args:
            s                    : [P, dim] noise vectors from ask()
            fitness_per_type_rmse: [P, T+1] per-type RMSE (type 0..T-1 from
                                   structures containing that type, type T = global)
            samples              : [P, dim] candidate parameter vectors

        Returns:
            s_sorted : [P, dim] composite noise — each column permuted by
                       its type's ranking
        """
        T = self.cfg.num_types
        P = self.pop_size
        Q = self.dim_q
        H = self.cfg.num_neurons
        n_per_type = Q * H + H + H

        # Add per-type regularization to per-type RMSE → [T+1] fitness values
        fitness_per_type = []
        for t in range(T):
            type_params = self._extract_type_params_batched(samples, t)  # [P, n_per_type]
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)
            fitness_per_type.append(fitness_per_type_rmse[:, t] + l1 + l2)

        # Global ranking (type T): regularize over all typed params, excluding b1
        typed = samples[:, :self.n_typed]
        n_typed_total = self.n_typed
        if self.cfg.target_mode == 2:
            typed = tf.concat([typed, samples[:, self.n_primary:self.n_primary + self.n_typed]], axis=1)
            n_typed_total = 2 * self.n_typed
        global_l1 = self.lambda_1 * tf.reduce_sum(tf.abs(typed), axis=1) / n_typed_total
        global_l2 = self.lambda_2 * tf.sqrt(
            tf.reduce_sum(tf.square(typed), axis=1) / n_typed_total)
        fitness_per_type.append(fitness_per_type_rmse[:, T] + global_l1 + global_l2)

        # Sort each type's fitness independently → [T+1, P] rank indices
        # ranks_per_type[t] = indices that sort type t's fitness ascending
        ranks_per_type = [tf.argsort(f) for f in fitness_per_type]  # list of [P]

        # Build composite s_sorted: for each variable, permute by its type's ranking
        # Gather all T+1 sorted versions of s, then select per-variable
        # s_sorted_all[t] = s permuted by type t's ranking: [P, dim]
        s_sorted_all = tf.stack([tf.gather(s, r) for r in ranks_per_type])  # [T+1, P, dim]

        # type_of_variable[v] tells us which row of s_sorted_all to use for column v
        tov = self._type_of_variable_tf  # [dim] — cached tf.constant
        # We want s_sorted[:, v] = s_sorted_all[tov[v], :, v]
        # Transpose to [dim, T+1, P] so we can index [v, tov[v]] -> [P]
        s_by_var = tf.transpose(s_sorted_all, [2, 0, 1])  # [dim, T+1, P]
        v_indices = tf.range(self.dim)  # [dim]
        gather_idx = tf.stack([v_indices, tov], axis=1)  # [dim, 2] — [v, tov[v]]
        s_selected = tf.gather_nd(s_by_var, gather_idx)  # [dim, P]
        s_sorted = tf.transpose(s_selected)  # [P, dim]

        return s_sorted

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
        gen_l1, gen_l2 = 0.0, 0.0
        val_fitness = float('inf')
        sigma_min = sigma_max = sigma_mean = sigma_median = float(cfg.init_sigma)
        train_start = time.perf_counter()

        # Report per-type structure coverage
        if self._per_type:
            tc = train_data["types_contained"].numpy()  # [S, T]
            for t in range(cfg.num_types):
                coverage = tc[:, t].sum() / S_train
                print(f"  Type {t}: present in {coverage:.1%} of structures")

        for gen in range(cfg.num_generations):
            t0 = time.perf_counter()

            samples, s = self.ask()

            # Select batch: None = full train set, int = random subset
            if cfg.batch_size is None:
                batch_data = train_data
            else:
                batch_idx_tf = tf.argsort(
                    self.tf_rng.uniform(shape=[S_train]))[:cfg.batch_size]
                gather_keys = ["descriptors", "gradients", "grad_index",
                               "positions", "Z_int", "boxes", "targets",
                               "atom_mask", "neighbor_mask"]
                if "types_contained" in train_data:
                    gather_keys.append("types_contained")
                batch_data = {
                    key: tf.gather(train_data[key], batch_idx_tf)
                    for key in gather_keys
                }

            t1 = time.perf_counter()

            # Evaluate entire population on GPU
            if self._per_type:
                # Per-type mode: get per-type RMSE [P, T+1], then build composite gradients
                fitness_per_type_rmse = self.evaluate_population(
                    samples, batch_data, return_per_type=True)
                fitness = fitness_per_type_rmse[:, -1]  # global RMSE for reporting
            else:
                fitness = self.evaluate_population(samples, batch_data)

            # GPU sync: extract scalar metrics
            avg_fitness = float(tf.reduce_mean(fitness))
            best_rmse = float(tf.reduce_min(fitness))
            worst_rmse = float(tf.reduce_max(fitness))

            t2 = time.perf_counter()

            # Regularization at current mean for reporting (sample every 100 gens to avoid GPU sync)
            if cfg.toggle_regularization and gen % 100 == 0:
                gen_l1, gen_l2 = self.compute_regularization(self.mu)
            elif not cfg.toggle_regularization:
                gen_l1, gen_l2 = 0, 0

            # Rank and update (GPU)
            if self._per_type:
                s_sorted = self._build_per_type_gradients(s, fitness_per_type_rmse, samples)
            else:
                ranks = tf.argsort(fitness)
                s_sorted = tf.gather(s, ranks)
            self.update(self.utilities, s_sorted)

            t3 = time.perf_counter()

            # Validate with updated mean (skip on non-val generations)
            _do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
            if _do_val:
                val_fitness = self.validate(val_data, self.mu)

            t4 = time.perf_counter()

            history["generation"].append(gen)
            history["train_loss"].append(avg_fitness)
            history["val_loss"].append(val_fitness)
            history["L1"].append(gen_l1)
            history["L2"].append(gen_l2)
            history["best_rmse"].append(best_rmse)
            history["worst_rmse"].append(worst_rmse)

            # Sigma stats: sample every 100 gens to avoid GPU→CPU transfer
            if gen % 100 == 0:
                sigma_np = self.sigma.numpy()
                sigma_min = float(np.min(sigma_np))
                sigma_max = float(np.max(sigma_np))
                sigma_mean = float(np.mean(sigma_np))
                sigma_median = float(np.median(sigma_np))
            history["sigma_min"].append(sigma_min)
            history["sigma_max"].append(sigma_max)
            history["sigma_mean"].append(sigma_mean)
            history["sigma_median"].append(sigma_median)

            # VRAM/RAM/CPU monitoring: sample every 100 gens to avoid per-gen overhead
            if gen % 100 == 0:
                if tf.config.list_physical_devices('GPU'):
                    vram_info = tf.config.experimental.get_memory_info('GPU:0')
                    history["vram_mb"].append(vram_info["peak"] / 1024 / 1024)
                else:
                    history["vram_mb"].append(0.0)
                try:
                    ram_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    history["ram_mb"].append(ram_peak_kb / 1024)
                except Exception:
                    history["ram_mb"].append(0.0)
                try:
                    history["cpu_load"].append(os.getloadavg()[0])
                except (OSError, AttributeError):
                    history["cpu_load"].append(0.0)

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
                    f"train RMSE: {avg_fitness:.6f}  "
                    f"val RMSE: {val_fitness:.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping (only update on val generations)
            if _do_val:
                if val_fitness < best_val_loss:
                    best_val_loss = val_fitness
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

            # Stagnation response: interpolate or noise injection
            if (cfg.sigma_reset_patience is not None
                    and cfg.stagnation_response is not None
                    and gens_without_improvement >= cfg.sigma_reset_patience):
                if cfg.stagnation_response == 'interpolate':
                    alpha = cfg.sigma_interpolate_alpha
                    init_vec = tf.fill([self.dim], cfg.init_sigma)
                    self.sigma.assign(self.sigma + alpha * (init_vec - self.sigma))
                    print(f"\nSigma interpolation at generation {gen + 1}: "
                          f"alpha={alpha}, stagnant for "
                          f"{gens_without_improvement} gens")
                elif cfg.stagnation_response == 'noise':
                    # Inject positive noise inversely proportional to current sigma:
                    # small components get a large boost, large ones nearly unchanged.
                    # noise_i = base_std * (init_sigma / sigma_i) * |z_i|
                    # Using |z| ensures noise is always positive (sigma can only grow).
                    noise_std = 10.0 ** cfg.sigma_noise_scale
                    scale = cfg.init_sigma / tf.maximum(self.sigma, 1e-12)
                    noise = noise_std * scale * tf.abs(self.tf_rng.normal(shape=[self.dim]))
                    self.sigma.assign(self.sigma + noise)
                    print(f"\nSigma noise injection at generation {gen + 1}: "
                          f"base_std=10^{cfg.sigma_noise_scale}={noise_std:.2e}, "
                          f"scale range=[{float(tf.reduce_min(scale)):.2f}, "
                          f"{float(tf.reduce_max(scale)):.2f}], "
                          f"stagnant for {gens_without_improvement} gens")
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
        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = tf.reduce_sum(batch_data["atom_mask"], axis=1)  # [B]
            preds = preds / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]

        diff = preds - batch_data["targets"]
        # Apply shear weighting for polarizability off-diagonal components
        if self._pol_weights is not None:
            diff_sq = tf.square(diff) * self._pol_weights
        else:
            diff_sq = tf.square(diff)
        rmse = tf.sqrt(tf.reduce_mean(diff_sq))
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

        For multi-element systems, computes per-type regularization (GPUMD NEP4):
        each type's parameters are penalized separately, averaged across types,
        then added to a global regularization term.

        Args:
            param_vectors : [P, dim] float32

        Returns:
            reg : [P] float32 — L1 + L2 penalty per candidate
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        if T > 1:
            n_per_type = Q * H + H + H  # W0_t + b0_t + W1_t

            total_l1 = tf.zeros([tf.shape(param_vectors)[0]])
            total_l2 = tf.zeros([tf.shape(param_vectors)[0]])

            for t in range(T):
                type_params = self._extract_type_params_batched(param_vectors, t)  # [P, n_per_type]
                total_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
                total_l2 += self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)

            # Average per-type + global over typed params only (excludes b1)
            typed = param_vectors[:, :self.n_typed]  # [P, n_typed]
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                typed2 = param_vectors[:, self.n_primary:self.n_primary + self.n_typed]
                typed = tf.concat([typed, typed2], axis=1)
                n_typed_total = 2 * self.n_typed
            global_l1 = self.lambda_1 * tf.reduce_sum(tf.abs(typed), axis=1) / n_typed_total
            global_l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(typed), axis=1) / n_typed_total)

            return total_l1 / T + global_l1 + total_l2 / T + global_l2
        else:
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(param_vectors), axis=1) / self.dim
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(param_vectors), axis=1) / self.dim)
            return l1 + l2

    def _extract_type_params_batched(self, param_vectors: tf.Tensor, t: int) -> tf.Tensor:
        """Extract type-t parameters from batched flat vectors.

        Args:
            param_vectors : [P, dim] float32
            t             : type index

        Returns:
            [P, Q*H + H + H] float32 — type-t params for each candidate
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H

        return tf.concat([
            param_vectors[:, w0_start:w0_end],
            param_vectors[:, b0_start:b0_end],
            param_vectors[:, w1_start:w1_end],
        ], axis=1)

    def evaluate_population(self, samples_tf: tf.Tensor, batch_data: dict[str, tf.Tensor],
                            include_regularization: bool = True,
                            return_per_type: bool = False) -> tf.Tensor:
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
            include_regularization : bool — if False, return raw RMSE without
                regularization (used by per-type ranking which applies reg separately)
            return_per_type : bool — if True, return [P, T+1] per-type RMSE
                (GPUMD-style: each type's RMSE from structures containing that type,
                 plus global RMSE at index T). Implies include_regularization=False.

        Returns:
            fitness : [P] float32 — RMSE (+ regularization if enabled) per candidate
                      OR [P, T+1] if return_per_type=True
        """
        P = self.pop_size
        B = batch_data["descriptors"].shape[0]
        T_dim = batch_data["targets"].shape[1]
        pop_chunk = self.cfg.population_chunk_size if self.cfg.population_chunk_size is not None else P
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else B

        batch_keys = ["descriptors", "gradients", "grad_index",
                      "positions", "Z_int", "boxes", "targets",
                      "atom_mask", "neighbor_mask"]

        T = self.cfg.num_types
        T_dim_f = tf.cast(T_dim, tf.float32)
        B_f = tf.cast(B, tf.float32)
        use_mae = self._use_mae
        use_inv_weight = self._use_inv_weight

        # Precompute inverse-magnitude weights once: w[b] = 1 / max(||target_b||^2, eps)
        if use_inv_weight:
            tgt_norm_sq = tf.reduce_sum(tf.square(batch_data["targets"]), axis=1)  # [B]
            inv_weights = 1.0 / tf.maximum(tgt_norm_sq, self._inv_weight_eps)      # [B]
            # Normalise so weights sum to B (preserves RMSE scale)
            inv_weights = inv_weights * (B_f / tf.reduce_sum(inv_weights))

        all_fitness = []

        for p_start in range(0, P, pop_chunk):
            p_end = min(p_start + pop_chunk, P)
            candidates = samples_tf[p_start:p_end]  # [C, dim]

            # Accumulate per-structure error across structure chunks
            chunk_parts = []  # list of [C, B_chunk_i]

            for s_start in range(0, B, struct_chunk):
                s_end = min(s_start + struct_chunk, B)
                struct_batch = {key: batch_data[key][s_start:s_end]
                                for key in batch_keys}
                chunk_err = self._evaluate_chunk(candidates, struct_batch)
                chunk_parts.append(chunk_err)          # [C, B_chunk_i]

            # Concatenate structure chunks → [C, B]
            per_struct_err = tf.concat(chunk_parts, axis=1)

            # Apply inverse-magnitude weighting
            if use_inv_weight:
                per_struct_err = per_struct_err * inv_weights[tf.newaxis, :]  # [C, B]

            if return_per_type:
                tc = batch_data["types_contained"]  # [B, T]
                per_type_parts = []
                for t in range(T):
                    tc_t = tc[:, t]                                    # [B] float mask
                    n_t = tf.maximum(tf.reduce_sum(tc_t), 1.0)         # count
                    type_err = tf.einsum('cb,b->c', per_struct_err, tc_t)  # [C]
                    if use_mae:
                        type_fitness = type_err / (n_t * T_dim_f)
                    else:
                        type_fitness = tf.sqrt(type_err / (n_t * T_dim_f))
                    per_type_parts.append(type_fitness)
                # Global (index T)
                global_err = tf.reduce_sum(per_struct_err, axis=1)     # [C]
                if use_mae:
                    global_fitness = global_err / (B_f * T_dim_f)
                else:
                    global_fitness = tf.sqrt(global_err / (B_f * T_dim_f))
                per_type_parts.append(global_fitness)
                all_fitness.append(tf.stack(per_type_parts, axis=1))   # [C, T+1]
            else:
                total_err = tf.reduce_sum(per_struct_err, axis=1)      # [C]
                if use_mae:
                    chunk_fitness = total_err / (B_f * T_dim_f)
                else:
                    chunk_fitness = tf.sqrt(total_err / (B_f * T_dim_f))
                all_fitness.append(chunk_fitness)

        if return_per_type:
            fitness = tf.concat(all_fitness, axis=0)  # [P, T+1]
        else:
            fitness = tf.concat(all_fitness, axis=0)  # [P]

        if not return_per_type and include_regularization and self.cfg.toggle_regularization:
            reg = self.compute_regularization_tf(samples_tf)
            fitness = fitness + reg

        return fitness

    @tf.function(reduce_retracing=True)
    def _evaluate_chunk(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate a chunk of C candidates on B structures.

        Returns per-structure error (SSE or SAE depending on loss_type) so that
        structure chunks can be aggregated correctly in evaluate_population.

        Args:
            chunk_samples : [C, dim]
            batch_data    : dict of [B, ...] tensors

        Returns:
            err : [C, B] — per-structure sum of squared (or absolute) errors
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

        # Pre-scale descriptors and gradients once for all candidates
        # (avoids redundant per-candidate scaling inside vectorized_map)
        if self.model._scale_descriptors:
            scale = self.model._descriptor_mean
            desc = desc / scale
            grads = grads / scale

        # Precompute per-atom normalization factor once (not per-candidate)
        _scale_preds = self.cfg.scale_targets and self.cfg.target_mode == 1
        if _scale_preds:
            num_atoms = tf.reduce_sum(amask, axis=1)  # [B]
            inv_num_atoms = 1.0 / tf.maximum(num_atoms, 1.0)  # [B]
            inv_num_atoms = inv_num_atoms[:, tf.newaxis]  # [B, 1]

        # Reconstruct weights for all candidates in chunk
        params = self.reconstruct_params_tf(chunk_samples)

        _use_mae = self._use_mae

        if self.cfg.target_mode == 2:
            W0, b0, W1, b1, W0p, b0p, W1p, b1p = params
            pol_weights = self._pol_weights  # [6] component weights

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p = args
                preds = self.model.predict_batch(
                    desc, grads, gidx, pos, Z, boxes, amask, nmask,
                    w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p,
                    prescaled=True,
                )
                diff = preds - targets  # [B, 6]
                if _use_mae:
                    return tf.reduce_sum(tf.abs(diff) * tf.sqrt(pol_weights), axis=1)  # [B]
                else:
                    return tf.reduce_sum(tf.square(diff) * pol_weights, axis=1)  # [B]

            stacked = (W0, b0, W1, b1, W0p, b0p, W1p, b1p)
        else:
            W0, b0, W1, b1 = params

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1 = args
                preds = self.model.predict_batch(
                    desc, grads, gidx, pos, Z, boxes, amask, nmask,
                    w0, bb0, w1, bb1, None, None, None, None,
                    prescaled=True,
                )
                if _scale_preds:
                    preds = preds * inv_num_atoms

                diff = preds - targets
                if _use_mae:
                    return tf.reduce_sum(tf.abs(diff), axis=1)     # [B]
                else:
                    return tf.reduce_sum(tf.square(diff), axis=1)  # [B]

            stacked = (W0, b0, W1, b1)

        return tf.vectorized_map(_forward_one_candidate, stacked)  # [C, B]
