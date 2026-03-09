import numpy as np
import tensorflow as tf
from TNEPconfig import TNEPconfig

def _set_model_params(model, *params):
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

    Maintains a diagonal Gaussian search distribution N(μ, diag(σ²)) over
    the flattened parameter vector of the TNEP model.  Each generation:
      1. Sample pop_size candidates: z_p = μ + σ * s_p,  s_p ~ N(0,1)
      2. Evaluate fitness (RMSE) for each candidate on a random batch
      3. Rank candidates by fitness, pair with log-shaped utilities
      4. Update:  μ  ← μ + σ * Σ_p u_p * s_p
                  σ  ← σ * exp(η_σ * Σ_p u_p * (s_p² - 1))

    Total parameter count = num_types * dim_q * num_neurons   (W0)
                          + num_types * num_neurons            (b0)
                          + num_types * num_neurons            (W1)
                          + 1                                  (b1)
    """

    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        self.dim_q = self.cfg.dim_q
        self.batch_size = self.cfg.batch_size
        self.rng = np.random.default_rng(self.cfg.seed)

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

        # Search distribution parameters: μ (mean) and σ (std dev)
        # GPUMD initialises mu in [-1, 1] (see snes.cu line 6709)
        self.mu = self.rng.uniform(-1.0, 1.0, size=self.dim)
        self.sigma = np.full(self.dim, self.cfg.init_sigma, float)

        auto_pop = int(4 + (3 * np.log(self.dim)))
        self.pop_size = self.cfg.pop_size if self.cfg.pop_size is not None else auto_pop


        # Resolve auto-default regularization: sqrt(dim * 1e-6 / num_types)
        # GPUMD divides by num_types for type-specific ANN (version != 3)
        auto_lambda = np.sqrt(self.dim * 1e-6 / self.cfg.num_types)
        self.lambda_1 = self.cfg.lambda_1 if self.cfg.lambda_1 is not None else auto_lambda
        self.lambda_2 = self.cfg.lambda_2 if self.cfg.lambda_2 is not None else auto_lambda

        self.eta_sigma = self.compute_eta_sigma()
        self.utilities = self.compute_utilities()

    def calc_rmse(self, y_true, y_pred):
        """Compute RMSE between prediction and target.

        Args:
            y_true : scalar or [3] tensor — target value
            y_pred : same shape as y_true — predicted value

        Returns:
            float — root mean squared error over all elements
        """
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        diff = y_pred - y_true
        mse = tf.reduce_mean(tf.square(diff))
        rmse_val = tf.sqrt(mse)

        return float(rmse_val.numpy())

    def compute_regularization(self, param_vector):
        """Compute L1 and L2 regularization penalties (GPUMD formula).

        L1 = lambda_1 * sum(|params|) / dim
        L2 = lambda_2 * sqrt(sum(params^2) / dim)

        Args:
            param_vector : ndarray [dim] — flat parameter vector

        Returns:
            l1 : float — L1 penalty
            l2 : float — L2 penalty
        """
        pv = np.asarray(param_vector)
        l1 = self.lambda_1 * np.sum(np.abs(pv)) / self.dim
        l2 = self.lambda_2 * np.sqrt(np.sum(pv ** 2) / self.dim)
        return float(l1), float(l2)

    def compute_eta_sigma(self) -> float:
        """Compute the σ learning rate from per-type parameter dimensionality.

        GPUMD (version != 3) divides by num_types for type-specific ANNs,
        giving a larger step size that accounts for per-type independence.

        Returns:
            η_σ : float — controls how fast σ adapts.
                  η_σ = (3 + ln(num)) / (5 * sqrt(num)) / 2
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
        contribute negative.  Computed once at init.

        Returns:
            utilities : ndarray [pop_size] — weights indexed by rank (0 = best).
        """

        λ = self.pop_size

        ranks = np.arange(λ)
        ranks += 1

        raw = np.log((λ * 0.5) + 1.0) - np.log(ranks)
        raw = np.maximum(0.0, raw)

        # Normalise to sum=1, then shift to zero-mean
        total = tf.reduce_sum(raw).numpy()
        if total > 0:
            raw /= total
        else:
            print("Utility calc failed due to negative total")
        utilities = raw - 1.0 / λ
        print("utilities = ", utilities)
        return utilities

    def ask(self):
        """Sample pop_size candidate parameter vectors from N(μ, diag(σ²)).

        Returns:
            samples : ndarray [pop_size, dim] — candidate parameter vectors
            s       : ndarray [pop_size, dim] — standard normal noise used
        """
        s = self.rng.standard_normal(size=(self.pop_size, self.dim))
        samples = self.mu + s * self.sigma
        return samples, s

    def update(self, utilities, s):
        """Update μ and σ using fitness-ranked noise vectors.

        Args:
            utilities : ndarray [pop_size]     rank-based weights (best first)
            s         : ndarray [pop_size, dim] noise vectors sorted by fitness
                        (s[0] = noise of best individual, s[-1] = worst)

        Mutates self.mu and self.sigma in place.
        """
        grad_mu = np.zeros(self.dim, dtype=float)
        grad_sigma = np.zeros(self.dim, dtype=float)

        for i in range(self.pop_size):
            grad_mu    += utilities[i] * s[i]
            grad_sigma += utilities[i] * (s[i]**2 - 1.0)

        self.mu += self.sigma * grad_mu
        self.sigma = self.sigma * np.exp(self.eta_sigma * grad_sigma)

    def fit(self, train_data, val_data):
        """Run the SNES training loop for num_generations generations.

        Each generation: sample pop_size candidates, evaluate each on a
        random batch of batch_size structures, rank by RMSE, update μ/σ.
        After updating, sets the model weights to the current μ.

        Args:
            train_data : dict with keys descriptors, gradients, grad_index,
                         positions, Z_int, targets, boxes (lists over structures)
            val_data   : same structure, used for validation each generation

        Returns:
            history : dict with keys generation, train_loss, val_loss (lists)
        """
        print("Fitting model...")
        train_descriptors = train_data["descriptors"]
        train_gradients = train_data["gradients"]
        train_grad_index = train_data["grad_index"]
        train_positions = train_data["positions"]
        train_z = train_data["Z_int"]
        train_targets = train_data["targets"]
        boxes = train_data["boxes"]
        cfg = self.cfg
        history = {
            "generation": [],
            "train_loss": [],
            "val_loss": [],
            "L1": [],
            "L2": [],
        }

        # Early stopping state
        best_val_loss = float('inf')
        best_mu = self.mu.copy()
        best_sigma = self.sigma.copy()
        gens_without_improvement = 0

        for gen in range(cfg.num_generations):
            samples, s = self.ask()
            fitness_matrix = np.zeros(shape=self.pop_size, dtype=float)

            # Select one random batch per generation — shared across all candidates
            batch_indices = np.arange(len(train_positions))
            self.rng.shuffle(batch_indices)
            batch_indices = batch_indices[:cfg.batch_size]

            # Evaluate each candidate on the same batch
            # Loss = RMSE + L1 + L2, matching GPUMD's total fitness (see snes.cu regularize)
            for i in range(self.pop_size):
                params = self.reconstruct_params(samples[i])
                _set_model_params(self.model, *params)

                rmse = []
                for j in batch_indices:
                    y_pred = self.model.predict(
                        train_descriptors[j], train_gradients[j],
                        train_grad_index[j], train_positions[j],
                        train_z[j], boxes[j])
                    rmse.append(self.calc_rmse(train_targets[j], y_pred))

                rmse_val = float(tf.reduce_mean(rmse))
                if cfg.toggle_regularization:
                    l1, l2 = self.compute_regularization(samples[i])
                    fitness_matrix[i] = rmse_val + l1 + l2
                else:
                    fitness_matrix[i] = rmse_val

            avg_fitness = tf.reduce_mean(fitness_matrix)
            # Compute regularization at current mean for reporting
            if cfg.toggle_regularization:
                gen_l1, gen_l2 = self.compute_regularization(self.mu)
            else:
                gen_l1, gen_l2 = 0, 0
            # Rank by fitness (ascending) and pair with utilities for update
            ranks = np.argsort(fitness_matrix)
            s_sorted = s[ranks]

            # Update distribution parameters, then validate with current mean
            self.update(self.utilities, s_sorted)

            # Set model weights to the updated mean for validation
            params = self.reconstruct_params(self.mu)
            _set_model_params(self.model, *params)

            val_fitness = self.validate(val_data)

            history["generation"].append(gen)
            history["train_loss"].append(avg_fitness)
            history["val_loss"].append(val_fitness)
            history["L1"].append(gen_l1)
            history["L2"].append(gen_l2)

            print(f"Generation {gen + 1}/{cfg.num_generations} complete, "
                  f"train RMSE: {float(avg_fitness):.4f}, val RMSE: {float(val_fitness):.4f}, "
                  f"L1: {gen_l1:.6f}, L2: {gen_l2:.6f}")

            # Early stopping check
            val_loss_scalar = float(val_fitness)
            if val_loss_scalar < best_val_loss:
                best_val_loss = val_loss_scalar
                best_mu = self.mu.copy()
                best_sigma = self.sigma.copy()
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"Early stopping at generation {gen + 1} "
                      f"(no improvement for {cfg.patience} generations)")
                self.mu = best_mu
                self.sigma = best_sigma
                params = self.reconstruct_params(self.mu)
                _set_model_params(self.model, *params)
                break

        # Restore best parameters even if early stopping didn't trigger
        if cfg.patience is not None:
            self.mu = best_mu
            self.sigma = best_sigma
            params = self.reconstruct_params(self.mu)
            _set_model_params(self.model, *params)

        return history

    def validate(self, val_data):
        """Compute mean RMSE on a random subset of val_size validation structures.

        Args:
            val_data : dict, same structure as train_data

        Returns:
            fitness : scalar tf.Tensor — mean RMSE across sampled structures
        """
        rmse = []
        val_descriptors = val_data["descriptors"]
        val_gradients = val_data["gradients"]
        val_grad_index = val_data["grad_index"]
        val_positions = val_data["positions"]
        val_z = val_data["Z_int"]
        val_targets = val_data["targets"]
        boxes = val_data["boxes"]

        indices = np.arange(len(val_positions))
        self.rng.shuffle(indices)

        for j in indices[:self.cfg.val_size]:
            y_pred = self.model.predict(
                val_descriptors[j], val_gradients[j], val_grad_index[j],
                val_positions[j], val_z[j], boxes[j])
            rmse.append(self.calc_rmse(val_targets[j], y_pred))

        fitness = tf.reduce_mean(rmse)
        return fitness

    def reconstruct_params(self, param_vector):
        """Reconstruct TNEP parameters from a flat vector.

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