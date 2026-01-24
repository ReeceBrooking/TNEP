import numpy as np
import tensorflow as tf
from TNEPconfig import TNEPconfig
from FitnessCalc import FitnessCalc

def _set_model_params(model, W0, b0, W1, b1):
    model.W0.assign(W0)
    model.b0.assign(b0)
    model.W1.assign(W1)
    model.b1.assign(b1)

class SNES:
    """
    Python SNES that mirrors the GPUMD C++ SNES.

    Core features mirrored:
      - Sampling: z = μ + σ * s, with s ~ N(0,1).
      - Utilities: log-shaped, clipped with max(0, ...), normalized,
        then shifted to sum to zero.
      - Per-type ranking: for each type t, we sort the population by
        fitness[:, t] and store type-specific rankings.
      - Per-dimension updates: each variable v looks at the ranking
        of its own type (type_of_variable[v]) and uses that to pair s
        and utilities.
      - Updates:
          μ_v    ← μ_v + σ_v * Σ_p u_p s_(p,v)
          σ_v    ← σ_v * exp(η_σ * Σ_p u_p (s_(p,v)^2 - 1))
      - η_σ: computed from dimensionality as in C++:
          num = dim       (version == 3)
          num = dim / T   (version != 3, with T = num_types)
          η_σ = (3 + log(num)) / (5 * sqrt(num)) / 2
    """

    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        self.dim_q = self.cfg.dim_q
        self.batch_size = self.cfg.batch_size
        self.rng = np.random.default_rng(self.cfg.seed)

        # total number of parameters in TNEP
        n_W0 = self.cfg.num_types * self.cfg.dim_q * self.cfg.num_neurons
        n_b0 = self.cfg.num_types * self.cfg.num_neurons
        n_W1 = self.cfg.num_types * self.cfg.num_neurons
        n_b1 = 1

        self.dim = n_W0 + n_b0 + n_W1 + n_b1

        # Mean and std vectors: μ, σ
        self.mu = self.rng.uniform(-0.5, 0.5, size=self.dim)   # like C++ (r1 - 0.5)*2
        self.sigma = np.full(self.dim, self.cfg.init_sigma, float)  # para.sigma0

        # Precompute η_sigma as in C++ SNES constructor
        self.eta_sigma = self.compute_eta_sigma()
        self.utilities = self.compute_utilities()

    # ------------------------------------------------------------------ #
    # η_sigma and utilities                                              #
    # ------------------------------------------------------------------ #
    def compute_eta_sigma(self) -> float:
        """
        Mirror C++:

            int num = number_of_variables;
            if (para.version != 3) {
                num /= para.num_types;
            }
            eta_sigma = (3.0f + std::log(num)) / (5.0f * sqrt(num)) / 2.0f;

        Notes:
          - We guard against division by zero when num_types == 0 by
            using num_types = 1 in that case.
        """
        num = float(self.dim_q)

        # Avoid weird edge cases
        num = max(num, 1.0)
        eta_sigma = (3.0 + np.log(num)) / (5.0 * np.sqrt(num)) / 2.0
        return float(eta_sigma)

    def compute_utilities(self) -> np.ndarray:
        """
        Mirror SNES::calculate_utility():

            for n in 0..popsize-1:
                utility[n] = max(0, log(λ*0.5 + 1) - log(n+1))
            normalize to sum=1
            then utility[n] = utility[n]/sum - 1/λ

        In C++ these are later used for all types (same u for each type).
        """
        # Utilities should be calculated in init stage, then applied as a map to the sorted fitness matrix to avoid computing utilities over and over again

        λ = self.cfg.pop_size

        ranks = np.arange(λ)
        ranks += 1

        raw = np.log(λ * 0.5 + 1.0) - np.log(ranks)
        raw = np.maximum(0.0, raw)     # max(0, ...)

        # sum the columns and divide columns by sum if more than zero
        sum = tf.reduce_sum(raw).numpy()

        if sum > 0:
            raw /= sum
        # normalize to sum = 1
        utilities = raw - 1.0 / λ
        print("utilities = ", utilities)
        return utilities

    # ------------------------------------------------------------------ #
    # Sampling: create population (gpu_create_population)                #
    # ------------------------------------------------------------------ #
    def ask(self):
        """
        Sample a population from N(μ, diag(σ²)).

        Mirrors gpu_create_population:

            s ~ N(0,1)
            population = μ + σ * s

        Returns
        -------
        population : (popsize, dim) array
            Candidate parameter vectors.
        """
        # s_(p,v) ~ N(0, 1)
        s = self.rng.standard_normal(size=(self.cfg.pop_size, self.dim))
        # z_(p,v) = μ_v + σ_v * s_(p,v)
        samples = self.mu + s * self.sigma
        return samples, s

    # ------------------------------------------------------------------ #
    # Update: use per-type ranking + utilities to update μ, σ           #
    # ------------------------------------------------------------------ #
    def update(self, utilities, s):
        """
        Update μ and σ using provided fitnesses and stored samples s.

        Parameters
        ----------
        fitness_matrix : np.ndarray, shape (popsize, total_types)
            fitness_matrix[p, t] should be the (scalar) total loss
            for individual p and type t, including regularization
            etc. This mirrors the 'fitness[p + (7*t + 0)*population_size]'
            column used in C++ when sorting.

            - Smaller is better (we MINIMIZE).
            - total_types = num_types + 1, index t in [0..num_types].

        Behavior:
          - For each type t, we sort the population indices by
            fitness_matrix[:, t] ascending → index[t, :]
          - For each variable v, we look up type = type_of_variable[v],
            and use index[type, :] as the ordering for s_(p,v).
          - We then apply the same μ/σ update as gpu_update_mu_and_sigma.
        """
        cfg = self.cfg

        grad_mu = np.zeros(self.dim, dtype=float)
        grad_sigma = np.zeros(self.dim, dtype=float)

        for i in range(cfg.pop_size):

            # accumulate gradients
            grad_mu    += utilities[i] * s[i]
            grad_sigma += utilities[i] * (s[i]**2 - 1.0)

        # Update μ and σ
        self.mu += self.sigma * grad_mu
        self.sigma = self.sigma * np.exp(self.eta_sigma * grad_sigma)

    # ------------------------------------------------------------------ #
    # Complete optimization loop (CPU-only version of SNES::compute)    #
    # ------------------------------------------------------------------ #
    def fit(self, train_data, val_data):
        """
        Run SNES for a fixed number of generations (training mode only).

        Parameters
        ----------
        fitness_fn : callable
            Function that takes population (popsize, dim) and returns
            fitness_matrix (popsize, num_types + 1), where each entry
            is the scalar 'total loss' for that type, to be minimized.
        n_generations : int
            Number of SNES generations.
        callback : callable or None
            If given, called as:
                callback(gen, mu, sigma, best_fitness_per_type)

        Returns
        -------
        mu : (dim,) array
            Final μ.
        sigma : (dim,) array
            Final σ.
        history : list of np.ndarray
            Each entry is (num_types+1,) array of best fitness per type.
        """
        train_descriptors = train_data["descriptors"]
        train_gradients = train_data["gradients"]
        train_positions = train_data["positions"]
        train_z = train_data["Z_int"]
        train_targets = train_data["targets"]
        boxes = train_data["boxes"]
        cfg = self.cfg
        history = {
            "generation": [],
            "train_loss": [],
            "val_loss": [],
        }
        for gen in range(cfg.num_generations):
            # 1. Sample population and reconstruct weights and biases
            samples, s = self.ask()
            # cycle through each sample
            fitness_matrix = np.zeros(shape = cfg.pop_size, dtype = float)
            loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
            for i in range(cfg.pop_size):
                W0, b0, W1, b1 = self.reconstruct_params(samples[i])
                # update model layers with sample weights and biases
                _set_model_params(self.model, W0, b0, W1, b1)

                # 2. Compute fitness per individual and type
                mse = 0.0
                mse = tf.convert_to_tensor(mse, dtype = float)
                for j in range(cfg.batch_size):
                    # TODO Change to random sample selector from training dataset
                    descriptors = train_descriptors[j]
                    gradients = train_gradients[j]
                    targets = train_targets[j]
                    Z = train_z[j]
                    positions = train_positions[j]
                    box = boxes[j]

                    # TODO Loss function
                    """ 
                        Temporary FitnessCalc replacement
                    """
                    # Forward pass
                    y_pred = self.model.predict(descriptors, gradients, positions, Z, box)

                    # MSE per-sample
                    assert tf.shape(targets) == tf.shape(y_pred)
                    mse += loss_fn(targets, y_pred)  # same shape as targets

                # RMSE
                rmse = tf.sqrt(mse)

                # Total fitness = sum over features
                fitness_matrix[i] = tf.reduce_sum(rmse)

                assert fitness_matrix.shape[0] == cfg.pop_size

            print("Fitness matrix at generation " + str(gen) + " = " + str(fitness_matrix))
            avg_fitness = tf.reduce_mean(fitness_matrix)
            # Rank fitness and index to utilities
            ranks = np.argsort(fitness_matrix)
            s_sorted = np.zeros_like(s)
            for r in range(len(ranks)):
                s_sorted[ranks[r]] = s[r]

            # 3. SNES update
            self.update(self.utilities, s_sorted)
            val_fitness = self.validate(val_data)
            history["generation"].append(gen)
            history["train_loss"].append(avg_fitness)
            history["val_loss"].append(val_fitness)

        # 4. Reconstruct weights from flat parameter vector and update model
        W0, b0, W1, b1 = self.reconstruct_params(self.mu)
        _set_model_params(self.model, W0, b0, W1, b1)

        return history

    def validate(self, val_data):
        mse = 0.0
        val_descriptors = val_data["descriptors"]
        val_gradients = val_data["gradients"]
        val_positions = val_data["positions"]
        val_z = val_data["Z_int"]
        val_targets = val_data["targets"]
        boxes = val_data["boxes"]
        for j in range(self.cfg.val_size):
            # TODO Change to random sample selector from validation dataset
            loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
            descriptors = val_descriptors[j]
            gradients = val_gradients[j]
            positions = val_positions[j]
            Z = val_z[j]
            targets = val_targets[j]
            box = boxes[j]

            # Loss function
            """ 
                Temporary FitnessCalc replacement
            """
            # Forward pass
            y_pred = self.model.predict(descriptors, gradients, positions, Z, box)

            mse += loss_fn(targets, y_pred)

        # RMSE
        rmse = tf.sqrt(mse)
        fitness = tf.reduce_sum(rmse)
        return fitness

    def reconstruct_params(self, param_vector):
        """
        Reconstruct TNEP parameters (W0, b0, W1, b1) from a flat vector.

        Parameters
        ----------
        param_vector : np.ndarray, shape (d,)
            Flat parameter vector containing all TNEP weights and biases.

        Returns
        -------
        W0 : np.ndarray, shape (num_types, dim_q, num_neurons)
        b0 : np.ndarray, shape (num_types, num_neurons)
        W1 : np.ndarray, shape (num_types, num_neurons)
        b1 : float
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

        offset = 0

        W0_flat = pv[offset: offset + n_W0]
        offset += n_W0
        b0_flat = pv[offset: offset + n_b0]
        offset += n_b0
        W1_flat = pv[offset: offset + n_W1]
        offset += n_W1
        b1_flat = pv[offset: offset + n_b1]

        W0 = W0_flat.reshape((T, Q, H))
        b0 = b0_flat.reshape((T, H))
        W1 = W1_flat.reshape((T, H))
        b1 = float(b1_flat[0])

        # Add per type handling
        return W0, b0, W1, b1