from __future__ import annotations

import sys
import time
import numpy as np
import torch
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from TNEP import TNEP

from data import select_structure_range


def _set_model_params(model: TNEP, *params: torch.Tensor) -> None:
    """Assign weight tensors directly into the TNEP model.

    For modes 0/1: (W0, b0, W1, b1)
    For mode 2:    (W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol)
    """
    model.W0.copy_(params[0])
    model.b0.copy_(params[1])
    model.W1.copy_(params[2])
    model.b1.copy_(params[3])
    if len(params) == 8:
        model.W0_pol.copy_(params[4])
        model.b0_pol.copy_(params[5])
        model.W1_pol.copy_(params[6])
        model.b1_pol.copy_(params[7])


class SNES:
    """Separable Natural Evolution Strategy optimizer for TNEP (PyTorch).

    Maintains a diagonal Gaussian search distribution N(mu, diag(sigma^2)) over
    the flattened parameter vector.  Each generation:
      1. Sample pop_size candidates: z_p = mu + sigma * s_p,  s_p ~ N(0,1)
      2. Evaluate fitness (RMSE) for each candidate on a random batch
      3. Rank candidates by fitness, pair with log-shaped utilities
      4. Update:  mu    <- mu + sigma * sum_p u_p * s_p
                  sigma <- sigma * exp(eta_sigma * sum_p u_p * (s_p^2 - 1))
    """

    def __init__(self, model: TNEP) -> None:
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.dim_q = cfg.dim_q
        self.batch_size = cfg.batch_size
        self.device = model.device

        # Random generator
        if cfg.seed is not None:
            self.rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        else:
            self.rng = torch.Generator(device=self.device)
            self.rng.seed()

        # Total number of trainable parameters
        n_W0 = cfg.num_types * cfg.dim_q * cfg.num_neurons
        n_b0 = cfg.num_types * cfg.num_neurons
        n_W1 = cfg.num_types * cfg.num_neurons
        n_b1 = 1
        self.n_typed = n_W0 + n_b0 + n_W1
        self.n_primary = n_W0 + n_b0 + n_W1 + n_b1
        if cfg.target_mode == 2:
            self.dim = 2 * self.n_primary
        else:
            self.dim = self.n_primary

        # Search distribution — GPUMD initialises mu in [-1, 1]
        rng_np = np.random.default_rng(cfg.seed)
        mu_init = rng_np.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        self.mu = torch.tensor(mu_init, device=self.device)
        self.sigma = torch.full((self.dim,), cfg.init_sigma,
                                dtype=torch.float32, device=self.device)

        auto_pop = int(4 + (3 * np.log(self.dim)))
        self.pop_size = cfg.pop_size if cfg.pop_size is not None else auto_pop

        # Regularization
        auto_lambda = np.sqrt(self.dim * 1e-6 / cfg.num_types)
        self.lambda_1 = cfg.lambda_1 if cfg.lambda_1 is not None else auto_lambda
        self.lambda_2 = cfg.lambda_2 if cfg.lambda_2 is not None else auto_lambda

        if cfg.stagnation_response is not None:
            assert cfg.stagnation_response in ('interpolate', 'noise')

        # Composite loss flags
        self._use_dir_loss = (cfg.direction_loss_weight is not None and cfg.target_mode >= 1)
        self._dir_eps = cfg.direction_loss_eps if self._use_dir_loss else 1e-6
        self._log_magnitude = (self._use_dir_loss and cfg.magnitude_loss_type == "log")

        # Polarizability shear weight
        if cfg.target_mode == 2:
            shear_sq = cfg.lambda_shear ** 2
            self._pol_weights = torch.tensor(
                [1.0, 1.0, 1.0, shear_sq, shear_sq, shear_sq],
                dtype=torch.float32, device=self.device)
        else:
            self._pol_weights = None

        # Per-type ranking
        self._per_type = (cfg.per_type_regularization
                          and cfg.toggle_regularization
                          and cfg.num_types > 1)
        if self._per_type:
            self._type_of_variable = self._build_type_of_variable()

        self.eta_sigma = cfg.eta_sigma if cfg.eta_sigma is not None else self.compute_eta_sigma()
        self.utilities = torch.tensor(
            self.compute_utilities(), dtype=torch.float32, device=self.device)

    def compute_regularization(self, param_vector: torch.Tensor) -> tuple[float, float]:
        """Compute L1 and L2 regularization penalties."""
        pv = param_vector.float()
        T = self.cfg.num_types
        n_per_type = self.n_primary // T

        if T > 1:
            total_l1 = 0.0
            total_l2 = 0.0
            for t in range(T):
                tp = self._extract_type_params(pv, t)
                total_l1 += self.lambda_1 * float(torch.sum(torch.abs(tp))) / n_per_type
                total_l2 += self.lambda_2 * float(torch.sqrt(torch.sum(tp ** 2) / n_per_type))

            typed = pv[:self.n_typed]
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                typed = torch.cat([typed, pv[self.n_primary:self.n_primary + self.n_typed]])
                n_typed_total = 2 * self.n_typed
            l1 = total_l1 / T + self.lambda_1 * float(torch.sum(torch.abs(typed))) / n_typed_total
            l2 = total_l2 / T + self.lambda_2 * float(torch.sqrt(torch.sum(typed ** 2) / n_typed_total))
        else:
            l1 = self.lambda_1 * float(torch.sum(torch.abs(pv))) / self.dim
            l2 = self.lambda_2 * float(torch.sqrt(torch.sum(pv ** 2) / self.dim))
        return l1, l2

    def _extract_type_params(self, pv: torch.Tensor, t: int) -> torch.Tensor:
        """Extract parameters belonging to atom type t from flat vector."""
        T, Q, H = self.cfg.num_types, self.dim_q, self.cfg.num_neurons
        w0_start = t * Q * H
        w0_end = w0_start + Q * H
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H
        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H
        return torch.cat([pv[w0_start:w0_end], pv[b0_start:b0_end], pv[w1_start:w1_end]])

    def _extract_type_params_batched(self, pv: torch.Tensor, t: int) -> torch.Tensor:
        """Extract type-t parameters from batched flat vectors [P, dim]."""
        T, Q, H = self.cfg.num_types, self.dim_q, self.cfg.num_neurons
        w0_start = t * Q * H
        w0_end = w0_start + Q * H
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H
        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H
        return torch.cat([pv[:, w0_start:w0_end], pv[:, b0_start:b0_end],
                          pv[:, w1_start:w1_end]], dim=1)

    def _build_type_of_variable(self) -> np.ndarray:
        """Build array mapping each parameter index to its atom type."""
        T, Q, H = self.cfg.num_types, self.dim_q, self.cfg.num_neurons

        def _ann_types():
            tov = np.empty(self.n_primary, dtype=np.int32)
            offset = 0
            for t in range(T):
                tov[offset:offset + Q * H] = t
                offset += Q * H
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            tov[offset] = T  # b1 = global
            return tov

        if self.cfg.target_mode == 2:
            return np.concatenate([_ann_types(), _ann_types()])
        return _ann_types()

    def _build_per_type_gradients(
        self,
        s: torch.Tensor,
        fitness_per_type_rmse: torch.Tensor,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """Build composite noise matrix with per-type rankings."""
        T = self.cfg.num_types
        Q, H = self.dim_q, self.cfg.num_neurons
        n_per_type = Q * H + H + H

        fitness_per_type = []
        for t in range(T):
            type_params = self._extract_type_params_batched(samples, t)
            l1 = self.lambda_1 * torch.sum(torch.abs(type_params), dim=1) / n_per_type
            l2 = self.lambda_2 * torch.sqrt(
                torch.sum(type_params ** 2, dim=1) / n_per_type)
            fitness_per_type.append(fitness_per_type_rmse[:, t] + l1 + l2)

        # Global ranking
        typed = samples[:, :self.n_typed]
        n_typed_total = self.n_typed
        if self.cfg.target_mode == 2:
            typed = torch.cat([typed, samples[:, self.n_primary:self.n_primary + self.n_typed]], dim=1)
            n_typed_total = 2 * self.n_typed
        global_l1 = self.lambda_1 * torch.sum(torch.abs(typed), dim=1) / n_typed_total
        global_l2 = self.lambda_2 * torch.sqrt(
            torch.sum(typed ** 2, dim=1) / n_typed_total)
        fitness_per_type.append(fitness_per_type_rmse[:, T] + global_l1 + global_l2)

        # Sort each type's fitness independently
        ranks_per_type = [torch.argsort(f) for f in fitness_per_type]

        # Build composite s_sorted
        s_sorted_all = torch.stack([s[r] for r in ranks_per_type])  # [T+1, P, dim]
        tov = torch.tensor(self._type_of_variable, dtype=torch.long, device=self.device)
        s_by_var = s_sorted_all.permute(2, 0, 1)  # [dim, T+1, P]
        v_indices = torch.arange(self.dim, device=self.device)
        s_selected = s_by_var[v_indices, tov]  # [dim, P]
        s_sorted = s_selected.T  # [P, dim]
        return s_sorted

    def compute_eta_sigma(self) -> float:
        """Compute sigma learning rate."""
        num = float(self.dim) / self.cfg.num_types
        num = max(num, 1.0)
        return ((3.0 + np.log(num)) / (5.0 * np.sqrt(num))) / 2.0

    def compute_utilities(self) -> np.ndarray:
        """Precompute rank-based utility weights."""
        lam = self.pop_size
        ranks = np.arange(lam) + 1
        raw = np.log((lam * 0.5) + 1.0) - np.log(ranks)
        raw = np.maximum(0.0, raw)
        total = np.sum(raw)
        if total > 0:
            raw /= total
        utilities = raw - 1.0 / lam
        return utilities

    def ask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample pop_size candidates from N(mu, diag(sigma^2))."""
        s = torch.randn(self.pop_size, self.dim, generator=self.rng, device=self.device)
        samples = self.mu + s * self.sigma
        return samples, s

    def update(self, utilities: torch.Tensor, s: torch.Tensor) -> None:
        """Update mu and sigma using fitness-ranked noise vectors."""
        grad_mu = torch.einsum('p,pd->d', utilities, s)
        grad_sigma = torch.einsum('p,pd->d', utilities, s ** 2 - 1.0)
        self.mu.add_(self.sigma * grad_mu)
        self.sigma.mul_(torch.exp(self.eta_sigma * grad_sigma))

    def fit(self, train_data: dict[str, torch.Tensor], val_data: dict[str, torch.Tensor],
            plot_callback: Callable | None = None) -> dict:
        """Run the SNES training loop."""
        print("Fitting model...")
        cfg = self.cfg
        S_train = int(train_data["targets"].shape[0])

        history = {
            "generation": [], "train_loss": [], "val_loss": [],
            "L1": [], "L2": [], "best_rmse": [], "worst_rmse": [],
            "sigma_min": [], "sigma_max": [], "sigma_mean": [], "sigma_median": [],
            "sigma_resets": [], "vram_mb": [], "ram_mb": [], "cpu_load": [],
            "timing": {
                "sample_batch": [], "evaluate": [], "rank_update": [],
                "validate": [], "overhead": [],
            },
        }

        best_val_loss = float('inf')
        best_mu = self.mu.clone()
        best_sigma = self.sigma.clone()
        gens_without_improvement = 0
        train_start = time.perf_counter()

        # Precompute all-types-present check
        if self._per_type:
            Z_int_np = train_data["Z_int"].cpu().numpy()
            atom_batch_np = train_data["atom_batch"].cpu().numpy()
            self._train_all_types_present = True
            for t in range(cfg.num_types):
                atoms_of_type_t = Z_int_np == t
                structs_with_t = set(atom_batch_np[atoms_of_type_t])
                if len(structs_with_t) < S_train:
                    self._train_all_types_present = False
                    break
            if self._train_all_types_present:
                print("Per-type RMSE: all types present in all structures — using fast path")

        for gen in range(cfg.num_generations):
            t0 = time.perf_counter()

            samples, s = self.ask()

            # Select batch
            if cfg.batch_size is None:
                batch_data = train_data
            else:
                perm = torch.randperm(S_train, generator=self.rng, device=self.device)[:cfg.batch_size]
                perm_sorted, _ = torch.sort(perm)
                # Re-collate from flat data using offset-based slicing
                # For simplicity, gather contiguous ranges where possible
                batch_data = self._select_batch(train_data, perm_sorted)

            t1 = time.perf_counter()

            # Evaluate population
            if self._per_type:
                fitness_per_type_rmse = self.evaluate_population(
                    samples, batch_data, return_per_type=True)
                fitness = fitness_per_type_rmse[:, -1]
            else:
                fitness = self.evaluate_population(samples, batch_data)

            avg_fitness = float(fitness.mean())
            best_rmse = float(fitness.min())
            worst_rmse = float(fitness.max())

            # VRAM snapshot
            if torch.cuda.is_available():
                history["vram_mb"].append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            else:
                history["vram_mb"].append(0.0)

            # RAM and CPU
            try:
                import psutil
                history["ram_mb"].append(psutil.Process().memory_info().rss / 1024 / 1024)
                history["cpu_load"].append(psutil.cpu_percent())
            except ImportError:
                history["ram_mb"].append(0.0)
                history["cpu_load"].append(0.0)

            t2 = time.perf_counter()

            # Regularization for reporting
            if cfg.toggle_regularization:
                gen_l1, gen_l2 = self.compute_regularization(self.mu)
            else:
                gen_l1, gen_l2 = 0, 0

            # Rank and update
            if self._per_type:
                s_sorted = self._build_per_type_gradients(s, fitness_per_type_rmse, samples)
            else:
                ranks = torch.argsort(fitness)
                s_sorted = s[ranks]
            self.update(self.utilities, s_sorted)

            t3 = time.perf_counter()

            # Validate
            val_fitness = self.validate(val_data, self.mu)

            t4 = time.perf_counter()

            history["generation"].append(gen)
            history["train_loss"].append(avg_fitness)
            history["val_loss"].append(float(val_fitness))
            history["L1"].append(gen_l1)
            history["L2"].append(gen_l2)
            history["best_rmse"].append(best_rmse)
            history["worst_rmse"].append(worst_rmse)
            sigma_np = self.sigma.cpu().numpy()
            history["sigma_min"].append(float(sigma_np.min()))
            history["sigma_max"].append(float(sigma_np.max()))
            history["sigma_mean"].append(float(sigma_np.mean()))
            history["sigma_median"].append(float(np.median(sigma_np)))

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
                    f"val RMSE: {float(val_fitness):.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping
            val_loss_scalar = float(val_fitness)
            if val_loss_scalar < best_val_loss:
                best_val_loss = val_loss_scalar
                best_mu = self.mu.clone()
                best_sigma = self.sigma.clone()
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"\nEarly stopping at generation {gen + 1} "
                      f"(no improvement for {cfg.patience} generations)")
                self.mu.copy_(best_mu)
                self.sigma.copy_(best_sigma)
                break

            # Stagnation response
            if (cfg.sigma_reset_patience is not None
                    and cfg.stagnation_response is not None
                    and gens_without_improvement >= cfg.sigma_reset_patience):
                if cfg.stagnation_response == 'interpolate':
                    alpha = cfg.sigma_interpolate_alpha
                    init_vec = torch.full((self.dim,), cfg.init_sigma, device=self.device)
                    self.sigma.add_(alpha * (init_vec - self.sigma))
                    print(f"\nSigma interpolation at generation {gen + 1}: "
                          f"alpha={alpha}, stagnant for "
                          f"{gens_without_improvement} gens")
                elif cfg.stagnation_response == 'noise':
                    noise_std = 10.0 ** cfg.sigma_noise_scale
                    scale = cfg.init_sigma / self.sigma.clamp(min=1e-12)
                    noise = noise_std * scale * torch.abs(
                        torch.randn(self.dim, generator=self.rng, device=self.device))
                    self.sigma.add_(noise)
                    print(f"\nSigma noise injection at generation {gen + 1}: "
                          f"base_std=10^{cfg.sigma_noise_scale}={noise_std:.2e}, "
                          f"scale range=[{float(scale.min()):.2f}, "
                          f"{float(scale.max()):.2f}], "
                          f"stagnant for {gens_without_improvement} gens")
                gens_without_improvement = 0
                history["sigma_resets"].append(gen)

            t5 = time.perf_counter()
            history["timing"]["sample_batch"].append(t1 - t0)
            history["timing"]["evaluate"].append(t2 - t1)
            history["timing"]["rank_update"].append(t3 - t2)
            history["timing"]["validate"].append(t4 - t3)
            history["timing"]["overhead"].append(t5 - t4)

            # Periodic plotting
            if gen + 1 < cfg.num_generations:
                if (plot_callback is not None
                        and cfg.plot_interval is not None
                        and (gen + 1) % cfg.plot_interval == 0):
                    params = self.reconstruct_params_torch(best_mu)
                    _set_model_params(self.model, *params)
                    plot_callback(history, gen + 1)
                    params = self.reconstruct_params_torch(self.mu)
                    _set_model_params(self.model, *params)

        print()
        self.mu.copy_(best_mu)
        self.sigma.copy_(best_sigma)
        params = self.reconstruct_params_torch(self.mu)
        _set_model_params(self.model, *params)
        return history

    def _select_batch(self, train_data: dict[str, torch.Tensor],
                      sorted_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select a sub-batch of structures by sorted indices from flat data."""
        indices_np = sorted_indices.cpu().numpy()
        S = int(train_data["targets"].shape[0])

        # Check if indices form a contiguous range
        if len(indices_np) > 0 and indices_np[-1] - indices_np[0] + 1 == len(indices_np):
            return select_structure_range(train_data, int(indices_np[0]), int(indices_np[-1]) + 1)

        # Non-contiguous: re-collate from individual structure ranges
        sub_batches = []
        for idx in indices_np:
            sub_batches.append(select_structure_range(train_data, int(idx), int(idx) + 1))

        # Merge sub-batches
        all_desc, all_Z, all_pos, all_ab = [], [], [], []
        all_grads, all_es, all_ed, all_eb = [], [], [], []
        all_targets, all_boxes, all_na = [], [], []
        atom_offset = 0
        edge_offset = 0
        atom_offset_list = [0]
        edge_offset_list = [0]
        for i, sb in enumerate(sub_batches):
            n_atoms = int(sb["num_atoms"].sum())
            n_edges = sb["gradients"].shape[0]
            all_desc.append(sb["descriptors"])
            all_Z.append(sb["Z_int"])
            all_pos.append(sb["positions"])
            all_ab.append(sb["atom_batch"] + i)
            all_grads.append(sb["gradients"])
            all_es.append(sb["edge_src"] + atom_offset)
            all_ed.append(sb["edge_dst"] + atom_offset)
            all_eb.append(sb["edge_batch"] + i)
            all_targets.append(sb["targets"])
            all_boxes.append(sb["boxes"])
            all_na.append(sb["num_atoms"])
            atom_offset += n_atoms
            edge_offset += n_edges
            atom_offset_list.append(atom_offset)
            edge_offset_list.append(edge_offset)

        return {
            "descriptors": torch.cat(all_desc),
            "Z_int": torch.cat(all_Z),
            "positions": torch.cat(all_pos),
            "atom_batch": torch.cat(all_ab),
            "gradients": torch.cat(all_grads),
            "edge_src": torch.cat(all_es),
            "edge_dst": torch.cat(all_ed),
            "edge_batch": torch.cat(all_eb),
            "targets": torch.cat(all_targets),
            "boxes": torch.cat(all_boxes),
            "num_atoms": torch.cat(all_na),
            "atom_offsets": torch.tensor(atom_offset_list, dtype=torch.int64,
                                         device=all_desc[0].device),
            "edge_offsets": torch.tensor(edge_offset_list, dtype=torch.int64,
                                         device=all_desc[0].device),
        }

    def validate(self, val_data: dict[str, torch.Tensor],
                 mu: torch.Tensor | None = None) -> float:
        """Compute RMSE on validation data."""
        if self.cfg.val_size is not None:
            S_val = int(val_data["targets"].shape[0])
            perm = torch.randperm(S_val, generator=self.rng, device=self.device)[:self.cfg.val_size]
            perm_sorted, _ = torch.sort(perm)
            batch_data = self._select_batch(val_data, perm_sorted)
        else:
            batch_data = val_data

        if mu is not None:
            params = self.reconstruct_params_torch(mu)
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

        with torch.no_grad():
            preds = self.model.predict_flat(batch_data, W0, b0, W1, b1, W0p, b0p, W1p, b1p)

        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = batch_data["num_atoms"].float().clamp(min=1.0).unsqueeze(1)
            preds = preds / num_atoms

        if self._use_dir_loss:
            targets = batch_data["targets"]
            pred_norm = torch.linalg.norm(preds, dim=1)
            tgt_norm = torch.linalg.norm(targets, dim=1)
            eps = self._dir_eps
            if self._log_magnitude:
                mag_rmse = torch.sqrt(torch.mean(
                    (torch.log(pred_norm.clamp(min=eps)) - torch.log(tgt_norm.clamp(min=eps))) ** 2))
            else:
                mag_rmse = torch.sqrt(torch.mean((pred_norm - tgt_norm) ** 2))
            dot = torch.sum(preds * targets, dim=1)
            denom = (pred_norm * tgt_norm).clamp(min=eps)
            cos_sim = dot / denom
            mask = (tgt_norm > eps).float()
            n_valid = mask.sum().clamp(min=1.0)
            dir_loss = 1.0 - torch.sum(cos_sim * mask) / n_valid
            return float(mag_rmse + self.cfg.direction_loss_weight * dir_loss)
        else:
            diff = preds - batch_data["targets"]
            if self._pol_weights is not None:
                rmse = torch.sqrt(torch.mean(diff ** 2 * self._pol_weights))
            else:
                rmse = torch.sqrt(torch.mean(diff ** 2))
            return float(rmse)

    def reconstruct_params(self, param_vector: np.ndarray) -> tuple:
        """Reconstruct TNEP parameters from a flat numpy vector."""
        pv = np.asarray(param_vector, dtype=np.float32)
        T, Q, H = self.cfg.num_types, self.dim_q, self.cfg.num_neurons
        n_W0, n_b0, n_W1, n_b1 = T * Q * H, T * H, T * H, 1

        def _extract(pv, offset):
            W0 = pv[offset:offset + n_W0].reshape(T, Q, H); offset += n_W0
            b0 = pv[offset:offset + n_b0].reshape(T, H); offset += n_b0
            W1 = pv[offset:offset + n_W1].reshape(T, H); offset += n_W1
            b1 = float(pv[offset]); offset += n_b1
            return W0, b0, W1, b1, offset

        W0, b0, W1, b1, offset = _extract(pv, 0)
        if self.cfg.target_mode == 2:
            W0p, b0p, W1p, b1p, _ = _extract(pv, offset)
            return W0, b0, W1, b1, W0p, b0p, W1p, b1p
        return W0, b0, W1, b1

    def reconstruct_params_torch(self, param_vectors: torch.Tensor) -> tuple:
        """Reconstruct TNEP weight tensors from flat torch tensor.

        Works for single [dim] or batched [P, dim] vectors.
        """
        T, Q, H = self.cfg.num_types, self.dim_q, self.cfg.num_neurons
        n_W0, n_b0, n_W1, n_b1 = T * Q * H, T * H, T * H, 1
        is_batched = param_vectors.dim() == 2

        def _extract(pv, offset):
            shape_W0 = (-1, T, Q, H) if is_batched else (T, Q, H)
            shape_bW = (-1, T, H) if is_batched else (T, H)
            W0 = pv[..., offset:offset + n_W0].reshape(shape_W0); offset += n_W0
            b0 = pv[..., offset:offset + n_b0].reshape(shape_bW); offset += n_b0
            W1 = pv[..., offset:offset + n_W1].reshape(shape_bW); offset += n_W1
            b1 = pv[..., offset]; offset += n_b1
            return W0, b0, W1, b1, offset

        W0, b0, W1, b1, offset = _extract(param_vectors, 0)
        if self.cfg.target_mode == 2:
            W0p, b0p, W1p, b1p, _ = _extract(param_vectors, offset)
            return W0, b0, W1, b1, W0p, b0p, W1p, b1p
        return W0, b0, W1, b1

    def compute_regularization_torch(self, param_vectors: torch.Tensor) -> torch.Tensor:
        """Compute L1+L2 regularization for batched param vectors [P, dim]."""
        T = self.cfg.num_types
        Q, H = self.dim_q, self.cfg.num_neurons

        if T > 1:
            n_per_type = Q * H + H + H
            P = param_vectors.shape[0]
            total_l1 = torch.zeros(P, device=self.device)
            total_l2 = torch.zeros(P, device=self.device)

            for t in range(T):
                tp = self._extract_type_params_batched(param_vectors, t)
                total_l1 += self.lambda_1 * torch.sum(torch.abs(tp), dim=1) / n_per_type
                total_l2 += self.lambda_2 * torch.sqrt(torch.sum(tp ** 2, dim=1) / n_per_type)

            typed = param_vectors[:, :self.n_typed]
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                typed = torch.cat([typed, param_vectors[:, self.n_primary:self.n_primary + self.n_typed]], dim=1)
                n_typed_total = 2 * self.n_typed
            global_l1 = self.lambda_1 * torch.sum(torch.abs(typed), dim=1) / n_typed_total
            global_l2 = self.lambda_2 * torch.sqrt(torch.sum(typed ** 2, dim=1) / n_typed_total)
            return total_l1 / T + global_l1 + total_l2 / T + global_l2
        else:
            l1 = self.lambda_1 * torch.sum(torch.abs(param_vectors), dim=1) / self.dim
            l2 = self.lambda_2 * torch.sqrt(torch.sum(param_vectors ** 2, dim=1) / self.dim)
            return l1 + l2

    def _precompute_shared(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pre-compute quantities shared across all candidates.

        1. Scale descriptors and gradients
        2. Compute edge displacements
        3. Pad to static shapes for torch.compile

        Returns a new dict with padded tensors ready for predict_padded_batched.
        """
        from scatter_ops import edge_displacements
        out = dict(batch_data)  # shallow copy

        # Scale descriptors and gradients once
        if self.model._scale_descriptors:
            scale = self.model._descriptor_mean
            out["desc_scaled"] = out["descriptors"] / scale
            out["grads_scaled"] = out["gradients"] / scale
        else:
            out["desc_scaled"] = out["descriptors"]
            out["grads_scaled"] = out["gradients"]

        # Edge displacements (for modes 1 and 2)
        if self.cfg.target_mode >= 1:
            dr, rij = edge_displacements(
                out["positions"], out["boxes"],
                out["edge_src"], out["edge_dst"], out["edge_batch"])
            out["dr"] = dr
            out["rij2"] = rij ** 2

        # Pad to static shapes for compiled batched evaluation
        out = self.model.pad_batch(out)

        return out

    def evaluate_population(self, samples: torch.Tensor, batch_data: dict[str, torch.Tensor],
                            include_regularization: bool = True,
                            return_per_type: bool = False) -> torch.Tensor:
        """Evaluate all SNES candidates on a batch of structures.

        Uses torch.vmap to vectorise the forward pass over the population
        dimension. Chunks along both population and structure dimensions
        to limit VRAM.
        """
        P = self.pop_size
        S = int(batch_data["targets"].shape[0])
        T_dim = int(batch_data["targets"].shape[1])
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else S
        pop_chunk = self.cfg.population_chunk_size if self.cfg.population_chunk_size is not None else P
        T = self.cfg.num_types

        # Precompute per-type structure indices
        if return_per_type:
            all_types_present = getattr(self, '_train_all_types_present', False)
            if not all_types_present:
                Z_int_np = batch_data["Z_int"].cpu().numpy()
                ab_np = batch_data["atom_batch"].cpu().numpy()
                type_struct_indices = []
                all_types_present_check = True
                for t in range(T):
                    atoms_of_t = Z_int_np == t
                    structs_with_t = np.unique(ab_np[atoms_of_t])
                    type_struct_indices.append(structs_with_t)
                    if len(structs_with_t) < S:
                        all_types_present_check = False
                type_struct_indices.append(np.arange(S))
                if all_types_present_check:
                    all_types_present = True

        # Pre-compute struct batches with shared quantities (scaled desc, edge displacements)
        # so they are reused across population chunks instead of recomputed each time
        struct_batches = []
        for s_start in range(0, S, struct_chunk):
            s_end = min(s_start + struct_chunk, S)
            sb = select_structure_range(batch_data, s_start, s_end)
            sb = self._precompute_shared(sb)
            struct_batches.append((sb, (s_end - s_start) * T_dim))

        all_fitness = []

        for p_start in range(0, P, pop_chunk):
            p_end = min(p_start + pop_chunk, P)
            candidates = samples[p_start:p_end]
            C = p_end - p_start

            if return_per_type:
                if all_types_present:
                    # Fast path: evaluate once, broadcast
                    total_sse = torch.zeros(C, device=self.device)
                    total_elements = 0
                    for sb, n_elem in struct_batches:
                        chunk_sse = self._evaluate_chunk(candidates, sb)
                        total_sse += chunk_sse
                        total_elements += n_elem
                    global_rmse = torch.sqrt(total_sse / total_elements)
                    all_fitness.append(global_rmse.unsqueeze(1).expand(-1, T + 1).clone())
                else:
                    # Slow path: per-type evaluation
                    per_type_rmse = []
                    for t_idx in range(T + 1):
                        indices = type_struct_indices[t_idx]
                        B_t = len(indices)
                        if B_t == 0:
                            per_type_rmse.append(torch.zeros(C, device=self.device))
                            continue
                        type_sse = torch.zeros(C, device=self.device)
                        type_elements = 0
                        for s_i in range(0, B_t, struct_chunk):
                            s_j = min(s_i + struct_chunk, B_t)
                            sub_indices = indices[s_i:s_j]
                            struct_batch = self._select_batch(
                                batch_data, torch.tensor(sub_indices, device=self.device))
                            chunk_sse = self._evaluate_chunk(candidates, struct_batch)
                            type_sse += chunk_sse
                            type_elements += (s_j - s_i) * T_dim
                        type_rmse_val = torch.sqrt(type_sse / max(type_elements, 1))
                        per_type_rmse.append(type_rmse_val)
                    all_fitness.append(torch.stack(per_type_rmse, dim=1))
            else:
                total_sse = torch.zeros(C, device=self.device)
                total_elements = 0
                for sb, n_elem in struct_batches:
                    chunk_sse = self._evaluate_chunk(candidates, sb)
                    total_sse += chunk_sse
                    total_elements += n_elem
                chunk_fitness = torch.sqrt(total_sse / total_elements)
                all_fitness.append(chunk_fitness)

        if return_per_type:
            fitness = torch.cat(all_fitness, dim=0)
        else:
            fitness = torch.cat(all_fitness)

        if not return_per_type and include_regularization and self.cfg.toggle_regularization:
            reg = self.compute_regularization_torch(samples)
            fitness = fitness + reg

        return fitness

    @torch.no_grad()
    def _evaluate_chunk(self, chunk_samples: torch.Tensor,
                        batch_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate C candidates in parallel via padded batched forward pass.

        Uses padded tensors with torch.compile for fused GPU execution,
        matching TF's @tf.function + padded tensor performance.

        Returns:
            sse : [C] sum of squared errors per candidate
        """
        if "desc_pad" not in batch_data:
            batch_data = self._precompute_shared(batch_data)

        params = self.reconstruct_params_torch(chunk_samples)  # batched [C, ...]
        if self.cfg.target_mode == 2:
            W0, b0, W1, b1, W0p, b0p, W1p, b1p = params
            # Mode 2 padded path not yet implemented — fall back to sequential
            C = chunk_samples.shape[0]
            sse = torch.zeros(C, device=self.device)
            targets = batch_data["targets"]
            for c in range(C):
                preds = self.model.predict_flat(
                    batch_data, W0[c], b0[c], W1[c], b1[c], W0p[c], b0p[c], W1p[c], b1p[c])
                diff = preds - targets
                sse[c] = torch.sum(diff ** 2 * self._pol_weights) if self._pol_weights is not None else torch.sum(diff ** 2)
            return sse

        W0, b0, W1, b1 = params

        preds = self.model.predict_padded_batched(
            batch_data, W0, b0, W1, b1)  # [C, S, T_dim]

        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = batch_data["num_atoms"].float().clamp(min=1.0)
            preds = preds / num_atoms.unsqueeze(0).unsqueeze(2)

        targets = batch_data["targets"]          # [S, T_dim]
        diff = preds - targets.unsqueeze(0)      # [C, S, T_dim]

        if self._pol_weights is not None:
            sse = torch.sum(diff ** 2 * self._pol_weights, dim=(1, 2))
        else:
            sse = torch.sum(diff ** 2, dim=(1, 2))

        return sse
