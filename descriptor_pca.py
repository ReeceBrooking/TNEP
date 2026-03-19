from __future__ import annotations

import numpy as np
import tensorflow as tf


class DescriptorPCA:
    """PCA compression for SOAP-turbo descriptors.

    Fits PCA on training descriptors, then transforms descriptors
    and gradients to reduced dimension. Resembles SOAP-turbo's
    compress_mode="linear" but applied as post-processing.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None    # [n_components, dim_q]
        self.mean_: np.ndarray | None = None          # [dim_q]
        self.explained_variance_ratio_: np.ndarray | None = None  # [n_components]

    def fit(self, descriptors: list[tf.Tensor]) -> DescriptorPCA:
        """Fit PCA on training descriptors.

        Args:
            descriptors: list of [N_i, dim_q] tensors (one per structure)

        Returns:
            self
        """
        all_desc = tf.concat(descriptors, axis=0).numpy()  # [total_atoms, dim_q]
        self.mean_ = np.mean(all_desc, axis=0)              # [dim_q]
        centered = all_desc - self.mean_                     # [total_atoms, dim_q]

        # Covariance matrix [dim_q, dim_q] — symmetric, use eigh
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order — reverse for descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        self.components_ = eigenvectors[:, :self.n_components].T  # [n_components, dim_q]

        cumvar = np.cumsum(self.explained_variance_ratio_)
        print(f"PCA fit: {self.n_components} components explain "
              f"{cumvar[-1]:.4f} of variance")
        print(f"  Top-5 cumulative: {cumvar[:5]}")
        return self

    def transform_descriptors(self, descriptors: list[tf.Tensor]) -> list[tf.Tensor]:
        """Project descriptors to PCA space.

        Args:
            descriptors: list of [N_i, dim_q] tensors

        Returns:
            list of [N_i, n_components] tensors
        """
        P = self.components_.T.astype(np.float32)   # [dim_q, n_components]
        mean = self.mean_.astype(np.float32)          # [dim_q]
        return [tf.constant((d.numpy() - mean) @ P) for d in descriptors]

    def transform_gradients(self, gradients: list[list[tf.Tensor]]) -> list[list[tf.Tensor]]:
        """Project descriptor gradients to PCA space.

        Gradients are dq/dR, so the chain rule for q_compressed = (q - mean) @ P
        gives d(q_compressed)/dR = dq/dR @ P (mean is constant).

        Args:
            gradients: list of (list of [M_i, 3, dim_q] tensors)

        Returns:
            list of (list of [M_i, 3, n_components] tensors)
        """
        P = tf.constant(self.components_.T, dtype=tf.float32)  # [dim_q, n_components]
        result = []
        for struct_grads in gradients:
            result.append([tf.constant(tf.einsum('mdi,ip->mdp', g, P)) for g in struct_grads])
        return result

    def transform(
        self,
        descriptors: list[tf.Tensor],
        gradients: list[list[tf.Tensor]],
    ) -> tuple[list[tf.Tensor], list[list[tf.Tensor]]]:
        """Transform both descriptors and gradients."""
        return self.transform_descriptors(descriptors), self.transform_gradients(gradients)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Serialize for model saving."""
        return {
            "pca_components": self.components_.astype(np.float32),
            "pca_mean": self.mean_.astype(np.float32),
            "pca_explained_variance_ratio": self.explained_variance_ratio_.astype(np.float32),
            "pca_n_components": np.array(self.n_components, dtype=np.int32),
        }

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> DescriptorPCA:
        """Deserialize from saved model."""
        n = int(d["pca_n_components"])
        pca = cls(n_components=n)
        pca.components_ = d["pca_components"].astype(np.float32)
        pca.mean_ = d["pca_mean"].astype(np.float32)
        pca.explained_variance_ratio_ = d["pca_explained_variance_ratio"].astype(np.float32)
        return pca
