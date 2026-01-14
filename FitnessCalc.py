# FitnessCalc.py

import numpy as np
import tensorflow as tf
from typing import Tuple


class FitnessCalc:
    """
    Compute fitness (to be MINIMIZED) for a given TNEP model and dataset.

    This implementation assumes:
      - model.predict(q, Z) -> per-atom or per-structure outputs of shape
        compatible with `targets`.
      - 'fitness' is a scalar per candidate = sum of RMSE over target features.

    You can change the reduction as needed.
    """

    def __init__(self, model):
        self.model = model
        # per-sample MSE, we will reduce manually
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")

    def calculate(
        self,
        descriptors,
        Z,
        targets
    ) -> float:
        """
        Compute scalar fitness for the CURRENT parameters of `self.model`.

        Parameters
        ----------
        descriptors : np.ndarray
            Shape e.g. (n_struct, N, dim_q) or (N_total, dim_q). You may need to
            adapt this to how you feed TNEP.
        Z : np.ndarray
            Atom types aligned with descriptors (same batch/atom layout).
        targets : np.ndarray
            Target values aligned per structure or per atom.

        Returns
        -------
        fitness : float
            Scalar loss to be minimized (sum of RMSE over target dimensions).
        """
        # Convert to tensors
    #    descriptors_tf = tf.convert_to_tensor(descriptors, dtype=tf.float32)
    #    Z_tf = tf.convert_to_tensor(Z, dtype=tf.int32)
    #    targets_tf = tf.convert_to_tensor(targets, dtype=tf.float32)

        # Forward pass – adapt as needed (per-atom vs per-structure)
        y_pred = self.model.predict(descriptors, Z)
        #print(y_pred)

        # MSE per-sample
        mse = self.loss_fn(targets, y_pred)  # same shape as targets
        print("mean squared error = ", mse)

        # RMSE per feature/component
        rmse_per_feature = tf.sqrt(tf.reduce_mean(mse, axis=0))

        # Total fitness = sum over features
        fitness = tf.reduce_sum(rmse_per_feature)

        return fitness
