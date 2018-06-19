from __future__ import division
import tensorflow as tf
import numpy as np
import settings


class Ice:
    """
    Holds all absorption and scattering coefficients and manages the TF
    distributions. For now the ice is only divided into layers in z-direction
    for absorption. Scattering is homogenous for now.

    Parameters
    ----------
    trainable : boolean
        Decides if the ice parameters are supposed to be trainable TF
        variables or simply constants. Overwrites the placeholders flag.

    placeholders : boolean
        If True the parameters will be TF placeholders.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, trainable=False, placeholders=False):
        self._trainable = trainable
        if not trainable:
            self._placeholders = placeholders
        else:
            self._placeholders = False

    def init(self, l_abs=[100, 100], l_scat=25, l_z=100):
        """
        Initializes the ice layers.

        Parameters
        ----------
        l_abs : float
            Absorption length in meters.
        l_scat : float
            Scattering length in meters.
        l_scat : float
            Total detector length in z direction in meters.
        """
        if self._trainable:
            self.l_abs = tf.Variable(l_abs, dtype=settings.FLOAT_PRECISION)
            self.l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)
        elif self._placeholders:
            self.l_abs = tf.placeholder(dtype=settings.FLOAT_PRECISION,
                                        shape=())
            self.l_scat = tf.placeholder(dtype=settings.FLOAT_PRECISION,
                                         shape=())
        else:
            self.l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)
            self.l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)

        self._scat_pdf = tf.distributions.Exponential(1/self.l_scat)
        self.N_layer = len(l_abs)

        dz = l_z/self.N_layer
        self.dz = dz
        z_l = np.arange(0, l_z, dz)
        z_l[0] = -1e6
        self._z_l = tf.constant(z_l, dtype=tf.float32)
        z_h = np.arange(dz, l_z + dz, dz)
        z_h[-1] = 1e6
        self._z_h = tf.constant(z_h, dtype=tf.float32)

    # -------------------------- TF Graph Building ----------------------------
    def tf_sample_scatter(self):
        """
        Samples scattering lengths.

        Returns
        -------
        Tensor for the sampled scattering lengths of shape(?).
        """
        return self._scat_pdf.sample(settings.BATCH_SIZE)

    def tf_get_layer_distance(self, r_0, r_1, v, d):
        """
        Calculates the travel distance in each layer for each photon.

        Parameters
        ----------
        r_0 : TF Tensor, shape(?, 3)
            Photon starting positions (scattering point).
        r_1 : TF Tensor, shape(?, 3)
            Photon end positions after (next scattering or hit)>
        v : TF Tensor, shape(?, 3) or None
            Normalized direction vectors r_1 - r_0. Redundant but since it is
            already calculated before it should be passed and not calculated
            again.
        d : TF Tensor, shape(?)
            The distance between r_1 and r_0. Also redundant but already known
            beforehand.

        Returns
        -------
        TF Tensor of shape(?, N_layers) where each entry is the traveled
        distance of the corresponding photon in the corresponding layer.
        """
        # grab z coordinates from start and end vectors
        z_0 = tf.where(r_0[:, 2] < r_1[:, 2], r_0[:, 2], r_1[:, 2])
        z_1 = tf.where(r_0[:, 2] > r_1[:, 2], r_0[:, 2], r_1[:, 2])

        # initialize the distance vector (traveled distance in each layer)
        d_z = tf.zeros([settings.BATCH_SIZE, self.N_layer],
                       dtype=settings.FLOAT_PRECISION)

        # expand and tile for where
        z_0 = tf.tile(tf.expand_dims(z_0, 1), [1, self.N_layer])
        z_1 = tf.tile(tf.expand_dims(z_1, 1), [1, self.N_layer])

        z_l = tf.tile(tf.expand_dims(self._z_l, 0), [settings.BATCH_SIZE, 1])
        z_h = tf.tile(tf.expand_dims(self._z_h, 0), [settings.BATCH_SIZE, 1])

        # completely traversed layers
        d_z += tf.where(tf.logical_and(z_l > z_0, z_h < z_1),
                        self.dz*tf.ones_like(d_z),
                        tf.zeros_like(d_z))

        # starting layer
        d_z += tf.where(tf.logical_and(z_l < z_0, z_h > z_0),
                        z_h - z_0,
                        tf.zeros_like(d_z))

        # last layer
        d_z += tf.where(tf.logical_and(z_l < z_1, z_h > z_1),
                        z_1 - z_l,
                        tf.zeros_like(d_z))

        # rescale to real direction, since v is normalized the dot product and
        # therefore cos of the angle is simply the z component of v
        d_layer = d_z/tf.expand_dims(tf.abs(v[:, 2]), 1)

        # OR only in one layer
        d_layer = tf.where(
            tf.logical_and(tf.logical_and(z_l < z_0, z_h > z_0),
                           tf.logical_and(z_l < z_1, z_h > z_1)),
            tf.tile(tf.expand_dims(d, 1), [1, self.N_layer]), d_layer)
        return d_layer
