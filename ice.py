from __future__ import division
import tensorflow as tf
import settings


class Ice:
    """
    Holds all absorption and scattering coefficients. For now the ice is only
    divided into layers in z-direction.

    Actually for now only homogeneous ice is implemented.

    Parameters
    ----------
    trainable : boolean
        Decides if the ice parameters are supposed to be trainable TF
        variables or simply constants.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, trainable=False):
        self._trainable = trainable
        self._homogenous = False

    def homogeneous_init(self, l_abs=100, l_scat=25):
        """
        Initializes homogeneous ice.

        Parameters
        ----------
        l_abs : float
            Absorption length in meters.
        l_scat : float
            Scattering length in meters.
        """
        # only train absorption for now without time information
        self._homogenous = True
        if self._trainable:
            self._l_abs = tf.Variable(l_abs, dtype=settings.FLOAT_PRECISION)
            # self._l_scat = tf.Variable(l_scat, dtype=settings.FLOAT_PRECISION)
        else:
            self._l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)
        self._l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)

        self._abs_pdf = tf.distributions.Exponential(1/self._l_abs)
        self._scat_pdf = tf.distributions.Exponential(1/self._l_scat)

    def random_init(self, n_layers=10, z_start=0, z_end=1000):
        """
        Initializes the ice randomly.

        Parameters
        ----------
        """
        # TODO: Implement

    # -------------------------- TF Graph Building ----------------------------
    def tf_get_coefficients(self, r):
        """
        Builds the subgraph which grabs the ice coefficients depending on the
        given photon position.

        Parameters
        ----------
        r : TF tensor, 3d vector
            Photon location.

        Returns
        -------
        The absorption and scattering coefficients at the given position r
        inside of the computational graph.
        """
        # TODO: Implement properly for layers
        if self._homogenous:
            return (self._l_abs, self._l_scat)

    def tf_sample_absorption(self, r):
        """
        Samples absorption lengths.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            Photon locations.

        Returns
        -------
        Tensor for the sampled absorption lengths of shape(?).
        """
        return self._abs_pdf.sample(tf.shape(r)[0])

    def tf_sample_scatter(self, r):
        """
        Samples scattering lengths.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            Photon locations.

        Returns
        -------
        Tensor for the sampled scattering lengths of shape(?).
        """
        return self._scat_pdf.sample(tf.shape(r)[0])
