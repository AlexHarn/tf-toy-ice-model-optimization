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
        # only train scattering, which triggers the gradient sign bug
        self._homogenous = True
        if self._trainable:
            self.l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)
            self.l_scat = tf.Variable(l_scat, dtype=settings.FLOAT_PRECISION)
        elif self._placeholders:
            self.l_abs = tf.placeholder(dtype=settings.FLOAT_PRECISION,
                                        shape=())
            self.l_scat = tf.placeholder(dtype=settings.FLOAT_PRECISION,
                                         shape=())
        else:
            self.l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)
            self.l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)

        self._abs_pdf = tf.distributions.Exponential(1/self.l_abs)
        self._scat_pdf = tf.distributions.Exponential(1/self.l_scat)

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
            return (self.l_abs, self.l_scat)

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
