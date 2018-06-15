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
        # only train absorption for now
        self._homogenous = True
        if self._trainable:
            self.l_abs = tf.Variable(l_abs, dtype=settings.FLOAT_PRECISION)
        elif self._placeholders:
            self.l_abs = tf.placeholder(dtype=settings.FLOAT_PRECISION,
                                        shape=())
        else:
            self.l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)

        self.l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)

        self._abs_pdf = tf.distributions.Exponential(1/self.l_abs)
        self._scat_pdf = tf.distributions.Exponential(1/self.l_scat)

    # -------------------------- TF Graph Building ----------------------------
    def tf_sample_scatter(self):
        """
        Samples scattering lengths.

        Returns
        -------
        Tensor for the sampled scattering lengths of shape(?).
        """
        return self._scat_pdf.sample(settings.BATCH_SIZE)
