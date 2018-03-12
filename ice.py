from __future__ import division
import tensorflow as tf
import settings


class Ice:
    """
    Holds all absorbtion and scattering coefficients. For now the ice is only
    devided into layers in z-direction.

    Actually for now only homogenous ice is implemented.
    """
    def __init__(self, trainable=False):
        """
        Constructor.

        Parameters
        ----------
        trainable: boolean. Decides if the ice parameters are supposed to be
                   trainable tf variables or simply constants.
        """
        self._trainable = trainable

    def homogeneous_init(self, l_abs=100, l_scat=25):
        """
        Initializes homogeneous ice.

        Parameters
        ----------
        l_abs: absorbtion length in meters
        l_scat: scattering length in meters
        """
        if self._trainable:
            self._l_abs = tf.Variable(l_abs, dtype=settings.FLOAT_PRECISION)
            self._l_scat = tf.Variable(l_scat, dtype=settings.FLOAT_PRECISION)
        else:
            self._l_abs = tf.constant(l_abs, dtype=settings.FLOAT_PRECISION)
            self._l_scat = tf.constant(l_scat, dtype=settings.FLOAT_PRECISION)

    def random_init(self, n_layers=10, z_start=0, z_end=1000):
        """
        Initializes the ice randomly.

        Parameters
        ----------

        """
        # TODO: Implement

    def tf_get_coefficients(self, r):
        """
        Builds the subgraph which grabs the ice coefficients depending on the
        given photon position.

        Parameters
        ----------
        r: TF tensor photon location
        """
        # TODO: Implement properly for layers
        return (self._l_abs, self._l_scat)
