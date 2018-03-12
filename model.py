from __future__ import division
import tensorflow as tf
import numpy as np
import settings


class Model:
    """
    Implementation of the simplified photon propagation model.
    """
    def __init__(self, ice, detector):
        """
        Initializes the model by building the computational graph.

        Parameters
        ----------
        detector: Detector class object, which holds information on the DOMs
        ice: Ice class object, which holds the scattering and absorption
             coefficients
        """
        # set attributes
        self._ice = ice
        self._detector = detector

        # define the computational graph
        self._r0 = tf.placeholder(settings.FLOAT_PRECISION, shape=(None, 3))
        self._v0 = tf.placeholder(settings.FLOAT_PRECISION, shape=(None, 3))
        self.final_positions = self._r0

    def init_cascade(self, x, y, z, n_photons=10000):
        """
        Initializes a cascade at position (x, y, z). All photons start at
        exactly this initial position. The initial directions a sampled
        uniformly.

        Parameters
        ----------
        x: x coordinate of the cascade
        y: y coordinate of the cascade
        z: z coordinate of the cascade
        n_photons: number of photons to initialize
        """
        self.r0 = np.zeros(shape=((n_photons, 3)), dtype=np.float)
        self.v0 = np.zeros(shape=((n_photons, 3)), dtype=np.float)

        self.r0[:, 0] = x
        self.r0[:, 1] = y
        self.r0[:, 2] = z

        thetas = np.random.uniform(0, np.pi, size=n_photons)
        phis = np.random.uniform(0, 2*np.pi, size=n_photons)
        sinTs = np.sin(thetas)

        self.v0[:, 0] = sinTs*np.cos(phis)
        self.v0[:, 1] = sinTs*np.sin(phis)
        self.v0[:, 2] = np.cos(thetas)

    @property
    def feed_dict(self):
        """
        Returns the feed_dict for the TensorFlow session.
        """
        if not (hasattr(self, 'r0') and hasattr(self, 'v0')):
            raise NoPhotonsInitializedException("Photons have to be "
                                                "initialized first!")
        return {self._r0: self.r0, self._v0: self.v0}


class NoPhotonsInitializedException(Exception):
    pass
