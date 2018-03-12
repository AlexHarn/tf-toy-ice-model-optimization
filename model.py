from __future__ import division
import tensorflow as tf


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
        self._ice = ice
        self._detector = detector
