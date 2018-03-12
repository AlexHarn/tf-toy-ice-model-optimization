from __future__ import division
import tensorflow as tf
import numpy as np


class Detector:
    """
    Holds information on the DOMs. Currently the DOMs are cubes not spheres.
    """
    def __init__(self, l_x=1000, l_y=1000, l_z=1000, dom_radius=0.124,
                 doms_per_string=60, nx_strings=10, ny_strings=10):
        """
        Creates the DOM-list.
        """
        self._dom_radius = dom_radius
        dx = l_x/nx_strings
        dy = l_y/ny_strings
        dz = l_z/doms_per_string
        self.doms = np.array([[x*dx, y*dy, z*dz] for x in range(nx_strings) for
                              y in range(ny_strings) for z in
                              range(doms_per_string)])

    def tf_check_for_hits(self, r1, r2):
        """
        Builds the subgraph to check if the line between r1 and r2 hits a DOM.
        If a hit occurs it is saved in the hitlist.

        Parameters
        ----------
        r1: tf tensor. Starting point of the line
        r2: tf tensor. End point of the line

        Returns
        -------
        tensor of rank 1 with as many entries as there are DOMs. Each entry
        corresponds to the DOM at the same position in the DOM-list. The
        entries are 0 if the DOM was not hit and 1 if the DOM was hit. Very
        sparse.
        """
        # TODO: Implement box intersection check
        return tf.constant(np.zeros(len(self.doms), dtype=np.int8))
