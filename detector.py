from __future__ import division
import numpy as np


class Detector:
    """
    Holds information on the DOMs. Currently the DOMs are cubes not spheres.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, l_x=1000, l_y=1000, l_z=1000, dom_radius=0.124,
                 doms_per_string=60, nx_strings=10, ny_strings=10):
        """
        Creates the DOM-list.

        Parameters
        ----------
        l_x : float or int, length in m
            Detector length in x-direction.
        l_y : float or int, lenth in m
            Detector length in y-direction.
        l_z : float or int, lenth in m
            Detector length in z-direction.
        dom_radius : float, radius in m
            The radius of the DOMs. For now DOMs are approximated by cubes with
            an edge length of two times the radius.
        doms_per_string : integer
            Number of DOMs per string. Strings go down in z-direction.
        nx_strings : integer
            Number of strins in x-direction.
        ny_strings : integer
            Number of strins in y-direction.
        """
        self._dom_radius = dom_radius
        dx = l_x/nx_strings
        dy = l_y/ny_strings
        dz = l_z/doms_per_string
        self.doms = np.array([[x*dx, y*dy, z*dz] for x in range(nx_strings) for
                              y in range(ny_strings) for z in
                              range(doms_per_string)])

    # -------------------------- TF Graph Building ----------------------------
    def tf_check_for_hits(self, r1, r2):
        """
        Builds the subgraph to check if the line between r1 and r2 hits a DOM.
        If a hit occurs it is saved in the hitlist.

        Parameters
        ----------
        r1 : TF tensor, 3d vector
            Starting point of the line.
        r2 : TF tensor, 3d vector
            End point of the line.

        Returns
        -------
        """
        # TODO: Implement box intersection check
