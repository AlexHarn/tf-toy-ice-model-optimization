from __future__ import division
import numpy as np
import tensorflow as tf
import settings


class Detector:
    """
    Holds information on the DOMs.
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
            The radius of the DOMs.
        doms_per_string : integer
            Number of DOMs per string. Strings go down in z-direction.
        nx_strings : integer
            Number of strins in x-direction.
        ny_strings : integer
            Number of strins in y-direction.
        """
        self._dom_radius = dom_radius
        dx = l_x/(nx_strings - 1)
        dy = l_y/(ny_strings - 1)
        dz = l_z/(doms_per_string - 1)
        self.doms = np.array([[x*dx, y*dy, z*dz] for x in range(nx_strings) for
                              y in range(ny_strings) for z in
                              range(doms_per_string)])

    # -------------------------- TF Graph Building ----------------------------
    def tf_check_for_hits(self, r1, r2):
        """
        Builds the subgraph to check if the lines between r1 and r2 hit DOMs.
        If a hit occurs the according scale factor < 1 is returned.

        Parameters
        ----------
        r1 : TF tensor, shape(?, 3)
            Starting points of the lines.
        r2 : TF tensor, shape(?, 3)
            End points of the lines.

        Returns
        -------
        Scale factors t so that r += d*v*t are the points of the hits. Or one
        where no hit occurs.
        """
        # TODO: Maybe find a better way to do this...
        x2x1diff = tf.tile([r2 - r1], [len(self.doms), 1, 1])
        x2x1diff = tf.transpose(x2x1diff, perm=[1, 0, 2])
        x2x1diffnorm = tf.norm(x2x1diff, axis=-1)

        x1 = tf.tile([r1], [len(self.doms), 1, 1])
        x1 = tf.transpose(x1, perm=[1, 0, 2])
        x1x0diff = x1 - self.doms

        ds = tf.norm(tf.cross(x2x1diff, x1x0diff), axis=-1)/x2x1diffnorm
        ts = -tf.einsum('ijk,ijk->ij', x1x0diff,
                        x2x1diff)/tf.square(x2x1diffnorm)

        t = -tf.reduce_max(-tf.where(
            tf.logical_and(tf.logical_and(ds < self._dom_radius, ts >= 0.), ts
                           <= 1.), ts,
            tf.ones([tf.shape(r1)[0], len(self.doms)],
                    dtype=settings.FLOAT_PRECISION)),
                           axis=-1)
        return t

    def tf_soft_count_hits(self, final_positions):
        """
        Counts the hits for each DOM using softsign for differentiability.

        Parameters
        ----------
        final_positions : tf tensor of shape (?, 3)
            The final positions after every photon has been absorbed or hit a
            DOM.

        Returns
        -------
        The DOM hitlist which contains a tensorflow variable for each DOM as a
        measure for the number of photons that ended up inside the DOM.
        """
        # calculate distances of every photon to every DOM
        hitlist = []
        for dom in self.doms:
            d = tf.norm(final_positions - dom, axis=1)
            hitlist.append(
                tf.reduce_sum(-tf.nn.softsign((d - self._dom_radius)*10000) +
                              1)/2)
        return hitlist

    def tf_count_hits(self, final_positions):
        """
        Counts the hits for each DOM exactly, which is not differentiable.

        Parameters
        ----------
        final_positions : tf tensor of shape (?, 3)
            The final positions after every photon has been absorbed or hit a
            DOM.

        Returns
        -------
        The DOM hitlist which contains the number of hits for each DOM as a
        tensorflow variable.
        """
        # calculate distances of every photon to every DOM
        hitlist = []
        for dom in self.doms:
            d = tf.norm(final_positions - dom, axis=1)
            hitlist.append(tf.reduce_sum(tf.where(d <= self._dom_radius,
                                                  tf.ones(tf.shape(d)[0]),
                                                  tf.zeros(tf.shape(d)[0]))))
        return hitlist
