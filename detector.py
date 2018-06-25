from __future__ import division
import numpy as np
import tensorflow as tf
import settings


class Detector:
    """
    Defines the detector and provides the graph building methods for hit
    detection and hit counting.

    Parameters
    ----------
    l_x : float or int, length in m
        Detector length in x-direction.
    l_y : float or int, length in m
        Detector length in y-direction.
    l_z : float or int, length in m
        Detector length in z-direction.
    dom_radius : float, radius in m
        The radius of the DOMs.
    doms_per_string : integer
        Number of DOMs per string. Strings go down in z-direction.
    nx_strings : integer
        Number of strings in x-direction.
    ny_strings : integer
        Number of strings in y-direction.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, l_x=1000, l_y=1000, l_z=1000, dom_radius=0.124,
                 doms_per_string=60, nx_strings=10, ny_strings=10):
        self._dom_radius = dom_radius
        dx = l_x/(nx_strings - 1)
        dy = l_y/(ny_strings - 1)
        dz = l_z/(doms_per_string - 1)
        self.doms = np.array([[x*dx, y*dy, z*dz] for x in range(nx_strings) for
                              y in range(ny_strings) for z in
                              range(doms_per_string)])

        self.tf_doms = tf.constant(np.expand_dims(self.doms, axis=0),
                                   dtype=settings.FLOAT_PRECISION)

        self._l_x = l_x
        self._l_y = l_y
        self._l_z = l_z

    # -------------------------- TF Graph Building ----------------------------
    def tf_check_for_hits(self, r, d, v):
        """
        Builds the subgraph to check if the lines between r and r + d*v  hit
        DOMs.  If a hit occurs the according scale factor < 1 is returned.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            Starting points of the lines.
        d : TF tensor, shape(?)
            Lengths of the lines.
        v : TF tensor, shape(?, 3)
            Normalized direction vectors of the lines.

        Returns
        -------
        Scale factors t so that r + d*v*t are the points of the hits. Or one
        where no hit occurs.
        """
        v_exp = tf.expand_dims(v, axis=1)
        r_exp = tf.expand_dims(r, axis=1)
        d_exp = tf.expand_dims(d, axis=1)

        # define plane with normal vector v and anchor point r
        # define second plane with normal vector v and anchor point as dom
        # ts are the distances between these planes
        diff_doms_r = self.tf_doms - r_exp
        ts_exp = tf.reduce_sum(v_exp * diff_doms_r, axis=2, keep_dims=True)

        # closest approach point is
        # r + t*v
        # calculate norm of vector from closest
        # approach point to dom
        ds_exp = tf.norm(-diff_doms_r + v_exp*ts_exp, axis=2, keep_dims=True)

        # remove last dimension again
        ds = tf.squeeze(ds_exp, axis=2)
        ts = tf.squeeze(ts_exp, axis=2)/d_exp

        # filter closest hit from valid hits, 1 if no hit occurs
        t = -tf.reduce_max(-tf.where(
            tf.logical_and(tf.logical_and(ds < self._dom_radius, ts >= 0.), ts
                           <= 1.), ts, tf.ones_like(ts)), axis=-1)
        return tf.stop_gradient(t)

    def tf_soft_count_hits(self, final_positions):
        """
        Counts the hits for each DOM using softsign for differentiability.

        Parameters
        ----------
        final_positions : TF tensor of shape (?, 3)
            The final positions after every photon has been absorbed or hit a
            DOM.

        Returns
        -------
        The DOM hitlist which contains a TF variable for each DOM as a measure
        for the number of photons that ended up inside the DOM.
        """
        final_positions_exp = tf.expand_dims(final_positions, axis=1)

        # calculate distances of every photon to every DOM
        ds = tf.norm(final_positions_exp - self.tf_doms, axis=2)
        hitlist = tf.reduce_sum(-tf.nn.softsign((ds - self._dom_radius))
                                + 1, axis=0) / 2

        return hitlist

    def tf_count_hits(self, final_positions):
        """
        Counts the hits for each DOM exactly, which is not differentiable.

        Parameters
        ----------
        final_positions : TF tensor of shape (?, 3)
            The final positions after every photon has been absorbed or hit a
            DOM.

        Returns
        -------
        The DOM hitlist which contains the number of hits for each DOM as a TF
        variable.
        """
        final_positions_exp = tf.expand_dims(final_positions, axis=1)

        # calculate distances of every photon to every DOM
        ds = tf.norm(final_positions_exp - self.tf_doms, axis=2)
        hitlist = tf.reduce_sum(tf.where(ds <= self._dom_radius,
                                         tf.ones_like(ds),
                                         tf.zeros_like(ds)),
                                axis=0)

        return tf.stop_gradient(hitlist)

    def tf_expected_hits(self, final_positions, traveled_distances, ice):
        """
        Calculates the expected hits for each DOM. Only homogeneous right now.

        Parameters
        ----------
        final_positions : TF tensor, shape(?, 3)
            Final photon positions.
        traveled_distances : TF tensor, shape(?)
            The traveled distance of each photon.
        ice : Ice class object
            The ice which specifies absorption.

        Returns
        -------
        The expected hits for each DOM. TF Tensor of shape (DOMS,).
        """
        final_positions_exp = tf.expand_dims(final_positions, axis=1)

        # calculate distances of every photon to every DOM
        ds = tf.norm(final_positions_exp - self.tf_doms, axis=2)
        hit_mask = ds <= self._dom_radius

        # calculate hit probability p for each photon, which is the 1 - p_abs
        # where p_abs is the probability for the photon to be absorbed at a
        # distance smaller than the traveled distance before it hit the DOM
        p = tf.exp(-tf.reduce_sum(
            1./tf.expand_dims(ice.l_abs, 0)*traveled_distances,
            axis=1))
        p_exp = tf.tile(tf.expand_dims(p, axis=1),
                        [1,
                         settings.NX_STRINGS
                         * settings.NY_STRINGS
                         * settings.DOMS_PER_STRING])

        hitlist = tf.reduce_sum(tf.where(hit_mask,
                                         p_exp,
                                         tf.zeros_like(p_exp)),
                                axis=0)
        return hitlist

    def tf_sample_hits(self, final_positions, traveled_distances, ice):
        """
        Calculates the probability for each photon to reach the DOM and than
        samples a uniform number to decide whether or not to count the photon.

        Parameters
        ----------
        final_positions : TF tensor, shape(?, 3)
            Final photon positions.
        traveled_distances : TF tensor, shape(?)
            The traveled distance of each photon.
        ice : Ice class object
            The ice which specifies absorption.

        Returns
        -------
        The sampled hits for each DOM. TF Tensor of shape (DOMS,).
        """
        final_positions_exp = tf.expand_dims(final_positions, axis=1)

        # calculate distances of every photon to every DOM
        ds = tf.norm(final_positions_exp - self.tf_doms, axis=2)
        hit_mask = ds <= self._dom_radius

        # calculate hit probability p for each photon, which is the 1 - p_abs
        # where p_abs is the probability for the photon to be absorbed at a
        # distance smaller than the traveled distance before it hit the DOM
        p = tf.exp(-tf.reduce_sum(
            1./tf.expand_dims(ice.l_abs, 0)*traveled_distances,
            axis=1))
        z = tf.distributions.Uniform().sample(
            settings.BATCHES_PER_STEP*settings.BATCH_SIZE)
        hits = tf.where(z < p, tf.ones_like(p), tf.zeros_like(p))

        hits_exp = tf.tile(tf.expand_dims(hits, axis=1),
                           [1,
                            settings.NX_STRINGS
                            * settings.NY_STRINGS
                            * settings.DOMS_PER_STRING])

        hitlist = tf.reduce_sum(tf.where(hit_mask,
                                         hits_exp,
                                         tf.zeros_like(hits_exp)),
                                axis=0)
        return hitlist
