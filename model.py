from __future__ import division
import tensorflow as tf
import numpy as np
import settings


class Model:
    """
    Implementation of the simplified photon propagation model.

    Parameters
    ----------
    detector : Detector class object
        The detector object, which holds information on the DOMs.
    ice : Ice class object
        The ice object, Which holds the scattering and absorption
        coefficients.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, ice, detector):
        # set attributes
        self._ice = ice
        self._detector = detector

        # start defining the computational graph
        self.r_cascades = tf.placeholder(settings.FLOAT_PRECISION,
                                         shape=(settings.CASCADES_PER_STEP,
                                                3))

        # initialize uniform pdf
        self._uni_pdf = tf.distributions.Uniform()

        # initialize cascades
        self.tf_init_cascades()

        # propagate
        self.tf_propagate()

    def tf_init_cascades(self):
        """
        Builds the subgraph to initialize cascades at positions
        self.r_cascades.  All photons start at exactly these initial positions.
        The initial directions are sampled uniformly. For now all cascades
        contain the same number of photons n_photons/shape(self.r_cascades)
        """
        self._r0 = tf.tile(
            self.r_cascades,
            [int(settings.BATCH_SIZE/settings.CASCADES_PER_STEP), 1])

        thetas = self._uni_pdf.sample(settings.BATCH_SIZE)*np.pi
        phis = self._uni_pdf.sample(settings.BATCH_SIZE)*2*np.pi
        sinTs = tf.sin(thetas)

        self._v0 = tf.transpose([sinTs*tf.cos(phis), sinTs*tf.sin(phis),
                                 tf.cos(thetas)])

    # ------------------------------ Simulation -------------------------------
    def tf_sample_normal_vectors(self, v):
        """
        Samples normalized random 3d vectors with uniformly distributed
        direction which is perpendicular to v.

        Parameters
        ----------
        v : TF tensor, shape(?, 3)
            The vectors for which random normal vectors are desired.

        Returns
        -------
        The random normal vector tensor of shape(?, 3).
        """
        # sample random vectors uniformly in all directions
        thetas = self._uni_pdf.sample(tf.shape(v)[0])*np.pi
        phis = self._uni_pdf.sample(tf.shape(v)[0])*2*np.pi
        sinTs = tf.sin(thetas)

        # construct normal vectors by computing the cross products
        n = tf.cross(tf.transpose([sinTs*tf.cos(phis), sinTs*tf.sin(phis),
                                   tf.cos(thetas)]), v)
        return n/tf.norm(n, axis=-1, keep_dims=True)

    def tf_scatter(self, v):
        """
        Scatter the given direction tensor v.

        Parameters
        ----------
        v : TF tensor, shape(?, 3)
            Direction vectors of the photons which are being scattered

        Returns
        -------
        The scattered direction tensor of shape(?, 3)
        """
        uni_pdf = tf.distributions.Uniform()
        u = uni_pdf.sample(settings.BATCH_SIZE) - 0.1
        u = tf.expand_dims(u, axis=1)
        u = tf.tile(u, [1, 3])
        return tf.where(u > 0, tf.ones_like(u)*v, -tf.ones_like(u)*v)

    def tf_propagate(self):
        """
        Propagates the photons until they are absorbed or hit a DOM.
        """
        def body(d_abs, r, v, t):
            # sample distances until next scattering
            d_scat = self._ice.tf_sample_scatter(r)

            # make sure we stop the propagation after d_abs
            d_abs = tf.where(d_abs > 0., d_abs, tf.zeros_like(d_abs))

            # if the distance is longer than the remaining distance until
            # absorption only propagate to absorption
            d = tf.where(d_scat < d_abs, d_scat, d_abs)

            # check for hits and stop inside the DOM if hit
            rel_d_til_hit = self._detector.tf_check_for_hits(r, d, v)
            d_abs = tf.where(rel_d_til_hit < 1., tf.zeros_like(d), d_abs - d)

            # propagate
            r += tf.expand_dims(d*rel_d_til_hit, axis=-1)*v

            # log traveltimes (or distance, just differ by constant speed)
            t += d*rel_d_til_hit

            # stop propagating if the photon is outside the cutoff radius
            if settings.CUTOFF_RADIUS:
                d_abs = tf.where(tf.norm(r - np.array([self._detector._l_x/2,
                                                       self._detector._l_y/2,
                                                       self._detector._l_z/2]),
                                         axis=-1) < settings.CUTOFF_RADIUS
                                 * np.linalg.norm([self._detector._l_x,
                                                   self._detector._l_y,
                                                   self._detector._l_z])/2,
                                 d_abs, tf.zeros_like(d_abs))

            # scatter photons which have not been stopped yet
            v = tf.where(d_abs > 0., self.tf_scatter(v), v)

            return [d_abs, r, v, t]

        results = tf.while_loop(
            lambda d_abs, r, v, t: tf.less(0., tf.reduce_max(d_abs)),
            lambda d_abs, r, v, t: body(d_abs, r, v, t),
            [self._ice.tf_sample_absorption(self._r0), self._r0, self._v0,
             tf.zeros([tf.shape(self._r0)[0]],
                      dtype=settings.FLOAT_PRECISION)],
            parallel_iterations=1)

        self.final_positions = results[1]
        self.arrival_times = results[3]
