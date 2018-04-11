from __future__ import division
import tensorflow as tf
import tfquaternion as tfq
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
    def tf_sample_normal_vectors(self, r):
        """
        Samples normalized random 3d vectors with uniformly distributed
        direction which is perpendicular to r.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            The vectors for which random normal vectors are desired.

        Returns
        -------
        The random normal vector tensor of shape(?, 3).
        """
        # sample random vectors uniformly in all directions
        thetas = self._uni_pdf.sample(tf.shape(r)[0])*np.pi
        phis = self._uni_pdf.sample(tf.shape(r)[0])*2*np.pi
        sinTs = tf.sin(thetas)

        # construct normal vectors by computing the cross products
        v = tf.cross(tf.transpose([sinTs*tf.cos(phis), sinTs*tf.sin(phis),
                                   tf.cos(thetas)]), r)
        return v/tf.norm(v, axis=-1, keep_dims=True)

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
        # sample cos(theta)
        cosTs = 2*self._uni_pdf.sample(tf.shape(v)[0])**(1/19) - 1
        cosT2s = tf.sqrt((cosTs + 1)/2)
        sinT2s = tf.sqrt((1 - cosTs)/2)

        ns = tf.transpose(self.tf_sample_normal_vectors(v) *
                          tf.expand_dims(sinT2s, axis=-1))
        # ignore the fact that n could be parallel to v, what's the probability
        # of that happening?

        q = tfq.Quaternion(tf.transpose([cosT2s, ns[0], ns[1], ns[2]]))
        return tfq.rotate_vector_by_quaternion(q, v)

    def tf_propagate(self):
        """
        Propagates the photons until they are absorbed or hit a DOM.
        """
        def body(d_abs, r, v):
            # sample distances until next scattering
            d_scat = self._ice.tf_sample_scatter(r)

            # make sure we stop the propagation after d_abs
            d_abs = tf.where(d_abs > 0., d_abs,
                             tf.zeros(tf.shape(d_abs),
                                      dtype=settings.FLOAT_PRECISION))

            # if the distance is longer than the remaining distance until
            # absorption only propagate to absorption
            d = tf.where(d_scat < d_abs, d_scat, d_abs)

            # check for hits and stop inside the DOM if hit
            t = self._detector.tf_check_for_hits(r, d, v)
            d_abs = tf.where(t < 1., tf.zeros(tf.shape(d),
                                              dtype=settings.FLOAT_PRECISION),
                             d_abs - d)

            # propagate
            r = r + tf.expand_dims(d*t, axis=-1)*v

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

            # scatter
            v = self.tf_scatter(v)

            return [d_abs, r, v]

        self.final_positions = tf.while_loop(
            lambda d_abs, r, v: tf.less(0., tf.reduce_max(d_abs)),
            lambda d_abs, r, v: body(d_abs, r, v),
            [self._ice.tf_sample_absorption(self._r0), self._r0, self._v0])[1]
