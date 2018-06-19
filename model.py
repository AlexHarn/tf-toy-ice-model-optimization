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
        self.r_cascades. All photons start at exactly these initial positions.
        The initial directions are sampled uniformly. For now all cascades
        contain the same number of photons n_photons/shape(self.r_cascades).
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
            Direction vectors of the photons which are being scattered.

        Returns
        -------
        The scattered direction tensor of shape(?, 3).
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
        def body(stopped, r, v, t, d_layer):
            """
            Body of the propagation loop.

            Parameters
            ----------
            stopped : TF tensor, shape(?)
                1 for photons which hit a DOM or reached the cutoff, 0 for
                photons which are still going.
            r : TF tensor, shape(?, 3)
                photon positions.
            v : TF tensor, shape(?, 3)
                normalized photon direcitons.
            t : TF tensor, shape(?)
                Travel time/distance of each photon.
            d_layer : TF tensor, shape(?, N_layer)
                Traveled distance of each photon in each layer.

            Returns
            -------
            (stopped, r, v, t) for next iteration.
            """
            # sample distances until next scattering
            d_scat = self._ice.tf_sample_scatter()

            # check for hits and stop inside the DOM if hit
            rel_d_til_hit = tf.where(stopped < 0.5,
                                     self._detector.tf_check_for_hits(r,
                                                                      d_scat,
                                                                      v),
                                     tf.zeros_like(d_scat))

            # propagate
            r_next = r + tf.expand_dims(d_scat*rel_d_til_hit, axis=-1)*v

            # log traveltimes (or distance, just differ by constant speed)
            d = d_scat*rel_d_til_hit
            t += d
            d_layer += tf.where(stopped < 0.5,
                                self._ice.tf_get_layer_distance(r, r_next, v,
                                                                d),
                                tf.zeros_like(d_layer))

            r = r_next

            stopped = tf.where(rel_d_til_hit < 1., tf.ones_like(stopped),
                               tf.zeros_like(stopped))

            # stop propagating if the photon is outside the cutoff radius
            if settings.CUTOFF_RADIUS:
                stopped = \
                    tf.where(tf.norm(r - np.array([self._detector._l_x/2,
                                                   self._detector._l_y/2,
                                                   self._detector._l_z/2]),
                                     axis=-1) < settings.CUTOFF_RADIUS *
                             np.linalg.norm([self._detector._l_x,
                                             self._detector._l_y,
                                             self._detector._l_z])/2, stopped,
                             tf.ones_like(stopped))

            # stop propagating if the photon reached the cutoff travel distance
            stopped = tf.where(t < settings.CUTOFF_DISTANCE, stopped,
                               tf.ones_like(stopped))

            # scatter photons which have not been stopped yet
            v = tf.where(stopped < 0.5, self.tf_scatter(v), v)

            return [stopped, r, v, t, d_layer]

        results = tf.while_loop(
            lambda stopped, r, v, t, d_layer:
                tf.greater(0.5, tf.reduce_min(stopped)),
            lambda stopped, r, v, t, d_layer:
                body(stopped, r, v, t, d_layer),
                [tf.zeros(settings.BATCH_SIZE), self._r0, self._v0,
                 tf.zeros([tf.shape(self._r0)[0]],
                          dtype=settings.FLOAT_PRECISION),
                 tf.zeros([settings.BATCH_SIZE, len(settings.L_ABS_START)],
                          dtype=settings.FLOAT_PRECISION)],
            parallel_iterations=1)

        self.final_positions = results[1]
        self.traveled_distances = results[3]
        self.traveled_layer_distance = results[4]
