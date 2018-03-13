from __future__ import division
import tensorflow as tf
import tfquaternion as tfq
import numpy as np
import settings


class Model:
    """
    Implementation of the simplified photon propagation model.
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, ice, detector):
        """
        Initializes the model by building the computational graph.

        Parameters
        ----------
        detector : Detector class object
            The detector object, which holds information on the DOMs.
        ice : Ice class object
            The ice object, Which holds the scattering and absorption
            coefficients.
        """
        # set attributes
        self._ice = ice
        self._detector = detector

        # define the computational graph
        self._r0 = tf.placeholder(settings.FLOAT_PRECISION, shape=(None, 3))
        self._v0 = tf.placeholder(settings.FLOAT_PRECISION, shape=(None, 3))

        # init uniform pdf
        self._uni_pdf = tf.distributions.Uniform()

        # propagate
        self.final_positions = tf.map_fn(lambda x: self.tf_propagate(x[0],
                                                                     x[1]),
                                         tf.stack([self._r0, self._v0],
                                                  axis=1))

    def init_cascade(self, x, y, z, n_photons=10000):
        """
        Initializes a cascade at position (x, y, z). All photons start at
        exactly this initial position. The initial directions a sampled
        uniformly.

        Parameters
        ----------
        x : float
            x coordinate of the cascade.
        y : float
            y coordinate of the cascade.
        z : float
            z coordinate of the cascade.
        n_photons : integer
            Number of photons to initialize.
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

    # ------------------------------ Simulation -------------------------------
    def tf_sample_normal_vector(self, r):
        """
        Samples a normalized random 3d vector with uniformly distributed
        direction which is perpendicular to r.

        Parameters
        ----------
        r : TF tensor, 3d vector
            The vector for which a random normal vector is desired.

        Returns
        -------
        The random normal vector tensor.
        """
        theta = self._uni_pdf.sample(1)[0]*np.pi
        phi = self._uni_pdf.sample(1)[0]*2*np.pi
        sinT = tf.sin(theta)
        v = tf.cross([sinT*tf.cos(phi), sinT*tf.sin(phi), tf.cos(theta)], r)
        return v/tf.norm(v)

    def tf_scatter(self, r):
        # sample cos(theta)
        cosT = 2*self._uni_pdf.sample(1)[0]**(1/19) - 1
        cosT2 = tf.sqrt((cosT + 1)/2)
        sinT2 = tf.sqrt((1 - cosT)/2)

        v = self.tf_sample_normal_vector(r)*sinT2
        # ignore the fact that v could be parallel to r, what's the probability
        # of that happening?

        q = tfq.Quaternion([cosT2, v[0], v[1], v[2]])

        return tfq.quaternion_to_vector3d(q*tfq.vector3d_to_quaternion(r)/q)

    def tf_propagate(self, r, v):
        """
        Propagates a single photon until it is absorbed.

        Parameters
        ----------
        r : tf tensor, 3d vector
            initial position of the photon
        v : tf tensor, 3d vector
            initial direction of the photon

        Returns
        -------
        Final position tensor of the photon after absorbtion.
        """

        def body(d_abs, r, v):
            # sample distance until next scattering
            d_scat = self._ice.tf_sample_scatter(r)

            # if the distance is longer than the remaining distance until
            # absorbtion only propagate to absorbtion
            # d = tf.cond(d_scat < d_abs, lambda: d_scat, lambda: d_abs)
            d = tf.where(d_scat < d_abs, d_scat, d_abs)

            # propagate
            d_abs -= d
            r += d*v

            # scatter
            v = self.tf_scatter(v)

            return [d_abs, r, v]

        return tf.while_loop(
            lambda d_abs, r, v: tf.less(0., d_abs),
            lambda d_abs, r, v: body(d_abs, r, v),
            [self._ice.tf_sample_absorbtion(r), r, v])[1]

    # ------------------------------ Properties -------------------------------
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
