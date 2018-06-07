from __future__ import division
import tensorflow as tf
import settings


class Ice:
    # ---------------------------- Initialization -----------------------------
    def __init__(self, trainable=False, placeholders=False, l_abs=[100],
                 l_scat=[25]):
        """
        Holds all absorption and scattering coefficients. For now the ice is
        only divided into layers in z-direction of equal length.

        Parameters
        ----------
        trainable : boolean
            Decides if the ice parameters are supposed to be trainable TF
            variables or simply constants. Overwrites the placeholders flag.

        placeholders : boolean
            If True the parameters will be TF placeholders.

        l_abs : list of floats
            A list that contains the absorption length in meter for each layer.

        l_abs : list of floats
            A list that contains the scattering length in meter for each layer.
        """
        self._trainable = trainable
        if not trainable:
            self._placeholders = placeholders
        else:
            self._placeholders = False

        assert len(l_abs) == len(l_scat)
        self._n_layers = len(l_abs)

        self.l_abs = []
        self.l_scat = []
        self._abs_pdf = []
        self._scat_pdf = []

        for i in range(self._n_layers):
            if self._trainable:
                self.l_abs.append(tf.Variable(l_abs[i],
                                              dtype=settings.FLOAT_PRECISION,
                                              name='l_abs_pred'))
                self.l_scat.append(tf.Variable(l_scat[i],
                                               dtype=settings.FLOAT_PRECISION,
                                               name='l_scat_pred'))
            elif self._placeholders:
                self.l_abs.append(tf.placeholder(
                    dtype=settings.FLOAT_PRECISION, shape=()))
                self.l_scat.append(tf.placeholder(
                    dtype=settings.FLOAT_PRECISION, shape=()))
            else:
                self.l_abs.append(tf.constant(l_abs[i],
                                              dtype=settings.FLOAT_PRECISION))
                self.l_scat.append(tf.constant(l_scat[i],
                                               dtype=settings.FLOAT_PRECISION))

            self._abs_pdf.append(
                tf.distributions.Exponential(1/self.l_abs[i]))
            self._scat_pdf.append(
                tf.distributions.Exponential(1/self.l_scat[i]))

    # -------------------------- TF Graph Building ----------------------------
    def tf_sample_absorption(self, r):
        """
        Samples absorption lengths.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            Photon locations.

        Returns
        -------
        Tensor for the sampled absorption lengths of shape(?).
        """
        sampled = self._abs_pdf[0].sample(tf.shape(r)[0])
        for layer in range(1, self._n_layers):
            sampled = tf.where(r[:, 2] <
                               layer*settings.LENGTH_Z/self._n_layers, sampled,
                               self._abs_pdf[layer].sample(tf.shape(r)[0]))
        return sampled

    def tf_sample_scatter(self, r):
        """
        Samples scattering lengths.

        Parameters
        ----------
        r : TF tensor, shape(?, 3)
            Photon locations.

        Returns
        -------
        Tensor for the sampled scattering lengths of shape(?).
        """
        sampled = self._scat_pdf[0].sample(tf.shape(r)[0])
        for layer in range(1, self._n_layers):
            sampled = tf.where(r[:, 2] <
                               layer*settings.LENGTH_Z/self._n_layers, sampled,
                               self._scat_pdf[layer].sample(tf.shape(r)[0]))
        return sampled
