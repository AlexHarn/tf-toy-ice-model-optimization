from __future__ import division
import tensorflow as tf
import numpy as np

import settings
from logger import Logger


def tf_propagate(r0, v0, l_abs, l_scat, max_z, tag):
    """
    Propagates the photons until they are absorbed.

    Parameters
    ----------
    r0: TF tensor of shape(?)
        Initial positions

    v0: TF tensor of shape(?)
        Initial directions
    """
    # initialize distributions
    abs_pdf = tf.distributions.Exponential(1/l_abs)
    scat_pdf = tf.distributions.Exponential(1/l_scat)
    uni_pdf = tf.distributions.Uniform()

    t0 = tf.zeros_like(r0)
    i0 = tf.constant(0)

    def body(d_abs, r, v, t, i):
        # count iterations
        i += 1

        # sample distances until next scattering
        d_scat = scat_pdf.sample(settings.BATCH_SIZE)

        # make sure we stop the propagation after d_abs
        d_abs = tf.where(d_abs > 0., d_abs, tf.zeros_like(d_abs))

        # if the distance is longer than the remaining distance until
        # absorption only propagate to absorption
        d = tf.where(d_scat < d_abs, d_scat, d_abs)

        # maximally go to boundary
        d = tf.where(tf.abs(r + d*v) > max_z,
                     d - (tf.abs(r + d*v) - max_z) + 1, 
                     d)

        # propagate
        # r = tf.Print(r, [r],'[{}]r'.format(tag), summarize=5)
        r += d*v
        d_abs -= d

        # log distance traveled
        # r = tf.Print(r, [t],'[{}]t'.format(tag), summarize=5)
        t += d

        # stop photons which have reached a minimal distance to either side:
        d_abs = tf.where(tf.abs(r) > max_z, tf.zeros_like(d_abs), d_abs)

        # scatter photons
        v = uni_pdf.sample(settings.BATCH_SIZE) - 0.3
        v = tf.where(v > 0, tf.ones_like(v), -tf.ones_like(v))

        return [d_abs, r, v, t, i]

    d_abs, r, v, t, i =  tf.while_loop(
        lambda d_abs, r, v, t, i: tf.less(0., tf.reduce_max(d_abs)),
        # lambda d_abs, r, v, t, i: tf.less(i, 4),
        lambda d_abs, r, v, t, i: body(d_abs, r, v, t, i),
        [abs_pdf.sample(settings.BATCH_SIZE), r0, v0, t0, i0],
        parallel_iterations=1)

    return t


def tf_soft_count_hits(final_positions, tf_doms):
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
    ds = tf.abs(final_positions_exp - tf_doms)
    soft = -tf.nn.softsign((ds - settings.DOM_RADIUS)) + 1
    hitlist = tf.reduce_sum(soft, axis=0) / 2

    return hitlist


def tf_count_hits(final_positions, tf_doms):
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
    ds = tf.abs(final_positions_exp - tf_doms)
    hitlist = tf.reduce_sum(tf.where(ds <= settings.DOM_RADIUS,
                                     tf.ones_like(ds),
                                     tf.zeros_like(ds)),
                            axis=0)

    return tf.stop_gradient(hitlist)


if __name__ == '__main__':
    # ---------------------------- Initialization -----------------------------
    # initialize the ice
    l_abs_true = tf.constant(settings.L_ABS_TRUE,
                             dtype=settings.FLOAT_PRECISION)
    l_scat_true = tf.constant(settings.L_SCAT_TRUE,
                              dtype=settings.FLOAT_PRECISION)

    l_abs_pred = tf.constant(settings.L_ABS_START,
                             dtype=settings.FLOAT_PRECISION)
    l_scat_pred = tf.Variable(settings.L_SCAT_START,
                              dtype=settings.FLOAT_PRECISION)

    # initialize the detector
    dz = settings.LENGTH_Z/(settings.DOMS_PER_STRING - 1)
    doms = np.linspace(0, settings.LENGTH_Z, dz)

    tf_doms = tf.constant(np.expand_dims(doms, axis=0),
                          dtype=settings.FLOAT_PRECISION)

    # initialize uniform pdf
    uni_pdf = tf.distributions.Uniform()

    # initialize the starting positions and directions
    tf_r_cascade = tf.placeholder(settings.FLOAT_PRECISION, shape=())

    r0_true = tf.tile([tf_r_cascade], [settings.BATCH_SIZE])
    r0_pred = tf.tile([tf_r_cascade], [settings.BATCH_SIZE])

    v0_true = uni_pdf.sample(settings.BATCH_SIZE) - 0.5
    v0_true = tf.where(v0_true > 0, tf.ones_like(v0_true),
                       -tf.ones_like(v0_true))
    v0_pred = uni_pdf.sample(settings.BATCH_SIZE) - 0.5
    v0_pred = tf.where(v0_pred > 0, tf.ones_like(v0_pred),
                       -tf.ones_like(v0_pred))

    # # start all with same directions
    # v0_true = tf.ones_like(v0_true)
    # v0_pred = tf.ones_like(v0_pred)

    # propagate
    time_traveled_true = tf_propagate(r0_true, v0_true, l_abs_true,
                                        l_scat_true, settings.BOUNDARY, 'real')
    time_traveled_pred = tf_propagate(r0_pred, v0_pred, l_abs_pred,
                                        l_scat_pred, settings.BOUNDARY, 'fake')

    sum_time_traveled_true = tf.reduce_sum(time_traveled_true)
    sum_time_traveled_pred = tf.reduce_sum(time_traveled_pred)

    # define loss
    loss = tf.reduce_sum(tf.squared_difference(sum_time_traveled_true,
                                                sum_time_traveled_pred))

    # initialize the optimizer
    optimizer = tf.train.AdamOptimizer(settings.LEARNING_RATE)

    # define operation to apply the gradients
    minimize = optimizer.minimize(loss)

    # don't allocate the entire vRAM initially
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)

    # initialize all variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    # initialize the logger
    logger = Logger(logdir='./log/', overwrite=True)
    logger.register_variables(['loss', 'l_abs_pred', 'l_scat_pred'],
                              print_all=True)

    logger.message("Starting...")
    for step in range(1, settings.MAX_STEPS + 1):
        # sample cascade position for this step
        #r_cascade = np.random.uniform(high=settings.LENGTH_X)
        r_cascade =0.

        result = session.run([minimize, loss, l_abs_pred, l_scat_pred, 
                              time_traveled_pred],
                             feed_dict={tf_r_cascade: r_cascade})

        # get updated parameters
        logger.log(step, result[1:-1])
        #print('t:',result[-1])

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

    logger.message("Done.")