from __future__ import division
import tensorflow as tf
import numpy as np

from ice import Ice
from detector import Detector
from model import Model
from logger import Logger
import settings


if __name__ == '__main__':
    # ---------------------------- Initialization -----------------------------
    # set random seeds
    if settings.RANDOM_SEED:
        tf.set_random_seed(settings.RANDOM_SEED)
        np.random.seed(settings.RANDOM_SEED)

    # initialize the ice
    ice_true = Ice()
    ice_true.homogeneous_init(l_abs=settings.L_ABS_TRUE,
                              l_scat=settings.L_SCAT_TRUE)

    ice_pred = Ice(trainable=True)
    ice_pred.homogeneous_init(l_abs=settings.L_ABS_START,
                              l_scat=settings.L_SCAT_START)

    # initialize the detector
    detector = Detector(dom_radius=settings.DOM_RADIUS,
                        nx_strings=settings.NX_STRINGS,
                        ny_strings=settings.NY_STRINGS,
                        doms_per_string=settings.DOMS_PER_STRING,
                        l_x=settings.LENGTH_X,
                        l_y=settings.LENGTH_Y,
                        l_z=settings.LENGTH_Z)

    # initialize the models
    model_true = Model(ice_true, detector)
    model_pred = Model(ice_pred, detector)

    if settings.CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        # don't allocate the entire vRAM initially
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)

    # define hitlists
    hits_true_biased = detector.tf_count_hits(model_true.final_positions)
    hits_pred_soft = detector.tf_soft_count_hits(model_pred.final_positions)
    hits_pred_hard = detector.tf_count_hits(model_pred.final_positions)

    # (reverse) bias correct true hits and take logs
    corrector = tf.stop_gradient(
        hits_pred_soft - hits_pred_hard)*tf.reduce_sum(hits_pred_hard) \
        / tf.reduce_sum(hits_true_biased)

    hits_true = tf.log(hits_true_biased + corrector + 1)
    hits_pred = tf.log(hits_pred_soft + 1)

    # define loss
    loss = tf.reduce_sum(tf.squared_difference(hits_true, hits_pred))

    # initialize the optimizer
    optimizer = tf.train.AdamOptimizer(settings.LEARNING_RATE)

    # define operation to apply the gradients
    minimize = optimizer.minimize(loss)

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
        r_cascades = [[np.random.uniform(high=settings.LENGTH_X),
                       np.random.uniform(high=settings.LENGTH_Y),
                       np.random.uniform(high=settings.LENGTH_Z)]]

        result = session.run([minimize, loss, ice_pred.l_abs, ice_pred.l_scat],
                             feed_dict={model_true.r_cascades: r_cascades,
                                        model_pred.r_cascades: r_cascades})

        # get updated parameters
        logger.log(step, result[1:])

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

    logger.message("Done.")
