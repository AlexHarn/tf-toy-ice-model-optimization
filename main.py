from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange

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
    hits_true = detector.tf_count_hits(model_true.final_positions)
    hits_pred = detector.tf_soft_count_hits(model_pred.final_positions)

    # define loss
    loss = tf.reduce_sum(tf.squared_difference(hits_true, hits_pred))

    # init optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=settings.LEARNING_RATE)

    # grab all trainable variables
    trainable_variables = tf.trainable_variables()

    # define variables to save the gradients in each batch
    accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                         trainable=False) for tv in
                             trainable_variables]

    # define operation to reset the accumulated gradients to zero
    reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                       accumulated_gradients]

    # define the gradients
    gradients = optimizer.compute_gradients(loss, trainable_variables)

    # Note: Gradients is a list of tuples containing the gradient and the
    # corresponding variable so gradient[0] is the actual gradient. Also divide
    # the gradients by BATCHES_PER_STEP so the learning rate still refers to
    # steps not batches.

    # define operation to propagate a batch and accumulating the gradients
    propagate_batch = [
        accumulated_gradient.assign_add(gradient[0]/settings.BATCHES_PER_STEP)
        for accumulated_gradient, gradient in zip(accumulated_gradients,
                                                  gradients)]

    # define operation to apply the gradients
    apply_gradients = optimizer.apply_gradients([
        (accumulated_gradient, gradient[1]) for accumulated_gradient, gradient
        in zip(accumulated_gradients, gradients)])

    # initialize all variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    # initialize the logger
    logger = Logger()

    print("Starting...")
    for i in range(settings.N_STEPS):
        # sample cascade positions for this step
        r_cascades = [[np.random.uniform(high=settings.LENGTH_X),
                       np.random.uniform(high=settings.LENGTH_Y),
                       np.random.uniform(high=settings.LENGTH_Z)]
                      for i in range(settings.CASCADES_PER_STEP)]

        # propagate in batches
        for j in trange(settings.BATCHES_PER_STEP, leave=False):
            session.run(propagate_batch, feed_dict={model_true.r_cascades:
                                                    r_cascades,
                                                    model_pred.r_cascades:
                                                    r_cascades})

        # apply accumulated gradients
        session.run(apply_gradients)

        # reset accumulated gradients to zero and get updated parameters
        result = session.run([reset_gradients, ice_pred._l_abs,
                              ice_pred._l_scat])
        logger.log(i, *result[1:])

        if i % settings.WRITE_INTERVAL == 0:
            logger.write()
