from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange

from ice import Ice
from detector import Detector
from model import Model
from logger import Logger
import settings

# ------------------------------ Initialization -------------------------------
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
                          l_scat=settings.L_SCAT_TRUE)

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

# define hitlists
hits_true = detector.tf_sample_hits(model_true.final_positions,
                                    model_true.traveled_distances, ice_true)
hits_pred = detector.tf_expected_hits(model_pred.final_positions,
                                      model_pred.traveled_distances, ice_pred)

# take logs for loss
log_hits_true = tf.log(tf.ones_like(hits_true) + hits_true)
log_hits_pred = tf.log(tf.ones_like(hits_pred) + hits_pred)

# define loss
loss = tf.reduce_sum(tf.squared_difference(log_hits_true, log_hits_pred))

# crate variable for learning rate
tf_learning_rate = tf.Variable(settings.INITIAL_LEARNING_RATE,
                               trainable=False,
                               dtype=settings.FLOAT_PRECISION)

# create update operation for learning rate
if settings.LEARNING_DECAY_MODE == 'Linear':
    update_learning_rate = tf.assign(tf_learning_rate, tf_learning_rate -
                                     settings.LEARNING_DECR)
elif settings.LEARNING_DECAY_MODE == 'Exponential':
    update_learning_rate = tf.assign(tf_learning_rate, tf_learning_rate *
                                     settings.LEARNING_DECR)
else:
    raise ValueError(settings.LEARNING_DECAY_MODE +
                     " is not a supported decay mode!")

# initialize the optimizer
if settings.OPTIMIZER == 'Adam':
    optimizer = tf.train.AdamOptimizer(tf_learning_rate,
                                       **settings.ADAM_SETTINGS)
elif settings.OPTIMIZER == 'GradientDescent':
    optimizer = tf.train.GradientDescentOptimizer(tf_learning_rate)
else:
    raise ValueError(settings.OPTIMIZER+" is not a supported optimizer!")

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

# define variable and operations to track the average batch loss
average_loss = tf.Variable(0., trainable=False)
update_loss = average_loss.assign_add(loss/settings.BATCHES_PER_STEP)
reset_loss = average_loss.assign(0.)

if __name__ == '__main__':
    if settings.CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
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
        # sample cascade positions for this step
        r_cascades = [[50, 50, 25]
                      for i in range(settings.CASCADES_PER_STEP)]

        # propagate in batches
        for i in trange(settings.BATCHES_PER_STEP, leave=False):
            session.run([propagate_batch, update_loss],
                        feed_dict={model_true.r_cascades: r_cascades,
                                   model_pred.r_cascades: r_cascades})

        # apply accumulated gradients
        session.run(apply_gradients)

        # get updated parameters
        result = session.run([average_loss, ice_pred.l_abs, ice_pred.l_scat])
        logger.log(step, result)

        # reset variables for next step
        session.run([reset_gradients, reset_loss])

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

        if settings.LEARNING_DECAY and step % settings.LEARNING_STEPS == 0:
            learning_rate = session.run(update_learning_rate)
            logger.message("Learning rate decreased to {:2.4f}"
                           .format(learning_rate), step)
            if learning_rate <= 0:
                break
    logger.message("Done.")
