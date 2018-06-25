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
ice_true.init(l_abs=settings.L_ABS_TRUE, l_scat=settings.L_SCAT_TRUE)

ice_pred = Ice(trainable=True)
ice_pred.init(l_abs=settings.L_ABS_START, l_scat=settings.L_SCAT_TRUE)

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

# save final positions and traveled layer distances as variables for each batch
final_positions_true = []
final_positions_pred = []
traveled_layer_distance_true = []
traveled_layer_distance_pred = []

for i in range(settings.BATCHES_PER_STEP):
    final_positions_true.append(tf.Variable(
        tf.zeros((settings.BATCH_SIZE, 3), dtype=settings.FLOAT_PRECISION)))
    final_positions_pred.append(tf.Variable(
        tf.zeros((settings.BATCH_SIZE, 3), dtype=settings.FLOAT_PRECISION)))

    traveled_layer_distance_true.append(tf.Variable(
        tf.zeros((settings.BATCH_SIZE, settings.N_LAYER),
                 dtype=settings.FLOAT_PRECISION)))
    traveled_layer_distance_pred.append(tf.Variable(
        tf.zeros((settings.BATCH_SIZE, settings.N_LAYER),
                 dtype=settings.FLOAT_PRECISION)))

# define hitlists
hits_true = detector.tf_sample_hits(
    tf.concat(final_positions_true, 0),
    tf.concat(traveled_layer_distance_true, 0),
    ice_true)

hits_pred = detector.tf_expected_hits(
    tf.concat(final_positions_pred, 0),
    tf.concat(traveled_layer_distance_pred, 0),
    ice_pred)

# Dimas likelihood, take the logarithm for stability
mu = (hits_pred + hits_true)/(2*settings.BATCHES_PER_STEP)
logLR_doms = hits_pred*(
    tf.log(mu) - tf.log(hits_pred/settings.BATCHES_PER_STEP)) + \
    hits_true*(tf.log(mu) - tf.log(hits_true/settings.BATCHES_PER_STEP))

loss = -tf.reduce_sum(tf.where(tf.is_nan(logLR_doms),
                               tf.zeros_like(logLR_doms), logLR_doms))

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

# create operation to minimize the loss
optimize = optimizer.minimize(loss)

# grab all trainable variables
trainable_variables = tf.trainable_variables()

# define operations to propagate each batch
propagate_batch = [[final_positions_true[i].assign(model_true.final_positions),
                    final_positions_pred[i].assign(model_pred.final_positions),
                    traveled_layer_distance_true[i].assign(
                        model_true.traveled_layer_distance),
                    traveled_layer_distance_pred[i].assign(
                        model_pred.traveled_layer_distance)]
                   for i in range(settings.BATCHES_PER_STEP)]

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
    logger.register_variables(['loss'] + ['l_abs_pred_{}'.format(i) for i in
                                          range(len(settings.L_ABS_TRUE))],
                              print_all=True)
    logger.message("Starting...")

    for step in range(1, settings.MAX_STEPS + 1):
        # sample cascade positions for this step
        r_cascades = [[settings.LENGTH_X/2,
                       settings.LENGTH_Y/2,
                       np.random.uniform(high=settings.LENGTH_Z)]
                      for i in range(settings.CASCADES_PER_STEP)]

        # propagate in batches
        for batch in trange(settings.BATCHES_PER_STEP, leave=False):
            session.run(propagate_batch[batch],
                        feed_dict={model_true.r_cascades: r_cascades,
                                   model_pred.r_cascades: r_cascades})

        # compute and apply gradients and get the loss with this data
        optimizer_step_loss = np.zeros(settings.OPTIMIZER_STEPS_PER_SIMULATION)
        for optimizer_step in trange(settings.OPTIMIZER_STEPS_PER_SIMULATION,
                                     leave=False):
            optimizer_step_loss[optimizer_step] = \
                session.run([optimize, loss])[1]

        # calculate average loss
        step_loss = np.mean(optimizer_step_loss)

        # get updated parameters
        result = session.run(ice_pred.l_abs)
        logger.log(step, [step_loss] + result.tolist())

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

        if settings.LEARNING_DECAY and step % settings.LEARNING_STEPS == 0:
            learning_rate = session.run(update_learning_rate)
            logger.message("Learning rate decreased to {:2.4f}"
                           .format(learning_rate), step)
            if learning_rate <= 0:
                break
    logger.message("Done.")
