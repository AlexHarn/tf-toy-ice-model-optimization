from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange

from ice import Ice
from detector import Detector
from model import Model
from logger import Logger
import settings

# --------------------------------- Settings ----------------------------------
L_ABS = np.linspace(80, 120, 9)
L_SCAT = np.linspace(20, 30, 9)


# ------------------------- General Initialization ----------------------------
# set random seeds
if settings.RANDOM_SEED:
    tf.set_random_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

if settings.CPU_ONLY:
    config = tf.ConfigProto(device_count={'GPU': 0})
else:
    # don't allocate the entire vRAM initially
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

# ----------------------------- Model Definition ------------------------------
# initialize the ice
ice_true = Ice()
ice_true.homogeneous_init(l_abs=settings.L_ABS_TRUE,
                          l_scat=settings.L_SCAT_TRUE)

ice_pred = Ice(placeholders=True)
ice_pred.homogeneous_init()

# initialize the detector
detector = Detector(dom_radius=settings.DOM_RADIUS,
                    nx_strings=settings.NX_STRINGS,
                    ny_strings=settings.NY_STRINGS,
                    doms_per_string=settings.DOMS_PER_STRING,
                    l_x=settings.LENGTH_X, l_y=settings.LENGTH_Y,
                    l_z=settings.LENGTH_Z)

# initialize the models
model_true = Model(ice_true, detector)
model_pred = Model(ice_pred, detector)


# ----------------------------------- Loss ------------------------------------
# define hitlists
hits_true_biased = detector.tf_count_hits(model_true.final_positions)
hits_pred_soft = detector.tf_soft_count_hits(model_pred.final_positions)
hits_pred_hard = detector.tf_count_hits(model_pred.final_positions)

# (reverse) bias correct true hits and take logs
corrector = tf.stop_gradient(hits_pred_soft - hits_pred_hard) \
    * tf.reduce_sum(hits_pred_hard)/tf.reduce_sum(hits_true_biased)

hits_true = tf.log(hits_true_biased + corrector + 1)
hits_pred = tf.log(hits_pred_soft + 1)

# define loss
loss = tf.reduce_sum(tf.squared_difference(hits_true, hits_pred))

# define variable and operations to track the average batch loss
average_loss = tf.Variable(0., trainable=False)
update_loss = average_loss.assign_add(loss/settings.BATCHES_PER_STEP)
reset_loss = average_loss.assign(0.)

if __name__ == '__main__':
    # --------------------------------- Run -----------------------------------
    # initialize all variables
    session.run(tf.global_variables_initializer())

    # initialize the logger
    logger = Logger(logdir='/data/user/aharnisch/log/gridsearch/')
    logger.register_variables(['loss', 'l_abs', 'l_scat'], print_all=True)

    # initialize grid array
    grid = np.zeros(shape=(len(L_ABS), len(L_SCAT)))

    logger.message("Starting...")
    for i in range(len(L_ABS)):
        for j in range(len(L_SCAT)):
            # sample cascade positions for this step
            r_cascades = [[50, 50, 25]
                          for n in range(settings.CASCADES_PER_STEP)]

            # propagate in batches
            for k in trange(settings.BATCHES_PER_STEP, leave=False):
                session.run([update_loss],
                            feed_dict={model_true.r_cascades: r_cascades,
                                       model_pred.r_cascades: r_cascades,
                                       ice_pred.l_abs: L_ABS[i],
                                       ice_pred.l_scat: L_SCAT[j]})

            # get updated parameters
            grid[i][j] = session.run([average_loss])[0]

            # log
            logger.log(i*len(L_SCAT) + j + 1, (grid[i][j], L_ABS[i],
                                               L_SCAT[j]))
            logger.write()

            # reset loss
            session.run([reset_loss])
