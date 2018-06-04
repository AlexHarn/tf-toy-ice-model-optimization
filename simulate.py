from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange
import os

from ice import Ice
from detector import Detector
from model import Model
from logger import Logger
import settings

# ------------------------------ Initialization -------------------------------
# select device
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# set random seeds
if settings.RANDOM_SEED:
    tf.set_random_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

# initialize the ice
ice = Ice()
ice.homogeneous_init(l_abs=settings.L_ABS_TRUE,
                     l_scat=settings.L_SCAT_TRUE)

# initialize the detector
detector = Detector(dom_radius=settings.DOM_RADIUS,
                    nx_strings=settings.NX_STRINGS,
                    ny_strings=settings.NY_STRINGS,
                    doms_per_string=settings.DOMS_PER_STRING,
                    l_x=settings.LENGTH_X,
                    l_y=settings.LENGTH_Y,
                    l_z=settings.LENGTH_Z)

# initialize the model
model = Model(ice, detector)

# define hitlist
hitmask = detector.tf_dom_hitmask(model.final_positions)
hitlist = detector.tf_count_hits(model.final_positions)

if settings.CPU_ONLY:
    config = tf.ConfigProto(device_count={'GPU': 0})
else:
    # don't allocate the entire vRAM initially
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)


if __name__ == '__main__':
    # initialize all variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    # initialize the logger
    logger = Logger(logdir='/net/nfshome/home/aharnisch/hit_sim/1/')
    logger.register_variables(['cascade_x', 'cascade_y', 'cascade_z'] +
                              ['dom_{}_hits'.format(i) for i in range(27)],
                              print_variables=['cascade_x', 'cascade_y',
                                               'cascade_z'])

    logger.message("Starting...")
    for step in range(1, settings.MAX_STEPS + 1):
        # sample cascade positions for this step
        r_cascade = [np.random.uniform(high=settings.LENGTH_X),
                     np.random.uniform(high=settings.LENGTH_Y),
                     50.]

        # propagate in batches
        data = np.zeros(27)
        for i in trange(settings.BATCHES_PER_STEP, leave=False):
            data += session.run(hitlist, feed_dict={model.r_cascades:
                                                    [r_cascade]})
        logger.log(step, r_cascade + data.tolist())

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

    logger.message("Done.")
