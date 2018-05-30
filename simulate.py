from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt

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
    logger = Logger(logdir='./log/', overwrite=True)
    logger.register_variables(['average_arrival_time'], print_all=True)

    logger.message("Starting...")
    for step in range(1, settings.MAX_STEPS + 1):
        # sample cascade positions for this step
        r_cascades = [[50, 50, 25]]

        # propagate in batches
        for i in trange(settings.BATCHES_PER_STEP, leave=False):
            result = session.run([model.arrival_times, hitmask, hitlist],
                                 feed_dict={model.r_cascades: r_cascades})

        dom_arrival_times = []
        mean_arrival_times = []
        for i in range(len(detector.doms)):
            dom_arrival_times.append(result[0][result[1][:, i]])
            mean_arrival_times.append(np.mean(dom_arrival_times[-1]))
            # plt.hist(dom_arrival_times[-1], bins=50, label=str(i))
            # plt.axvline(np.linalg.norm(detector.doms[i] - r_cascades[0]), 0, 1,
                        # c='k', lw=2)
            # plt.legend()
            # plt.show()
            # plt.clf()

        print(mean_arrival_times)

        exit()

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

    logger.message("Done.")
