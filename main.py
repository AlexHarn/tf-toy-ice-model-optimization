from __future__ import division
import tensorflow as tf
import numpy as np
from ice import Ice
from detector import Detector
from model import Model
import settings


if __name__ == '__main__':
    # ---------------------------- Initialization -----------------------------
    if settings.RANDOM_SEED:
        tf.set_random_seed(settings.RANDOM_SEED)
        np.random.seed(settings.RANDOM_SEED)
    ice_true = Ice()
    ice_true.homogeneous_init()

    ice_pred = Ice(trainable=True)
    ice_pred.homogeneous_init(l_abs=80, l_scat=20)

    detector = Detector(dom_radius=settings.DOM_RADIUS,
                        nx_strings=settings.NX_STRINGS,
                        ny_strings=settings.NY_STRINGS,
                        doms_per_string=settings.DOMS_PER_STRING,
                        l_x=settings.LENGTH_X,
                        l_y=settings.LENGTH_Y,
                        l_z=settings.LENGTH_Z)

    model_true = Model(ice_true, detector)
    model_pred = Model(ice_pred, detector)

    if settings.CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        # don't allocate the entire vRAM initially
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)

    # define loss
    mean_true, std_true = tf.nn.moments(model_true.final_positions, axes=0)
    mean_pred, std_pred = tf.nn.moments(model_pred.final_positions, axes=0)
    loss = tf.squared_difference(mean_true, mean_pred) + \
        tf.squared_difference(std_true, std_pred)

    # init optimizer
    optimizer = \
        tf.train.AdamOptimizer(
            learning_rate=settings.LEARNING_RATE).minimize(loss)

    hits_true = detector.tf_count_hits(model_true.final_positions)
    hits_pred = detector.tf_soft_count_hits(model_pred.final_positions)

    # init variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    print("Starting...")
    r_cascade = [50, 50, 25]
    r = session.run([optimizer, hits_true, hits_pred],
                    feed_dict={model_true.r_cascade: r_cascade,
                               model_pred.r_cascade: r_cascade})
    print(r[1:])
