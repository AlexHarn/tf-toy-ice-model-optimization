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

    # initialize the ice
    ice_true = Ice()
    ice_true.homogeneous_init()

    ice_pred = Ice(trainable=True)
    ice_pred.homogeneous_init(l_abs=80, l_scat=20)

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
    optimizer = \
        tf.train.AdamOptimizer(
            learning_rate=settings.LEARNING_RATE).minimize(loss)

    # init variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    print("Starting...")
    for i in range(100000000):
        r_cascade = [np.random.uniform(high=settings.LENGTH_X),
                     np.random.uniform(high=settings.LENGTH_Y),
                     np.random.uniform(high=settings.LENGTH_Z)]
        result = session.run([optimizer, loss, ice_pred._l_abs,
                              ice_pred._l_scat],
                             feed_dict={model_true.r_cascade: r_cascade,
                                        model_pred.r_cascade: r_cascade})

        print(("[{:08d}] loss: {:2.3f} l_abs_pred: {:2.3f} l_scat_pred:"
               " {:2.3f}") .format(i, *result[1:]))
