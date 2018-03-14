from __future__ import division
import tensorflow as tf
from ice import Ice
from detector import Detector
from model import Model
import settings


if __name__ == '__main__':
    # ---------------------------- Initialization -----------------------------
    ice_true = Ice()
    ice_true.homogeneous_init()

    ice_pred = Ice(trainable=True)
    ice_pred.homogeneous_init(l_abs=80, l_scat=20)

    detector = Detector(dom_radius=settings.DOM_RADIUS,
                        nx_strings=settings.NX_STRINGS,
                        ny_strings=settings.NY_STRINGS,
                        doms_per_string=settings.DOMS_PER_STRING)

    model_true = Model(ice_true, detector)
    model_pred = Model(ice_pred, detector)

    # setup TensorFLow, use CPU only for now
    if settings.CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(config=config)
    else:
        session = tf.Session()

    # define loss
    mean_true, std_true = tf.nn.moments(model_true.final_positions, axes=0)
    mean_pred, std_pred = tf.nn.moments(model_pred.final_positions, axes=0)
    loss = tf.squared_difference(mean_true, mean_pred) + \
        tf.squared_difference(std_true, std_pred)

    # init optimizer
    optimizer = \
        tf.train.AdamOptimizer(
            learning_rate=settings.LEARNING_RATE).minimize(loss)

    hits = detector.tf_count_hits(model_true.final_positions)
    # init variables
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    print("Starting training...")
    model_true.init_cascade(500, 500, 250, settings.N_PHOTONS)
    model_pred.init_cascade(500, 500, 250, settings.N_PHOTONS)
    feed_dict = model_true.feed_dict
    feed_dict.update(model_pred.feed_dict)
    print(session.run([optimizer, hits], feed_dict=feed_dict)[1])
    # for i in range(1000000000):
        # model_true.init_cascade(500, 500, 500, settings.N_PHOTONS)
        # model_pred.init_cascade(500, 500, 500, settings.N_PHOTONS)
        # result = session.run([optimizer, ice_pred._l_abs, ice_pred._l_scat],
                             # feed_dict={**model_true.feed_dict,
                                        # **model_pred.feed_dict})

        # print(('[{:08d}] l_abs_pred: {:2.3f} l_scat_pred: {:2.3f}')
              # .format(i, *result[1:]))
