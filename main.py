from __future__ import division
import tensorflow as tf
from ice import Ice
from detector import Detector
from model import Model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # ---------------------------- Initialization -----------------------------
    ice = Ice()
    ice.homogeneous_init()
    detector = Detector()
    model = Model(ice, detector)

    # initialize a very weak cascade in the middle of the detector
    model.init_cascade(500, 500, 500, n_photons=1000)

    # setup TensorFLow, use CPU only for now
    config = tf.ConfigProto(device_count={'GPU': 0})

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    # --------------------------------- Run -----------------------------------
    result = session.run(model.final_positions, feed_dict={**model.feed_dict})
    print(result)

    # visualize result for debugging
    # TODO: visualizer class with DOM plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    # draw photons
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], marker='.')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
