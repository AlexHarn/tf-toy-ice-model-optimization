from __future__ import division
from ice import Ice
from detector import Detector
from model import Model


if __name__ == '__main__':
    # initialize ice and detector
    ice = Ice()
    detector = Detector()

    # initialize the model
    model = Model(ice, detector)
