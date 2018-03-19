import tensorflow as tf

# -------------------------------- TensorFlow ---------------------------------
FLOAT_PRECISION = tf.float32
CPU_ONLY = False
RANDOM_SEED = 1234  # seed or False

# --------------------------------- Detector ----------------------------------
LENGTH_X = 100  # m
LENGTH_Y = 100  # m
LENGTH_Z = 100  # m
DOM_RADIUS = 5  # m
DOMS_PER_STRING = 3
NX_STRINGS = 3
NY_STRINGS = 3

# --------------------------------- Training ----------------------------------
LEARNING_RATE = 0.1
N_STEPS = 100000000
BATCHES_PER_STEP = 20
BATCH_SIZE = 25000
# each cascades contains BATCH_SIZE*BATCHES_PER_STEP/CASCADES_PER_STEP photons
CASCADES_PER_STEP = 5
