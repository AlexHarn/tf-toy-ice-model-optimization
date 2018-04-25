import tensorflow as tf

# -------------------------------- TensorFlow ---------------------------------
FLOAT_PRECISION = tf.float32
CPU_ONLY = False
RANDOM_SEED = False  # seed or False

# --------------------------------- Detector ----------------------------------
LENGTH_X = 100  # m
LENGTH_Y = 100  # m
LENGTH_Z = 100  # m
DOM_RADIUS = 5  # m
DOMS_PER_STRING = 3
NX_STRINGS = 3
NY_STRINGS = 3

BOUNDARY = 500

# ----------------------------------- Ice -------------------------------------
L_ABS_TRUE = 1000
L_SCAT_TRUE = 25
L_ABS_START = 1000
L_SCAT_START = 30

# ------------------------------- Propagation ---------------------------------
# Distance of the detector center at which to stop propagation to spare
# computation time in units of norm([LENGTH_X, LENGTH_Y, LENGTH_Z])/2.
# Set to False for no cutoff
CUTOFF_RADIUS = 1.1

# --------------------------------- Training ----------------------------------
MAX_STEPS = 100000000
BATCHES_PER_STEP = 1
BATCH_SIZE = 50000
# each cascades contains BATCH_SIZE*BATCHES_PER_STEP/CASCADES_PER_STEP photons
CASCADES_PER_STEP = 1

LEARNING_RATE = 1

# --------------------------------- Logging -----------------------------------
WRITE_INTERVAL = 1  # how many steps between each write
