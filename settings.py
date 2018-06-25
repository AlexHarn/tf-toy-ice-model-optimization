import tensorflow as tf

# -------------------------------- TensorFlow ---------------------------------
FLOAT_PRECISION = tf.float32
CPU_ONLY = False
RANDOM_SEED = False  # seed or False

# --------------------------------- Detector ----------------------------------
LENGTH_X = 100  # m
LENGTH_Y = 100  # m
LENGTH_Z = 1000  # m
DOM_RADIUS = 2.5  # m
DOMS_PER_STRING = 10
NX_STRINGS = 3
NY_STRINGS = 3

# ----------------------------------- Ice -------------------------------------
L_ABS_TRUE = [100, 90, 110, 105, 99, 102, 104, 50, 98, 115]
N_LAYER = len(L_ABS_TRUE)
L_ABS_START = [100]*N_LAYER
L_SCAT_TRUE = 25

# ------------------------------- Propagation ---------------------------------
# Maximum travel distance to propagete before cutoff, must be set
CUTOFF_DISTANCE = 1000.

# Distance of the detector center at which to stop propagation to spare
# computation time in units of norm([LENGTH_X, LENGTH_Y, LENGTH_Z])/2.
# Set to False for no cutoff
CUTOFF_RADIUS = 1.1

# --------------------------------- Training ----------------------------------
# One step includes one simulation of CASCADES_PER_STEP cascades which is
# fragmented into BATCHES_PER_STEP batches. Each batch contains BATCH_SIZE
# photons.
# So each cascade contains BATCH_SIZE*BATCHES_PER_STEP/CASCADES_PER_STEP
# photons.
MAX_STEPS = 100000000
BATCHES_PER_STEP = 200
BATCH_SIZE = 50000
CASCADES_PER_STEP = 100

# -------------------------------- Optimizer ----------------------------------
# How many times to reuse the same data for optimization
OPTIMIZER_STEPS_PER_SIMULATION = 1000
# The initial learning rate
INITIAL_LEARNING_RATE = 0.1
# True or False to activate/deactivate learning rate decay
LEARNING_DECAY = False
# Decay modes: Linear or Exponential
LEARNING_DECAY_MODE = 'Exponential'
# decrease the INITIAL_LEARNING_RATE every LEARNING_STEPS steps by
# LEARNING_DECR linearly or exponentially
LEARNING_DECR = 0.95
LEARNING_STEPS = 10

# supported optimizers: Adam, GradientDescent
OPTIMIZER = 'Adam'
ADAM_SETTINGS = dict(beta1=0.9, beta2=0.999, epsilon=1e-08)

# --------------------------------- Logging -----------------------------------
WRITE_INTERVAL = 1  # how many steps between each write
