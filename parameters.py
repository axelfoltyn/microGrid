
class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPISODE = 365 * 24 - 1
    EPISODE = 60
    STEPS_PER_TEST = 365 * 24 - 1


    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 1e-5
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.8
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.98
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 4e-6
    REPLAY_MEMORY_SIZE = 100000
    BATCH_SIZE = 256
    FREEZE_INTERVAL = 50
    DETERMINISTIC = False
    TARGET_UPDATE_INTERVAL = 2


# parameters used in the initialization of environments
class EnvParam:
    # The max value of the network purchase (None = no limit)
    MAX_BUY_ENERGY = None
    # The maximum value of the network sale (None = no limit)
    MAX_SELL_ENERGY = None
    # To know the production and consumption of the following
    PREDICTION = False
    # To have the deadline before the next solstice
    EQUINOX = True
    # Size of the production/consumption history
    LENGTH_HISTORY = 12