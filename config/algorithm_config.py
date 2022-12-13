class NavConstant:
    SMOOTHED_LOCALIZATION = True
    MEMORY_SUBSAMPLING = 4
    MIN_SHORTCUT_DISTANCE = 5
    WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.7
    INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.95
    SHORTCUT_WINDOW = 10
    MIN_LOOK_AHEAD = 1
    MAX_LOOK_AHEAD = 7
    PREDICTION_BATCH_SIZE = 1024
    NUMBER_OF_NEAREST_NEIGHBOURS = 5
    SMALL_SHORTCUTS_NUMBER = 2000
    LARGE_SHORTCUTS_NUMBER = 100000
    assert SMALL_SHORTCUTS_NUMBER <= LARGE_SHORTCUTS_NUMBER


class TrainingConstant:
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    POSITIVE_SAMPLE_DISTANCE = 10
    NEGATIVE_SAMPLE_MULTIPLIER = 5
    NUM_SAMPLING_PER_LEVEL = 500


class TestConstant:
    BATCH_SIZE = 32
    SIMILARITY_PROBABILITY_THRESHOLD = 0.95
    IGNORE_CLOSE_NODES = False
    NUM_SAMPLING_PER_LEVEL = 100


class NetworkConstant:
    # Input size
    NET_WIDTH = 1024
    NET_HEIGHT = 256
    NET_CHANNELS = 3
