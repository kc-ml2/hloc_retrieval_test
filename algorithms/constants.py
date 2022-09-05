#!/usr/bin/env python
from algorithms import resnet


# Loaded from environment
class EnvConstant:
    SMOOTHED_LOCALIZATION = True
    MEMORY_SUBSAMPLING = 4
    MIN_SHORTCUT_DISTANCE = 5
    MEMORY_MAX_FRAMES = None
    EDGE_EXPERIMENT_ID = "0103_R"
    EXPERIMENT_OUTPUT_FOLDER = "default_experiment"
    WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.7
    INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.95
    SHORTCUT_WINDOW = 10
    MIN_LOOK_AHEAD = 1
    MAX_LOOK_AHEAD = 7
    NUMBER_OF_TRIALS = 6


# Training
class TrainingConstant:
    LEARNING_RATE = 1e-04
    MODEL_CHECKPOINT_PERIOD = 100
    BATCH_SIZE = 64
    DUMP_AFTER_BATCHES = 100
    INF_EPOCHS = 1000000000
    EDGE_MAX_EPOCHS = 10000
    EDGE_EPISODES = 10
    EDGE_CLASSES = 2
    NEGATIVE_SAMPLE_MULTIPLIER = 5


# Testing
class NetworkConstant:
    JOINT_NETWORK = resnet.ResnetBuilder.build_resnet_18
    SIAMESE_NETWORK = resnet.ResnetBuilder.build_siamese_resnet_18
    ACTION_NETWORK = resnet.ResnetBuilder.build_resnet_18
    TESTING_BATCH_SIZE = 1024
    NUMBER_OF_NEAREST_NEIGHBOURS = 5
    SMALL_SHORTCUTS_NUMBER = 2000
    LARGE_SHORTCUTS_NUMBER = 100000
    assert SMALL_SHORTCUTS_NUMBER <= LARGE_SHORTCUTS_NUMBER
    # Input size
    NET_WIDTH = 160
    NET_HEIGHT = 120
    NET_CHANNELS = 3


# Paths
class PathConstant:
    EDGE_MODEL_WEIGHTS = ""
    SHORTCUTS_CACHE_FILE = ""
