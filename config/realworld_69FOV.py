class PathConfig:
    LOCALIZATION_TEST_PATH = "./output/output_realworld/"
    # SCENE_DIRECTORY = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    SCENE_DIRECTORY = "/data1/chlee/Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    TRAIN_IMAGE_PATH = ""
    TRAIN_LABEL_PATH = ""
    VALID_IMAGE_PATH = ""
    VALID_LABEL_PATH = ""
    TEST_IMAGE_PATH = ""
    TEST_LABEL_PATH = ""
    MODEL_WEIGHTS = "model.20230410-153553.threeview.69FOV.weights.hdf5"
    GPU_ID = 1


class CamConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    NUM_CAMERA = 3
    IMAGE_CONCAT = True
    WIDTH = 256
    HEIGHT = 192
    SENSOR_HEIGHT = 0.34
    if RGB_360_SENSOR:
        assert RGB_SENSOR is False
        assert NUM_CAMERA == 1
    if IMAGE_CONCAT:
        assert NUM_CAMERA > 1
        assert RGB_360_SENSOR is False


class ActionConfig:
    FORWARD_AMOUNT = 0.25
    BACKWARD_AMOUNT = 0.25
    TURN_LEFT_AMOUNT = 5.0
    TURN_RIGHT_AMOUNT = 5.0


class DataConfig:
    METERS_PER_PIXEL = 0.1
    REMOVE_ISOLATED = True


class TrainingConstant:
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    POSITIVE_SAMPLE_DISTANCE = 10
    NEGATIVE_SAMPLE_MULTIPLIER = 5
    NUM_SAMPLING_PER_LEVEL = 500


class TestConstant:
    BATCH_SIZE = 32
    SIMILARITY_PROBABILITY_THRESHOLD = 0.95
    NUM_SAMPLING_PER_LEVEL = 100


class NetworkConstant:
    # Input size
    NET_WIDTH = 256
    NET_HEIGHT = 192
    NET_CHANNELS = 3
    # Architecture
    NUM_EMBEDDING = 256
    TOP_HIDDEN = 3
