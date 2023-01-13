class PathConfig:
    SCENE_DIRECTORY = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    # SCENE_DIRECTORY = "/data1/chlee/Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    # TRAIN_IMAGE_PATH = "/data1/chlee/siamese_dataset/images"
    # TRAIN_LABEL_PATH = "./data/label_train.json"
    # VALID_IMAGE_PATH = "/data1/chlee/siamese_dataset/val_images"
    # VALID_LABEL_PATH = "./data/label_val.json"
    # TEST_IMAGE_PATH = "/data1/chlee/siamese_dataset/test_images"
    # TEST_LABEL_PATH = "./data/label_test.json"
    TRAIN_IMAGE_PATH = "/data1/chlee/siamese_dataset_four_view/images"
    TRAIN_LABEL_PATH = "./data/label_train_four_view.json"
    VALID_IMAGE_PATH = "/data1/chlee/siamese_dataset_four_view/val_images"
    VALID_LABEL_PATH = "./data/label_val_four_view.json"
    TEST_IMAGE_PATH = "/data1/chlee/siamese_dataset_four_view/test_images"
    TEST_LABEL_PATH = "./data/label_test_four_view.json"
    EDGE_MODEL_WEIGHTS = ""
    SHORTCUTS_CACHE_FILE = ""
    GPU_ID = 0


class CamNormalConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    FOUR_VIEW = False
    WIDTH = 256
    HEIGHT = 256
    SENSOR_HEIGHT = 0.5


class Cam360Config:
    RGB_SENSOR = False
    RGB_360_SENSOR = True
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    FOUR_VIEW = False
    WIDTH = 512
    HEIGHT = 256
    SENSOR_HEIGHT = 0.5


class CamFourViewConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    FOUR_VIEW = True
    WIDTH = 256
    HEIGHT = 256
    SENSOR_HEIGHT = 0.5


class CamGivenReferenceConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = True
    WIDTH = 256
    HEIGHT = 256
    SENSOR_HEIGHT = 0


class ActionConfig:
    FORWARD_AMOUNT = 0.25
    BACKWARD_AMOUNT = 0.25
    TURN_LEFT_AMOUNT = 5.0
    TURN_RIGHT_AMOUNT = 5.0


class DataConfig:
    METERS_PER_PIXEL = 0.1
    REMOVE_DUPLICATE_FRAMES = True
    TRANSLATION_THRESHOLD = 0.5
    INTERPOLATE_TRANSLATION = True
    INTERPOLATION_INTERVAL = 0.02
    IS_DENSE_GRAPH = True
    REMOVE_ISOLATED = True


class DisplayConfig:
    DISPLAY_OBSERVATION = True
    DISPLAY_PATH_MAP = True


class OutputConfig:
    SAVE_PATH_MAP = False
