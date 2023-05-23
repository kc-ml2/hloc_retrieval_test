class PathConfig:
    LOCALIZATION_CLASS_PATH = "relocalization.localization_netvlad_superpoint.LocalizationNetVLADSuperpoint"
    LOCALIZATION_TEST_PATH = "/data1/chlee/output/output_realworld/"
    # SCENE_DIRECTORY = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    SCENE_DIRECTORY = "/data1/chlee/Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    HLOC_OUTPUT = "/data1/chlee/netvlad_output/realworld_69FOV/"
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


class ActionConfig:
    FORWARD_AMOUNT = 0.25
    BACKWARD_AMOUNT = 0.25
    TURN_LEFT_AMOUNT = 5.0
    TURN_RIGHT_AMOUNT = 5.0


class DataConfig:
    DATA_FROM_SIM = False
    METERS_PER_PIXEL = 0.1
