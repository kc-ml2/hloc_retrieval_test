class PathConfig:
    LOCALIZATION_CLASS_PATH = "global_localization.netvlad_superpoint.LocalizationNetVLADSuperpoint"
    LOCALIZATION_TEST_PATH = "/data1/chlee/output/output_single_view_90FOV_HD/"
    SCENE_DIRECTORY = "/data1/chlee/Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    HLOC_OUTPUT = "/data1/chlee/netvlad_output/singleview_90FOV_HD/"
    HLOC_DB_ONLY_OUTPUT = "/data1/chlee/netvlad_DB/singleview_90FOV_HD/"
    MAP_DIR_PREFIX = "map_node_observation_level"
    QUERY_DIR_PREFIX = "test_query"
    POS_RECORD_FILE_PREFIX = "pos_record"


class CamConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    NUM_CAMERA = 4
    IMAGE_CONCAT = False
    WIDTH = 640
    HEIGHT = 640
    SENSOR_HEIGHT = 0.34


class ActionConfig:
    FORWARD_AMOUNT = 0.5
    BACKWARD_AMOUNT = 0.5
    TURN_LEFT_AMOUNT = 15.0
    TURN_RIGHT_AMOUNT = 15.0


class DataConfig:
    DATA_FROM_SIM = True
    METERS_PER_PIXEL = 0.1
    REMOVE_ISOLATED = True
    NUM_TEST_SAMPLE_PER_LEVEL = 100