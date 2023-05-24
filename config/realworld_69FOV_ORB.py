class PathConfig:
    LOCALIZATION_CLASS_PATH = "relocalization.localization_orb_matching.LocalizationOrbMatching"
    LOCALIZATION_TEST_PATH = "./output/output_realworld/"
    SCENE_DIRECTORY = "./Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    HLOC_OUTPUT = "./netvlad_output/realworld_69FOV/"
    MAP_DIR_PREFIX = "map_node_observation_level"
    QUERY_DIR_PREFIX = "test_query"
    POS_RECORD_FILE_PREFIX = "pos_record"


class CamConfig:
    NUM_CAMERA = 3
    IMAGE_CONCAT = True


class DataConfig:
    DATA_FROM_SIM = False
    METERS_PER_PIXEL = 0.1
