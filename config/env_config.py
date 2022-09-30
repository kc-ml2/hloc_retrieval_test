class CamNormalConfig:
    RGB_SENSOR = True
    RGB_360_SENSOR = False
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    WIDTH = 256
    HEIGHT = 256
    SENSOR_HEIGHT = 0.5


class Cam360Config:
    RGB_SENSOR = False
    RGB_360_SENSOR = True
    DEPTH_SENSOR = True
    SEMANTIC_SENSOR = False
    WIDTH = 512
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


class DisplayOnConfig:
    DISPLAY_OBSERVATION = True
    DISPLAY_PATH_MAP = True
    DISPLAY_SEMANTIC_OBJECT = False


class DisplayOffConfig:
    DISPLAY_OBSERVATION = False
    DISPLAY_PATH_MAP = False
    DISPLAY_SEMANTIC_OBJECT = False


class OutputConfig:
    SAVE_PATH_MAP = False
