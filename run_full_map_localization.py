import argparse
import os

import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from config.env_config import ActionConfig, CamFourViewConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from habitat_env.localization import Localization
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    parser.add_argument("--load-model", default="./model_weights/model.20221129-125905.32batch.4view.weights.best.hdf5")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path
    loaded_model = args.load_model

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    # Load pre-trained model & top network
    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        model, top_network, bottom_network = ResnetBuilder.load_model(loaded_model)

    # Main loop
    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            # Read binary topdown map
            binary_topdown_map = sim.topdown_map_list[level]

            # Set file path
            map_obs_dir = os.path.join(observation_path, f"map_node_observation_level_{level}")
            sample_dir = os.path.join(observation_path, f"test_sample_{level}")

            # Initialize localization instance
            localization = Localization(
                top_network,
                bottom_network,
                map_obs_dir,
                sample_dir=sample_dir,
                binary_topdown_map=binary_topdown_map,
            )
            localization.iterate_localization_with_sample(recolored_topdown_map)

        sim.close()
