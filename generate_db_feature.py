"""
Original code:
https://github.com/cvg/Hierarchical-Localization/blob/master/pipeline_Aachen.ipynb

The original code is released under Apache-2.0 license.

Modified by KC-ML2.
"""


import argparse
import os
from pathlib import Path

from hloc import extract_features

from global_localization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/singleview_90FOV_HD_interactive.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json

    config = load_config_module(module_name)
    image_dir = Path(config.PathConfig.LOCALIZATION_TEST_PATH)
    test_on_sim = config.DataConfig.DATA_FROM_SIM

    if test_on_sim:
        # If you use dataset generated from simulator
        scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)
        test_num_level = 0
        for scene_number in scene_list:
            # Find total number of levels
            for height in height_data:
                if scene_number in height:
                    test_num_level = test_num_level + 1
    else:
        # If you use your own dataset collected on real world
        scene_list = ["real world"]
        test_num_level = 1

    num_iteration = 0

    for scene_number in scene_list:
        if test_on_sim:
            sim = HabitatSimWithMap(scene_number, config, height_data)
            num_level = len(sim.recolored_topdown_map_list)
            scene_dirname = f"observation_{scene_number}"
            image_dir_by_scene = os.path.join(image_dir, scene_dirname)
        else:
            num_level = 0
            scene_dirname = ""
            image_dir_by_scene = image_dir

        for level in range(num_level):
            print("scene: ", scene_number, "    level: ", level)
            num_iteration = num_iteration + 1
            print(num_iteration, "/", test_num_level)

            # Set file path
            map_index = f"{config.PathConfig.MAP_DIR_PREFIX}_{level}"
            map_obs_dir = os.path.join(image_dir_by_scene, map_index)
            outputs = Path(os.path.join(config.PathConfig.HLOC_DB_ONLY_OUTPUT, scene_dirname, f"{level}"))

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            map_obs_list = [os.path.join(scene_dirname, map_index, file) for file in sorted_map_obs_file]

            # Set hloc config
            retrieval_conf = extract_features.confs["netvlad"]

            # Extract global descriptor(retrieval) feature with NetVLAD
            retrieval_features = extract_features.main(retrieval_conf, image_dir, outputs, image_list=map_obs_list)

        sim.close()
