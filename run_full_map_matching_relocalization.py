import argparse
import os

import numpy as np

from relocalization.orb_matching_localization import OrbMatchingLocalization
from relocalization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/singleview_90FOV.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    is_sparse = args.sparse
    is_visualize = args.visualize

    config = load_config_module(module_name)
    image_dir = config.PathConfig.LOCALIZATION_TEST_PATH

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    num_iteration = 0
    test_num_level = 0

    for scene_number in scene_list:
        # Find number of levels
        for height in height_data:
            if scene_number in height:
                test_num_level = test_num_level + 1

    # Main loop
    total_recall = []
    total_d1 = []
    total_d2 = []
    total_queries = 0

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)
        image_dir_by_scene = os.path.join(image_dir, f"observation_{scene_number}")

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            num_iteration = num_iteration + 1
            print(num_iteration, "/", test_num_level)

            # Read binary topdown map
            binary_topdown_map = sim.topdown_map_list[level]

            # Set file path
            map_obs_dir = os.path.join(image_dir_by_scene, f"{config.PathConfig.MAP_DIR_PREFIX}_{level}")
            query_dir = os.path.join(image_dir_by_scene, f"{config.PathConfig.QUERY_DIR_PREFIX}_{level}")

            # Initialize localization instance
            localization = OrbMatchingLocalization(
                config,
                map_obs_dir=map_obs_dir,
                query_dir=query_dir,
                binary_topdown_map=binary_topdown_map,
                sparse_map=is_sparse,
                visualize=is_visualize,
            )

            recall_list, d1_list, d2_list, num_queries = localization.iterate_localization_with_query()

            total_recall = total_recall + recall_list
            total_d1 = total_d1 + d1_list
            total_d2 = total_d2 + d2_list
            total_queries = total_queries + num_queries

        sim.close()

    print("Recall: ", sum(total_recall) / total_queries)
    print("Recall std: ", np.std(total_recall))
    print("Distance 1: ", (sum(total_d1) / total_queries) * 0.1)
    print("Distance 1 std: ", np.std(total_d1) * 0.1)
    print("Distance 2: ", (sum(total_d2) / total_queries) * 0.1)
    print("Distance 2 std: ", np.std(total_d2) * 0.1)
