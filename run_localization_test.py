import argparse
import os

import numpy as np

from relocalization.sim import HabitatSimWithMap
from utils.config_import import import_localization_class, load_config_module
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/concat_fourview_69FOV_HD.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--visualize", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    is_visualize = args.visualize

    config = load_config_module(module_name)
    image_dir = config.PathConfig.LOCALIZATION_TEST_PATH
    test_on_sim = config.DataConfig.DATA_FROM_SIM

    if test_on_sim:  # If you use dataset generated from simulator
        scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)
        test_num_level = 0
        # Find total number of levels
        for scene_number in scene_list:
            for height in height_data:
                if scene_number in height:
                    test_num_level = test_num_level + 1
    else:  # If you use your own dataset collected on real world
        scene_list = ["real world"]
        test_num_level = 1

    num_iteration = 0

    total_accuracy = []
    total_d1 = []
    total_d2 = []
    total_queries = 0

    recolored_topdown_map = None
    binary_topdown_map = None

    # Main loop
    for scene_number in scene_list:
        if test_on_sim:
            sim = HabitatSimWithMap(scene_number, config, height_data)  # Initialize simulator for member variables
            num_level = len(sim.recolored_topdown_map_list)
            scene_dirname = f"observation_{scene_number}"
            image_dir_by_scene = os.path.join(image_dir, scene_dirname)
        else:
            num_level = 1
            scene_dirname = ""
            image_dir_by_scene = image_dir

        for level in range(num_level):
            print("scene: ", scene_number, "    level: ", level)
            num_iteration = num_iteration + 1
            print(num_iteration, "/", test_num_level)

            # Read topdown map if you use dataset from simulation. No topdown map for real-world currently
            if test_on_sim:
                recolored_topdown_map = sim.recolored_topdown_map_list[level]
                binary_topdown_map = sim.topdown_map_list[level]

            # Set image file path
            map_obs_dir = os.path.join(image_dir_by_scene, f"{config.PathConfig.MAP_DIR_PREFIX}_{level}")
            query_dir = os.path.join(image_dir_by_scene, f"{config.PathConfig.QUERY_DIR_PREFIX}_{level}")

            # Import global localization method dynamically
            localization_class = import_localization_class(config.PathConfig.LOCALIZATION_CLASS_PATH)
            localization = localization_class(
                config,
                map_obs_dir,
                query_dir,
                binary_topdown_map=binary_topdown_map,
                visualize=is_visualize,
            )

            # Iterate global localization process with queries at current space and level
            accuracy_list, d1_list, d2_list, num_queries = localization.iterate_localization_with_query(
                recolored_topdown_map
            )

            # Accumulate results
            total_accuracy = total_accuracy + accuracy_list
            total_d1 = total_d1 + d1_list
            total_d2 = total_d2 + d2_list
            total_queries = total_queries + num_queries

        if test_on_sim:
            sim.close()

    print("Accuracy: ", sum(total_accuracy) / total_queries)
    print("Pos Distance: ", (sum(total_d1) / total_queries) * 0.1)
    print("Pos Distance std: ", np.std(total_d1) * 0.1)
    print("Node Distance: ", (sum(total_d2) / total_queries) * 0.1)
    print("Node Distance std: ", np.std(total_d2) * 0.1)
