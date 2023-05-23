import argparse
from pathlib import Path
import os

from hloc import extract_features, match_features, pairs_from_retrieval

from utils.habitat_utils import open_env_related_files
from relocalization.sim import HabitatSimWithMap
from utils.config_import import load_config_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/singleview_90FOV.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json

    config = load_config_module(module_name)
    image_dir = config.PathConfig.LOCALIZATION_TEST_PATH
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
            query_index = f"{config.PathConfig.QUERY_DIR_PREFIX}_{level}"
            map_obs_dir = os.path.join(image_dir_by_scene, map_index)
            query_dir = os.path.join(image_dir_by_scene, query_index)
            outputs = os.path.join(config.PathConfig.HLOC_OUTPUT, scene_dirname, f"{level}")

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            sorted_test_query_file = sorted(os.listdir(query_dir))

            map_obs_list = [os.path.join(scene_dirname, map_index, file) for file in sorted_map_obs_file]
            query_list = [os.path.join(scene_dirname, query_index, file) for file in sorted_test_query_file]
            total_image_list = map_obs_list + query_list

            retrieval_pairs = outputs / 'pairs-netvlad.txt'

            retrieval_conf = extract_features.confs['netvlad']
            feature_conf = extract_features.confs['superpoint_inloc']
            matcher_conf = match_features.confs['NN-superpoint']

            retrieval_features = extract_features.main(retrieval_conf, image_dir, outputs, image_list=total_image_list)
            print("extract retrieval feature")

            pairs_from_retrieval.main(retrieval_features, retrieval_pairs, num_matched=20, query_list=query_list, db_list=map_obs_list)
            print("get pair text file from retrieval feature")

            feature_path = extract_features.main(feature_conf, image_dir, outputs, image_list=total_image_list)
            print("extract superpoint feature")

            match_path = match_features.main(matcher_conf, retrieval_pairs, feature_conf['output'], outputs)
            print("get match result h5 file from superpoint feature")

        sim.close()
