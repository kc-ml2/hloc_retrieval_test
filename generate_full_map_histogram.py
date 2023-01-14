import argparse
import os

from relocalization.object_spatial_pyramid import ObjectSpatialPyramid
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    num_iteration = 0
    test_num_level = 0

    for scene_number in scene_list:
        # Find number of levels
        for height in height_data:
            if scene_number in height:
                test_num_level = test_num_level + 1

    # Make list to iterate for Siamese forward
    for scene_number in scene_list:
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        # Find number of levels
        num_level = 0
        for height in height_data:
            if scene_number in height:
                num_level = num_level + 1
        if num_level == 0:
            raise ValueError("Height data is not found.")

        for level in range(num_level):
            print("scene: ", scene_number, "    level: ", level)
            num_iteration = num_iteration + 1
            print(num_iteration, "/", test_num_level)

            map_obs_dir = os.path.join(observation_path, f"map_node_observation_level_{level}")
            sample_dir = os.path.join(observation_path, f"test_sample_{level}")

            object_spatial_pyramid = ObjectSpatialPyramid(map_obs_dir, sample_dir, load_cache=True)
