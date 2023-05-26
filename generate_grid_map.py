import argparse
import json
import os

import cv2

from global_localization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import get_entire_maps_by_levels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/concat_fourview_69FOV_HD.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--output-path", default="./data")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    output_path = args.output_path

    config = load_config_module(module_name)

    topdown_path = os.path.join(output_path, "topdown/")
    recolored_path = os.path.join(output_path, "recolored_topdown/")
    height_json_path = os.path.join(output_path, "map_height.json")
    os.makedirs(topdown_path, exist_ok=True)
    os.makedirs(recolored_path, exist_ok=True)

    generated_scene_num = 0
    height = {}

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        print(scene_number)
        print(generated_scene_num)

        sim = HabitatSimWithMap(scene_number, config)

        # Sample topdown maps randomly & get the largest maps by levels
        recolored_topdown_map_list, topdown_map_list, height_list = get_entire_maps_by_levels(
            sim, config.DataConfig.METERS_PER_PIXEL
        )

        # Export the largest maps as bmp image
        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            topdown_map = topdown_map_list[i]

            topdown_file_name = f"{scene_number}_{i}.bmp"
            recolored_file_name = f"{scene_number}_{i}.bmp"
            cv2.imwrite(os.path.join(topdown_path, topdown_file_name), topdown_map)
            cv2.imwrite(os.path.join(recolored_path, recolored_file_name), recolored_topdown_map)

            height[f"{scene_number}_{i}"] = float(height_list[i])

            with open(height_json_path, "w") as label_json:  # pylint: disable=unspecified-encoding
                json.dump(height, label_json, indent=4)

        sim.close()
        generated_scene_num = generated_scene_num + 1
