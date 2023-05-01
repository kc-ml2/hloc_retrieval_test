import argparse
import json
import os

import cv2

from relocalization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import get_entire_maps_by_levels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/singleview_90FOV.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_total.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json

    config = load_config_module(module_name)

    os.makedirs("./data/topdown/", exist_ok=True)
    os.makedirs("./data/recolored_topdown/", exist_ok=True)

    generated_scene_num = 0
    height = {}

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        print(scene_number)
        print(generated_scene_num)

        sim = HabitatSimWithMap(scene_number, config)

        # Sample random maps & get the largest maps by levels
        recolored_topdown_map_list, topdown_map_list, height_list = get_entire_maps_by_levels(
            sim, config.DataConfig.METERS_PER_PIXEL
        )

        # Store the largest maps
        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            topdown_map = topdown_map_list[i]

            cv2.imwrite(f"./data/topdown/{scene_number}_{i}.bmp", topdown_map)
            cv2.imwrite(f"./data/recolored_topdown/{scene_number}_{i}.bmp", recolored_topdown_map)

            height[f"{scene_number}_{i}"] = float(height_list[i])

            with open(height_json_path, "w") as label_json:  # pylint: disable=unspecified-encoding
                json.dump(height, label_json, indent=4)

        sim.close()
        generated_scene_num = generated_scene_num + 1
