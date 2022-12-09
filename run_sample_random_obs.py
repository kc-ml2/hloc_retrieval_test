import argparse
import json
import os

import cv2
from habitat.utils.visualizations import maps

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    output_path = args.output_path
    height_json_path = args.map_height_json

    # Open files
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    if scene_index is not None:
        scene_list = [scene_list[scene_index]]

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            observation_path = os.path.join(output_path, f"observation_{scene_number}")
            test_sample_path = os.path.join(observation_path, f"test_sample_{level}")
            os.makedirs(test_sample_path, exist_ok=True)

            pos_record_json = os.path.join(observation_path, f"pos_record_test_sample_{level}.json")
            pos_record = {}
            pos_record.update({"scene_number": scene_number})
            pos_record.update({"level": level})

            for k in range(TestConstant.NUM_SAMPLING_PER_LEVEL):
                while True:
                    random_point, random_rotation = sim.set_random_position()
                    if sim.closest_level == level:
                        break

                observations = sim.get_cam_observations()
                color_img = observations["all_view"]

                cv2.imwrite(test_sample_path + os.sep + f"{k:06d}.jpg", color_img)

                node_point = maps.to_grid(random_point[2], random_point[0], sim.recolored_topdown_map.shape[0:2], sim)
                sim_pos = {f"{k:06d}_sim": [[float(pos) for pos in random_point], random_rotation]}
                grid_pos = {f"{k:06d}_grid": [int(pnt) for pnt in node_point]}
                pos_record.update(sim_pos)
                pos_record.update(grid_pos)

            with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                json.dump(pos_record, record_json, indent=4)

        sim.close()
