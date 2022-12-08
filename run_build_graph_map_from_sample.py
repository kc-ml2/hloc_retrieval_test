import argparse
import json
import os

import cv2
import numpy as np

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, DataConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import convert_to_binarymap, convert_to_dense_topology, remove_isolated_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    parser.add_argument("--save-img", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path
    save_img = args.save_img

    # Open files
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    if scene_index is not None:
        scene_list = [scene_list[scene_index]]

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    # Main loop
    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            topdown_map = sim.topdown_map_list[level]
            if DataConfig.REMOVE_ISOLATED:
                topdown_map = remove_isolated_area(topdown_map)
            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            # Set file path
            result_cache = os.path.join(observation_path, f"similarity_matrix_test_sample_{level}.npy")
            pos_record = os.path.join(observation_path, f"pos_record_test_sample_{level}.json")

            with open(pos_record, "r") as f:  # pylint: disable=unspecified-encoding
                pos_record = json.load(f)

            with open(result_cache, "rb") as f:  # pylint: disable=unspecified-encoding
                similarity_matrix = np.load(f)

            if args.save_img:
                visualization_result_path = os.path.join(observation_path, f"localization_visualize_result_{level}")
                os.makedirs(visualization_result_path, exist_ok=True)

            for i in range(TestConstant.NUM_SAMPLING_PER_LEVEL):
                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)

                for node in graph.nodes():
                    draw_point_from_node(map_image, graph, node)

                grid_pos = pos_record[f"{i:06d}_grid"]
                sim_pos = pos_record[f"{i:06d}_sim"]
                graph.add_node("current")
                graph.nodes()["current"]["o"] = grid_pos

                agent_height = sim_pos[0][1]
                map_height = sim.height_list[level]
                print("Sample No.: ", i)
                print("Agent Height: ", agent_height, "    Map Height: ", map_height)

                similarity = similarity_matrix[:, i]
                node_with_max_value = np.argmax(similarity)
                similarity_set = similarity > TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
                print("Max value: ", similarity[node_with_max_value], "   Node: ", node_with_max_value)

                for idx, upper in enumerate(similarity_set):
                    if upper:
                        highlight_point_from_node(map_image, graph, idx, (0, 0, 122))

                highlight_point_from_node(map_image, graph, node_with_max_value, (255, 255, 0))
                highlight_point_from_node(map_image, graph, "current", (0, 255, 0))

                cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("localization", 1152, 1152)
                cv2.imshow("localization", map_image)

                if args.save_img:
                    cv2.imwrite(visualization_result_path + os.sep + f"{i:06d}.jpg", map_image)

                cv2.waitKey()
