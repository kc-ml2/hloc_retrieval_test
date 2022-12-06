import argparse
import json
import os
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np
from scipy.spatial.transform import Rotation

from config.env_config import ActionConfig, CamFourViewConfig, DataConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_dense_topology,
    convert_to_visual_binarymap,
    remove_isolated_area,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--map-debug", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    output_path = args.output_path
    height_json_path = args.map_height_json
    map_debug = args.map_debug

    # Open files
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    if scene_index:
        scene_list = scene_list[scene_index]

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            topdown_map = sim.topdown_map_list[level]
            visual_binary_map = convert_to_visual_binarymap(topdown_map)

            if DataConfig.REMOVE_ISOLATED:
                topdown_map = remove_isolated_area(topdown_map)

            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            if len(list(graph.nodes)) == 0:
                continue

            if map_debug:
                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
                for current_node in graph.nodes():
                    for node_id in graph.nodes():
                        draw_point_from_node(map_image, graph, node_id)
                    highlight_point_from_node(map_image, graph, current_node, (0, 255, 0))

                    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("map", 1152, 1152)
                    cv2.imshow("map", map_image)
                    cv2.waitKey()

            observation_path = os.path.join(output_path, f"observation_{scene_number}")
            map_obs_result_path = os.path.join(observation_path, f"map_node_observation_level_{level}")
            os.makedirs(map_obs_result_path, exist_ok=True)

            for node_id in graph.nodes():
                pos = maps.from_grid(
                    int(graph.nodes[node_id]["o"][0]),
                    int(graph.nodes[node_id]["o"][1]),
                    recolored_topdown_map.shape[0:2],
                    sim,
                    sim.pathfinder,
                )

                # Set random rotation at current position
                agent_state = habitat_sim.AgentState()
                agent_state.position = np.array([pos[1], sim.height_list[level], pos[0]])
                random_rotation = random.randint(0, 359)
                r = Rotation.from_euler("y", random_rotation, degrees=True)
                agent_state.rotation = r.as_quat()
                sim.agent.set_state(agent_state)

                observations = sim.get_cam_observations()
                color_img = observations["all_view"]

                cv2.imwrite(map_obs_result_path + os.sep + f"{node_id}.bmp", color_img)

        sim.close()
