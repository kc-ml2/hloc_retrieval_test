import argparse
import os
import random

import cv2
import habitat_sim
import networkx as nx
import numpy as np

from config.env_config import ActionConfig, CamNormalConfig, DataConfig, DisplayOnConfig, OutputConfig, PathConfig
from utils.habitat_utils import display_map, get_entire_maps_by_levels, init_map_display, make_cfg
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_dense_topology,
    convert_to_visual_binarymap,
    display_graph,
    generate_map_image,
    remove_isolated_area,
    visualize_path,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene = PathConfig.SCENE_DIRECTORY + os.sep + scene_number + os.sep + scene_number + ".glb"

        sim_settings = {
            "width": CamNormalConfig.WIDTH,
            "height": CamNormalConfig.HEIGHT,
            "scene": scene,
            "default_agent": 0,
            "sensor_height": CamNormalConfig.SENSOR_HEIGHT,
            "color_sensor": CamNormalConfig.RGB_SENSOR,
            "color_360_sensor": CamNormalConfig.RGB_360_SENSOR,
            "depth_sensor": CamNormalConfig.DEPTH_SENSOR,
            "semantic_sensor": CamNormalConfig.SEMANTIC_SENSOR,
            "seed": 1,
            "enable_physics": False,
            "forward_amount": ActionConfig.FORWARD_AMOUNT,
            "backward_amount": ActionConfig.BACKWARD_AMOUNT,
            "turn_left_amount": ActionConfig.TURN_LEFT_AMOUNT,
            "turn_right_amount": ActionConfig.TURN_RIGHT_AMOUNT,
        }

        cfg = make_cfg(sim_settings)
        sim = habitat_sim.Simulator(cfg)

        # The randomness is needed when choosing the actions
        random.seed(sim_settings["seed"])
        sim.seed(sim_settings["seed"])
        pathfinder_seed = 1

        # Set agent state
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.5, 0.0])  # world space
        agent.set_state(agent_state)

        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized")
        sim.pathfinder.seed(pathfinder_seed)

        recolored_topdown_map_list, topdown_map_list, _ = get_entire_maps_by_levels(sim, DataConfig.METERS_PER_PIXEL)

        if DisplayOnConfig.DISPLAY_PATH_MAP:
            init_map_display(window_name="colored_map")
            init_map_display(window_name="visual_binary_map")

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]
            visual_binary_map = convert_to_visual_binarymap(topdown_map)

            if DataConfig.REMOVE_ISOLATED:
                topdown_map = remove_isolated_area(topdown_map)

            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            node_list = None
            while node_list is None:
                try:
                    start = random.choice(list(graph.nodes))
                    end = random.choice(list(graph.nodes))
                    node_list = nx.shortest_path(graph, start, end)
                except nx.NetworkXNoPath:
                    pass

            if DisplayOnConfig.DISPLAY_PATH_MAP:
                print("Displaying recolored map:")
                display_map(recolored_topdown_map, window_name="colored_map", wait_for_key=True)
                print("Displaying visual binary map:")
                display_map(visual_binary_map, window_name="visual_binary_map", wait_for_key=True)
                print("Displaying graph:")
                display_graph(
                    visual_binary_map,
                    graph,
                    window_name="original graph",
                    node_only=DataConfig.IS_DENSE_GRAPH,
                    wait_for_key=True,
                )
                print("Displaying path:")
                visualize_path(visual_binary_map, graph, node_list, wait_for_key=True)

            if OutputConfig.SAVE_PATH_MAP:
                map_img = generate_map_image(
                    visual_binary_map, graph, node_only=DataConfig.IS_DENSE_GRAPH, line_edge=False
                )
                cv2.imwrite(f"./output/test/{scene_number}_{i}.bmp", map_img)

        cv2.destroyAllWindows()
        sim.close()
