import argparse
import json
import os
import random

import cv2
import networkx as nx

from config.env_config import ActionConfig, CamNormalConfig, DataConfig, DisplayConfig, OutputConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import display_map, init_map_display
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
    parser.add_argument("--scene-list-file", default="./data/scene_list_train.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output/skeleton")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json
    output_path = args.output_path

    if OutputConfig.SAVE_PATH_MAP:
        os.makedirs(output_path, exist_ok=True)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamNormalConfig, ActionConfig, PathConfig, height_data)

        if DisplayConfig.DISPLAY_PATH_MAP:
            init_map_display(window_name="colored_map")
            init_map_display(window_name="visual_binary_map")

        for i, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = sim.topdown_map_list[i]
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

            if DisplayConfig.DISPLAY_PATH_MAP:
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
                cv2.imwrite(output_path + os.sep + f"{scene_number}_{i}.bmp", map_img)

        cv2.destroyAllWindows()
        sim.close()
