import argparse
import random

import cv2
import habitat_sim
import numpy as np

from utils.habitat_utils import display_map, get_entire_maps_by_levels, init_map_display, make_cfg
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_topology,
    convert_to_visual_binarymap,
    display_graph,
    generate_map_image,
    prune_graph,
    remove_isolated_area,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = True
    rgb_360_sensor = False
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1
    display_path_map = False
    save_path_map = True
    erode_path_map = False
    remove_isolated = True

    check_radius = 3
    prune_iteration = 0
    noise_removal_threshold = 1000

    kernel = np.ones((5, 5), np.uint8)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
        scene = scene_directory + scene_number + "/" + scene_number + ".glb"

        sim_settings = {
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
            "scene": scene,  # Scene path
            "default_agent": 0,
            "sensor_height": 0,  # Height of sensors in meters
            "color_sensor": rgb_sensor,  # RGB sensor
            "color_360_sensor": rgb_360_sensor,
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
            "forward_amount": 0.25,
            "backward_amount": 0.25,
            "turn_left_amount": 5.0,
            "turn_right_amount": 5.0,
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

        recolored_topdown_map_list, topdown_map_list = get_entire_maps_by_levels(sim, meters_per_pixel)

        if display_path_map:
            init_map_display(window_name="colored_map")
            init_map_display(window_name="visual_binary_map")

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]
            visual_binary_map = convert_to_visual_binarymap(topdown_map)

            if erode_path_map:
                topdown_map = cv2.erode(topdown_map, kernel, iterations=1)
                topdown_map = cv2.dilate(topdown_map, kernel, iterations=1)

            if remove_isolated:
                topdown_map = remove_isolated_area(topdown_map)

            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_topology(binary_map)

            if display_path_map:
                print("Displaying recolored map:")
                display_map(recolored_topdown_map, window_name="colored_map", wait_for_key=True)
                print("Displaying visual binary map:")
                display_map(visual_binary_map, window_name="visual_binary_map", wait_for_key=True)
                print("Displaying graph:")
                display_graph(visual_binary_map, graph, window_name="original graph", wait_for_key=True)

            for _ in range(prune_iteration):
                print("Pruning graph")
                graph = prune_graph(graph, topdown_map, check_radius)

            if display_path_map and prune_iteration:
                print("Displaying pruned graph:")
                display_graph(visual_binary_map, graph, window_name="pruned_graph", wait_for_key=True, line_edge=True)

            if save_path_map:
                map_img = generate_map_image(visual_binary_map, graph, line_edge=False)
                cv2.imwrite(f"./output/pruned/{scene_number}_{i}.bmp", map_img)

        cv2.destroyAllWindows()
        sim.close()
