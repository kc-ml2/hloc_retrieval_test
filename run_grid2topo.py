import argparse
import random

import cv2
import habitat_sim
import numpy as np

from utils.habitat_utils import display_map, get_entire_maps_by_levels, make_cfg
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_topology,
    convert_to_visual_binarymap,
    display_graph,
    generate_map_image,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1

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
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
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

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]

            print("Displaying recolored map:")
            display_map(recolored_topdown_map, window_name="colored_map", wait_for_key=True)

            visual_binary_map = convert_to_visual_binarymap(topdown_map)
            print("Displaying visual binary map:")
            display_map(visual_binary_map, wait_for_key=True)

            binary_map = convert_to_binarymap(topdown_map)
            skeletonized_map, graph = convert_to_topology(binary_map)

            print("Displaying skeleton map:")
            display_map(skeletonized_map, window_name="skeleton")
            print("Displaying graph:")
            display_graph(visual_binary_map, graph)

            map_img = generate_map_image(visual_binary_map, graph)

            cv2.imwrite(f"./output/medial/{scene_number}_{i}.jpg", map_img)

            sim.close()
