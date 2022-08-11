import argparse
import random

import cv2
import habitat_sim
import numpy as np

from utils.habitat_utils import get_entire_maps_by_levels, make_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1

    generated_scene_num = 0

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
        scene = scene_directory + scene_number + "/" + scene_number + ".glb"
        print(scene_number)
        print(generated_scene_num)

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
        position = sim.pathfinder.get_random_navigable_point()

        recolored_topdown_map_list, topdown_map_list = get_entire_maps_by_levels(sim, meters_per_pixel)

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            topdown_map = topdown_map_list[i]

            cv2.imwrite(f"./output/topdown/{scene_number}_{i}.bmp", topdown_map)
            cv2.imwrite(f"./output/recolored_topdown/{scene_number}_{i}.bmp", recolored_topdown_map)

        sim.close()
        generated_scene_num = generated_scene_num + 1
