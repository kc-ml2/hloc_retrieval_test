import argparse
import json
import os
import random

import cv2
import habitat_sim
import numpy as np

from config.env_config import ActionConfig, CamNormalConfig, DataConfig
from utils.habitat_utils import get_entire_maps_by_levels, make_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_total.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json

    os.makedirs("./data/topdown/")
    os.makedirs("./data/recolored_topdown/")

    generated_scene_num = 0
    height = {}

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
        scene = scene_directory + scene_number + "/" + scene_number + ".glb"
        print(scene_number)
        print(generated_scene_num)

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
        position = sim.pathfinder.get_random_navigable_point()

        recolored_topdown_map_list, topdown_map_list, height_list = get_entire_maps_by_levels(
            sim, DataConfig.METERS_PER_PIXEL
        )

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            topdown_map = topdown_map_list[i]

            cv2.imwrite(f"./data/topdown/{scene_number}_{i}.bmp", topdown_map)
            cv2.imwrite(f"./data/recolored_topdown/{scene_number}_{i}.bmp", recolored_topdown_map)

            height[f"{scene_number}_{i}"] = float(height_list[i])

            with open(height_json_path, "w") as label_json:  # pylint: disable=unspecified-encoding
                json.dump(height, label_json, indent=4)

        sim.close()
        generated_scene_num = generated_scene_num + 1
