import argparse
import json
import os

import cv2
from habitat.utils.visualizations import maps
import habitat_sim

from config.env_config import ActionConfig, Cam360Config, DisplayConfig, PathConfig
from utils.habitat_utils import (
    display_map,
    display_opencv_cam,
    get_closest_map,
    get_map_from_database,
    init_map_display,
    init_opencv_cam,
    initialize_sim,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_train.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output/observations")
    parser.add_argument("--save-all", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json
    output_path = args.output_path
    is_save_all = args.save_all

    os.makedirs(output_path, exist_ok=True)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    # for scene_number in scene_list:
    scene_number = scene_list[0]
    sim = initialize_sim(scene_number, Cam360Config, ActionConfig, PathConfig)
    agent = sim.initialize_agent(0)
    recolored_topdown_map_list, _, _ = get_map_from_database(scene_number, height_data)

    agent_state = habitat_sim.AgentState()
    nav_point = sim.pathfinder.get_random_navigable_point()
    agent_state.position = nav_point  # world space
    agent.set_state(agent_state)

    img_id = 0

    if DisplayConfig.DISPLAY_PATH_MAP:
        init_map_display()

    if DisplayConfig.DISPLAY_OBSERVATION:
        init_opencv_cam()

    while True:
        observations = sim.get_sensor_observations()
        color_img = cv2.cvtColor(observations["color_360_sensor"], cv2.COLOR_BGR2RGB)

        if DisplayConfig.DISPLAY_OBSERVATION:
            key = display_opencv_cam(color_img)

        current_state = agent.get_state()
        position = current_state.position

        if key == ord("w"):
            action = "move_forward"
        if key == ord("s"):
            action = "move_backward"
        if key == ord("a"):
            action = "turn_left"
        if key == ord("d"):
            action = "turn_right"
        if is_save_all:
            cv2.imwrite(output_path + os.sep + f"{img_id:06d}.jpg", color_img)
            img_id = img_id + 1
        if key == ord("o"):
            if is_save_all:
                pass
            else:
                print("save image")
                cv2.imwrite(output_path + os.sep + f"{img_id:06d}.jpg", color_img)
                img_id = img_id + 1
                continue
        if key == ord("q"):
            break

        sim.step(action)

        if DisplayConfig.DISPLAY_PATH_MAP:
            recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)
            node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            display_map(recolored_topdown_map, key_points=[transposed_point])
