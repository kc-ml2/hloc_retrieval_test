import argparse
import json
import os

import cv2
from habitat.utils.visualizations import maps
import habitat_sim

from algorithms.yolo import Yolo
from config.env_config import ActionConfig, Cam360Config, PathConfig
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
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output/observations")
    parser.add_argument("--pos-record-json", default="./output/pos_record.json")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--save-except-rotation", action="store_true")
    parser.add_argument("--detection", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    output_path = args.output_path
    pos_record_json = args.pos_record_json
    is_save_all = args.save_all
    is_save_except_rotation = args.save_except_rotation
    is_detection = args.detection

    check_arg = is_save_all + is_save_except_rotation
    if check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    if is_detection:
        yolo = Yolo()

    os.makedirs(output_path, exist_ok=True)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    if scene_index >= len(scene_list):
        raise IndexError(f"Scene list index out of range. The range is from 0 to {len(scene_list) - 1}")

    scene_number = scene_list[scene_index]
    sim = initialize_sim(scene_number, Cam360Config, ActionConfig, PathConfig)
    agent = sim.initialize_agent(0)
    recolored_topdown_map_list, _, _ = get_map_from_database(scene_number, height_data)

    agent_state = habitat_sim.AgentState()
    nav_point = sim.pathfinder.get_random_navigable_point()
    agent_state.position = nav_point  # world space
    agent.set_state(agent_state)

    img_id = 0
    pos_record = {}
    pos_record.update({"scene_number": scene_number})

    # Initialize opencv display window
    init_map_display()
    init_opencv_cam()

    while True:
        observations = sim.get_sensor_observations()
        color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

        current_state = agent.get_state()
        position = current_state.position

        recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)
        node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)

        display_map(recolored_topdown_map, key_points=[node_point])

        if is_detection:
            detect_img = yolo.detect_img(color_img)
            key = display_opencv_cam(detect_img)
        else:
            key = display_opencv_cam(color_img)

        if key == ord("w"):
            action = "move_forward"
        if key == ord("s"):
            action = "move_backward"
        if key == ord("a"):
            action = "turn_left"
        if key == ord("d"):
            action = "turn_right"
        if key == ord("q"):
            break

        if is_save_all:
            cv2.imwrite(output_path + os.sep + f"{img_id:06d}.jpg", color_img)
            sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
            grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
            pos_record.update(sim_pos)
            pos_record.update(grid_pos)
            img_id = img_id + 1
        if is_save_except_rotation:
            if key == ord("w") or key == ord("s"):
                cv2.imwrite(output_path + os.sep + f"{img_id:06d}.jpg", color_img)
                sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
                grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
                pos_record.update(sim_pos)
                pos_record.update(grid_pos)
                img_id = img_id + 1
        if key == ord("o"):
            if is_save_all or is_save_except_rotation:
                pass
            else:
                print("save image")
                cv2.imwrite(output_path + os.sep + f"{img_id:06d}.jpg", color_img)
                sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
                grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
                pos_record.update(sim_pos)
                pos_record.update(grid_pos)
                img_id = img_id + 1
                continue

        sim.step(action)

    with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
        json.dump(pos_record, record_json, indent=4)
