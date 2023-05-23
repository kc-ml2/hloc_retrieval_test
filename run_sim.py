import argparse
import json
import os
import random

import cv2
from habitat.utils.visualizations import maps
import numpy as np

from relocalization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import (
    display_map,
    display_opencv_cam,
    init_map_display,
    init_opencv_cam,
    make_output_path,
    open_env_related_files,
    save_observation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/concat_fourview_69FOV_HD.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--save-except-rotation", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    output_path = args.output_path
    is_save_all = args.save_all
    is_save_except_rotation = args.save_except_rotation

    check_arg = is_save_all + is_save_except_rotation
    if check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    config = load_config_module(module_name)
    image_dir = config.PathConfig.LOCALIZATION_TEST_PATH

    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            image_dir_by_scene, pos_record_json = make_output_path(
                output_path, scene_number, config.PathConfig.POS_RECORD_FILE_PREFIX
            )

            img_id = 0
            pos_record = {}
            pos_record.update({"scene_number": scene_number})

            # Sample initial agent position from topdown map grey area
            # This is for fixing height of agent position
            binary_topdown_map = sim.topdown_map_list[level]
            explorable_area_index = list(zip(*np.where(binary_topdown_map == 1)))
            grid_pos = random.sample(explorable_area_index, 1)[0]
            sim.set_state_from_grid(grid_pos, level)

            # Initialize opencv display window
            init_map_display()
            init_opencv_cam()

            while True:
                # Get current position & set it to unified height
                current_state = sim.agent.get_state()
                current_state.position[1] = sim.height_list[level]
                position = current_state.position
                sim.agent.set_state(current_state)

                # Get camera observation
                observations = sim.get_cam_observations()
                color_img = observations["all_view"]

                key = display_opencv_cam(color_img)

                # Update map data
                previous_level = sim.closest_level
                sim.update_closest_map(position)
                map_image = cv2.cvtColor(sim.recolored_topdown_map, cv2.COLOR_GRAY2BGR)

                # If level is changed, re-initialize localization instance
                current_level = sim.closest_level

                node_point = maps.to_grid(position[2], position[0], sim.recolored_topdown_map.shape[0:2], sim)
                display_map(map_image, key_points=[node_point])

                # Set action according to key input
                if key == ord("w"):
                    action = "move_forward"
                if key == ord("s"):
                    action = "move_backward"
                if key == ord("a"):
                    action = "turn_left"
                if key == ord("d"):
                    action = "turn_right"
                if key == ord("n"):
                    break
                if key == ord("q"):
                    break

                # Save observation & position record according to the flag
                # Save observation every step
                if is_save_all:
                    save_observation(color_img, image_dir_by_scene, img_id, pos_record, position, node_point)
                    img_id = img_id + 1
                # Save observation only when forward & backward movement
                if is_save_except_rotation:
                    if key == ord("w") or key == ord("s"):
                        save_observation(color_img, image_dir_by_scene, img_id, pos_record, position, node_point)
                        img_id = img_id + 1
                # Save observation when "o" key input
                if key == ord("o"):
                    if is_save_all or is_save_except_rotation:
                        pass
                    else:
                        save_observation(color_img, image_dir_by_scene, img_id, pos_record, position, node_point)
                        img_id = img_id + 1
                        continue

                sim.step(action)

            file_saved = os.listdir(image_dir_by_scene)
            if file_saved:
                with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                    json.dump(pos_record, record_json, indent=4)
            else:
                os.rmdir(image_dir_by_scene)

        sim.close()

        if key == ord("q"):
            break
