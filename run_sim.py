import argparse
import json
import os

import cv2
from habitat.utils.visualizations import maps

from config.env_config import ActionConfig, CamFourViewConfig, DataConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import (
    display_map,
    display_opencv_cam,
    draw_point_from_node,
    init_map_display,
    init_opencv_cam,
    make_output_path,
    open_env_related_files,
)
from utils.skeletonize_utils import topdown_map_to_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--save-except-rotation", action="store_true")
    parser.add_argument("--detection", action="store_true")
    parser.add_argument("--localization", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    output_path = args.output_path
    is_save_all = args.save_all
    is_save_except_rotation = args.save_except_rotation
    is_detection = args.detection
    is_localization = args.localization

    check_arg = is_save_all + is_save_except_rotation
    if check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data, is_detection)
        observation_path, pos_record_json = make_output_path(output_path, scene_number)

        img_id = 0
        graph = 0
        pos_record = {}
        pos_record.update({"scene_number": scene_number})

        # Initialize opencv display window
        init_map_display()
        init_opencv_cam()

        while True:
            # Get camera observation
            observations = sim.get_cam_observations()
            color_img = observations["all_view"]

            # Get current position
            current_state = sim.agent.get_state()
            position = current_state.position

            # Update map data
            previous_level = sim.closest_level
            sim.update_closest_map(position)
            current_level = sim.closest_level
            map_image = cv2.cvtColor(sim.recolored_topdown_map, cv2.COLOR_GRAY2BGR)

            # If level is changed or graph is not initialized, build graph
            if previous_level != current_level or graph == 0 and is_localization:
                graph = topdown_map_to_graph(sim.topdown_map_list[current_level], DataConfig.REMOVE_ISOLATED)

            if is_localization:
                for node in graph.nodes():
                    draw_point_from_node(map_image, graph, node)

            node_point = maps.to_grid(position[2], position[0], sim.recolored_topdown_map.shape[0:2], sim)
            display_map(map_image, key_points=[node_point])

            # Display observation. If YOLO is available, display object detection result
            if is_detection:
                detect_img = sim.detect_img(observations)
                key = display_opencv_cam(detect_img)
            else:
                key = display_opencv_cam(color_img)

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
                cv2.imwrite(observation_path + os.sep + f"{img_id:06d}.jpg", color_img)
                sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
                grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
                pos_record.update(sim_pos)
                pos_record.update(grid_pos)
                img_id = img_id + 1
            # Save observation only when forward & backward movement
            if is_save_except_rotation:
                if key == ord("w") or key == ord("s"):
                    cv2.imwrite(observation_path + os.sep + f"{img_id:06d}.jpg", color_img)
                    sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
                    grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
                    pos_record.update(sim_pos)
                    pos_record.update(grid_pos)
                    img_id = img_id + 1
            # Save observation when "o" key input
            if key == ord("o"):
                if is_save_all or is_save_except_rotation:
                    pass
                else:
                    print("save image")
                    cv2.imwrite(observation_path + os.sep + f"{img_id:06d}.jpg", color_img)
                    sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
                    grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
                    pos_record.update(sim_pos)
                    pos_record.update(grid_pos)
                    img_id = img_id + 1
                    continue

            sim.step(action)

        file_saved = os.listdir(observation_path)
        if is_save_all or is_save_except_rotation or file_saved:
            with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                json.dump(pos_record, record_json, indent=4)
        else:
            os.rmdir(observation_path)

        sim.close()

        if key == ord("q"):
            break
