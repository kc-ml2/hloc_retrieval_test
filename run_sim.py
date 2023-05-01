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
    parser.add_argument("--config", default="config/singleview_90FOV.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--save-except-rotation", action="store_true")
    parser.add_argument("--localization", action="store_true")
    parser.add_argument("--load-model", default="./model_weights/model.20221129-125905.4view.weights.hdf5")
    parser.add_argument("--map-obs-path", default="./output")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    output_path = args.output_path
    is_save_all = args.save_all
    is_save_except_rotation = args.save_except_rotation
    is_localization = args.localization

    check_arg = is_save_all + is_save_except_rotation
    if check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    config = load_config_module(module_name)
    map_obs_path = config.PathConfig.LOCALIZATION_TEST_PATH
    loaded_model = config.PathConfig.MODEL_WEIGHTS

    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    # Load pre-trained model
    if is_localization:
        import tensorflow as tf

        from network.resnet import ResnetBuilder  # pylint: disable=ungrouped-imports
        from relocalization.localization import Localization  # pylint: disable=ungrouped-imports

        ResnetBuilder.restrict_gpu_memory(config)

        with tf.device(f"/device:GPU:{config.PathConfig.GPU_ID}"):
            model, top_network, bottom_network = ResnetBuilder.load_siamese_model(loaded_model, config.NetworkConstant)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            observation_path, pos_record_json = make_output_path(output_path, scene_number)

            img_id = 0
            pos_record = {}
            pos_record.update({"scene_number": scene_number})

            # Sample initial agent position from topdown map grey area
            # This is for fixing height of agent position
            binary_topdown_map = sim.topdown_map_list[level]
            explorable_area_index = list(zip(*np.where(binary_topdown_map == 1)))
            grid_pos = random.sample(explorable_area_index, 1)[0]
            sim.set_state_from_grid(grid_pos, level)

            if is_localization:
                # Set file path
                current_map_dir = os.path.join(
                    map_obs_path, f"observation_{scene_number}", f"map_node_observation_level_{sim.closest_level}"
                )
                # Initialize localization instance
                localization = Localization(
                    config,
                    top_network,
                    bottom_network,
                    current_map_dir,
                    binary_topdown_map=binary_topdown_map,
                )

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
                if previous_level != current_level and is_localization:
                    current_map_dir = os.path.join(
                        map_obs_path, f"observation_{scene_number}", f"map_node_observation_level_{current_level}"
                    )
                    localization = Localization(
                        config,
                        top_network,
                        bottom_network,
                        current_map_dir,
                        binary_topdown_map=sim.topdown_map_list[current_level],
                    )

                # Execute localization
                if is_localization:
                    obs_embedding = localization.calculate_embedding_from_observation(color_img)
                    localization_result = localization.localize_with_observation(obs_embedding)
                    map_image = localization.visualize_on_map(map_image, localization_result)

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
                    save_observation(color_img, observation_path, img_id, pos_record, position, node_point)
                    img_id = img_id + 1
                # Save observation only when forward & backward movement
                if is_save_except_rotation:
                    if key == ord("w") or key == ord("s"):
                        save_observation(color_img, observation_path, img_id, pos_record, position, node_point)
                        img_id = img_id + 1
                # Save observation when "o" key input
                if key == ord("o"):
                    if is_save_all or is_save_except_rotation:
                        pass
                    else:
                        save_observation(color_img, observation_path, img_id, pos_record, position, node_point)
                        img_id = img_id + 1
                        continue

                sim.step(action)

            file_saved = os.listdir(observation_path)
            if file_saved:
                with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                    json.dump(pos_record, record_json, indent=4)
            else:
                os.rmdir(observation_path)

        sim.close()

        if key == ord("q"):
            break
