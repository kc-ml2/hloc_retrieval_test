"""
Original code:
https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb

The original code is released under the MIT license.

Modified by KC-ML2.
"""


import argparse
import os
from pathlib import Path
import random
import time

import cv2
from habitat.utils.visualizations import maps
from hloc import extract_features, pairs_from_retrieval
import numpy as np

from global_localization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
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
    parser.add_argument("--config", default="config/singleview_90FOV_HD_interactive.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./output")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    output_path = args.output_path

    config = load_config_module(module_name)
    image_dir = config.PathConfig.LOCALIZATION_TEST_PATH

    retrieval_conf = extract_features.confs["netvlad"]
    model = extract_features.load_model(retrieval_conf)

    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            image_dir_by_scene, _ = make_output_path(
                output_path, scene_number, config.PathConfig.POS_RECORD_FILE_PREFIX
            )

            # Initialize map image
            current_state = sim.agent.get_state()
            current_state.position[1] = sim.height_list[level]
            position = current_state.position
            sim.update_closest_map(position)
            map_image = cv2.cvtColor(sim.recolored_topdown_map, cv2.COLOR_GRAY2BGR)
            topdown_map = sim.topdown_map_list[level]
            graph = topdown_map_to_graph(topdown_map, config.DataConfig.REMOVE_ISOLATED)
            for node_id in graph.nodes():
                draw_point_from_node(map_image, graph, node_id)

            scene_dirname = f"observation_{scene_number}"
            netvlad_path = os.path.join(config.PathConfig.HLOC_DB_ONLY_OUTPUT, scene_dirname, f"{level}")
            netvlad_outputs = Path(netvlad_path)
            interactive_path = Path(os.path.join(netvlad_outputs, "query.h5"))
            db_feature_path = Path(os.path.join(netvlad_outputs, "global-feats-netvlad.h5"))
            retrieval_pairs = Path("./output/interactive-pair.txt")

            # Set file path
            image_dir_prefix = os.path.join(image_dir, scene_dirname)
            map_index = f"{config.PathConfig.MAP_DIR_PREFIX}_{level}"
            query_index = f"interactive_{level}"
            map_obs_dir = os.path.join(image_dir_prefix, map_index)
            query_dir = os.path.join(image_dir_prefix, query_index)
            os.makedirs(query_dir, exist_ok=True)
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            map_obs_list = [os.path.join(scene_dirname, map_index, file) for file in sorted_map_obs_file]
            query_list = [os.path.join(scene_dirname, query_index, "query.jpg")]

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
                start = time.time()

                # Get current position & set it to unified height
                current_state = sim.agent.get_state()
                current_state.position[1] = sim.height_list[level]
                position = current_state.position
                sim.agent.set_state(current_state)

                # Get camera observation
                observations = sim.get_cam_observations()
                color_img = observations["front_view"]

                # Save current image for netvlad inference
                query_image_path = os.path.join(image_dir, query_list[0])
                cv2.imwrite(query_image_path, color_img)

                extract_features.run_model(
                    model,
                    retrieval_conf,
                    Path(image_dir),
                    feature_path=interactive_path,
                    image_list=query_list,
                    overwrite=True,
                )

                pairs_from_retrieval.main(
                    interactive_path,
                    retrieval_pairs,
                    num_matched=10,
                    query_list=query_list,
                    db_list=map_obs_list,
                    db_descriptors=db_feature_path,
                )

                with open(retrieval_pairs) as f:  # pylint: disable=unspecified-encoding
                    closest_node = int(f.readline()[-13:-7])
                    print(closest_node)
                f.close()

                copied_map = map_image.copy()
                draw_point_from_node(copied_map, graph, closest_node, color=(0, 255, 255), radius=1)

                node_point = maps.to_grid(position[2], position[0], sim.recolored_topdown_map.shape[0:2], sim)
                display_map(copied_map, key_points=[node_point])

                key = display_opencv_cam(color_img)

                # Update map data
                # If level is changed, re-initialize localization instance
                previous_level = sim.closest_level
                sim.update_closest_map(position)
                current_level = sim.closest_level

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

                sim.step(action)

                end = time.time()
                print(end - start)

        sim.close()

        if key == ord("q"):
            break
