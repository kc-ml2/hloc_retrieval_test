import argparse
import json
import os
import random

import cv2
import networkx as nx

from relocalization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import open_env_related_files
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_dense_topology,
    get_one_random_directed_adjacent_node,
    remove_isolated_area,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/singleview_90FOV.py")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    height_json_path = args.map_height_json
    is_train = args.train
    is_valid = args.valid
    is_test = args.test

    config = load_config_module(module_name)

    # Set dataset output path according to flag
    if is_train:
        scene_list_file = "./data/scene_list_train.txt"
        output_image_path = config.TRAIN_IMAGE_PATH
        label_json_file = config.TRAIN_LABEL_PATH
    if is_valid:
        scene_list_file = "./data/scene_list_val_unseen.txt"
        output_image_path = config.VALID_IMAGE_PATH
        label_json_file = config.VALID_LABEL_PATH
    if is_test:
        scene_list_file = "./data/scene_list_test.txt"
        output_image_path = config.TEST_IMAGE_PATH
        label_json_file = config.TEST_LABEL_PATH

    # Check if there is only one flag via train, valid, and test
    check_arg = is_train + is_test + is_valid
    if check_arg == 0 or check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    os.makedirs(output_image_path, exist_ok=True)

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path)

    label = {}
    total_scene_num = 0
    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)

        print("total scene: ", total_scene_num)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)
            topdown_map = sim.topdown_map_list[level]
            if config.DataConfig.REMOVE_ISOLATED:
                topdown_map = remove_isolated_area(topdown_map)
            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            if len(list(graph.nodes)) == 0:
                continue

            # Make ratio of positive & negative samples to be 5:5 with "random() < 0.5"
            for k in range(config.TrainingConstant.NUM_SAMPLING_PER_LEVEL):
                # Positive samples
                if random.random() < 0.5:
                    y = 1
                    error_code = None
                    while error_code != 0:
                        try:
                            start = random.choice(list(graph.nodes))
                        except IndexError:
                            break
                        # Get step between 1 ~ max positive distance
                        step = random.randint(1, config.TrainingConstant.POSITIVE_SAMPLE_DISTANCE)
                        previous_node = None
                        current_node = start

                        for _ in range(step):
                            # Randomly select a direction. Ignore visited (previous) node
                            next_node, error_code = get_one_random_directed_adjacent_node(
                                graph, current_node, previous_node
                            )
                            previous_node = current_node
                            current_node = next_node
                    end = next_node
                # Negative samples
                else:
                    y = 0
                    step = 0
                    max_iteration = 0
                    min_step = (
                        config.TrainingConstant.POSITIVE_SAMPLE_DISTANCE
                        * config.TrainingConstant.NEGATIVE_SAMPLE_MULTIPLIER
                    )
                    while step < min_step:
                        try:
                            start = random.choice(list(graph.nodes))
                            end = random.choice(list(graph.nodes))
                            step = len(nx.shortest_path(graph, start, end))
                            max_iteration = max_iteration + 1
                            if max_iteration > 20:
                                break
                        except nx.NetworkXNoPath:
                            continue
                        except IndexError:
                            break

                node_list = [start, end]

                if config.CamConfig.NUM_CAMERA > 1 and not config.CamConfig.IMAGE_CONCAT:
                    view_attr = "front_view"
                else:
                    view_attr = "all_view"

                random_rotation = 0
                for j, node in enumerate(node_list):
                    if view_attr == "front_view" and j == 1:
                        random_rotation = random_rotation + random.randint(-30, 30)
                        pos, random_rotation = sim.set_state_from_grid(
                            graph.nodes[node]["o"], level, rotation=random_rotation
                        )
                    if sim.cam_config.NUM_CAMERA == 1 and not sim.cam_config.RGB_360_SENSOR and j == 1:
                        random_rotation = random_rotation + random.randint(-30, 30)
                        pos, random_rotation = sim.set_state_from_grid(
                            graph.nodes[node]["o"], level, rotation=random_rotation
                        )
                    else:
                        pos, random_rotation = sim.set_state_from_grid(graph.nodes[node]["o"], level)
                    observations = sim.get_cam_observations()
                    color_img = observations[view_attr]

                    cv2.imwrite(output_image_path + os.sep + f"{scene_number}_{level:06d}_{k:06d}_{j}.jpg", color_img)
                    label_pos = {
                        f"{scene_number}_{level:06d}_{k:06d}_{j}": [
                            [float(pos[1]), float(sim.height_list[level]), float(pos[0])],
                            random_rotation,
                        ]
                    }
                    label.update(label_pos)

                label_similarity = {f"{scene_number}_{level:06d}_{k:06d}_similarity": y}
                label.update(label_similarity)

        total_scene_num = total_scene_num + 1
        sim.close()

    with open(label_json_file, "w") as label_json:  # pylint: disable=unspecified-encoding
        json.dump(label, label_json, indent=4)
