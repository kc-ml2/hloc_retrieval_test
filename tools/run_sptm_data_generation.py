import argparse
import json
import os
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

from config.algorithm_config import TrainingConstant
from config.env_config import ActionConfig, Cam360Config, DataConfig, PathConfig
from utils.habitat_utils import get_map_from_database, initialize_sim
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_dense_topology,
    convert_to_visual_binarymap,
    get_one_random_directed_adjacent_node,
    remove_isolated_area,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    height_json_path = args.map_height_json
    is_train = args.train
    is_valid = args.valid
    is_test = args.test

    if is_train:
        scene_list_file = "./data/scene_list_train.txt"
        output_image_path = PathConfig.TRAIN_IMAGE_PATH
        label_json_file = PathConfig.TRAIN_LABEL_PATH
    if is_valid:
        scene_list_file = "./data/scene_list_val_unseen.txt"
        output_image_path = PathConfig.VALID_IMAGE_PATH
        label_json_file = PathConfig.VALID_LABEL_PATH
    if is_test:
        scene_list_file = "./data/scene_list_test.txt"
        output_image_path = PathConfig.TEST_IMAGE_PATH
        label_json_file = PathConfig.TEST_LABEL_PATH

    check_arg = is_train + is_test + is_valid
    if check_arg == 0 or check_arg >= 2:
        raise ValueError("Argument Error. Put only one flag.")

    os.makedirs(output_image_path, exist_ok=True)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    label = {}
    total_scene_num = 0
    for scene_number in scene_list:
        sim = initialize_sim(scene_number, Cam360Config, ActionConfig, PathConfig)
        agent = sim.initialize_agent(0)
        recolored_topdown_map_list, topdown_map_list, height_list = get_map_from_database(scene_number, height_data)

        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.5, 0.0])  # world space
        agent.set_state(agent_state)

        print("total scene: ", total_scene_num)

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]
            visual_binary_map = convert_to_visual_binarymap(topdown_map)

            if DataConfig.REMOVE_ISOLATED:
                topdown_map = remove_isolated_area(topdown_map)

            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            if len(list(graph.nodes)) == 0:
                continue

            for k in range(TrainingConstant.NUM_SAMPLING_PER_LEVEL):
                if random.random() < 0.5:
                    y = 1
                    error_code = None
                    while error_code != 0:
                        try:
                            start = random.choice(list(graph.nodes))
                        except IndexError:
                            break
                        step = random.randint(1, TrainingConstant.POSITIVE_SAMPLE_DISTANCE)
                        previous_node = None
                        current_node = start

                        for _ in range(step):
                            next_node, error_code = get_one_random_directed_adjacent_node(
                                graph, current_node, previous_node
                            )
                            previous_node = current_node
                            current_node = next_node
                    end = next_node
                else:
                    y = 0
                    step = 0
                    max_iteration = 0
                    min_step = TrainingConstant.POSITIVE_SAMPLE_DISTANCE * TrainingConstant.NEGATIVE_SAMPLE_MULTIPLIER
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

                for j, node in enumerate(node_list):
                    pos = maps.from_grid(
                        int(graph.nodes[node]["o"][0]),
                        int(graph.nodes[node]["o"][1]),
                        recolored_topdown_map.shape[0:2],
                        sim,
                        sim.pathfinder,
                    )
                    agent_state.position = np.array([pos[1], height_list[i], pos[0]])
                    random_rotation = random.randint(0, 359)
                    r = Rotation.from_euler("y", random_rotation, degrees=True)
                    agent_state.rotation = r.as_quat()

                    agent.set_state(agent_state)
                    observations = sim.get_sensor_observations()
                    color_img = cv2.cvtColor(observations["color_360_sensor"], cv2.COLOR_BGR2RGB)

                    cv2.imwrite(output_image_path + os.sep + f"{scene_number}_{i:06d}_{k:06d}_{j}.bmp", color_img)
                    label_pos = {
                        f"{scene_number}_{i:06d}_{k:06d}_{j}": [
                            [float(pos[1]), float(height_list[i]), float(pos[0])],
                            random_rotation,
                        ]
                    }
                    label.update(label_pos)

                label_similarity = {f"{scene_number}_{i:06d}_{k:06d}_similarity": y}
                label.update(label_similarity)

        total_scene_num = total_scene_num + 1

        cv2.destroyAllWindows()
        sim.close()

        with open(label_json_file, "w") as label_json:  # pylint: disable=unspecified-encoding
            json.dump(label, label_json, indent=4)
