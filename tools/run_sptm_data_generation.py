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

from algorithms.constants import TrainingConstant
from utils.habitat_utils import display_opencv_cam, init_opencv_cam, make_cfg
from utils.skeletonize_utils import (
    convert_to_binarymap,
    convert_to_dense_topology,
    convert_to_visual_binarymap,
    get_one_random_directed_adjacent_node,
    remove_isolated_area,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = False
    rgb_360_sensor = True
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1
    is_dense_graph = True
    remove_isolated = True

    display = False

    num_sampling_per_level = 400

    label_json_path = "./output/images/label.json"
    label = {}

    total_scene_num = 0
    recolored_directory = "./output/recolored_topdown/"
    topdown_directory = "./output/topdown/"
    height_json_path = "./output/map_height.json"
    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
        scene = scene_directory + scene_number + "/" + scene_number + ".glb"

        num_levels = 0
        for root, dir, files in os.walk(topdown_directory):
            for file in files:
                if scene_number in file:
                    num_levels = num_levels + 1

        recolored_topdown_map_list = []
        topdown_map_list = []
        height_list = []
        for level in range(num_levels):
            height_list.append(height_data[scene_number + f"_{level}"])
            searched_recolored_topdown_map = cv2.imread(
                recolored_directory + scene_number + f"_{level}" + ".bmp", cv2.IMREAD_GRAYSCALE
            )
            searched_topdown_map = cv2.imread(
                topdown_directory + scene_number + f"_{level}" + ".bmp", cv2.IMREAD_GRAYSCALE
            )
            recolored_topdown_map_list.append(searched_recolored_topdown_map)
            topdown_map_list.append(searched_topdown_map)

        sim_settings = {
            "width": 512,  # Spatial resolution of the observations
            "height": 256,
            "scene": scene,  # Scene path
            "default_agent": 0,
            "sensor_height": 0.5,  # Height of sensors in meters
            "color_sensor": rgb_sensor,  # RGB sensor
            "color_360_sensor": rgb_360_sensor,
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
            "forward_amount": 0.25,
            "backward_amount": 0.25,
            "turn_left_amount": 5.0,
            "turn_right_amount": 5.0,
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

        if display:
            init_opencv_cam()

        print("total scene: ", total_scene_num)

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]
            visual_binary_map = convert_to_visual_binarymap(topdown_map)

            if remove_isolated:
                topdown_map = remove_isolated_area(topdown_map)

            binary_map = convert_to_binarymap(topdown_map)
            _, graph = convert_to_dense_topology(binary_map)

            if len(list(graph.nodes)) == 0:
                continue

            for k in range(num_sampling_per_level):
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

                    cv2.imwrite(f"./output/images/{scene_number}_{i:06d}_{k:06d}_{j}.bmp", color_img)
                    label[f"{scene_number}_{i:06d}_{k:06d}_{j}"] = [
                        [float(pos[1]), float(height_list[i]), float(pos[0])],
                        random_rotation,
                    ]

                label[f"{scene_number}_{i:06d}_{k:06d}_similarity"] = y

                with open(label_json_path, "w") as label_json:  # pylint: disable=unspecified-encoding
                    json.dump(label, label_json, indent=4)

                if display:
                    display_opencv_cam(color_img)

        total_scene_num = total_scene_num + 1

        cv2.destroyAllWindows()
        sim.close()
