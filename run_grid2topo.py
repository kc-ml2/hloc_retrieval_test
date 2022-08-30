import argparse
import random

import cv2
import habitat_sim
import numpy as np

from utils.habitat_utils import display_map, get_entire_maps_by_levels, init_map_display, make_cfg
from utils.skeletonize_utils import convert_to_topology, convert_to_visual_binarymap, display_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1

    check_radius = 3
    prune_iteration = 2
    noise_removal_threshold = 2

    kernel = np.ones((5, 5), np.uint8)

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    for scene_number in scene_list:
        scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
        scene = scene_directory + scene_number + "/" + scene_number + ".glb"

        sim_settings = {
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
            "scene": scene,  # Scene path
            "default_agent": 0,
            "sensor_height": 0,  # Height of sensors in meters
            "color_sensor": rgb_sensor,  # RGB sensor
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
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

        recolored_topdown_map_list, topdown_map_list = get_entire_maps_by_levels(sim, meters_per_pixel)

        init_map_display(window_name="colored_map")
        init_map_display(window_name="visual_binary_map")

        for i, recolored_topdown_map in enumerate(recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", i)
            topdown_map = topdown_map_list[i]

            print("Displaying recolored map:")
            display_map(recolored_topdown_map, window_name="colored_map", wait_for_key=True)

            visual_binary_map = convert_to_visual_binarymap(topdown_map)
            print("Displaying visual binary map:")
            display_map(visual_binary_map, window_name="visual_binary_map", wait_for_key=True)

            topdown_map = cv2.erode(topdown_map, kernel, iterations=1)
            topdown_map = cv2.dilate(topdown_map, kernel, iterations=1)

            contours, hierarchy = cv2.findContours(topdown_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < noise_removal_threshold:
                    cv2.fillPoly(topdown_map, [contour], 0)

            skeletonized_map, graph = convert_to_topology(topdown_map)

            print("Displaying original graph:")
            display_graph(visual_binary_map, graph, window_name="original graph", wait_for_key=True)

            # map_img = generate_map_image(visual_binary_map, graph, line_edge=False)
            # cv2.imwrite(f"./output/skeleton/{scene_number}_{i}.jpg", map_img)

            for _ in range(prune_iteration):
                end_node_list = []
                isolated_node_list = []
                root_node_list = []

                for node in graph.nodes:
                    if len(list(graph.neighbors(node))) == 1:
                        end_node_list.append(node)
                    if len(list(graph.neighbors(node))) == 0:
                        isolated_node_list.append(node)

                for isolated_node in isolated_node_list:
                    graph.remove_node(isolated_node)

                for end_node in end_node_list:
                    pnt = [int(graph.nodes[end_node]["o"][0]), int(graph.nodes[end_node]["o"][1])]
                    check_patch = topdown_map[
                        pnt[0] - check_radius : pnt[0] + check_radius, pnt[1] - check_radius : pnt[1] + check_radius
                    ]
                    if (2 in check_patch) or (0 in check_patch):
                        graph.remove_node(end_node)

                for node in graph.nodes:
                    if len(list(graph.neighbors(node))) == 2:
                        root_node_list.append(node)

                for root_node in root_node_list:
                    branch = list(graph.neighbors(root_node))
                    if len(branch) != 2:
                        continue
                    graph.remove_node(root_node)
                    graph.add_edge(branch[0], branch[1])
                    graph.edges[branch[0], branch[1]]["pts"] = []

            print("Displaying pruned graph:")
            display_graph(visual_binary_map, graph, window_name="pruned_graph", wait_for_key=True, line_edge=True)

            # map_img = generate_map_image(visual_binary_map, graph, line_edge=True)
            # cv2.imwrite(f"./output/pruned/{scene_number}_{i}.jpg", map_img)

        sim.close()
