import argparse
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np

from grid2topo.habitat_utils import convert_transmat_to_point_quaternion, display_map, display_opencv_cam, make_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args, _ = parser.parse_known_args()
    test_scene = args.scene

    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    pose_trace = np.load(directory + "001456_guide_pose_trace.npz")

    rgb_sensor = True
    depth_sensor = False
    semantic_sensor = False

    meters_per_pixel = 0.1

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": test_scene,  # Scene path
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

    # Load map image
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")
    sim.pathfinder.seed(pathfinder_seed)
    semantics = sim.semantic_scene

    for level in semantics.levels:
        print(f"Level id:{level.id}")

    nav_point_list = []
    closest_level_list = []
    for i in range(300):
        level_distance_dict = {}
        nav_point = sim.pathfinder.get_random_navigable_point()
        distance_list = []
        average_list = []
        for level in semantics.levels:
            for region in level.regions:
                distance = abs(region.aabb.center[1] - (nav_point[1] + 0.5))
                distance_list.append(distance)
            average = sum(distance_list) / len(distance_list)
            average_list.append(average)
        closest_level = average_list.index(min(average_list))
        nav_point_list.append(nav_point)
        closest_level_list.append(closest_level)

    desired_level = 0
    for i, point in enumerate(nav_point_list):
        if not sim.pathfinder.is_navigable(point):
            print("Sampled point is not navigable")
        if closest_level_list[i] == desired_level:
            topdown_map = maps.get_topdown_map(sim.pathfinder, height=point[1], meters_per_pixel=meters_per_pixel)
            recolor_palette = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
            recolored_topdown_map = recolor_palette[topdown_map]
            print(closest_level_list[i])
            node_point = maps.to_grid(point[2], point[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            display_map(recolored_topdown_map, [transposed_point], wait_for_key=True)

    img_id = 0
    ext_trans_mat_list = pose_trace["extrinsic_matrix"]

    while True:
        nodes = []
        for i in range(0, len(ext_trans_mat_list), 100):
            trans_mat = ext_trans_mat_list[i]
            position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
            agent_state.position = position
            print(position)
            agent_state.rotation = angle_quaternion
            agent.set_state(agent_state)
            observations = sim.get_sensor_observations()
            color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)
            key = display_opencv_cam(color_img)

            if key == ord("o"):
                print("save image")
                cv2.imwrite(f"./output/query{img_id}.jpg", color_img)
                cv2.imwrite(f"./output/db{img_id}.jpg", color_img)
                img_id = img_id + 1
                continue

            node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            nodes.append(transposed_point)
            display_map(recolored_topdown_map, nodes)
