import argparse
import random
import os
import gzip

import cv2
import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
import numpy as np
import jsonlines
from habitat.utils.visualizations import maps

from grid2topo.habitat_utils import display_opencv_cam, display_observation, make_cfg, display_map
from grid2topo.habitat_utils import convert_transmat_to_point_quaternion, convert_points_to_topdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args, _ = parser.parse_known_args()
    test_scene = args.scene

    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = True

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

    directory = "/data1/rxr_dataset/rxr-data/pose_traces/rxr_train/"
    file_list = os.listdir(directory)
    npzfile = np.load(directory + "001377_guide_pose_trace.npz")
    ext_trans_mat_list = npzfile["extrinsic_matrix"]
    print(len(ext_trans_mat_list))

    # Load map image
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()
    if not sim.pathfinder.is_navigable(nav_point):
        print("Sampled point is not navigable")
    # print(sim.pathfinder.get_bounds()[0])
    # print(sim.pathfinder.get_bounds()[0][0])
    # input()
    topdown_map = maps.get_topdown_map(sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel)
    recolor_palette = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    recolored_topdown_map = recolor_palette[topdown_map]

    img_id = 0


    trans_mat = ext_trans_mat_list[0]
    position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
    agent_state.position = position
    agent_state.rotation = angle_quaternion
    agent.set_state(agent_state)



    while True:
        nodes = []
        for i in range(0, len(ext_trans_mat_list), 300):
            # trans_mat = ext_trans_mat_list[i]
            # position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
            # agent_state.position = position
            # agent_state.rotation = angle_quaternion
            # agent.set_state(agent_state)
            observations = sim.get_sensor_observations()
            color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)
            key = display_opencv_cam(color_img)
            # display_observation(observations["color_sensor"], observations["semantic_sensor"], observations["depth_sensor"])

            if key == ord("w"):
                action = "move_forward"
            if key == ord("s"):
                action = "move_backward"
            if key == ord("a"):
                action = "turn_left"
            if key == ord("d"):
                action = "turn_right"
            if key == ord("o"):
                print("save image")
                cv2.imwrite(f"./output/query{img_id}.jpg", color_img)
                cv2.imwrite(f"./output/db{img_id}.jpg", color_img)
                img_id = img_id + 1
                continue
            sim.step(action)

            # node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            # reversed_point = (node_point[1], node_point[0])
            # nodes.append(reversed_point)
            # display_map(recolored_topdown_map, nodes)

