import argparse
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np

from grid2topo.cv_utils import draw_flow
from grid2topo.habitat_utils import (
    convert_transmat_to_point_quaternion,
    display_map,
    display_opencv_cam,
    extrinsic_mat_list_to_pos_angle_list,
    get_closest_map,
    get_entire_maps_by_levels,
    get_scene_by_eng_guide,
    interpolate_discrete_matrix,
    make_cfg,
    print_pose_diff,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-id")
    args, _ = parser.parse_known_args()
    instruction_id = int(args.trace_id)

    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    pose_trace = np.load(directory + str(instruction_id).zfill(6) + "_follower_pose_trace.npz")
    train_guide_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"

    # Search for scene glb file according to trace-id
    scene = get_scene_by_eng_guide(instruction_id, train_guide_file, scene_directory)

    rgb_sensor = True
    depth_sensor = False
    semantic_sensor = False

    optical_flow_overlay = False

    meters_per_pixel = 0.1
    translation_threshold = 0.5
    interpolation_interval = 0.1

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

    # Load map image
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")
    sim.pathfinder.seed(pathfinder_seed)

    recolored_topdown_map_list = get_entire_maps_by_levels(sim, meters_per_pixel)

    ext_trans_mat_list = pose_trace["extrinsic_matrix"]
    trans_mat = ext_trans_mat_list[0]
    position, _ = convert_transmat_to_point_quaternion(trans_mat)

    recolored_topdown_map = get_closest_map(sim, position, recolored_topdown_map_list)

    ext_trans_mat_list = interpolate_discrete_matrix(
        list(ext_trans_mat_list), interpolation_interval, translation_threshold
    )
    pos_trajectory, angle_trajectory = extrinsic_mat_list_to_pos_angle_list(ext_trans_mat_list)

    img_id = 0
    nodes = []
    prev = None

    for i in range(0, len(pos_trajectory), 1):
        position = pos_trajectory[i]
        angle_quaternion = angle_trajectory[i]
        agent_state.position = position

        if i == 0:
            prev_trans_mat = ext_trans_mat_list[i]

        print_pose_diff(i, ext_trans_mat_list[i], prev_trans_mat)
        prev_trans_mat = ext_trans_mat_list[i]

        agent_state.rotation = angle_quaternion
        agent.set_state(agent_state)
        observations = sim.get_sensor_observations()
        color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

        if optical_flow_overlay:
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            if prev is None:  # First frame
                prev = gray
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev=prev,
                    next=gray,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )
                draw_flow(color_img, flow)
                prev = gray

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
