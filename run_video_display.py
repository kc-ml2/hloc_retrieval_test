import argparse
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np

from utils.cv_utils import draw_flow
from utils.habitat_utils import (
    cal_pose_diff,
    convert_transmat_to_point_quaternion,
    display_map,
    display_opencv_cam,
    extrinsic_mat_list_to_pos_angle_list,
    get_closest_map,
    get_entire_maps_by_levels,
    get_scene_by_eng_guide,
    init_map_display,
    init_opencv_cam,
    interpolate_discrete_matrix,
    make_cfg,
    remove_duplicate_matrix,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-id")
    args, _ = parser.parse_known_args()
    instruction_id = int(args.trace_id)

    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    pose_trace = np.load(directory + str(instruction_id).zfill(6) + "_guide_pose_trace.npz")
    train_guide_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"

    # Search for scene glb file according to trace-id
    scene = get_scene_by_eng_guide(instruction_id, train_guide_file, scene_directory)

    rgb_sensor = True
    rgb_360_sensor = False
    depth_sensor = False
    semantic_sensor = False

    display_observation = True
    display_path_map = True
    display_semantic_object = True
    optical_flow_overlay = False

    remove_duplicate_frames = True
    interpolate_translation = True

    meters_per_pixel = 0.1
    translation_threshold = 0.5
    interpolation_interval = 0.02

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": scene,  # Scene path
        "default_agent": 0,
        "sensor_height": 0,  # Height of sensors in meters
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

    # Load map image
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")
    sim.pathfinder.seed(pathfinder_seed)

    recolored_topdown_map_list, _, _ = get_entire_maps_by_levels(sim, meters_per_pixel)

    ext_trans_mat_list = pose_trace["extrinsic_matrix"]
    trans_mat = ext_trans_mat_list[0]
    position, _ = convert_transmat_to_point_quaternion(trans_mat)

    recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)

    if display_semantic_object:
        level = sim.semantic_scene.levels[closest_level]

        door_pos_list = []
        for region in level.regions:
            for obj in region.objects:
                if obj.category.name() == "sofa":
                    door_point = maps.to_grid(
                        obj.aabb.center[2], obj.aabb.center[0], recolored_topdown_map.shape[0:2], sim
                    )
                    transposed_point = (door_point[1], door_point[0])

                    cv2.drawMarker(
                        img=recolored_topdown_map,
                        position=(int(transposed_point[0]), int(transposed_point[1])),
                        color=(0, 0, 255),
                        markerType=cv2.MARKER_DIAMOND,
                        markerSize=2,
                    )

    if remove_duplicate_frames:
        ext_trans_mat_list = remove_duplicate_matrix(ext_trans_mat_list)

    if interpolate_translation:
        ext_trans_mat_list = interpolate_discrete_matrix(
            list(ext_trans_mat_list), interpolation_interval, translation_threshold
        )

    pos_trajectory, angle_trajectory = extrinsic_mat_list_to_pos_angle_list(ext_trans_mat_list)

    img_id = 0
    nodes = []
    prev = None

    if display_observation:
        init_opencv_cam()

    if display_path_map:
        init_map_display()

    for i in range(0, len(pos_trajectory), 1):
        position = pos_trajectory[i]
        angle_quaternion = angle_trajectory[i]
        agent_state.position = position

        if i == 0:
            prev_trans_mat = ext_trans_mat_list[i]

        position_diff, _, rotation_diff = cal_pose_diff(ext_trans_mat_list[i], prev_trans_mat)
        prev_trans_mat = ext_trans_mat_list[i]

        print("Frame: ", i)
        print("Position diff: ", position_diff)
        print("Angle diff (euler[deg]): ", rotation_diff)

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

        if display_observation:
            key = display_opencv_cam(color_img)
            if key == ord("o"):
                print("save image")
                cv2.imwrite(f"./output/query{img_id}.jpg", color_img)
                cv2.imwrite(f"./output/db{img_id}.jpg", color_img)
                img_id = img_id + 1

        if display_path_map:
            node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            nodes.append(transposed_point)
            display_map(recolored_topdown_map, key_points=nodes)
