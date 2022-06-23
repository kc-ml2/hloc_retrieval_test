import argparse
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np

from grid2topo.habitat_utils import (
    convert_transmat_to_point_quaternion,
    display_map,
    display_opencv_cam,
    get_closest_map,
    get_entire_maps_by_levels,
    get_scene_by_eng_guide,
    make_cfg,
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
    depth_sensor = False
    semantic_sensor = False

    meters_per_pixel = 0.1

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

    # while True:
    #     sample1 = sim.pathfinder.get_random_navigable_point()
    #     sample2 = sim.pathfinder.get_random_navigable_point()

    #     path = habitat_sim.ShortestPath()
    #     path.requested_start = sample1
    #     path.requested_end = sample2
    #     found_path = sim.pathfinder.find_path(path)
    #     geodesic_distance = path.geodesic_distance
    #     path_points = path.points
    #     print(path_points)
    #     input()
    #     if path_points == []:
    #         continue
    #     else:
    #         break

    # recolored_topdown_map = get_closest_map(sim, sample1, recolored_topdown_map_list)

    # sample1_point = maps.to_grid(sample1[2], sample1[0], recolored_topdown_map.shape[0:2], sim)
    # transposed_point = (sample1_point[1], sample1_point[0])
    # cv2.drawMarker(
    #     img=recolored_topdown_map,
    #     position=(int(transposed_point[0]), int(transposed_point[1])),
    #     color=(0, 255, 0),
    #     markerType=cv2.MARKER_DIAMOND,
    #     markerSize=2,
    # )

    # sample2_point = maps.to_grid(sample2[2], sample2[0], recolored_topdown_map.shape[0:2], sim)
    # transposed_point = (sample2_point[1], sample2_point[0])
    # cv2.drawMarker(
    #     img=recolored_topdown_map,
    #     position=(int(transposed_point[0]), int(transposed_point[1])),
    #     color=(0, 255, 0),
    #     markerType=cv2.MARKER_DIAMOND,
    #     markerSize=2,
    # )

    # nodes = []
    # for position in path_points:
    #     agent_state.position = position
    #     print(position)
    #     agent.set_state(agent_state)
    #     observations = sim.get_sensor_observations()
    #     color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)
    #     key = display_opencv_cam(color_img)

    #     node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
    #     transposed_point = (node_point[1], node_point[0])
    #     nodes.append(transposed_point)
    #     display_map(recolored_topdown_map, nodes)

    # Option 1
    # TODO: Just use dataset from pose diff
    # TODO: check the number of data (especially orientation)
    # TODO: Only the amount of translation can be varied
    # TODO: Path interpolation must be implemented for translation variation
    # TODO: How to discriminate rotation and translation
    # TODO: Backward translation...?

    # Option 2
    # TODO: Implement size-variable action

    ext_trans_mat_list = pose_trace["extrinsic_matrix"]
    trans_mat = ext_trans_mat_list[0]
    position, _ = convert_transmat_to_point_quaternion(trans_mat)

    print(len(ext_trans_mat_list))
    input()

    # for i in range(0, len(ext_trans_mat_list), 100):
    #     position, angle_quaternion = convert_transmat_to_point_quaternion(ext_trans_mat_list[i])
    #     if (i + 100 <= len(ext_trans_mat_list)):
    #         next_pos, next_angle = convert_transmat_to_point_quaternion(ext_trans_mat_list[i + 100])
    #     print("current pos: ", position, angle_quaternion)
    #     print("next pos: ", next_pos, next_angle)
    #     print("pos diff: ", next_pos - position)
    #     print("angle diff: ", next_angle - angle_quaternion)
    #     print(i)
    #     input()

    # for i, trans_mat in enumerate(ext_trans_mat_list):
    #     position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
    #     if (i + 1 <= len(ext_trans_mat_list)):
    #         next_pos, next_angle = convert_transmat_to_point_quaternion(ext_trans_mat_list[i + 1])
    #     print("current pos: ", position, angle_quaternion)
    #     print("pos diff: ", next_pos - position)
    #     print("angle diff: ", next_angle - angle_quaternion)
    #     print(i)
    #     input()

    recolored_topdown_map = get_closest_map(sim, position, recolored_topdown_map_list)

    img_id = 0
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
