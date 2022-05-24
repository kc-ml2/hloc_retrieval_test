import argparse
import random

from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np

from grid2topo.habitat_utils import convert_transmat_to_point_quaternion, display_map, make_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args, _ = parser.parse_known_args()
    test_scene = args.scene

    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    instruction_id_list = ["001279", "001364", "001456", "001679", "002324", "002947"]

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
    nav_point = sim.pathfinder.get_random_navigable_point()
    if not sim.pathfinder.is_navigable(nav_point):
        print("Sampled point is not navigable")
    topdown_map = maps.get_topdown_map(sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel)
    recolor_palette = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    recolored_topdown_map = recolor_palette[topdown_map]

    img_id = 0
    trajectory_list = []

    for id in instruction_id_list:
        npzfile = np.load(directory + id + "_guide_pose_trace.npz")
        ext_trans_mat_list = npzfile["extrinsic_matrix"]

        nodes = []
        for i in range(0, len(ext_trans_mat_list), 10):
            trans_mat = ext_trans_mat_list[i]
            position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)

            node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            if transposed_point in nodes:
                pass
            else:
                nodes.append(transposed_point)
        trajectory_list.append(nodes)

    for trajectory in trajectory_list:
        display_map(recolored_topdown_map, trajectory)
