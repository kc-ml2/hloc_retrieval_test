import argparse
import random

import habitat_sim
from habitat.utils.visualizations import maps
import numpy as np

from grid2topo.habitat_utils import make_cfg, display_map, convert_points_to_topdown


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
        "sensor_height": 1.5,  # Height of sensors in meters
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
    pathfinder_seed = 4

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.5, 0.0])  # world space
    agent.set_state(agent_state)

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")

    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()

    if not sim.pathfinder.is_navigable(nav_point):
        print("Sampled point is not navigable")

    topdown_map = maps.get_topdown_map(sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel)

    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    recolored_topdown_map = recolor_map[topdown_map]

    print("Displaying the raw map from get_topdown_view:")
    display_map(topdown_map)
    input()

    print("Displaying recolored:")
    display_map(recolored_topdown_map)
    input()

    vis_points = [nav_point]
    xy_vis_points = convert_points_to_topdown(sim.pathfinder, vis_points, meters_per_pixel)
    print("\nDisplay the map with key_point overlay:")
    display_map(recolored_topdown_map, key_points=xy_vis_points)
    input()
