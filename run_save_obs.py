import argparse
import os
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim

from config.env_config import ActionConfig, Cam360Config, DataConfig, PathConfig
from utils.habitat_utils import (
    display_map,
    display_opencv_cam,
    get_closest_map,
    get_entire_maps_by_levels,
    init_map_display,
    init_opencv_cam,
    make_cfg,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    display_observation = True
    display_path_map = True

    os.makedirs("./output/images/")
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    # for scene_number in scene_list:
    scene_number = scene_list[0]
    scene = PathConfig.SCENE_DIRECTORY + os.sep + scene_number + os.sep + scene_number + ".glb"

    sim_settings = {
        "width": Cam360Config.WIDTH,
        "height": Cam360Config.HEIGHT,
        "scene": scene,
        "default_agent": 0,
        "sensor_height": Cam360Config.SENSOR_HEIGHT,
        "color_sensor": Cam360Config.RGB_SENSOR,
        "color_360_sensor": Cam360Config.RGB_360_SENSOR,
        "depth_sensor": Cam360Config.DEPTH_SENSOR,
        "semantic_sensor": Cam360Config.SEMANTIC_SENSOR,
        "seed": 1,
        "enable_physics": False,
        "forward_amount": ActionConfig.FORWARD_AMOUNT,
        "backward_amount": ActionConfig.BACKWARD_AMOUNT,
        "turn_left_amount": ActionConfig.TURN_LEFT_AMOUNT,
        "turn_right_amount": ActionConfig.TURN_RIGHT_AMOUNT,
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

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized")
    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()

    agent_state.position = nav_point  # world space
    agent.set_state(agent_state)

    img_id = 0

    if display_path_map:
        recolored_topdown_map_list, _, _ = get_entire_maps_by_levels(sim, DataConfig.METERS_PER_PIXEL)
        init_map_display()

    if display_observation:
        init_opencv_cam()

    while True:
        observations = sim.get_sensor_observations()
        color_img = cv2.cvtColor(observations["color_360_sensor"], cv2.COLOR_BGR2RGB)

        if display_observation:
            key = display_opencv_cam(color_img)

        current_state = agent.get_state()
        position = current_state.position

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
            cv2.imwrite(f"./output/images/query{img_id}.jpg", color_img)
            cv2.imwrite(f"./output/images/db{img_id}.jpg", color_img)
            img_id = img_id + 1
            continue
        if key == ord("q"):
            break

        sim.step(action)

        if display_path_map:
            recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)
            node_point = maps.to_grid(position[2], position[0], recolored_topdown_map.shape[0:2], sim)
            transposed_point = (node_point[1], node_point[0])
            display_map(recolored_topdown_map, key_points=[transposed_point])
