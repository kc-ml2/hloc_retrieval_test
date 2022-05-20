import argparse
import random

import cv2
import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel

from grid2topo.habitat_utils import display_opencv_cam, make_cfg

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
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
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

    # create and register new light setup:
    my_scene_lighting_setup = [LightInfo(vector=[2.0, 1.0, 0.6, 0.0], model=LightPositionModel.Global)]
    sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")
    cfg.sim_cfg.scene_light_setup = "my_scene_lighting"
    sim.reconfigure(cfg)

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

    while True:
        observations = sim.get_sensor_observations()
        color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)
        key = display_opencv_cam(color_img)

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
