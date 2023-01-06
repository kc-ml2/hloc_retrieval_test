import os
import random

import cv2
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

from algorithms.yolo import Yolo
from utils.habitat_utils import make_cfg, make_sim_setting_dict


class HabitatSimWithMap(habitat_sim.Simulator):
    """Inheritance instance of habitat_sim.Simulator. This class inlcudes config init, map loading, agent init."""

    def __init__(self, scene_number, cam_config, action_config, path_config, height_data=None, is_detection=None):
        self.scene_number = scene_number
        self.height_data = height_data

        # Make config
        scene = path_config.SCENE_DIRECTORY + os.sep + scene_number + os.sep + scene_number + ".glb"
        sim_settings = make_sim_setting_dict(scene, cam_config, action_config)
        cfg = make_cfg(sim_settings)

        super().__init__(cfg)

        # Set Flag
        self.is_four_view = cam_config.FOUR_VIEW
        self.four_view_angle = quaternion.from_rotation_vector([0, np.pi / 2, 0])
        self.four_view_line = np.zeros([cam_config.HEIGHT, 50, 3]).astype(np.uint8)

        # Set seed
        random.seed(sim_settings["seed"])
        self.seed(sim_settings["seed"])
        self.pathfinder.seed(sim_settings["seed"])

        # Load map from map database
        map_data = self.get_map_from_database()
        self.recolored_topdown_map_list = map_data[0]
        self.topdown_map_list = map_data[1]
        self.height_list = map_data[2]

        # Variable to store current map state
        self.recolored_topdown_map = None
        self.closest_level = 0

        # Initialize agent
        self.agent = self.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        nav_point = self.pathfinder.get_random_navigable_point()
        agent_state.position = nav_point  # world space
        self.agent.set_state(agent_state)

        # Initialize Yolo
        if is_detection:
            self.yolo = Yolo()

    def get_map_from_database(
        self, topdown_directory="./data/topdown/", recolored_directory="./data/recolored_topdown/"
    ):
        """Get map files from pre-made map."""
        num_levels = 0
        for _, _, files in os.walk(topdown_directory):
            for file in files:
                if self.scene_number in file:
                    num_levels = num_levels + 1

        recolored_topdown_map_list = []
        topdown_map_list = []
        height_list = []
        for level in range(num_levels):
            height_list.append(self.height_data[self.scene_number + f"_{level}"])
            searched_recolored_topdown_map = cv2.imread(
                recolored_directory + os.sep + self.scene_number + f"_{level}" + ".bmp", cv2.IMREAD_GRAYSCALE
            )
            searched_topdown_map = cv2.imread(
                topdown_directory + os.sep + self.scene_number + f"_{level}" + ".bmp", cv2.IMREAD_GRAYSCALE
            )
            recolored_topdown_map_list.append(searched_recolored_topdown_map)
            topdown_map_list.append(searched_topdown_map)

        return recolored_topdown_map_list, topdown_map_list, height_list

    def update_closest_map(self, position):
        """Find out which level the agent is on."""
        distance_list = []
        average_list = []

        for level in self.semantic_scene.levels:
            for region in level.regions:
                distance = abs(region.aabb.center[1] - position[1])
                distance_list.append(distance)
            average = sum(distance_list) / len(distance_list)
            average_list.append(average)

        self.closest_level = average_list.index(min(average_list))
        self.recolored_topdown_map = self.recolored_topdown_map_list[self.closest_level]

    def set_state_from_grid(self, grid_pos, level, random_rotation=True, rotation=None):
        """Set agent state from position from grid."""
        agent_state = habitat_sim.AgentState()
        pos = maps.from_grid(
            int(grid_pos[0]),
            int(grid_pos[1]),
            self.recolored_topdown_map_list[level].shape[0:2],
            self,
            self.pathfinder,
        )

        if random_rotation & bool(rotation):
            raise ValueError("Input Error. Put only one value between random_rotation and rotation.")
        if random_rotation:
            random_rotation = random.randint(0, 359)
            r = Rotation.from_euler("y", random_rotation, degrees=True)
            agent_state.rotation = r.as_quat()

        agent_state.position = np.array([pos[1], self.height_list[level], pos[0]])
        self.agent.set_state(agent_state)
        self.update_closest_map(agent_state.position)

        return pos, random_rotation

    def get_cam_observations(self):
        """Inherit the 'get_sensor_observations' method of the parent class."""
        cam_observations = {
            "all_view": None,
            "all_view_with_line": None,
            "front_view": None,
            "right_view": None,
            "back_view": None,
            "left_view": None,
        }

        if self.is_four_view:
            # Store original view for agent state restoration
            original_state = self.agent.get_state()
            current_state = self.agent.get_state()
            rotation = current_state.rotation

            # Get front view
            observations = self.get_sensor_observations()
            cam_observations["front_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

            # Turn agent for left view
            rotation = rotation * self.four_view_angle
            current_state.rotation = rotation
            self.agent.set_state(current_state)

            # Get left view
            observations = self.get_sensor_observations()
            cam_observations["left_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

            # Turn agent for back view
            rotation = rotation * self.four_view_angle
            current_state.rotation = rotation
            self.agent.set_state(current_state)

            # Get back view
            observations = self.get_sensor_observations()
            cam_observations["back_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

            # Turn agent for right view
            rotation = rotation * self.four_view_angle
            current_state.rotation = rotation
            self.agent.set_state(current_state)

            # Get right view
            observations = self.get_sensor_observations()
            cam_observations["right_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

            # Merge every view for "all_view"
            cam_observations["all_view"] = np.concatenate(
                [
                    cam_observations["front_view"],
                    cam_observations["right_view"],
                    cam_observations["back_view"],
                    cam_observations["left_view"],
                ],
                axis=1,
            )

            # Merge every view for "all_view_with_line"
            cam_observations["all_view_with_line"] = np.concatenate(
                [
                    cam_observations["front_view"],
                    self.four_view_line,
                    cam_observations["right_view"],
                    self.four_view_line,
                    cam_observations["back_view"],
                    self.four_view_line,
                    cam_observations["left_view"],
                ],
                axis=1,
            )

            # Recovery agent's state
            self.agent.set_state(original_state)

        else:
            observations = self.get_sensor_observations()
            cam_observations["all_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

        return cam_observations

    def detect_img(self, cam_observations):
        """Detect image with Yolo. Merge result images if needed."""
        if self.is_four_view:
            detect_img_front = self.yolo.detect_and_display(cam_observations["front_view"])
            detect_img_right = self.yolo.detect_and_display(cam_observations["right_view"])
            detect_img_back = self.yolo.detect_and_display(cam_observations["back_view"])
            detect_img_left = self.yolo.detect_and_display(cam_observations["left_view"])
            detect_img = np.concatenate(
                [
                    detect_img_front,
                    self.four_view_line,
                    detect_img_right,
                    self.four_view_line,
                    detect_img_back,
                    self.four_view_line,
                    detect_img_left,
                ],
                axis=1,
            )

        else:
            detect_img = self.yolo.detect_and_display(cam_observations["all_view"])

        return detect_img
