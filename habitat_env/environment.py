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

    def __init__(self, scene_number, cam_config, action_config, path_config, height_data=None):
        self.scene_number = scene_number
        self.height_data = height_data
        self.cam_config = cam_config

        # Make config
        scene = path_config.SCENE_DIRECTORY + os.sep + scene_number + os.sep + scene_number + ".glb"
        sim_settings = make_sim_setting_dict(scene, cam_config, action_config)
        cfg = make_cfg(sim_settings)

        super().__init__(cfg)

        # Set Flag
        self.is_four_view = cam_config.FOUR_VIEW
        self.four_view_angle = quaternion.from_rotation_vector([0, np.pi / 2, 0])
        self.blank_line = np.zeros([cam_config.HEIGHT, 50, 3]).astype(np.uint8)

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
        cam_observations = {"all_view": None}

        if self.is_four_view:
            cam_observations.update({"front_view": None, "right_view": None, "back_view": None, "left_view": None})

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

            # Recovery agent's state
            self.agent.set_state(original_state)

        else:
            observations = self.get_sensor_observations()
            cam_observations["all_view"] = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)

        return cam_observations

    def detect_img(self, cam_observations, yolo: Yolo):
        """Detect image with Yolo. Merge result images if needed."""
        obs = cam_observations

        if self.is_four_view:
            detect_imgs, detections = yolo.detect_and_display(
                np.array([obs["front_view"], obs["right_view"], obs["back_view"], obs["left_view"]])
            )

            detection_front, detection_right, detection_back, detection_left = detections
            img_front, img_right, img_back, img_left = detect_imgs

            detect_img = np.concatenate(
                [img_front, self.blank_line, img_right, self.blank_line, img_back, self.blank_line, img_left],
                axis=1,
            )

            for i, detect in enumerate([detection_front, detection_right, detection_back, detection_left]):
                boxes = detect[0]
                for box in boxes:
                    if len(box) > 0:
                        box[0] = box[0] + self.cam_config.WIDTH * i

            merged_box = detection_front[0] + detection_right[0] + detection_back[0] + detection_left[0]
            merged_confidence = detection_front[1] + detection_right[1] + detection_back[1] + detection_left[1]
            merged_classIDs = detection_front[2] + detection_right[2] + detection_back[2] + detection_left[2]

            detection_result = merged_box, merged_confidence, merged_classIDs

        else:
            detect_imgs, detections = yolo.detect_and_display(np.expand_dims(obs["all_view"], axis=0))
            detect_img = detect_imgs[0]
            detection_result = detections[0]

        return detect_img, detection_result
