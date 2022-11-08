import os
import random

import cv2
import habitat_sim

from utils.habitat_utils import make_cfg, make_sim_setting_dict


class HabitatSimWithMap(habitat_sim.Simulator):
    """Inheritance instance of habitat_sim.Simulator. This class inlcudes config init, map loading, agent init."""

    def __init__(self, scene_number, cam_config, action_config, path_config, height_data=None):
        self.scene_number = scene_number
        self.height_data = height_data

        # Make config
        scene = path_config.SCENE_DIRECTORY + os.sep + scene_number + os.sep + scene_number + ".glb"
        sim_settings = make_sim_setting_dict(scene, cam_config, action_config)
        cfg = make_cfg(sim_settings)

        super().__init__(cfg)

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
