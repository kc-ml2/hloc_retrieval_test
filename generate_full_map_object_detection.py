import argparse
import json
import os

import cv2

from config.env_config import ActionConfig, CamFourViewConfig, PathConfig
from network.yolo import Yolo
from relocalization.object_spatial_pyramid import ObjectSpatialPyramid
from relocalization.sim import HabitatSimWithMap
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)
    yolo = Yolo()

    num_iteration = 0
    test_num_level = 0

    for scene_number in scene_list:
        # Find number of levels
        for height in height_data:
            if scene_number in height:
                test_num_level = test_num_level + 1

    # Make list to iterate for Siamese forward
    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        # Find number of levels
        num_level = 0
        for height in height_data:
            if scene_number in height:
                num_level = num_level + 1
        if num_level == 0:
            raise ValueError("Height data is not found.")

        for level in range(num_level):
            print("scene: ", scene_number, "    level: ", level)
            num_iteration = num_iteration + 1
            print(num_iteration, "/", test_num_level)

            map_obs_dir = os.path.join(observation_path, f"map_node_observation_level_{level}")
            sample_dir = os.path.join(observation_path, f"test_sample_{level}")

            # Set output npy file name
            object_spatial_pyramid = ObjectSpatialPyramid(map_obs_dir, sample_dir)
            map_output = object_spatial_pyramid.map_detection_file
            sample_output = object_spatial_pyramid.sample_detection_file

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            sorted_test_sample_file = sorted(os.listdir(sample_dir))
            map_obs_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_file]
            sample_list = [sample_dir + os.sep + file for file in sorted_test_sample_file]

            print("Generating object detection result from map observations...")
            map_detection = {}

            for i, map_obs in enumerate(map_obs_list):
                img = cv2.imread(map_obs)
                cam_observations = {
                    "all_view": None,
                    "front_view": None,
                    "right_view": None,
                    "back_view": None,
                    "left_view": None,
                }

                cam_observations["all_view"] = img
                if sim.is_four_view:
                    cam_observations["front_view"] = img[:, 0:256]
                    cam_observations["right_view"] = img[:, 256:512]
                    cam_observations["back_view"] = img[:, 512:768]
                    cam_observations["left_view"] = img[:, 768:1024]

                _, detection_result = sim.detect_img(cam_observations, yolo)
                map_detection[f"{i:06d}"] = detection_result

                print(i, "/", len(map_obs_list), end="\r")

            print("Generating sample histogram...")
            sample_detection = {}

            for i, sample_obs in enumerate(sample_list):
                img = cv2.imread(sample_obs)
                cam_observations = {
                    "all_view": None,
                    "front_view": None,
                    "right_view": None,
                    "back_view": None,
                    "left_view": None,
                }

                cam_observations["all_view"] = img
                if sim.is_four_view:
                    cam_observations["front_view"] = img[:, 0:256]
                    cam_observations["right_view"] = img[:, 256:512]
                    cam_observations["back_view"] = img[:, 512:768]
                    cam_observations["left_view"] = img[:, 768:1024]

                _, detection_result = sim.detect_img(cam_observations, yolo)
                sample_detection[f"{i:06d}"] = detection_result

                print(i, "/", len(sample_list), end="\r")

            with open(map_output, "w") as f:  # pylint: disable=unspecified-encoding
                json.dump(map_detection, f, indent=4)

            with open(sample_output, "w") as f:  # pylint: disable=unspecified-encoding
                json.dump(sample_detection, f, indent=4)

        sim.close()
