import argparse
import gzip
import json
import os
from os import listdir
from os.path import isfile, join
import random

import cv2
import habitat_sim
import jsonlines
import numpy as np

from utils.habitat_utils import (
    extrinsic_mat_list_to_pos_angle_list,
    interpolate_discrete_matrix,
    make_cfg,
    remove_duplicate_matrix,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path")
    args, _ = parser.parse_known_args()
    output_path = args.output_path

    directory = "/data1/chlee/rxr_dataset/rxr-data/pose_traces/rxr_train/"
    # directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    entire_pose_file_list = [f for f in sorted(listdir(directory)) if isfile(join(directory, f))]

    train_guide_file = "/data1/chlee/rxr_dataset/rxr-data/rxr_train_guide.jsonl.gz"
    # train_guide_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    scene_directory = "/data1/chlee/Matterport3D/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"
    # scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"

    jsonl_file = gzip.open(train_guide_file)
    reader = jsonlines.Reader(jsonl_file)

    language_dict = {}
    eng_scene_dict = {}
    diff_data = {}

    for obj in reader:
        language_dict[obj["instruction_id"]] = obj["language"]
        if obj["language"] == "en-IN" or obj["language"] == "en-US":
            eng_scene_dict[obj["instruction_id"]] = obj["scan"]

    pose_file_list = entire_pose_file_list[0 : len(entire_pose_file_list)]
    diff_json_path = "/data1/chlee/output/diff_data.json"

    total_instruction_num = 0
    total_sampling_num = 0
    for i, pose_file in enumerate(pose_file_list):
        print("Total instruction number: ", total_instruction_num)
        print("Total sampling number: ", total_sampling_num)
        print(i, "/", len(pose_file_list))
        print(pose_file)
        is_follower = "follower" in pose_file

        instruction_id = int(pose_file[0:6])
        is_eng = language_dict[instruction_id] == "en-IN" or language_dict[instruction_id] == "en-US"
        if is_eng is not True:
            continue

        if is_follower:
            os.makedirs(f"/data1/chlee/output/img/{str(instruction_id).zfill(6)}_follwer")
        else:
            os.makedirs(f"/data1/chlee/output/img/{str(instruction_id).zfill(6)}_guide")

        scene = scene_directory + eng_scene_dict[instruction_id] + "/" + eng_scene_dict[instruction_id] + ".glb"
        pose_trace = np.load(directory + pose_file)

        rgb_sensor = True
        depth_sensor = False
        semantic_sensor = False

        meters_per_pixel = 0.1
        translation_threshold = 0.5
        interpolation_interval = 0.02
        sampling_frames = 40
        sampling_number = 2

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

        # Set agent state
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        ext_trans_mat_list = []
        deduplicated_mat_list = []
        pos_trajectory = []
        angle_trajectory = []

        ext_trans_mat_list = pose_trace["extrinsic_matrix"]
        deduplicated_mat_list = remove_duplicate_matrix(ext_trans_mat_list)
        deduplicated_mat_list = interpolate_discrete_matrix(
            list(deduplicated_mat_list), interpolation_interval, translation_threshold
        )
        pos_trajectory, angle_trajectory = extrinsic_mat_list_to_pos_angle_list(deduplicated_mat_list)

        for k in range(sampling_number):
            try:
                idx = random.randint(0, len(pos_trajectory) - sampling_frames)
            except ValueError:
                print(len(pos_trajectory))
                continue

            video_frames_mat = []
            for j in range(sampling_frames):
                video_frames_mat.append(deduplicated_mat_list[idx + j].tolist())

                position = pos_trajectory[idx + j]
                angle_quaternion = angle_trajectory[idx + j]
                agent_state.position = position
                agent_state.rotation = angle_quaternion
                agent.set_state(agent_state)
                observations = sim.get_sensor_observations()
                color_img = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB)
                if is_follower:
                    cv2.imwrite(
                        f"/data1/chlee/output/img/{str(instruction_id).zfill(6)}_follwer/{k}_{str(j).zfill(2)}.jpg",
                        color_img,
                    )
                else:
                    cv2.imwrite(
                        f"/data1/chlee/output/img/{str(instruction_id).zfill(6)}_guide/{k}_{str(j).zfill(2)}.jpg",
                        color_img,
                    )

            if is_follower:
                diff_data[f"{str(instruction_id).zfill(6)}_follwer_{k}"] = video_frames_mat
            else:
                diff_data[f"{str(instruction_id).zfill(6)}_guide_{k}"] = video_frames_mat
            with open(diff_json_path, "w") as diff_json:  # pylint: disable=unspecified-encoding
                json.dump(diff_data, diff_json, indent=4)

            total_sampling_num = total_sampling_num + 1

        sim.close()
        total_instruction_num = total_instruction_num + 1
