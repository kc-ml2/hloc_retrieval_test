import json
import os
from pathlib import Path
import re

import cv2
import h5py
import numpy as np
import tensorflow as tf


class LocalizationRealWorldNetVLADOnly:
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        config,
        map_obs_dir,
        match_path,
        query_dir=None,
        load_cache=True,
        instance_only=False,
        visualize=False,
        sparse_map=False,
    ):
        """Initialize localization instance with specific model & map data."""
        self.config = config
        self.map_obs_dir = map_obs_dir
        self.query_dir = query_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        self.query_prefix = os.path.basename(query_dir)

        with open(match_path) as f:
            self.retrieval_pairs = f.readlines()

        if config.CamConfig.IMAGE_CONCAT is True:
            self.num_frames_per_node = 1
        else:
            self.num_frames_per_node = config.CamConfig.NUM_CAMERA

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))

        self.map_id = map_cache_index[-1]

        # Open map pose record file
        self.map_pos_record_file = os.path.join(observation_path, f"pos_record_map_{self.map_id}.json")
        with open(self.map_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
            self.map_pos_record = json.load(f)

        sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
        self.num_map_graph_nodes = len(sorted_map_obs_file)

        if self.query_dir:
            query_cache_index = os.path.basename(os.path.normpath(query_dir))
            self.query_pos_record_file = os.path.join(observation_path, f"pos_record_{query_cache_index}.json")

            with open(self.query_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.query_pos_record = json.load(f)

            sorted_query_file = sorted(os.listdir(query_dir))
            self.num_query_graph_nodes = len(sorted_query_file)

        # Load cached npy file if the flag is true
        if load_cache and (instance_only is False):
            self._load_cache()

            # Initialize graph map from binary topdown map image
            self.map_pos_mat = np.zeros([self.num_map_graph_nodes, 2])

            for node_id in range(self.num_map_graph_nodes):
                self.map_pos_mat[node_id] = self.map_pos_record[f"{node_id:06d}_grid"]

    def _load_cache(self):
        """Load cached npy embedding file from map & query observations."""
        pass

    def calculate_embedding_from_observation(self, observation):
        """Calculate siamese embedding from observation image with botton network."""
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        with tf.device(f"/device:GPU:{self.config.PathConfig.GPU_ID}"):
            regulized_img = tf.image.convert_image_dtype(observation, tf.float32)
            obs_embedding = self.bottom_network.predict_on_batch(np.expand_dims(regulized_img, axis=0))

        obs_embedding = np.squeeze(obs_embedding)

        return obs_embedding

    def localize_with_observation(self, query_id: str):
        """Get localization result of current map according to input observation embedding."""
        high_simil_pair_list = []
        query = f"{self.query_prefix}_hd_concat/{query_id}"
        for pair in self.retrieval_pairs:
            if query in pair:
                high_simil_pair_list.append(pair)
        
        max_pair = high_simil_pair_list[0]
        map_node_with_max_value = int(re.search('/(.+?).jpg', max_pair[-14:]).group(1))
        high_similarity_set = []
        for high_pair in high_simil_pair_list:
            high_similarity_set.append(int(re.search('/(.+?).jpg', high_pair[-14:]).group(1)))

        return map_node_with_max_value, high_similarity_set

    def iterate_localization_with_query(self):
        """Execute localization & visualize with test query iteratively."""
        accuracy_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i in range(self.num_query_graph_nodes):
            print("query index: ", i)
            query_id = f"{i:06d}"
            result = self.localize_with_observation(query_id)

            grid_pos = np.array(self.query_pos_record[f"{i:06d}_grid"])

            accuracy = self.evaluate_accuracy(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, gt_node = self.evaluate_node_distance(result[0], grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

            if self.is_visualize:
                print("query No.: ", i)
                print("Accuracy", accuracy)
                print("Pose D: ", d1)
                print("Node D: ", d2)

                if accuracy is False:
                    if self.num_frames_per_node == 1:
                        predicted_path = os.path.join(self.map_obs_dir, f"{result[0]:06d}.jpg")
                        true_path = os.path.join(self.map_obs_dir, f"{gt_node:06d}.jpg")
                        predicted_img = cv2.imread(predicted_path)
                        true_img = cv2.imread(true_path)
                    else:
                        predicted_path_list = [
                            os.path.join(self.map_obs_dir, f"{result[0]:06d}_{idx}.jpg")
                            for idx in range(self.num_frames_per_node)
                        ]
                        true_path_list = [
                            os.path.join(self.map_obs_dir, f"{gt_node:06d}_{idx}.jpg")
                            for idx in range(self.num_frames_per_node)
                        ]
                        predicted_img_list = [cv2.imread(predicted_path) for predicted_path in predicted_path_list]
                        predicted_img = np.concatenate(predicted_img_list, axis=1)
                        true_img_list = [cv2.imread(true_path) for true_path in true_path_list]
                        true_img = np.concatenate(true_img_list, axis=1)

                    query_path = os.path.join(self.query_dir, f"{i:06d}.jpg")
                    query_img = cv2.imread(query_path)
                    blank_img = np.zeros([10, predicted_img.shape[1], 3], dtype=np.uint8)

                    if self.num_frames_per_node == 1:
                        match_img = np.concatenate([query_img, blank_img, predicted_img, blank_img, true_img], axis=0)
                    else:
                        padded_img = np.full(np.shape(predicted_img), 255, dtype=np.uint8)
                        x_offset = predicted_img.shape[1] // 2 - query_img.shape[1] // 2
                        padded_img[:, x_offset : x_offset + query_img.shape[1], :] = query_img
                        match_img = np.concatenate([padded_img, blank_img, predicted_img, blank_img, true_img], axis=0)

                    cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("localization", 1700, 700)
                    cv2.imshow("localization", match_img)

                key = cv2.waitKey()

                if key == ord("n"):
                    break

        k = i + 1
        print("Temporay Accuracy: ", sum(accuracy_list) / k)

        return accuracy_list, d1_list, d2_list, i + 1

    def get_ground_truth_nearest_node(self, grid_pos):
        """Get the nearest node by Euclidean distance."""
        current_pos_mat = np.zeros([self.num_map_graph_nodes, 2])
        current_pos_mat[:] = grid_pos
        distance_set = np.linalg.norm(self.map_pos_mat - current_pos_mat, axis=1)
        nearest_node = np.argmin(distance_set)

        return nearest_node

    def evaluate_pos_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from current position?"""
        predicted_grid_pos = self.map_pos_record[f"{map_node_with_max_value:06d}_grid"]
        distance = np.linalg.norm(np.array(predicted_grid_pos) - grid_pos)

        return distance

    def evaluate_node_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from the ground-truth nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)
        ground_truth_nearest_node_pos = self.map_pos_record[f"{ground_truth_nearest_node:06d}_grid"]
        predicted_nearest_node_pos = self.map_pos_record[f"{map_node_with_max_value:06d}_grid"]
        distance = np.linalg.norm(np.array(ground_truth_nearest_node_pos) - np.array(predicted_nearest_node_pos))

        return distance, ground_truth_nearest_node

    def evaluate_accuracy(self, map_node_with_max_value, grid_pos):
        """Is it the nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        ground_truth_pos = self.map_pos_record[f"{ground_truth_nearest_node:06d}_grid"]
        estimated_pos = self.map_pos_record[f"{map_node_with_max_value:06d}_grid"]

        step = np.linalg.norm(np.array(ground_truth_pos) - np.array(estimated_pos))

        result = step <= 10

        return bool(result)
