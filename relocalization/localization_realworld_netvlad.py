import json
import os

import cv2
import h5py
import numpy as np
import tensorflow as tf


class LocalizationRealWorldNetVLAD:
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        config,
        map_obs_dir,
        sample_dir=None,
        load_cache=True,
        instance_only=False,
        visualize=False,
        sparse_map=False,
    ):
        """Initialize localization instance with specific model & map data."""
        self.config = config
        self.map_obs_dir = map_obs_dir
        self.sample_dir = sample_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        if config.CamConfig.IMAGE_CONCAT is True:
            self.num_frames_per_node = 1
        else:
            self.num_frames_per_node = config.CamConfig.NUM_CAMERA

        match_path = "/home/chlee/reference/Hierarchical-Localization/output_hd_concat/feats-superpoint-n4096-r1600_matches-NN-mutual-dist.7_pairs-netvlad.h5"
        match_h5 = h5py.File(match_path, "r")

        self.score_dict = {}
        for test_sample_id in match_h5.keys():
            sample_id = match_h5[test_sample_id].name[-10:-4]
            current_score_dict = {}
            for netvlad_match_pair in match_h5[test_sample_id]:
                netvlad_matched_map_id = match_h5[test_sample_id][netvlad_match_pair].name[-10:-4]
                x = match_h5[test_sample_id][netvlad_match_pair]["matching_scores0"][:]
                current_score_dict.update({netvlad_matched_map_id: np.sum(x)})
            self.score_dict.update({sample_id: current_score_dict})

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))

        self.map_id = 0

        # Open map pose record file
        self.map_pos_record_file = os.path.join(observation_path, f"pos_record_map_{self.map_id}.json")
        with open(self.map_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
            self.map_pos_record = json.load(f)

        sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
        self.num_map_graph_nodes = len(sorted_map_obs_file)

        if self.sample_dir:
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_pos_record_file = os.path.join(observation_path, f"pos_record_{sample_cache_index}.json")

            with open(self.sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_pos_record = json.load(f)

            sorted_sample_file = sorted(os.listdir(sample_dir))
            self.num_sample_graph_nodes = len(sorted_sample_file)

        # Load cached npy file if the flag is true
        if load_cache and (instance_only is False):
            self._load_cache()

            # Initialize graph map from binary topdown map image
            self.map_pos_mat = np.zeros([self.num_map_graph_nodes, 2])

            for node_id in range(self.num_map_graph_nodes):
                self.map_pos_mat[node_id] = self.map_pos_record[f"{node_id:06d}_grid"]

    def _load_cache(self):
        """Load cached npy embedding file from map & sample observations."""
        pass

    def calculate_embedding_from_observation(self, observation):
        """Calculate siamese embedding from observation image with botton network."""
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        with tf.device(f"/device:GPU:{self.config.PathConfig.GPU_ID}"):
            regulized_img = tf.image.convert_image_dtype(observation, tf.float32)
            obs_embedding = self.bottom_network.predict_on_batch(np.expand_dims(regulized_img, axis=0))

        obs_embedding = np.squeeze(obs_embedding)

        return obs_embedding

    def localize_with_observation(self, sample_id: str):
        """Get localization result of current map according to input observation embedding."""
        current_score_dict = self.score_dict[sample_id]
        max_id = max(current_score_dict, key=current_score_dict.get)
        map_node_with_max_value = int(max_id)
        high_similarity_set = [int(key) for key in current_score_dict]

        return map_node_with_max_value, high_similarity_set

    def iterate_localization_with_sample(self):
        """Execute localization & visualize with test sample iteratively."""
        accuracy_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i in range(self.num_sample_graph_nodes):
            print("Sample index: ", i)
            sample_id = f"{i:06d}"
            result = self.localize_with_observation(sample_id)

            grid_pos = np.array(self.sample_pos_record[f"{i:06d}_grid"])

            accuracy = self.evaluate_accuracy(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, gt_node = self.evaluate_node_distance(result[0], grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

            if self.is_visualize:
                print("Sample No.: ", i)
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

                    sample_path = os.path.join(self.sample_dir, f"{i:06d}.jpg")
                    sample_img = cv2.imread(sample_path)
                    blank_img = np.zeros([10, predicted_img.shape[1], 3], dtype=np.uint8)

                    if self.num_frames_per_node == 1:
                        match_img = np.concatenate([sample_img, blank_img, predicted_img, blank_img, true_img], axis=0)
                    else:
                        padded_img = np.full(np.shape(predicted_img), 255, dtype=np.uint8)
                        x_offset = predicted_img.shape[1] // 2 - sample_img.shape[1] // 2
                        padded_img[:, x_offset : x_offset + sample_img.shape[1], :] = sample_img
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
