import json
import os

import cv2
import networkx as nx
import numpy as np
import tensorflow as tf

from config.algorithm_config import NetworkConstant, TestConstant
from config.env_config import DataConfig, PathConfig
from relocalization.object_spatial_pyramid import ObjectSpatialPyramid
from utils.habitat_utils import draw_point_from_grid_pos, draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import topdown_map_to_graph


class Localization:
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        top_network,
        bottom_network,
        map_obs_dir,
        sample_dir=None,
        is_detection=False,
        binary_topdown_map=None,
        load_cache=True,
        visualize=True,
    ):
        """Initialize localization instance with specific model & map data."""
        self.sample_dir = sample_dir
        self.is_detection = is_detection
        self.is_visualize = visualize

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            self.top_network = top_network
            self.bottom_network = bottom_network

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.map_embedding_file = os.path.join(observation_path, f"siamese_embedding_{map_cache_index}.npy")

        if self.sample_dir:
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_embedding_file = os.path.join(observation_path, f"siamese_embedding_{sample_cache_index}.npy")
            self.sample_pos_record_file = os.path.join(observation_path, f"pos_record_{sample_cache_index}.json")

            with open(self.sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_pos_record = json.load(f)

        # Initialize emny matrix and parameters for handling embeddings
        self.num_map_embedding = len(os.listdir(os.path.normpath(map_obs_dir)))
        self.dimension_map_embedding = NetworkConstant.NUM_EMBEDDING
        self.input_embedding_mat = np.zeros((self.num_map_embedding, 2 * self.dimension_map_embedding))

        # Load cached npy file if the flag is true
        if load_cache:
            self._load_cache()

            # Initialize graph map from binary topdown map image
            self.graph = topdown_map_to_graph(binary_topdown_map, DataConfig.REMOVE_ISOLATED)
            self.map_pos_mat = np.zeros([len(self.graph.nodes()), 2])

            for node_id in self.graph.nodes():
                self.map_pos_mat[node_id] = self.graph.nodes[node_id]["o"]

        # Initialize spatial pyramid matching instance
        if is_detection:
            self.object_pyramid = ObjectSpatialPyramid(map_obs_dir=map_obs_dir, sample_dir=sample_dir, load_cache=True)

    def _load_cache(self):
        """Load cached npy embedding file from map & sample observations."""
        with open(self.map_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
            map_embedding_mat = np.load(f)
            if np.shape(map_embedding_mat)[0] != self.num_map_embedding:
                raise ValueError("Dimension of the cache file is different with map record.")

        if self.sample_dir:
            with open(self.sample_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
                self.sample_embedding_mat = np.load(f)

        self.input_embedding_mat[:, : self.dimension_map_embedding] = map_embedding_mat

    def calculate_embedding_from_observation(self, observation):
        """Calculate siamese embedding from observation image with botton network."""
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            regulized_img = tf.image.convert_image_dtype(observation, tf.float32)
            obs_embedding = self.bottom_network.predict_on_batch(np.expand_dims(regulized_img, axis=0))

        obs_embedding = np.squeeze(obs_embedding)

        return obs_embedding

    def localize_with_observation(self, observation_embedding, detection_result=None):
        """Get localization result of current map according to input observation embedding."""

        self.input_embedding_mat[:, self.dimension_map_embedding :] = observation_embedding

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            predictions = self.top_network.predict_on_batch(self.input_embedding_mat)

        similarity = predictions[:, 1]
        map_node_with_max_value = np.argmax(similarity)
        high_similarity_set = [
            id for id in range(len(similarity)) if similarity[id] > TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
        ]

        if self.is_detection and detection_result[0] != []:
            if detection_result is None:
                raise ValueError("Detection result is required for localization with object detection.")

            current_histogram, _, _, _ = self.object_pyramid.make_spatial_histogram(detection_result)
            current_histogram_batch = np.zeros(self.object_pyramid.map_histogram_batch.shape)
            current_histogram_batch[:] = current_histogram

            scores = np.sum(np.minimum(self.object_pyramid.map_histogram_batch, current_histogram_batch), axis=1)

            all_max_ids = np.argwhere(scores == np.amax(scores))
            all_max_ids = all_max_ids // self.object_pyramid.low_pyramid_dim
            all_max_ids = all_max_ids.flatten().tolist()

            for id in all_max_ids:
                if id == map_node_with_max_value:
                    return map_node_with_max_value, high_similarity_set, similarity

            filtered_similarity = []

            if high_similarity_set == []:
                for id in all_max_ids:
                    filtered_similarity.append(similarity[id])
                map_node_with_max_value = all_max_ids[np.argmax(filtered_similarity)]

            else:
                for i, high_sim_id in enumerate(high_similarity_set):
                    if (high_sim_id in all_max_ids) is False:
                        high_similarity_set.pop(i)

                if high_similarity_set == []:
                    for id in all_max_ids:
                        filtered_similarity.append(similarity[id])
                    map_node_with_max_value = all_max_ids[np.argmax(filtered_similarity)]

                else:
                    for high_sim_id in high_similarity_set:
                        filtered_similarity.append(similarity[high_sim_id])
                    map_node_with_max_value = high_similarity_set[np.argmax(filtered_similarity)]

            # print("Detection: ", detection_result)
            # print("Max value id with detection: ", max_id)
            # print("node in high sim set: ", max_id in high_similarity_set)

        return map_node_with_max_value, high_similarity_set, similarity

    def visualize_on_map(self, map_image, result):
        """Visualize localization result."""
        map_node_with_max_value, high_similarity_set, similarity = result

        print("Max value: ", similarity[map_node_with_max_value], "   Node: ", map_node_with_max_value)

        for node in self.graph.nodes():
            draw_point_from_node(map_image, self.graph, node)

        for node in high_similarity_set:
            highlight_point_from_node(map_image, self.graph, node, (0, 0, 122))

        highlight_point_from_node(map_image, self.graph, map_node_with_max_value, (255, 255, 0))

        return map_image

    def iterate_localization_with_sample(self, recolored_topdown_map):
        """Execute localization & visualize with test sample iteratively."""
        accuracy_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i, sample_embedding in enumerate(self.sample_embedding_mat):
            if self.is_detection:
                detection_result = self.object_pyramid.sample_detection_result[f"{i:06d}"]
                result = self.localize_with_observation(sample_embedding, detection_result)
            else:
                result = self.localize_with_observation(sample_embedding)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]

            accuracy = self.evaluate_accuracy(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2 = self.evaluate_node_distance(result[0], grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

            if self.is_visualize:
                print("Sample No.: ", i)
                print("Accuracy", accuracy)
                print("Pose D: ", d1)
                print("Node D: ", d2)

                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
                map_image = self.visualize_on_map(map_image, result)
                draw_point_from_grid_pos(map_image, grid_pos, (0, 255, 0))

                cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("localization", 1152, 1152)
                cv2.imshow("localization", map_image)

                key = cv2.waitKey()

                if key == ord("n"):
                    break

        return accuracy_list, d1_list, d2_list, i + 1

    def get_ground_truth_nearest_node(self, grid_pos):
        """Get the nearest node by Euclidean distance."""
        current_pos_mat = np.zeros([len(self.graph.nodes()), 2])
        current_pos_mat[:] = grid_pos
        distance_set = np.linalg.norm(self.map_pos_mat - current_pos_mat, axis=1)
        nearest_node = np.argmin(distance_set)

        return nearest_node

    def evaluate_pos_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from current position?"""
        predicted_grid_pos = self.graph.nodes[map_node_with_max_value]["o"]
        distance = np.linalg.norm(predicted_grid_pos - grid_pos)

        return distance

    def evaluate_node_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from the ground-truth nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)
        ground_truth_nearest_node_pos = self.graph.nodes[ground_truth_nearest_node]["o"]
        predicted_nearest_node_pos = self.graph.nodes[map_node_with_max_value]["o"]
        distance = np.linalg.norm(ground_truth_nearest_node_pos - predicted_nearest_node_pos)

        return distance

    def evaluate_accuracy(self, map_node_with_max_value, grid_pos):
        """Is it the nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        try:
            step = len(nx.shortest_path(self.graph, ground_truth_nearest_node, map_node_with_max_value))
            result = step <= 10
        except nx.exception.NetworkXNoPath:
            result = False

        return result
