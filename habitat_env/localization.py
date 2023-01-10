import json
import os

import cv2
import numpy as np
import tensorflow as tf

from config.algorithm_config import NetworkConstant, TestConstant
from config.env_config import DataConfig, PathConfig
from utils.habitat_utils import draw_point_from_grid_pos, draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import topdown_map_to_graph


class Localization:
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        top_network,
        bottom_network,
        binary_topdown_map,
        map_obs_dir,
        sample_dir=None,
        is_detection=False,
        num_support=80,
    ):
        """Initialize localization instance with specific model & map data."""
        self.graph = topdown_map_to_graph(binary_topdown_map, DataConfig.REMOVE_ISOLATED)
        self.is_detection = is_detection

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            self.top_network = top_network
            self.bottom_network = bottom_network

        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        map_embedding_file = os.path.join(observation_path, f"siamese_embedding_{map_cache_index}.npy")

        with open(map_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
            map_embedding_mat = np.load(f)

        if sample_dir:
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            sample_embedding_file = os.path.join(observation_path, f"siamese_embedding_{sample_cache_index}.npy")
            sample_pos_record_file = os.path.join(observation_path, f"pos_record_{sample_cache_index}.json")

            with open(sample_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
                self.sample_embedding_mat = np.load(f)
            with open(sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_pos_record = json.load(f)

        num_map_embedding = np.shape(map_embedding_mat)[0]
        self.dimension_map_embedding = np.shape(map_embedding_mat)[1]

        self.input_embedding_mat = np.zeros((num_map_embedding, 2 * self.dimension_map_embedding))
        self.input_embedding_mat[:, : self.dimension_map_embedding] = map_embedding_mat

        self.map_pos_mat = np.zeros([len(self.graph.nodes()), 2])

        for node_id in self.graph.nodes():
            self.map_pos_mat[node_id] = self.graph.nodes[node_id]["o"]

        if is_detection:
            self.num_support = num_support

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
        if self.is_detection is True and detection_result is None:
            raise ValueError("Detection result is required for localization with object detection.")

        self.input_embedding_mat[:, self.dimension_map_embedding :] = observation_embedding

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            predictions = self.top_network.predict_on_batch(self.input_embedding_mat)

        similarity = predictions[:, 1]
        map_node_with_max_value = np.argmax(similarity)
        high_similarity_set = [
            id for id in range(len(similarity)) if similarity[id] > TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
        ]

        if detection_result is not None:
            histogram = self.make_spatial_histogram(detection_result)

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
        for i, sample_embedding in enumerate(self.sample_embedding_mat):
            print("Sample No.: ", i)

            map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)

            result = self.localize_with_observation(sample_embedding)
            map_image = self.visualize_on_map(map_image, result)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]
            draw_point_from_grid_pos(map_image, grid_pos, (0, 255, 0))

            print("Accuracy", self.evaluate_accuracy(result[0], grid_pos))
            print("Pose D: ", self.evaluate_pos_distance(result[0], grid_pos))
            print("Node D: ", self.evaluate_node_distance(result[0], grid_pos))

            cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("localization", 1152, 1152)
            cv2.imshow("localization", map_image)

            key = cv2.waitKey()

            if key == ord("n"):
                break

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

        return ground_truth_nearest_node == map_node_with_max_value

    def make_spatial_histogram(self, detection_result, pyramid_level=2, split_per_level=4, height_criteria=100):
        """Make spatial pyramid histogram for matching."""
        boxes, _, classIDs = detection_result
        spatial_width_interval = NetworkConstant.NET_WIDTH / split_per_level**pyramid_level

        histogram_low = np.zeros([self.num_support, split_per_level**pyramid_level])
        histogram_middle = np.zeros([self.num_support, split_per_level ** (pyramid_level - 1)])
        histogram_top = np.zeros(self.num_support)

        histogram_total = (
            histogram_top.tolist() + histogram_middle.flatten().tolist() + histogram_low.flatten().tolist()
        )

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                spatial_pos = int(box[0] // spatial_width_interval)
                histogram_low[classIDs[i], spatial_pos] = histogram_low[classIDs[i], spatial_pos] + 1

            for i in range(split_per_level):
                start_idx = i * split_per_level
                end_idx = start_idx + 4
                histogram_middle[:, i] = np.sum(histogram_low[:, start_idx:end_idx], axis=1)

            histogram_top = np.sum(histogram_low, axis=1)

            histogram_sum = np.sum(histogram_top, axis=0)
            histogram_low = histogram_low / histogram_sum
            histogram_middle = histogram_middle / histogram_sum
            histogram_top = histogram_top / histogram_sum

            histogram_total = (
                histogram_top.tolist() + histogram_middle.flatten().tolist() + histogram_low.flatten().tolist()
            )

        return histogram_total
