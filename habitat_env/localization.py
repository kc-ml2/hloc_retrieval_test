import json
import os

import cv2
import numpy as np
import tensorflow as tf

from config.algorithm_config import TestConstant
from config.env_config import PathConfig
from utils.habitat_utils import draw_point_from_grid_pos, draw_point_from_node, highlight_point_from_node


class Localization:
    """Class for localization methods according to the given map."""

    def __init__(self, sim, top_network, bottom_network, map_obs_dir, sample_dir=None):
        """Initialize localization instance with specific model & map data."""
        self.sim = sim

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

    def calculate_embedding_from_observation(self):
        """Calculate siamese embedding from observation image with botton network."""

    def localize_with_observation(self, observation_embedding):
        """Get localization result of current map according to input observation embedding."""
        self.input_embedding_mat[:, self.dimension_map_embedding :] = observation_embedding

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            predictions = self.top_network.predict_on_batch(self.input_embedding_mat)

        similarity = predictions[:, 1]
        map_node_with_max_value = np.argmax(similarity)
        high_similarity_set = [
            id for id in range(len(similarity)) if similarity[id] > TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
        ]

        return map_node_with_max_value, high_similarity_set, similarity

    def iterate_localization_with_sample(self, graph, recolored_topdown_map):
        """Visualize localization result from test sample iteratively."""
        for i, sample_embedding in enumerate(self.sample_embedding_mat):
            print("Sample No.: ", i)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]
            map_node_with_max_value, high_similarity_set, similarity = self.localize_with_observation(sample_embedding)

            map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)

            print("Max value: ", similarity[map_node_with_max_value], "   Node: ", map_node_with_max_value)

            for node in graph.nodes():
                draw_point_from_node(map_image, graph, node)

            for node in high_similarity_set:
                highlight_point_from_node(map_image, graph, node, (0, 0, 122))

            highlight_point_from_node(map_image, graph, map_node_with_max_value, (255, 255, 0))
            draw_point_from_grid_pos(map_image, grid_pos, (0, 255, 0))

            cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("localization", 1152, 1152)
            cv2.imshow("localization", map_image)

            cv2.waitKey()
