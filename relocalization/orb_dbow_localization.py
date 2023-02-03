import json
import os
import time

import cv2
import numpy as np

from config.env_config import DataConfig
from relocalization.dbow import dbow
from utils.habitat_utils import draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import topdown_map_to_graph


class OrbDbowLocalization:
    """Class for localization methods with orb bag of the binary words method."""

    def __init__(
        self,
        map_obs_dir,
        sample_dir=None,
        binary_topdown_map=None,
        visualize=False,
        sparse_map=False,
    ):
        """Initialize localization instance with specific model & map data."""
        self.map_obs_dir = map_obs_dir
        self.sample_dir = sample_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))

        # Make list to iterate
        sorted_map_obs_id = sorted(os.listdir(map_obs_dir))
        map_obs_file_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_id]

        if self.sample_dir:
            sorted_test_sample_file = sorted(os.listdir(sample_dir))
            self.sample_list = [sample_dir + os.sep + file for file in sorted_test_sample_file]
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_pos_record_file = os.path.join(observation_path, f"pos_record_{sample_cache_index}.json")

            with open(self.sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_pos_record = json.load(f)

        # Initialize map graph from binary topdown map
        self.graph = topdown_map_to_graph(binary_topdown_map, DataConfig.REMOVE_ISOLATED, sparse_map=sparse_map)
        self.num_map_graph_nodes = len(self.graph.nodes())

        # Initialize graph map from binary topdown map image
        self.map_pos_mat = np.zeros([len(self.graph.nodes()), 2])
        for node_id in self.graph.nodes():
            self.map_pos_mat[node_id] = self.graph.nodes[node_id]["o"]

        # Initiate ORB detector
        self.orb = cv2.ORB_create(
            # nfeatures=40,
            nfeatures=100,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        # self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if self.is_sparse_map:
            map_obs_file_list = map_obs_file_list[: self.num_map_graph_nodes]

        orb_db_images = []
        for map_obs_file in map_obs_file_list:
            orb_db_images.append(cv2.imread(map_obs_file))

        start = time.time()

        # Create Vocabulary
        print("Creating vocabulary...")
        n_clusters = 10
        depth = 2
        vocabulary = dbow.Vocabulary(orb_db_images, n_clusters, depth, self.orb)

        # Create a database
        print("Creating database...")
        i = 0
        self.db = dbow.Database(vocabulary)
        for i, image in enumerate(orb_db_images):
            print(i, end="\r", flush=True)
            _, descs = self.orb.detectAndCompute(image, None)
            descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
            self.db.add(descs)
        print(i)

        end = time.time()
        print("Done DB generation. Elapsed time: ", end - start)

    def localize_with_observation(self, current_img):
        """Get localization result of current map according to input observation embedding."""
        # Query the database
        _, descs = self.orb.detectAndCompute(current_img, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        scores = self.db.query(descs)
        map_node_with_max_value = np.argmax(scores)

        return map_node_with_max_value

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

    def iterate_localization_with_sample(self):
        """Execute localization & visualize with test sample iteratively."""
        accuracy_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i, sample_path in enumerate(self.sample_list):
            sample_img = cv2.imread(sample_path)
            map_node_with_max_value = self.localize_with_observation(sample_img)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]

            accuracy = self.evaluate_accuracy(map_node_with_max_value, grid_pos)
            d1 = self.evaluate_pos_distance(map_node_with_max_value, grid_pos)
            d2, _ = self.evaluate_node_distance(map_node_with_max_value, grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

        k = i + 1

        print("Temporay Accuracy: ", sum(accuracy_list) / k)
        print("Temporay Distance 1: ", (sum(d1_list) / k) * 0.1)
        print("Temporay Distance 2: ", (sum(d2_list) / k) * 0.1)

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

        return distance, ground_truth_nearest_node

    def evaluate_accuracy(self, map_node_with_max_value, grid_pos):
        """Is it the nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        ground_truth_pos = self.graph.nodes()[ground_truth_nearest_node]["o"]
        estimated_pos = self.graph.nodes()[map_node_with_max_value]["o"]

        step = np.linalg.norm(ground_truth_pos - estimated_pos)

        result = step <= 10

        return bool(result)
