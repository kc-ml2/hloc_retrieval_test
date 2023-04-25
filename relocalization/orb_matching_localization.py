import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.habitat_utils import draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import topdown_map_to_graph


class OrbMatchingLocalization:
    """Class for localization methods with orb bag of the binary words method."""

    def __init__(
        self,
        map_obs_dir,
        sample_dir=None,
        binary_topdown_map=None,
        visualize=False,
        sparse_map=False,
        num_frames_per_node=1,
    ):
        """Initialize localization instance with specific model & map data."""
        self.map_obs_dir = map_obs_dir
        self.sample_dir = sample_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))

        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.obs_path = os.path.basename(observation_path)
        self.map_id = map_cache_index[-1]

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
        self.graph = topdown_map_to_graph(binary_topdown_map, config.DataConfig.REMOVE_ISOLATED, sparse_map=sparse_map)
        self.num_map_graph_nodes = len(self.graph.nodes())

        # Initialize graph map from binary topdown map image
        self.map_pos_mat = np.zeros([len(self.graph.nodes()), 2])
        for node_id in self.graph.nodes():
            self.map_pos_mat[node_id] = self.graph.nodes[node_id]["o"]

        # Initiate ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=200,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        start = time.time()
        self.desc_db = []
        num_nonetype = 0

        if self.is_sparse_map:
            map_obs_file_list = map_obs_file_list[: num_frames_per_node * self.num_map_graph_nodes]

        for i in range(0, len(map_obs_file_list), num_frames_per_node):
            frame_list = []
            for k in range(num_frames_per_node):
                frame_list.append(cv2.imread(map_obs_file_list[i + k]))
            db_image = np.concatenate(frame_list, axis=1)
            _, db_des = self.orb.detectAndCompute(db_image, None)
            if db_des is None:
                db_des = np.zeros([1, 32], dtype=np.uint8)
                num_nonetype = num_nonetype + 1

            self.desc_db.append(db_des)

        end = time.time()
        print("Number of NoneType in DB: ", num_nonetype)
        print("DB generation elapsed time: ", end - start)

        start = time.time()
        num_nonetype = 0
        self.desc_query = []

        for sample_obs_file in self.sample_list:
            sample_image = cv2.imread(sample_obs_file)
            _, sample_des = self.orb.detectAndCompute(sample_image, None)
            if sample_des is None:
                sample_des = np.zeros([1, 32], dtype=np.uint8)
                num_nonetype = num_nonetype + 1
            self.desc_query.append(sample_des)

        end = time.time()
        print("Number of NoneType in Query: ", num_nonetype)
        print("Query generation elapsed time: ", end - start)

    def localize_with_observation(self, i):
        """Get localization result of current map according to input observation embedding."""
        # Query the database
        print(i, end="\r", flush=True)

        scores = []

        for db_des in self.desc_db:
            matches = self.bf.match(self.desc_query[i], db_des)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:30]

            scores.append(sum([match.distance for match in matches]))

        map_node_with_max_value = np.argmin(scores)
        high_similarity_set = sorted(range(len(scores)), key=lambda i: scores[i])[:30]
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        return map_node_with_max_value, high_similarity_set, normalized_scores

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

        high_dist_list = []
        high_simil_list = []

        for i in range(len(self.sample_list)):
            result = self.localize_with_observation(i)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]

            accuracy = self.evaluate_accuracy(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, _ = self.evaluate_node_distance(result[0], grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

            for high_node in result[1]:
                high_dist, _ = self.evaluate_node_distance(high_node, grid_pos)
                high_dist_list.append(high_dist)
                high_simil_list.append(result[2][high_node])

        high_dist_list = [dist / 10 for dist in high_dist_list]

        plt.clf()
        # # plt.ylim((0, 150))
        # plt.xlim((0, 1.0))
        # plt.scatter(high_simil_list, high_dist_list, marker=".", s=1)
        # plt.xlabel("normalized L2 distance between local features from two images")
        # plt.ylabel("actual distance between two positions [m]")
        # # plt.show()
        # plt.savefig(f"./output_fig/orb/{self.obs_path}_{self.map_id}.jpg", dpi=300)

        k = i + 1
        print("Temporay Accuracy: ", sum(accuracy_list) / k)

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
