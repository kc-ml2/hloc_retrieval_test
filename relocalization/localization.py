import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config.algorithm_config import NetworkConstant
from config.env_config import DataConfig, PathConfig
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
        binary_topdown_map=None,
        load_cache=True,
        instance_only=False,
        visualize=False,
        sparse_map=False,
        num_views=1,
    ):
        """Initialize localization instance with specific model & map data."""
        self.map_obs_dir = map_obs_dir
        self.sample_dir = sample_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            self.top_network = top_network
            self.bottom_network = bottom_network

        # Set file name from sim & record name
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.map_embedding_file = os.path.join(observation_path, f"siamese_embedding_{map_cache_index}.npy")

        self.obs_path = os.path.basename(observation_path)
        self.map_id = map_cache_index[-1]

        if self.sample_dir:
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_embedding_file = os.path.join(observation_path, f"siamese_embedding_{sample_cache_index}.npy")
            self.sample_pos_record_file = os.path.join(observation_path, f"pos_record_{sample_cache_index}.json")

            with open(self.sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_pos_record = json.load(f)

        if instance_only is False:
            # Initialize map graph from binary topdown map
            self.graph = topdown_map_to_graph(binary_topdown_map, DataConfig.REMOVE_ISOLATED, sparse_map=sparse_map)

            # Initialize emny matrix and parameters for handling embeddings
            self.num_views = num_views
            self.num_map_embedding = len(os.listdir(os.path.normpath(map_obs_dir)))
            self.num_map_graph_nodes = len(self.graph.nodes())
            self.dimension_map_embedding = NetworkConstant.NUM_EMBEDDING
            self.input_embedding_mat = np.zeros(
                (self.num_views * self.num_map_graph_nodes, 2 * self.dimension_map_embedding)
            )

        # Load cached npy file if the flag is true
        if load_cache and (instance_only is False):
            self._load_cache()

            # Initialize graph map from binary topdown map image
            self.map_pos_mat = np.zeros([self.num_map_graph_nodes, 2])

            for node_id in range(self.num_map_graph_nodes):
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

    def _load_cache(self):
        """Load cached npy embedding file from map & sample observations."""
        with open(self.map_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
            map_embedding_mat = np.load(f)
            if np.shape(map_embedding_mat)[0] != self.num_map_embedding:
                raise ValueError("Dimension of the cache file is different with map record.")

        if self.sample_dir:
            with open(self.sample_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
                self.sample_embedding_mat = np.load(f)

        self.input_embedding_mat[:, : self.dimension_map_embedding] = map_embedding_mat[
            : self.num_views * self.num_map_graph_nodes
        ]

    def calculate_embedding_from_observation(self, observation):
        """Calculate siamese embedding from observation image with botton network."""
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            regulized_img = tf.image.convert_image_dtype(observation, tf.float32)
            obs_embedding = self.bottom_network.predict_on_batch(np.expand_dims(regulized_img, axis=0))

        obs_embedding = np.squeeze(obs_embedding)

        return obs_embedding

    def localize_with_observation(self, observation_embedding, current_img=None):
        """Get localization result of current map according to input observation embedding."""

        self.input_embedding_mat[:, self.dimension_map_embedding :] = observation_embedding

        with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
            predictions = self.top_network.predict_on_batch(self.input_embedding_mat)

        similarity = predictions[:, 1]
        map_node_with_max_value = np.argmax(similarity) // self.num_views

        if self.num_views == 1:
            high_similarity_set = sorted(range(len(similarity)), key=lambda k: similarity[k])[-30:]
        else:
            high_similarity_set_unfolded = sorted(range(len(similarity)), key=lambda k: similarity[k])[-30:]
            high_similarity_set = [high_node // self.num_views for high_node in high_similarity_set_unfolded]

        if current_img is not None:
            orb_distance_list = []
            for high_id in high_similarity_set:
                if self.num_views == 1:
                    predicted_path = os.path.join(self.map_obs_dir, f"{high_id:06d}.jpg")
                    predicted_img = cv2.imread(predicted_path)
                else:
                    frame_list = []
                    for frame_idx in range(self.num_views):
                        frame_path = os.path.join(self.map_obs_dir, f"{high_id:06d}_{frame_idx}.jpg")
                        frame_list.append(cv2.imread(frame_path))

                    predicted_img = np.concatenate(frame_list, axis=1)

                _, sample_des = self.orb.detectAndCompute(current_img, None)
                _, predicted_des = self.orb.detectAndCompute(predicted_img, None)

                predicted_matches = self.bf.match(sample_des, predicted_des)
                predicted_matches = sorted(predicted_matches, key=lambda x: x.distance)
                predicted_matches = predicted_matches[:30]

                orb_distance_list.append(sum([match.distance for match in predicted_matches]))

            min_distance_index = np.argmin(orb_distance_list)
            map_node_with_max_value = high_similarity_set[min_distance_index]

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
        high_dist_list = []
        high_simil_list = []

        for i, sample_embedding in enumerate(self.sample_embedding_mat):
            sample_path = os.path.join(self.sample_dir, f"{i:06d}.jpg")
            sample_img = cv2.imread(sample_path)
            # result = self.localize_with_observation(sample_embedding, current_img=sample_img)
            result = self.localize_with_observation(sample_embedding)

            grid_pos = self.sample_pos_record[f"{i:06d}_grid"]

            accuracy = self.evaluate_accuracy(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, gt_node = self.evaluate_node_distance(result[0], grid_pos)

            accuracy_list.append(accuracy)
            d1_list.append(d1)
            d2_list.append(d2)

            for high_node in result[1]:
                high_dist, _ = self.evaluate_node_distance(high_node, grid_pos)
                high_dist_list.append(high_dist)

                node_idx = high_node * self.num_views

                single_node_distance_list = []
                for d in range(self.num_views):
                    feature_distance = 1.0 - result[2][node_idx + d]
                    single_node_distance_list.append(feature_distance)
                high_simil_list.append(min(single_node_distance_list))

            if self.is_visualize:
                print("Sample No.: ", i)
                print("Accuracy", accuracy)
                print("Pose D: ", d1)
                print("Node D: ", d2)
                print("Predicted node similarity: ", result[2][result[0]])
                print("GT node similarity: ", result[2][gt_node])

                if accuracy is False:
                    if self.num_views == 1:
                        predicted_path = os.path.join(self.map_obs_dir, f"{result[0]:06d}.jpg")
                        true_path = os.path.join(self.map_obs_dir, f"{gt_node:06d}.jpg")
                        predicted_img = cv2.imread(predicted_path)
                        true_img = cv2.imread(true_path)
                    else:
                        predicted_path_list = [
                            os.path.join(self.map_obs_dir, f"{result[0]:06d}_{idx}.jpg")
                            for idx in range(self.num_views)
                        ]
                        true_path_list = [
                            os.path.join(self.map_obs_dir, f"{gt_node:06d}_{idx}.jpg") for idx in range(self.num_views)
                        ]
                        predicted_img_list = [cv2.imread(predicted_path) for predicted_path in predicted_path_list]
                        predicted_img = np.concatenate(predicted_img_list, axis=1)
                        true_img_list = [cv2.imread(true_path) for true_path in true_path_list]
                        true_img = np.concatenate(true_img_list, axis=1)

                    sample_path = os.path.join(self.sample_dir, f"{i:06d}.jpg")
                    sample_img = cv2.imread(sample_path)
                    blank_img = np.zeros([10, predicted_img.shape[1], 3], dtype=np.uint8)

                    if self.num_views == 1:
                        match_img = np.concatenate([sample_img, blank_img, predicted_img, blank_img, true_img], axis=0)
                    else:
                        padded_img = np.full(np.shape(predicted_img), 255, dtype=np.uint8)
                        x_offset = predicted_img.shape[1] // 2 - sample_img.shape[1] // 2
                        padded_img[:, x_offset : x_offset + sample_img.shape[1], :] = sample_img
                        match_img = np.concatenate([padded_img, blank_img, predicted_img, blank_img, true_img], axis=0)

                    cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("localization", 1700, 700)
                    cv2.imshow("localization", match_img)

                # Visualize position on map
                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
                map_image = self.visualize_on_map(map_image, result)
                draw_point_from_grid_pos(map_image, grid_pos, (0, 255, 0))

                cv2.namedWindow("map", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("map", 512, 512)
                cv2.imshow("map", map_image)

                key = cv2.waitKey()

                if key == ord("n"):
                    break

        # plot for distance distribution visualization
        high_dist_list = [dist / 10 for dist in high_dist_list]

        plt.clf()
        # # plt.ylim((0, 150))
        # plt.xlim((0, 1.0))
        # plt.scatter(high_simil_list, high_dist_list, marker=".", s=1)
        # plt.xlabel("distance between two images using Siamese network")
        # plt.ylabel("actual distance between two positions [m]")
        # # plt.show()
        # plt.savefig(f"./output_fig/siamese/{self.obs_path}_{self.map_id}.jpg", dpi=300)

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
