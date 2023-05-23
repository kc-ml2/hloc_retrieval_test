import json
import os
from abc import abstractmethod

import cv2
import numpy as np

from utils.habitat_utils import draw_point_from_grid_pos, draw_point_from_node, highlight_point_from_node
from utils.skeletonize_utils import topdown_map_to_graph


class LocalizationBase:
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        config,
        map_obs_dir,
        query_dir,
        binary_topdown_map=None,
        visualize=False,
        sparse_map=False,
    ):
        """Initialize localization instance with specific model & map data."""
        self.config = config
        self.map_obs_dir = map_obs_dir
        self.query_dir = query_dir
        self.is_visualize = visualize
        self.is_sparse_map = sparse_map

        if config.CamConfig.IMAGE_CONCAT is True:
            self.num_frames_per_node = 1
        else:
            self.num_frames_per_node = config.CamConfig.NUM_CAMERA

        # Set file name from sim & record name
        image_dir_by_scene = os.path.dirname(os.path.normpath(map_obs_dir))
        map_dirname = os.path.basename(os.path.normpath(map_obs_dir))
        self.level = map_dirname[-1]

        self.test_on_sim = config.DataConfig.DATA_FROM_SIM
        if self.test_on_sim:
            self.scene_dirname = os.path.basename(image_dir_by_scene)
        else:
            self.scene_dirname = ""

        self.query_dirname = os.path.basename(os.path.normpath(query_dir))
        self.query_pos_record_file = os.path.join(
            image_dir_by_scene, f"{config.PathConfig.POS_RECORD_FILE_PREFIX}_{self.query_dirname}.json"
        )
        self.map_pos_record_file = os.path.join(
            image_dir_by_scene, f"{config.PathConfig.POS_RECORD_FILE_PREFIX}_{map_dirname}.json"
        )

        self.sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
        self.sorted_query_file = sorted(os.listdir(query_dir))
        self.num_query_graph_nodes = len(self.sorted_query_file)
        self.num_map_embedding = len(os.listdir(os.path.normpath(map_obs_dir)))

        with open(self.query_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
            self.query_pos_record = json.load(f)

        # If test data is from sim, generate graph to get coordinate info
        # Else, load coordinate info from record file
        if self.test_on_sim:
            # Initialize map graph from binary topdown map
            self.graph = topdown_map_to_graph(
                binary_topdown_map, config.DataConfig.REMOVE_ISOLATED, sparse_map=sparse_map
            )
            # Initialize empty matrix and parameters for handling embeddings
            self.num_map_graph_nodes = len(self.graph.nodes())
        else:
            self.num_map_graph_nodes = len(self.sorted_map_obs_file)
            # self.num_map_graph_nodes = int(self.num_map_embedding / self.num_frames_per_node)

            with open(self.map_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.map_pos_record = json.load(f)

        # Initialize graph map from binary topdown map image
        self.map_pos_mat = np.zeros([self.num_map_graph_nodes, 2])

        for node_id in range(self.num_map_graph_nodes):
            if self.test_on_sim:
                self.map_pos_mat[node_id] = self.graph.nodes[node_id]["o"]
            else:
                self.map_pos_mat[node_id] = np.array(self.map_pos_record[f"{node_id:06d}_grid"])

    @abstractmethod
    def localize_with_observation(self, query_id: str):
        pass
    
    def iterate_localization_with_query(self, recolored_topdown_map=None):
        """Execute localization & visualize with test query iteratively."""
        recall_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i in range(self.num_query_graph_nodes):
            query_id = f"{i:06d}"
            result = self.localize_with_observation(query_id)

            grid_pos = np.array(self.query_pos_record[f"{i:06d}_grid"])

            recall = self.evaluate_recall(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, gt_node = self.evaluate_node_distance(result[0], grid_pos)

            recall_list.append(recall)
            d1_list.append(d1)
            d2_list.append(d2)

            if self.is_visualize:
                print("query No.: ", i)
                print("Recall", recall)
                print("Pose D: ", d1)
                print("Node D: ", d2)

                if recall is False:
                    self.display_observation_comparison(result, gt_node, i)

                # Visualize position on map
                if recolored_topdown_map is not None and self.test_on_sim:
                    map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
                    map_image = self.visualize_on_map(map_image, result)
                    draw_point_from_grid_pos(map_image, grid_pos, (0, 255, 0))

                    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("map", 512, 512)
                    cv2.imshow("map", map_image)

                key = cv2.waitKey()
                if key == ord("n"):
                    break

        k = i + 1
        print("Temporay Recall: ", sum(recall_list) / k)

        return recall_list, d1_list, d2_list, i + 1

    def visualize_on_map(self, map_image, result):
        """Visualize localization result."""
        map_node_with_max_value, high_similarity_set = result

        for node in self.graph.nodes():
            draw_point_from_node(map_image, self.graph, node)

        for node in high_similarity_set:
            highlight_point_from_node(map_image, self.graph, node, (0, 0, 122))

        highlight_point_from_node(map_image, self.graph, map_node_with_max_value, (255, 255, 0))

        return map_image

    def display_observation_comparison(self, result, gt_node, i):
        """Display current. predicted, ground-truth observations simultaneously."""
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

    def get_ground_truth_nearest_node(self, grid_pos):
        """Get the nearest node by Euclidean distance."""
        current_pos_mat = np.zeros([self.num_map_graph_nodes, 2])
        current_pos_mat[:] = grid_pos
        distance_set = np.linalg.norm(self.map_pos_mat - current_pos_mat, axis=1)
        nearest_node = np.argmin(distance_set)

        return nearest_node

    def evaluate_pos_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from current position?"""
        if self.test_on_sim:
            predicted_grid_pos = self.graph.nodes[map_node_with_max_value]["o"]
        else:
            predicted_grid_pos = np.array(self.map_pos_record[f"{map_node_with_max_value:06d}_grid"])

        distance = np.linalg.norm(predicted_grid_pos - grid_pos)

        return distance

    def evaluate_node_distance(self, map_node_with_max_value, grid_pos):
        """How far is the predicted node from the ground-truth nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        if self.test_on_sim:
            ground_truth_nearest_node_pos = self.graph.nodes[ground_truth_nearest_node]["o"]
            predicted_nearest_node_pos = self.graph.nodes[map_node_with_max_value]["o"]
        else:
            ground_truth_nearest_node_pos = np.array(self.map_pos_record[f"{ground_truth_nearest_node:06d}_grid"])
            predicted_nearest_node_pos = np.array(self.map_pos_record[f"{map_node_with_max_value:06d}_grid"])

        distance = np.linalg.norm(ground_truth_nearest_node_pos - predicted_nearest_node_pos)

        return distance, ground_truth_nearest_node

    def evaluate_recall(self, map_node_with_max_value, grid_pos):
        """Is it the nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        if self.test_on_sim:
            ground_truth_pos = self.graph.nodes()[ground_truth_nearest_node]["o"]
            estimated_pos = self.graph.nodes()[map_node_with_max_value]["o"]
        else:
            ground_truth_pos = np.array(self.map_pos_record[f"{ground_truth_nearest_node:06d}_grid"])
            estimated_pos = np.array(self.map_pos_record[f"{map_node_with_max_value:06d}_grid"])

        step = np.linalg.norm(ground_truth_pos - estimated_pos)

        result = step <= 10

        return bool(result)
