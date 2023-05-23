import json
import os
import time

import cv2
import numpy as np


class OrbMatchingLocalizationRealWorld:
    """Class for localization methods with orb bag of the binary words method."""

    def __init__(
        self,
        config,
        map_obs_dir,
        query_dir=None,
        visualize=False,
        sparse_map=False,
    ):
        """Initialize localization instance with specific model & map data."""
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
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.obs_path = os.path.basename(image_dir_by_scene)
        self.map_id = map_cache_index[-1]

        # Open map pose record file
        self.map_pos_record_file = os.path.join(image_dir_by_scene, f"pos_record_map_{self.map_id}.json")
        with open(self.map_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
            self.map_pos_record = json.load(f)

        # Make list to iterate
        sorted_map_obs_id = sorted(os.listdir(map_obs_dir))
        map_obs_file_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_id]

        if self.query_dir:
            sorted_test_query_file = sorted(os.listdir(query_dir))
            self.query_list = [query_dir + os.sep + file for file in sorted_test_query_file]
            query_cache_index = os.path.basename(os.path.normpath(query_dir))
            self.query_pos_record_file = os.path.join(image_dir_by_scene, f"pos_record_{query_cache_index}.json")

            with open(self.query_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.query_pos_record = json.load(f)

        # Initialize emny matrix and parameters for handling embeddings
        self.num_map_embedding = len(os.listdir(os.path.normpath(map_obs_dir)))
        self.num_map_graph_nodes = int(self.num_map_embedding / self.num_frames_per_node)

        # Initialize graph map from binary topdown map image
        self.map_pos_mat = np.zeros([self.num_map_graph_nodes, 2])

        for node_id in range(self.num_map_graph_nodes):
            self.map_pos_mat[node_id] = self.map_pos_record[f"{node_id:06d}_grid"]

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
            map_obs_file_list = map_obs_file_list[: self.num_frames_per_node * self.num_map_graph_nodes]

        for i in range(0, len(map_obs_file_list), self.num_frames_per_node):
            frame_list = []
            for k in range(self.num_frames_per_node):
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

        for query_obs_file in self.query_list:
            query_image = cv2.imread(query_obs_file)
            _, query_des = self.orb.detectAndCompute(query_image, None)
            if query_des is None:
                query_des = np.zeros([1, 32], dtype=np.uint8)
                num_nonetype = num_nonetype + 1
            self.desc_query.append(query_des)

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

    def iterate_localization_with_query(self):
        """Execute localization & visualize with test query iteratively."""
        recall_list = []
        d1_list = []
        d2_list = []
        i = 0

        for i in range(len(self.query_list)):
            print("query index: ", i)
            result = self.localize_with_observation(i)

            grid_pos = np.array(self.query_pos_record[f"{i:06d}_grid"])

            recall = self.evaluate_recall(result[0], grid_pos)
            d1 = self.evaluate_pos_distance(result[0], grid_pos)
            d2, _ = self.evaluate_node_distance(result[0], grid_pos)

            recall_list.append(recall)
            d1_list.append(d1)
            d2_list.append(d2)

        k = i + 1
        print("Temporay Recall: ", sum(recall_list) / k)

        return recall_list, d1_list, d2_list, i + 1

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

    def evaluate_recall(self, map_node_with_max_value, grid_pos):
        """Is it the nearest node?"""
        ground_truth_nearest_node = self.get_ground_truth_nearest_node(grid_pos)

        ground_truth_pos = self.map_pos_record[f"{ground_truth_nearest_node:06d}_grid"]
        estimated_pos = self.map_pos_record[f"{map_node_with_max_value:06d}_grid"]

        step = np.linalg.norm(np.array(ground_truth_pos) - np.array(estimated_pos))

        result = step <= 10

        return bool(result)
