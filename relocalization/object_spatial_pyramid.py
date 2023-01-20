import json
import os

import numpy as np

from config.algorithm_config import NetworkConstant


class ObjectSpatialPyramid:
    """Class for object-based spatial pyramid matching."""

    def __init__(self, map_obs_dir, graph, sample_dir=None, num_support=80, load_cache=False):
        """Initialize spatial pyramid instance with specific map data."""
        # Initialize path
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.map_detection_file = os.path.join(observation_path, f"object_detection_{map_cache_index}.json")
        self.map_detection_result = None

        self.graph = graph

        # Set parameters
        self.num_support = num_support
        self.num_map_node = len(self.graph.nodes())
        self.pyramid_level = 2
        self.direction_split_per_level = 2
        self.distance_split_per_level = 2

        self.mid_direction_split = self.direction_split_per_level ** (self.pyramid_level - 1)
        self.mid_distance_split = self.distance_split_per_level ** (self.pyramid_level - 1)
        self.low_direction_split = self.direction_split_per_level**self.pyramid_level
        self.low_distance_split = self.distance_split_per_level**self.pyramid_level
        self.mid_pyramid_dim = self.mid_direction_split * self.mid_distance_split
        # self.mid_pyramid_dim = self.direction_split_per_level ** (self.pyramid_level - 1)
        self.low_pyramid_dim = self.low_direction_split * self.low_distance_split
        # self.low_pyramid_dim = self.direction_split_per_level**self.pyramid_level
        self.all_pyramid_dim = 1 + self.low_pyramid_dim + self.mid_pyramid_dim

        self.sample_dir = sample_dir

        if sample_dir:
            # Initialize path
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_detection_file = os.path.join(observation_path, f"object_detection_{sample_cache_index}.json")
            self.sample_detection_result = None

        if load_cache:
            self._load_cache()
            self.map_histogram_batch = np.zeros(
                [self.num_map_node * self.low_direction_split, self.num_support * self.all_pyramid_dim]
            )
            self._generate_map_histogram_batch()

    def _load_cache(self):
        with open(self.map_detection_file, "r") as f:  # pylint: disable=unspecified-encoding
            self.map_detection_result = json.load(f)

        if self.sample_dir:
            with open(self.sample_detection_file, "r") as f:  # pylint: disable=unspecified-encoding
                self.sample_detection_result = json.load(f)

    def _generate_map_histogram_batch(self):
        """Generate map histogram batch with incremental rotation value."""
        for i in range(self.num_map_node):
            current_map_detection = self.map_detection_result[f"{i:06d}"]
            current_map_histogram_set = self.make_spatial_histogram(current_map_detection)
            flatten_total = current_map_histogram_set[0]
            self.map_histogram_batch[i * self.low_direction_split] = flatten_total

            for j in range(self.low_direction_split - 1):
                rotated_histogram_set = self._rotate_histogram(current_map_histogram_set)
                idx = i * self.low_direction_split + j + 1
                self.map_histogram_batch[idx] = rotated_histogram_set[0]
                current_map_histogram_set = rotated_histogram_set

    def _rotate_histogram(self, histogram_set):
        """Rotate histogram by one split step angle."""
        flatten_total, histogram_top, histogram_mid, histogram_low = histogram_set

        if np.sum(flatten_total) == 0:
            pass
        else:
            histogram_low = np.roll(histogram_low, self.low_distance_split, axis=1)

            histogram_mid = np.zeros(histogram_mid.shape)
            histogram_mid = self.convert_low_to_mid(histogram_low, histogram_mid)

            flatten_total = np.concatenate([histogram_top, histogram_mid.flatten(), histogram_low.flatten()])

        return flatten_total, histogram_top, histogram_mid, histogram_low

    def make_spatial_histogram(self, detection_result):
        """Make spatial pyramid histogram for matching."""
        boxes, _, classIDs = detection_result
        spatial_width_interval = NetworkConstant.NET_WIDTH / self.low_direction_split
        spatial_distance_interval = NetworkConstant.NET_HEIGHT / self.low_distance_split

        histogram_low = np.zeros([self.num_support, self.low_pyramid_dim])
        histogram_mid = np.zeros([self.num_support, self.mid_pyramid_dim])
        histogram_top = np.zeros(self.num_support)
        histogram_total = np.concatenate([histogram_top, histogram_mid.flatten(), histogram_low.flatten()])

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                box_height = 255 if box[3] >= NetworkConstant.NET_HEIGHT else box[3]
                spatial_x_pos = int(box[0] // spatial_width_interval)
                spatial_y_pos = int(box_height // spatial_distance_interval)
                pos = self.low_distance_split * spatial_x_pos + spatial_y_pos
                histogram_low[classIDs[i], pos] = histogram_low[classIDs[i], pos] + 1

            histogram_mid = self.convert_low_to_mid(histogram_low, histogram_mid)
            histogram_top = np.sum(histogram_low, axis=1)

            histogram_sum = np.sum(histogram_top, axis=0)
            histogram_low = histogram_low / histogram_sum * 0.5
            histogram_mid = histogram_mid / histogram_sum * 0.25
            histogram_top = histogram_top / histogram_sum * 0.25

            histogram_total = np.concatenate([histogram_top, histogram_mid.flatten(), histogram_low.flatten()])

        return histogram_total, histogram_top, histogram_mid, histogram_low

    def convert_low_to_mid(self, histogram_low, black_histogram_mid):
        for i, hg in enumerate(histogram_low):
            temp = hg.reshape(-1, self.distance_split_per_level).sum(1).reshape(-1, self.mid_distance_split)
            temp = np.transpose(temp)
            temp = temp.reshape(-1, self.direction_split_per_level).sum(1).reshape(-1, self.mid_direction_split)
            temp = np.transpose(temp)
            black_histogram_mid[i] = temp.flatten()

        histogram_mid = black_histogram_mid

        return histogram_mid

    # TODO
    # def calculate_histogram_intersection(self, histogram_a, histogram_b):
    #     """Calculate histogram intersection."""

    # def calculate_batch_histogram_intersection(self, map_histogram, current_histogram):
    #     """Calculate histogram intersection with all histogram from map observations."""
