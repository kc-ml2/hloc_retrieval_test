import os

import numpy as np

from config.algorithm_config import NetworkConstant


class ObjectSpatialPyramid:
    """Class for object-based spatial pyramid matching."""

    def __init__(self, map_obs_dir, sample_dir=None, num_support=80):
        """Initialize spatial pyramid instance with specific map data."""
        observation_path = os.path.dirname(os.path.normpath(map_obs_dir))
        map_cache_index = os.path.basename(os.path.normpath(map_obs_dir))
        self.map_histogram_file = os.path.join(observation_path, f"spatial_histogram_{map_cache_index}.npy")

        if sample_dir:
            sample_cache_index = os.path.basename(os.path.normpath(sample_dir))
            self.sample_histogram_file = os.path.join(observation_path, f"spatial_histogram_{sample_cache_index}.npy")

        self.num_support = num_support

    def make_spatial_histogram(self, detection_result, pyramid_level=2, split_per_level=4):
        """Make spatial pyramid histogram for matching."""
        boxes, _, classIDs = detection_result
        spatial_width_interval = NetworkConstant.NET_WIDTH / split_per_level**pyramid_level

        histogram_low = np.zeros([self.num_support, split_per_level**pyramid_level])
        histogram_mid = np.zeros([self.num_support, split_per_level ** (pyramid_level - 1)])
        histogram_top = np.zeros(self.num_support)

        histogram_total = np.concatenate([histogram_top, histogram_mid.flatten(), histogram_low.flatten()])

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                spatial_pos = int(box[0] // spatial_width_interval)
                histogram_low[classIDs[i], spatial_pos] = histogram_low[classIDs[i], spatial_pos] + 1

            for i in range(split_per_level):
                start_idx = i * split_per_level
                end_idx = start_idx + 4
                histogram_mid[:, i] = np.sum(histogram_low[:, start_idx:end_idx], axis=1)

            histogram_top = np.sum(histogram_low, axis=1)

            histogram_sum = np.sum(histogram_top, axis=0)
            histogram_low = histogram_low / histogram_sum * 0.5
            histogram_mid = histogram_mid / histogram_sum * 0.25
            histogram_top = histogram_top / histogram_sum * 0.25

            histogram_total = np.concatenate([histogram_top, histogram_mid.flatten(), histogram_low.flatten()])

        return histogram_total

    # TODO
    # def calculate_histogram_intersection(self, histogram_a, histogram_b):
    #     """Calculate histogram intersection."""

    # def calculate_batch_histogram_intersection(self, map_hostogram, current_histogram):
    #     """Calculate histogram intersection with all histogram from map observations."""
