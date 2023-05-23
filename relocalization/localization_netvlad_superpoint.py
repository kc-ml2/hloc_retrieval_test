import os

import h5py
import numpy as np

from relocalization.localization_base import LocalizationBase


class LocalizationNetVLADSuperpoint(LocalizationBase):
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
        super().__init__(
            config,
            map_obs_dir,
            query_dir,
            binary_topdown_map=binary_topdown_map,
            visualize=visualize,
            sparse_map=sparse_map,
        )

        self.hloc_result = self._load_hloc_result()

    def _load_hloc_result(self):
        if self.test_on_sim:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname, self.level)
        else:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname)

        hloc_output_file = os.path.join(
            hloc_output_dir, "feats-superpoint-n4096-r1600_matches-NN-mutual-dist.7_pairs-netvlad.h5"
        )

        score_dict = {}
        match_h5 = h5py.File(hloc_output_file, "r")

        for _, query_dict in match_h5.items():
            query_id_string = query_dict.name[-10:-4]
            current_score_dict = {}

            for map_id in query_dict:
                map_id_string = query_dict[map_id].name[-10:-4]
                scores = query_dict[map_id]["matching_scores0"][:]
                current_score_dict.update({map_id_string: np.sum(scores)})

            score_dict.update({query_id_string: current_score_dict})

        return score_dict

    def localize_with_observation(self, query_id: str):
        """Get localization result of current map according to input observation embedding."""
        current_score_dict = self.hloc_result[query_id]
        max_id = max(current_score_dict, key=current_score_dict.get)
        map_node_with_max_value = int(max_id)
        high_similarity_set = [int(key) for key in current_score_dict]

        return map_node_with_max_value, high_similarity_set
