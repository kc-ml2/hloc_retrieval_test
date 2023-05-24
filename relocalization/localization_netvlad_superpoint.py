import os
import re

import h5py
import numpy as np

from relocalization.localization_base import LocalizationBase


class LocalizationNetVLADSuperpoint(LocalizationBase):
    """Class for global localization methods according to the given map and queries with NetVLAD+Superpoint."""

    def __init__(
        self,
        config,
        map_obs_dir,
        query_dir,
        binary_topdown_map=None,
        visualize=False,
    ):
        """Initialize localization instance with specific map & query image directories."""
        super().__init__(
            config,
            map_obs_dir,
            query_dir,
            binary_topdown_map=binary_topdown_map,
            visualize=visualize,
        )

        self.hloc_result = self._load_hloc_result()

    def _load_hloc_result(self):
        """Load SuperPoint matching result h5 file which is generated by hloc."""
        if self.test_on_sim:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname, self.level)
        else:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname)

        hloc_output_file = os.path.join(
            hloc_output_dir, "feats-superpoint-n4096-r1600_matches-NN-mutual-dist.7_pairs-netvlad.h5"
        )

        match_h5 = h5py.File(hloc_output_file, "r")

        # Store data by query ID in matching score dictionary variable
        score_dict = {}
        for _, query_dict in match_h5.items():
            query_id_string = re.split("[/ . -]", query_dict.name)[-2]

            current_score_dict = {}
            for map_id in query_dict:
                map_id_string = re.split("[/ . -]", query_dict[map_id].name)[-2]
                scores = query_dict[map_id]["matching_scores0"][:]
                current_score_dict.update({map_id_string: np.sum(scores)})

            score_dict.update({query_id_string: current_score_dict})

        return score_dict

    def localize_with_observation(self, query_id: str):
        """Get global localization result according to query id input."""
        current_score_dict = self.hloc_result[query_id]
        # Get map node ID with maximum matching score value
        max_id_string = max(current_score_dict, key=current_score_dict.get)

        # If node has multiple observations, extract node id from file name string
        if "_" in max_id_string:
            map_node_with_max_value = int(max_id_string.split("_")[0])
            high_similarity_set = [int(key.split("_")[0]) for key in current_score_dict]
        else:
            map_node_with_max_value = int(max_id_string)
            high_similarity_set = [int(key) for key in current_score_dict]

        return map_node_with_max_value, high_similarity_set
