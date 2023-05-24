import os
import re

from relocalization.localization_base import LocalizationBase


class LocalizationNetVLADOnly(LocalizationBase):
    """Class for localization methods according to the given map."""

    def __init__(
        self,
        config,
        map_obs_dir,
        query_dir,
        binary_topdown_map=None,
        visualize=False,
    ):
        """Initialize localization instance with specific model & map data."""
        super().__init__(
            config,
            map_obs_dir,
            query_dir,
            binary_topdown_map=binary_topdown_map,
            visualize=visualize,
        )

        self.query_prefix = os.path.join(self.scene_dirname, self.query_dirname)
        self.hloc_result = self._load_hloc_result()

    def _load_hloc_result(self):
        if self.test_on_sim:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname, self.level)
        else:
            hloc_output_dir = os.path.join(self.config.PathConfig.HLOC_OUTPUT, self.scene_dirname)

        hloc_output_file = os.path.join(hloc_output_dir, "pairs-netvlad.txt")

        with open(hloc_output_file) as f:  # pylint: disable=unspecified-encoding
            retrieval_pairs = f.readlines()

        return retrieval_pairs

    def localize_with_observation(self, query_id: str):
        """Get localization result of current map according to input observation embedding."""
        high_simil_pair_list = []
        query = f"{self.query_prefix}/{query_id}"
        for pair in self.hloc_result:
            if query in pair:
                high_simil_pair_list.append(pair)

        max_pair = high_simil_pair_list[0]
        map_node_with_max_value = int(re.search("/(.+?).jpg", max_pair[-14:]).group(1))
        high_similarity_set = []
        for high_pair in high_simil_pair_list:
            high_similarity_set.append(int(re.search("/(.+?).jpg", high_pair[-14:]).group(1)))

        return map_node_with_max_value, high_similarity_set
