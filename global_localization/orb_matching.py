import os
import time

import cv2
import numpy as np

from global_localization.base import LocalizationBase


class LocalizationOrbMatching(LocalizationBase):
    """Class for global localization methods with ORB local feature.
    This class does not use DBOW(Bag of the Words) method. It uses brute force matching of ORB features.
    As we tested, DBOW is good for reducing computing resource, but not good for localization accuaracy.
    """

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

        self.desc_db, self.desc_query = self._generate_ORB_feature_database()

    def _generate_ORB_feature_database(self):
        """Initialize ORB instance & get ORB local features from images."""
        # Make image list to iterate
        map_obs_file_list = [self.map_obs_dir + os.sep + file for file in self.sorted_map_obs_file]
        query_list = [self.query_dir + os.sep + file for file in self.sorted_query_file]

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
        num_nonetype = 0

        # Extract local descriptors(ORB) from all map images, and store them as database
        desc_map = []
        for i in range(0, len(map_obs_file_list), self.num_frames_per_node):
            # Concatenate images if multiple images are assinged to a single node
            frame_list = []
            for k in range(self.num_frames_per_node):
                frame_list.append(cv2.imread(map_obs_file_list[i + k]))
            map_image = np.concatenate(frame_list, axis=1)

            _, map_des = self.orb.detectAndCompute(map_image, None)  # Extract local descriptor
            if map_des is None:  # Error exception code for Nonetype result
                map_des = np.zeros([1, 32], dtype=np.uint8)
                num_nonetype = num_nonetype + 1

            desc_map.append(map_des)  # Add current local descriptor to database(list)

        end = time.time()
        print("Number of NoneType in DB: ", num_nonetype)
        print("DB generation elapsed time: ", end - start)

        start = time.time()
        num_nonetype = 0

        # Extract local descriptors(ORB) from all query images, and store them as database
        desc_query = []
        for query_obs_file in query_list:
            query_image = cv2.imread(query_obs_file)
            _, query_des = self.orb.detectAndCompute(query_image, None)  # Extract local descriptor

            if query_des is None:  # Error exception code for Nonetype result
                query_des = np.zeros([1, 32], dtype=np.uint8)
                num_nonetype = num_nonetype + 1

            desc_query.append(query_des)  # Add current local descriptor to database(list)

        end = time.time()
        print("Number of NoneType in Query: ", num_nonetype)
        print("Query generation elapsed time: ", end - start)

        return desc_map, desc_query

    def localize_with_observation(self, query_id: str):
        """Get global localization result according to query id input."""
        # Query the database
        i = int(query_id)
        print(i, end="\r", flush=True)

        scores = []
        for db_des in self.desc_db:
            matches = self.bf.match(self.desc_query[i], db_des)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:30]  # Use top 30 matches when calculating the score

            scores.append(sum([match.distance for match in matches]))

        # Get the highest similarity node ID
        map_node_with_max_value = np.argmin(scores)

        # Get top 20 high similarity set
        high_similarity_set = sorted(range(len(scores)), key=lambda i: scores[i])[:20]

        return map_node_with_max_value, high_similarity_set
