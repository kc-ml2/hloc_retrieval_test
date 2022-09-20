import os.path

import networkx as nx
import numpy as np
from numpy import mean, median

from algorithms import resnet
from algorithms.constants import EnvConstant, NetworkConstant, PathConstant
from algorithms.sptm_utils import get_distance


def load_keras_model(number_of_input_frames, number_of_actions, path, load_method=resnet.ResnetBuilder.build_resnet_18):
    result = load_method(
        (number_of_input_frames * NetworkConstant.NET_CHANNELS, NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH),
        number_of_actions,
    )
    result.load_weights(path)
    result.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return result


def top_number_to_threshold(n, top_number, values):
    top_number = min([top_number, n])
    threshold = np.percentile(values, (n - top_number) * 100 / float(n))
    return threshold


def sieve(shortcuts, top_number):
    if top_number == 0:
        return []
    probabilities = shortcuts[:, 0]
    n = shortcuts.shape[0]
    threshold = top_number_to_threshold(n, top_number, probabilities)
    print("Confidence threshold for top", top_number, "out of", n, ":", threshold)
    sieved_shortcut_indexes = []
    for index in range(n):
        if probabilities[index] >= threshold:
            sieved_shortcut_indexes.append(index)
    return shortcuts[sieved_shortcut_indexes]


class InputProcessor:
    def __init__(self):
        self.edge_model = load_keras_model(2, 2, PathConstant.EDGE_MODEL_WEIGHTS, NetworkConstant.SIAMESE_NETWORK)
        self.bottom_network = resnet.ResnetBuilder.build_bottom_network(
            self.edge_model, (NetworkConstant.NET_CHANNELS, NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH)
        )
        self.top_network = resnet.ResnetBuilder.build_top_network(self.edge_model)
        self.tensor_to_predict = np.array([])

    def set_memory_buffer(self, keyframes):
        memory_codes = self.bottom_network.predict(np.array(keyframes))
        list_to_predict = []
        for index in range(len(keyframes)):
            x = np.concatenate((memory_codes[0], memory_codes[index]), axis=0)
            list_to_predict.append(x)
        self.tensor_to_predict = np.array(list_to_predict)

    def append_to_memory_buffer(self, keyframe):
        expanded_keyframe = np.expand_dims(keyframe, axis=0)
        memory_code = self.bottom_network.predict(expanded_keyframe)
        x = np.concatenate((memory_code, memory_code), axis=1)
        self.tensor_to_predict = np.concatenate((self.tensor_to_predict, x), axis=0)

    def predict_single_input(self, input):
        input_code = np.squeeze(self.bottom_network.predict(np.expand_dims(input, axis=0), batch_size=1))
        for index in range(self.tensor_to_predict.shape[0]):
            self.tensor_to_predict[index][0 : (input_code.shape[0])] = input_code
        probabilities = self.top_network.predict(self.tensor_to_predict, batch_size=NetworkConstant.TESTING_BATCH_SIZE)
        return probabilities[:, 1]

    def get_memory_size(self):
        return self.tensor_to_predict.shape[0]


class SPTM:
    def __init__(self):
        self.input_processor = InputProcessor()
        self.graph = nx.Graph()
        self.shortest_paths = []
        self.shortest_distances = []
        self.shortcuts = np.array([])
        self.shortcuts_cache_file = PathConstant.SHORTCUTS_CACHE_FILE

    def set_memory_buffer(self, keyframes):
        self.input_processor.set_memory_buffer(keyframes)

    def append_to_memory_buffer(self, keyframe):
        self.input_processor.append_to_memory_buffer(keyframe)

    def predict_single_input(self, input):
        return self.input_processor.predict_single_input(input)

    def get_memory_size(self):
        return self.input_processor.get_memory_size()

    def add_double_sided_edge(self, first, second):
        self.graph.add_edge(first, second)
        self.graph.add_edge(second, first)

    def add_double_forward_biased_edge(self, first, second):
        self.graph.add_edge(first, second)
        self.graph.add_edge(second, first, {"weight": 1000000000})

    @staticmethod
    def smooth_shortcuts_matrix(shortcuts_matrix, keyframe_coordinates):
        for first, _ in enumerate(shortcuts_matrix):
            for second in range(first + 1, len(shortcuts_matrix)):
                shortcuts_matrix[first][second] = (
                    shortcuts_matrix[first][second] + shortcuts_matrix[second][first]
                ) / 2.0
        shortcuts = []
        for first in range(len(shortcuts_matrix)):
            for second in range(first + 1 + EnvConstant.MIN_SHORTCUT_DISTANCE, len(shortcuts_matrix)):
                values = []
                for shift in range(-EnvConstant.SHORTCUT_WINDOW, EnvConstant.SHORTCUT_WINDOW + 1):
                    first_shifted = first + shift
                    second_shifted = second + shift
                    if 0 <= first_shifted < len(shortcuts_matrix) and 0 <= second_shifted < len(shortcuts_matrix):
                        values.append(shortcuts_matrix[first_shifted][second_shifted])
                quality = median(values)
                distance = get_distance(keyframe_coordinates[first], keyframe_coordinates[second])
                shortcuts.append((quality, first, second, distance))
        return np.array(shortcuts)

    def compute_shortcuts(self, keyframes, keyframe_coordinates):
        self.set_memory_buffer(keyframes)
        if not os.path.isfile(self.shortcuts_cache_file):
            shortcuts_matrix = []
            for first, _ in enumerate(keyframes):
                probabilities = self.predict_single_input(keyframes[first])
                shortcuts_matrix.append(probabilities)
                print("Finished:", float(first * 100) / float(len(keyframes)), "%")
            shortcuts = self.smooth_shortcuts_matrix(shortcuts_matrix, keyframe_coordinates)
            shortcuts = sieve(shortcuts, NetworkConstant.LARGE_SHORTCUTS_NUMBER)
            np.save(self.shortcuts_cache_file, shortcuts)
        else:
            shortcuts = np.load(self.shortcuts_cache_file)
        self.shortcuts = sieve(shortcuts, NetworkConstant.SMALL_SHORTCUTS_NUMBER)

    def get_number_of_shortcuts(self):
        return len(self.shortcuts)

    def get_shortcut(self, index):
        return (int(self.shortcuts[index, 1]), int(self.shortcuts[index, 2]))

    def get_shortcuts(self):
        return self.shortcuts

    def build_graph(self, keyframes, keyframe_coordinates):
        self.set_memory_buffer(keyframes)
        memory_size = self.get_memory_size()
        self.graph.add_nodes_from(range(memory_size))
        for first in range(memory_size - 1):
            # self.add_double_forward_biased_edge(first, first + 1)
            self.add_double_sided_edge(first, first + 1)
        self.compute_shortcuts(keyframes, keyframe_coordinates)
        for index in range(self.get_number_of_shortcuts()):
            edge = self.get_shortcut(index)
            first, second = edge
            assert abs(first - second) > EnvConstant.MIN_SHORTCUT_DISTANCE
            self.add_double_sided_edge(*edge)

    def find_nn(self, input):
        probabilities = self.predict_single_input(input)
        best_index = np.argmax(probabilities)
        best_probability = np.max(probabilities)
        return best_index, best_probability, probabilities

    def compute_shortest_paths(self, graph_goal):
        self.shortest_paths = nx.shortest_path(self.graph, target=graph_goal, weight="weight")
        self.shortest_distances = [len(value) - 1 for value in self.shortest_paths.values()]
        print("Mean shortest_distances to goal:", mean(self.shortest_distances))
        print("Median shortest_distances to goal:", median(self.shortest_distances))

    def get_shortest_paths_and_distances(self):
        return self.shortest_paths, self.shortest_distances

    @staticmethod
    def _find_neighbours_by_threshold(threshold, probabilities):
        nns = []
        for index, probability in enumerate(probabilities):
            if probability >= threshold:
                nns.append(index)
        return nns

    def find_neighbours_by_threshold(self, input, threshold):
        probabilities = self.predict_single_input(input)
        return self._find_neighbours_by_threshold(threshold, probabilities)

    def find_knn(self, input, k):
        probabilities = self.predict_single_input(input)
        threshold = top_number_to_threshold(self.get_memory_size(), k, probabilities)
        return self._find_neighbours_by_threshold(threshold, probabilities)

    def find_knn_median_threshold(self, input, k, threshold):
        probabilities = self.predict_single_input(input)
        knn_threshold = top_number_to_threshold(self.get_memory_size(), k, probabilities)
        final_threshold = max([threshold, knn_threshold])
        nns = self._find_neighbours_by_threshold(final_threshold, probabilities)
        nns.sort()
        if nns:
            # nn = nns[len(nns) / 2]
            nn = nns[int(len(nns) / 2)]
            return nn, probabilities, nns
        else:
            return None, probabilities, nns

    def find_nn_threshold(self, input, threshold):
        nn, probability, probabilities = self.find_nn(input)
        if probability < threshold:
            return None, None
        else:
            return nn, probabilities

    def find_smoothed_nn(self, input):
        nn = None
        if EnvConstant.SMOOTHED_LOCALIZATION:
            nn, probabilities = self.find_nn_on_last_shortest_path(input)
        if nn is None:
            nn, probabilities, _ = self.find_knn_median_threshold(
                input, NetworkConstant.NUMBER_OF_NEAREST_NEIGHBOURS, EnvConstant.INTERMEDIATE_REACHABLE_GOAL_THRESHOLD
            )
        return nn, probabilities