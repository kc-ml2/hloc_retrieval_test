import math

import cv2


def get_distance(first_point, second_point):
    return math.sqrt((first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2)


def color2gray(input):
    return cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)


def downsample(input, factor):
    for _ in range(factor):
        input = cv2.pyrDown(input)
    return input
