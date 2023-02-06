import math
import os
import random

import cv2
import keras
import numpy as np
import tensorflow as tf


def get_distance(first_point, second_point):
    return math.sqrt((first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2)


def color2gray(input):
    return cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)


def downsample(input, factor):
    for _ in range(factor):
        input = cv2.pyrDown(input)
    return input


def preprocess_paired_image_file(image_name, label, file_directory, extension):
    """Preprocess & concatenate two images."""
    anchor_file = file_directory + os.sep + image_name + f"_0{extension}"
    target_file = file_directory + os.sep + image_name + f"_1{extension}"

    anchor_image_string = tf.io.read_file(anchor_file)
    target_image_string = tf.io.read_file(target_file)

    if extension == ".bmp":
        anchor_image = tf.image.decode_bmp(anchor_image_string, channels=3)
        target_image = tf.image.decode_bmp(target_image_string, channels=3)
    if extension == ".jpg":
        anchor_image = tf.image.decode_jpeg(anchor_image_string, channels=3)
        target_image = tf.image.decode_jpeg(target_image_string, channels=3)

    anchor_image = tf.image.convert_image_dtype(anchor_image, tf.float32)
    target_image = tf.image.convert_image_dtype(target_image, tf.float32)

    input_image = tf.concat((anchor_image, target_image), 2)

    return input_image, label


def preprocess_single_view_paired_image_file(image_name, label, file_directory, extension):
    """Preprocess & concatenate two images."""
    anchor_file = file_directory + os.sep + image_name + f"_0{extension}"
    target_file = file_directory + os.sep + image_name + f"_1{extension}"

    anchor_image_string = tf.io.read_file(anchor_file)
    target_image_string = tf.io.read_file(target_file)

    if extension == ".bmp":
        anchor_image = tf.image.decode_bmp(anchor_image_string, channels=3)
        target_image = tf.image.decode_bmp(target_image_string, channels=3)
    if extension == ".jpg":
        anchor_image = tf.image.decode_jpeg(anchor_image_string, channels=3)
        target_image = tf.image.decode_jpeg(target_image_string, channels=3)

    slice_start = random.randint(0, 767)
    sliced_target_image = target_image[:, slice_start : slice_start + 256, :]

    front = tf.constant(0, shape=(256, slice_start, 3), dtype=tf.uint8)
    end = tf.constant(0, shape=(256, 1024 - slice_start - 256, 3), dtype=tf.uint8)
    modified_target_image = tf.concat((front, sliced_target_image, end), 1)

    anchor_image = tf.image.convert_image_dtype(anchor_image, tf.float32)
    modified_target_image = tf.image.convert_image_dtype(modified_target_image, tf.float32)

    input_image = tf.concat((anchor_image, modified_target_image), 2)

    return input_image, label


def preprocess_single_image_file(obs_file):
    """Preprocess one images from file name."""
    image_string = tf.io.read_file(obs_file)
    image = tf.image.decode_jpeg(image_string, channels=3)
    input_image = tf.image.convert_image_dtype(image, tf.float32)

    return input_image


def list_image_name_label_wo_index(sorted_image_file, label_data):
    image_name_list = []
    for image_file in sorted_image_file:
        if image_file[0:-6] in image_name_list:
            pass
        else:
            image_name_list.append(image_file[0:-6])

    y_list = []
    for image_name in image_name_list:
        label_value = label_data[image_name + "_similarity"]
        y_list.append(keras.utils.to_categorical(np.array(label_value), num_classes=2))

    return image_name_list, y_list
