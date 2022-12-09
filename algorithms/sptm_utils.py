import math
import os

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


def preprocess_image(image_name, label, file_directory, extension):
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


def preprocess_image_wo_label(obs_id_pair, anchor_file_dir, target_file_dir, extension):
    """Preprocess & concatenate two images from observations."""
    anchor_file = anchor_file_dir + os.sep + obs_id_pair[0] + extension
    target_file = target_file_dir + os.sep + obs_id_pair[1] + extension

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
