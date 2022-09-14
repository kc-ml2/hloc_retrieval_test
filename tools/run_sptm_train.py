import json
import os

import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

from algorithms.constants import NetworkConstant, TrainingConstant

file_directory = "/data1/chlee/siamese_dataset/images/"
label_directory = "./output/label_all.json"
sorted_image_file = sorted(os.listdir(file_directory))
with open(label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
    label_data = json.load(label_json)


def preprocess_image(filename, label):
    """Preprocess & concatenate two images."""
    anchor_file = file_directory + filename + "_0.bmp"
    target_file = file_directory + filename + "_1.bmp"

    anchor_image_string = tf.io.read_file(anchor_file)
    target_image_string = tf.io.read_file(target_file)
    anchor_image = tf.image.decode_bmp(anchor_image_string, channels=3)
    target_image = tf.image.decode_bmp(target_image_string, channels=3)
    anchor_image = tf.image.convert_image_dtype(anchor_image, tf.float32)
    target_image = tf.image.convert_image_dtype(target_image, tf.float32)

    input_image = tf.concat((anchor_image, target_image), 2)

    return input_image, label


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

with tf.device("/device:GPU:1"):

    dataset = tf.data.Dataset.from_tensor_slices((image_name_list, y_list))
    dataset = dataset.map(preprocess_image)

    # for x, y in dataset.take(1):
    #     print(np.shape(x))
    #     print(y)
    #     input()

    val_dataset = dataset.take(10000)
    train_dataset = dataset.skip(10000)

    train_dataset = train_dataset.batch(128)
    val_dataset = val_dataset.batch(128)

    siamese = NetworkConstant.SIAMESE_NETWORK
    model = siamese((NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS))
    adam = keras.optimizers.Adam(
        learning_rate=TrainingConstant.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001
    )
    checkpointer = ModelCheckpoint(filepath="model.weights.best.hdf5", verbose=1, save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    model.summary()
    model.fit(x=train_dataset, epochs=10000, validation_data=val_dataset, callbacks=[checkpointer])
