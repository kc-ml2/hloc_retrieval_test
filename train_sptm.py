import json
import os
import random
import time

import keras
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from config.algorithm_config import NetworkConstant, TrainingConstant
from config.env_config import PathConfig
from network.resnet import ResnetBuilder
from utils.network_utils import list_image_name_label_wo_index, preprocess_paired_image_file

if __name__ == "__main__":
    random.seed(1)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    train_file_directory = PathConfig.TRAIN_IMAGE_PATH
    valid_file_directory = PathConfig.VALID_IMAGE_PATH

    train_label_directory = PathConfig.TRAIN_LABEL_PATH
    valid_label_directory = PathConfig.VALID_LABEL_PATH

    sorted_train_image_file = sorted(os.listdir(train_file_directory))
    sorted_valid_image_file = sorted(os.listdir(valid_file_directory))

    img_extension = sorted_train_image_file[0][-4:]

    with open(train_label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
        train_label_data = json.load(label_json)
    with open(valid_label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
        valid_label_data = json.load(label_json)

    train_image_list, train_y_list = list_image_name_label_wo_index(sorted_train_image_file, train_label_data)
    valid_image_list, valid_y_list = list_image_name_label_wo_index(sorted_valid_image_file, valid_label_data)

    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_y_list))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_list, valid_y_list))

        train_dataset = train_dataset.shuffle(len(train_image_list), reshuffle_each_iteration=True)

        train_dataset = train_dataset.map(
            lambda x, y: preprocess_paired_image_file(x, y, train_file_directory, img_extension)
        )
        valid_dataset = valid_dataset.map(
            lambda x, y: preprocess_paired_image_file(x, y, valid_file_directory, img_extension)
        )

        train_dataset = train_dataset.batch(TrainingConstant.BATCH_SIZE)
        valid_dataset = valid_dataset.batch(TrainingConstant.BATCH_SIZE)

        siamese = ResnetBuilder.build_siamese_resnet_18
        model = siamese((NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS))

        adam = keras.optimizers.Adam(
            learning_rate=TrainingConstant.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001
        )
        checkpointer = ModelCheckpoint(
            filepath=f"model_weights/model.{timestr}.weights.hdf5", verbose=1, save_best_only=True
        )
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
        model.summary()
        model.fit(x=train_dataset, epochs=10000, validation_data=valid_dataset, callbacks=[checkpointer])
