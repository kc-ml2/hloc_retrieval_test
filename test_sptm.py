import argparse
import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf

from algorithms.sptm_utils import list_image_name_label_wo_index, preprocess_image
from config.algorithm_config import TestConstant
from config.env_config import PathConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", required=True)
    parser.add_argument("--generate-img", action="store_true")
    args, _ = parser.parse_known_args()
    loaded_model = args.load_model
    generate_img = args.generate_img

    correct_directory = "./output/sptm_result/correct/"
    wrong_directory = "./output/sptm_result/wrong/"

    if generate_img:
        os.makedirs(correct_directory, exist_ok=True)
        os.makedirs(wrong_directory, exist_ok=True)

    file_directory = PathConfig.TEST_IMAGE_PATH
    label_directory = PathConfig.TEST_LABEL_PATH
    sorted_image_file = sorted(os.listdir(file_directory))
    with open(label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
        label_data = json.load(label_json)

    img_extension = sorted_image_file[0][-4:]
    image_name_list, y_list = list_image_name_label_wo_index(sorted_image_file, label_data)

    y_list_for_compare = []
    for image_name in image_name_list:
        label_value = label_data[image_name + "_similarity"]
        y_list_for_compare.append(label_value)

    with tf.device("/device:GPU:1"):
        dataset = tf.data.Dataset.from_tensor_slices((image_name_list, y_list))
        dataset = dataset.map(lambda x, y: preprocess_image(x, y, file_directory, img_extension))
        dataset = dataset.batch(TestConstant.BATCH_SIZE)

        # Test
        model = keras.models.load_model(loaded_model)
        predictions = model.predict(dataset)
        predictions = tf.math.argmax(predictions, -1)
        pred_np = predictions.numpy()
        compare_chart = np.concatenate(
            (np.expand_dims(y_list_for_compare, axis=-1), np.expand_dims(pred_np, axis=-1)), axis=1
        )

        num_correct = 0
        for i, image_name in enumerate(image_name_list):
            if compare_chart[i][0] == 0:
                distance = "Far"
            if compare_chart[i][0] == 1:
                distance = "Near"

            if compare_chart[i][0] == compare_chart[i][1]:
                prediction = "Correct"
                output_path = correct_directory
                num_correct = num_correct + 1
            if compare_chart[i][0] != compare_chart[i][1]:
                prediction = "Wrong"
                output_path = wrong_directory

            if generate_img:
                anchor_img = cv2.imread(file_directory + os.sep + image_name + f"_0{img_extension}")
                target_img = cv2.imread(file_directory + os.sep + image_name + f"_1{img_extension}")
                separation_block = np.zeros((np.shape(anchor_img)[0], 100, 3), dtype=np.uint8)
                concatenated_img = np.concatenate((anchor_img, separation_block), axis=1)
                concatenated_img = np.concatenate((concatenated_img, target_img), axis=1)

                put_str = f"{distance}_{prediction}"

                cv2.putText(
                    img=concatenated_img,
                    text=put_str,
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                cv2.imwrite(output_path + os.sep + f"{image_name}_{distance}_{prediction}.jpg", concatenated_img)

            print(i)
        print("Number of correct answer: ", num_correct)
