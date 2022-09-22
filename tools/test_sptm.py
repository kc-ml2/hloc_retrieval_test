import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf

from algorithms.constants import TrainingConstant
from algorithms.sptm_utils import list_image_name_label_wo_index, preprocess_image

if __name__ == "__main__":
    file_directory = "/data1/chlee/siamese_dataset/images/"
    # file_directory = "./output/images/"
    label_directory = "./output/label_all.json"
    sorted_image_file = sorted(os.listdir(file_directory))
    with open(label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
        label_data = json.load(label_json)

    image_name_list, y_list = list_image_name_label_wo_index(file_directory, sorted_image_file, label_data)

    y_list_for_compare = []
    for image_name in image_name_list:
        label_value = label_data[image_name + "_similarity"]
        y_list_for_compare.append(label_value)

    with tf.device("/device:GPU:1"):
        dataset = tf.data.Dataset.from_tensor_slices((image_name_list, y_list))
        dataset = dataset.map(lambda x, y: preprocess_image(x, y, file_directory))

        val_dataset = dataset.take(10000)
        train_dataset = dataset.skip(10000)

        val_dataset = val_dataset.batch(TrainingConstant.BATCH_SIZE)
        train_dataset = train_dataset.batch(TrainingConstant.BATCH_SIZE)

        # Test
        model = keras.models.load_model("./model_weights/model0916.weights.best.hdf5")
        predictions = model.predict(val_dataset)
        predictions = tf.math.argmax(predictions, -1)
        pred_np = predictions.numpy()
        compare_chart = np.concatenate(
            (np.expand_dims(y_list_for_compare[:10000], axis=-1), np.expand_dims(pred_np, axis=-1)), axis=1
        )

        num_correct = 0
        for i, image_name in enumerate(image_name_list[:10000]):
            anchor_img = cv2.imread(file_directory + image_name + "_0.bmp")
            target_img = cv2.imread(file_directory + image_name + "_1.bmp")

            separation_block = np.zeros((np.shape(anchor_img)[0], 100, 3), dtype=np.uint8)
            concatenated_img = np.concatenate((anchor_img, separation_block), axis=1)
            concatenated_img = np.concatenate((concatenated_img, target_img), axis=1)

            if compare_chart[i][0] == 0:
                distance = "Far"
            if compare_chart[i][0] == 1:
                distance = "Near"

            if compare_chart[i][0] == compare_chart[i][1]:
                prediction = "Correct"
                num_correct = num_correct + 1
            if compare_chart[i][0] != compare_chart[i][1]:
                prediction = "Wrong"

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

            cv2.imwrite(f"./output/test/{image_name}_{distance}_{prediction}.bmp", concatenated_img)
            print(i)
        print("Number of correct answer: ", num_correct)
