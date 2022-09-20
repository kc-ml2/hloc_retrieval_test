import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf


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


if __name__ == "__main__":
    # file_directory = "/data1/chlee/siamese_dataset/images/"
    file_directory = "./output/images/"
    label_directory = "./output/label_all.json"
    sorted_image_file = sorted(os.listdir(file_directory))
    with open(label_directory, "r") as label_json:  # pylint: disable=unspecified-encoding
        label_data = json.load(label_json)

    image_name_list = []
    for image_file in sorted_image_file:
        if image_file[0:-6] in image_name_list:
            pass
        else:
            image_name_list.append(image_file[0:-6])

    y_list = []
    y_print_list = []
    for image_name in image_name_list:
        label_value = label_data[image_name + "_similarity"]
        y_list.append(keras.utils.to_categorical(np.array(label_value), num_classes=2))
        y_print_list.append(label_value)

    with tf.device("/device:GPU:1"):

        dataset = tf.data.Dataset.from_tensor_slices((image_name_list, y_list))
        dataset = dataset.map(preprocess_image)

        val_dataset = dataset.take(10000)
        train_dataset = dataset.skip(10000)

        train_dataset = train_dataset.batch(128)
        val_dataset = val_dataset.batch(128)

        # Test
        model = keras.models.load_model("model.weights.best.hdf5")
        predictions = model.predict(val_dataset)
        predictions = tf.math.argmax(predictions, -1)
        pred_np = predictions.numpy()
        compare_chart = np.concatenate(
            (np.expand_dims(y_print_list[:10000], axis=-1), np.expand_dims(pred_np, axis=-1)), axis=1
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
