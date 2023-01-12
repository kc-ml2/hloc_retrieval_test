import os

import cv2
import numpy as np


class Yolo:
    def __init__(
        self,
        yolo_cfg_path="./config/yolo",
        yolo_weight_path="./model_weights/yolov3.weights",
        confidence=0.5,
        threshold=0.3,
    ):
        self.confidence = confidence
        self.threshold = threshold

        # Load the COCO class labels our YOLO model was trained on
        labels_path = os.path.sep.join([yolo_cfg_path, "coco.names"])
        with open(labels_path) as label_file:  # pylint: disable=unspecified-encoding
            self.labels = label_file.read().strip().split("\n")

        # Initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        # Derive the paths to the model configuration
        config_path = os.path.sep.join([yolo_cfg_path, "yolov3.cfg"])

        # Load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, yolo_weight_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    # def detect_object(self, image):
    #     if isinstance(image, str):
    #         image = cv2.imread(image)

    #     (H, W) = image.shape[:2]

    #     # Construct a blob from the input image and then perform a forward pass of the YOLO object detector,
    #     # giving us our bounding boxes and associated probabilities
    #     blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    def detect_object(self, image_batch):
        (H, W) = image_batch[0].shape[:2]
        image_list = np.array([image_batch[0], image_batch[1]])

        # Construct a blob from the input image and then perform a forward pass of the YOLO object detector,
        # giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(image_list, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        print(blob.shape)
        input()
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        # Initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        for output in layer_outputs:
            for detection in output:
                # Extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filter out weak predictions by ensuring the detected probability is
                # greater than the minimum probability
                if confidence > self.confidence:
                    # Scale the bounding box coordinates back relative to the size of the image,
                    # keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box
                    # followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([int(centerX), int(centerY), int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        suppressed_boxes = []
        suppressed_confidences = []
        suppressed_classIDs = []

        for idx in idxs:
            suppressed_boxes.append(boxes[idx])
            suppressed_confidences.append(confidences[idx])
            suppressed_classIDs.append(classIDs[idx])

        return suppressed_boxes, suppressed_confidences, suppressed_classIDs

    def display_detection_on_img(self, detection_result, image):
        boxes, confidences, classIDs = detection_result

        # Ensure at least one detection exists
        if len(boxes) > 0:
            # Loop over the indexes we are keeping
            for i, box in enumerate(boxes):
                width = box[2]
                height = box[3]
                corner_x = int(box[0] - width / 2)
                corner_y = int(box[1] - height / 2)

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (corner_x, corner_y), (corner_x + width, corner_y + height), color, 2)
                text = f"{self.labels[classIDs[i]]}: {confidences[i]:.4f}"
                cv2.putText(image, text, (corner_x, corner_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

    def detect_and_display(self, image_batch):
        detection_result_batch = self.detect_object(image_batch)
        image_batch = self.display_detection_on_img(detection_result_batch, image_batch)

        return image_batch, detection_result_batch

    # def detect_and_display(self, image):
    #     detection_result = self.detect_object(image)
    #     image = self.display_detection_on_img(detection_result, image)

    #     return image, detection_result