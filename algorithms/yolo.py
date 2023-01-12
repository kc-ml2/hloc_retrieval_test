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

    def detect_object(self, image_batch):
        batch_size = image_batch.shape[0]
        (H, W) = image_batch[0].shape[:2]

        # Construct a blob from the input image and then perform a forward pass of the YOLO object detector,
        # giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImages(image_batch, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        # Initialize lists of detected bounding boxes, confidences, and class IDs, respectively
        idxs_list = []
        detection_results = []
        boxes = []
        confidences = []
        classIDs = []
        suppressed_boxes = []
        suppressed_confidences = []
        suppressed_classIDs = []

        for i in range(batch_size):
            boxes.append([])
            confidences.append([])
            classIDs.append([])
            suppressed_boxes.append([])
            suppressed_confidences.append([])
            suppressed_classIDs.append([])

            for output in layer_outputs:
                detection_list = output[i]
                for detection in detection_list:
                    # Extract the class ID and confidence (i.e., probability) of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # Filter out weak predictions by probability
                    if confidence > self.confidence:
                        # Scale the bounding box coordinates back relative to the size of the image,
                        # keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box
                        # followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Update our list of bounding box coordinates, confidences, and class IDs
                        boxes[i].append([int(centerX), int(centerY), int(width), int(height)])
                        confidences[i].append(float(confidence))
                        classIDs[i].append(classID)

            # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes[i], confidences[i], self.confidence, self.threshold)
            idxs_list.append(idxs)

        for i in range(batch_size):
            idxs = idxs_list[i]
            for idx in idxs:
                suppressed_boxes[i].append(boxes[i][idx])
                suppressed_confidences[i].append(confidences[i][idx])
                suppressed_classIDs[i].append(classIDs[i][idx])

            suppressed_detection = (suppressed_boxes[i], suppressed_confidences[i], suppressed_classIDs[i])
            detection_results.append(suppressed_detection)

        return detection_results

    def display_detection_on_img(self, image_batch, detection_results):
        batch_size = image_batch.shape[0]

        for batch_idx in range(batch_size):
            boxes, confidences, classIDs = detection_results[batch_idx]
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
                    cv2.rectangle(
                        image_batch[batch_idx], (corner_x, corner_y), (corner_x + width, corner_y + height), color, 2
                    )
                    text = f"{self.labels[classIDs[i]]}: {confidences[i]:.4f}"
                    cv2.putText(
                        image_batch[batch_idx], text, (corner_x, corner_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        return image_batch

    def detect_and_display(self, image_batch):
        detection_results = self.detect_object(image_batch)
        image_batch = self.display_detection_on_img(image_batch, detection_results)

        return image_batch, detection_results
