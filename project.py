import numpy as np
import os
import pandas as pd
import time as t
import cv2
import argparse as ap

# parse the arguments by constructing the argument parse
argument_parser = ap.ArgumentParser()
argument_parser.add_argument("-i", "--image", required=True,
	help="path to input image")
argument_parser.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
argument_parser.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(argument_parser.parse_args())

# loading the COCO class labels on which our YOLO model was trained
labels_path = "coco.names"
LABELS = open(labels_path).read()

# for representation of each possible class label we initialize the list of colors
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# deriving the paths to the YOLO weights and model configuration
weights_path =  "yolov3.weights"
config_path =  "yolov3.cfg"

# loading our YOLO object detector trained on the COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# loading our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and performing a forward pass of the YOLO object detector
# this process gives us the bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=True)
net.setInput(blob)
start = t.time()
layerOutputs = net.forward(ln)
end = t.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.2f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and class ids
boxes = []
confidences = []
class_ids = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class id and confidence of the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out the weak predictions by ensuring the detected probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding box followed by the boxe's width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences and class ids
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			class_ids.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(indexes) > 0:
	# loop over the indexes we are keeping
	for i in indexes.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a rectangle bounding box and label on the image
		color = [int(c) for c in COLORS[class_ids[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.2f}".format(LABELS[class_ids[i]], confidences[i])
		cv2.putText(image, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 1)

# display the output image
cv2.imshow("Output Image", image)
cv2.waitKey(0)
