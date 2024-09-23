import cv2
import os
import numpy as np
	

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

recognition_classes = ['diarypage', 'date', 'row', 'location']

confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.8 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(f"model/dairytable_model.onnx")

def list_images(path):
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]
	onlyfiles.sort()
	return onlyfiles

def DetectionProcess(original_image):
	[height, width, _] = original_image.shape
	length = max((height, width))
	image = np.zeros((length, length, 3), np.uint8)
	image[0:height, 0:width] = original_image
	scale = length / INPUT_WIDTH

	blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(INPUT_WIDTH, INPUT_WIDTH), swapRB=True)
	detection_model.setInput(blob)
	outputs = detection_model.forward()

	outputs = np.array([cv2.transpose(outputs[0])])
	rows = outputs.shape[1]

	boxes = []
	scores = []
	class_ids = []

	for i in range(rows):
		classes_scores = outputs[0][i][4:]
		(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
		if maxScore >= confThreshold:
			box = [
				outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
				outputs[0][i][2], outputs[0][i][3]]
			boxes.append(box)
			scores.append(maxScore)
			class_ids.append(maxClassIndex)

	result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

	detections = []
	for i in range(len(result_boxes)):
		index = result_boxes[i]
		box = boxes[index]
		detection = {
			'class_id': class_ids[index],
			'class_name': recognition_classes[class_ids[index]],
			'confidence': scores[index],
			'box': box,
			'scale': scale}
		detections.append(detection)
	return detections


def DetectDiaryTable(img):
	detections = DetectionProcess(img)
	detected_cards = []
	for detection in detections:
		class_id, class_name, confidence, box, scale = \
			detection['class_id'], detection['class_name'], detection['confidence'], detection['box'], detection[
				'scale']
		print(detection['class_name'])
		print(detection['confidence'])
		left, top, right, bottom = round(box[0] * scale), round(box[1] * scale), round(
			(box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)

		detected_cards.append([left, top, right, bottom])
		cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

		cv2.putText(img, detection['class_name'], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
	height, width, _ = img.shape
	
	cv2.imshow("Diarypage_Detection", cv2.resize(img, (int(width/2), int(height/2))))
	cv2.waitKey(0)

if __name__ == '__main__':
	ls_images = list_images(dir_path + "/images/")
	for input_image in ls_images:
		inputFilename = os.path.join(dir_path + "/images/", input_image)
		img = cv2.imread(inputFilename)
		DetectDiaryTable(img)





