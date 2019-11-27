# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from scipy.spatial import distance
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[61:64]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-r", "--picamera", type=int, default=-1,
# 	help="whether or not the Raspberry Pi camera should be used")
# args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() #获取人脸
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
# time.sleep(2.0)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] #68个特征点数据库中获取左眼
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
YAWN_THRESH = 10
thresh = 0.20
frame_check = 40
flag0=0
flag=0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)  #converting to NumPy Array矩阵运算
		mouth = shape[Start:End]
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye) #眼睛长宽比
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0#长宽比平均值
		lipdistance = lip_distance(shape)
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) #将眼眶可视化
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		if (lipdistance > YAWN_THRESH):
			print(lipdistance)
			flag0 += 1
			print ("yawning time: ", flag0)
			if flag0 >= 40:
				cv2.putText(frame, "Yawn Alert", (10, 150),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "Yawn Alert", (220, 150),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Don't yawn")
		else:
			flag0 = 0

		if (ear < thresh):
			flag += 1
			print ("eyes closing time: ",flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,250),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Open your eyes!")
		else:
			flag = 0
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
