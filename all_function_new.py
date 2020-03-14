"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance
from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import dlib
import pyttsx3
import pyaudio
import wave
import time
import speech_recognition as sr
from gaze_tracking import GazeTracking
engine = pyttsx3.init()
rate = 0.9
volume = 1.0
engine.setProperty('rate',rate)
engine.setProperty('volume', volume)
#AUDIO_OUTPUT ='hello.wav'
# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] #68个特征点数据库中获取左眼
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
YAWN_THRESH = 10
thresh = 0.20
frame_check = 40
flag0=0
flag=0
detector = dlib.get_frontal_face_detector() #获取人脸
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

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

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def audio_record(out_file, rec_time):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 #16bit编码格式
    CHANNELS = 1 #单声道
    RATE = 16000 #16000采样频率
    
    p = pyaudio.PyAudio()
    # 创建音频流 
    stream = p.open(format=FORMAT, # 音频流wav格式
                    channels=CHANNELS, # 单声道
                    rate=RATE, # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Start Recording...")

    frames = [] # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # 录制完成
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording Done...")

    # 保存音频文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def sphinx_recog(out_file):
	
	r = sr.Recognizer()

	harvard = sr.AudioFile(out_file)

	with harvard as source:

		audio = r.record(source)

	# recognize speech using Sphinx

	try:
		q = r.recognize_sphinx(audio);
		print("Sphinx thinks you said " + q)
		if isSubstring2('increase',q):
			if isSubstring2('volume',q):
				volume+=0.1
				engine.setProperty('volume',volume)
			if isSubstring2('rate',q):
				rate+=0.1
				engine.setProperty('rate',rate)
		if isSubstring2('decrease',q):
			if isSubstring2('volume',q):
				volume-=0.1
				engine.setProperty('volume',volume)
			if isSubstring2('rate',q):
				rate-=0.1
				engine.setProperty('rate',rate)

	except sr.UnknownValueError:

		print("Sphinx could not understand audio")

	except sr.RequestError as e:

		print("Sphinx error; {0}".format(e))

def isSubstring2(s1,s2):
    tag = False
    if s2.find(s1) != -1:
        tag = True
    return tag

def main():
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        engine.say("Warning: video source not assigned, default webcam will be used")
        engine.runAndWait()
        video_src = 0
    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()
    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()
    gaze = GazeTracking()
    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter() 
    head_flag = 0
    gaze_flag = 0
    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break
        
        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]
        #audio_record(AUDIO_OUTPUT, 3) 
        #sphinx_recog(AUDIO_OUTPUT)
        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)
	
        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        text = ""
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            rects = detector(gray, 0)
            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()
            #
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

                    if (lipdistance > YAWN_THRESH):
                        #print(lipdistance)
                        flag0 += 1
                        print ("yawning time: ", flag0)
                        if flag0 >= 40:
                            cv2.putText(frame, "Yawn Alert", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "Yawn Alert", (220, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            engine.say("don't yawn")
                            engine.runAndWait()
                            flag0 = 0
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
                            engine.say("open your eyes")
                            engine.runAndWait()
                            flag = 0
                    else:
                        flag = 0
                    if gaze.is_right():
                        print("Looking right")
                        text = "Looking right"
                    elif gaze.is_left():
                        print("Looking left")
                        text = "Looking left"
                    elif gaze.is_up():
                        text = "Looking up"
                    else:
                        text ="Looking center"
                    if text is not "Looking center":
                        gaze_flag+=1
                        if gaze_flag >= 20:
                            engine.say("look forward")
                            engine.runAndWait()
                            gaze_flag = 0
                    else:
                        gaze_flag =0;
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)
            # get angles
            angles = pose_estimator.get_angles(pose[0],pose[1])
            if ((-8 > angles[0] or angles[0]> 8)or(-8 > angles[1] or angles[1]> 8)):
            	head_flag += 1
            	if head_flag >= 40:
                    print(angles[0])
                    engine.say("please look ahead")
                    engine.runAndWait()
            else:
            	head_flag = 0
            # pose_estimator.draw_info(frame, angles)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            pose_estimator.draw_annotation_box(
                frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # Uncomment following line to draw head axes on frame.
            pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])
            #pose_estimator.show_3d_model

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()


if __name__ == '__main__':
    main()
