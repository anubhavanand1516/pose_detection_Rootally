import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture("KneeBend.mp4")
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (600, 400))
    results = pose.process(img)
  
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    cv2.imshow("Pose Detection", img)
    h, w, c = img.shape   
    opImg = np.zeros([h, w, c])  
    opImg.fill(255)  
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    cv2.imshow("Pose Label", opImg)
    print(results.pose_landmarks)
    cv2.waitKey(1)
