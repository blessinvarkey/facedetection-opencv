import cv2
import numpy as np
from random import randrange
#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    #rectangle around the face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x, y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),5)

    cv2.imshow('Video', frame)
    cv2.waitKey(1)

print('no error')
