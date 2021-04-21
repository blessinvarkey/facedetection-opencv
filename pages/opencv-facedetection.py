import cv2
import numpy as np
from random import randrange
#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Choose an image to detect face
img = cv2.imread('woman.jpg')
#Convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
print(face_coordinates)
#rectangle around the face
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x, y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),5)
#(x,y,w,h) = face_coordinates[1]
#cv2.rectangle(img,(x, y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),5)
cv2.imshow('Face', img)
cv2.waitKey(10000)

print('no error')
