import cv2
import numpy as np


#Read an image
img = cv2.imread("Road.jpg")
#cv2.putText(img, "Car", (300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
print(img.shape)

cv2.imshow("Road", img)

cv2.waitKey(10000)
