import numpy as np
import cv2 as cv
from env import path_xml

faceCascade = cv.CascadeClassifier(path_xml)

cap = cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

while True:
    ret,photo = cap.read()
    cv.imshow('video',photo)
    k = cv.waitKey(30) & 0xFF
    if k == ord('s'):
        break
cap.release()
cv.destroyAllWindows()