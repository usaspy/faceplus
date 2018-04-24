import numpy as np
import cv2 as cv
from env import path_xml

faceCascade = cv.CascadeClassifier(path_xml)

cap = cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

while True:
    ret,photo = cap.read()
    gray = cv.cvtColor(photo,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )
    print(faces)
    for(x,y,w,h) in faces:
        cv.rectangle(photo,(x,y),(x+w,y+h),(255,255,0),1)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = photo[y:y+h,x:x+w]
    cv.imshow('video',photo)
    k = cv.waitKey(30) & 0xFF
    if k == ord('s'):
        break
cap.release()
cv.destroyAllWindows()
s
