#头像采集器

import cv2 as cv
import os
from env import path_xml

cam = cv.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv.CascadeClassifier(path_xml)

face_id = input('\nEnter the user id press <RETURN> ..')
print('正在初始化，请直视摄像头...')

count = 0
while(True):
    ret,img = cam.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),1)
        count = count + 1
        cv.imwrite("datadir/User." + str(face_id) + "." + str(count) + ".jpg",gray[y:y+h,x:x+w])
        cv.imshow('qp', gray[y:y + h, x:x + w])
    cv.imshow('image',img)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
    elif count >= 120:
        break
print(" 结束头像采集..")
cam.release()
cv.destroyAllWindows()