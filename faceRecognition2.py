
import cv2
import numpy as np
import time
import os
from env import path_xml
from multiprocessing import Pool

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

names = {1: 'zhang hong', 2: 'zhang dong yang', 3: 'luo ran', 4: 'outman'}

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

cam.set(cv2.CAP_PROP_FPS, 30)
faceCascade = cv2.CascadeClassifier(path_xml)
def check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    print(faces)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(id, "<>", confidence)
        if confidence < 100:
            name = names[id]
            confidence = "Match:  {0}%".format(round(confidence))
        else:
            name = "unknown"
            confidence = "Match: {0}%".format(round(100 - confidence))
        cv2.putText(img, str(name), (x, y - 5), font, 1, (255, 255, 0), 2)
        cv2.putText(img, str(confidence), (x, y - 30), font, 1, (255, 255, 255), 1)

        fps = cam.get(cv2.CAP_PROP_FPS)
        cv2.putText(img, 'fps:' + str(int(fps)), (x, y - 55), font, 1, (255, 255, 255), 1)
    return img





if __name__ == '__main__':
    p = Pool(3)

    while True:
        ret, img = cam.read()
        r = p.apply_async(func=check, args=(img,))

        cv2.imshow('Vedio', r.get())
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
