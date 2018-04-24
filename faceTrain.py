#分析器训练
import cv2 as cv
import os
import numpy as np
from PIL import Image
from env import path_xml

# Path for face image database
path = './datadir'
recognizer = cv.face.LBPHFaceRecognizer_create()
face_detector = cv.CascadeClassifier(path_xml)

#取得图像数据和标签名
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = imagePath.split(".")[2]
        faces = face_detector.detectMultiScale(img_numpy)
        for x,y,w,h in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("正在学习头像，这需要一段时间，请等待...")
faces,ids = getImagesAndLabels(path)
print(ids)
ids = list(map(int,ids))
print(ids)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')

print("学习完成！分析了%d个用户的照片"%len(np.unique(ids)))
