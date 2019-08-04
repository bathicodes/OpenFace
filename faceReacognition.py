import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)

    return faces,gray_img



