import numpy as np
import cv2
import os

os.chdir("C:\\Users\\PC_datascience")
os.chdir(".\\Documents\\GitHub\\computer-vision")
# Load an color image in grayscale

face_cascade = cv2.CascadeClassifier('.\\input_xml\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('.\\input_xml\\haarcascade_eye.xml')
img = cv2.imread('input\\legoland.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#test sur img
faces = face_cascade.detectMultiScale(img, 1.25, 3)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # regions of interest
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_color)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#test sur gray
faces = face_cascade.detectMultiScale(gray, 1.25, 3)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    # regions of interest
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()