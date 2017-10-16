# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 23:02:21 2017

@author: PC_datascience
"""
import os
import numpy as np
import cv2

os.chdir("C:\\Users\\PC_datascience")
os.chdir(".\\Documents\\GitHub\\computer-vision")
face_cascade = cv2.CascadeClassifier('.\\input_xml\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('.\\input_xml\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#test sur gray
faces = face_cascade.detectMultiScale(gray, 1.25, 3)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    # regions of interest
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

   # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()