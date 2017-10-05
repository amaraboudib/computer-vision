import numpy as np
import cv2
import os

os.chdir("C:\\Users\\PC_datascience")
os.chdir(".\\Documents\\GitHub\\computer-vision")

# Load an color image in grayscale
img = cv2.imread('input\messi.jpg')

print(img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()