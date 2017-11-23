import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('121.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

for i in range(30,60,1):
	for j in range(30,60,1):
		print("Pixel %d-%d - Saturacao:%d"%(i, j, hsv[i][j][1]))

plt.imshow(hist,interpolation = 'nearest')
plt.show()

cv2.imshow('image',img)
cv2.waitKey(0)

hsv[56][56][1] = 0

img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR);

cv2.imshow('image',img)
cv2.waitKey(0)
