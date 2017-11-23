from sys import argv
import numpy as np
import cv2

im = cv2.imread('26.jpg')

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY_INV)

ind = np.argwhere(thresh==0)
cov = np.cov(ind)
#values, vectors = np.linalg.eig(cov)																																																																																		
