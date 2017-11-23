from sys import argv
import numpy as np
import cv2

im = cv2.imread(argv[1]+'.jpg')

blur = cv2.blur(im,(3,3))

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 210, 255, cv2.THRESH_BINARY)

dst = cv2.inpaint(im,thresh,10,cv2.INPAINT_TELEA)

cv2.imshow('image',im)
cv2.waitKey(0)
cv2.imshow('image',blur)
cv2.waitKey(0)
cv2.imshow('image2',thresh)
cv2.waitKey(0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
