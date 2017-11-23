import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import skimage.filters
import cv2
import math
import imutils
from sys import argv
import colorsys
#import sklearn
#from sklearn import cluster

data = 'tst1.jpg'

img = cv2.imread(data)
print(np.shape(img))
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
#tst = cv2.Canny(img, 100, 200)
'''
cv2.imshow('Gray Scale Image',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Canny Image',tst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
y, x = np.nonzero(thresh)

#x = x - np.mean(x)
#y = y - np.mean(y)
coords = np.vstack([x, y])


cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1  
x_v2, y_v2 = evec2

theta1 = math.atan(abs((x_v1)/(y_v1)))
'''

#img2 = cv2.imread(data)
#img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

sum_line = np.zeros(((np.shape(img)[0],3)))

aux = 0
for x in range(0, 360, 5):
	print np.shape(img)
	rot = imutils.rotate(img,x)
	x_mean = np.shape(img)[1]/2
	for j in range(np.shape(img)[0]):
		range_hor = range(x_mean-5,x_mean+5,1)
		for i in range_hor:
			aux = aux+img[j][i]
		aux = aux/len(range_hor)
		print aux
		sum_line[j] = aux
print sum_line
		

cv2.imshow("Rotated Image", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = (np.shape(rot))[1]
d = (np.shape(rot))[2]

t = np.nanmean(np.where(rot!=0,rot,np.nan),axis=1)

imt = np.zeros((400,400,3), np.uint8)

for i in range((np.shape(t)[0])):
	cv2.rectangle(imt, (175, i*2), (225,(i+1)*2), t[i], 3)

img = cv2.cvtColor(imt, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
tst = cv2.Canny(img, 20, 50)

for i in range((np.shape(t)[0])-1):
	dist = np.linalg.norm(t[i]-t[i+1])
	print dist
		

cv2.imshow("Rotated (Problematic)", imt)
cv2.waitKey(0)
cv2.destroyAllWindows()

#red, blue, yellow, and gray
boundaries = [
([17, 15, 100], [50, 56, 200]),
([86, 31, 4], [220, 88, 50]),
([25, 146, 190], [62, 174, 250]),
([103, 86, 65], [145, 133, 128])]

mask = cv2.inRange(imt, np.array([17, 15, 100], dtype="uint8"), np.array([50, 56, 200], dtype="uint8"))
output = cv2.bitwise_and(imt, imt, mask=mask)


cv2.imshow("Rotated (Problematic)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()





