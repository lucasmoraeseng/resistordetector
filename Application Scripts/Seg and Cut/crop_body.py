import numpy as np
import cv2
import PIL
import sklearn
import random

from PIL import Image
from sklearn import cluster

data = '0_3.jpg'

img = cv2.imread(data)
mean = np.shape(img)[0]/2
img_crop = (img[mean-10:mean+10,:])

cv2.imshow("Cropped Image", img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

t = np.mean(img_crop,axis=0)
#Ttemp = t.astype(np.float64)

img_crop = np.zeros((400,400,3), np.uint8)

for i in range((np.shape(t)[0])):
	#print i
	#print Tu8[i]
	cv2.rectangle(img_crop, (175, i*2), (225,(i+1)*2), t[i], 3)

cv2.imshow("Cropped Image", img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

t_shape = t.reshape(1,t.shape[0],t.shape[1])
print t_shape
img_gray = cv2.cvtColor(t_shape, cv2.COLOR_RGB2GRAY)

img_hsv = cv2.cvtColor(t_shape,cv2.COLOR_BGR2HSV)




'''
cv2.imshow("Gray Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
KMEANS ALTERNATIVA

imageVec = np.asarray(img_crop)
imageRec = np.reshape(imageVec,(-1,3))

clusterer = sklearn.cluster.KMeans(6,n_init=1,max_iter = 1000)
labels = clusterer.fit(imageRec)
centers = clusterer.cluster_centers_
print centers
'''


