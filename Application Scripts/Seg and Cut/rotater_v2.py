import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import skimage.filters
import cv2
import math
import imutils
from sys import argv

data = argv[1]+'.jpg'

#img = scipy.misc.imread(data, flatten=True)
#print(np.shape(img))


img = cv2.imread(data)
print(np.shape(img))
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)


cv2.imshow('image',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


y, x = np.nonzero(thresh)
print np.shape(x)
print np.shape(y)


#x = x - np.mean(x)
#y = y - np.mean(y)
coords = np.vstack([x, y])


cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)
print evals
print evecs

sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1  
x_v2, y_v2 = evec2

'''
deg = 45+math.degrees(math.atan(evec1[1]/evec1[0]))

img2 = cv2.imread(data)
rot = imutils.rotate(img2,deg)

print deg

cv2.imshow("Rotated (Problematic)", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

scale = 20
plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')
plt.plot(x, y, 'k.')
plt.axis('equal')
plt.gca().invert_yaxis()  



theta1 = math.atan(abs((x_v1)/(y_v1)))
theta2 = math.atan(abs((x_v2)/(y_v2)))    

img2 = cv2.imread(data)
rot = imutils.rotate(img2,math.degrees(theta1))
rot2 = imutils.rotate(img2,math.degrees(theta2))

print math.degrees(theta1)
print math.degrees(theta2)

cv2.imshow("Theta1", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Theta2", rot2)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = (np.shape(rot))[1]
d = (np.shape(rot))[2]

t = np.nanmean(np.where(rot!=0,rot,np.nan),axis=1)

imt = np.zeros((400,400,3), np.uint8)

for i in range((np.shape(t)[0])):
	#print i
	#print t[i]
	cv2.rectangle(imt, (175, i*2), (225,(i+1)*2), t[i], 3)

cv2.imshow("Rotated (Problematic)", imt)
cv2.waitKey(0)
cv2.destroyAllWindows()

#hist = cv2.calcHist([rot],[0],None,[256],[0,256])


'''
rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat *coords

x_transformed, y_transformed = transformed_mat.A
plt.plot(x_transformed, y_transformed, 'g.')
plt.show()



def raw_moment(data, i_order, j_order):
  nrows, ncols = data.shape
  y_indices, x_indicies = np.mgrid[:nrows, :ncols]
  return (data * x_indicies**i_order * y_indices**j_order).sum()


def moments_cov(data):
  data_sum = data.sum()
  m10 = raw_moment(data, 1, 0)
  m01 = raw_moment(data, 0, 1)
  x_centroid = m10 / data_sum
  y_centroid = m01 / data_sum
  u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
  u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
  u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
  cov = np.array([[u20, u11], [u11, u02]])
  return cov


img = scipy.misc.imread('/home/lindomar/Desktop/Snake Adaptive Segmentation/oval.png', flatten=1)
cov = moments_cov(img)
evals, evecs = np.linalg.eig(cov)'''


