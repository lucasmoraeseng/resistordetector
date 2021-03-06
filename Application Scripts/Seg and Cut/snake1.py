import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import skimage.filters
import cv2
import math
import imutils

data = '134.jpg'

img = scipy.misc.imread(data, flatten=True)
print(np.shape(img))


#img = cv2.imread('26.jpg')
#print(np.shape(img))
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


y, x = np.nonzero(img)


x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])


cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)


sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1  
x_v2, y_v2 = evec2

print evals
print evec1
print evec2
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



theta = math.atan((x_v1)/(y_v1))  

img2 = cv2.imread(data)
rot = imutils.rotate(img2,math.degrees(theta))

print theta

cv2.imshow("Rotated (Problematic)", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()


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


