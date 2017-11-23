import cv2
import os

ImgCounter = 150
MainPath = 'Originals'

for img in os.listdir(MainPath):
    imgReader = cv2.imread(MainPath+'/'+img)
    SizeY = int(imgReader.shape[1]*0.25)
    SizeX = int(imgReader.shape[0]*0.25)
    imgResized = cv2.resize(imgReader,(SizeY,SizeX))
    cv2.imwrite(str(ImgCounter)+'.jpg',imgResized)
    ImgCounter+=1

