import os
import cv2
import subprocess
from shutil import copyfile
import HaarLibraries
from HaarLibraries import RemoveDirectory
from HaarLibraries import GetExtension
from HaarLibraries import CreateDirIfNotExists
from HaarLibraries import CopyFolderImage
from HaarLibraries import CustomImage
from HaarLibraries import AdjImages
from HaarLibraries import FallOnePathIn
from HaarLibraries import CreateBGFile
from HaarLibraries import CreateInfoFile
from HaarLibraries import GetImageNamePositive
from HaarLibraries import GetAllPositivesImages
from HaarLibraries import Callopencv_createsamples
from HaarLibraries import Callopencv_createsamples_vec
from HaarLibraries import Callopencv_train
from HaarLibraries import ReadFile
from HaarLibraries import PrepareOpenCV
####-----------MAIN----------####
Mode = 0 #0 - Full
         #1 - OnlyTrain

PositiveVectorObj = []

PathPos = '../PosFinal'
PathNeg = '../NegFinal'
OpenCVPath = '../OpenCV_Haar'

#SizeNegImage = (512,384)
SizeNegImage = (256,192)
SizePosImage = (50,50)

Mode = input('Enter Mode (0-Full,1-OnlyTrain): ')

if Mode == 0:

    RemoveDirectory(PathPos)
    RemoveDirectory(PathNeg)
    RemoveDirectory(OpenCVPath)

    AdjImages('../Pos/PosCell',PathPos,SizePosImage)
    AdjImages('../Neg/Background',PathNeg,SizeNegImage)

    CreateInfoFile(1,PathPos)
    CreateBGFile(1,PathNeg)

    CreateDirIfNotExists(OpenCVPath)

    if not os.path.exists(OpenCVPath+'/trainHaar.py'):
        copyfile('trainHaar.py',OpenCVPath+'/trainHaar.py')

    PrepareOpenCV(OpenCVPath,PositiveVectorObj)
else:
    if not os.path.exists(OpenCVPath+'/trainHaar.py'):
        copyfile('trainHaar.py',OpenCVPath+'/trainHaar.py')
    else:
        os.remove(OpenCVPath+'/trainHaar.py')
        copyfile('trainHaar.py',OpenCVPath+'/trainHaar.py')
        
        
    



