import PIL
import numpy as np
import sklearn
from PIL import Image
from sklearn import cluster
import random
from random import randint
import collections
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

# Function
# Name: getImageRec
# Description: Sample ramdomic an image
# Input: picName(string)
# Output: result (num array)

def getImageRec(picName):
	numberSamples = 2000;
	image = Image.open(picName)
	imageVec = np.asarray(image)
	imageRec = np.reshape(imageVec,(-1,3))
	result = []
	imageSize = imageVec.shape
	imageSizeY = imageSize[0]
	imageSizeX = imageSize[1]
	muX = imageSizeX//2
	muY = imageSizeY//2
	ConstSigma = 1.5
	sigmaX = math.sqrt(muX)*ConstSigma
	sigmaY = math.sqrt(muY)*ConstSigma
	randX = np.random.normal(muX,sigmaX,numberSamples)	
	randY = np.random.normal(muY,sigmaY,numberSamples)	

	for i in range(numberSamples):
		#randNum = random.randrange(0, len(imageRec))
		randPiX = int(randX[i])
		randPiY = int(randY[i])
		if randPiX >= imageSizeX:
			randPiX = imageSizeX-1
		if randPiX < 0:
			randPiX = 0
		if randPiY >= imageSizeY:
			randPiY = imageSizeY-1
		if randPiY < 0:
			randPiY = 0
			
		result.append(imageVec[randPiX][randPiY])				
		#result.append(imageRec[randNum])
	return result

# Function
# Name: predictCluster
# Description: Get labels from image
# Input: name(string)
# Output: cluster.predict (num array)

def predictCluster(name):
	im = Image.open(name)
	imageVec = np.asarray(im)
	imageRec = np.reshape(im,(-1,3))
	return clusterer.predict(imageRec)

# Function
# Name: unZip
# Description: Convert a pointer List to a array with a specific 
# length
# Input: List(zipped List), collum (num), NumOfElements(num)
# Output: result (num array)

def unZip(List,collum,NumOfElements):
	result = []
	ListCount = 0
	for i in range(0,NumOfElements):
		if ListCount < len(List):
			if List[ListCount][0] == i:
				result.append(List[ListCount][collum])
				ListCount+=1
			else:
				result.append(0)
		else:
			result.append(0)
	return result

# Function
# Name: GetHistogramData
# Description: Get histogram values from cluster predict
# Input: Lbl(num array)
# Output: LblCnt(num array)

def GetHistogramData(Lbl):
	LblCnt = collections.Counter(Lbl)
	LblCnt = sorted(LblCnt.items())
	#del LblCnt[0]
	LblCnt = list(LblCnt)
	LblCnt = unZip(LblCnt,1,bins)
	return LblCnt	

# Class
# Name: FileComp
# Description: Stores file comparision data

class FileComp(object):

	# Function
	# Name: __init__
	# Description: __init__
	# Input: FileName(string), DistTarg(num)
	# Output: nothing

	def __init__(self,FileName, DistTarg):
		self.FileName = FileName
		self.DistTarg = DistTarg

# Function
# Name: GetRandomPic
# Description: Get a random picture
# Input: nothing
# Output: picStrTar(string)

def GetRandomPic():
	randPic = random.randrange(0,totalNumPic)
	randAng = random.randrange(0,finalAngle,Step)

	picStrTar='obj'+str(randPic+1)+'__'+str(randAng)+fileExt
	
	return picStrTar

# Function
# Name: SearchAll
# Description: Search a picture in all database
# Input: lblTarg (num array)
# Output: FileCompList (FileComp array)

def SearchAll(LblTarg):
	FileCompList = []
	f = open('./../Analisys.txt', 'w')
	for i in range(totalNumPic):
		print 'Picture: obj' + str(i+1)
		for j in range(initialAngle, finalAngle, Step):
			picString = 'obj'+str(i+1)+'__'+str(j)+fileExt #Modification: add fileExt
			LabelSch = predictCluster(picString)
			LabelSchCount = GetHistogramData(LabelSch)
			LabelSch_Ar = np.array(LabelSchCount)
		
			dist = np.linalg.norm(LblTarg-LabelSch_Ar)
			
			f.write('PicName: ' + picString + ' dist: {}\n'.format(dist))  # python will convert \n to os.linesep
			FileC = FileComp(picString,dist)			
			FileCompList.append(FileC)
	
	f.close()  

	import operator
	FileCompList.sort(key=operator.attrgetter('DistTarg'))
	return FileCompList

# Function
# Name: SearchAllHisto
# Description: Search a picture in all database memory
# Input: lblTarg (num array)
# Output: FileCompList (FileComp array)

def SearchAllHisto(LblTarg):
	global FileHisto	
	FileCompList = []
	f = open('./../Analisys.txt', 'w')
	for i in range(totalNumPic):
		for j in range(initialAngle, finalAngle, Step):
			MaxItems = (finalAngle - initialAngle)/Step
			FileH = FileHisto[i*MaxItems+j/Step]
			
			dist = np.linalg.norm(LblTarg-FileH.HistoData)
			
			f.write('PicName: ' + FileH.FileName + ' dist: {}\n'.format(dist))  # python will convert \n to os.linesep
			FileC = FileComp(FileH.FileName,dist)			
			FileCompList.append(FileC)
	
	f.close()  

	import operator
	FileCompList.sort(key=operator.attrgetter('DistTarg'))
	return FileCompList


# Function
# Name: LoadHisto
# Description: Load histogram data from database
# Input: nothing
# Output: nothing

def LoadHisto():
	global FileHisto	
	FileHisto = []
	for i in range(totalNumPic):
		print 'Picture: obj' + str(i+1)
		for j in range(initialAngle, finalAngle, Step):
			picString = 'obj'+str(i+1)+'__'+str(j)+fileExt #Modification: add fileExt
			LabelSch = predictCluster(picString)
			LabelSchCount = GetHistogramData(LabelSch)
			LabelSch_Ar = np.array(LabelSchCount)
			FileH = PicHisto(picString,LabelSch_Ar)
			FileHisto.append(FileH)

# Class
# Name: PicHisto
# Description: Stores file histogram data

class PicHisto(object):

	# Function
	# Name: __init__
	# Description: __init__
	# Input: FileName(string), HistoData(num array)
	# Output: nothing

	def __init__(self,FileName, HistoData):
		self.FileName = FileName
		self.HistoData = HistoData



# Function
# Name: FindPic
# Description: Find a random picture
# Input: picName (string)
# Output: nothing

def FindPic(picName):
	picStringTarg = picName
	print "File to search: " + picStringTarg

	LabelTarg  = predictCluster(picStringTarg)
	LabelTarCount = GetHistogramData(LabelTarg)
	LabelTar_Ar = np.array(LabelTarCount)
	#FileCList = SearchAll(LabelTar_Ar)
	FileCList = SearchAllHisto(LabelTar_Ar)

	for it in range(0, 10):
		print "FileName: {}".format(FileCList[it].FileName)
		print "DistTarg: {}".format(FileCList[it].DistTarg)
		print "---------------------------------------"
		#Modification: displaying images
		img = Image.open(FileCList[it].FileName)
		img.show(title=(FileCList[it].FileName+' - Similarity Level: '+str(it)))
	
# Function
# Name: Create3DHisto
# Description: Create 3D histogram from 2 images
# Input: picName1 (string), picName2 (string)
# Output: nothing

def Create3DHisto(picName1, picName2):
	global centers
	picStringTarg = picName1
	picStringComp = picName2
	print "File to search: " + picStringTarg
	print "File to compare: " + picStringComp

	LabelTarg  = predictCluster(picStringTarg)
	LabelTarCount = GetHistogramData(LabelTarg)
	LabelTar_Ar = np.array(LabelTarCount)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xData = centers[:,0]
	yData = centers[:,1]
	zData = centers[:,2]
			
	ax.scatter(xData,yData,zData,s=LabelTarCount)
	fig.suptitle(picStringTarg, fontsize=12)	
	ax.set_xlabel('Red axis')
	ax.set_ylabel('Green axis')
	ax.set_zlabel('Blue axis')

	fig1 = plt.figure()
	ay = fig1.add_subplot(111, projection='3d')

	LabelSch = predictCluster(picStringComp)
	LabelSchCount = GetHistogramData(LabelSch)
	LabelSch_Ar = np.array(LabelSchCount)

	ay.scatter(xData,yData,zData,s=LabelSchCount)
	fig1.suptitle(picStringComp, fontsize=12)	
	ay.set_xlabel('Red axis')
	ay.set_ylabel('Green axis')
	ay.set_zlabel('Blue axis')
	
	dist = np.linalg.norm(LabelTar_Ar-LabelSch_Ar)
	print 'Euclidean distance: {}'.format(dist)		
	img = Image.open(picStringTarg)
	img.show(title=(picStringTarg))		
	img = Image.open(picStringComp)
	img.show(title=(picStringComp))

	plt.show()




###############################################################
### MAIN
###############################################################

#Number of pictures
totalNumPic = 100

#Parameters of COIL images
initialAngle = 0
finalAngle = 360 
Step = 10
fileExt = '.ppm'

#Parameters of KMeans
bins = 128

#Debug Parameters
Analisys = 0
AnalisysIdle = 0

FileHisto = []

for i in range(totalNumPic):
	for j in range(initialAngle, finalAngle, Step):
		if j == 0:
			picString = 'obj'+str(i+1)+'__'+str(j)+fileExt
			imageRecFinal = getImageRec(picString)
		else:
			imageRecFinal = np.concatenate((imageRecFinal, getImageRec(picString)))

#Fitting cluster
clusterer = sklearn.cluster.KMeans(bins,n_init=1,max_iter = 1000)
labels = clusterer.fit(imageRecFinal)
centers = clusterer.cluster_centers_

# Load Histogram data
LoadHisto()

## Configure Mode
WorkDone = 1
while WorkDone:
	op = 0
	print 'Modes:'
	print '1 -> Find a random picture'
	print '2 -> Find a specific picture'
	print '3 -> Compare 3D Histogram'
	print '4 -> Compare 3D Histogram of a random pic with their position 0'
	print '5 -> Exit'	
	op = input('Enter your option: ')


	if op == 1:
		FindPic(GetRandomPic())	
	elif op == 2:
		pictureName = input('Enter filename: ')
		FindPic(pictureName)
	elif op == 3:	
		pictureName1 = input('Enter filename1: ')
		pictureName2 = input('Enter filename2: ')
		Create3DHisto(pictureName1, pictureName2)		
	elif op == 4:	
		pictureName1 = GetRandomPic()
		pictureName2 = pictureName1[0:pictureName1.index('_')]+'__0'+fileExt
		Create3DHisto(pictureName1, pictureName2)
	elif op == 5:
		WorkDone = 0
	else:
		print 'Command does not acknowledge.'
		print ''
		



		










