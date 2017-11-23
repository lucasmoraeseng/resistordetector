# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import os
import HaarLibraries

objCascade = cv2.CascadeClassifier("HaarXML/ResNew22_cascade.xml")

def GetName(Line):
    dotPos = 0
    dotCount =0
    fileExt = ''
    for charExt in Line:
        if charExt == ' ':
            dotPos = dotCount 
        dotCount +=1
    
    fileExt = Line[:dotPos]
    
    return fileExt

def GetNum(Line):
    dotPos = 0
    dotCount =0
    fileExt = ''
    for charExt in Line:
        if charExt == ' ':
            dotPos = dotCount 
        dotCount +=1
    fileExt = Line[dotPos+1:]
    #print fileExt
    fileExt = int(float(fileExt))
    return fileExt

def ReadTestImage(FileIn):
    fileRead = open(FileIn,'r')
    fileLines = fileRead.readlines()
    OutVector = []
    for fLine in fileLines:
        fLineName = GetName (fLine)
        fLineCount = GetNum (fLine)
        OutVector.append((fLineName,fLineCount))
    return OutVector


def TestImage(TestPath, Sample,ResultFilePath,ResultPath,CroppedPath,Cascade,ShowImage):
    global objCascade
    fileResult = open(TestPath+'/'+ResultFilePath,'a')
    img = cv2.imread(TestPath+'/'+Sample[0],cv2.IMREAD_GRAYSCALE)
    imgColor = cv2.imread(TestPath+'/'+Sample[0])
    #print 'shape img: %s -> shape imgColor: %s' %(str(img.shape),str(imgColor.shape))

    # Detect faces in the image
    objRes = objCascade.detectMultiScale(
        img,
        scaleFactor=1.02,
        minNeighbors=20,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    #print("Found {0} faces!".format(len(faces)))
    fileResult.write('Filename: '+ Sample[0]+'\n')
    fileResult.write('\tExpectec Result: ' + str(Sample[1])+'\n')
    fileResult.write('\tReceived Result: ' + str(len(objRes))+'\n')  
    objCount = 0  

    imgColored = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    IndexIma = 0
    for (x, y, w, h) in objRes:
        cv2.rectangle(imgColored, (x, y), (x+w, y+h), (0, 255, 0), 1)
        fileResult.write('\t\tObject'+str(objCount)+': ')
        fileResult.write('x: '+str(x)+' y: '+str(y))
        fileResult.write(' w: '+str(w)+' h: '+str(h)+'\n')
        imgCrop = imgColor[y:y+h,x:x+w].copy()
        #imgCrop = imgColored[y:h,x:w]
        cv2.imwrite(CroppedPath + '/'+ str(IndexIma)+'_'+Sample[0],imgCrop)
        IndexIma += 1
    fileResult.close()

    cv2.imwrite(ResultPath+'/'+Sample[0],imgColored)
    if ShowImage == 1:
        cv2.imshow('frame', imgColored)
        let = cv2.waitKey(200)
    
    ReturnValue = 0    
    if len(objRes) == Sample[1]:
       ReturnValue =1
       
    return ReturnValue     
    
#----------Main----------------#

TestPath = 'TestDB'
TestFile = 'test.dat'
fileOut = 'ResultTest.dat'
ResultDB = 'ResultDB'
CroppedDB = 'CroppedDB'
PerResult = 0
SumTest = 0


if os.path.exists(ResultDB):
    HaarLibraries.RemoveDirectory(ResultDB)

if os.path.exists(CroppedDB):
    HaarLibraries.RemoveDirectory(CroppedDB)

if os.path.exists(fileOut):
    os.remove(fileOut)

HaarLibraries.CreateDirIfNotExists(ResultDB)
HaarLibraries.CreateDirIfNotExists(CroppedDB)

for Test in ReadTestImage(TestPath+ '/'+TestFile):
    SumTest +=1
    Result = TestImage(TestPath,Test,fileOut,ResultDB,CroppedDB,objCascade,1)
    print 'SumTest: '+str(SumTest)+ ' Result: '+ str(Result) 
    if Result == 1:
        PerResult+=1

PerResult = (PerResult/(SumTest*1.0))*100
    
print 'Result %s in %s images' % (PerResult,SumTest)        
fileResult = open(fileOut,'a')
fileResult.write('\n\n\nResult: '+ str(PerResult))
fileResult.write(' in ' + str(SumTest) + ' images\n')
fileResult.close()



