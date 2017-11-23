import subprocess
import os
import cv2
import subprocess
from shutil import copyfile

def Callopencv_createsamples_vec(FilePos,FileNeg,Vector,Size,ImgSize):
    Commands = ['opencv_createsamples']
    Commands += ['-info',FilePos]
    Commands += ['-bg',FileNeg]
    Commands += ['-vec',Vector]
    Commands += ['-num',str(Size)]
    Commands += ['-w',str(ImgSize[0])]
    Commands += ['-h',str(ImgSize[1])]
    print Commands
    subprocess.call(Commands)

def RemoveFinalSamples(CvPath):
    if os.path.exists(CvPath + '/finalVector.vec'):
       os.remove(CvPath + '/finalVector.vec')

def ReadFile(InfoPath):
    fRead = open(InfoPath,'r')
    fLines = fRead.readlines()
    return fLines

def GetAllListFiles(Location):
    if os.path.exists(Location+ '/FullList.dat'):
       os.remove(Location+ '/FullList.dat')
    FileWrite = open(Location+'/FullList.dat','a')
    
    for FileRead in os.listdir(Location):
        if GetExtension(Location+'/'+FileRead) == '.lst':
            fRead = open(Location+'/'+FileRead,'r')
            fReadLines = fRead.readlines()
            for fLine in fReadLines:
                FileWrite.write(fLine)
            fRead.close()
    FileWrite.close()

def GetExtension(FileName):
    fileExt = FileName[len(FileName)-5:]
    dotPos = 0
    dotCount =0
    for charExt in fileExt:
        if charExt == '.':
            dotPos = dotCount 
        dotCount +=1
    
    fileExt = fileExt[dotPos:]
    
    return fileExt

def Callopencv_train(OutputDir,Vector,FileNeg,numPos,numNeg,numStages, ImgSize):
    Commands = ['opencv_traincascade']
    Commands += ['-data',OutputDir]
    Commands += ['-vec',Vector]
    Commands += ['-bg',FileNeg]
    Commands += ['-numPos',str(numPos)]
    Commands += ['-numNeg',str(numNeg)]
    Commands += ['-numStages',str(numStages)]
    Commands += ['-precalcValBufSize','1024']
    Commands += ['-precalcIdxBufSize', '1024']
    Commands += ['-featureType','HAAR']
    Commands += ['-minHitRate','0.995']
    Commands += ['-maxFalseAlarmRate','0.5']
    Commands += ['-mode','ALL']
    Commands += ['-w',str(ImgSize[0])]
    Commands += ['-h',str(ImgSize[1])]
    print Commands
    subprocess.call(Commands)   

def JoinAndTrain_openCV(CVPath,Sta):    
    DBCounter = 0
    NegativeDB = ReadFile('../bg.txt')
    GetAllListFiles('info')
    DBAll = ReadFile('info/FullList.dat')
    print str(len(NegativeDB)) + '-'+str(len(DBAll))    
    DBCounter = len(DBAll)
    RemoveFinalSamples(CVPath)
    Callopencv_createsamples_vec('info/FullList.dat','../bg.txt','finalVector.vec',DBCounter,(48,48))    
    Callopencv_train('data','finalVector.vec','../bg.txt',len(NegativeDB)*2.1,len(NegativeDB),Sta,(48,48))

def Train_openCV(CVPath,Sta):
    DBCounter = 0
    NegativeDB = ReadFile('../bg.txt')
    GetAllListFiles('info')
    DBAll = ReadFile('info/FullList.dat')
    print str(len(NegativeDB)) + '-'+str(len(DBAll))    
    DBCounter = len(DBAll)
    Callopencv_train('data','finalVector.vec','../bg.txt',len(NegativeDB)*2.1,len(NegativeDB),Sta,(48,48))


##-----------MAIN---------------------##
Stages = input('Enter number of stages: ')
if Stages >7:
    JoinAndTrain_openCV('./', Stages)
else:
    Stages = input('Enter number of stages: ')
    Train_openCV('./', Stages)



