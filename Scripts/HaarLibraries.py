import os
import cv2
import subprocess
from shutil import copyfile

class PositiveFiles:
    Images = []
    def __init__(self,MainImage,ListOfImages):
        self.Main = MainImage        
        if len(ListOfImages)>0: 
            self.Images+=ListOfImages
    def IncludeImages(self,ListOfImages):
        self.Images += ListOfImages       

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

def RemoveDirectory(Dir):
    if os.path.exists(Dir):
        for fileList in os.listdir(Dir):
            if os.path.isdir(Dir+'/'+fileList):
                RemoveDirectory(Dir+'/'+fileList)                
            else:
                os.remove(Dir+'/'+fileList)
        subprocess.call(['rmdir',Dir])

def CreateDirIfNotExists(CreatePath):
    if not os.path.exists(CreatePath):  
        os.makedirs(CreatePath)

def CopyFolderImage(Input,Output):
    CopyFolderCounter = 0
    if os.path.exists(Output):
        for fileList in os.listdir(Input):
            if os.path.isdir(fileList):
                CopyFolderImage(Input+'/'+fileList,Output+'/'+fileList)
            else:        
                copyfile(Input+'/'+fileList,Output+'/'+str(CopyFolderCounter)+'.jpg')
            

def CustomImage(ImgPath,ImgSize,ImgNewName):
    print "Customizing img: " + ImgPath
    print "   new img Name: " + ImgNewName
    imgRead = cv2.imread(ImgPath,cv2.IMREAD_GRAYSCALE)
    
    #print "       SizeOld : " + str(ImgSize) 
    #print "       SizeNew : " + str(imgRead.shape) 
    Passed = 0
    try:
        Resized = cv2.resize(imgRead, ImgSize)
        cv2.imwrite(ImgNewName,Resized)
        Passed = 1
    except:
        os.remove(ImgPath)
        Passed = 0
        print 'Removing File: ' + ImgPath
    return Passed
    
def AdjImages(PathIn,PathOut,ImgSize):
    CreateDirIfNotExists(PathOut)
    ImgNameCounter = 0
    for imgInput in os.listdir(PathIn):
        ReturnCustom = CustomImage(PathIn+'/'+imgInput, ImgSize,PathOut+'/'+str(ImgNameCounter)+GetExtension(PathIn+'/'+imgInput))          
        if ReturnCustom == 1:        
            ImgNameCounter+=1

def FallOnePathIn(StrPath):
    Count=0
    SpacePos = 0
    FirstFound = 0
    for CharLine in StrPath:
        if CharLine == '/' and not FirstFound:
            SpacePos = Count
            FirstFound = 1
        Count+=1
    return StrPath[SpacePos+1:]

def CreateBGFile(ClearFile,NegPath):
    if ClearFile== 1:    
        if os.path.exists('../bg.txt'):        
            os.remove('../bg.txt')
    for NegImg in os.listdir(NegPath):
        line = NegPath + '/'+NegImg+'\n'
        with open('../bg.txt','a') as f:
            f.write(FallOnePathIn(line))

def CreateInfoFile(ClearFile,PosPath):
    if ClearFile== 1:    
        if os.path.exists('../info.dat'):        
            os.remove('../info.dat')
    for PosImg in os.listdir(PosPath):
        imgRead = cv2.imread(PosPath+'/'+PosImg,cv2.IMREAD_GRAYSCALE)
        imgReadShape = imgRead.shape
        ShapeXImg = str(imgReadShape[0])
        ShapeYImg = str(imgReadShape[1])        
        line = PosPath+'/'+PosImg+ ' 1 0 0 ' + ShapeXImg+' '+ShapeYImg+'\n'
        with open('../info.dat','a') as f:
            f.write(line)

def GetImageNamePositive(StringLine):
    Count=0
    SpacePos = 0
    FirstFound = 0
    for CharLine in StringLine:
        if CharLine == ' ' and not FirstFound:
            SpacePos = Count
            FirstFound = 1
        Count+=1
    return StringLine[0:SpacePos]

def GetAllPositivesImages(ListPosFile,PositiveVector):
    AllPosFiles = ReadFile(ListPosFile)
    for File in AllPosFiles:   
        FileName = GetImageNamePositive(File)
        TempPositive = PositiveFiles(FileName,[])
        PositiveVector.append(TempPositive)

def Callopencv_createsamples(FileToCreate,InfoLocation,ListOfBg,Angles,Inten,Size,ImgSize,BgColor,BgThresh):
    Commands = ['opencv_createsamples']
    Commands += ['-img',FileToCreate]
    Commands += ['-bg',ListOfBg]
    Commands += ['-info',InfoLocation]
    Commands += ['-pngoutput','info']
    Commands += ['-maxxangle',str(Angles[0])]
    Commands += ['-maxyangle',str(Angles[1])]
    Commands += ['-maxzangle',str(Angles[2])]
    Commands += ['-maxidev',str(Inten)]
    Commands += ['-num',str(Size)]
    Commands += ['-bgcolor',str(BgColor)]
    Commands += ['-bgthresh',str(BgThresh)]
    Commands += ['-w',str(ImgSize[0])]
    Commands += ['-h',str(ImgSize[1])]
    #print Commands
    subprocess.call(Commands)
    #return Commands

def Callopencv_createsamples_vec(FilePos,FileNeg,Vector,Size,ImgSize):
    Commands = ['opencv_createsamples']
    Commands += ['-info',FilePos]
    Commands += ['-bg',FileNeg]
    Commands += ['-vec',Vector]
    Commands += ['-num',str(Size)]
    Commands += ['-w',str(ImgSize[0])]
    Commands += ['-h',str(ImgSize[1])]
    #print Commands
    subprocess.call(Commands)

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
    #print Commands
    subprocess.call(Commands)   

def ReadFile(InfoPath):
    fRead = open(InfoPath,'r')
    fLines = fRead.readlines()
    return fLines

def PrepareOpenCV(CVPath,PositiveVector):   
    CVCounter = 0
    DBCounter = 0
    CreateDirIfNotExists(CVPath+'/data')
    CreateDirIfNotExists(CVPath+'/info')
    GetAllPositivesImages('../info.dat',PositiveVector)
    NegativeDB = ReadFile('../bg.txt')
    for PositiveImage in PositiveVector:
        #CreateDirIfNotExists(CVPath+'/info/'+str(CVCounter))
        InfoLocation = CVPath+'/info/'+'info'+str(CVCounter) +'.lst'
        Callopencv_createsamples(PositiveImage.Main,InfoLocation,'../bg.txt',(1.1,1.1,1.1),30,len(NegativeDB),(48,48),0,8)   
        #CreatedSamples = ReadFile(CVPath+'/info/info.lst')
        #PositiveImage.IncludeImages(CreatedSamples)
        CVCounter +=1    
        DBCounter += len(NegativeDB)


        
    



