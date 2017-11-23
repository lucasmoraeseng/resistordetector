import numpy as np
import cv2
import PIL
import sklearn
import random
from matplotlib import pyplot as plt
import sys

from PIL import Image
from sklearn import cluster
from scipy import signal

import ColorFinder

objCascade = cv2.CascadeClassifier("HaarXML/ResNew22_cascade.xml")

def GetDataLine(Data,Line):
    ToReturn = []
    for I in range(Data.shape[0]):
        ToReturn += [Data[Line][I]]
    return ToReturn

def GetDataCollum(Data,Collum):
    ToReturn = []
    for I in range(Data.shape[1]):
        ToReturn+=[Data[I][Collum]]
    return ToReturn

def GetMeanConvColour(Data,Kernel):
    Result = np.zeros((3))
    DataB = Data[:,:,0]
    DataG = Data[:,:,1]
    DataR = Data[:,:,2]
    Result[0] = signal.convolve2d(DataB,Kernel,mode='valid',boundary='symm')/9
    Result[1] = signal.convolve2d(DataG,Kernel,mode='valid',boundary='symm')/9
    Result[2] = signal.convolve2d(DataR,Kernel,mode='valid',boundary='symm')/9
    #print Result
    return Result

def RemoveHighlight(ImageIn):
    _imgOut = ImageIn.copy()
    ImgGray = cv2.cvtColor(ImageIn.copy(), cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(ImgGray,cv2.cv.CV_32F,1,0,-1)
    gradY = cv2.Sobel(ImgGray,cv2.cv.CV_32F,0,1,-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (5, 5))
    (_, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    Kern = np.array([[1,1,1],[1,1,1],[1,1,1]])

    for i in range(_imgOut.shape[0]):
        for j in range(_imgOut.shape[1]):
            if thresh[i,j] >127:
                _imgOut[i,j] = [_imgOut[i,j,0],_imgOut[i,j,1],_imgOut[i,j,2]]
            else:
                _imgOut[i,j] = 0
                if i>0 and j>0 and i<_imgOut.shape[0]-2 and j< _imgOut.shape[1]-2:
                    _imgOut[i,j] = GetMeanConvColour(_imgOut[i-1:i+2,j-1:j+2],Kern)
    return _imgOut

def CropResistor(Data):
    imgGray = cv2.cvtColor(Data.copy(), cv2.COLOR_BGR2GRAY)

    imgGray = cv2.blur(imgGray,(12,12))
    imgGray = cv2.blur(imgGray,(12,12))
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(imgGray,cv2.cv.CV_32F,1,0,-1)
    gradY = cv2.Sobel(imgGray,cv2.cv.CV_32F,0,1,-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    ThreshShape = thresh.shape

    LineIni = 0
    LineEnd = 0

    for i in range(ThreshShape[0]):
        DataX = GetDataLine(thresh,i)
        MaxiX = max(DataX)
        #print MaxiX
        if LineIni == 0:
            if MaxiX > 0:
                LineIni = i
        else:
            if LineEnd == 0:
                if MaxiX == 0:
                    LineEnd = i
    
    assert not(LineIni == 0 and LineEnd==0)
    
    if LineEnd == 0:
        LineEnd = ThreshShape[0]

    imgFinal = Data.copy()
    imgFinal = imgFinal[LineIni:LineEnd+1,:]

    #cv2.imshow("Cropped Image", gradient)
    #cv2.waitKey(0)
    #cv2.imshow("Cropped Image", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return imgFinal

def GetDrawResistor(DataIn):
	Data = DataIn.copy()
	t = np.uint8(np.mean(Data,axis=0))

	DrawResistor = np.zeros((Data.shape[0],Data.shape[1],3), np.uint8)
	DRWidth = Data.shape[1]
	DRHeight = Data.shape[0]

	for i in range((np.shape(t)[0])):
		cv2.rectangle(Data,(i,0), (i+1,DRHeight), np.float64(t[i]), 1)
	
	return Data

def GetRangeColors(Data,Pos):
	ColorOut = []
	ColorData = []

	#ColorData = Data[0:Pos,:]

	for i in range(0,len(Pos),2):
		ColorData = Data[:,Pos[i]:Pos[i+1]]
		ColorTemp = ColorData[:]
		MeanColorLines = np.mean(ColorTemp,axis=1)
		MeanColor = np.mean(MeanColorLines,axis=0)
		ColorOut += [MeanColor.tolist()]

	return np.uint8(ColorOut)	
		
def DrawFinalResistor(Stripes,SizeFinal):
    ResFinalBack = np.zeros(SizeFinal,dtype = np.uint8)
    StripeWidth = int(SizeFinal[1]/Stripes.shape[0])   
    #print StripeWidth 
    for i in range(Stripes.shape[0]):
        cv2.rectangle(ResFinalBack,(i*StripeWidth,0), (((i+1)*StripeWidth)-1,SizeFinal[1]), np.float64(Stripes[i]), -1)
        #print 'StripeColor: ' + str(Stripes[i]) + ' X0: ' + str(i*StripeWidth)+ ' X1: ' + str(((i+1)*StripeWidth)-1) + ' Y1: ' + str(SizeFinal[1]) 
    return ResFinalBack
	
def CheckResistor(imgIn):
    imgGray = cv2.cvtColor(imgIn.copy(), cv2.COLOR_BGR2GRAY)

    imgGray = cv2.blur(imgGray,(12,12))
    imgGray = cv2.blur(imgGray,(12,12))
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(imgGray,cv2.cv.CV_32F,1,0,-1)
    gradY = cv2.Sobel(imgGray,cv2.cv.CV_32F,0,1,-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (12, 12))
    (_, thresh) = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    ThreshShape = thresh.shape

    LineIni = 0
    LineEnd = 0

    for i in range(ThreshShape[0]):
        DataX = GetDataLine(thresh,i)
        MaxiX = max(DataX)
        #print MaxiX
        if LineIni == 0:
            if MaxiX > 0:
                LineIni = i
        else:
            if LineEnd == 0:
                if MaxiX == 0:
                    LineEnd = i    
    if LineEnd - LineIni > 78:
        print 'Size: ' + str(LineIni) + ' '+ str(LineEnd)
        return True
    else:
        print 'Size: ' + str(LineIni) + ' '+ str(LineEnd)
        return False

def GetRangeDiffColors(Data,Pos):
	ColorOut = []
	ColorData = []

	#ColorData = Data[0:Pos,:]

	ColorData = Data[:,0:Pos[0]]
	ColorTemp = ColorData[:]
	MeanColorLines = np.mean(ColorTemp,axis=1)
	MeanColor = np.mean(MeanColorLines,axis=0)
	ColorOut += [MeanColor.tolist()]    

	for i in range(0,len(Pos)-1,1):
		ColorData = Data[:,Pos[i]:Pos[i+1]]
		ColorTemp = ColorData[:]
		MeanColorLines = np.mean(ColorTemp,axis=1)
		MeanColor = np.mean(MeanColorLines,axis=0)
		ColorOut += [MeanColor.tolist()]

	ColorData = Data[:,Pos[len(Pos)-1]:]
	ColorTemp = ColorData[:]
	MeanColorLines = np.mean(ColorTemp,axis=1)
	MeanColor = np.mean(MeanColorLines,axis=0)
	ColorOut += [MeanColor.tolist()]  

	return np.uint8(ColorOut)	

def FindStripesColor(ImageData, Debug, Test):
    imgOriginal = ImageData[:]
    imgTr = RemoveHighlight(imgOriginal)

    imgBox = CropResistor(imgTr)
    imgBoxShape = imgBox.shape

    MiddleX = int(imgBoxShape[0]/2)
    MiddleY = int(imgBoxShape[1]/2)

    Range = int(imgBoxShape[0]*0.2)

    ResistorRange = imgBox[MiddleX-Range:MiddleX+Range+1,:]
    #ResRangeClean = cv2.blur(ResistorRange,(12,12))
    ResRangeClean = cv2.bilateralFilter(ResistorRange,18,75,75)
    #ResRangeClean = cv2.medianBlur(ResistorRange,5)


    #######################################################
    ## body treatment
    #######################################################
    DrawResistor = GetDrawResistor(ResRangeClean)

    #DRBlur = cv2.blur(DrawResistor, (3, 3))
    #DRBlur = cv2.medianBlur(DrawResistor,5)

    #DRGray = cv2.cvtColor(DRBlur, cv2.COLOR_BGR2GRAY)
    DRGray = cv2.cvtColor(DrawResistor, cv2.COLOR_BGR2GRAY)

    gradGrayX = cv2.Sobel(DRGray,cv2.cv.CV_32F,1,0,-1)
    gradGrayY = cv2.Sobel(DRGray,cv2.cv.CV_32F,0,1,-1)

    # subtract the y-gradient from the x-gradient
    gradientGray = cv2.subtract(gradGrayX, gradGrayY)
    gradientGray = cv2.convertScaleAbs(gradientGray) 

    (_, threshGray) = cv2.threshold(gradientGray, 30, 255, cv2.THRESH_BINARY)


    xStem = []
    yStem = []

    DiffDistance = 5

    for i in range(DrawResistor.shape[1]-DiffDistance):
        Point1 = np.float64(DrawResistor[DrawResistor.shape[0]/2,i+DiffDistance])	
        Point2 = np.float64(DrawResistor[DrawResistor.shape[0]/2,i])
        Diff = np.linalg.norm(Point1-Point2)    
        xStem += [i]
        yStem += [Diff]

    del xStem[0:3]
    del yStem[0:3] 

    del xStem[len(xStem)-3:len(xStem)]
    del yStem[len(yStem)-3:len(yStem)]

    maxY = max(yStem)

    for i in range(len(yStem)):
	    yStem[i] = yStem[i]/maxY

    BorderXColor = []
    BorderYColor = []

    for i in range(len(yStem)):
        if (yStem[i] > 0.5):
            BorderYColor += [yStem[i]]
            BorderXColor += [xStem[i]]

    if Debug:
        print BorderXColor
        print BorderYColor
        print min(BorderYColor)

    BorderXFinal = []
    BorderYFinal = []

    BorderAcc = []
    for i in range(len(BorderXColor)-1):
        BorderAcc += [BorderXColor[i]]
        #print BorderAcc
        #print 'Ite: '+ str(i)
        if(BorderXColor[i+1] - BorderAcc[-1:][0])>2:
            BMean = np.mean(BorderAcc)
            BorderMean = int(BMean)
            #print BorderAcc
            #print 'Mean: ' + str(BorderMean) + ' MeanFloat: ' + str(BMean)
            BorderXFinal += [BorderMean]		
            BorderAcc = []
        if(i == len(BorderXColor)-2):
            BorderAcc += [BorderXColor[i+1]]
            BMean = np.mean(BorderAcc)
            BorderMean = int(BMean)
            #print BorderAcc		
            #print 'Mean: ' + str(BorderMean) + ' MeanFloat: ' + str(BMean)
            BorderXFinal += [BorderMean]		
            BorderAcc = []
    
    if Test:    
        ColorDiffAxis = GetRangeDiffColors(DrawResistor,BorderXFinal)
        ResFinalDiffDrw = DrawFinalResistor(ColorDiffAxis.copy(),(100,100,3))
        ColorDiffMean = np.mean(ColorDiffAxis,axis=0)
        print ColorDiffAxis
        print ColorDiffMean

        DiffAxis = []
        DiffXAxis = []    
        
        for i in range(len(ColorDiffAxis)-1):
            PAxis1 = np.float64(ColorDiffAxis[i])	
            PAxis2 = np.float64(ColorDiffAxis[i+1])
            TempDiff = np.linalg.norm(PAxis2-PAxis1)         
            DiffAxis +=[TempDiff] 
            DiffXAxis +=[i]

    if not Test:
        if Debug:
            print BorderXFinal
        if len(BorderXFinal) %2 != 0:
            BorderXF = [] 
            BorderYF = []
            for i in range(len(BorderXFinal)):
                for j in range(len(BorderXColor)):
                 if BorderXColor[j] == BorderXFinal[i]:
                     BorderXF += [BorderXColor[j]]
                     BorderYF += [BorderYColor[j]]
                     break
            BorderYMin = min(BorderYF)
            BorderXMinPos = 0        
            if Debug:            
                print BorderXF            
                print BorderYF
                print BorderYMin
        
            for i in range(len(BorderXF)):
                if BorderYMin == BorderYF[i]:
                    BorderXMinPos = BorderXF[i]
                    break

            for i in range(len(BorderXFinal)):
                if BorderXFinal[i] == BorderXMinPos:
                    del BorderXFinal[i]
                    break                        
        if Debug:
            print BorderXFinal

        ColorAxis = GetRangeColors(DrawResistor,BorderXFinal)

        if Debug:
            print ColorAxis

        Colors = ColorFinder.GetRangesColors(ColorAxis)

        if Debug:
            print Colors

        ResFinalDrw = DrawFinalResistor(ColorAxis.copy(),(100,100,3))

    fig1 = plt.figure(1)
    fig1.clf()
    #plt.ion() # enables interactive mode

    plt221 = fig1.add_subplot(2,2,1)
    plt221.imshow(cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB))
    plt221.set_title('Original')

    plt222 = fig1.add_subplot(2,2,2)
    #plt222.imshow(threshGray,cmap = 'gray')
    if not Test:
        plt222.imshow(cv2.cvtColor(ResFinalDrw, cv2.COLOR_BGR2RGB))
    else:
        plt222.imshow(cv2.cvtColor(ResFinalDiffDrw, cv2.COLOR_BGR2RGB))
    plt222.set_title('Final Resistor Draw')

    plt223 = fig1.add_subplot(2,2,3)
    plt223.imshow(cv2.cvtColor(ResRangeClean, cv2.COLOR_BGR2RGB))
    plt223.set_title('Boxed')

    plt224 = fig1.add_subplot(2,2,4)
    #plt224.imshow(cv2.cvtColor(DrawResistor, cv2.COLOR_BGR2RGB))
    #plt224.imshow(cv2.cvtColor(ResistorRange, cv2.COLOR_BGR2RGB))          
    plt224.plot(xStem,yStem)
    plt224.set_title('Diff')

    fig1.show()
    #fig2 = plt.figure(2)
    #fig2.clf()
    
    #if not Test:
    #    plt311_ = fig2.add_subplot(1,1,1)
    #else:
    #    plt311_ = fig2.add_subplot(2,1,1)            
    #plt311_.plot(xStem,yStem)

    #if Test:
    #    plt312_ = fig2.add_subplot(2,1,2)
    #    plt312_.plot(DiffXAxis,DiffAxis)

    #fig2.show()

    if not Test:
        return Colors
    else:
        return [-1]
    
def FindInImage(Sample):
    global objCascade
    imgColor = cv2.imread(Sample) 
    if imgColor.shape[0] > 512:
        SizeY = int(imgColor.shape[1]*0.5)
        SizeX = int(imgColor.shape[0]*0.5)
        imgResized = cv2.resize(imgColor,(SizeY,SizeX))
        cv2.imshow('frame', imgResized)
    else:
        cv2.imshow('frame', imgColor)
    let = cv2.waitKey(0)
    imgGray = cv2.cvtColor(imgColor.copy(), cv2.COLOR_BGR2GRAY)
    #print 'shape img: %s -> shape imgColor: %s' %(str(img.shape),str(imgColor.shape))

    # Detect faces in the image
    objRes = objCascade.detectMultiScale(
        imgGray,
        scaleFactor=1.02,
        minNeighbors=20,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    #print("Found {0} faces!".format(len(faces)))
    objCount = 0  

    #imgColored = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    imgColored = imgColor[:]     
   
    IndexIma = 0
    ImageListName = []
    for (x, y, w, h) in objRes:
        cv2.rectangle(imgColored, (x, y), (x+w, y+h), (0, 255, 0), 1)
        imgCrop = imgColor[y:y+h,x:x+w].copy()
        #imgCrop = imgColored[y:h,x:w]
        ImageListName += ['./Temp/'+ str(IndexIma)+'.jpg']
        cv2.imwrite(ImageListName[IndexIma],imgCrop)
        IndexIma += 1

           
    return ImageListName   

def resval(resistor,Debug):
    colorBand = 	{
        'black' : 0,
        'brown' : 1,
        'red' : 2,
        'orange' : 3,
        'yellow' : 4,
        'green' : 5,
        'blue' : 6,
        'violet' : 7,
        'grey' : 8,
        'white' : 9,
        'unidentified' : 1
        }

    resCom = 	[10, 11, 12, 13, 15, 16, 18, 20, 22, 24,
                27, 30, 33, 36, 39, 43, 47, 51, 56, 62,
                68, 75, 82, 91
                ]

    if len(resistor) != 3:
        if Debug:
            print 'Resistance not identified'
        return 0
    else:
        if resistor[0] == 'black' and resistor[2] == 'black':
            if Debug:
                print 'Resistance not identified'
            return 0  
        else:
            valueF = float(str(colorBand[resistor[0]])+str(colorBand[resistor[1]]))
            if valueF in resCom:            
                valueF *=  pow(10,colorBand[resistor[2]])
            else:
                valueF = -1    
            
            valueR = float(str(colorBand[resistor[2]])+str(colorBand[resistor[1]]))
            if valueR in resCom: 
                valueR *=  pow(10,colorBand[resistor[0]])
            else:
                valueR = -1               
                      
            if valueF == -1 and valueR == -1:
                return -1
            elif valueF == -1:
                return valueR
            elif valueR == -1:
                return valueF
            elif valueF == valueR:
                return valueF
            else:
                return [valueF, valueR]

    return 0
                
#######################################################
#######################################################
##### MAIN
#######################################################
#######################################################

#imgSRC  = ['./DB/0_96G.jpg']
#imgSRC += ['./DB/0_75G.jpg']
#imgSRC += ['./DB/0_14.jpg' ]
#imgSRC = '0_3.jpg'

ResistorsInImage = FindInImage('./TestDB/' + sys.argv[1] +'.jpg')   
##        2, 16, 
print ResistorsInImage

imgSRC = ResistorsInImage[:]

for i in range(len(imgSRC)):
    print 'Processing image: ' + imgSRC[i]
    imgOriginal = cv2.imread(imgSRC[i])
    imgOrigShape = imgOriginal.shape

    IsResistor = CheckResistor(imgOriginal)

    if IsResistor:
        ValueColor =  FindStripesColor(imgOriginal,False,False)
        print ValueColor
        CanBeRead = 1
        for i in range(len(ValueColor)):
            if ValueColor[i] == 'unidentified':
                CanBeRead = 0
                break

        if CanBeRead != 0:     
            print resval(ValueColor,False)

    raw_input("Press Enter to continue...")


