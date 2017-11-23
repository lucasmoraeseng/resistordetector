# Program to detect the resistor color band
import numpy as np
import cv2

def ColorRanges(HSV_Color):
    colorMap = {
    	0  : 'black'       ,
    	1  : 'brown'       ,
        2  : 'red'         ,
        3  : 'orange'      ,
        4  : 'yellow'      ,
        5  : 'green'       ,
        6  : 'blue'        ,
        7  : 'violet'      ,
        8  : 'grey'        ,
        9  : 'white'       ,
        10 : 'red'         ,
        12 : 'unidentified'
    }

    # Bounderies definition
    # Black, brown, red, orange, yellow, green, blue, violet, grey, gold, silver
    boundaries = [
    [  0, 00, 00], [179,255, 65], #black
    [  8, 70, 51], [ 16,255,180], #brown
    [125,178,100], [179,255,255], #red
    [  6,160,135], [ 12,255,255], #orange
    [ 24,127,178], [ 30,255,255], #yellow
    [ 40, 30, 70], [ 74,255,255], #green
    [ 91,127,127], [121,255,255], #blue
    [129, 50,127], [170, 90,255], #violet
    [  0,  0, 76], [179, 25,200], #grey
    [  0,  0,204], [179, 25,255], #white  
    [  0,140,100], [  8,255,255]  #red  
    ]

    # Default condition
    color = 12
    for i in range(0,len(boundaries),2):	
        LowerColor = np.array(boundaries[i],dtype='uint8')
        HigherColor = np.array(boundaries[i+1],dtype='uint8')
        Mask = cv2.inRange(HSV_Color,LowerColor,HigherColor)
        #print 'Mask: ' + str(Mask) + ' MaskShape: ' + str(Mask.shape)    
        if Mask[0][0] == 255:
            color = i//2  

    return colorMap[color]

def GetColorFromBGR(BGR_Color):
    ToConvert = BGR_Color.reshape(1,1,3)
    hsv_conv = cv2.cvtColor(ToConvert,cv2.COLOR_BGR2HSV)
    print 'Color: '+ str(hsv_conv)
    Color = ColorRanges(hsv_conv)
    return Color

def GetRangesColors(BGR_array):
    ColorRanges = []
    for i in range(len(BGR_array)):
        ColorRanges += [GetColorFromBGR(np.array(BGR_array[i],dtype='uint8'))]
    return ColorRanges

