#!/usr/bin/env python
import rospy
import roslib
roslib.load_manifest('thesispro')
import numpy as np
import cv2 as cv
import sys
from thesispro.msg import screen
from thesispro.msg import localizaton
from thesispro.msg import localizationSymbols
from thesispro.msg import localizationProgramOptions
from sensor_msgs.msg import Image
from enum import Enum
from cv_bridge import CvBridge, CvBridgeError

def checkStatus(croppedSymbol):
    global status
    im2, contours, hierarchy = cv.findContours(croppedSymbol, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]
    try:
         perimeter = cv.arcLength(contours[0],True)
         if perimeter>10:
                status = True
    except Exception:
         status = False
    return status

def SymbolRecognitionFunc(image,rSymbol):#this takes big images
    croppedSymbol = image[rSymbol[1]:rSymbol[2],rSymbol[0]:rSymbol[3]]
    isSymbolOn = checkStatus(croppedSymbol)
    return isSymbolOn

class image_converter: #To get Image as message
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("imgConnection", Image, self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8") #take scene frame in grayscale
            #gray = cv.cvtColor(cv_image,cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(cv_image,(5,5),0)
            #Otsu's thresholding after Gaussian filtering
            ret3,resimg = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) #binarizing the frame

            #Program Recognition
            rProgram1 = [int(ProgramOptionCoordinates.prg1left[0]),int(ProgramOptionCoordinates.prg1left[1]),int(ProgramOptionCoordinates.prg2left[1]),int(ProgramOptionCoordinates.prg1right[0])]
            rProgram2 = [int(ProgramOptionCoordinates.prg2left[0]),int(ProgramOptionCoordinates.prg2left[1]),int(ProgramOptionCoordinates.prg3left[1]),int(ProgramOptionCoordinates.prg2right[0])]
            rProgram3 = [int(ProgramOptionCoordinates.prg3left[0]),int(ProgramOptionCoordinates.prg3left[1]),int(ProgramOptionCoordinates.prg4left[1]),int(ProgramOptionCoordinates.prg3right[0])]
            rProgram4 = [int(ProgramOptionCoordinates.prg4left[0]),int(ProgramOptionCoordinates.prg4left[1]),int(ProgramOptionCoordinates.prg5left[1]),int(ProgramOptionCoordinates.prg4right[0])]
            rProgram5 = [int(ProgramOptionCoordinates.prg5left[0]),int(ProgramOptionCoordinates.prg5left[1]),int(ProgramOptionCoordinates.prg6left[1]),int(ProgramOptionCoordinates.prg5right[0])]
            rProgram6 = [int(ProgramOptionCoordinates.prg6left[0]),int(ProgramOptionCoordinates.prg6left[1]),int(ProgramOptionCoordinates.prg7left[1]),int(ProgramOptionCoordinates.prg6right[0])]
            rProgram7 = [int(ProgramOptionCoordinates.prg7left[0]),int(ProgramOptionCoordinates.prg7left[1]),int(ProgramOptionCoordinates.prg8left[1]),int(ProgramOptionCoordinates.prg7right[0])]
            rProgram8 = [int(ProgramOptionCoordinates.prg8left[0]),int(ProgramOptionCoordinates.prg8left[1]),int(ProgramOptionCoordinates.prg9left[1]),int(ProgramOptionCoordinates.prg8right[0])]
            rProgram9 = [int(ProgramOptionCoordinates.prg9left[0]),int(ProgramOptionCoordinates.prg9left[1]),int(ProgramOptionCoordinates.prg10left[1]),int(ProgramOptionCoordinates.prg9right[0])]
            rProgram10 = [int(ProgramOptionCoordinates.prg10left[0]),int(ProgramOptionCoordinates.prg10left[1]),int(ProgramOptionCoordinates.prg11left[1]),int(ProgramOptionCoordinates.prg10right[0])]
            imCropPrg1 = resimg[rProgram1[1]:rProgram1[2],rProgram1[0]:rProgram1[3]]
            imCropPrg2 = resimg[rProgram2[1]:rProgram2[2],rProgram2[0]:rProgram2[3]]
            imCropPrg3 = resimg[rProgram3[1]:rProgram3[2],rProgram3[0]:rProgram3[3]]
            imCropPrg4 = resimg[rProgram4[1]:rProgram4[2],rProgram4[0]:rProgram4[3]]
            imCropPrg5 = resimg[rProgram5[1]:rProgram5[2],rProgram5[0]:rProgram5[3]]
            imCropPrg6 = resimg[rProgram6[1]:rProgram6[2],rProgram6[0]:rProgram6[3]]
            imCropPrg7 = resimg[rProgram7[1]:rProgram7[2],rProgram7[0]:rProgram7[3]]
            imCropPrg8 = resimg[rProgram8[1]:rProgram8[2],rProgram8[0]:rProgram8[3]]
            imCropPrg9 = resimg[rProgram9[1]:rProgram9[2],rProgram9[0]:rProgram9[3]]
            imCropPrg10 = resimg[rProgram10[1]:rProgram10[2],rProgram10[0]:rProgram10[3]]

            #checking whether program led is active or inactive
            if checkStatus(imCropPrg1) == True:
                msg.program = 1
            elif checkStatus(imCropPrg2) == True:
                msg.program = 2
            elif checkStatus(imCropPrg3) == True:
                msg.program = 3
            elif checkStatus(imCropPrg4) == True:
                msg.program = 4
            elif checkStatus(imCropPrg5) == True:
                msg.program = 5
            elif checkStatus(imCropPrg6) == True:
                msg.program = 6
            elif checkStatus(imCropPrg7) == True:
                msg.program = 7
            elif checkStatus(imCropPrg8) == True:
                msg.program = 8
            elif checkStatus(imCropPrg9) == True:
                msg.program = 9
            elif checkStatus(imCropPrg10) == True:
                msg.program = 10
            else:
                print("Error occured or the washing machine is not active!!!")

            #the same for wash option area
            #Option recognition
            rOption1 = [int(ProgramOptionCoordinates.option1left[0]),int(ProgramOptionCoordinates.option1left[1]),int(ProgramOptionCoordinates.option2left[1]),int(ProgramOptionCoordinates.option1right[0])]
            rOption2 = [int(ProgramOptionCoordinates.option2left[0]),int(ProgramOptionCoordinates.option2left[1]),int(ProgramOptionCoordinates.option3left[1]),int(ProgramOptionCoordinates.option2right[0])]
            rOption3 = [int(ProgramOptionCoordinates.option3left[0]),int(ProgramOptionCoordinates.option3left[1]),int(ProgramOptionCoordinates.option4left[1]),int(ProgramOptionCoordinates.option3right[0])]
            imCropOption1 = resimg[rOption1[1]:rOption1[2],rOption1[0]:rOption1[3]]
            imCropOption2 = resimg[rOption2[1]:rOption2[2],rOption2[0]:rOption2[3]]
            imCropOption3 = resimg[rOption3[1]:rOption3[2],rOption3[0]:rOption3[3]]
            if checkStatus(imCropOption1) == True:
                msg.option = 1
            elif checkStatus(imCropOption2) == True:
                msg.option = 2
            elif checkStatus(imCropOption3) == True:
                msg.option = 3
            else:
                print("Error occured or not any option chosen!!!")

            #Symbol On/Off status recognition
            rSymbol1 = [int(SymbolCoordinates.s1upleft[0]), int(SymbolCoordinates.s1upleft[1]), int(SymbolCoordinates.s1bottomleft[1]), int(SymbolCoordinates.s1upright[0])]
            rSymbol2 = [int(SymbolCoordinates.s2upleft[0]), int(SymbolCoordinates.s2upleft[1]), int(SymbolCoordinates.s2bottomleft[1]), int(SymbolCoordinates.s2upright[0])]
            rSymbol3 = [int(SymbolCoordinates.s3upleft[0]), int(SymbolCoordinates.s3upleft[1]), int(SymbolCoordinates.s3bottomleft[1]), int(SymbolCoordinates.s3upright[0])]
            rSymbol4 = [int(SymbolCoordinates.s4upleft[0]), int(SymbolCoordinates.s4upleft[1]), int(SymbolCoordinates.s4bottomleft[1]), int(SymbolCoordinates.s4upright[0])]
            rSymbol5 = [int(SymbolCoordinates.s5upleft[0]), int(SymbolCoordinates.s5upleft[1]), int(SymbolCoordinates.s5bottomleft[1]), int(SymbolCoordinates.s5upright[0])]
            rSymbol6 = [int(SymbolCoordinates.s6upleft[0]), int(SymbolCoordinates.s6upleft[1]), int(SymbolCoordinates.s6bottomleft[1]), int(SymbolCoordinates.s6upright[0])]
            rSymbol7 = [int(SymbolCoordinates.s7upleft[0]), int(SymbolCoordinates.s7upleft[1]), int(SymbolCoordinates.s7bottomleft[1]), int(SymbolCoordinates.s7upright[0])]
            rSymbol8 = [int(SymbolCoordinates.s8upleft[0]), int(SymbolCoordinates.s8upleft[1]), int(SymbolCoordinates.s8bottomleft[1]), int(SymbolCoordinates.s8upright[0])]
            rSymbol9 = [int(SymbolCoordinates.s9upleft[0]), int(SymbolCoordinates.s9upleft[1]), int(SymbolCoordinates.s9bottomleft[1]), int(SymbolCoordinates.s9upright[0])]
            rSymbol10 = [int(SymbolCoordinates.s10upleft[0]), int(SymbolCoordinates.s10upleft[1]), int(SymbolCoordinates.s10bottomleft[1]), int(SymbolCoordinates.s10upright[0])]
            rSymbol11 = [int(SymbolCoordinates.s11upleft[0]), int(SymbolCoordinates.s11upleft[1]), int(SymbolCoordinates.s11bottomleft[1]), int(SymbolCoordinates.s11upright[0])]
            #NormalTM2L7 or DailyTM2L7
            rSymbol12 = [int(SymbolCoordinates.s12upleft[0]), int(SymbolCoordinates.s12upleft[1]), int(SymbolCoordinates.s12bottomleft[1]), int(SymbolCoordinates.s12upright[0])]
            rSymbol13 = [int(SymbolCoordinates.s13upleft[0]), int(SymbolCoordinates.s13upleft[1]), int(SymbolCoordinates.s13bottomleft[1]), int(SymbolCoordinates.s13upright[0])]
            #rSymbol14 = [int(SymbolCoordinates.s14upleft[0]), int(SymbolCoordinates.s14upleft[1]), int(SymbolCoordinates.s14bottomleft[1]), int(SymbolCoordinates.s14upright[0])]
            #rSymbol15 = [int(SymbolCoordinates.s15upleft[0]), int(SymbolCoordinates.s15upleft[1]), int(SymbolCoordinates.s15bottomleft[1]), int(SymbolCoordinates.s15upright[0])]
            #rSymbol16 = [int(SymbolCoordinates.s16upleft[0]), int(SymbolCoordinates.s16upleft[1]), int(SymbolCoordinates.s16bottomleft[1]), int(SymbolCoordinates.s16upright[0])]
            #rSymbol17 = [int(SymbolCoordinates.s17upleft[0]), int(SymbolCoordinates.s17upleft[1]), int(SymbolCoordinates.s17bottomleft[1]), int(SymbolCoordinates.s17upright[0])]
            #rSymbol18 = [int(SymbolCoordinates.s18upleft[0]), int(SymbolCoordinates.s18upleft[1]), int(SymbolCoordinates.s18bottomleft[1]), int(SymbolCoordinates.s18upright[0])]
            #rSymbol19 = [int(SymbolCoordinates.s19upleft[0]), int(SymbolCoordinates.s19upleft[1]), int(SymbolCoordinates.s19bottomleft[1]), int(SymbolCoordinates.s19upright[0])]
            #rSymbol20 = [int(SymbolCoordinates.s20upleft[0]), int(SymbolCoordinates.s20upleft[1]), int(SymbolCoordinates.s20bottomleft[1]), int(SymbolCoordinates.s20upright[0])]
            #rSymbol21 = [int(SymbolCoordinates.s21upleft[0]), int(SymbolCoordinates.s21upleft[1]), int(SymbolCoordinates.s21bottomleft[1]), int(SymbolCoordinates.s21upright[0])]
            #rSymbol22 = [int(SymbolCoordinates.s22upleft[0]), int(SymbolCoordinates.s22upleft[1]), int(SymbolCoordinates.s22bottomleft[1]), int(SymbolCoordinates.s22upright[0])]
            isS1_OnOff = SymbolRecognitionFunc(resimg,rSymbol1)
            isS2_OnOff = SymbolRecognitionFunc(resimg,rSymbol2)
            isS3_OnOff = SymbolRecognitionFunc(resimg,rSymbol3)
            isS4_OnOff = SymbolRecognitionFunc(resimg,rSymbol4)
            isS5_OnOff = SymbolRecognitionFunc(resimg,rSymbol5)
            isS6_OnOff = SymbolRecognitionFunc(resimg,rSymbol6)
            isS7_OnOff = SymbolRecognitionFunc(resimg,rSymbol7)
            isS8_OnOff = SymbolRecognitionFunc(resimg,rSymbol8)
            isS9_OnOff = SymbolRecognitionFunc(resimg,rSymbol9)
            isS10_OnOff = SymbolRecognitionFunc(resimg,rSymbol10)
            isS11_OnOff = SymbolRecognitionFunc(resimg,rSymbol11)
            isS12_OnOff = SymbolRecognitionFunc(resimg,rSymbol12)
            isS13_OnOff = SymbolRecognitionFunc(resimg,rSymbol13)
            #isS14_OnOff = SymbolRecognitionFunc(resimg,rSymbol14)
            #isS15_OnOff = SymbolRecognitionFunc(resimg,rSymbol15)
            #isS16_OnOff = SymbolRecognitionFunc(resimg,rSymbol16)
            #isS17_OnOff = SymbolRecognitionFunc(resimg,rSymbol17)
            #isS18_OnOff = SymbolRecognitionFunc(resimg,rSymbol18)
            #isS19_OnOff = SymbolRecognitionFunc(resimg,rSymbol19)
            #isS20_OnOff = SymbolRecognitionFunc(resimg,rSymbol20)
            #isS21_OnOff = SymbolRecognitionFunc(resimg,rSymbol21)
            #isS22_OnOff = SymbolRecognitionFunc(resimg,rSymbol22)
            if isS12_OnOff == False:
                msg.NormalTM2L7 = True
                msg.DailyTM2L7 = False
                msg.SuperQuickTM2L7 = False
            else:
                msg.NormalTM2L7 = False
                msg.DailyTM2L7 = True
            #msg.SuperQuickTM2L7 = isS3_OnOff
            msg.Economy = isS11_OnOff
            msg.Prewash = isS9_OnOff
            msg.Stain = isS10_OnOff
            msg.ExtraRinse = isS7_OnOff
            #msg.Soft = isS8_OnOff
            msg.Night_Cycle = isS8_OnOff
            msg.Rinse_Hold = isS7_OnOff
            msg.Steam_Anticrease = isS6_OnOff
            msg.Low_Steam = False
            msg.Medium_Steam = False
            #msg.Steam_Max = isS14_OnOff
            msg.Prewashing = isS9_OnOff
            msg.Washing = isS1_OnOff
            msg.Rinsing = isS2_OnOff
            msg.Spinning = isS3_OnOff
            #msg.Drying = isS19_OnOff
            msg.Delay = True
            msg.Steaming = isS5_OnOff
            #msg.Anticrease =

            pub.publish(msg)
            cv.waitKey(1)
            if cv.waitKey(1) & 0xFF == ord('q'):
               cv.destroyAllWindows()
        except CVBridgeError as e:
            print(e)


def callback_symbol_coordinates(data):
    SymbolCoordinates.s1upleft = data.s1upleft
    SymbolCoordinates.s1upright = data.s1upright
    SymbolCoordinates.s1bottomleft = data.s1bottomleft
    SymbolCoordinates.s1bottomright = data.s1bottomright
    #
    SymbolCoordinates.s2upleft = data.s2upleft
    SymbolCoordinates.s2upright = data.s2upright
    SymbolCoordinates.s2bottomleft = data.s2bottomleft
    SymbolCoordinates.s2bottomright = data.s2bottomright
    #
    SymbolCoordinates.s3upleft = data.s3upleft
    SymbolCoordinates.s3upright = data.s3upright
    SymbolCoordinates.s3bottomleft = data.s3bottomleft
    SymbolCoordinates.s3bottomright = data.s3bottomright
    #
    SymbolCoordinates.s4upleft = data.s4upleft
    SymbolCoordinates.s4upright = data.s4upright
    SymbolCoordinates.s4bottomleft = data.s4bottomleft
    SymbolCoordinates.s4bottomright = data.s4bottomright
    #
    SymbolCoordinates.s5upleft = data.s5upleft
    SymbolCoordinates.s5upright = data.s5upright
    SymbolCoordinates.s5bottomleft = data.s5bottomleft
    SymbolCoordinates.s5bottomright = data.s5bottomright
    #
    SymbolCoordinates.s6upleft = data.s6upleft
    SymbolCoordinates.s6upright = data.s6upright
    SymbolCoordinates.s6bottomleft = data.s6bottomleft
    SymbolCoordinates.s6bottomright = data.s6bottomright
    #
    SymbolCoordinates.s7upleft = data.s7upleft
    SymbolCoordinates.s7upright = data.s7upright
    SymbolCoordinates.s7bottomleft = data.s7bottomleft
    SymbolCoordinates.s7bottomright = data.s7bottomright
    #
    SymbolCoordinates.s8upleft = data.s8upleft
    SymbolCoordinates.s8upright = data.s8upright
    SymbolCoordinates.s8bottomleft = data.s8bottomleft
    SymbolCoordinates.s8bottomright = data.s8bottomright
    #
    SymbolCoordinates.s9upleft = data.s9upleft
    SymbolCoordinates.s9upright = data.s9upright
    SymbolCoordinates.s9bottomleft = data.s9bottomleft
    SymbolCoordinates.s9bottomright = data.s9bottomright
    #
    SymbolCoordinates.s10upleft = data.s10upleft
    SymbolCoordinates.s10upright = data.s10upright
    SymbolCoordinates.s10bottomleft = data.s10bottomleft
    SymbolCoordinates.s10bottomright = data.s10bottomright
    #
    SymbolCoordinates.s11upleft = data.s11upleft
    SymbolCoordinates.s11upright = data.s11upright
    SymbolCoordinates.s11bottomleft = data.s11bottomleft
    SymbolCoordinates.s11bottomright = data.s11bottomright
    #
    SymbolCoordinates.s12upleft = data.s12upleft
    SymbolCoordinates.s12upright = data.s12upright
    SymbolCoordinates.s12bottomleft = data.s12bottomleft
    SymbolCoordinates.s12bottomright = data.s12bottomright
    #
    SymbolCoordinates.s13upleft = data.s13upleft
    SymbolCoordinates.s13upright = data.s13upright
    SymbolCoordinates.s13bottomleft = data.s13bottomleft
    SymbolCoordinates.s13bottomright = data.s13bottomright

def callback_program_options(data):
    ProgramOptionCoordinates.prg1left = data.prg1left
    ProgramOptionCoordinates.prg1right = data.prg1right

    ProgramOptionCoordinates.prg2left = data.prg2left
    ProgramOptionCoordinates.prg2right = data.prg2right

    ProgramOptionCoordinates.prg3left = data.prg3left
    ProgramOptionCoordinates.prg3right = data.prg3right

    ProgramOptionCoordinates.prg4left = data.prg4left
    ProgramOptionCoordinates.prg4right = data.prg4right

    ProgramOptionCoordinates.prg5left = data.prg5left
    ProgramOptionCoordinates.prg5right = data.prg5right

    ProgramOptionCoordinates.prg6left = data.prg6left
    ProgramOptionCoordinates.prg6right = data.prg6right

    ProgramOptionCoordinates.prg7left = data.prg7left
    ProgramOptionCoordinates.prg7right = data.prg7right

    ProgramOptionCoordinates.prg8left = data.prg8left
    ProgramOptionCoordinates.prg8right = data.prg8right

    ProgramOptionCoordinates.prg9left = data.prg9left
    ProgramOptionCoordinates.prg9right = data.prg9right

    ProgramOptionCoordinates.prg10left = data.prg10left
    ProgramOptionCoordinates.prg10right = data.prg10right

    ProgramOptionCoordinates.prg11left = data.prg11left
    ProgramOptionCoordinates.prg11right = data.prg11right

    ProgramOptionCoordinates.option1left = data.option1left
    ProgramOptionCoordinates.option1right = data.option1right

    ProgramOptionCoordinates.option2left = data.option2left
    ProgramOptionCoordinates.option2right = data.option2right

    ProgramOptionCoordinates.option3left = data.option3left
    ProgramOptionCoordinates.option3right = data.option3right

    ProgramOptionCoordinates.option4left = data.option4left
    ProgramOptionCoordinates.option4right = data.option4right


if __name__ == "__init__":
    global SymbolCoordinates
    global ProgramOptionCoordinates

if __name__ == "__main__":
    pub = rospy.Publisher('symbol_program_option_info',screen,queue_size = 1)
    rospy.init_node('symbol_program_option_info', anonymous = True)
    msg = screen()

    #making reference variables for wash function and wash program and options.
    SymbolCoordinates = localizationSymbols()
    ProgramOptionCoordinates = localizationProgramOptions()
    #---------------------------------------------------------------------------
    #taking wash function coordinates
    symbolcoordinates = rospy.Subscriber("SymbolLocalizer",localizationSymbols,callback_symbol_coordinates)
    #taking wash program and option coordinates
    programoptioncoordinates = rospy.Subscriber("ProgramOptionLocalizer",localizationProgramOptions,callback_program_options)
    ic = image_converter() #for image receiving. thresholding and recognition process is handling here
    #---------------------------------------------------------------------------
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.release()
    cv.destroyAllWindows()
