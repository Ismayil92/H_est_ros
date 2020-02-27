#!/usr/bin/env python
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import rospy
import roslib
roslib.load_manifest('ui_interpretation')
from ui_interpretation.msg import screen
from ui_interpretation.msg import localizaton
from ui_interpretation.msg import localizationSymbols
from ui_interpretation.msg import localizationProgramOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import datetime
#import tensorflow as tf

#Reference coordinates on the model image
class coordinates:

    #------------------------DIGITS---------------------------------------------
    #hour
    hour1_coordinates = [1146,114,61,100]
    hour2_coordinates = [1215,114,61,100]
    hour3_coordinates = [1279,114,61,100]

    #kg
    kg1_coordinates = [713,123,760-713,75]
    kg2_coordinates = [760,123,760-713,75]

    #temperature
    temp1_coordinates = [740,338,784-740,408-338]
    temp2_coordinates = [784,338,784-740,408-338]

    #centrifugation
    centr1_coordinates = [859,338,887-859,70]
    centr2_coordinates = [887,338,44,70]
    centr3_coordinates = [887+44,338,44,70]
    centr4_coordinates = [887+2*44,338,44,70]

    #speed
    speed_coordinates = [1263,333,1310-1263,73]

    #----------------------SYMBOLS(Wash Functions)------------------------------

    #Symbol1 Washing
    washing_coordinates = [703,225,66,70]
    #Symbol2 Rinsing
    rinsing_coordinates = [771,225,54,70]
    #Symbol3 Spinning
    spinning_coordinates = [831,225,43,70]
    #Symbol4
    symbol4_coordinates = [884,225,42,70]
    #Symbol5 Steam
    steam_coordinates = [937,225,55,70]
    #Symbol6 Anticrease
    anticrease_coordinates = [1002,225,58,70]
    #Symbol7 Rinse_Hold
    rinsehold_coordinates = [871,290,63,53]
    #Symbol8 Night_Cycle
    nightcycle_coordinates = [934,288,60,51]
    #Symbol9 Prewash
    prewash_coordinates = [1033,340,78,67]
    #Symbol10 Stain
    stain_coordinates = [1109,339,72,68]
    #3 different symbols on the Speed symbol
    #Symbol 11 Eco
    eco_coordinates = [1253,214,80,40]
    #Symbol 12 time symbol Daily and SuperQuick TM under ECO symbol
    dailysuperquick_coordinates = [1200,258,66,70]
    #Symbol 13 saat isaresinin solunda qalan isare
    symbol13_coordinates = [1265,254,70,74]

    #----------------------ProgramOptionCoordinates----------------------------------------------
    #Wash Program coordinates on model image
    algod_coordinates = [286,111,22,44]
    sinteticos_coordinates = [286,157,22,44]
    delicados_coordinates = [286,202,22,44]
    lana_coordinates = [286,247,22,44]
    vapor_coordinates = [286,296,22,44]
    okopower_coordinates = [286,343,22,44]
    antialergia_coordinates = [286,385,22,44]
    twentymin_coordinates = [286,434,22,44]
    outdoor_coordinates = [286,479,22,44]
    jeans_coordinates = [286,526,22,44]
    #Wash Option Coordinates on model image
    aclarado_coordinates = [1412,326,26,44]
    centrif_coordinates = [1412,370,26,44]
    drenar_coordinates = [1412,416,26,44]

if __name__ == "__main__":
    #--ROS Configuration----------------
    digit_coord_pub = rospy.Publisher("DigitLocalizer",localizaton,queue_size=10) #12 symbol publisher
    symbol_coord_pub = rospy.Publisher("SymbolLocalizer",localizationSymbols,queue_size=10) #wash function publisher
    program_option_coord_pub = rospy.Publisher("ProgramOptionLocalizer",localizationProgramOptions, queue_size=10) #wash program and option publisher
    image_pub = rospy.Publisher("imgConnection",Image,queue_size=1) #Scene frame publisher
    bridge = CvBridge()
    rospy.init_node('stream_and_coordinates_info',anonymous = True)
    DigitCoordinates = localizaton()
    SymbolCoordinates = localizationSymbols()
    ProgramOptionCoordinates = localizationProgramOptions()
    #-----------------------------------
    dc = coordinates()
    #model image
    img_object = cv.imread('/home/ismayil/catkin_ws/src/thesispro/resources/model_images/image3.jpg', cv.IMREAD_COLOR)
    img_object = cv.cvtColor(img_object,cv.COLOR_BGR2GRAY)
    if img_object is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    #Create Descriptor matcher
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    cap = cv.VideoCapture(1)
    #cap.set(3,1920)
    #cap.set(4,1080)
    if(cap.isOpened() == False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, img_scene = cap.read()
        if ret==True:
            img_scene=cv.cvtColor(img_scene,cv.COLOR_BGR2GRAY)
            try:
                image_pub.publish(bridge.cv2_to_imgmsg(img_scene, "mono8"))
            except CvBridgeError as e:
                print(e)
            #img_scene = cv.GaussianBlur(img_scene,(5,5),0)
            #Otsu's thresholding after Gaussian filtering
            #ret3,img_scene = cv.threshold(img_scene,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
            #-- Step 2: Matching descriptor vectors with a FLANN based matcher
            # Since SURF is a floating-point descriptor NORM_L2 is used
            knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

            #-- Filter matches using the Lowe's ratio test to eliminate outliers
            ratio_thresh = 0.75
            good_matches = []
            for m,n in knn_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            #-- Draw matches
            img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
            #cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #-- Localize the object
            obj = np.empty((len(good_matches),2), dtype=np.float32)
            scene = np.empty((len(good_matches),2), dtype=np.float32)
            for i in range(len(good_matches)):
                #-- Get the keypoints from the good matches in both images
                obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
                obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
                scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
                scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

            #TO FIND HOMOGRAPHY
            H, maskMatrix =  cv.findHomography(obj, scene, cv.RANSAC)
            matches_mask = maskMatrix.ravel().tolist()


            #-------------------DIGIT CORNERS-------------------------#
            #img.shape[0] -> number of rows or height
            #img.shape[1] -> number of columns or width
            #img.shape[2] -> number of channels

            #first point coordinate
            #x->int(rect_crop[0])
            #y->int(rect_crop[1])
            #crop_img = img[y:y+h, x:x+w]

            #-- Get the corners from the image_1 ( the object to be "detected" )
            h, w = img_object.shape
            object_corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            scene_corners = cv.perspectiveTransform(object_corners, H)

            hour1_corners = np.empty((4,1,2), dtype=np.float32)
            hour1_corners[0,0,0] = float(dc.hour1_coordinates[0])    #left top corner -x
            hour1_corners[0,0,1] = float(dc.hour1_coordinates[1])    #left top corner -y
            hour1_corners[1,0,0] = float(dc.hour1_coordinates[0]) + float(dc.hour1_coordinates[2]) #right top corner -x
            hour1_corners[1,0,1] = float(dc.hour1_coordinates[1])    #right top corner -y
            hour1_corners[2,0,0] = float(dc.hour1_coordinates[0])+float(dc.hour1_coordinates[2]) #right bottom corner -x
            hour1_corners[2,0,1] = float(dc.hour1_coordinates[1])+float(dc.hour1_coordinates[3]) #right bottom corner -y
            hour1_corners[3,0,0] = float(dc.hour1_coordinates[0])               #left bottom corner -x
            hour1_corners[3,0,1] = float(dc.hour1_coordinates[1])+float(dc.hour1_coordinates[3]) #left bottom corner -y

            hour2_corners = np.empty((4,1,2), dtype=np.float32)
            hour2_corners[0,0,0] = float(dc.hour2_coordinates[0])    #left top corner -x
            hour2_corners[0,0,1] = float(dc.hour2_coordinates[1])    #left top corner -y
            hour2_corners[1,0,0] = float(dc.hour2_coordinates[0]) + float(dc.hour2_coordinates[2]) #right top corner -x
            hour2_corners[1,0,1] = float(dc.hour2_coordinates[1])    #right top corner -y
            hour2_corners[2,0,0] = float(dc.hour2_coordinates[0])+float(dc.hour2_coordinates[2]) #right bottom corner -x
            hour2_corners[2,0,1] = float(dc.hour2_coordinates[1])+float(dc.hour2_coordinates[3]) #right bottom corner -y
            hour2_corners[3,0,0] = float(dc.hour2_coordinates[0])               #left bottom corner -x
            hour2_corners[3,0,1] = float(dc.hour2_coordinates[1])+float(dc.hour2_coordinates[3]) #left bottom corner -y


            hour3_corners = np.empty((4,1,2), dtype=np.float32)
            hour3_corners[0,0,0] = float(dc.hour3_coordinates[0])    #left top corner -x
            hour3_corners[0,0,1] = float(dc.hour3_coordinates[1])    #left top corner -y
            hour3_corners[1,0,0] = float(dc.hour3_coordinates[0]) + float(dc.hour3_coordinates[2]) #right top corner -x
            hour3_corners[1,0,1] = float(dc.hour3_coordinates[1])    #right top corner -y
            hour3_corners[2,0,0] = float(dc.hour3_coordinates[0])+float(dc.hour3_coordinates[2]) #right bottom corner -x
            hour3_corners[2,0,1] = float(dc.hour3_coordinates[1])+float(dc.hour3_coordinates[3]) #right bottom corner -y
            hour3_corners[3,0,0] = float(dc.hour3_coordinates[0])               #left bottom corner -x
            hour3_corners[3,0,1] = float(dc.hour3_coordinates[1])+float(dc.hour3_coordinates[3]) #left bottom corner -y


            kg1_corners = np.empty((4,1,2), dtype=np.float32)
            kg1_corners[0,0,0] = float(dc.kg1_coordinates[0])    #left top corner -x
            kg1_corners[0,0,1] = float(dc.kg1_coordinates[1])    #left top corner -y
            kg1_corners[1,0,0] = float(dc.kg1_coordinates[0]) + float(dc.kg1_coordinates[2]) #right top corner -x
            kg1_corners[1,0,1] = float(dc.kg1_coordinates[1])    #right top corner -y
            kg1_corners[2,0,0] = float(dc.kg1_coordinates[0])+float(dc.kg1_coordinates[2]) #right bottom corner -x
            kg1_corners[2,0,1] = float(dc.kg1_coordinates[1])+float(dc.kg1_coordinates[3]) #right bottom corner -y
            kg1_corners[3,0,0] = float(dc.kg1_coordinates[0])               #left bottom corner -x
            kg1_corners[3,0,1] = float(dc.kg1_coordinates[1])+float(dc.kg1_coordinates[3]) #left bottom corner -y

            kg2_corners = np.empty((4,1,2), dtype=np.float32)
            kg2_corners[0,0,0] = float(dc.kg2_coordinates[0])    #left top corner -x
            kg2_corners[0,0,1] = float(dc.kg2_coordinates[1])    #left top corner -y
            kg2_corners[1,0,0] = float(dc.kg2_coordinates[0]) + float(dc.kg2_coordinates[2]) #right top corner -x
            kg2_corners[1,0,1] = float(dc.kg2_coordinates[1])    #right top corner -y
            kg2_corners[2,0,0] = float(dc.kg2_coordinates[0])+float(dc.kg2_coordinates[2]) #right bottom corner -x
            kg2_corners[2,0,1] = float(dc.kg2_coordinates[1])+float(dc.kg2_coordinates[3]) #right bottom corner -y
            kg2_corners[3,0,0] = float(dc.kg2_coordinates[0])               #left bottom corner -x
            kg2_corners[3,0,1] = float(dc.kg2_coordinates[1])+float(dc.kg2_coordinates[3]) #left bottom corner -y

            temp1_corners = np.empty((4,1,2), dtype=np.float32)
            temp1_corners[0,0,0] = float(dc.temp1_coordinates[0])    #left top corner -x
            temp1_corners[0,0,1] = float(dc.temp1_coordinates[1])    #left top corner -y
            temp1_corners[1,0,0] = float(dc.temp1_coordinates[0]) + float(dc.temp1_coordinates[2]) #right top corner -x
            temp1_corners[1,0,1] = float(dc.temp1_coordinates[1])    #right top corner -y
            temp1_corners[2,0,0] = float(dc.temp1_coordinates[0])+float(dc.temp1_coordinates[2]) #right bottom corner -x
            temp1_corners[2,0,1] = float(dc.temp1_coordinates[1])+float(dc.temp1_coordinates[3]) #right bottom corner -y
            temp1_corners[3,0,0] = float(dc.temp1_coordinates[0])               #left bottom corner -x
            temp1_corners[3,0,1] = float(dc.temp1_coordinates[1])+float(dc.temp1_coordinates[3]) #left bottom corner -y

            temp2_corners = np.empty((4,1,2), dtype=np.float32)
            temp2_corners[0,0,0] = float(dc.temp2_coordinates[0])    #left top corner -x
            temp2_corners[0,0,1] = float(dc.temp2_coordinates[1])    #left top corner -y
            temp2_corners[1,0,0] = float(dc.temp2_coordinates[0]) + float(dc.temp2_coordinates[2]) #right top corner -x
            temp2_corners[1,0,1] = float(dc.temp2_coordinates[1])    #right top corner -y
            temp2_corners[2,0,0] = float(dc.temp2_coordinates[0])+float(dc.temp2_coordinates[2]) #right bottom corner -x
            temp2_corners[2,0,1] = float(dc.temp2_coordinates[1])+float(dc.temp2_coordinates[3]) #right bottom corner -y
            temp2_corners[3,0,0] = float(dc.temp2_coordinates[0])               #left bottom corner -x
            temp2_corners[3,0,1] = float(dc.temp2_coordinates[1])+float(dc.temp2_coordinates[3]) #left bottom corner -y

            centr1_corners = np.empty((4,1,2), dtype=np.float32)
            centr1_corners[0,0,0] = float(dc.centr1_coordinates[0])    #left top corner -x
            centr1_corners[0,0,1] = float(dc.centr1_coordinates[1])    #left top corner -y
            centr1_corners[1,0,0] = float(dc.centr1_coordinates[0]) + float(dc.centr1_coordinates[2]) #right top corner -x
            centr1_corners[1,0,1] = float(dc.centr1_coordinates[1])    #right top corner -y
            centr1_corners[2,0,0] = float(dc.centr1_coordinates[0])+float(dc.centr1_coordinates[2]) #right bottom corner -x
            centr1_corners[2,0,1] = float(dc.centr1_coordinates[1])+float(dc.centr1_coordinates[3]) #right bottom corner -y
            centr1_corners[3,0,0] = float(dc.centr1_coordinates[0])               #left bottom corner -x
            centr1_corners[3,0,1] = float(dc.centr1_coordinates[1])+float(dc.centr1_coordinates[3]) #left bottom corner -y

            centr2_corners = np.empty((4,1,2), dtype=np.float32)
            centr2_corners[0,0,0] = float(dc.centr2_coordinates[0])    #left top corner -x
            centr2_corners[0,0,1] = float(dc.centr2_coordinates[1])    #left top corner -y
            centr2_corners[1,0,0] = float(dc.centr2_coordinates[0]) + float(dc.centr2_coordinates[2]) #right top corner -x
            centr2_corners[1,0,1] = float(dc.centr2_coordinates[1])    #right top corner -y
            centr2_corners[2,0,0] = float(dc.centr2_coordinates[0])+float(dc.centr2_coordinates[2]) #right bottom corner -x
            centr2_corners[2,0,1] = float(dc.centr2_coordinates[1])+float(dc.centr2_coordinates[3]) #right bottom corner -y
            centr2_corners[3,0,0] = float(dc.centr2_coordinates[0])               #left bottom corner -x
            centr2_corners[3,0,1] = float(dc.centr2_coordinates[1])+float(dc.centr2_coordinates[3]) #left bottom corner -y

            centr3_corners = np.empty((4,1,2), dtype=np.float32)
            centr3_corners[0,0,0] = float(dc.centr3_coordinates[0])    #left top corner -x
            centr3_corners[0,0,1] = float(dc.centr3_coordinates[1])    #left top corner -y
            centr3_corners[1,0,0] = float(dc.centr3_coordinates[0]) + float(dc.centr3_coordinates[2]) #right top corner -x
            centr3_corners[1,0,1] = float(dc.centr3_coordinates[1])    #right top corner -y
            centr3_corners[2,0,0] = float(dc.centr3_coordinates[0])+float(dc.centr3_coordinates[2]) #right bottom corner -x
            centr3_corners[2,0,1] = float(dc.centr3_coordinates[1])+float(dc.centr3_coordinates[3]) #right bottom corner -y
            centr3_corners[3,0,0] = float(dc.centr3_coordinates[0])               #left bottom corner -x
            centr3_corners[3,0,1] = float(dc.centr3_coordinates[1])+float(dc.centr3_coordinates[3]) #left bottom corner -y

            centr4_corners = np.empty((4,1,2), dtype=np.float32)
            centr4_corners[0,0,0] = float(dc.centr4_coordinates[0])    #left top corner -x
            centr4_corners[0,0,1] = float(dc.centr4_coordinates[1])    #left top corner -y
            centr4_corners[1,0,0] = float(dc.centr4_coordinates[0]) + float(dc.centr4_coordinates[2]) #right top corner -x
            centr4_corners[1,0,1] = float(dc.centr4_coordinates[1])    #right top corner -y
            centr4_corners[2,0,0] = float(dc.centr4_coordinates[0])+float(dc.centr4_coordinates[2]) #right bottom corner -x
            centr4_corners[2,0,1] = float(dc.centr4_coordinates[1])+float(dc.centr4_coordinates[3]) #right bottom corner -y
            centr4_corners[3,0,0] = float(dc.centr4_coordinates[0])               #left bottom corner -x
            centr4_corners[3,0,1] = float(dc.centr4_coordinates[1])+float(dc.centr4_coordinates[3]) #left bottom corner -y

            speed_corners = np.empty((4,1,2), dtype=np.float32)
            speed_corners[0,0,0] = float(dc.speed_coordinates[0])    #left top corner -x
            speed_corners[0,0,1] = float(dc.speed_coordinates[1])    #left top corner -y
            speed_corners[1,0,0] = float(dc.speed_coordinates[0])+float(dc.speed_coordinates[2]) #right top corner -x
            speed_corners[1,0,1] = float(dc.speed_coordinates[1])    #right top corner -y
            speed_corners[2,0,0] = float(dc.speed_coordinates[0])+float(dc.speed_coordinates[2]) #right bottom corner -x
            speed_corners[2,0,1] = float(dc.speed_coordinates[1])+float(dc.speed_coordinates[3]) #right bottom corner -y
            speed_corners[3,0,0] = float(dc.speed_coordinates[0]) #left bottom corner -x
            speed_corners[3,0,1] = float(dc.speed_coordinates[1])+float(dc.speed_coordinates[3]) #left bottom corner -y


            #-------------------Symbol (Wash Function) image crops----------------------------------------------
            #Wash Function region coordinates on the user interface
            washing_corners = np.empty((4,1,2), dtype=np.float32)
            washing_corners[0,0,0] = float(dc.washing_coordinates[0])    #left top corner -x
            washing_corners[0,0,1] = float(dc.washing_coordinates[1])    #left top corner -y
            washing_corners[1,0,0] = float(dc.washing_coordinates[0])+float(dc.washing_coordinates[2]) #right top corner -x
            washing_corners[1,0,1] = float(dc.washing_coordinates[1])    #right top corner -y
            washing_corners[2,0,0] = float(dc.washing_coordinates[0])+float(dc.washing_coordinates[2]) #right bottom corner -x
            washing_corners[2,0,1] = float(dc.washing_coordinates[1])+float(dc.washing_coordinates[3]) #right bottom corner -y
            washing_corners[3,0,0] = float(dc.washing_coordinates[0]) #left bottom corner -x
            washing_corners[3,0,1] = float(dc.washing_coordinates[1])+float(dc.washing_coordinates[3]) #left bottom corner -y

            rinsing_corners = np.empty((4,1,2), dtype=np.float32)
            rinsing_corners[0,0,0] = float(dc.rinsing_coordinates[0])    #left top corner -x
            rinsing_corners[0,0,1] = float(dc.rinsing_coordinates[1])    #left top corner -y
            rinsing_corners[1,0,0] = float(dc.rinsing_coordinates[0])+float(dc.rinsing_coordinates[2]) #right top corner -x
            rinsing_corners[1,0,1] = float(dc.rinsing_coordinates[1])    #right top corner -y
            rinsing_corners[2,0,0] = float(dc.rinsing_coordinates[0])+float(dc.rinsing_coordinates[2]) #right bottom corner -x
            rinsing_corners[2,0,1] = float(dc.rinsing_coordinates[1])+float(dc.rinsing_coordinates[3]) #right bottom corner -y
            rinsing_corners[3,0,0] = float(dc.rinsing_coordinates[0]) #left bottom corner -x
            rinsing_corners[3,0,1] = float(dc.rinsing_coordinates[1])+float(dc.rinsing_coordinates[3]) #left bottom corner -y

            spinning_corners = np.empty((4,1,2), dtype=np.float32)
            spinning_corners[0,0,0] = float(dc.spinning_coordinates[0])    #left top corner -x
            spinning_corners[0,0,1] = float(dc.spinning_coordinates[1])    #left top corner -y
            spinning_corners[1,0,0] = float(dc.spinning_coordinates[0])+float(dc.spinning_coordinates[2]) #right top corner -x
            spinning_corners[1,0,1] = float(dc.spinning_coordinates[1])    #right top corner -y
            spinning_corners[2,0,0] = float(dc.spinning_coordinates[0])+float(dc.spinning_coordinates[2]) #right bottom corner -x
            spinning_corners[2,0,1] = float(dc.spinning_coordinates[1])+float(dc.spinning_coordinates[3]) #right bottom corner -y
            spinning_corners[3,0,0] = float(dc.spinning_coordinates[0]) #left bottom corner -x
            spinning_corners[3,0,1] = float(dc.spinning_coordinates[1])+float(dc.spinning_coordinates[3]) #left bottom corner -y

            symbol4_corners = np.empty((4,1,2), dtype=np.float32)
            symbol4_corners[0,0,0] = float(dc.symbol4_coordinates[0])    #left top corner -x
            symbol4_corners[0,0,1] = float(dc.symbol4_coordinates[1])    #left top corner -y
            symbol4_corners[1,0,0] = float(dc.symbol4_coordinates[0])+float(dc.symbol4_coordinates[2]) #right top corner -x
            symbol4_corners[1,0,1] = float(dc.symbol4_coordinates[1])    #right top corner -y
            symbol4_corners[2,0,0] = float(dc.symbol4_coordinates[0])+float(dc.symbol4_coordinates[2]) #right bottom corner -x
            symbol4_corners[2,0,1] = float(dc.symbol4_coordinates[1])+float(dc.symbol4_coordinates[3]) #right bottom corner -y
            symbol4_corners[3,0,0] = float(dc.symbol4_coordinates[0]) #left bottom corner -x
            symbol4_corners[3,0,1] = float(dc.symbol4_coordinates[1])+float(dc.symbol4_coordinates[3]) #left bottom corner -y

            steam_corners = np.empty((4,1,2), dtype=np.float32)
            steam_corners[0,0,0] = float(dc.steam_coordinates[0])    #left top corner -x
            steam_corners[0,0,1] = float(dc.steam_coordinates[1])    #left top corner -y
            steam_corners[1,0,0] = float(dc.steam_coordinates[0])+float(dc.steam_coordinates[2]) #right top corner -x
            steam_corners[1,0,1] = float(dc.steam_coordinates[1])    #right top corner -y
            steam_corners[2,0,0] = float(dc.steam_coordinates[0])+float(dc.steam_coordinates[2]) #right bottom corner -x
            steam_corners[2,0,1] = float(dc.steam_coordinates[1])+float(dc.steam_coordinates[3]) #right bottom corner -y
            steam_corners[3,0,0] = float(dc.steam_coordinates[0]) #left bottom corner -x
            steam_corners[3,0,1] = float(dc.steam_coordinates[1])+float(dc.steam_coordinates[3]) #left bottom corner -y

            anticrease_corners = np.empty((4,1,2), dtype=np.float32)
            anticrease_corners[0,0,0] = float(dc.anticrease_coordinates[0])    #left top corner -x
            anticrease_corners[0,0,1] = float(dc.anticrease_coordinates[1])    #left top corner -y
            anticrease_corners[1,0,0] = float(dc.anticrease_coordinates[0])+float(dc.anticrease_coordinates[2]) #right top corner -x
            anticrease_corners[1,0,1] = float(dc.anticrease_coordinates[1])    #right top corner -y
            anticrease_corners[2,0,0] = float(dc.anticrease_coordinates[0])+float(dc.anticrease_coordinates[2]) #right bottom corner -x
            anticrease_corners[2,0,1] = float(dc.anticrease_coordinates[1])+float(dc.anticrease_coordinates[3]) #right bottom corner -y
            anticrease_corners[3,0,0] = float(dc.anticrease_coordinates[0]) #left bottom corner -x
            anticrease_corners[3,0,1] = float(dc.anticrease_coordinates[1])+float(dc.anticrease_coordinates[3]) #left bottom corner -y

            rinsehold_corners = np.empty((4,1,2), dtype=np.float32)
            rinsehold_corners[0,0,0] = float(dc.rinsehold_coordinates[0])    #left top corner -x
            rinsehold_corners[0,0,1] = float(dc.rinsehold_coordinates[1])    #left top corner -y
            rinsehold_corners[1,0,0] = float(dc.rinsehold_coordinates[0])+float(dc.rinsehold_coordinates[2]) #right top corner -x
            rinsehold_corners[1,0,1] = float(dc.rinsehold_coordinates[1])    #right top corner -y
            rinsehold_corners[2,0,0] = float(dc.rinsehold_coordinates[0])+float(dc.rinsehold_coordinates[2]) #right bottom corner -x
            rinsehold_corners[2,0,1] = float(dc.rinsehold_coordinates[1])+float(dc.rinsehold_coordinates[3]) #right bottom corner -y
            rinsehold_corners[3,0,0] = float(dc.rinsehold_coordinates[0]) #left bottom corner -x
            rinsehold_corners[3,0,1] = float(dc.rinsehold_coordinates[1])+float(dc.rinsehold_coordinates[3]) #left bottom corner -y

            nightcycle_corners = np.empty((4,1,2), dtype=np.float32)
            nightcycle_corners[0,0,0] = float(dc.nightcycle_coordinates[0])    #left top corner -x
            nightcycle_corners[0,0,1] = float(dc.nightcycle_coordinates[1])    #left top corner -y
            nightcycle_corners[1,0,0] = float(dc.nightcycle_coordinates[0])+float(dc.nightcycle_coordinates[2]) #right top corner -x
            nightcycle_corners[1,0,1] = float(dc.nightcycle_coordinates[1])    #right top corner -y
            nightcycle_corners[2,0,0] = float(dc.nightcycle_coordinates[0])+float(dc.nightcycle_coordinates[2]) #right bottom corner -x
            nightcycle_corners[2,0,1] = float(dc.nightcycle_coordinates[1])+float(dc.nightcycle_coordinates[3]) #right bottom corner -y
            nightcycle_corners[3,0,0] = float(dc.nightcycle_coordinates[0]) #left bottom corner -x
            nightcycle_corners[3,0,1] = float(dc.nightcycle_coordinates[1])+float(dc.nightcycle_coordinates[3]) #left bottom corner -y

            prewash_corners = np.empty((4,1,2), dtype=np.float32)
            prewash_corners[0,0,0] = float(dc.prewash_coordinates[0])    #left top corner -x
            prewash_corners[0,0,1] = float(dc.prewash_coordinates[1])    #left top corner -y
            prewash_corners[1,0,0] = float(dc.prewash_coordinates[0])+float(dc.prewash_coordinates[2]) #right top corner -x
            prewash_corners[1,0,1] = float(dc.prewash_coordinates[1])    #right top corner -y
            prewash_corners[2,0,0] = float(dc.prewash_coordinates[0])+float(dc.prewash_coordinates[2]) #right bottom corner -x
            prewash_corners[2,0,1] = float(dc.prewash_coordinates[1])+float(dc.prewash_coordinates[3]) #right bottom corner -y
            prewash_corners[3,0,0] = float(dc.prewash_coordinates[0]) #left bottom corner -x
            prewash_corners[3,0,1] = float(dc.prewash_coordinates[1])+float(dc.prewash_coordinates[3]) #left bottom corner -y

            stain_corners = np.empty((4,1,2), dtype=np.float32)
            stain_corners[0,0,0] = float(dc.stain_coordinates[0])    #left top corner -x
            stain_corners[0,0,1] = float(dc.stain_coordinates[1])    #left top corner -y
            stain_corners[1,0,0] = float(dc.stain_coordinates[0])+float(dc.stain_coordinates[2]) #right top corner -x
            stain_corners[1,0,1] = float(dc.stain_coordinates[1])    #right top corner -y
            stain_corners[2,0,0] = float(dc.stain_coordinates[0])+float(dc.stain_coordinates[2]) #right bottom corner -x
            stain_corners[2,0,1] = float(dc.stain_coordinates[1])+float(dc.stain_coordinates[3]) #right bottom corner -y
            stain_corners[3,0,0] = float(dc.stain_coordinates[0]) #left bottom corner -x
            stain_corners[3,0,1] = float(dc.stain_coordinates[1])+float(dc.stain_coordinates[3]) #left bottom corner -y

            eco_corners = np.empty((4,1,2), dtype=np.float32)
            eco_corners[0,0,0] = float(dc.eco_coordinates[0])    #left top corner -x
            eco_corners[0,0,1] = float(dc.eco_coordinates[1])    #left top corner -y
            eco_corners[1,0,0] = float(dc.eco_coordinates[0])+float(dc.eco_coordinates[2]) #right top corner -x
            eco_corners[1,0,1] = float(dc.eco_coordinates[1])    #right top corner -y
            eco_corners[2,0,0] = float(dc.eco_coordinates[0])+float(dc.eco_coordinates[2]) #right bottom corner -x
            eco_corners[2,0,1] = float(dc.eco_coordinates[1])+float(dc.eco_coordinates[3]) #right bottom corner -y
            eco_corners[3,0,0] = float(dc.eco_coordinates[0]) #left bottom corner -x
            eco_corners[3,0,1] = float(dc.eco_coordinates[1])+float(dc.eco_coordinates[3]) #left bottom corner -y

            daily_corners = np.empty((4,1,2), dtype=np.float32)
            daily_corners[0,0,0] = float(dc.dailysuperquick_coordinates[0])    #left top corner -x
            daily_corners[0,0,1] = float(dc.dailysuperquick_coordinates[1])    #left top corner -y
            daily_corners[1,0,0] = float(dc.dailysuperquick_coordinates[0])+float(dc.dailysuperquick_coordinates[2]) #right top corner -x
            daily_corners[1,0,1] = float(dc.dailysuperquick_coordinates[1])    #right top corner -y
            daily_corners[2,0,0] = float(dc.dailysuperquick_coordinates[0])+float(dc.dailysuperquick_coordinates[2]) #right bottom corner -x
            daily_corners[2,0,1] = float(dc.dailysuperquick_coordinates[1])+float(dc.dailysuperquick_coordinates[3]) #right bottom corner -y
            daily_corners[3,0,0] = float(dc.dailysuperquick_coordinates[0]) #left bottom corner -x
            daily_corners[3,0,1] = float(dc.dailysuperquick_coordinates[1])+float(dc.dailysuperquick_coordinates[3]) #left bottom corner -y

            symbol13_corners = np.empty((4,1,2), dtype=np.float32)
            symbol13_corners[0,0,0] = float(dc.symbol13_coordinates[0])    #left top corner -x
            symbol13_corners[0,0,1] = float(dc.symbol13_coordinates[1])    #left top corner -y
            symbol13_corners[1,0,0] = float(dc.symbol13_coordinates[0])+float(dc.symbol13_coordinates[2]) #right top corner -x
            symbol13_corners[1,0,1] = float(dc.symbol13_coordinates[1])    #right top corner -y
            symbol13_corners[2,0,0] = float(dc.symbol13_coordinates[0])+float(dc.symbol13_coordinates[2]) #right bottom corner -x
            symbol13_corners[2,0,1] = float(dc.symbol13_coordinates[1])+float(dc.symbol13_coordinates[3]) #right bottom corner -y
            symbol13_corners[3,0,0] = float(dc.symbol13_coordinates[0]) #left bottom corner -x
            symbol13_corners[3,0,1] = float(dc.symbol13_coordinates[1])+float(dc.symbol13_coordinates[3]) #left bottom corner -y


            #--------------------ProgramOptionCoordinates-----------------------
            #Wash program region corner coordinates on the model image
            algod_corners = np.empty((4,1,2), dtype=np.float32)
            algod_corners[0,0,0] = float(dc.algod_coordinates[0])    #left top corner -x
            algod_corners[0,0,1] = float(dc.algod_coordinates[1])    #left top corner -y
            algod_corners[1,0,0] = float(dc.algod_coordinates[0])+float(dc.algod_coordinates[2]) #right top corner -x
            algod_corners[1,0,1] = float(dc.algod_coordinates[1])    #right top corner -y
            algod_corners[2,0,0] = float(dc.algod_coordinates[0])+float(dc.algod_coordinates[2]) #right bottom corner -x
            algod_corners[2,0,1] = float(dc.algod_coordinates[1])+float(dc.algod_coordinates[3]) #right bottom corner -y
            algod_corners[3,0,0] = float(dc.algod_coordinates[0]) #left bottom corner -x
            algod_corners[3,0,1] = float(dc.algod_coordinates[1])+float(dc.algod_coordinates[3]) #left bottom corner -y

            sinteticos_corners = np.empty((4,1,2), dtype=np.float32)
            sinteticos_corners[0,0,0] = float(dc.sinteticos_coordinates[0])    #left top corner -x
            sinteticos_corners[0,0,1] = float(dc.sinteticos_coordinates[1])    #left top corner -y
            sinteticos_corners[1,0,0] = float(dc.sinteticos_coordinates[0])+float(dc.sinteticos_coordinates[2]) #right top corner -x
            sinteticos_corners[1,0,1] = float(dc.sinteticos_coordinates[1])    #right top corner -y
            sinteticos_corners[2,0,0] = float(dc.sinteticos_coordinates[0])+float(dc.sinteticos_coordinates[2]) #right bottom corner -x
            sinteticos_corners[2,0,1] = float(dc.sinteticos_coordinates[1])+float(dc.sinteticos_coordinates[3]) #right bottom corner -y
            sinteticos_corners[3,0,0] = float(dc.sinteticos_coordinates[0]) #left bottom corner -x
            sinteticos_corners[3,0,1] = float(dc.sinteticos_coordinates[1])+float(dc.sinteticos_coordinates[3]) #left bottom corner -y

            delicados_corners = np.empty((4,1,2), dtype=np.float32)
            delicados_corners[0,0,0] = float(dc.delicados_coordinates[0])    #left top corner -x
            delicados_corners[0,0,1] = float(dc.delicados_coordinates[1])    #left top corner -y
            delicados_corners[1,0,0] = float(dc.delicados_coordinates[0])+float(dc.delicados_coordinates[2]) #right top corner -x
            delicados_corners[1,0,1] = float(dc.delicados_coordinates[1])    #right top corner -y
            delicados_corners[2,0,0] = float(dc.delicados_coordinates[0])+float(dc.delicados_coordinates[2]) #right bottom corner -x
            delicados_corners[2,0,1] = float(dc.delicados_coordinates[1])+float(dc.delicados_coordinates[3]) #right bottom corner -y
            delicados_corners[3,0,0] = float(dc.delicados_coordinates[0]) #left bottom corner -x
            delicados_corners[3,0,1] = float(dc.delicados_coordinates[1])+float(dc.delicados_coordinates[3]) #left bottom corner -y

            lana_corners = np.empty((4,1,2), dtype=np.float32)
            lana_corners[0,0,0] = float(dc.lana_coordinates[0])    #left top corner -x
            lana_corners[0,0,1] = float(dc.lana_coordinates[1])    #left top corner -y
            lana_corners[1,0,0] = float(dc.lana_coordinates[0])+float(dc.lana_coordinates[2]) #right top corner -x
            lana_corners[1,0,1] = float(dc.lana_coordinates[1])    #right top corner -y
            lana_corners[2,0,0] = float(dc.lana_coordinates[0])+float(dc.lana_coordinates[2]) #right bottom corner -x
            lana_corners[2,0,1] = float(dc.lana_coordinates[1])+float(dc.lana_coordinates[3]) #right bottom corner -y
            lana_corners[3,0,0] = float(dc.lana_coordinates[0]) #left bottom corner -x
            lana_corners[3,0,1] = float(dc.lana_coordinates[1])+float(dc.lana_coordinates[3]) #left bottom corner -y

            vapor_corners = np.empty((4,1,2), dtype=np.float32)
            vapor_corners[0,0,0] = float(dc.vapor_coordinates[0])    #left top corner -x
            vapor_corners[0,0,1] = float(dc.vapor_coordinates[1])    #left top corner -y
            vapor_corners[1,0,0] = float(dc.vapor_coordinates[0])+float(dc.vapor_coordinates[2]) #right top corner -x
            vapor_corners[1,0,1] = float(dc.vapor_coordinates[1])    #right top corner -y
            vapor_corners[2,0,0] = float(dc.vapor_coordinates[0])+float(dc.vapor_coordinates[2]) #right bottom corner -x
            vapor_corners[2,0,1] = float(dc.vapor_coordinates[1])+float(dc.vapor_coordinates[3]) #right bottom corner -y
            vapor_corners[3,0,0] = float(dc.vapor_coordinates[0]) #left bottom corner -x
            vapor_corners[3,0,1] = float(dc.vapor_coordinates[1])+float(dc.vapor_coordinates[3]) #left bottom corner -y

            oko_corners = np.empty((4,1,2), dtype=np.float32)
            oko_corners[0,0,0] = float(dc.okopower_coordinates[0])    #left top corner -x
            oko_corners[0,0,1] = float(dc.okopower_coordinates[1])    #left top corner -y
            oko_corners[1,0,0] = float(dc.okopower_coordinates[0])+float(dc.okopower_coordinates[2]) #right top corner -x
            oko_corners[1,0,1] = float(dc.okopower_coordinates[1])    #right top corner -y
            oko_corners[2,0,0] = float(dc.okopower_coordinates[0])+float(dc.okopower_coordinates[2]) #right bottom corner -x
            oko_corners[2,0,1] = float(dc.okopower_coordinates[1])+float(dc.okopower_coordinates[3]) #right bottom corner -y
            oko_corners[3,0,0] = float(dc.okopower_coordinates[0]) #left bottom corner -x
            oko_corners[3,0,1] = float(dc.okopower_coordinates[1])+float(dc.okopower_coordinates[3]) #left bottom corner -y

            antialergia_corners = np.empty((4,1,2), dtype=np.float32)
            antialergia_corners[0,0,0] = float(dc.antialergia_coordinates[0])    #left top corner -x
            antialergia_corners[0,0,1] = float(dc.antialergia_coordinates[1])    #left top corner -y
            antialergia_corners[1,0,0] = float(dc.antialergia_coordinates[0])+float(dc.antialergia_coordinates[2]) #right top corner -x
            antialergia_corners[1,0,1] = float(dc.antialergia_coordinates[1])    #right top corner -y
            antialergia_corners[2,0,0] = float(dc.antialergia_coordinates[0])+float(dc.antialergia_coordinates[2]) #right bottom corner -x
            antialergia_corners[2,0,1] = float(dc.antialergia_coordinates[1])+float(dc.antialergia_coordinates[3]) #right bottom corner -y
            antialergia_corners[3,0,0] = float(dc.antialergia_coordinates[0]) #left bottom corner -x
            antialergia_corners[3,0,1] = float(dc.antialergia_coordinates[1])+float(dc.antialergia_coordinates[3]) #left bottom corner -y

            twentymin_corners = np.empty((4,1,2), dtype=np.float32)
            twentymin_corners[0,0,0] = float(dc.twentymin_coordinates[0])    #left top corner -x
            twentymin_corners[0,0,1] = float(dc.twentymin_coordinates[1])    #left top corner -y
            twentymin_corners[1,0,0] = float(dc.twentymin_coordinates[0])+float(dc.twentymin_coordinates[2]) #right top corner -x
            twentymin_corners[1,0,1] = float(dc.twentymin_coordinates[1])    #right top corner -y
            twentymin_corners[2,0,0] = float(dc.twentymin_coordinates[0])+float(dc.twentymin_coordinates[2]) #right bottom corner -x
            twentymin_corners[2,0,1] = float(dc.twentymin_coordinates[1])+float(dc.twentymin_coordinates[3]) #right bottom corner -y
            twentymin_corners[3,0,0] = float(dc.twentymin_coordinates[0]) #left bottom corner -x
            twentymin_corners[3,0,1] = float(dc.twentymin_coordinates[1])+float(dc.twentymin_coordinates[3]) #left bottom corner -y

            outdoor_corners = np.empty((4,1,2), dtype=np.float32)
            outdoor_corners[0,0,0] = float(dc.outdoor_coordinates[0])    #left top corner -x
            outdoor_corners[0,0,1] = float(dc.outdoor_coordinates[1])    #left top corner -y
            outdoor_corners[1,0,0] = float(dc.outdoor_coordinates[0])+float(dc.outdoor_coordinates[2]) #right top corner -x
            outdoor_corners[1,0,1] = float(dc.outdoor_coordinates[1])    #right top corner -y
            outdoor_corners[2,0,0] = float(dc.outdoor_coordinates[0])+float(dc.outdoor_coordinates[2]) #right bottom corner -x
            outdoor_corners[2,0,1] = float(dc.outdoor_coordinates[1])+float(dc.outdoor_coordinates[3]) #right bottom corner -y
            outdoor_corners[3,0,0] = float(dc.outdoor_coordinates[0]) #left bottom corner -x
            outdoor_corners[3,0,1] = float(dc.outdoor_coordinates[1])+float(dc.outdoor_coordinates[3]) #left bottom corner -y

            jeans_corners = np.empty((4,1,2), dtype=np.float32)
            jeans_corners[0,0,0] = float(dc.jeans_coordinates[0])    #left top corner -x
            jeans_corners[0,0,1] = float(dc.jeans_coordinates[1])    #left top corner -y
            jeans_corners[1,0,0] = float(dc.jeans_coordinates[0])+float(dc.jeans_coordinates[2]) #right top corner -x
            jeans_corners[1,0,1] = float(dc.jeans_coordinates[1])    #right top corner -y
            jeans_corners[2,0,0] = float(dc.jeans_coordinates[0])+float(dc.jeans_coordinates[2]) #right bottom corner -x
            jeans_corners[2,0,1] = float(dc.jeans_coordinates[1])+float(dc.jeans_coordinates[3]) #right bottom corner -y
            jeans_corners[3,0,0] = float(dc.jeans_coordinates[0]) #left bottom corner -x
            jeans_corners[3,0,1] = float(dc.jeans_coordinates[1])+float(dc.jeans_coordinates[3]) #left bottom corner -y

            #Wash Option coordinates on the reference image
            aclarado_corners = np.empty((4,1,2), dtype=np.float32)
            aclarado_corners[0,0,0] = float(dc.aclarado_coordinates[0])    #left top corner -x
            aclarado_corners[0,0,1] = float(dc.aclarado_coordinates[1])    #left top corner -y
            aclarado_corners[1,0,0] = float(dc.aclarado_coordinates[0])+float(dc.aclarado_coordinates[2]) #right top corner -x
            aclarado_corners[1,0,1] = float(dc.aclarado_coordinates[1])    #right top corner -y
            aclarado_corners[2,0,0] = float(dc.aclarado_coordinates[0])+float(dc.aclarado_coordinates[2]) #right bottom corner -x
            aclarado_corners[2,0,1] = float(dc.aclarado_coordinates[1])+float(dc.aclarado_coordinates[3]) #right bottom corner -y
            aclarado_corners[3,0,0] = float(dc.aclarado_coordinates[0]) #left bottom corner -x
            aclarado_corners[3,0,1] = float(dc.aclarado_coordinates[1])+float(dc.aclarado_coordinates[3]) #left bottom corner -y

            centrif_corners = np.empty((4,1,2), dtype=np.float32)
            centrif_corners[0,0,0] = float(dc.centrif_coordinates[0])    #left top corner -x
            centrif_corners[0,0,1] = float(dc.centrif_coordinates[1])    #left top corner -y
            centrif_corners[1,0,0] = float(dc.centrif_coordinates[0])+float(dc.centrif_coordinates[2]) #right top corner -x
            centrif_corners[1,0,1] = float(dc.centrif_coordinates[1])    #right top corner -y
            centrif_corners[2,0,0] = float(dc.centrif_coordinates[0])+float(dc.centrif_coordinates[2]) #right bottom corner -x
            centrif_corners[2,0,1] = float(dc.centrif_coordinates[1])+float(dc.centrif_coordinates[3]) #right bottom corner -y
            centrif_corners[3,0,0] = float(dc.centrif_coordinates[0]) #left bottom corner -x
            centrif_corners[3,0,1] = float(dc.centrif_coordinates[1])+float(dc.centrif_coordinates[3]) #left bottom corner -y

            drenar_corners = np.empty((4,1,2), dtype=np.float32)
            drenar_corners[0,0,0] = float(dc.drenar_coordinates[0])    #left top corner -x
            drenar_corners[0,0,1] = float(dc.drenar_coordinates[1])    #left top corner -y
            drenar_corners[1,0,0] = float(dc.drenar_coordinates[0])+float(dc.drenar_coordinates[2]) #right top corner -x
            drenar_corners[1,0,1] = float(dc.drenar_coordinates[1])    #right top corner -y
            drenar_corners[2,0,0] = float(dc.drenar_coordinates[0])+float(dc.drenar_coordinates[2]) #right bottom corner -x
            drenar_corners[2,0,1] = float(dc.drenar_coordinates[1])+float(dc.drenar_coordinates[3]) #right bottom corner -y
            drenar_corners[3,0,0] = float(dc.drenar_coordinates[0]) #left bottom corner -x
            drenar_corners[3,0,1] = float(dc.drenar_coordinates[1])+float(dc.drenar_coordinates[3]) #left bottom corner -y


            ####################################PERSPECTIVE TRANSFORM#########################################################
            #Applying perspective transformation to map coordinates from model to scene using Homography transformation matrix
            #scene_corners = cv.perspectiveTransform(obj_corners, H)
            rect_hour_1_coords_scene = cv.perspectiveTransform(hour1_corners, H)
            rect_hour_2_coords_scene = cv.perspectiveTransform(hour2_corners, H)
            rect_hour_3_coords_scene = cv.perspectiveTransform(hour3_corners, H)
            rect_kg_1_coords_scene = cv.perspectiveTransform(kg1_corners, H)
            rect_kg_2_coords_scene = cv.perspectiveTransform(kg2_corners, H)
            rect_temp_1_coords_scene = cv.perspectiveTransform(temp1_corners, H)
            rect_temp_2_coords_scene = cv.perspectiveTransform(temp2_corners, H)
            rect_centr_1_coords_scene = cv.perspectiveTransform(centr1_corners, H)
            rect_centr_2_coords_scene  = cv.perspectiveTransform(centr2_corners, H)
            rect_centr_3_coords_scene =  cv.perspectiveTransform(centr3_corners, H)
            rect_centr_4_coords_scene = cv.perspectiveTransform(centr4_corners, H)
            rect_speed_coords_scene = cv.perspectiveTransform(speed_corners, H)

            #----------Symbol coordinates on the scene mapped from model image ----------------
            rect_washing_coords_scene = cv.perspectiveTransform(washing_corners,H)
            rect_rinsing_coords_scene = cv.perspectiveTransform(rinsing_corners,H)
            rect_spinning_coords_scene = cv.perspectiveTransform(spinning_corners,H)
            rect_symbol4_coords_scene = cv.perspectiveTransform(symbol4_corners,H)
            rect_steam_coords_scene = cv.perspectiveTransform(steam_corners,H)
            rect_anticrease_coords_scene = cv.perspectiveTransform(anticrease_corners,H)
            rect_rinsehold_coords_scene = cv.perspectiveTransform(rinsehold_corners,H)
            rect_nightcycle_coords_scene = cv.perspectiveTransform(nightcycle_corners,H)
            rect_prewash_coords_scene = cv.perspectiveTransform(prewash_corners,H)
            rect_stain_coords_scene = cv.perspectiveTransform(stain_corners,H)
            rect_eco_coords_scene = cv.perspectiveTransform(eco_corners,H)
            rect_dailysuperquick_coords_scene = cv.perspectiveTransform(daily_corners,H)
            rect_symbol13_coords_scene = cv.perspectiveTransform(symbol13_corners,H)

            #----Program and Option coordinates on the scene mapped from model image -----------
            rect_algod_coords_scene = cv.perspectiveTransform(algod_corners,H)
            rect_sinteticos_coords_scene = cv.perspectiveTransform(sinteticos_corners,H)
            rect_delicados_coords_scene = cv.perspectiveTransform(delicados_corners,H)
            rect_lana_coords_scene = cv.perspectiveTransform(lana_corners,H)
            rect_vapor_coords_scene = cv.perspectiveTransform(vapor_corners,H)
            rect_okopower_coords_scene = cv.perspectiveTransform(oko_corners,H)
            rect_antialergia_coords_scene = cv.perspectiveTransform(antialergia_corners,H)
            rect_twentymin_coords_scene = cv.perspectiveTransform(twentymin_corners,H)
            rect_outdoor_coords_scene = cv.perspectiveTransform(outdoor_corners,H)
            rect_jeans_coords_scene = cv.perspectiveTransform(jeans_corners,H)

            rect_aclarado_coords_scene = cv.perspectiveTransform(aclarado_corners,H)
            rect_centrif_coords_scene = cv.perspectiveTransform(centrif_corners,H)
            rect_drenar_coords_scene = cv.perspectiveTransform(drenar_corners,H)

            #############################DRAW LINES on the SCENE IMAGE##############
            #-- Draw lines between the corners (the mapped object in the scene)
            #Hour lines on the scene
            img_scene = cv.polylines(img_scene,[np.int32(rect_hour_1_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_hour_2_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_hour_3_coords_scene)],True,(255,0,0),2)
            #Kilogram lines on the scene
            img_scene = cv.polylines(img_scene,[np.int32(rect_kg_1_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_kg_2_coords_scene)],True,(255,0,0),2)
            #Temperature lines on the scene
            img_scene = cv.polylines(img_scene,[np.int32(rect_temp_1_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_temp_2_coords_scene)],True,(255,0,0),2)
            #Centrifugation
            img_scene = cv.polylines(img_scene,[np.int32(rect_centr_1_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_centr_2_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_centr_3_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_centr_4_coords_scene)],True,(255,0,0),2)
            #Speed lines on the scene
            img_scene = cv.polylines(img_scene,[np.int32(rect_speed_coords_scene)],True,(255,0,0),2)
            #---------------------Draw rectangular around symbols-----------
            #Symbol1  Washing
            img_scene = cv.polylines(img_scene,[np.int32(rect_washing_coords_scene)],True,(255,0,0),2)
            #Symbol2 Rinsing
            img_scene = cv.polylines(img_scene,[np.int32(rect_rinsing_coords_scene)],True,(255,0,0),2)
            #Symbol3 Spinning
            img_scene = cv.polylines(img_scene,[np.int32(rect_spinning_coords_scene)],True,(255,0,0),2)
            #Symbol4
            img_scene = cv.polylines(img_scene,[np.int32(rect_symbol4_coords_scene)],True,(255,0,0),2)
            #Symbol5 Steam
            img_scene = cv.polylines(img_scene,[np.int32(rect_steam_coords_scene)],True,(255,0,0),2)
            #Symbol6 Anticrease
            img_scene = cv.polylines(img_scene,[np.int32(rect_anticrease_coords_scene)],True,(255,0,0),2)
            #Symbol7 Rinse_Hold
            img_scene = cv.polylines(img_scene,[np.int32(rect_rinsehold_coords_scene)],True,(255,0,0),2)
            #Symbol8 Night_Cycle
            img_scene = cv.polylines(img_scene,[np.int32(rect_nightcycle_coords_scene)],True,(255,0,0),2)
            #Symbol9 Prewash
            img_scene = cv.polylines(img_scene,[np.int32(rect_prewash_coords_scene)],True,(255,0,0),2)
            #Symbol10 Stain
            img_scene = cv.polylines(img_scene,[np.int32(rect_stain_coords_scene)],True,(255,0,0),2)
            #Symbol11 Eco
            img_scene = cv.polylines(img_scene,[np.int32(rect_eco_coords_scene)],True,(255,0,0),2)
            #Symbol12 DailyTM2L7
            img_scene = cv.polylines(img_scene,[np.int32(rect_dailysuperquick_coords_scene)],True,(255,0,0),2)
            #Symbol13 Symbol13
            img_scene = cv.polylines(img_scene,[np.int32(rect_symbol13_coords_scene)],True,(255,0,0),2)
            #----Draw rectangular on Program and Option led lights
            img_scene = cv.polylines(img_scene,[np.int32(rect_algod_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_sinteticos_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_delicados_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_lana_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_vapor_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_okopower_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_antialergia_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_twentymin_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_outdoor_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_jeans_coords_scene)],True,(255,0,0),2)
            #Draw rectangular on Option led lights on the scene
            img_scene = cv.polylines(img_scene,[np.int32(rect_aclarado_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_centrif_coords_scene)],True,(255,0,0),2)
            img_scene = cv.polylines(img_scene,[np.int32(rect_drenar_coords_scene)],True,(255,0,0),2)


            #---------------ROS MESSAGE FOR DIGITS--------------------------
            DigitCoordinates.firsthourupleft = rect_hour_1_coords_scene[0][0]
            DigitCoordinates.firsthourupright = rect_hour_1_coords_scene[1][0]
            DigitCoordinates.firsthourbottomleft = rect_hour_1_coords_scene[3][0]
            DigitCoordinates.firsthourbottomright = rect_hour_1_coords_scene[2][0]

            DigitCoordinates.secondhourupleft = rect_hour_2_coords_scene[0][0]
            DigitCoordinates.secondhourupright = rect_hour_2_coords_scene[1][0]
            DigitCoordinates.secondhourbottomleft = rect_hour_2_coords_scene[3][0]
            DigitCoordinates.secondhourbottomright = rect_hour_2_coords_scene[2][0]

            DigitCoordinates.thirdhourupleft = rect_hour_3_coords_scene[0][0]
            DigitCoordinates.thirdhourupright = rect_hour_3_coords_scene[1][0]
            DigitCoordinates.thirdhourbottomleft = rect_hour_3_coords_scene[3][0]
            DigitCoordinates.thirdhourbottomright = rect_hour_3_coords_scene[2][0]
            #Kilogram coordinates on the scene
            DigitCoordinates.firstkgupleft = rect_kg_1_coords_scene[0][0]
            DigitCoordinates.firstkgupright = rect_kg_1_coords_scene[1][0]
            DigitCoordinates.firstkgbottomleft = rect_kg_1_coords_scene[3][0]
            DigitCoordinates.firstkgbottomright = rect_kg_1_coords_scene[2][0]

            DigitCoordinates.secondkgupleft = rect_kg_2_coords_scene[0][0]
            DigitCoordinates.secondkgupright = rect_kg_2_coords_scene[1][0]
            DigitCoordinates.secondkgbottomleft = rect_kg_2_coords_scene[3][0]
            DigitCoordinates.secondkgbottomright = rect_kg_2_coords_scene[2][0]
            #Temperature coordinates on the scene
            DigitCoordinates.firsttempupleft = rect_temp_1_coords_scene[0][0]
            DigitCoordinates.firsttempupright = rect_temp_1_coords_scene[1][0]
            DigitCoordinates.firsttempbottomleft = rect_temp_1_coords_scene[3][0]
            DigitCoordinates.firsttempbottomright = rect_temp_1_coords_scene[2][0]

            DigitCoordinates.secondtempupleft = rect_temp_2_coords_scene[0][0]
            DigitCoordinates.secondtempupright = rect_temp_2_coords_scene[1][0]
            DigitCoordinates.secondtempbottomleft = rect_temp_2_coords_scene[3][0]
            DigitCoordinates.secondtempbottomright = rect_temp_2_coords_scene[2][0]
            #Centrifugation coordinates on the scene
            DigitCoordinates.firstcentrupleft = rect_centr_1_coords_scene[0][0]
            DigitCoordinates.firstcentrupright = rect_centr_1_coords_scene[1][0]
            DigitCoordinates.firstcentrbottomleft = rect_centr_1_coords_scene[3][0]
            DigitCoordinates.firstcentrbottomright = rect_centr_1_coords_scene[2][0]

            DigitCoordinates.secondcentrupleft = rect_centr_2_coords_scene[0][0]
            DigitCoordinates.secondcentrupright = rect_centr_2_coords_scene[1][0]
            DigitCoordinates.secondcentrbottomleft = rect_centr_2_coords_scene[3][0]
            DigitCoordinates.secondcentrbottomright = rect_centr_2_coords_scene[2][0]

            DigitCoordinates.thirdcentrupleft = rect_centr_3_coords_scene[0][0]
            DigitCoordinates.thirdcentrupright = rect_centr_3_coords_scene[1][0]
            DigitCoordinates.thirdcentrbottomleft = rect_centr_3_coords_scene[3][0]
            DigitCoordinates.thirdcentrbottomright = rect_centr_3_coords_scene[2][0]

            DigitCoordinates.fourthcentrupleft = rect_centr_4_coords_scene[0][0]
            DigitCoordinates.fourthcentrupright = rect_centr_4_coords_scene[1][0]
            DigitCoordinates.fourthcentrbottomleft = rect_centr_4_coords_scene[3][0]
            DigitCoordinates.fourthcentrbottomright = rect_centr_4_coords_scene[2][0]

            DigitCoordinates.speedupleft = rect_speed_coords_scene[0][0]
            DigitCoordinates.speedupright = rect_speed_coords_scene[1][0]
            DigitCoordinates.speedbottomleft = rect_speed_coords_scene[3][0]
            DigitCoordinates.speedbottomright = rect_speed_coords_scene[2][0]

            #-----------------ROS MESSAGE FOR SYMBOLS (Wash Functions)------------------------
            SymbolCoordinates.s1upleft = rect_washing_coords_scene[0][0]
            SymbolCoordinates.s1upright = rect_washing_coords_scene[1][0]
            SymbolCoordinates.s1bottomleft = rect_washing_coords_scene[3][0]
            SymbolCoordinates.s1bottomright = rect_washing_coords_scene[2][0]

            SymbolCoordinates.s2upleft = rect_rinsing_coords_scene[0][0]
            SymbolCoordinates.s2upright = rect_rinsing_coords_scene[1][0]
            SymbolCoordinates.s2bottomleft = rect_rinsing_coords_scene[3][0]
            SymbolCoordinates.s2bottomright = rect_rinsing_coords_scene[2][0]

            SymbolCoordinates.s3upleft = rect_spinning_coords_scene[0][0]
            SymbolCoordinates.s3upright = rect_spinning_coords_scene[1][0]
            SymbolCoordinates.s3bottomleft = rect_spinning_coords_scene[3][0]
            SymbolCoordinates.s3bottomright = rect_spinning_coords_scene[2][0]


            SymbolCoordinates.s4upleft = rect_symbol4_coords_scene[0][0]
            SymbolCoordinates.s4upright = rect_symbol4_coords_scene[1][0]
            SymbolCoordinates.s4bottomleft = rect_symbol4_coords_scene[3][0]
            SymbolCoordinates.s4bottomright = rect_symbol4_coords_scene[2][0]

            SymbolCoordinates.s5upleft = rect_steam_coords_scene[0][0]
            SymbolCoordinates.s5upright = rect_steam_coords_scene[1][0]
            SymbolCoordinates.s5bottomleft = rect_steam_coords_scene[3][0]
            SymbolCoordinates.s5bottomright = rect_steam_coords_scene[2][0]

            SymbolCoordinates.s6upleft = rect_anticrease_coords_scene[0][0]
            SymbolCoordinates.s6upright = rect_anticrease_coords_scene[1][0]
            SymbolCoordinates.s6bottomleft = rect_anticrease_coords_scene[3][0]
            SymbolCoordinates.s6bottomright = rect_anticrease_coords_scene[2][0]

            SymbolCoordinates.s7upleft = rect_rinsehold_coords_scene[0][0]
            SymbolCoordinates.s7upright = rect_rinsehold_coords_scene[1][0]
            SymbolCoordinates.s7bottomleft = rect_rinsehold_coords_scene[3][0]
            SymbolCoordinates.s7bottomright = rect_rinsehold_coords_scene[2][0]

            SymbolCoordinates.s8upleft = rect_nightcycle_coords_scene[0][0]
            SymbolCoordinates.s8upright = rect_nightcycle_coords_scene[1][0]
            SymbolCoordinates.s8bottomleft = rect_nightcycle_coords_scene[3][0]
            SymbolCoordinates.s8bottomright = rect_nightcycle_coords_scene[2][0]

            SymbolCoordinates.s9upleft = rect_prewash_coords_scene[0][0]
            SymbolCoordinates.s9upright = rect_prewash_coords_scene[1][0]
            SymbolCoordinates.s9bottomleft = rect_prewash_coords_scene[3][0]
            SymbolCoordinates.s9bottomright = rect_prewash_coords_scene[2][0]

            SymbolCoordinates.s10upleft = rect_stain_coords_scene[0][0]
            SymbolCoordinates.s10upright = rect_stain_coords_scene[1][0]
            SymbolCoordinates.s10bottomleft = rect_stain_coords_scene[3][0]
            SymbolCoordinates.s10bottomright = rect_stain_coords_scene[2][0]

            SymbolCoordinates.s11upleft = rect_eco_coords_scene[0][0]
            SymbolCoordinates.s11upright = rect_eco_coords_scene[1][0]
            SymbolCoordinates.s11bottomleft = rect_eco_coords_scene[3][0]
            SymbolCoordinates.s11bottomright = rect_eco_coords_scene[2][0]

            SymbolCoordinates.s12upleft = rect_dailysuperquick_coords_scene[0][0]
            SymbolCoordinates.s12upright = rect_dailysuperquick_coords_scene[1][0]
            SymbolCoordinates.s12bottomleft = rect_dailysuperquick_coords_scene[3][0]
            SymbolCoordinates.s12bottomright = rect_dailysuperquick_coords_scene[2][0]

            SymbolCoordinates.s13upleft = rect_symbol13_coords_scene[0][0]
            SymbolCoordinates.s13upright = rect_symbol13_coords_scene[1][0]
            SymbolCoordinates.s13bottomleft = rect_symbol13_coords_scene[3][0]
            SymbolCoordinates.s13bottomright = rect_symbol13_coords_scene[2][0]

            #ROS MESSAGES FOR PROGRAM AND OPTION LED LIGHTS
            ProgramOptionCoordinates.prg1left = rect_algod_coords_scene[0][0]
            ProgramOptionCoordinates.prg1right = rect_algod_coords_scene[1][0]
            ProgramOptionCoordinates.prg2left = rect_algod_coords_scene[3][0]
            ProgramOptionCoordinates.prg2right = rect_algod_coords_scene[2][0]

            ProgramOptionCoordinates.prg2left = rect_sinteticos_coords_scene[0][0]
            ProgramOptionCoordinates.prg2right = rect_sinteticos_coords_scene[1][0]
            ProgramOptionCoordinates.prg3left = rect_sinteticos_coords_scene[3][0]
            ProgramOptionCoordinates.prg3right = rect_sinteticos_coords_scene[2][0]

            ProgramOptionCoordinates.prg3left = rect_delicados_coords_scene[0][0]
            ProgramOptionCoordinates.prg3right = rect_delicados_coords_scene[1][0]
            ProgramOptionCoordinates.prg4left = rect_delicados_coords_scene[3][0]
            ProgramOptionCoordinates.prg4right = rect_delicados_coords_scene[2][0]

            ProgramOptionCoordinates.prg4left = rect_lana_coords_scene[0][0]
            ProgramOptionCoordinates.prg4right = rect_lana_coords_scene[1][0]
            ProgramOptionCoordinates.prg5left = rect_lana_coords_scene[3][0]
            ProgramOptionCoordinates.prg5right = rect_lana_coords_scene[2][0]

            ProgramOptionCoordinates.prg5left = rect_vapor_coords_scene[0][0]
            ProgramOptionCoordinates.prg5right = rect_vapor_coords_scene[1][0]
            ProgramOptionCoordinates.prg6left = rect_vapor_coords_scene[3][0]
            ProgramOptionCoordinates.prg6right = rect_vapor_coords_scene[2][0]

            ProgramOptionCoordinates.prg6left = rect_okopower_coords_scene[0][0]
            ProgramOptionCoordinates.prg6right = rect_okopower_coords_scene[1][0]
            ProgramOptionCoordinates.prg7left = rect_okopower_coords_scene[3][0]
            ProgramOptionCoordinates.prg7right = rect_okopower_coords_scene[2][0]

            ProgramOptionCoordinates.prg7left = rect_antialergia_coords_scene[0][0]
            ProgramOptionCoordinates.prg7right = rect_antialergia_coords_scene[1][0]
            ProgramOptionCoordinates.prg8left = rect_antialergia_coords_scene[3][0]
            ProgramOptionCoordinates.prg8right = rect_antialergia_coords_scene[2][0]

            ProgramOptionCoordinates.prg8left = rect_twentymin_coords_scene[0][0]
            ProgramOptionCoordinates.prg8right = rect_twentymin_coords_scene[1][0]
            ProgramOptionCoordinates.prg9left = rect_twentymin_coords_scene[3][0]
            ProgramOptionCoordinates.prg9right = rect_twentymin_coords_scene[2][0]

            ProgramOptionCoordinates.prg9left = rect_outdoor_coords_scene[0][0]
            ProgramOptionCoordinates.prg9right = rect_outdoor_coords_scene[1][0]
            ProgramOptionCoordinates.prg10left = rect_outdoor_coords_scene[3][0]
            ProgramOptionCoordinates.prg10right = rect_outdoor_coords_scene[2][0]

            ProgramOptionCoordinates.prg10left = rect_jeans_coords_scene[0][0]
            ProgramOptionCoordinates.prg10right = rect_jeans_coords_scene[1][0]
            ProgramOptionCoordinates.prg11left = rect_jeans_coords_scene[3][0]
            ProgramOptionCoordinates.prg11right = rect_jeans_coords_scene[2][0]


            #Publishing coordinate messages
            digit_coord_pub.publish(DigitCoordinates)
            symbol_coord_pub.publish(SymbolCoordinates)
            program_option_coord_pub.publish(ProgramOptionCoordinates)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            #-- Show detected matches
            cv.imshow('Good Matches & Object detection', img_scene)
            cv.waitKey(1)
    cv.release()
    cv.destroyAllWindows()
