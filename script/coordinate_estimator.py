#!/usr/bin/env python

from __future__ import print_function 
import cv2 as cv
import numpy as np 
import rospy 
import os
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError 
from enum import Enum

#Global Variables 
bridge = CvBridge()
rgbImg = np.zeros([640,480],dtype = np.uint8)
grayImg = np.zeros([640,480], dtype = np.uint8)
sceneRGBImg = np.zeros([640,480], dtype=np.uint8)
refRGBImg = np.zeros([640,480], dtype = np.uint8)
refGrayImg = np.zeros([640,480], dtype = np.uint8)
H = np.empty((3,3), dtype = np.float32)

def openRefImages(image_name):
    global refRGBImg 
    global refGrayImg
    abs_drc = os.path.abspath(__file__)
    path_to_scripts_ind = abs_drc.find('script')
    path_to_project = abs_drc[0:path_to_scripts_ind]
    PATH_TO_IMAGE = os.path.join(path_to_project, 'Data/images/', image_name)
    refRGBImg = cv.imread(PATH_TO_IMAGE, cv.IMREAD_COLOR)
    if refRGBImg is None:
            print('Could not open or find the image!')
            exit(0)
    refGrayImg = cv.cvtColor(refRGBImg,cv.COLOR_BGR2GRAY)
    
    

def frameCallback(data): #Subscribing Video Frames
    global rgbImg
    global grayImg
    try:
        rgbImg = bridge.imgmsg_to_cv2(data,"bgr8")
        grayImg = cv.cvtColor(rgbImg, cv.COLOR_BGR2GRAY)        
    except CvBridgeError as e:
        print(e)

class Symbols(Enum):
    WASHING = 0
    RINSING = 1
    SPINNING = 2
    SYMBOL4 = 3
    STEAM = 4
    ANTICREASE = 5
    RINSEHOLD = 6
    NIGHTCYCLE = 7
    PREWASH = 8
    STAIN = 9
    ECO = 10
    DAILY = 11
    SYMBOL = 12
class Programs(Enum):
    ALGOD = 0
    SINTETICOS = 1
    DELICADOS = 2
    LANA = 3
    VAPOR = 4
    OKO = 5
    ANTIALERGIA = 6
    TWENTYMIN = 7
    OUTDOOR = 8
    JEANS = 9

class Options(Enum):
    ACLARADO = 0
    CENTRIF = 1
    DRENAR = 2
    

class des_coordinates: #Desired coordinates on the model image [x,y,width,height]
    #Digits
    hour_coord = np.array([[1146,114,61,100],
    [1215,114,61,100],
    [1279,114,61,100]], dtype=np.int32)

    kg_coord = np.array([[713,123,760-713,75],
    [760,123,760-713,75]],dtype=np.int32)
   
    temp_coord = np.array([[740,338,784-740,408-338],
    [784,338,784-740,408-338]],dtype=np.int32)

    centr_coord = np.array([[859,338,887-859,70],
    [887,338,44,70],
    [887+44,338,44,70],
    [887+2*44,338,44,70]],dtype=np.int32)

    speed_coord = np.array([[1263,333,1310-1263,73]],dtype=np.int32)

    #Symbols
    washing_coord = np.array([[703,225,66,70]],dtype=np.int32)
    rinsing_coord = np.array([[771,225,54,70]],dtype=np.int32)
    spinning_coord = np.array([[831,225,43,70]],dtype=np.int32)
    symbol4_coord = np.array([[884,225,42,70]],dtype=np.int32)
    steam_coord = np.array([[937,225,55,70]],dtype=np.int32)
    anticrease_coord = np.array([[1002,225,58,70]],dtype=np.int32)
    rinsehold_coord = np.array([[871,290,63,53]],dtype=np.int32)
    nightcycle_coord = np.array([[934,288,60,51]],dtype=np.int32)
    prewash_coord = np.array([[1033,340,78,67]],dtype=np.int32)
    stain_coord = np.array([[1109,339,72,68]],dtype=np.int32)
    eco_coord = np.array([[1253,214,80,40]],dtype=np.int32)
    dailysuperquick_coord = np.array([[1200,258,66,70]],dtype=np.int32)
    symbol13_coord = np.array([[1265,254,70,74]],dtype=np.int32)   
   
    symbol_coords = np.array([
        washing_coord,
        rinsing_coord,
        spinning_coord,
        symbol4_coord,
        steam_coord,
        anticrease_coord,
        rinsehold_coord,
        nightcycle_coord,
        prewash_coord,
        stain_coord,
        eco_coord,
        dailysuperquick_coord, 
        symbol13_coord]).squeeze()
    
    #Program
    algod_coord = np.array([[286,111,22,44]],dtype=np.int32)
    sinteticos_coord = np.array([[286,157,22,44]],dtype=np.int32)
    delicados_coord = np.array([[286,202,22,44]],dtype=np.int32)
    lana_coord = np.array([[286,247,22,44]],dtype=np.int32)
    vapor_coord = np.array([[286,296,22,44]],dtype=np.int32)
    okopower_coord = np.array([[286,343,22,44]],dtype=np.int32)
    antialergia_coord = np.array([[286,385,22,44]],dtype=np.int32)
    twentymin_coord = np.array([[286,434,22,44]],dtype=np.int32)
    outdoor_coord = np.array([[286,479,22,44]],dtype=np.int32)
    jeans_coord = np.array([[286,526,22,44]],dtype=np.int32)

    program_coords = np.array([
        algod_coord,
        sinteticos_coord,
        delicados_coord,
        lana_coord,
        vapor_coord,
        okopower_coord,
        antialergia_coord,
        twentymin_coord,
        outdoor_coord,
        jeans_coord
    ]).squeeze()

    #Option
    aclarado_coord = np.array([[1412,326,26,44]],dtype=np.int32)
    centrif_coord = np.array([[1412,370,26,44]],dtype=np.int32)
    drenar_coord = np.array([[1412,416,26,44]],dtype=np.int32)

    option_coords = np.array([
        aclarado_coord,
        centrif_coord,
        drenar_coord
    ]).squeeze()

class DesiredROIs:
    dc = des_coordinates()
    #digit ROI
    hour_corners = np.empty((3,4,1,2), dtype = np.float32)
    kg_corners = np.empty((2,4,1,2), dtype = np.float32)
    temp_corners = np.empty((2,4,1,2), dtype = np.float32)
    centr_corners = np.empty((4,4,1,2), dtype = np.float32)
    speed_corners = np.empty((1,4,1,2), dtype = np.float32)
    #symbol ROI
    symbol_corners = np.empty((13,4,1,2), dtype = np.float32) 
    #Program ROI
    program_corners = np.empty((10,4,1,2), dtype = np.float32)   
    #Option ROI 
    option_corners = np.empty((3,4,1,2), dtype = np.float32)
 
    #Kilogram
    for i in range(2): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,0])
                if j==0 and k==1:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,1])
                if j==1 and k==0:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,0])+float(dc.kg_coord[i,2])
                if j==1 and k==1:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,1])
                if j==2 and k==0:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,0])+float(dc.kg_coord[i,2])
                if j==2 and k==1:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,1])+float(dc.kg_coord[i,3])
                if j==3 and k==0:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,0])
                if j==3 and k==1:
                    kg_corners[i,j,0,k] = float(dc.kg_coord[i,1])+float(dc.kg_coord[i,3])
    #Hour 
    for i in range(3): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,0])
                if j==0 and k==1:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,1])
                if j==1 and k==0:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,0])+float(dc.hour_coord[i,2])
                if j==1 and k==1:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,1])
                if j==2 and k==0:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,0])+float(dc.hour_coord[i,2])
                if j==2 and k==1:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,1])+float(dc.hour_coord[i,3])
                if j==3 and k==0:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,0])
                if j==3 and k==1:
                    hour_corners[i,j,0,k] = float(dc.hour_coord[i,1])+float(dc.hour_coord[i,3])
    #Temperature 
    for i in range(2): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,0])
                if j==0 and k==1:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,1])
                if j==1 and k==0:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,0])+float(dc.temp_coord[i,2])
                if j==1 and k==1:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,1])
                if j==2 and k==0:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,0])+float(dc.temp_coord[i,2])
                if j==2 and k==1:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,1])+float(dc.temp_coord[i,3])
                if j==3 and k==0:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,0])
                if j==3 and k==1:
                    temp_corners[i,j,0,k] = float(dc.temp_coord[i,1])+float(dc.temp_coord[i,3])
    #Centrifugation
    for i in range(4): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,0])
                if j==0 and k==1:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,1])
                if j==1 and k==0:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,0])+float(dc.centr_coord[i,2])
                if j==1 and k==1:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,1])
                if j==2 and k==0:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,0])+float(dc.centr_coord[i,2])
                if j==2 and k==1:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,1])+float(dc.centr_coord[i,3])
                if j==3 and k==0:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,0])
                if j==3 and k==1:
                    centr_corners[i,j,0,k] = float(dc.centr_coord[i,1])+float(dc.centr_coord[i,3])
    #Speed
    for i in range(1): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,0])
                if j==0 and k==1:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,1])
                if j==1 and k==0:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,0])+float(dc.speed_coord[i,2])
                if j==1 and k==1:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,1])
                if j==2 and k==0:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,0])+float(dc.speed_coord[i,2])
                if j==2 and k==1:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,1])+float(dc.speed_coord[i,3])
                if j==3 and k==0:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,0])
                if j==3 and k==1:
                    speed_corners[i,j,0,k] = float(dc.speed_coord[i,1])+float(dc.speed_coord[i,3])
    #Symbols
    for i in range(13): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,0])
                if j==0 and k==1:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,1])
                if j==1 and k==0:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,0])+float(dc.symbol_coords[i,2])
                if j==1 and k==1:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,1])
                if j==2 and k==0:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,0])+float(dc.symbol_coords[i,2])
                if j==2 and k==1:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,1])+float(dc.symbol_coords[i,3])
                if j==3 and k==0:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,0])
                if j==3 and k==1:
                    symbol_corners[i,j,0,k] = float(dc.symbol_coords[i,1])+float(dc.symbol_coords[i,3])
    #Programs
    for i in range(10): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,0])
                if j==0 and k==1:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,1])
                if j==1 and k==0:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,0])+float(dc.program_coords[i,2])
                if j==1 and k==1:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,1])
                if j==2 and k==0:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,0])+float(dc.program_coords[i,2])
                if j==2 and k==1:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,1])+float(dc.program_coords[i,3])
                if j==3 and k==0:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,0])
                if j==3 and k==1:
                    program_corners[i,j,0,k] = float(dc.program_coords[i,1])+float(dc.program_coords[i,3])
    #Options 
    for i in range(3): #element
        for j in range(4): #corners
            for k in range(2): #x or y
                if j==0 and k ==0:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,0])
                if j==0 and k==1:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,1])
                if j==1 and k==0:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,0])+float(dc.option_coords[i,2])
                if j==1 and k==1:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,1])
                if j==2 and k==0:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,0])+float(dc.option_coords[i,2])
                if j==2 and k==1:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,1])+float(dc.option_coords[i,3])
                if j==3 and k==0:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,0])
                if j==3 and k==1:
                    option_corners[i,j,0,k] = float(dc.option_coords[i,1])+float(dc.option_coords[i,3])
    
    
class SURF_detector:
    _min_hessian = 400  
    _lowe_ratio_thresh = 0.75     
    
    def __init__(self):
        self.detector = cv.xfeatures2d.SURF_create(hessianThreshold=SURF_detector._min_hessian)
        self.keypoints_ref, self.descriptors_ref = self.detector.detectAndCompute(refGrayImg,None)
        self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    def compute(self):
        keypoints_scene, descriptors_scene = self.detector.detectAndCompute(grayImg,None)
        #  Step 2: Matching descriptor vectors with a FLANN based matcher
        #  Since SURF is a floating-point descriptor NORM_L2 is used
        knn_matches = []
        try:
            knn_matches = self.matcher.knnMatch(self.descriptors_ref, descriptors_scene, 2)
        except:
            print("Error occured in knn matching")
        #  Filter matches using the Lowe's ratio test to eliminate outliers 
        good_matches = [] 
        for m,n in knn_matches:
            if m.distance < SURF_detector._lowe_ratio_thresh * n.distance:
                good_matches.append(m)
        #  Localize the object
        obj = np.empty((len(good_matches),2), dtype=np.float32)
        scene = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            #-- Get the keypoints from the good matches in both images
            obj[i,0] = self.keypoints_ref[good_matches[i].queryIdx].pt[0]
            obj[i,1] = self.keypoints_ref[good_matches[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        # to find Homography Matrix 
        global H
        try:
            H, maskMatrix = cv.findHomography(obj, scene, cv.RANSAC)
            matches_mask = maskMatrix.ravel().tolist()
        except Exception as ex:
            print(ex)   
        return H    

def perspectiveTransform(ROI_in,H):
    ROI_scene = np.empty_like(ROI_in)
    try:
        for i in range(ROI_in.shape[0]):
            ROI_scene[i] = cv.perspectiveTransform(ROI_in[i],H)
    except:
        print("Problem occured in perspective transformation computation")
    return ROI_scene

def drawROI(ROI_scene,img_scene):
    color = (255,0,0)
    depth = 2 
    for i in range(ROI_scene.shape[0]):
        out_scene = cv.polylines(img_scene,[np.int32(ROI_scene[i])],True,color,depth)
    return out_scene


if __name__ == "__main__":
    rospy.init_node('coordinate_estimator',anonymous=True)
    rate = rospy.Rate(10)
    imageTopic = "/camera/color/image_raw"  
    image_sub = rospy.Subscriber(imageTopic, Image, frameCallback, queue_size=1)
    openRefImages("refRGBImage.jpg") 
    surf = SURF_detector()
    refCorners = DesiredROIs()    
    while not rospy.is_shutdown():
        H = surf.compute()     
        print(H)   
        # Perspective Transform 
        hour_coord_scene = perspectiveTransform(refCorners.hour_corners, H)
        kg_coord_scene = perspectiveTransform(refCorners.kg_corners, H)
        temp_coord_scene = perspectiveTransform(refCorners.temp_corners, H)
        centrif_coord_scene = perspectiveTransform(refCorners.centr_corners, H)
        speed_coord_scene = perspectiveTransform(refCorners.speed_corners, H)
        symbol_coord_scene = perspectiveTransform(refCorners.symbol_corners, H)
        program_coord_scene = perspectiveTransform(refCorners.program_corners, H)
        option_coord_scene = perspectiveTransform(refCorners.option_corners, H)
        #Draw line to show ROIs
        sceneRGBImg = drawROI(hour_coord_scene,rgbImg)
        sceneRGBImg = drawROI(kg_coord_scene,rgbImg)
        sceneRGBImg = drawROI(temp_coord_scene,rgbImg)
        sceneRGBImg = drawROI(centrif_coord_scene,rgbImg)
        sceneRGBImg = drawROI(speed_coord_scene,rgbImg)
        sceneRGBImg = drawROI(symbol_coord_scene,rgbImg)
        sceneRGBImg = drawROI(program_coord_scene,rgbImg)
        sceneRGBImg = drawROI(option_coord_scene,rgbImg)
        cv.namedWindow("RGB Image")
        cv.imshow("RGB Image",rgbImg) 
        cv.namedWindow("scene image")
        cv.imshow("scene image",sceneRGBImg)
        cv.waitKey(3)
        rate.sleep()
    cv.release()
    cv.destroyAllWindows()
        









