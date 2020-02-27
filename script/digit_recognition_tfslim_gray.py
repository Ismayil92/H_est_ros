#!/usr/bin/env python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Process
import argparse
import rospy
import roslib
roslib.load_manifest('thesispro')
import numpy as np
import tensorflow as tf
import cv2 as cv
import Tkinter
import tkFileDialog
import os
import easygui
import sys
from thesispro.msg import screen
from thesispro.msg import localizaton
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import datetime



def checkStatus(croppedSymbol):
    global status
    #gray = cv.cvtColor(croppedSymbol,cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(gray,(5,5),0)
    #Otsu's thresholding after Gaussian filtering
    #ret3,threshcrop = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
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

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


myinput_width = 96
myinput_height = 96

def read_tensor_from_image_file(
        image,
        input_height=myinput_height,
        input_width=myinput_width,
        input_mean=0,
        input_std=255):
    #r = cv.selectROI("Image", resimg, False, False)
    # Crop image
    # imCrop = resimg[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # Numpy array
    np_image_data = np.asarray(image)
    np_image_data = np.stack((np_image_data,)*3, -1)
    float_caster = tf.cast(np_image_data, tf.float32)
    # maybe insert float convertion here - see edit remark!
    # np_final = np.expand_dims(np_image_data, axis=0)
    dims_expander = tf.expand_dims(float_caster, 0)

    #input_name = "file_reader"
    #output_name = "normalized"
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def classification(image,graph,labels):
    t = read_tensor_from_image_file(
        image,
        input_height=myinput_height,
        input_width=myinput_width,
        input_mean=input_mean,
        input_std=input_std)

    input_layer = "input"
    output_layer = "MobilenetV2/Predictions/Reshape_1"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    if (str(labels[top_k[0]]) == '0' or str(labels[top_k[0]]) == '-'):
        reply = '0'
    elif (str(labels[top_k[0]]) == '1'):
        reply = '1'
    elif (str(labels[top_k[0]]) == '2'):
        reply = '2'
    elif (str(labels[top_k[0]]) == '3'):
        reply = '3'
    elif (str(labels[top_k[0]]) == '4'):
        reply = '4'
    elif (str(labels[top_k[0]]) == '5'):
        reply = '5'
    elif (str(labels[top_k[0]]) == '6'):
        reply = '6'
    elif (str(labels[top_k[0]]) == '7'):
        reply = '7'
    elif (str(labels[top_k[0]]) == '8'):
        reply = '8'
    elif (str(labels[top_k[0]]) == '9'):
        reply = '9'

    return reply


class image_converter: #To get Image as message
    def __init__(self):
        self.bridge = CvBridge()
        self.graph = load_graph(model_file)
        self.labels = load_labels(label_file)
        self.image_sub = rospy.Subscriber("imgConnection", Image, self.callback)
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            blur = cv.GaussianBlur(cv_image,(5,5),0)
            #Otsu's thresholding after Gaussian filtering
            ret3,resimg = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            #Centrifugation
            rCentr1 = [int(DigitCoordinates.firstcentrupleft[0]), int(DigitCoordinates.firstcentrupleft[1]), int(DigitCoordinates.firstcentrbottomleft[1]), int(DigitCoordinates.firstcentrupright[0])]
            rCentr2 = [int(DigitCoordinates.secondcentrupleft[0]), int(DigitCoordinates.secondcentrupleft[1]), int(DigitCoordinates.secondcentrbottomleft[1]), int(DigitCoordinates.secondcentrupright[0])]
            rCentr3 = [int(DigitCoordinates.thirdcentrupleft[0]), int(DigitCoordinates.thirdcentrupleft[1]), int(DigitCoordinates.thirdcentrbottomleft[1]), int(DigitCoordinates.thirdcentrupright[0])]
            rCentr4 = [int(DigitCoordinates.fourthcentrupleft[0]), int(DigitCoordinates.fourthcentrupleft[1]), int(DigitCoordinates.fourthcentrbottomleft[1]), int(DigitCoordinates.fourthcentrupright[0])]
            #Kilogram
            rKg1 = [int(DigitCoordinates.firstkgupleft[0]), int(DigitCoordinates.firstkgupleft[1]), int(DigitCoordinates.firstkgbottomleft[1]), int(DigitCoordinates.firstkgupright[0])]
            rKg2 = [int(DigitCoordinates.secondkgupleft[0]), int(DigitCoordinates.secondkgupleft[1]), int(DigitCoordinates.secondkgbottomleft[1]), int(DigitCoordinates.secondkgupright[0])]
            #Temperature
            rTemp1 = [int(DigitCoordinates.firsttempupleft[0]), int(DigitCoordinates.firsttempupleft[1]), int(DigitCoordinates.firsttempbottomleft[1]), int(DigitCoordinates.firsttempupright[0])]
            rTemp2 = [int(DigitCoordinates.secondtempupleft[0]), int(DigitCoordinates.secondtempupleft[1]), int(DigitCoordinates.secondtempbottomleft[1]), int(DigitCoordinates.secondtempupright[0])]
            #Hour
            rh1 = [int(DigitCoordinates.firsthourupleft[0]), int(DigitCoordinates.firsthourupleft[1]), int(DigitCoordinates.firsthourbottomleft[1]), int(DigitCoordinates.firsthourupright[0])]
            rh2 = [int(DigitCoordinates.secondhourupleft[0]), int(DigitCoordinates.secondhourupleft[1]), int(DigitCoordinates.secondhourbottomleft[1]), int(DigitCoordinates.secondhourupright[0])]
            rh3 = [int(DigitCoordinates.thirdhourupleft[0]), int(DigitCoordinates.thirdhourupleft[1]), int(DigitCoordinates.thirdhourbottomleft[1]), int(DigitCoordinates.thirdhourupright[0])]
            #Speed
            rSpeed1 = [int(DigitCoordinates.speedupleft[0]), int(DigitCoordinates.speedupleft[1]), int(DigitCoordinates.speedbottomleft[1]), int(DigitCoordinates.speedupright[0])]
            #-------------------------------------Digit Cropped---------------------------------------------
            imCropCentr1 = resimg[rCentr1[1]:rCentr1[2],rCentr1[0]:rCentr1[3]]
            imCropCentr2 = resimg[rCentr2[1]:rCentr2[2],rCentr2[0]:rCentr2[3]]
            imCropCentr3 = resimg[rCentr3[1]:rCentr3[2],rCentr3[0]:rCentr3[3]]
            imCropCentr4 = resimg[rCentr4[1]:rCentr4[2],rCentr4[0]:rCentr4[3]]

            imCropKg1 = resimg[rKg1[1]:rKg1[2],rKg1[0]:rKg1[3]]
            imCropKg2 = resimg[rKg2[1]:rKg2[2],rKg2[0]:rKg2[3]]

            imCropTemp1 = resimg[rTemp1[1]:rTemp1[2],rTemp1[0]:rTemp1[3]]
            imCropTemp2 = resimg[rTemp2[1]:rTemp2[2],rTemp2[0]:rTemp2[3]]

            imCropHour1 = resimg[rh1[1]:rh1[2],rh1[0]:rh1[3]]
            imCropHour2 = resimg[rh2[1]:rh2[2],rh2[0]:rh2[3]]
            imCropHour3 = resimg[rh3[1]:rh3[2],rh3[0]:rh3[3]]

            imCropSpeed = resimg[rh1[1]:rh1[2],rh1[0]:rh1[3]]
            #-----------------------------------------------------------------------------------------------
            centrifugation = {}
            kilogram = {}
            temperature = {}
            hour = {}
            speed = {}
            #-----------------------------------------------------------------------------------------------
            if checkStatus(imCropCentr1)==True:
                centrifugation[0] = classification(imCropCentr1,self.graph,self.labels)
            else:
                centrifugation[0] = 0
            if checkStatus(imCropCentr2)==True:
                centrifugation[1] = classification(imCropCentr2,self.graph,self.labels)
            else:
                centrifugation[1] = 0
            if checkStatus(imCropCentr3)==True:
                centrifugation[2] = classification(imCropCentr3,self.graph,self.labels)
            else:
                centrifugation[2] = 0
            if checkStatus(imCropCentr4)==True:
                centrifugation[3] = classification(imCropCentr4,self.graph,self.labels)
            else:
                centrifugation[3] = 0

            if checkStatus(imCropKg1)==True:
                kilogram[0] = classification(imCropKg1,self.graph,self.labels)
            else:
                kilogram[0] = 0
            if checkStatus(imCropKg2)==True:
                kilogram[1] = classification(imCropKg2,self.graph,self.labels)
            else:
                kilogram[1] = 0

            if checkStatus(imCropTemp1)==True:
                temperature[0] = classification(imCropTemp1,self.graph,self.labels)
            else:
                temperature[0] = 0
            if checkStatus(imCropTemp2)==True:
                temperature[1] = classification(imCropTemp2,self.graph,self.labels)
            else:
                temperature[1] = 0

            if checkStatus(imCropHour1)==True:
                hour[0] = classification(imCropHour1,self.graph,self.labels)
            else:
                hour[0] = 0
            if checkStatus(imCropHour2)==True:
                hour[1] = classification(imCropHour2,self.graph,self.labels)
            else:
                hour[1] = 0
            if checkStatus(imCropHour3)==True:
                hour[2] = classification(imCropHour3,self.graph,self.labels)
            else:
                hour[2] = 0

            if checkStatus(imCropSpeed)==True:
                speed[0] = classification(imCropSpeed,self.graph,self.labels)
            else:
                speed[0] = '0'



            spin_number = int(str(centrifugation[0])+str(centrifugation[1])+str(centrifugation[2])+str(centrifugation[3]))
            kg = float(str(kilogram[0])+'.'+str(kilogram[1]))
            temperatureWashing = int(str(temperature[0])+str(temperature[1]))
            time = int(str(hour[0]) + str(hour[1]) + str(hour[2]))
            spinSpeed = int(speed[0])



            msg.User_select_Rinse_Nums = spin_number
            msg.Kilogram = kg
            msg.temperature = temperatureWashing
            msg.User_Select_Delay_Time = time
            msg.Spin_speed = spinSpeed


            print(msg.User_select_Rinse_Nums)
            print(msg.Kilogram)
            print(msg.temperature)
            print(msg.User_Select_Delay_Time)
            print(msg.Spin_speed)
            pub.publish(msg)
            cv.waitKey(1)


            if cv.waitKey(1) & 0xFF == ord('q'):
               cv.destroyAllWindows()
        except CvBridgeError as e:
            print(e)




def callback_coordinates(data):

    #Hour
    DigitCoordinates.firsthourupleft = data.firsthourupleft
    DigitCoordinates.firsthourupright = data.firsthourupright
    DigitCoordinates.firsthourbottomleft = data.firsthourbottomleft
    DigitCoordinates.firsthourbottomright = data.firsthourbottomright
    #
    DigitCoordinates.secondhourupleft = data.secondhourupleft
    DigitCoordinates.secondhourupright = data.secondhourupright
    DigitCoordinates.secondhourbottomleft = data.secondhourbottomleft
    DigitCoordinates.secondhourbottomright = data.secondhourbottomright
    #
    DigitCoordinates.thirdhourupleft = data.thirdhourupleft
    DigitCoordinates.thirdhourupright = data.thirdhourupright
    DigitCoordinates.thirdhourbottomleft = data.thirdhourbottomleft
    DigitCoordinates.thirdhourbottomright = data.thirdhourbottomright

    #Kilogram
    DigitCoordinates.firstkgupleft = data.firstkgupleft
    DigitCoordinates.firstkgupright = data.firstkgupright
    DigitCoordinates.firstkgbottomleft = data.firstkgbottomleft
    DigitCoordinates.firstkgbottomright = data.firstkgbottomright
    #
    DigitCoordinates.secondkgupleft = data.secondkgupleft
    DigitCoordinates.secondkgupright = data.secondkgupright
    DigitCoordinates.secondkgbottomleft = data.secondkgbottomleft
    DigitCoordinates.secondkgbottomright = data.secondkgbottomright

    #Temperature
    DigitCoordinates.firsttempupleft = data.firsttempupleft
    DigitCoordinates.firsttempupright = data.firsttempupright
    DigitCoordinates.firsttempbottomleft = data.firsttempbottomleft
    DigitCoordinates.firsttempbottomright = data.firsttempbottomright
    #
    DigitCoordinates.secondtempupleft = data.secondtempupleft
    DigitCoordinates.secondtempupright = data.secondtempupright
    DigitCoordinates.secondtempbottomleft = data.secondtempbottomleft
    DigitCoordinates.secondtempbottomright = data.secondtempbottomright

    #Centrifugation
    DigitCoordinates.firstcentrupleft = data.firstcentrupleft
    DigitCoordinates.firstcentrupright = data.firstcentrupright
    DigitCoordinates.firstcentrbottomleft = data.firstcentrbottomleft
    DigitCoordinates.firstcentrbottomright = data.firstcentrbottomright
    #
    DigitCoordinates.secondcentrupleft = data.secondcentrupleft
    DigitCoordinates.secondcentrupright = data.secondcentrupright
    DigitCoordinates.secondcentrbottomleft = data.secondcentrbottomleft
    DigitCoordinates.secondcentrbottomright = data.secondcentrbottomright
    #
    DigitCoordinates.thirdcentrupleft = data.thirdcentrupleft
    DigitCoordinates.thirdcentrupright = data.thirdcentrupright
    DigitCoordinates.thirdcentrbottomleft = data.thirdcentrbottomleft
    DigitCoordinates.thirdcentrbottomright = data.thirdcentrbottomright
    #
    DigitCoordinates.fourthcentrupleft = data.fourthcentrupleft
    DigitCoordinates.fourthcentrupright = data.fourthcentrupright
    DigitCoordinates.fourthcentrbottomleft = data.fourthcentrbottomleft
    DigitCoordinates.fourthcentrbottomright = data.fourthcentrbottomright
    #
    DigitCoordinates.speedupleft = data.speedupleft
    DigitCoordinates.speedupright = data.speedupright
    DigitCoordinates.speedbottomleft = data.speedbottomleft
    DigitCoordinates.speedbottomright = data.speedbottomright

     #Centrifugation
    rCentr1 = [int(DigitCoordinates.firstcentrupleft[0]), int(DigitCoordinates.firstcentrupleft[1]), int(DigitCoordinates.firstcentrbottomleft[1]), int(DigitCoordinates.firstcentrupright[0])]
    rCentr2 = [int(DigitCoordinates.secondcentrupleft[0]), int(DigitCoordinates.secondcentrupleft[1]), int(DigitCoordinates.secondcentrbottomleft[1]), int(DigitCoordinates.secondcentrupright[0])]
    rCentr3 = [int(DigitCoordinates.thirdcentrupleft[0]), int(DigitCoordinates.thirdcentrupleft[1]), int(DigitCoordinates.thirdcentrbottomleft[1]), int(DigitCoordinates.thirdcentrupright[0])]
    rCentr4 = [int(DigitCoordinates.fourthcentrupleft[0]), int(DigitCoordinates.fourthcentrupleft[1]), int(DigitCoordinates.fourthcentrbottomleft[1]), int(DigitCoordinates.fourthcentrupright[0])]
    #Kilogram
    rKg1 = [int(DigitCoordinates.firstkgupleft[0]), int(DigitCoordinates.firstkgupleft[1]), int(DigitCoordinates.firstkgbottomleft[1]), int(DigitCoordinates.firstkgupright[0])]
    rKg2 = [int(DigitCoordinates.secondkgupleft[0]), int(DigitCoordinates.secondkgupleft[1]), int(DigitCoordinates.secondkgbottomleft[1]), int(DigitCoordinates.secondkgupright[0])]
    #Temperature
    rTemp1 = [int(DigitCoordinates.firsttempupleft[0]), int(DigitCoordinates.firsttempupleft[1]), int(DigitCoordinates.firsttempbottomleft[1]), int(DigitCoordinates.firsttempupright[0])]
    rTemp2 = [int(DigitCoordinates.secondtempupleft[0]), int(DigitCoordinates.secondtempupleft[1]), int(DigitCoordinates.secondtempbottomleft[1]), int(DigitCoordinates.secondtempupright[0])]
    #Hour
    rh1 = [int(DigitCoordinates.firsthourupleft[0]), int(DigitCoordinates.firsthourupleft[1]), int(DigitCoordinates.firsthourbottomleft[1]), int(DigitCoordinates.firsthourupright[0])]
    rh2 = [int(DigitCoordinates.secondhourupleft[0]), int(DigitCoordinates.secondhourupleft[1]), int(DigitCoordinates.secondhourbottomleft[1]), int(DigitCoordinates.secondhourupright[0])]
    rh3 = [int(DigitCoordinates.thirdhourupleft[0]), int(DigitCoordinates.thirdhourupleft[1]), int(DigitCoordinates.thirdhourbottomleft[1]), int(DigitCoordinates.thirdhourupright[0])]
    #Speed
    rSpeed1 = [int(DigitCoordinates.speedupleft[0]), int(DigitCoordinates.speedupleft[1]), int(DigitCoordinates.speedbottomleft[1]), int(DigitCoordinates.speedupright[0])]




if __name__ == "__init__":
    global DigitCoordinates
    global rh1,rh2,rh3,rh4

if __name__ == "__main__":
    #Model file = path of the frozen graph file
    model_file = "/home/ismayil/catkin_ws/src/thesispro/resources/models/Slim/frozen_mobilenetV2_1.0_96_gray.pb"
    #label_list
    label_file = "/home/ismayil/catkin_ws/src/thesispro/resources/models/Slim/labels.txt"
    input_height = myinput_height
    input_width = myinput_width
    input_mean = 0
    input_std = 255

    #ROS PUBLISHER
    rospy.init_node('user_interface_info', anonymous = True)
    pub = rospy.Publisher('user_interface_info', screen, queue_size = 1)
    msg = screen()
    #rate = rospy.Rate(1) # 10hz
    #to get Digit Coordinates
    DigitCoordinates = localizaton()
    coordinates = rospy.Subscriber("DigitLocalizer",localizaton,callback_coordinates)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.release()
    cv.destroyAllWindows()
