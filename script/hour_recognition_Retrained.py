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



def checkStatus(croppedSymbol):#finding contours pixels
    global status
    im2, contours, hierarchy = cv.findContours(croppedSymbol, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]
    try:
         perimeter = cv.arcLength(contours[0],True) #finding the length of the contour in that region of interest
         if perimeter>10: #if the number of contour pixel is greater than 10, the led is on.
                status = True
    except Exception:
         status = False
    return status

def SymbolRecognitionFunc(image,rSymbol):#cropping image region and checking status of that region
    croppedSymbol = image[rSymbol[1]:rSymbol[2],rSymbol[0]:rSymbol[3]]
    isSymbolOn = checkStatus(croppedSymbol)
    return isSymbolOn

def load_graph(model_file): #loading the graph file
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

#input size setting
myinput_width = 96
myinput_height = 96

def read_tensor_from_image_file(
        image,
        input_height=myinput_height,
        input_width=myinput_width,
        input_mean=0,
        input_std=255):
    # Numpy array
    np_image_data = np.asarray(image)
    np_image_data = np.stack((np_image_data,)*3, -1)
    float_caster = tf.cast(np_image_data, tf.float32)
    # maybe insert float convertion here - see edit remark!
    dims_expander = tf.expand_dims(float_caster, 0)
    input_name = "file_reader"
    output_name = "normalized"
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file): #loading labels
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

    input_layer = "input" #input layer name of the neural network
    output_layer = "MobilenetV2/Predictions/Reshape_1" #output layer name of the neural network
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
        self.image_sub = rospy.Subscriber("imgConnection", Image, self.callback) #receiving the scene frame


    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            #gray = cv.cvtColor(cv_image,cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(cv_image,(5,5),0)
            #Otsu's thresholding after Gaussian filtering
            ret3,resimg = cv.threshold(resimg,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            #Hour
            rh1 = [int(DigitCoordinates.firsthourupleft[0]), int(DigitCoordinates.firsthourupleft[1]), int(DigitCoordinates.firsthourbottomleft[1]), int(DigitCoordinates.firsthourupright[0])]
            rh2 = [int(DigitCoordinates.secondhourupleft[0]), int(DigitCoordinates.secondhourupleft[1]), int(DigitCoordinates.secondhourbottomleft[1]), int(DigitCoordinates.secondhourupright[0])]
            rh3 = [int(DigitCoordinates.thirdhourupleft[0]), int(DigitCoordinates.thirdhourupleft[1]), int(DigitCoordinates.thirdhourbottomleft[1]), int(DigitCoordinates.thirdhourupright[0])]

            imCropHour1 = resimg[rh1[1]:rh1[2],rh1[0]:rh1[3]]
            imCropHour2 = resimg[rh2[1]:rh2[2],rh2[0]:rh2[3]]
            imCropHour3 = resimg[rh3[1]:rh3[2],rh3[0]:rh3[3]]

            hour = {}
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

            time = int(str(hour[0]) + str(hour[1]) + str(hour[2]))
            msg.User_Select_Delay_Time = time
            print(msg.User_Select_Delay_Time)
            #time = '0'+ str(hour[0])# + ':' + str(hour[1]) + str(hour[2])
            #myhour = datetime.datetime.strptime(time,'%H:%M').time()
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


    rh1 = [int(DigitCoordinates.firsthourupleft[0]), int(DigitCoordinates.firsthourupleft[1]), int(DigitCoordinates.firsthourbottomleft[1]), int(DigitCoordinates.firsthourupright[0])]
    rh2 = [int(DigitCoordinates.secondhourupleft[0]), int(DigitCoordinates.secondhourupleft[1]), int(DigitCoordinates.secondhourbottomleft[1]), int(DigitCoordinates.secondhourupright[0])]
    rh3 = [int(DigitCoordinates.thirdhourupleft[0]), int(DigitCoordinates.thirdhourupleft[1]), int(DigitCoordinates.thirdhourbottomleft[1]), int(DigitCoordinates.thirdhourupright[0])]



if __name__ == "__init__":
    global DigitCoordinates
    global rh1

if __name__ == "__main__":
    #model graph file and label file directories
    model_file = "/home/ismayil/catkin_ws/src/thesispro/resources/models/Slim/frozen_mobilenetV2_1.0_96_gray.pb"
    label_file = "/home/ismayil/catkin_ws/src/thesispro/resources/models/Slim/labels.txt"
    input_height = myinput_height
    input_width = myinput_width
    input_mean = 0
    input_std = 255

    #ROS PUBLISHER
    pub = rospy.Publisher('hour_info', screen, queue_size = 1)
    rospy.init_node('hour_info', anonymous = True)
    msg = screen()
    #rate = rospy.Rate(10) # 10hz
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
