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
        input_width=myinput_height,
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
            ret3,resimg = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            #Kilogram
            rKg1 = [int(DigitCoordinates.firstkgupleft[0]), int(DigitCoordinates.firstkgupleft[1]), int(DigitCoordinates.firstkgbottomleft[1]), int(DigitCoordinates.firstkgupright[0])]
            rKg2 = [int(DigitCoordinates.secondkgupleft[0]), int(DigitCoordinates.secondkgupleft[1]), int(DigitCoordinates.secondkgbottomleft[1]), int(DigitCoordinates.secondkgupright[0])]

            imCropKg1 = resimg[rKg1[1]:rKg1[2],rKg1[0]:rKg1[3]]
            imCropKg2 = resimg[rKg2[1]:rKg2[2],rKg2[0]:rKg2[3]]

            kilogram = {}
            if checkStatus(imCropKg1)==True:
                kilogram[0] = classification(imCropKg1,self.graph,self.labels)
            else:
                kilogram[0] = 0
            if checkStatus(imCropKg2)==True:
                kilogram[1] = classification(imCropKg2,self.graph,self.labels)
            else:
                kilogram[1] = 0
            kg = float(str(kilogram[0])+'.'+str(kilogram[1]))
            msg.Kilogram = kg
            print(msg.Kilogram)
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

    rh1 = [int(DigitCoordinates.firstkgupleft[0]), int(DigitCoordinates.firstkgupleft[1]), int(DigitCoordinates.firstkgbottomleft[1]), int(DigitCoordinates.firstkgupright[0])]
    rh2 = [int(DigitCoordinates.secondkgupleft[0]), int(DigitCoordinates.secondkgupleft[1]), int(DigitCoordinates.secondkgbottomleft[1]), int(DigitCoordinates.secondkgupright[0])]




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
    #Codes for parallel programming --------------------------------------------
    #p1 = Process(target=classificationTemperature)
    #p1.start()
    #p2 = Process(target=classificationCentrifugation)
    #p2.start()
    #p1.join()
    #p2.join()
    #---------------------------------------------------------------------------

    #ROS PUBLISHER
    pub = rospy.Publisher('kg_info', screen, queue_size = 1)
    rospy.init_node('kg_info', anonymous = True)


    msg = screen()
    rate = rospy.Rate(10) # 10hz
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
