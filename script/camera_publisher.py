#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

if __name__ == "__main__":

    rospy.init_node('VideoPublisher', anonymous=True)
    bridge = CvBridge()
    VideoRaw = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=2)
    rate = rospy.Rate(1)
    cam = cv.VideoCapture('/home/ismayil/catkin_ws/src/ui_interpretation/Data/images/video.avi')
    if(cam.isOpened()==False):
        print("Error opening video stream of file")
    while (cam.isOpened()):
        meta, frame = cam.read()
        if meta == True:
            
            try:
                msg_frame = bridge.cv2_to_imgmsg(frame)
                VideoRaw.publish(msg_frame, "bgr8")
            except CvBridgeError as e:
                print(e) 
            cv.imshow("goruntu", frame)
            cv.waitKey(3)       
            #rate.sleep()

    cv.release()
    cv.DestroyAllWindows()
  
