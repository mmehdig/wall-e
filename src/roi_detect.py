#!/usr/bin/env python

import roslib
import rospy
import cv
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy

bridge = CvBridge()

cv2_img_rgb = None
background = None
rgbd = dict()


def image_callback(data):
    global cv2_img_rgb
    cv2_img_rgb = bridge.imgmsg_to_cv2(data, "bgr8")


def depth_callback(data):
    global rgbd
    global background

    cv2_img_depth = bridge.imgmsg_to_cv2(data, "32FC1")

    for y, row in enumerate(cv2_img_rgb):
        for x, color in enumerate(row):
            z = cv2_img_depth[y][x]
            z = z[0]
            rgbd[(x, y)] = ((color[0], color[0], color[0], float(z)))
            print rgbd[(x, y)]
    if not background:
        background = rgbd.copy()

# def detect_object:


if __name__ == '__main__':
    rospy.init_node("walle-roi_detect")

    """ Subscribe to the raw camera image topic and set the image processing callback """
    rospy.Subscriber("/camera/rgb/image_color", Image, image_callback, queue_size=1)
    rospy.Subscriber("/camera/depth/image", Image, depth_callback, queue_size=1)


    print "hello, hello"
    rospy.spin()