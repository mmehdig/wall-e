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
from math import sqrt, isnan

bridge = CvBridge()

cv2_img_rgb = None
background = []
background_avg = None
rgbd = dict()
current_view = []

def image_callback(data):
    global cv2_img_rgb
    cv2_img_rgb = bridge.imgmsg_to_cv2(data, "bgr8")


def depth_callback(data):
    global rgbd
    global background
    global current_view

    if len(background) < 30 or not captureImage:
        cv2_img_depth = bridge.imgmsg_to_cv2(data, "32FC1")

        for y, row in enumerate(cv2_img_rgb):
            for x, color in enumerate(row):
                z = cv2_img_depth[y][x]
                z = z[0]
                rgbd[(x, y)] = ((color[0], color[0], color[0], float(z)))
        if len(background) < 30:
            background.append(rgbd.copy())
			print "Background step ", len(background)
        elif len(current_view) < 30:
            currentFrame = rgbd.copy()
            captureImage = False
            print "Image step", len(current_view)
            find_object()

def average_bkg_and_cv(lst):
        toReturn = dict()
        for x in range(0,640):
                for y in range(0,480):
                	summed = (0,0,0,0)
                	for i in lst[(x,y)]:
                		summed = (summed[0]+lst[0], summed[1] + lst[1], summed[2] + lst[2], summed[3]+lst[3])
                        toReturn[(x,y)] = float(summed/len(lst[(x,y)])) 
        return toReturn

def calc_foreground(bkg, cv):
        toReturn = dict()
        for x in range(0,640):
                for y in range(0,480):
                        toReturn[(x,y)] = (bkg[0] - cv[0], bkg[1] - cv[1], bkg[2] - cv[2], bkg[3] - cv[3])

def average_depths(lst):
        sum = 0
        for i in lst:
                sum += float(lst[3])
        return sum/len(lst)

def find_object():
        global background_avg
        if not background_avg:
                background_avg = average_bkg_and_cv(background)
        current_view_avg = average_bkg_and_cv(current_view)
        foreground = calc_foreground(background_avg, current_view_avg)
        print foreground[(232,263)]
	
	horizontal_diffs = []
        for x in range[10,640]:
                for y in range[0,480]:
                        this_foreground = []
                        for i in range(0,10):
                                this_foreground.append(foreground[(x-i,y])
                        print average_depths(this_foreground)
                        if average_depths(this_foreground) > 0.1:
                                horizontal_diffs.append((x-5),y)
        print horizonal_diffs
	

if __name__ == '__main__':
    rospy.init_node("walle-roi_detect")

    """ Subscribe to the raw camera image topic and set the image processing callback """
    rospy.Subscriber("/camera/rgb/image_color", Image, image_callback, queue_size=1)
    rospy.Subscriber("/camera/depth/image", Image, depth_callback, queue_size=1)
    print "Started!"


    captureImage = False
    rospy.spin()
