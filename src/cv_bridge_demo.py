#!/usr/bin/env python

""" cv_bridge_demo.py - Version 0.1 2011-05-29

    A ROS-to-OpenCV node that uses cv_bridge to map a ROS image topic and optionally a ROS
    depth image topic to the equivalent OpenCV image stream(s).
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2011 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""

import roslib;# roslib.load_manifest('rbx1_vision')
import rospy
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import String

class cvBridgeDemo():
    def __init__(self):
        self.node_name = "walle_recognizer"
        
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.node_name
        cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        cv.MoveWindow(self.cv_window_name, 25, 75)
        
        # And one for the depth image
        cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Depth Image", 25, 350)
        self.known_objects = []
        self.sift = cv2.SIFT()

        self.current_depth = None
        self.current_color = None

        # Create the cv_bridge object
        self.bridge = CvBridge()

        self.pub = rospy.Publisher("/walle/recognizer/publish", String)
        rospy.Subscriber("/walle/recognizer/listen", String, self.receive, queue_size=1)
        self.rate = rospy.Rate(10) # 10hz
        
        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e
        
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(frame, dtype=np.uint8)
        
        # Process the frame using the process_image() function
        display_image = self.process_image(frame)
                       
        # Display the image.
        cv2.imshow(self.node_name, display_image)
        
        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        self.current_color = display_image
        if self.keystroke > 0:
            print self.keystroke
            cc = chr(self.keystroke & 255).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")
            if cc == 'i':
                self.recognize_object(display_image)
            if cc == 'l':
                self.learn_new_object(display_image)
                # save the features to the database

    def learn_new_object(self, frame, name):
        kp, des = self.sift.detectAndCompute(frame,None)
        self.known_objects.append((name, kp, des))
        print 'ok, %s' % name
        self.send('ok, %s' % name)

    def recognize_object(self,frame):

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        kp, des = self.sift.detectAndCompute(frame,None)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        scores = []
        for name, kp2, des2 in self.known_objects:
            matches = flann.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            if len(good) > 10:
                scores.append((len(good), name))
                print len(good), name
        if scores:
            self.send(str(sorted(scores)[-1]))
        else:
            self.send("I don't know")

    def depth_callback(self, ros_image):
        #Use cv_bridge() to convert the ROS image to OpenCV format
        try:
        #    The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "16UC1")
        except CvBridgeError, e:
             print e
        #
        #Convert the depth image to a Numpy array since most cv2 functions

        depth_array = np.array(depth_image, dtype=np.float32)
        self.current_depth = np.copy(depth_array)
        max_of_frame = 1000
        depth_array[depth_array < 1] = max_of_frame
        depth_array[depth_array > 1000] = max_of_frame

        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

        # Process the depth image
        depth_display_image = self.process_depth_image(depth_array)
    
        # Display the result
        cv2.imshow("Depth Image", depth_display_image)
          
    def process_image(self, frame):
        # Convert to greyscale
        grey = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
        #print frame[self.current_depth > 0.9]
        frame[np.tile(self.current_depth > 1000, (1, 1, 3))] = 255
        frame[np.tile(self.current_depth < 1, (1, 1, 3))] = 255
        return self.orb_function(frame)

    def orb_function(self, frame):
        #img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.FeatureDetector_create("ORB")
        descriptor = cv2.DescriptorExtractor_create("ORB")
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

        # find the keypoints with ORB
        kp = orb.detect(frame)

        # compute the descriptors with ORB
        kp, des = descriptor.compute(frame, kp)

        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(frame,kp,color=(0,255,0), flags=0)
        return img2


    def good_Features(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
        try:
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(img,(x,y),3,255,-1)
        except:
            pass
        return img

    def process_depth_image(self, frame):
        return self.good_Features(frame)

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()

    def send(self, data):
        print "sending", data
        self.pub.publish(data)
        self.rate.sleep()

    def receive(self, msg):
        if isinstance(msg, String):
            data = msg.data
        else:
            data = msg
        if data[:8] == "this is ":
            print data[8:]
            self.learn_new_object(self.current_color, data[8:])
        elif data == "what is this?":
            print 'finding out what this is..'
            self.recognize_object(self.current_color)
        else:
            print "I don't understand", data


def main(args):
    print("hello!")

    try:
        cvBridgeDemo()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    
