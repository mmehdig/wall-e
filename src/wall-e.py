#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import rospy
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import codecs


class Recognizer():
    def __init__(self):
        self.demo_on = False
        self.node_name = "walle_recognizer"
        
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.node_name
        cv.NamedWindow("RGB Image", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("RGB Image", 25, 75)
        
        # And one for the depth image
        cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Depth Image", 25, 350)
        self.known_objects = []
        self.sift = cv2.ORB()

        self.current_depth = None
        # self.current_color = None
        self.current_stage = None

        # Create the cv_bridge object
        self.bridge = CvBridge()

        self.pub = rospy.Publisher("/walle/recognizer/publish", String)
        self.rate = rospy.Rate(10)   # 10hz

        self.last_recognized = None

        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        rospy.Subscriber("/walle/recognizer/listen", String, self.receive, queue_size=1)

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
        cv2.imshow("RGB Image", display_image)

        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        if self.keystroke > 0:
            cc = chr(self.keystroke & 255).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")
            if cc == 'i':
                self.recognize_object(display_image)
            if cc == 'l':
                self.learn_new_object(display_image)
                # save the features to the database

        if self.current_stage is not None:
            method = self.current_stage[0]
            arguments = [display_image] + self.current_stage[1:]
            method(*arguments)
            self.current_stage = None


    def depth_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
        #    The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "16UC1")
        except CvBridgeError, e:
             print e
        #
        # Convert the depth image to a Numpy array since most cv2 functions

        depth_array = np.array(depth_image, dtype=np.float32)
        self.current_depth = np.copy(depth_array)

        max_depth = 1000
        depth_array[depth_array < 1] = max_depth
        depth_array[depth_array > 1000] = max_depth

        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

        # Process the depth image
        depth_display_image = self.process_depth_image(depth_array)
    
        # Display the result
        cv2.imshow("Depth Image", depth_display_image)

    def process_depth_image(self, frame):
        return frame
        # return self.good_Features(frame)

    def process_image(self, frame):
        # Convert to greyscale
        # grey = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
        #print frame[self.current_depth > 0.9]

        # region of interest is 1 to 1000 millimeters distance from camera
        # set all not interesting points white (255,255,255):
        frame[np.tile(self.current_depth > 1000, (1, 1, 3))] = 255
        frame[np.tile(self.current_depth < 1, (1, 1, 3))] = 255

        # show SIFT features
        if self.demo_on:
            kp, des = self.sift.detectAndCompute(frame, None)
            frame = cv2.drawKeypoints(frame, kp, color=(0, 255, 0), flags=0)

        #return self.orb_function(frame)
        return frame

    # def orb_function(self, frame):
    #     #img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #     #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     orb = cv2.FeatureDetector_create("ORB")
    #     descriptor = cv2.DescriptorExtractor_create("ORB")
    #     matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    #
    #     # find the keypoints with ORB
    #     kp = orb.detect(frame)
    #
    #     # compute the descriptors with ORB
    #     kp, des = descriptor.compute(frame, kp)
    #
    #     # draw only keypoints location,not size and orientation
    #     img2 = cv2.drawKeypoints(frame,kp,color=(0,255,0), flags=0)
    #     return img2

    # def good_Features(self, frame):
    #     img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #
    #     gray = np.float32(gray)
    #     corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
    #     try:
    #         corners = np.int0(corners)
    #         for i in corners:
    #             x,y = i.ravel()
    #             cv2.circle(img,(x,y),3,255,-1)
    #     except:
    #         pass
    #     return img

    def learn_new_object(self, frame, name):
        kp, des = self.sift.detectAndCompute(frame, None)
        self.known_objects.append((name, kp, des))
        self.send(u'{"ok":"%s"}' % name)

    def reinforce_object(self, name):
        self.known_objects.append((name, self.last_recognized[0], self.last_recognized[1]))
        self.send(u'{"ok":"%s"}' % name)

    def recognize_object(self, frame):
        # extract sift features:
        kp, des = self.sift.detectAndCompute(frame, None)
        self.last_recognized = (kp, des)
        # setup FLANN parameters (Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        scores = []
        for name, kp2, des2 in self.known_objects:
            matches = flann.knnMatch(des, des2, k=2)
            print len(des), len(des2)
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            if len(good) > 10:
                scores.append((len(good)*1.0/len(des2), name))

                # debugging, print number of good matching keypoints for each known object
                print len(good), name

        # send out thoughts about object recognition
        if scores:
            self.send(self.jsonify(sorted(scores, reverse=True), "detected"))
        else:
            self.send(u'{"detected":null}')

    def jsonify(self, data, name):
        if isinstance(data, list):
            return u'{"' + name + u'":[' + u",".join(map(lambda t: u'{"%s":%f}' % t, data)) + u']}'

        return u'{"error":500}'

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()

    def send(self, data):
        # debugging:
        print "sending >>>", data

        self.pub.publish(data)
        self.rate.sleep()

    def receive(self, msg):
        if isinstance(msg, String):
            data = msg.data
        else:
            data = msg

        data = codecs.decode(data, 'utf-8')
        # debugging:
        print "received <<<", data

        if data[:7] == u"learn: ":
            print data[7:]
            # self.send(u'{"ok":"%s"}' % data[7:])

            self.current_stage = [self.learn_new_object, data[8:]]
            # self.learn_new_object(self.current_color, data[8:])

        elif data == u"what is this?":
            print u'finding out what this is..'
            # scores = [(u'häst', 0.456789), (u'book', 0.1234)]
            # self.send(self.jsonify(sorted(scores, reverse=True), "detected"))

            self.current_stage = [self.recognize_object]
            # self.recognize_object(self.current_color)
        elif data == u"last":
            self.current_stage = [self.reinforce_object]
        else:
            print u"I don't understand", data


def main(args):
    print("Running the recognizer ...!")

    try:
        Recognizer()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
