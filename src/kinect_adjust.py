#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64


def move_it(angle):
    if angle != target_angle:
        pub.publish(target_angle)
        print target_angle, angle
    rate.sleep()

if __name__ == '__main__':
    rospy.init_node("walle-kinect_adjust")
    rate = rospy.Rate(10) # 10hz

    try:
        target_angle = input("Give me angle [-31:31] :")
        target_angle = float(target_angle)
    except:
        target_angle = 0

    pub = rospy.Publisher("/tilt_angle", Float64)
    rospy.Subscriber("/cur_tilt_angle", Float64, move_it)

    rospy.spin()