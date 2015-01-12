#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def listen(data):
    pub.publish(data)
    rate.sleep()

if __name__ == '__main__':
    rospy.init_node("walle_ear")
    rate = rospy.Rate(10) # 10hz

    pub = rospy.Publisher("/walle/dialog/listen", String)

    try:
        while not rospy.is_shutdown():
            user_input = raw_input("Just say whatever you want:\n")
            listen(user_input)
    except rospy.ROSInterruptException:
        pass
