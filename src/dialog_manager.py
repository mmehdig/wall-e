#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def say(msg):
    """
    if data == "ok, ..."
    if data == ???
    """
    data = msg.data

    if data[:3] == "ok,":
        print data
    elif data:
        # TODO: decide about form of string data ...
        print "I think", data

if __name__ == '__main__':
    rospy.init_node("walle_dm")
    rospy.Subscriber("/walle/dialog/listen", String, say, queue_size=1)
    rospy.Subscriber("/walle/dialog/thoughts", String, say, queue_size=1)

    rospy.spin()