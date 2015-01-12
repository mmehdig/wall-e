#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def listen(msg):
    """
    process the message if it needs any process...
    """
    # TODO: you can remove this line if you don't like echos in dialog manager
    say("user said \"" + msg.data + "\"")


def say(msg):
    """
    if data == "ok, ..."
    if data == ???
    """
    if isinstance(msg, String):
        data = msg.data
    else:
        data = msg

    if data[:3] == "ok,":
        print data
    elif data:
        # TODO: decide about form of string data ...
        print "I think,", data

if __name__ == '__main__':
    rospy.init_node("walle_dm")
    rospy.Subscriber("/walle/dialog/listen", String, listen, queue_size=1)
    rospy.Subscriber("/walle/dialog/thoughts", String, say, queue_size=1)



    rospy.spin()