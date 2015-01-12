#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def received(msg):
    """
    msg.data contains the response. Responses currently are:
        the string (!!!) (x, 'y'), e.g. "269, 'a newspaper')"
        where x is an int, the number matched, and y is a String with the name that was given to the object
        I'd say x < 40 is unsure. It does not return anything lower than 10, and it always only returns one object.
        Support for multiple objects (list of tuples of int and str) would be appreciated, if guesses are close.

        the string 'I don't know', when kinect can not figure out what the object is (currently means x < 10, this
        may be changed later)

        the string 'ok, a newspaper'
    """
    print "received: \"" + msg.data + "\""



def listen(data):
    """function used for sending data, only strings should be sent!
    currently supported:
        this is x
        what is this?
    """
    pub.publish(data)
    rate.sleep()

if __name__ == '__main__':
    rospy.init_node("walle_dm")
    rospy.Subscriber("/walle/recognizer/publish", String, received, queue_size=1)
    rate = rospy.Rate(10) # 10hz

    pub = rospy.Publisher("/walle/recognizer/listen", String)

    try:
        while not rospy.is_shutdown():
            user_input = raw_input("Talk to kinect:\n")
            listen(user_input)
    except rospy.ROSInterruptException:
        pass


    rospy.spin()
