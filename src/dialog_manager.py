#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import json


def language_understanding(text):
    """
    Natural Language Understanding:
    natural language sentences should be converted to known commands and robot language:
        learn: x
        what is this?

    :param text: user input text
    :type text: str
    :return: str
    """

    # normalized:
    text = text.lower()\
        .replace("?", "")\
        .strip()

    if text[:8] == "this is ":
        text = "learn: " + text[8:]
    elif text[:6] == "learn ":
        text = "learn: " + text[6:]
    elif text[:13] == "this item is " or text[:13] == "now, you see ":
        text = "learn: " + text[13:]

    if text[:7] == "learn: ":
        if text[6:11] == " not ":
            text = "not learn: " + text[11:]
        elif text[6:9] == " a ":
            text = "learn: " + text[9:]
        elif text[6:10] == " an ":
            text = "learn: " + text[10:]
        elif text[6:11] == " the ":
            text = "learn: " + text[11:]
        else:
            text = "learn: " + text[7:]

    elif text in ["what about this", "who is this", "what is this", "what do you know about this"]:
        text = "what is this?"

    data = text
    return data


def language_generation(data):
    """
    :param data: json format string, comes from robot
    :type data: str
    :return: str
    """
    data = json.load(data)

    if 'detected' in data:
        if data['detected'] is None:
            text = "I have no idea!"
        else:
            text = "It is " + data['detected'][0].keys()[0]
    elif 'ok' in data:
        text = "ok, " + data['ok']
    return text


def received(msg):
    """
    :type msg: String
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
    text = language_generation(msg.data)

    print text


def listen(text):
    """
    function used for sending data, only strings should be sent!
    it will convert user input to robot language before sending to robot.
    """
    # language understanding
    data = language_understanding(text)
    print data
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
