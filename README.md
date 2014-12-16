wall-e
======

A Project in Embodied and Situated Language Processing. ([The course webpage](http://gul.gu.se/public/courseId/65966/lang-en/publicPage.do?item=27403175))


Requirments
======

* (We have tested this package on Ubuntu 12.04 with ROS-groovy)
* [freenect_stack](http://wiki.ros.org/freenect_stack)


Installation
======

First, you need to create a workspace, if you have a workspace you can go to `src` folder and skip this part:

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

Then, you can download the package, and compile it:

```
git clone https://github.com/mmehdig/wall-e.git walle
```

If you don't have [kinect_aux](http://wiki.ros.org/kinect_aux), you can download it here. First you should have required packages:

```
sudo apt-get install libusb-1.0-0 libusb-1.0-0-dev
git clone https://github.com/muhrix/kinect_aux.git
```



```
cd ..
catkin_make
```

Now make sure that environment is ready for `rosrun`

```
 source devel/setup.bash 
```


Run
======
First, you need to run freenect in one terminal:

```
roslaunch freenect_launch freenect.launch
```

Then, in another terminal:

```
rosrun walle roi_detect.py
```

To adjust Kinect head by moving the motor:

```
rosrun walle kinect_adjust.py
```
