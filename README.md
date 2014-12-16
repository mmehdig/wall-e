wall-e
======

Project in Embodied and Situated Language Processing


Installation
======

First, you need to create a workspace, if you have a workspace you can go ti `src` folder and skip this part:

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

Then, you can download the package, and compile it:

```
git clone https://github.com/mmehdig/wall-e.git walle
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

