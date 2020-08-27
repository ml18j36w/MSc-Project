## About the Project

This project contains packages for use with ROS that enable a robot to detect a person and follow them in real time through a known environment. Obstacle avoidance is also incorporated into the person following module, and a safe distance between the robot and the person being followed is maintained at all times.

## Installing the package

Full installation instructions can be found at https://gitlab.com/sc16sw/person_follower_robot/wikis/Installation-Process

## Running the Software

# Prerequisites

*  In order for spatial transformations to work during pose estimation, a map is required. Therefore, the environment must be mapped.

*  The project requires that subscriptions to both a colour image topic and a depth image topic are available, such as those provided by RGB-D cameras.

*  The catkin workspace must have already been built using `catkin_build`

*  Before running a module, the catkin workspace must be sourced. This can be done by navigating to the top level of the workspace and entering the following command:
    ```
    source devel/setup.bash
    ```

    This must be done for each new terminal. Alternatively, you can add the above command to `.bashrc` which will ensure that any new terminals do not require the catkin workspace to be sourced.

# How to run the software

In order to use the project, two different nodes must run as follows:

1. The first node that is required is the person detection module. In order to run the person detection module, open a terminal and enter the following command:
```
rosrun person_follower human_detection.py
```

2. The second node that is required is the person follower module. In order to run the person follower module, open a terminal and enter the following command:
```
rosrun person_follower follower.py
```

The person detection module publishes a `detections` topic, over which messages containing the centre coordinate of every detection can be accessed.

The person follower module subscribes to the `detections` topic in order to receive the detection messages. The centre points of detections are then used for pose estimation, and the trajectory that the robot should follow is calculated.
