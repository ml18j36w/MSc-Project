## About the Project

This project contains packages for use with ROS that enable a robot to welcoming visitors in real time through a known environment.

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

```
rosrun person_follower welcoming.py
```
The robot will recognise differnt visitors and interact with users by command.


