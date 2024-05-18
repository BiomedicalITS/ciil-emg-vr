FROM ros:humble

# install ros package
RUN apt-get update && apt-get install -y \
    build-essential \
    ros-${ROS_DISTRO}-ur \
    ros-${ROS_DISTRO}-demo-nodes-cpp \
    ros-${ROS_DISTRO}-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

# launch ros package
#CMD ["ros2", "launch", "ur_robot_driver", "ur_control.launch.py", "ur_type:=ur5e", "robot_ip:=172.22.22.1", "use_fake_hardware:=true", "launch_rviz:=true"]
CMD ["ros2", "launch", "ur_robot_driver", "ur_control.launch.py", "ur_type:=ur5e", "robot_ip:=172.22.22.1"]
# is "reverse_ip:=<ipp of docker host>" necessary?? https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/348#issuecomment-1180105662