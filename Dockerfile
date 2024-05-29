FROM ros:noetic

SHELL ["/bin/bash", "-c"]
WORKDIR /root/

# install ros package
RUN apt-get update && apt-get install -y \
    git \
    inetutils-ping \
    build-essential

#RUN rm -rf /var/lib/apt/lists/*

# http://wiki.ros.org/catkin/Tutorials/create_a_workspace
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    mkdir -p catkin_ws/src && \
    cd catkin_ws && \
    catkin_make && \
    echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc && \
    cd src && catkin_create_pkg move_group_python rospy geometry_msgs

ADD move_group_python catkin_ws/src/move_group_python/launch

RUN cd catkin_ws && \
    source devel/setup.bash && \
    catkin_make

# TODO doesnt work yet. Gotta build the package etc
