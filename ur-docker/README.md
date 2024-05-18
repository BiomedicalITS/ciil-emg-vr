# README

This directory is used to build the custom ROS UR5 Docker and manage it.

See the `ur_robot_controller` [documentation](https://docs.ros.org/en/ros2_packages/humble/api/ur_robot_driver/index.html).

General ROS2 Docker [tutorial](https://wiki.ros.org/docker/Tutorials/Docker).

## Setting up

First, follow the instructions in the linked documentation above. Then, from the project's root:

- `cd ur-docker`
- Build the Docker image: `./build-docker.sh`
- Start a Docker container: `./start-docker.sh`
- `cd ..`

## Running ROS nodes in Docker

See [this page](https://docs.ros.org/en/humble/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html).

## Using a bare ros Docker image

This can be useful for example to test if the ur Docker is working properly.

```bash
docker run -it --rm ros:humble bash

apt update; apt install -y
    ros-humble-ros2-control
    ros-humble-ros2-controllers
    ros-humble-ur
```
