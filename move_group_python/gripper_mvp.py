#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from robotiq_hande_ros_driver.srv import gripper_service
import socket
import json

if __name__ == "__main__":
    rospy.loginfo("using service")
    gripper_srv = rospy.ServiceProxy("gripper_service", gripper_service)
    rospy.init_node("gripper_test_node")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("0.0.0.0", 5112))
        while True:
            cmd = json.loads(s.recv(2048))
            print(cmd)

            if cmd["action"] == "open":
                # open gripper
                response = gripper_srv(position=0, speed=255, force=25)
            elif cmd["action"] == "close":
                # close gripper
                response = gripper_srv(position=255, speed=255, force=25)
