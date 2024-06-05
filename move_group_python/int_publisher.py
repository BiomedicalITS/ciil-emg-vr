#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64
import socket
import json

def publisher():
    rospy.init_node('target_pose_publisher', anonymous=True)
    arm_pub = rospy.Publisher('/arm_command', Int64, queue_size = 10)
    grip_pub = rospy.Publisher('/gripper_command', Int64, queue_size = 10)
    wrist_pub = rospy.Publisher('/wrist_command', Int64, queue_size = 10)
    rate = rospy.Rate(10)  # 10 Hz

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("192.168.50.39", 5112))
        s.setblocking(False)
        while not rospy.is_shutdown():
            # Read command from armband
            try:
                cmd = json.loads(s.recv(2048))
                print(cmd)
            except:
                continue
                        
            if "gripper" in cmd.keys():
                gripper_action = cmd["gripper"]
                grip_pub.publish(gripper_action)

            elif "arm" in cmd.keys():
                arm_action = cmd["arm"]
                arm_pub.publish(arm_action)
                
            elif "wrist" in cmd.keys():
                wrist_action = cmd["wrist"]
                wrist_pub.publish(wrist_action)
            else:
                continue

            # Sleep to maintain the rate
            rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass  
