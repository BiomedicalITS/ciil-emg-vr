#!/usr/bin/env python

import rospy
import geometry_msgs.msg
import socket
import json


def publisher():
    rospy.init_node("target_pose_publisher", anonymous=True)
    pub = rospy.Publisher("/target_pose", geometry_msgs.msg.Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("0.0.0.0", 5112))
        # s.setblocking(False)

        x0, y0, z0 = 0, 0, 0
        fact = 0.1
        while not rospy.is_shutdown():

            cmd = json.loads(s.recv(2048))
            print(cmd)
            if cmd["direction"] == "forward":
                x0 += fact
            elif cmd["direction"] == "backward":
                x0 -= fact
            elif cmd["direction"] == "up":
                z0 += fact
            elif cmd["direction"] == "down":
                z0 -= fact
            elif cmd["direction"] == "left":
                y0 += fact
            elif cmd["direction"] == "right":
                y0 -= fact

            pose = geometry_msgs.msg.Pose()
            pose.position.x = x0  # Update with the desired position
            pose.position.y = y0  # Update with the desired position
            pose.position.z = z0  # Update with the desired position
            pose.position.x = 0.1
            pose.position.y = 0.0
            pose.position.z = 0.0
            pose.orientation.w = 1.0

            rospy.loginfo(f"Publishing new target pose: {pose}")
            pub.publish(pose)
            rate.sleep()


if __name__ == "__main__":
    try:
        print("HELLO")
        publisher()
    except rospy.ROSInterruptException:
        pass
