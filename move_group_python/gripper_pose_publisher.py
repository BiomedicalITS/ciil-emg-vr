'''
#!/usr/bin/env python

import rospy
import geometry_msgs.msg

def publisher():
    rospy.init_node('target_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/target_pose', geometry_msgs.msg.Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        pose = geometry_msgs.msg.Pose()
        pose.position.x = 0.4  # Update with the desired position
        pose.position.y = 0.1  # Update with the desired position
        pose.position.z = 0.4  # Update with the desired position
        pose.orientation.w = 1.0

        rospy.loginfo(f"Publishing new target pose: {pose}")
        pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
'''

#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import geometry_msgs.msg

def publisher():
    rospy.init_node('target_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/target_pose', geometry_msgs.msg.Pose, queue_size=10)
    grip_pub = rospy.Publisher('/gripper_command', String, queue_size = 10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Read position from user input
        x = float(input("Enter x position: "))
        y = float(input("Enter y position: "))
        z = float(input("Enter z position: "))
        gripper = input("Enter gripper position: ").strip()

        # Create a new Pose message
        pose = geometry_msgs.msg.Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0  # Assuming the same orientation for simplicity

        # Log and publish the pose
        rospy.loginfo(f"Publishing new target pose: {pose}")
        pub.publish(pose)
        grip_pub.publish(gripper)

        # Sleep to maintain the rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass