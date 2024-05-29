#!/usr/bin/env python

import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String
import sys

class RealTimeMoveGroupInterface:
    def __init__(self):
        super(RealTimeMoveGroupInterface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('real_time_move_group', anonymous=True)

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        group = moveit_commander.MoveGroupCommander(group_name)

        self.robot = robot
        self.scene = scene
        self.group = group

        rospy.Subscriber("/target_pose", geometry_msgs.msg.Pose, self.move_to_pose)

    def move_to_pose(self, target_pose):
        group = self.group
        group.set_pose_target(target_pose)

        plan = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        current_pose = group.get_current_pose().pose
        if not self.all_close(target_pose, current_pose, 0.01):
            rospy.logwarn("Target pose not reached")

    def all_close(self, goal, actual, tolerance):
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False
        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)
        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
        return True

if __name__ == '__main__':
    try:
        interface = RealTimeMoveGroupInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

