#!/usr/bin/env python

import rospy
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import String
from robotiq_hande_ros_driver.srv import gripper_service
import sys
import copy
import threading

class RealTimeMoveGroupInterface:
    def __init__(self):
        super(RealTimeMoveGroupInterface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('real_time_move_group', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        self.gripper_srv = rospy.ServiceProxy('gripper_service', gripper_service)
        self.gripper_srv.wait_for_service()

        self.current_direction = None
        self.lock = threading.Lock()

        rospy.Subscriber("/arm_command", String, self.update_direction)
        rospy.Subscriber("/gripper_command", String, self.control_gripper)

        self.movement_thread = threading.Thread(target=self.continuous_move)
        self.movement_thread.start()

    def update_direction(self, direction):
        with self.lock:
            self.current_direction = direction.data

    def continuous_move(self):
        rate = rospy.Rate(10)  # 10 Hz
        movement = 0.01

        while not rospy.is_shutdown():
            if self.current_direction:
                with self.lock:
                    direction = self.current_direction

                group = self.group
                waypoints = []

                wpose = group.get_current_pose().pose
                if direction == "up":
                    wpose.position.z += movement
                elif direction == "down":
                    wpose.position.z -= movement
                elif direction == "left":
                    wpose.position.y += movement
                elif direction == "right":
                    wpose.position.y -= movement
                elif direction == "forward":
                    wpose.position.x += movement
                elif direction == "backward":
                    wpose.position.x -= movement
                else:
                    rospy.logwarn("Unknown direction command: %s", direction)
                    continue

                waypoints.append(copy.deepcopy(wpose))

                (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
                group.execute(plan, wait=False)  # Non-blocking execution

            rate.sleep()

    def control_gripper(self, command):
        if command.data == "open":
            self.gripper_srv(position=0, speed=255, force=255)
        elif command.data == "close":
            self.gripper_srv(position=255, speed=255, force=255)
        else:
            rospy.logwarn("Unknown gripper command: %s", command.data)

if __name__ == '__main__':
    try:
        interface = RealTimeMoveGroupInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
