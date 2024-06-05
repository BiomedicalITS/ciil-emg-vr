#!/usr/bin/env python

import rospy
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import Int64
from robotiq_hande_ros_driver.srv import gripper_service
import sys
import copy
import threading
import math

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

        self.current_arm_direction = None
        self.arm_lock = threading.Lock()
        
        self.current_wrist_direction = None 
        self.wrist_lock = threading.Lock()

        rospy.Subscriber("/arm_command", Int64, self.update_arm_direction)
        rospy.Subscriber("/gripper_command", Int64, self.control_gripper)
        rospy.Subscriber("/wrist_command", Int64, self.update_wrist_direction)

        self.arm_movement_thread = threading.Thread(target=self.control_arm)
        self.arm_movement_thread.start()
        
        self.wrist_movement_thread = threading.Thread(target=self.control_wrist)
        self.wrist_movement_thread.start()

    def update_arm_direction(self, direction):
        with self.arm_lock:
            self.current_arm_direction = direction.data

    def control_arm(self):
        rate = rospy.Rate(10)  # 10 Hz
        movement = 0.1

        while not rospy.is_shutdown():
            if self.current_arm_direction:
                with self.arm_lock:
                    direction = self.current_arm_direction

                group = self.group
                waypoints = []

                wpose = group.get_current_pose().pose
                if direction == 5:
                    wpose.position.z += movement
                elif direction == 6:
                    wpose.position.z -= movement
                elif direction == 3:
                    wpose.position.x += movement
                elif direction == 4:
                    wpose.position.x -= movement
                elif direction == 1:
                    wpose.position.y += movement
                elif direction == 2:
                    wpose.position.y -= movement
                elif direction == 0:
                    wpose = wpose
                else:
                    rospy.logwarn("Unknown arm command")
                    continue

                waypoints.append(copy.deepcopy(wpose))

                (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
                group.execute(plan, wait=False)  # Non-blocking execution

            rate.sleep()
         
    def update_wrist_direction(self, direction):
        with self.wrist_lock:
            self.current_wrist_direction = direction.data
            
    def control_wrist(self):
        rate = rospy.Rate(10) #10 Hz
        movement = math.pi / 4
        
        while not rospy.is_shutdown():
            if self.current_wrist_direction:
                with self.wrist_lock:
                    direction = self.current_wrist_direction
    
                group = self.group
                joint_goal = group.get_current_joint_values()
        
                if direction == 0:
                    joint_goal = joint_goal
                elif direction == 1:
                    joint_goal[3] += movement
                elif direction == 2:
                    joint_goal[3] -= movement
                elif direction == 3:
                    joint_goal[4] -= movement
                elif direction == 4:
                    joint_goal[4] += movement
                else:
                    rospy.logwarn("Unknown wrist command")
                    continue
        
                group.go(joint_goal, wait=False)
                group.stop()
                
            rate.sleep()
        
    def control_gripper(self, command):
        command = command.data
        if command == 0:
            self.gripper_srv(position=255, speed=255, force=255)
        elif command == 1:
            self.gripper_srv(position=255, speed=255, force=255)
        elif command == 2:
            self.gripper_srv(position=0, speed=255, force=255)
        else:
            rospy.logwarn("Unknown gripper command")
            

if __name__ == '__main__':
    try:
        interface = RealTimeMoveGroupInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
