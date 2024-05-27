#!/usr/bin/env python

#
# Code adapted from https://github.com/ros-planning/moveit_tutorials/blob/kinetic-devel/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
# and https://web.archive.org/web/20240317214020/https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
# to UR5e robot and Python3 
#
from math import pi
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def all_close(goal, actual, tolerance):
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

class MoveGroupTutorial(object):
  def __init__(self):
    super(MoveGroupTutorial, self).__init__()

    # Initialize `moveit_commander` and a `rospy` node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_tutorial_ur5e', anonymous=True)

    # Instantiate `RobotCommander` object
    # Provides info such as the robot's kinematic model 
    # and current joint states
    robot = moveit_commander.RobotCommander()

    # Instantiate `PlanningSceneInterface` object
    # Provides remote interface for getting, setting, and updating
    # robot's internal understanding of surroundings
    scene = moveit_commander.PlanningSceneInterface()

    # Instantiate `MoveGroupCommander` object
    # An interface to a planning group
    group_name = "manipulator" # Planning group name for UR5e
    group = moveit_commander.MoveGroupCommander(group_name)

    # Create `DisplayTrajectory` publisher
    # To display trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path",
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    # Get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()
    print("============ Reference frame: %s" % planning_frame)

    # Get the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print("============ End effector: %s" % eef_link)

    # Get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Robot Groups:", robot.get_group_names())

    # Print the entire state of the robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.group = group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names

  def go_to_up_state(self):
    group = self.group

    joint_goal = group.get_current_joint_values()
    print(type(joint_goal), joint_goal)

    # Set joint goal for UR5e:
    joint_goal[0] = 0
    joint_goal[1] = -pi * 0.5
    joint_goal[2] = 0
    joint_goal[3] = -pi * 0.5
    joint_goal[4] = 0
    joint_goal[5] = 0    

    group.go(joint_goal, wait=True)
    group.stop() # Ensures no residual movement

    current_joints = self.group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)

  def go_to_pose_goal(self):
    group = self.group

    current_pose = group.get_current_pose().pose
    print("Current pose: ", current_pose)

    # Set pose goal:
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.1
    pose_goal.position.z = 0.4
    group.set_pose_target(pose_goal)

    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    # For testing:
    current_pose = group.get_current_pose().pose
    print("New current pose: ", current_pose)

    return all_close(pose_goal, current_pose, 0.01)

  def plan_cartesian_path(self, scale=1):
    # You can plan Cartesian paths directly by specifying a list of waypoints 
    # for the end-effector to go through

    group = self.group

    current_pose = group.get_current_pose().pose
    print("Current pose: ", current_pose)

    waypoints = []

    wpose = group.get_current_pose().pose 
    wpose.position.z -= scale * 0.1 # First move up (z)
    wpose.position.y += scale * 0.2 # and sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.y += scale * 0.1 # Second move forward/backwards in (x)
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.y -= scale * 0.1 # Third move sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    (plan, fraction) = group.compute_cartesian_path(
                                       waypoints, # waypoints to follow 
                                       0.01,      # eef_step 
                                       0.0)       # jump_threshold 

    # Note: this is just planning, we are not asking move_group to actually
    # move the robot yet
    return plan, fraction

  def display_trajectory(self, plan):
    robot = self.robot
    display_trajectory_publisher = self.display_trajectory_publisher

    # A 'DisplayTrajectory'_msg has two primary fields
    # trajectory_start: populated with current robot state
    # trajectory: we add our plan to this field
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)

    display_trajectory_publisher.publish(display_trajectory)

  def execute_plan(self, plan):
    group = self.group

    # Use execute if you would like the robot to follow the plan that has been
    # computed
    group.execute(plan, wait=True)

  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    box_name = self.box_name
    scene = self.scene

    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      attached_objects = scene.get_attached_objects([box_name])
      is_attached = len(attached_objects.keys()) > 0

      is_known = box_name in scene.get_known_object_names()

      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True

      rospy.sleep(0.1)
      seconds = rospy.get_time()

    return False

  def add_box(self, timeout=4):
    box_name = self.box_name
    scene = self.scene

    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "ee_link"  # Adjust this frame_id if necessary
    box_pose.pose.orientation.w = 1.0
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))

    self.box_name = box_name
    return self.wait_for_state_update(box_is_known=True, timeout=timeout)

  def attach_box(self, timeout=4):
    box_name = self.box_name
    robot = self.robot
    scene = self.scene
    eef_link = self.eef_link
    group_names = self.group_names

    grasping_group = 'gripper'  # Adjust this to your actual gripper group name if necessary
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)

    return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)

  def detach_box(self, timeout=4):
    box_name = self.box_name
    scene = self.scene
    eef_link = self.eef_link

    scene.remove_attached_object(eef_link, name=box_name)

    return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)

  def remove_box(self, timeout=4):
    box_name = self.box_name
    scene = self.scene

    scene.remove_world_object(box_name)

    return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)


def main():
  try:
    print("============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ...")
    input()
    tutorial = MoveGroupTutorial()

    print("============ Press `Enter` to execute a movement using a joint state goal ...")
    input()
    tutorial.go_to_up_state()

    print("============ Press `Enter` to execute a movement using a pose goal ...")
    input()
    tutorial.go_to_pose_goal()

    print("============ Press `Enter` to execute go to up state ...")
    input()
    tutorial.go_to_up_state()

    print("============ Press `Enter` to plan and display a Cartesian path ...")
    input()
    cartesian_plan, fraction = tutorial.plan_cartesian_path()

    print("============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ...")
    input()
    tutorial.display_trajectory(cartesian_plan)

    print("============ Press `Enter` to execute a saved path ...")
    input()
    tutorial.execute_plan(cartesian_plan)

    print ("============ Python tutorial demo complete!")

  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return
  
if __name__ == '__main__':
  main()
