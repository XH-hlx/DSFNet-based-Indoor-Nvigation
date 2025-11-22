#!/usr/bin/env python3
"""
Command Converter: Position Command -> Velocity Command
Converts Ego-Planner position commands to velocity commands for Go2
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
import tf.transformations as tft


class CmdConverter:
    def __init__(self):
        rospy.init_node('cmd_converter', anonymous=False)

        # Parameters
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.8)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)
        self.control_frequency = rospy.get_param('~control_frequency', 30.0)

        # PID gains
        self.kp_linear = rospy.get_param('~kp_linear', 1.0)
        self.kp_angular = rospy.get_param('~kp_angular', 2.0)

        # State
        self.current_odom = None
        self.target_pos = None
        self.target_yaw = None

        # Publishers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Subscribers
        self.pos_cmd_sub = rospy.Subscriber('/planning/pos_cmd', PositionCommand, self.pos_cmd_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/visual_slam/odom', Odometry, self.odom_callback, queue_size=1)

        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_frequency), self.control_loop)

        rospy.loginfo("Command Converter initialized")

    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom = msg

    def pos_cmd_callback(self, msg):
        """Receive position command from planner"""
        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.target_yaw = msg.yaw

    def control_loop(self, event):
        """Main control loop"""
        if self.current_odom is None or self.target_pos is None:
            return

        # Get current position
        current_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])

        # Get current yaw
        orientation_q = self.current_odom.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tft.euler_from_quaternion(orientation_list)

        # Position error (only XY for ground robot)
        pos_error = self.target_pos[:2] - current_pos[:2]
        distance = np.linalg.norm(pos_error)

        # Yaw error
        yaw_error = self.normalize_angle(self.target_yaw - yaw)

        # Control law
        cmd_vel = Twist()

        if distance > 0.05:  # Position threshold
            # Desired heading
            desired_yaw = np.arctan2(pos_error[1], pos_error[0])
            heading_error = self.normalize_angle(desired_yaw - yaw)

            # Angular velocity (turn towards goal first)
            if abs(heading_error) > 0.2:  # ~11 degrees
                cmd_vel.angular.z = self.kp_angular * heading_error
                cmd_vel.linear.x = 0.0  # Don't move forward while turning
            else:
                # Linear velocity
                cmd_vel.linear.x = self.kp_linear * distance

                # Angular velocity (fine adjustment)
                cmd_vel.angular.z = self.kp_angular * heading_error

        else:
            # At target position, only adjust yaw
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.kp_angular * yaw_error

        # Saturate velocities
        cmd_vel.linear.x = np.clip(cmd_vel.linear.x, -self.max_linear_vel, self.max_linear_vel)
        cmd_vel.angular.z = np.clip(cmd_vel.angular.z, -self.max_angular_vel, self.max_angular_vel)

        # Publish
        self.vel_pub.publish(cmd_vel)

        # Debug
        rospy.loginfo_throttle(1.0, f"Distance: {distance:.2f}m, Linear: {cmd_vel.linear.x:.2f}, Angular: {cmd_vel.angular.z:.2f}")

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


if __name__ == '__main__':
    try:
        converter = CmdConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
