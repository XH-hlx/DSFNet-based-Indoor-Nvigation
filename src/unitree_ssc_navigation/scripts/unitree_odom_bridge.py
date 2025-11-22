#!/usr/bin/env python3
"""
Unitree Odometry Bridge Node
Publishes odometry from Unitree robot to the topic expected by ego-planner
Also provides pose information for the planner
"""

import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Imu


class UnitreeOdomBridge:
    def __init__(self):
        rospy.init_node('unitree_odom_bridge', anonymous=False)

        # Parameters
        self.odom_input_topic = rospy.get_param('~odom_input_topic', '/trunk_odom')
        self.odom_output_topic = rospy.get_param('~odom_output_topic', '/visual_slam/odom')
        self.pose_output_topic = rospy.get_param('~pose_output_topic', '/mavros/local_position/pose')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.world_frame = rospy.get_param('~world_frame', 'world')

        # TF broadcaster and listener
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Publishers
        self.odom_pub = rospy.Publisher(self.odom_output_topic, Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher(self.pose_output_topic, PoseStamped, queue_size=10)

        # Subscriber
        self.odom_sub = rospy.Subscriber(self.odom_input_topic, Odometry, self.odom_callback, queue_size=10)

        # Camera to base transform (adjust based on your robot's camera mounting)
        # For Go1, camera is typically mounted on the front, facing forward
        # Camera position relative to base_link (x forward, y left, z up)
        self.cam_to_base_trans = rospy.get_param('~camera_x', 0.28)
        self.cam_to_base_y = rospy.get_param('~camera_y', 0.0)
        self.cam_to_base_z = rospy.get_param('~camera_z', 0.03)

        rospy.loginfo(f"Unitree Odom Bridge initialized!")
        rospy.loginfo(f"  Input: {self.odom_input_topic}")
        rospy.loginfo(f"  Output: {self.odom_output_topic}")
        rospy.loginfo(f"  Pose: {self.pose_output_topic}")

    def odom_callback(self, msg):
        """Forward odometry and publish transforms"""
        try:
            # Republish odometry
            odom_out = msg
            odom_out.header.frame_id = self.world_frame
            odom_out.child_frame_id = self.base_frame
            self.odom_pub.publish(odom_out)

            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header = odom_out.header
            pose_msg.pose = odom_out.pose.pose
            self.pose_pub.publish(pose_msg)

            # Broadcast TF: world -> base_link
            self.tf_broadcaster.sendTransform(
                (msg.pose.pose.position.x,
                 msg.pose.pose.position.y,
                 msg.pose.pose.position.z),
                (msg.pose.pose.orientation.x,
                 msg.pose.pose.orientation.y,
                 msg.pose.pose.orientation.z,
                 msg.pose.pose.orientation.w),
                msg.header.stamp,
                self.base_frame,
                self.world_frame
            )

            # Broadcast TF: base_link -> camera_link
            # Camera is mounted on the front of the robot
            self.tf_broadcaster.sendTransform(
                (self.cam_to_base_trans, self.cam_to_base_y, self.cam_to_base_z),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                msg.header.stamp,
                'camera_link',
                self.base_frame
            )

            # Broadcast TF: camera_link -> camera_color_optical_frame
            # ROS optical frame convention: z forward, x right, y down
            self.tf_broadcaster.sendTransform(
                (0, 0, 0),
                tf.transformations.quaternion_from_euler(-np.pi/2, 0, -np.pi/2),
                msg.header.stamp,
                self.camera_frame,
                'camera_link'
            )

        except Exception as e:
            rospy.logerr(f"Error in odom callback: {e}")


def main():
    try:
        node = UnitreeOdomBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
