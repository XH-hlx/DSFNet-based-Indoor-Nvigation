#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

path = Path()

def odom_cb(msg):
    global path
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    path.header = msg.header
    path.poses.append(pose)
    path_pub.publish(path)

if __name__ == '__main__':
    rospy.init_node('path_publisher')
    path_pub = rospy.Publisher('/robot_path', Path, queue_size=10)
    rospy.Subscriber('/odom', Odometry, odom_cb)
    rospy.spin()