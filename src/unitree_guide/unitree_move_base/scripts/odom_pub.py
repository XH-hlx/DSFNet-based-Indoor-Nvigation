#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import LinkStates

def cb(msg):
    try:
        idx = msg.name.index('go1_gazebo::base')
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        odom.pose.pose = msg.pose[idx]
        odom.twist.twist = msg.twist[idx]
        pub.publish(odom)
    except:
        pass

if __name__ == '__main__':
    rospy.init_node('odom_pub')
    pub = rospy.Publisher('/odom', Odometry, queue_size=1)
    rospy.Subscriber('/gazebo/link_states', LinkStates, cb)
    rospy.spin()