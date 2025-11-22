#!/usr/bin/env python
import rospy
import tf
from gazebo_msgs.msg import LinkStates

def link_states_cb(msg):
    try:
        idx = msg.name.index('go1_gazebo::base')
        pose = msg.pose[idx]
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            rospy.Time.now(),   # use_sim_time=true 会让这个时间跟 Gazebo /clock 保持一致
            'base',
            'odom'
        )
    except ValueError:
        pass

if __name__=='__main__':
    rospy.init_node('odom_broadcaster')
    rospy.Subscriber('/gazebo/link_states', LinkStates, link_states_cb, queue_size=1)
    rospy.spin()

#!/usr/bin/env python
# import rospy
# import tf
# from gazebo_msgs.msg import LinkStates

# class OdomBroadcaster:
#     def __init__(self):
#         self.br = tf.TransformBroadcaster()
#         self.last_pose = None
#         rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_cb, queue_size=1)
    
#     def link_states_cb(self, msg):
#         try:
#             idx = msg.name.index('go1_gazebo::base')
#             pose = msg.pose[idx]
#             # 只有位置/姿态发生变化才广播
#             cur_pose = (pose.position.x, pose.position.y, pose.position.z,
#                         pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
#             if cur_pose == self.last_pose:
#                 return  # 防止疯狂刷同样的数据
#             self.last_pose = cur_pose
#             # 用仿真时间
#             stamp = rospy.Time.now()
#             self.br.sendTransform(
#                 (pose.position.x, pose.position.y, pose.position.z),
#                 (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
#                 stamp,
#                 'base',  # 注意base_link和你TF树要一致
#                 'odom'
#             )
#         except ValueError:
#             pass

# if __name__ == '__main__':
#     rospy.init_node('odom_broadcaster')
#     OdomBroadcaster()
#     rospy.spin()
