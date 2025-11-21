#!/usr/bin/env python3
"""
SSC to Planner Bridge Node
Converts SSC point cloud to format suitable for ego-planner
Performs ground segmentation and obstacle extraction
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import struct


class SSCToPlannerBridge:
    def __init__(self):
        rospy.init_node('ssc_to_planner_bridge', anonymous=False)

        # Parameters
        self.ground_height_threshold = rospy.get_param('~ground_height_threshold', 0.3)  # 30cm
        self.max_height = rospy.get_param('~max_height', 2.0)  # 2m
        self.min_height = rospy.get_param('~min_height', -0.5)  # -50cm
        self.max_range = rospy.get_param('~max_range', 10.0)  # 10m

        # Publishers
        self.obstacle_cloud_pub = rospy.Publisher('/ssc/obstacle_cloud', PointCloud2, queue_size=1)
        self.ground_cloud_pub = rospy.Publisher('/ssc/ground_cloud', PointCloud2, queue_size=1)
        self.planner_cloud_pub = rospy.Publisher('/pcl_render_node/cloud', PointCloud2, queue_size=1)

        # Subscriber
        self.ssc_sub = rospy.Subscriber('/ssc/point_cloud', PointCloud2, self.ssc_callback, queue_size=1)

        rospy.loginfo("SSC to Planner Bridge Node initialized!")

    def ssc_callback(self, cloud_msg):
        """Process SSC point cloud and extract obstacles for planner"""
        try:
            # Parse point cloud
            points = []
            ground_points = []
            obstacle_points = []

            for point in pc2.read_points(cloud_msg, skip_nans=True):
                x, y, z = point[0], point[1], point[2]

                # Filter by range
                dist = np.sqrt(x*x + y*y + z*z)
                if dist > self.max_range:
                    continue

                # Filter by height
                if z < self.min_height or z > self.max_height:
                    continue

                # Classify as ground or obstacle
                if z < self.ground_height_threshold:
                    ground_points.append([x, y, z])
                else:
                    obstacle_points.append([x, y, z])
                    points.append([x, y, z])  # For planner input

            # Create header
            header = Header()
            header.stamp = cloud_msg.header.stamp
            header.frame_id = cloud_msg.header.frame_id

            # Publish obstacle cloud for visualization
            if len(obstacle_points) > 0:
                fields = [
                    pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                ]
                obstacle_msg = pc2.create_cloud(header, fields, obstacle_points)
                self.obstacle_cloud_pub.publish(obstacle_msg)

            # Publish ground cloud for visualization
            if len(ground_points) > 0:
                ground_msg = pc2.create_cloud(header, fields, ground_points)
                self.ground_cloud_pub.publish(ground_msg)

            # Publish to planner
            if len(points) > 0:
                planner_msg = pc2.create_cloud(header, fields, points)
                self.planner_cloud_pub.publish(planner_msg)
                rospy.loginfo_throttle(2.0, f"Published {len(points)} obstacle points to planner")
            else:
                rospy.logwarn_throttle(5.0, "No obstacle points to publish")

        except Exception as e:
            rospy.logerr(f"Error in SSC callback: {e}")
            import traceback
            traceback.print_exc()


def main():
    try:
        node = SSCToPlannerBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
