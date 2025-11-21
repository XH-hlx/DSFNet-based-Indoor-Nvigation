#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64, String
from geometry_msgs.msg import Point

class SofaBoundingBox:
    def __init__(self, model_name):
        self.model_name = model_name
        self.area = None
        self.corners = []
        self.name = None
        
        rospy.init_node('get_sofa_bounding_box')
        rospy.Subscriber(f"/gazebo/{model_name}/area", Float64, self.area_callback)
        rospy.Subscriber(f"/gazebo/{model_name}/corners", Point, self.corners_callback)
        rospy.Subscriber(f"/gazebo/{model_name}/name", String, self.name_callback)

    def area_callback(self, msg):
        self.area = msg.data

    def corners_callback(self, msg):
        self.corners.append(msg)
        # 限制最多 8 个角点（2D 四角 + 3D 边界框的 4 个额外角点）
        if len(self.corners) > 8:
            self.corners.pop(0)

    def name_callback(self, msg):
        self.name = msg.data

    def print_info(self):
        rospy.sleep(1)  # 等待数据接收
        if self.name and self.area is not None and len(self.corners) >= 4:
            print(f"物体名称: {self.name}")
            print(f"占用面积: {self.area:.2f} 平方米")
            print("四角坐标 (2D XY 平面投影):")
            for i in range(4):
                print(f"角{i+1}: ({self.corners[i].x:.2f}, {self.corners[i].y:.2f})")
            print("多面体角点 (3D 坐标，共 {0} 个):".format(len(self.corners) - 4))
            for i in range(4, len(self.corners)):
                print(f"角{i-3}: ({self.corners[i].x:.2f}, {self.corners[i].y:.2f}, {self.corners[i].z:.2f})")
        else:
            print("等待数据...")

if __name__ == '__main__':
    model_name = 'aws_robomaker_residential_SofaC_01'
    sofa_bbox = SofaBoundingBox(model_name)
    try:
        while not rospy.is_shutdown():
            sofa_bbox.print_info()
            rospy.sleep(1)
    except rospy.ROSInterruptException:
        pass