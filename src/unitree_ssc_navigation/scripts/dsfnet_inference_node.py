#!/usr/bin/env python3
"""
DSFNet Inference ROS Node
Performs Semantic Scene Completion using RGB-D images from the Unitree Go1 robot
"""

import rospy
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import struct

# Add DSFNet to Python path
DSFNET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'DSFNet')
sys.path.insert(0, DSFNET_PATH)

from models.DDRNet import SSC_RGBD_DDRNet
from models.projection_layer import compute_CP_mega_matrix
from config import colorMap


class DSFNetInferenceNode:
    def __init__(self):
        rospy.init_node('dsfnet_inference_node', anonymous=False)

        # Parameters
        self.model_path = rospy.get_param('~model_path', '')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = rospy.get_param('~num_classes', 12)
        self.voxel_size = rospy.get_param('~voxel_size', [240, 144, 240])
        self.voxel_size_lr = [60, 36, 60]

        # Camera intrinsics (default for RealSense D435)
        self.fx = rospy.get_param('~fx', 387.229248046875)
        self.fy = rospy.get_param('~fy', 387.229248046875)
        self.cx = rospy.get_param('~cx', 321.04638671875)
        self.cy = rospy.get_param('~cy', 243.44969177246094)

        # Image size
        self.img_height = rospy.get_param('~img_height', 480)
        self.img_width = rospy.get_param('~img_width', 640)

        # Voxel parameters
        self.voxel_origin = np.array([0, -1.44, 0])  # Origin of voxel grid in camera frame
        self.voxel_unit = 0.02  # 2cm per voxel

        # Initialize model
        rospy.loginfo("Loading DSFNet model...")
        self.model = SSC_RGBD_DDRNet(num_classes=self.num_classes).to(self.device)

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            rospy.loginfo(f"Model loaded from {self.model_path}")
        else:
            rospy.logwarn("No model path provided or file does not exist. Using random weights.")

        self.model.eval()

        # Precompute projection matrices
        self.compute_projection_matrices()

        # CV Bridge
        self.bridge = CvBridge()

        # Semantic cost mapping (12 classes: empty, ceiling, floor, wall, window, chair, bed, sofa, table, tvs, furniture, objects)
        self.semantic_costs = rospy.get_param('~semantic_costs', [
            0.1,   # 0: empty - very low cost
            1.0,   # 1: ceiling - high cost
            0.3,   # 2: floor - low cost, traversable
            1.0,   # 3: wall - high cost
            0.9,   # 4: window - high cost
            0.7,   # 5: chair - medium-high cost
            0.8,   # 6: bed - high cost
            0.8,   # 7: sofa - high cost
            0.7,   # 8: table - medium-high cost
            0.6,   # 9: tvs - medium cost
            0.75,  # 10: furniture - medium-high cost
            0.65   # 11: objects - medium cost
        ])

        # Publishers
        self.ssc_pointcloud_pub = rospy.Publisher('/ssc/point_cloud', PointCloud2, queue_size=1)
        self.ssc_occupied_pub = rospy.Publisher('/ssc/occupied_cloud', PointCloud2, queue_size=1)
        self.ssc_semantic_pub = rospy.Publisher('/ssc_inference_node/semantic_cloud_xyz', PointCloud2, queue_size=1)  # For ego-planner
        self.depth_viz_pub = rospy.Publisher('/ssc/depth_viz', Image, queue_size=1)

        # Subscribers with time synchronization
        self.rgb_sub = Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = Subscriber('/camera/depth/image_raw', Image)

        # Synchronize RGB and Depth
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        rospy.loginfo("DSFNet Inference Node initialized and ready!")

    def compute_projection_matrices(self):
        """Precompute projection matrices for 3D voxelization"""
        # For full resolution
        self.P = compute_CP_mega_matrix(
            self.voxel_origin,
            self.voxel_unit,
            self.img_width,
            self.img_height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.voxel_size[0],
            self.voxel_size[1],
            self.voxel_size[2]
        ).long().to(self.device)

        # For low resolution
        self.P4 = compute_CP_mega_matrix(
            self.voxel_origin,
            self.voxel_unit * 4,
            self.img_width,
            self.img_height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.voxel_size_lr[0],
            self.voxel_size_lr[1],
            self.voxel_size_lr[2]
        ).long().to(self.device)

    def preprocess_depth(self, depth_img):
        """Preprocess depth image"""
        # Normalize depth to meters if needed
        if depth_img.max() > 10:  # Likely in mm
            depth_img = depth_img.astype(np.float32) / 1000.0
        else:
            depth_img = depth_img.astype(np.float32)

        # Resize to expected size
        if depth_img.shape != (self.img_height, self.img_width):
            depth_img = cv2.resize(depth_img, (self.img_width, self.img_height))

        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_img).float().unsqueeze(0).unsqueeze(0)
        return depth_tensor.to(self.device)

    def preprocess_rgb(self, rgb_img):
        """Preprocess RGB image"""
        # Convert BGR to RGB if needed
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Resize to expected size
        if rgb_img.shape[:2] != (self.img_height, self.img_width):
            rgb_img = cv2.resize(rgb_img, (self.img_width, self.img_height))

        # Normalize to [0, 1]
        rgb_img = rgb_img.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor.to(self.device)

    def image_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB-D images"""
        try:
            # Convert ROS images to OpenCV format
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            # Handle different depth encodings
            if depth_msg.encoding == '16UC1':
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            elif depth_msg.encoding == '32FC1':
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            else:
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Preprocess images
            rgb_tensor = self.preprocess_rgb(rgb_img)
            depth_tensor = self.preprocess_depth(depth_img)

            # Create predicted depth (for now, use input depth)
            pred_depth_tensor = depth_tensor.clone()

            # Inference
            with torch.no_grad():
                start_time = rospy.Time.now()

                # Forward pass
                ssc_output, aux1, aux2 = self.model(
                    x_depth=depth_tensor,
                    x_rgb=rgb_tensor,
                    pred_depth=pred_depth_tensor.squeeze(0).squeeze(0),
                    p=self.P,
                    p4=self.P4
                )

                inference_time = (rospy.Time.now() - start_time).to_sec()
                rospy.loginfo(f"Inference time: {inference_time:.3f}s ({1.0/inference_time:.1f} FPS)")

            # Get predicted class labels
            ssc_pred = torch.argmax(ssc_output, dim=1).cpu().numpy()[0]  # (60, 36, 60)

            # Convert SSC to point cloud and publish
            self.publish_ssc_pointcloud(ssc_pred, rgb_msg.header)

            # Publish semantic cloud for ego-planner (XYZI format)
            self.publish_semantic_cloud(ssc_pred, rgb_msg.header)

            # Publish depth visualization
            self.publish_depth_viz(depth_img, rgb_msg.header)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
            import traceback
            traceback.print_exc()

    def publish_depth_viz(self, depth_img, header):
        """Publish depth visualization"""
        try:
            depth_viz = depth_img.copy()
            if depth_viz.max() > 0:
                depth_viz = (depth_viz / depth_viz.max() * 255).astype(np.uint8)
            else:
                depth_viz = np.zeros_like(depth_viz, dtype=np.uint8)

            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding='bgr8')
            depth_msg.header = header
            self.depth_viz_pub.publish(depth_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish depth viz: {e}")

    def publish_ssc_pointcloud(self, ssc_pred, header):
        """Convert SSC voxel grid to point cloud and publish"""
        try:
            points = []
            occupied_points = []

            # Voxel resolution parameters for low-res output (60x36x60)
            voxel_unit_lr = self.voxel_unit * 4  # 8cm per voxel

            # Iterate through voxel grid
            for x in range(self.voxel_size_lr[0]):
                for y in range(self.voxel_size_lr[1]):
                    for z in range(self.voxel_size_lr[2]):
                        class_id = ssc_pred[x, y, z]

                        # Skip empty voxels (class 0)
                        if class_id == 0:
                            continue

                        # Compute voxel center in camera frame
                        vx = self.voxel_origin[0] + x * voxel_unit_lr + voxel_unit_lr / 2
                        vy = self.voxel_origin[1] + y * voxel_unit_lr + voxel_unit_lr / 2
                        vz = self.voxel_origin[2] + z * voxel_unit_lr + voxel_unit_lr / 2

                        # Get color from colorMap
                        if class_id < len(colorMap):
                            color = colorMap[class_id]
                            rgb = struct.unpack('I', struct.pack('BBBB', color[2], color[1], color[0], 255))[0]
                        else:
                            rgb = 0xFFFFFF

                        point = [vz, -vx, -vy, rgb]  # Transform to ROS coordinate (z forward, x right, y down)
                        points.append(point)

                        # Also create occupied cloud (all occupied voxels)
                        occupied_points.append([vz, -vx, -vy, 0xFF0000])  # Red for occupied

            # Create PointCloud2 message with RGB
            header.frame_id = 'camera_color_optical_frame'

            if len(points) > 0:
                fields = [
                    pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
                ]

                pc_msg = pc2.create_cloud(header, fields, points)
                self.ssc_pointcloud_pub.publish(pc_msg)

                # Publish occupied cloud
                occupied_msg = pc2.create_cloud(header, fields, occupied_points)
                self.ssc_occupied_pub.publish(occupied_msg)

                rospy.loginfo(f"Published point cloud with {len(points)} points")
            else:
                rospy.logwarn("No occupied voxels found in SSC output")

        except Exception as e:
            rospy.logerr(f"Error publishing SSC point cloud: {e}")
            import traceback
            traceback.print_exc()

    def publish_semantic_cloud(self, ssc_pred, header):
        """Publish semantic point cloud in XYZI format for ego-planner integration"""
        try:
            points = []
            voxel_unit_lr = self.voxel_unit * 4  # 8cm per voxel

            # Iterate through voxel grid
            for x in range(self.voxel_size_lr[0]):
                for y in range(self.voxel_size_lr[1]):
                    for z in range(self.voxel_size_lr[2]):
                        class_id = int(ssc_pred[x, y, z])

                        # Skip empty voxels (class 0)
                        if class_id == 0:
                            continue

                        # Compute voxel center in camera frame
                        vx = self.voxel_origin[0] + x * voxel_unit_lr + voxel_unit_lr / 2
                        vy = self.voxel_origin[1] + y * voxel_unit_lr + voxel_unit_lr / 2
                        vz = self.voxel_origin[2] + z * voxel_unit_lr + voxel_unit_lr / 2

                        # Transform to ROS coordinate (z forward, x right, y down)
                        px = vz
                        py = -vx
                        pz = -vy

                        # Use intensity field to encode semantic label
                        intensity = float(class_id)

                        # Append [x, y, z, intensity(label)]
                        points.append([px, py, pz, intensity])

            if len(points) > 0:
                # Create PointCloud2 with XYZI fields
                header_copy = Header()
                header_copy.stamp = header.stamp
                header_copy.frame_id = 'camera_color_optical_frame'

                fields = [
                    pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('intensity', 12, pc2.PointField.FLOAT32, 1),
                ]

                cloud_msg = pc2.create_cloud(header_copy, fields, points)
                self.ssc_semantic_pub.publish(cloud_msg)

                rospy.loginfo_throttle(2.0, f"Published semantic cloud with {len(points)} points for ego-planner")
            else:
                rospy.logwarn_throttle(5.0, "No occupied voxels found for semantic cloud")

        except Exception as e:
            rospy.logerr(f"Error publishing semantic cloud: {e}")
            import traceback
            traceback.print_exc()


def main():
    try:
        node = DSFNetInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
