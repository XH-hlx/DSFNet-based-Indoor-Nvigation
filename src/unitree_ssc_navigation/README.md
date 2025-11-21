# Unitree SSC Navigation

Integrated navigation system for Unitree Go1 robot using:
- **DSFNet**: Deep Semantic Scene Completion from RGB-D images
- **Ego-Planner**: B-spline based trajectory planning
- **Unitree Guide**: Quadruped locomotion controller

## System Architecture

```
┌──────────────┐
│ Camera       │
│ (RGB-D)      │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ DSFNet Inference     │  ← SSC from RGB-D
│ dsfnet_inference.py  │
└──────┬───────────────┘
       │ /ssc/point_cloud
       ▼
┌──────────────────────┐
│ SSC to Planner       │  ← Ground segmentation
│ ssc_to_planner.py    │
└──────┬───────────────┘
       │ /ssc/obstacle_cloud
       ▼
┌──────────────────────┐
│ Ego-Planner          │  ← Path planning
│ ego_planner_node     │
└──────┬───────────────┘
       │ /planning/bspline
       ▼
┌──────────────────────┐
│ Trajectory Server    │  ← MPC tracking
│ traj_server          │
└──────┬───────────────┘
       │ /cmd_vel
       ▼
┌──────────────────────┐
│ Unitree Controller   │  ← Locomotion
│ junior_ctrl          │
└──────────────────────┘
```

## Installation

### 1. Prerequisites

```bash
# Install ROS dependencies
sudo apt-get install ros-noetic-cv-bridge ros-noetic-tf ros-noetic-message-filters

# Install Python dependencies
pip3 install torch torchvision numpy opencv-python
```

### 2. Build the workspace

```bash
cd /home/xh/Desktop/ros1/main/ros1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### 3. Prepare DSFNet model

Place your trained DSFNet model weights in a directory, e.g.:
```bash
mkdir -p ~/models/dsfnet
# Copy your model.pth to ~/models/dsfnet/model.pth
```

## Usage

### Full System (Simulation)

```bash
# Terminal 1: Launch the navigation system
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=true \
    use_dsfnet:=true \
    dsfnet_model_path:=~/models/dsfnet/model.pth

# Terminal 2: Start the Unitree controller
rosrun unitree_guide junior_ctrl
```

Once running:
1. Press `2` to enter FixedStand mode
2. Press `4` to enter Trotting mode
3. In RViz, use `2D Nav Goal` to set navigation targets

### Test DSFNet Only

```bash
# With rosbag or camera
roslaunch unitree_ssc_navigation test_dsfnet.launch \
    dsfnet_model_path:=~/models/dsfnet/model.pth \
    rgb_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw
```

### Real Robot

```bash
# Terminal 1: Configure network
cd src/unitree_ros_to_real/unitree_legged_real
sudo ./ipconfig.sh

# Terminal 2: Launch navigation (without simulator)
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=false \
    use_dsfnet:=true \
    dsfnet_model_path:=~/models/dsfnet/model.pth

# Terminal 3: Start Unitree controller (high-level)
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel

# Terminal 4: Start navigation controller
rosrun unitree_guide junior_ctrl
```

## Topic Overview

### Input Topics
- `/camera/color/image_raw` - RGB image (sensor_msgs/Image)
- `/camera/depth/image_raw` - Depth image (sensor_msgs/Image)
- `/trunk_odom` - Robot odometry (nav_msgs/Odometry)

### Output Topics
- `/ssc/point_cloud` - Semantic scene completion (sensor_msgs/PointCloud2)
- `/ssc/obstacle_cloud` - Obstacles for planner (sensor_msgs/PointCloud2)
- `/ssc/ground_cloud` - Ground points (sensor_msgs/PointCloud2)
- `/planning/bspline` - Planned trajectory (ego_planner/Bspline)
- `/cmd_vel` - Velocity commands (geometry_msgs/Twist)

### Visualization Topics
- `/ssc/depth_viz` - Depth visualization (sensor_msgs/Image)
- `/planning/pos_cmd` - Position command visualization
- `/move_base_simple/goal` - Goal marker (geometry_msgs/PoseStamped)

## Parameters

### DSFNet Parameters
Edit `config/camera_params.yaml` or pass as launch arguments:
- `fx, fy, cx, cy` - Camera intrinsics
- `img_width, img_height` - Image dimensions
- `voxel_unit` - Voxel size in meters (default: 0.02)
- `num_classes` - Number of semantic classes (default: 12)

### Ego-Planner Parameters
- `map_size_x, map_size_y, map_size_z` - Planning map size
- `max_vel` - Maximum velocity (default: 0.8 m/s)
- `max_acc` - Maximum acceleration (default: 1.5 m/s²)
- `planning_horizon` - Planning horizon (default: 5.0 m)

### SSC Bridge Parameters
- `ground_height_threshold` - Height below which voxels are ground (default: 0.3m)
- `max_height` - Maximum obstacle height (default: 2.0m)
- `max_range` - Maximum obstacle range (default: 10.0m)

## Troubleshooting

### No DSFNet model
If you don't have a trained model, the system will run with random weights (for testing only). To train a model:
1. Prepare NYU-SSCAD dataset
2. Follow DSFNet training instructions in the original repository
3. Copy trained weights to your model path

### Camera topics not found
Check available camera topics:
```bash
rostopic list | grep camera
```

Remap topics in launch file if needed:
```xml
<arg name="rgb_topic" default="/your/rgb/topic" />
<arg name="depth_topic" default="/your/depth/topic" />
```

### Unitree controller not responding
1. Ensure robot is in correct mode (press L2+A for FixedStand)
2. Check `/cmd_vel` topic is being published
3. Verify traj_server is running: `rosnode info /traj_server`

### Slow inference
DSFNet requires GPU for real-time performance. Check:
```bash
# Verify CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"
```

For faster inference, reduce image resolution or use a lighter model.

## File Structure

```
unitree_ssc_navigation/
├── launch/
│   ├── unitree_ssc_navigation.launch  # Main system launch
│   └── test_dsfnet.launch             # DSFNet test
├── scripts/
│   ├── dsfnet_inference_node.py       # DSFNet ROS node
│   ├── ssc_to_planner_bridge.py       # SSC to planner converter
│   └── unitree_odom_bridge.py         # Odometry bridge
├── config/
│   ├── camera_params.yaml             # Camera parameters
│   ├── navigation.rviz                # RViz config (full system)
│   └── dsfnet_test.rviz               # RViz config (DSFNet only)
├── models/
│   └── (place model weights here)
└── README.md
```

## Performance Notes

- **DSFNet inference**: ~100-200ms on GPU (RTX 3060+)
- **Ego-planner**: ~50-100ms per replan
- **Control frequency**: 30Hz (/cmd_vel)
- **Expected FPS**: 5-10 Hz end-to-end

## Citation

If you use this system, please cite:

```bibtex
@article{dsfnet,
  title={Deep Semantic Scene Completion},
  author={...},
  journal={...},
  year={2021}
}

@inproceedings{egoplanner,
  title={EGO-Planner: An ESDF-free Gradient-based Local Planner},
  author={Zhou, Xin and others},
  booktitle={IEEE RA-L},
  year={2021}
}
```

## License

This integration package is provided as-is. Individual components maintain their original licenses:
- DSFNet: Original repository license
- Ego-Planner: Original repository license
- Unitree SDK: Unitree Robotics license
