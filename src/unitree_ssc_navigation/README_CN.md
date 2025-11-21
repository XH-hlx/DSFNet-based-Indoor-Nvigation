# Unitree SSC 导航系统

基于Unitree Go1机器狗的智能导航系统，集成了：
- **DSFNet**: 基于RGB-D图像的深度语义场景补全
- **Ego-Planner**: 基于B样条的轨迹规划
- **Unitree Guide**: 四足机器人运动控制器

## 系统架构

```
┌──────────────┐
│ 相机(RGB-D)   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ DSFNet推理节点        │  ← RGB-D图像转换为语义场景补全
│ dsfnet_inference.py  │
└──────┬───────────────┘
       │ /ssc/point_cloud
       ▼
┌──────────────────────┐
│ SSC到规划器桥接       │  ← 地面分割和障碍物提取
│ ssc_to_planner.py    │
└──────┬───────────────┘
       │ /ssc/obstacle_cloud
       ▼
┌──────────────────────┐
│ Ego-Planner          │  ← 路径规划
│ ego_planner_node     │
└──────┬───────────────┘
       │ /planning/bspline
       ▼
┌──────────────────────┐
│ 轨迹跟踪服务器        │  ← MPC轨迹跟踪
│ traj_server          │
└──────┬───────────────┘
       │ /cmd_vel
       ▼
┌──────────────────────┐
│ Unitree控制器         │  ← 四足运动控制
│ junior_ctrl          │
└──────────────────────┘
```

## 安装步骤

### 1. 安装依赖

```bash
# 安装ROS依赖
sudo apt-get install ros-noetic-cv-bridge ros-noetic-tf ros-noetic-message-filters

# 安装Python依赖
pip3 install torch torchvision numpy opencv-python
```

### 2. 编译工作空间

```bash
cd /home/xh/Desktop/ros1/main/ros1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### 3. 准备DSFNet模型

将训练好的DSFNet模型权重放到指定目录：
```bash
mkdir -p ~/models/dsfnet
# 将你的 model.pth 复制到 ~/models/dsfnet/model.pth
```

## 使用方法

### 方法1: 使用快速启动脚本（推荐）

```bash
cd /home/xh/Desktop/ros1/main/ros1_ws/src/unitree_ssc_navigation/scripts
./quick_start.sh
```

按照提示选择运行模式：
1. 完整系统（仿真）
2. 仅测试DSFNet
3. 完整系统（真实机器人）

### 方法2: 手动启动完整系统（仿真）

```bash
# 终端1: 启动导航系统
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=true \
    use_dsfnet:=true \
    dsfnet_model_path:=~/models/dsfnet/model.pth

# 终端2: 启动Unitree控制器
rosrun unitree_guide junior_ctrl
```

启动后操作步骤：
1. 按 `2` 进入FixedStand（固定站立）模式
2. 按 `4` 进入Trotting（小跑）模式
3. 在RViz中使用 `2D Nav Goal` 设置导航目标

### 仅测试DSFNet

```bash
# 使用rosbag或真实相机
roslaunch unitree_ssc_navigation test_dsfnet.launch \
    dsfnet_model_path:=~/models/dsfnet/model.pth \
    rgb_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw
```

### 在真实机器人上运行

```bash
# 终端1: 配置网络
cd src/unitree_ros_to_real/unitree_legged_real
sudo ./ipconfig.sh

# 终端2: 启动导航系统（关闭仿真器）
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=false \
    use_dsfnet:=true \
    dsfnet_model_path:=~/models/dsfnet/model.pth

# 终端3: 启动真实机器人接口（高级控制）
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel

# 终端4: 启动导航控制器
rosrun unitree_guide junior_ctrl
```

**注意事项**：
- 真实机器人需要先通过手柄让机器人站起来
- 确保机器人周围有足够的安全空间
- 建议先在仿真中测试

## 话题说明

### 输入话题
- `/camera/color/image_raw` - RGB图像 (sensor_msgs/Image)
- `/camera/depth/image_raw` - 深度图像 (sensor_msgs/Image)
- `/trunk_odom` - 机器人里程计 (nav_msgs/Odometry)

### 输出话题
- `/ssc/point_cloud` - 语义场景补全点云 (sensor_msgs/PointCloud2)
- `/ssc/obstacle_cloud` - 障碍物点云（用于规划器）(sensor_msgs/PointCloud2)
- `/ssc/ground_cloud` - 地面点云 (sensor_msgs/PointCloud2)
- `/planning/bspline` - 规划的轨迹 (ego_planner/Bspline)
- `/cmd_vel` - 速度命令 (geometry_msgs/Twist)

### 可视化话题
- `/ssc/depth_viz` - 深度可视化 (sensor_msgs/Image)
- `/planning/pos_cmd` - 位置命令可视化
- `/move_base_simple/goal` - 目标标记 (geometry_msgs/PoseStamped)

## 参数配置

### DSFNet参数
编辑 `config/camera_params.yaml` 或作为launch参数传递：
- `fx, fy, cx, cy` - 相机内参
- `img_width, img_height` - 图像尺寸
- `voxel_unit` - 体素大小（米）(默认: 0.02)
- `num_classes` - 语义类别数量 (默认: 12)

### Ego-Planner参数
- `map_size_x, map_size_y, map_size_z` - 规划地图大小
- `max_vel` - 最大速度 (默认: 0.8 m/s)
- `max_acc` - 最大加速度 (默认: 1.5 m/s²)
- `planning_horizon` - 规划视野 (默认: 5.0 m)

### SSC桥接参数
- `ground_height_threshold` - 地面高度阈值 (默认: 0.3m)
- `max_height` - 最大障碍物高度 (默认: 2.0m)
- `max_range` - 最大障碍物距离 (默认: 10.0m)

## 常见问题

### 没有DSFNet模型怎么办？
如果没有训练好的模型，系统会使用随机权重运行（仅用于测试）。要训练模型：
1. 准备NYU-SSCAD数据集
2. 按照DSFNet原始仓库的训练说明进行训练
3. 将训练好的权重复制到模型路径

### 找不到相机话题
检查可用的相机话题：
```bash
rostopic list | grep camera
```

如需重映射话题，在launch文件中修改：
```xml
<arg name="rgb_topic" default="/你的/rgb/话题" />
<arg name="depth_topic" default="/你的/depth/话题" />
```

### Unitree控制器无响应
1. 确保机器人处于正确模式（按 L2+A 进入FixedStand）
2. 检查 `/cmd_vel` 话题是否在发布
3. 验证traj_server是否运行: `rosnode info /traj_server`

### 推理速度慢
DSFNet需要GPU才能实时运行。检查：
```bash
# 验证CUDA是否可用
python3 -c "import torch; print(torch.cuda.is_available())"
```

为了更快的推理速度，可以：
- 降低图像分辨率
- 使用更轻量的模型
- 确保使用GPU运行

### 检查系统依赖
```bash
cd /home/xh/Desktop/ros1/main/ros1_ws/src/unitree_ssc_navigation/scripts
python3 check_dependencies.py
```

## 文件结构

```
unitree_ssc_navigation/
├── launch/
│   ├── unitree_ssc_navigation.launch  # 主系统启动文件
│   └── test_dsfnet.launch             # DSFNet测试启动文件
├── scripts/
│   ├── dsfnet_inference_node.py       # DSFNet ROS节点
│   ├── ssc_to_planner_bridge.py       # SSC到规划器转换器
│   ├── unitree_odom_bridge.py         # 里程计桥接
│   ├── check_dependencies.py          # 依赖检查工具
│   └── quick_start.sh                 # 快速启动脚本
├── config/
│   ├── camera_params.yaml             # 相机参数
│   └── (RViz配置文件)
├── models/
│   └── (模型权重文件)
└── README_CN.md
```

## 性能指标

- **DSFNet推理**: ~100-200ms (RTX 3060+)
- **Ego-planner**: ~50-100ms 每次重规划
- **控制频率**: 30Hz (/cmd_vel)
- **端到端帧率**: 5-10 Hz

## 调试技巧

### 查看节点运行状态
```bash
rosnode list
rosnode info /dsfnet_inference
```

### 查看话题数据
```bash
rostopic list
rostopic echo /ssc/point_cloud -n 1
rostopic hz /cmd_vel
```

### 查看TF树
```bash
rosrun tf view_frames
# 查看生成的frames.pdf
```

### RViz可视化
在RViz中添加以下显示：
- PointCloud2: `/ssc/obstacle_cloud` - 显示障碍物
- Path: `/planning/path` - 显示规划路径
- RobotModel: 显示机器人模型
- TF: 显示坐标系

## 贡献与反馈

如有问题或建议，请联系开发者或在GitHub上提issue。

## 许可证

本集成包按原样提供。各个组件保留其原始许可证：
- DSFNet: 原始仓库许可证
- Ego-Planner: 原始仓库许可证
- Unitree SDK: Unitree Robotics许可证
