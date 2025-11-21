# 快速开始指南

## 🚀 5分钟快速启动

### 1. 编译工作空间（首次使用）

```bash
cd /home/xh/Desktop/ros1/main/ros1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### 2. 检查依赖

```bash
cd src/unitree_ssc_navigation/scripts
python3 check_dependencies.py
```

### 3. 启动系统

**方式A: 使用快速启动脚本（推荐）**
```bash
cd /home/xh/Desktop/ros1/main/ros1_ws/src/unitree_ssc_navigation/scripts
./quick_start.sh
```

**方式B: 手动启动**

仿真环境：
```bash
# 终端1
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=true \
    use_dsfnet:=false

# 终端2（启动后再运行）
rosrun unitree_guide junior_ctrl
# 按 2 -> FixedStand
# 按 4 -> Trotting
# 在RViz中使用2D Nav Goal设置目标
```

## 📁 已创建的文件

### 核心节点
- `scripts/dsfnet_inference_node.py` - DSFNet语义场景补全推理
- `scripts/ssc_to_planner_bridge.py` - SSC点云处理和转换
- `scripts/unitree_odom_bridge.py` - 里程计桥接

### Launch文件
- `launch/unitree_ssc_navigation.launch` - 完整系统启动
- `launch/test_dsfnet.launch` - DSFNet测试

### 工具脚本
- `scripts/check_dependencies.py` - 依赖检查
- `scripts/quick_start.sh` - 快速启动脚本

### 配置文件
- `config/camera_params.yaml` - 相机参数配置

### 文档
- `README.md` - 英文详细文档
- `README_CN.md` - 中文详细文档
- `QUICK_START_CN.md` - 本文件

### DSFNet补充文件
- `DSFNet/models/projection_layer.py` - 2D到3D投影层
- `DSFNet/models/visual.py` - 可视化工具

## 🎮 基本操作

### Unitree控制器键盘命令

进入模式：
- `0` - Passive（被动模式）
- `2` - FixedStand（固定站立）
- `4` - Trotting（小跑模式，可移动）

运动控制（在Trotting模式下）：
- `W` - 前进
- `S` - 后退
- `A` - 左移
- `D` - 右移
- `J` - 左转
- `L` - 右转
- `Space` - 停止

### RViz操作

1. **设置导航目标**：
   - 点击工具栏的 `2D Nav Goal`
   - 在地图上点击并拖动设置目标位置和方向

2. **查看规划结果**：
   - 绿色线条：规划的路径
   - 彩色点云：语义场景补全结果
   - 红色点云：检测到的障碍物

## 🔧 常用调试命令

```bash
# 查看所有ROS节点
rosnode list

# 查看某个节点的详细信息
rosnode info /dsfnet_inference

# 查看所有话题
rostopic list

# 查看话题发布频率
rostopic hz /cmd_vel

# 查看话题内容（一条消息）
rostopic echo /ssc/obstacle_cloud -n 1

# 查看TF树
rosrun tf view_frames
evince frames.pdf
```

## ⚙️ 系统配置

### 修改规划参数

编辑 `launch/unitree_ssc_navigation.launch`:

```xml
<!-- 地图大小 -->
<arg name="map_size_x" value="20.0"/>
<arg name="map_size_y" value="20.0"/>
<arg name="map_size_z" value="2.5"/>

<!-- 速度限制 -->
<arg name="max_vel" value="0.8" />    <!-- 降低速度更安全 -->
<arg name="max_acc" value="1.5" />

<!-- 规划视野 -->
<arg name="planning_horizon" value="5.0" />
```

### 修改相机参数

编辑 `config/camera_params.yaml` 或在launch文件中修改：

```yaml
# 相机内参
fx: 387.229248046875
fy: 387.229248046875
cx: 321.04638671875
cy: 243.44969177246094

# 图像尺寸
width: 640
height: 480
```

## 🐛 常见问题快速解决

### 问题1: 控制器无响应
```bash
# 检查话题
rostopic echo /cmd_vel -n 1
rostopic info /cmd_vel

# 重启控制器
# Ctrl+C 停止 junior_ctrl
rosrun unitree_guide junior_ctrl
# 重新按 2 -> 4
```

### 问题2: 找不到相机话题
```bash
# 列出所有相机相关话题
rostopic list | grep camera

# 在launch文件中重映射
<arg name="rgb_topic" default="/你的相机/rgb" />
<arg name="depth_topic" default="/你的相机/depth" />
```

### 问题3: DSFNet推理慢
```bash
# 检查是否使用GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 如果没有GPU，可以暂时关闭DSFNet
roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
    use_sim:=true \
    use_dsfnet:=false  # 关闭DSFNet
```

### 问题4: 编译错误
```bash
# 清理并重新编译
cd /home/xh/Desktop/ros1/main/ros1_ws
catkin_make clean
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## 📊 系统性能

预期性能指标：
- DSFNet推理: 100-200ms (需要GPU)
- 路径规划: 50-100ms
- 控制频率: 30Hz
- 端到端延迟: ~200-300ms

建议硬件：
- CPU: 4核+
- RAM: 8GB+
- GPU: NVIDIA RTX 2060+ (用于DSFNet)

## 📚 下一步

1. **阅读完整文档**：
   - 中文：`README_CN.md`
   - 英文：`README.md`

2. **调整参数**：
   - 根据你的环境调整地图大小
   - 根据机器人性能调整速度限制

3. **添加功能**：
   - 集成SLAM系统
   - 添加更多传感器
   - 实现多点导航

## 🆘 获取帮助

如遇到问题：
1. 查看详细文档 `README_CN.md`
2. 运行依赖检查 `python3 check_dependencies.py`
3. 检查ROS日志 `rosnode list` 和 `rosnode info`

## 🎯 测试清单

启动系统前的检查：
- [ ] ROS环境已source
- [ ] 工作空间已编译
- [ ] 依赖检查通过
- [ ] （如使用DSFNet）模型文件存在
- [ ] （如使用真机）网络已配置

运行测试：
- [ ] Gazebo仿真启动成功
- [ ] 机器人站立（FixedStand）
- [ ] RViz显示正常
- [ ] 可以设置导航目标
- [ ] 机器人能够移动
- [ ] 路径规划正常工作

祝使用愉快！🎉
