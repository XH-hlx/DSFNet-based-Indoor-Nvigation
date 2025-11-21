#!/bin/bash
# Quick start script for Unitree SSC Navigation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Unitree SSC Navigation - Quick Start${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if ROS is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS environment not sourced${NC}"
    echo "Please run: source /opt/ros/noetic/setup.bash"
    exit 1
fi

# Check if workspace is built
WORKSPACE_DIR="/home/xh/Desktop/ros1/main/ros1_ws"
if [ ! -f "$WORKSPACE_DIR/devel/setup.bash" ]; then
    echo -e "${YELLOW}Workspace not built. Building now...${NC}"
    cd "$WORKSPACE_DIR"
    catkin_make
fi

# Source workspace
echo "Sourcing workspace..."
source "$WORKSPACE_DIR/devel/setup.bash"

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"
python3 "$WORKSPACE_DIR/src/unitree_ssc_navigation/scripts/check_dependencies.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}Some dependencies are missing. Please install them first.${NC}"
    exit 1
fi

# Ask user for mode
echo -e "\n${GREEN}Select mode:${NC}"
echo "1) Full system (simulation)"
echo "2) Test DSFNet only"
echo "3) Full system (real robot)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting full system in simulation...${NC}"
        echo -e "${YELLOW}Note: You need to start unitree controller separately:${NC}"
        echo -e "${YELLOW}  Terminal 2: rosrun unitree_guide junior_ctrl${NC}"
        echo ""
        read -p "DSFNet model path (leave empty to skip DSFNet): " model_path

        if [ -z "$model_path" ]; then
            roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
                use_sim:=true \
                use_dsfnet:=false
        else
            roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
                use_sim:=true \
                use_dsfnet:=true \
                dsfnet_model_path:="$model_path"
        fi
        ;;
    2)
        echo -e "\n${GREEN}Starting DSFNet test mode...${NC}"
        read -p "DSFNet model path: " model_path
        read -p "RGB topic [/camera/color/image_raw]: " rgb_topic
        rgb_topic=${rgb_topic:-/camera/color/image_raw}
        read -p "Depth topic [/camera/depth/image_raw]: " depth_topic
        depth_topic=${depth_topic:-/camera/depth/image_raw}

        roslaunch unitree_ssc_navigation test_dsfnet.launch \
            dsfnet_model_path:="$model_path" \
            rgb_topic:="$rgb_topic" \
            depth_topic:="$depth_topic"
        ;;
    3)
        echo -e "\n${GREEN}Starting full system on real robot...${NC}"
        echo -e "${RED}WARNING: Make sure you have:${NC}"
        echo -e "${RED}  1. Configured network (sudo ./ipconfig.sh)${NC}"
        echo -e "${RED}  2. Started real robot interface${NC}"
        echo -e "${RED}  3. Robot is in safe position${NC}"
        read -p "Continue? [y/N]: " confirm

        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Aborted."
            exit 0
        fi

        read -p "DSFNet model path (leave empty to skip DSFNet): " model_path

        if [ -z "$model_path" ]; then
            roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
                use_sim:=false \
                use_dsfnet:=false
        else
            roslaunch unitree_ssc_navigation unitree_ssc_navigation.launch \
                use_sim:=false \
                use_dsfnet:=true \
                dsfnet_model_path:="$model_path"
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
