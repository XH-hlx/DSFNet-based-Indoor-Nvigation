#!/bin/bash

# ========================================
# Unitree Go2 Semantic Navigation Simulation
# Quick Start Script
# ========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Unitree Go2 Semantic Navigation${NC}"
echo -e "${BLUE}  Gazebo Simulation Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check workspace
if [ ! -d "$HOME/Desktop/ros1/main/ros1_ws/devel" ]; then
    echo -e "${RED}ERROR: Workspace not found at ~/ros1_ws${NC}"
    echo -e "${YELLOW}Please build the workspace first${NC}"
    exit 1
fi

# Source workspace
echo -e "${GREEN}[1/5] Sourcing ROS workspace...${NC}"
source $HOME/Desktop/ros1/main/ros1_ws/devel/setup.bash

# Check DSFNet model
MODEL_PATH="$HOME/Desktop/ros1/main/ros1_ws/DSFNet/models/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}WARNING: DSFNet model not found at:${NC}"
    echo -e "${YELLOW}  $MODEL_PATH${NC}"
    echo -e "${YELLOW}SSC will run without trained model (random predictions)${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if roscore is running
echo -e "${GREEN}[2/5] Checking roscore...${NC}"
if ! pgrep -x "rosmaster" > /dev/null; then
    echo -e "${YELLOW}roscore not running, starting...${NC}"
    roscore &
    sleep 3
fi

# Kill previous instances
echo -e "${GREEN}[3/5] Cleaning up previous instances...${NC}"
rosnode kill -a 2>/dev/null || true
pkill -9 -f "gzclient|gzserver" 2>/dev/null || true
sleep 2

# Launch simulation
echo -e "${GREEN}[4/5] Launching Go2 simulation...${NC}"
echo -e "${BLUE}This will start:${NC}"
echo -e "  - Gazebo with Go2 model"
echo -e "  - DSFNet semantic scene completion"
echo -e "  - Ego-Planner with semantic costs"
echo -e "  - RViz visualization"
echo ""

# Set parameters
USE_SEMANTIC=${USE_SEMANTIC:-true}
LAMBDA_SEMANTIC=${LAMBDA_SEMANTIC:-5.0}
WORLD=${WORLD:-small_house}

# Robot spawn position (can be customized via environment variables)
SPAWN_X=${SPAWN_X:-7.058412}
SPAWN_Y=${SPAWN_Y:--0.776664}
SPAWN_Z=${SPAWN_Z:-0.6}
SPAWN_YAW=${SPAWN_YAW:-0.0}

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  World: $WORLD"
echo -e "  Semantic Cost: $USE_SEMANTIC"
echo -e "  Lambda Semantic: $LAMBDA_SEMANTIC"
echo -e "  Model: $MODEL_PATH"
echo -e "  Spawn Position: x=$SPAWN_X, y=$SPAWN_Y, z=$SPAWN_Z, yaw=$SPAWN_YAW"
echo ""

# Launch
echo -e "${GREEN}[5/5] Starting system...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

roslaunch unitree_ssc_navigation go2_semantic_navigation_sim.launch \
    dsfnet_model_path:=$MODEL_PATH \
    use_semantic_cost:=$USE_SEMANTIC \
    lambda_semantic:=$LAMBDA_SEMANTIC \
    world_name:=$WORLD \
    spawn_x:=$SPAWN_X \
    spawn_y:=$SPAWN_Y \
    spawn_z:=$SPAWN_Z \
    spawn_yaw:=$SPAWN_YAW \
    rviz:=true

# Cleanup on exit
echo -e "${GREEN}Shutting down...${NC}"
rosnode kill -a
pkill -9 -f "gzclient|gzserver"

echo -e "${GREEN}Done!${NC}"
