#!/usr/bin/env python3
"""
Check dependencies for Unitree SSC Navigation system
"""

import sys
import os

def check_python_packages():
    """Check required Python packages"""
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'rospy': 'ROS Python',
        'sensor_msgs': 'ROS sensor_msgs',
        'geometry_msgs': 'ROS geometry_msgs',
        'nav_msgs': 'ROS nav_msgs',
    }

    missing = []
    print("Checking Python packages...")
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    return missing

def check_dsfnet():
    """Check DSFNet availability"""
    print("\nChecking DSFNet...")
    dsfnet_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'DSFNet'
    )

    if not os.path.exists(dsfnet_path):
        print(f"  ✗ DSFNet not found at {dsfnet_path}")
        return False

    print(f"  ✓ DSFNet found at {dsfnet_path}")

    # Check for required files
    required_files = [
        'models/DDRNet.py',
        'models/DDR.py',
        'config.py',
    ]

    missing_files = []
    for file in required_files:
        full_path = os.path.join(dsfnet_path, file)
        if os.path.exists(full_path):
            print(f"    ✓ {file}")
        else:
            print(f"    ✗ {file} - MISSING")
            missing_files.append(file)

    return len(missing_files) == 0

def check_ros_packages():
    """Check ROS packages"""
    print("\nChecking ROS packages...")
    import subprocess

    required_packages = [
        'ego_planner',
        'unitree_guide',
        'waypoint_generator',
    ]

    missing = []
    for pkg in required_packages:
        try:
            result = subprocess.run(
                ['rospack', 'find', pkg],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"  ✓ {pkg}")
            else:
                print(f"  ✗ {pkg} - NOT FOUND")
                missing.append(pkg)
        except Exception as e:
            print(f"  ✗ {pkg} - ERROR: {e}")
            missing.append(pkg)

    return missing

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print(f"  ⚠ CUDA not available - will use CPU (slow)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

    return True

def main():
    print("=" * 60)
    print("Unitree SSC Navigation - Dependency Checker")
    print("=" * 60)

    # Check Python packages
    missing_python = check_python_packages()

    # Check DSFNet
    dsfnet_ok = check_dsfnet()

    # Check ROS packages
    missing_ros = check_ros_packages()

    # Check CUDA
    cuda_ok = check_cuda()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = True

    if missing_python:
        print(f"✗ Missing Python packages: {', '.join(missing_python)}")
        print("  Install with: pip3 install torch torchvision opencv-python numpy")
        all_ok = False
    else:
        print("✓ All Python packages installed")

    if not dsfnet_ok:
        print("✗ DSFNet not properly set up")
        all_ok = False
    else:
        print("✓ DSFNet available")

    if missing_ros:
        print(f"✗ Missing ROS packages: {', '.join(missing_ros)}")
        print("  Build workspace: catkin_make")
        all_ok = False
    else:
        print("✓ All ROS packages found")

    if not cuda_ok:
        print("⚠ CUDA not available - inference will be slow")
    else:
        print("✓ CUDA available")

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All dependencies satisfied! System ready to use.")
        return 0
    else:
        print("✗ Some dependencies are missing. Please install them.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
