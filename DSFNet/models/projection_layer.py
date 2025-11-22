#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Projection layer for 2D to 3D voxel conversion
Projects 2D features into 3D voxel space based on depth information
"""

import torch
import torch.nn as nn
import numpy as np


def compute_CP_mega_matrix(voxel_origin, voxel_unit, img_w, img_h, fx, fy, cx, cy, voxel_w, voxel_h, voxel_d):
    """
    Compute Camera Projection mega matrix for projecting pixels to voxels

    Args:
        voxel_origin: Origin of voxel grid in camera frame [x, y, z]
        voxel_unit: Size of each voxel in meters
        img_w, img_h: Image width and height
        fx, fy: Focal lengths
        cx, cy: Principal point
        voxel_w, voxel_h, voxel_d: Voxel grid dimensions

    Returns:
        P: Projection matrix mapping (u,v) pixels to voxel indices [img_h, img_w]
    """
    # Create meshgrid for all pixels
    v, u = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')

    # For each pixel, we need to determine which voxel it projects to at each depth
    # This is a simplified version - stores voxel index for average depth
    # In practice, this should be computed per-sample based on actual depth

    # Placeholder: map each pixel to a voxel index
    # The actual projection depends on depth value
    # This matrix will be used to index into the voxel grid

    P = torch.zeros((img_h, img_w), dtype=torch.long)

    # For now, create a simple mapping
    # In the actual implementation, this would be computed based on depth
    # The matrix stores linear indices into the voxel grid
    for i in range(img_h):
        for j in range(img_w):
            # Back-project pixel to 3D ray (assuming unit depth)
            # This is a simplified version
            z = 1.0  # placeholder depth
            x = (j - cx) * z / fx
            y = (i - cy) * z / fy

            # Convert to voxel coordinates
            vx = int((x - voxel_origin[0]) / voxel_unit)
            vy = int((y - voxel_origin[1]) / voxel_unit)
            vz = int((z - voxel_origin[2]) / voxel_unit)

            # Clamp to voxel grid bounds
            vx = max(0, min(voxel_w - 1, vx))
            vy = max(0, min(voxel_h - 1, vy))
            vz = max(0, min(voxel_d - 1, vz))

            # Convert to linear index
            linear_idx = vx * (voxel_h * voxel_d) + vy * voxel_d + vz
            P[i, j] = linear_idx

    return torch.from_numpy(P.astype(np.int64))


class Project2Dto3D(nn.Module):
    """
    Project 2D features to 3D voxel grid using depth information
    """
    def __init__(self, voxel_w, voxel_h, voxel_d):
        """
        Args:
            voxel_w, voxel_h, voxel_d: Voxel grid dimensions (width, height, depth)
        """
        super(Project2Dto3D, self).__init__()
        self.voxel_w = voxel_w
        self.voxel_h = voxel_h
        self.voxel_d = voxel_d

    def forward(self, features_2d, projection_indices):
        """
        Project 2D features into 3D voxel grid

        Args:
            features_2d: 2D features [B, C, H, W]
            projection_indices: Projection indices [H, W] mapping pixels to voxels

        Returns:
            features_3d: 3D voxel features [B, C, voxel_w, voxel_h, voxel_d]
        """
        B, C, H, W = features_2d.shape

        # Initialize 3D voxel grid
        features_3d = torch.zeros(
            (B, C, self.voxel_w, self.voxel_h, self.voxel_d),
            dtype=features_2d.dtype,
            device=features_2d.device
        )

        # Flatten voxel grid for easier indexing
        features_3d_flat = features_3d.view(B, C, -1)

        # Flatten 2D features
        features_2d_flat = features_2d.view(B, C, H * W)

        # Flatten projection indices
        proj_flat = projection_indices.view(-1)

        # Project features
        # For each pixel, add its features to the corresponding voxel
        # Using scatter_add to handle multiple pixels projecting to same voxel
        for b in range(B):
            for c in range(C):
                features_3d_flat[b, c].scatter_add_(
                    0, proj_flat, features_2d_flat[b, c]
                )

        # Reshape back to 3D
        features_3d = features_3d_flat.view(B, C, self.voxel_w, self.voxel_h, self.voxel_d)

        return features_3d


def project_depth_to_voxels(depth, projection_matrix, voxel_shape):
    """
    Helper function to project depth image to voxel occupancy

    Args:
        depth: Depth image [B, H, W]
        projection_matrix: Projection indices [H, W]
        voxel_shape: (voxel_w, voxel_h, voxel_d)

    Returns:
        voxel_occupancy: Binary occupancy grid [B, voxel_w, voxel_h, voxel_d]
    """
    B, H, W = depth.shape
    voxel_w, voxel_h, voxel_d = voxel_shape

    # Initialize voxel grid
    voxel_occupancy = torch.zeros(
        (B, voxel_w, voxel_h, voxel_d),
        dtype=torch.float32,
        device=depth.device
    )

    # Flatten
    voxel_flat = voxel_occupancy.view(B, -1)
    depth_flat = depth.view(B, H * W)
    proj_flat = projection_matrix.view(-1)

    # Mark occupied voxels
    for b in range(B):
        valid_depth = depth_flat[b] > 0
        valid_indices = proj_flat[valid_depth]
        voxel_flat[b, valid_indices] = 1.0

    return voxel_occupancy.view(B, voxel_w, voxel_h, voxel_d)
