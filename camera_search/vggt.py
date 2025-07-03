# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

# 简化版VGGT - 暂时移除复杂依赖
class VGGTSimplified(nn.Module, PyTorchModelHubMixin):
    """简化版VGGT模型 - 用于测试集成"""
    
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 创建简单的占位符层
        self.dummy_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.dummy_linear = nn.Linear(64, 3)  # 输出3D点
        
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        简化的前向传播 - 返回与真实VGGT相同的接口
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        
        B, S, C, H, W = images.shape
        
        # 创建占位符输出
        predictions = {}
        
        # 简单的卷积处理
        dummy_features = []
        for i in range(S):
            feat = self.dummy_conv(images[0, i])  # [3, H, W] -> [64, H, W]
            feat = torch.mean(feat, dim=[2, 3])   # [64]
            dummy_features.append(feat)
        
        dummy_features = torch.stack(dummy_features)  # [S, 64]
        
        # 生成占位符的世界坐标点
        world_points = torch.randn(B, S, H, W, 3, device=images.device)
        world_points_conf = torch.rand(B, S, H, W, device=images.device)
        
        # 生成占位符的深度图
        depth = torch.rand(B, S, H, W, 1, device=images.device)
        depth_conf = torch.rand(B, S, H, W, device=images.device)
        
        # 生成占位符的相机姿态编码
        pose_enc = torch.randn(B, S, 9, device=images.device)
        
        predictions.update({
            "pose_enc": pose_enc,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "images": images
        })
        
        # 如果有查询点，添加跟踪结果
        if query_points is not None:
            N = query_points.shape[-2]
            track = torch.randn(B, S, N, 2, device=images.device)
            vis = torch.rand(B, S, N, device=images.device)
            conf = torch.rand(B, S, N, device=images.device)
            
            predictions.update({
                "track": track,
                "vis": vis,
                "conf": conf
            })
        
        return predictions

# 为了兼容性，创建别名
VGGT = VGGTSimplified
