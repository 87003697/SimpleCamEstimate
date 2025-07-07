"""
VGGT助手模块 - 基于V2M4实现，使用真实的VGGT模型
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import cv2

@dataclass
class VGGTResult:
    """VGGT推理结果 - 与DUSt3R结果接口兼容"""
    reference_pc: np.ndarray           # 参考图像点云
    rendered_pcs: List[np.ndarray]     # 渲染图像点云列表
    depth_maps: List[np.ndarray]       # 深度图列表
    confidence_scores: List[float]     # 置信度分数

class VGGTHelper:
    """VGGT助手类 - 基于V2M4实现，使用真实的VGGT模型"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.is_loaded = False
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
    def load_model(self):
        """加载真实的VGGT模型 - 基于V2M4实现 (无异常处理版本)"""
        print("🔄 正在加载真实的VGGT模型...")
        
        # 使用V2M4中的正确加载方式 - 直接执行，不捕获异常
        from .vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        
        print("✅ VGGT模型加载成功")
        self.is_loaded = True
        
    def _preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """预处理图像 - 转换为VGGT格式 (基于V2M4实现)"""
        processed_images = []
        
        # VGGT的标准输入尺寸
        target_size = 518  # 与V2M4保持一致
        
        for img in images:
            # 确保图像是RGB格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 转换为tensor并归一化到[0,1]
                if img.dtype == np.uint8:
                    img_tensor = torch.from_numpy(img).float() / 255.0
                else:
                    img_tensor = torch.from_numpy(img).float()
                
                # 调整维度顺序: HWC -> CHW
                img_tensor = img_tensor.permute(2, 0, 1)
                
                # 调整尺寸到target_size
                height, width = img_tensor.shape[1], img_tensor.shape[2]
                
                # 保持宽高比调整到target_size
                if height != target_size or width != target_size:
                    img_tensor = F.interpolate(
                        img_tensor.unsqueeze(0),
                        size=(target_size, target_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)
                
                processed_images.append(img_tensor)
        
        # 堆叠成batch: [S, 3, H, W]
        return torch.stack(processed_images).to(self.device)
    
    def _extract_point_clouds(self, predictions: dict, images: torch.Tensor) -> List[np.ndarray]:
        """从VGGT预测结果中提取点云 - 基于V2M4实现"""
        point_clouds = []
        
        # 获取世界坐标点云 - 与V2M4保持一致的处理方式
        world_points = predictions["world_points"][0].detach()  # [S, H, W, 3]
        
        # 插值到原图像尺寸
        world_points = F.interpolate(
            world_points.permute(0, 3, 1, 2),  # [S, H, W, 3] -> [S, 3, H, W]
            size=images.shape[-1],
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)  # [S, 3, H', W'] -> [S, H', W', 3]
        
        for i, pts in enumerate(world_points):
            # 提取有效点云 (非黑色像素区域) - 与V2M4保持一致
            valid_mask = images[i].permute(1, 2, 0).sum(-1) > 0  # [H, W]
            points = pts.view(-1, 3)[valid_mask.view(-1)]  # 选择有效点
            
            # 转换为numpy
            point_clouds.append(points.cpu().numpy())
        
        return point_clouds
    
    def inference(self, reference_image: np.ndarray, rendered_views: List[np.ndarray]) -> VGGTResult:
        """VGGT推理 - 使用真实的VGGT模型 (无异常处理版本)"""
        
        if not self.is_loaded:
            self.load_model()
            
        # 直接执行，不检查模型状态
        # 1. 预处理图像
        all_images = [reference_image] + rendered_views
        images_tensor = self._preprocess_images(all_images)
        
        # 2. VGGT推理 - 使用与V2M4相同的方式
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images_tensor)
        
        # 3. 提取点云
        point_clouds = self._extract_point_clouds(predictions, images_tensor)
        
        # 4. 构建结果
        reference_pc = point_clouds[0] if len(point_clouds) > 0 else np.array([]).reshape(0, 3)
        rendered_pcs = point_clouds[1:] if len(point_clouds) > 1 else []
        
        # 提取深度图
        depth_maps = []
        if "depth" in predictions:
            depth_tensor = predictions["depth"][0].detach()  # [S, H, W, 1]
            for i in range(depth_tensor.shape[0]):
                depth_map = depth_tensor[i, :, :, 0].cpu().numpy()
                depth_maps.append(depth_map)
        
        # 计算置信度分数
        confidence_scores = []
        if "world_points_conf" in predictions:
            conf_tensor = predictions["world_points_conf"][0].detach()  # [S, H, W]
            for i in range(conf_tensor.shape[0]):
                avg_conf = conf_tensor[i].mean().item()
                confidence_scores.append(avg_conf)
        else:
            confidence_scores = [0.8 for _ in rendered_pcs]
        
        return VGGTResult(
            reference_pc=reference_pc,
            rendered_pcs=rendered_pcs,
            depth_maps=depth_maps,
            confidence_scores=confidence_scores
        ) 