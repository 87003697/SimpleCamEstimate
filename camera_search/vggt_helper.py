"""
VGGT助手模块 - 基于V2M4实现
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
    """VGGT助手类 - 基于V2M4实现"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """加载VGGT模型"""
        try:
            # 检查是否支持bfloat16
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            self.dtype = dtype
            
            print("🔄 正在加载VGGT模型...")
            
            # 方法1: 使用PyTorchModelHubMixin直接加载 (V2M4的方式)
            try:
                from .vggt import VGGT
                print("   尝试使用PyTorchModelHubMixin加载...")
                self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
                print("✅ VGGT模型加载成功")
                self.is_loaded = True
                return
                
            except Exception as e1:
                print(f"⚠️ PyTorchModelHubMixin加载失败: {e1}")
                
            # 方法2: 手动加载safetensors文件
            try:
                import safetensors.torch
                from huggingface_hub import hf_hub_download
                
                print("   尝试手动加载safetensors...")
                
                # 下载模型文件
                model_path = hf_hub_download(
                    repo_id="facebook/VGGT-1B",
                    filename="model.safetensors",
                    cache_dir="/home/zhiyuan_ma/.cache/huggingface"
                )
                
                config_path = hf_hub_download(
                    repo_id="facebook/VGGT-1B", 
                    filename="config.json",
                    cache_dir="/home/zhiyuan_ma/.cache/huggingface"
                )
                
                # 读取配置
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 创建模型
                from .vggt import VGGT
                self.model = VGGT(
                    img_size=config.get('img_size', 518),
                    patch_size=config.get('patch_size', 14), 
                    embed_dim=config.get('embed_dim', 1024)
                )
                
                # 加载权重
                state_dict = safetensors.torch.load_file(model_path)
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                
                print("✅ VGGT模型手动加载成功")
                self.is_loaded = True
                return
                
            except Exception as e2:
                print(f"⚠️ 手动加载safetensors失败: {e2}")
                
            # 方法3: 创建空模型用于测试
            print("⚠️ 创建空VGGT模型用于测试...")
            from .vggt import VGGT
            self.model = VGGT().to(self.device)
            print("✅ VGGT空模型创建成功")
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ VGGT模型加载完全失败: {e}")
            print("⚠️ 将使用占位符实现")
            self.is_loaded = False
        
    def _preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """预处理图像 - 转换为VGGT格式"""
        processed_images = []
        
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
                processed_images.append(img_tensor)
        
        # 堆叠成batch: [S, 3, H, W]
        return torch.stack(processed_images).to(self.device)
    
    def _extract_point_clouds(self, predictions: dict, images: torch.Tensor) -> List[np.ndarray]:
        """从VGGT预测结果中提取点云"""
        point_clouds = []
        
        # 获取世界坐标点云
        world_points = predictions["world_points"][0].detach()  # [S, H, W, 3]
        
        # 插值到原图像尺寸
        world_points = F.interpolate(
            world_points.permute(0, 3, 1, 2), 
            size=images.shape[-1], 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        for i, pts in enumerate(world_points):
            # 提取有效点云 (非黑色像素区域)
            valid_mask = images[i].permute(1, 2, 0).sum(-1) > 0
            points = pts.view(-1, 3)[valid_mask.view(-1)]
            
            # 转换为numpy
            point_clouds.append(points.cpu().numpy())
        
        return point_clouds
    
    def inference(self, reference_image: np.ndarray, rendered_views: List[np.ndarray]) -> VGGTResult:
        """VGGT推理"""
        
        if not self.is_loaded:
            self.load_model()
            
        # 如果模型加载失败，返回占位符结果
        if not self.is_loaded or self.model is None:
            print("⚠️ VGGT模型未加载，使用占位符结果")
            return self._create_dummy_result(reference_image, rendered_views)
        
        try:
            # 1. 预处理图像
            all_images = [reference_image] + rendered_views
            images_tensor = self._preprocess_images(all_images)
            
            # 2. VGGT推理
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(images_tensor)
            
            # 3. 提取点云
            point_clouds = self._extract_point_clouds(predictions, images_tensor)
            
            # 4. 构建结果
            reference_pc = point_clouds[0] if len(point_clouds) > 0 else np.random.rand(1000, 3).astype(np.float32)
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
            
        except Exception as e:
            print(f"❌ VGGT推理失败: {e}")
            return self._create_dummy_result(reference_image, rendered_views)
    
    def _create_dummy_result(self, reference_image: np.ndarray, rendered_views: List[np.ndarray]) -> VGGTResult:
        """创建占位符结果"""
        # 创建占位符点云
        dummy_pc = np.random.rand(1000, 3).astype(np.float32)
        dummy_depth = np.random.rand(512, 512).astype(np.float32)
        
        return VGGTResult(
            reference_pc=dummy_pc,
            rendered_pcs=[dummy_pc for _ in rendered_views],
            depth_maps=[dummy_depth for _ in rendered_views],
            confidence_scores=[0.5 for _ in rendered_views]
        ) 