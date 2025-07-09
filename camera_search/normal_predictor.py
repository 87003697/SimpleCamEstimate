"""
简化的Normal预测器 - 为相机搜索算法优化
最小依赖，专注于核心功能
"""

import torch
import numpy as np
from PIL import Image
import cv2

class SimpleNormalPredictor:
    """简化的法线图预测器"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
    
    def load_model(self):
        """加载StableNormal模型"""
        if self.model is not None:
            return
        
        print("🎨 Loading Normal Predictor...")
        
        # 尝试加载高级模型
        try:
            self.model = torch.hub.load(
                "hugoycj/StableNormal", 
                "StableNormal_turbo",
                trust_remote_code=True, 
                yoso_version="yoso-normal-v1-8-1"
            )
            
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            print(f"✅ StableNormal model loaded successfully on {self.device}")
            self.use_advanced_model = True
            
        except Exception as e:
            print(f"   ⚠️ Failed to load StableNormal model: {e}")
            print("   ⚠️ Using basic processing instead")
            self.use_advanced_model = False
            self.model = "basic_processing"
    
    def predict(self, image_pil):
        """预测图像的法线图"""
        self.load_model()
        
        if self.use_advanced_model:
            # 使用高级模型
            with torch.no_grad():
                normal_map = self.model(image_pil)
                return normal_map
        else:
            # 使用基本处理
            return self._basic_normal_prediction(image_pil)
    
    def _basic_normal_prediction(self, image_pil):
        """基本法线图预测 - 基于图像处理"""
        # 转换为numpy数组
        image_np = np.array(image_pil)
        
        # 转换为灰度图
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 构建法线图
        normal_x = grad_x / 255.0
        normal_y = grad_y / 255.0
        normal_z = np.ones_like(normal_x) * 0.5
        
        # 归一化
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x = normal_x / (length + 1e-8)
        normal_y = normal_y / (length + 1e-8)
        normal_z = normal_z / (length + 1e-8)
        
        # 转换到 [0, 1] 范围
        normal_map = np.stack([
            (normal_x + 1) / 2,
            (normal_y + 1) / 2,
            (normal_z + 1) / 2
        ], axis=2)
        
        # 转换为PIL图像
        normal_map = (normal_map * 255).astype(np.uint8)
        return Image.fromarray(normal_map) 