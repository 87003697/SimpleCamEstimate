"""
简化的Normal预测器 - 为相机搜索算法优化
最小依赖，专注于核心功能
"""

import torch
from PIL import Image
import os
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download

class SimpleNormalPredictor:
    """简化的法线图预测器"""
    
    def __init__(self, device="cuda", weights_dir=None, yoso_version=None):
        # 尝试从配置文件获取配置
        try:
            # 将项目根目录添加到sys.path
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            sys.path.insert(0, str(project_root))
            
            from config import get_stablenormal_config
            config = get_stablenormal_config()
            
            self.device = device if device != "cuda" else config['device']
            self.weights_dir = weights_dir if weights_dir is not None else config['weights_dir']
            self.yoso_version = yoso_version if yoso_version is not None else config['yoso_version']
            
            print(f"🔧 Using StableNormal config from config.py")
            
        except ImportError:
            print("⚠️ config.py not found, using default settings")
            # 使用默认配置
            self.device = device
            self.weights_dir = weights_dir if weights_dir is not None else "./weights"
            self.yoso_version = yoso_version if yoso_version is not None else "yoso-normal-v1-8-1"
        
        self.model = None
        
        # 确保权重目录存在
        os.makedirs(self.weights_dir, exist_ok=True)
    
    def _cache_weights(self) -> None:
        """缓存模型权重"""
        model_id = f"Stable-X/{self.yoso_version}"
        local_path = os.path.join(self.weights_dir, self.yoso_version)
        
        if os.path.exists(local_path):
            print(f"   📁 Model weights already cached at: {local_path}")
            return
        
        print(f"   📥 Downloading model weights: {model_id}")
        snapshot_download(
            repo_id=model_id, 
            local_dir=local_path, 
            force_download=False
        )
        print(f"   ✅ Weights cached at: {local_path}")
    
    def load_model(self):
        """加载StableNormal模型"""
        if self.model is not None:
            return
        
        print("🎨 Loading Normal Predictor...")
        
        # 缓存权重
        self._cache_weights()
        
        # 尝试本地加载
        local_repo_path = os.path.join(
            torch.hub.get_dir(), 
            'hugoycj_StableNormal_main'
        )
        
        if os.path.exists(local_repo_path):
            print("   🔄 Loading from local cache...")
            self.model = torch.hub.load(
                local_repo_path,
                "StableNormal_turbo",
                yoso_version=self.yoso_version,
                source='local',
                local_cache_dir=self.weights_dir,
                device=self.device,  # 修复：传递正确的设备参数
            )
        else:
            print("   🔄 Loading from remote...")
            self.model = torch.hub.load(
                "hugoycj/StableNormal", 
                "StableNormal_turbo",
                trust_remote_code=True, 
                yoso_version=self.yoso_version,
                local_cache_dir=self.weights_dir,
                device=self.device  # 修复：传递正确的设备参数
            )
        
        # 配置模型 - 确保所有组件都在正确设备上
        self.model = self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        print(f"   ✅ StableNormal model loaded successfully on {self.device}")
    
    def predict(self, image: Image.Image, 
                resolution: int = 768, match_input_resolution: bool = True, 
                data_type: str = 'object') -> Image.Image:
        """预测图像的法线图"""
        self.load_model()
        
        # 直接使用PIL Image，无需转换
        with torch.no_grad():
            normal_map = self.model(
                image,
                resolution=resolution,
                match_input_resolution=match_input_resolution,
                data_type=data_type
            )
            return normal_map


# 便捷函数
def create_normal_predictor(device="cuda", weights_dir=None, 
                           yoso_version=None, 
                           load_immediately=False) -> SimpleNormalPredictor:
    """工厂函数：创建normal predictor"""
    predictor = SimpleNormalPredictor(
        device=device, 
        weights_dir=weights_dir, 
        yoso_version=yoso_version
    )
    
    if load_immediately:
        predictor.load_model()
    
    return predictor 