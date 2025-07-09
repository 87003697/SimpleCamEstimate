"""
DUSt3R助手模块
简化封装DUSt3R功能，严格基于原始v2m4_camera_search.py实现
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
from pathlib import Path

# 导入GPU性能监控
from .gpu_profiler import profile_stage

# 设置DUSt3R路径
def setup_dust3r_paths():
    """设置本地DUSt3R核心路径"""
    # 尝试从配置文件获取路径
    try:
        # 将项目根目录添加到sys.path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        sys.path.insert(0, str(project_root))
        
        from config import get_dust3r_paths
        dust3r_config = get_dust3r_paths()
        
        dust3r_core_path = dust3r_config['core_path']
        dust3r_lib_path = dust3r_config['lib_path']
        croco_path = dust3r_config['croco_path']
        
        print(f"🔧 Using DUSt3R paths from config.py")
        
    except ImportError:
        print("⚠️ config.py not found, using default paths")
        # 使用默认路径（原始实现）
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        dust3r_core_path = project_root / "_reference" / "MeshSeriesGen" / "models" / "dust3r" / "dust3r_core"
        dust3r_lib_path = dust3r_core_path / "dust3r"
        croco_path = dust3r_core_path / "croco"
    
    if not dust3r_core_path.exists():
        raise FileNotFoundError(f"DUSt3R core path不存在: {dust3r_core_path}")
    
    paths_to_add = [str(dust3r_core_path), str(dust3r_lib_path), str(croco_path)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return dust3r_core_path

# 初始化DUSt3R路径
DUST3R_CORE_PATH = setup_dust3r_paths()

# DUSt3R导入
import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner

@dataclass
class DUSt3RResult:
    """DUSt3R推理结果"""
    reference_pc: np.ndarray           # 参考图像点云
    rendered_pcs: List[np.ndarray]     # 渲染图像点云列表
    confidences: List[Optional[np.ndarray]]  # 置信度列表
    alignment_loss: float              # 对齐损失

class DUSt3RHelper:
    """DUSt3R的简化封装 - 基于原始v2m4实现"""
    
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model_path = model_path
        self.image_size = 512
        self.model = self._load_dust3r_model()
        
    def _load_dust3r_model(self):
        """加载DUSt3R模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"DUSt3R模型路径不存在: {self.model_path}")
        
        model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        return model
    
    @profile_stage("DUSt3R_Inference")
    def inference(self, reference_image: np.ndarray, 
                 rendered_views: List[np.ndarray]) -> DUSt3RResult:
        """DUSt3R推理 - 核心功能"""
        
        # 1. 图像预处理
        processed_reference = self._preprocess_image_for_dust3r(reference_image)
        processed_views = [self._preprocess_image_for_dust3r(img) for img in rendered_views]
        
        # 2. 保存临时图像文件 (DUSt3R需要文件路径)
        all_images = [processed_reference] + processed_views
        temp_paths = self._save_temp_images(all_images)
        
        # 3. DUSt3R标准流程
        processed_images = load_images(temp_paths, size=self.image_size)
        pairs = make_pairs(processed_images, scene_graph='complete', symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1)
        
        # 4. 全局对齐 - 使用更快的设置
        scene = global_aligner(output, device=self.device)
        loss = scene.compute_global_alignment(niter=300, lr=0.01)
        
        # 5. 提取结果
        point_clouds = scene.get_pts3d()
        confidences = scene.get_conf()
        
        # 6. 清理临时文件
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)
        
        # 7. 构造结果
        reference_pc = point_clouds[0].detach().cpu().numpy()
        rendered_pcs = [pc.detach().cpu().numpy() for pc in point_clouds[1:]]
        confidence_arrays = [conf.detach().cpu().numpy() if conf is not None else None 
                           for conf in confidences]
        
        return DUSt3RResult(
            reference_pc=reference_pc,
            rendered_pcs=rendered_pcs,
            confidences=confidence_arrays,
            alignment_loss=loss
        )
    
    def _preprocess_image_for_dust3r(self, image: np.ndarray) -> np.ndarray:
        """DUSt3R图像预处理 - 基于原始实现"""
        # 确保图像是RGB格式
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"不支持的图像格式: {image.shape}")
        
        processed_img = image.copy()
        
        # 调整尺寸到DUSt3R要求的大小
        if processed_img.shape[:2] != (self.image_size, self.image_size):
            processed_img = cv2.resize(processed_img, (self.image_size, self.image_size), 
                                     interpolation=cv2.INTER_LANCZOS4)
        
        # 确保数据类型
        if processed_img.dtype != np.uint8:
            processed_img = np.clip(processed_img * 255, 0, 255).astype(np.uint8)
        
        return processed_img
    
    def _save_temp_images(self, images: List[np.ndarray]) -> List[str]:
        """保存临时图像文件"""
        # 尝试使用配置文件中的临时目录
        try:
            # 将项目根目录添加到sys.path
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            sys.path.insert(0, str(project_root))
            
            from config import get_output_paths
            output_paths = get_output_paths()
            temp_dir = output_paths['temp_dir']
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_paths = []
            for i, img in enumerate(images):
                temp_path = temp_dir / f'dust3r_temp_{i:03d}.jpg'
                cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                temp_paths.append(str(temp_path))
            return temp_paths
            
        except ImportError:
            # 使用默认的临时文件机制
            temp_paths = []
            for i, img in enumerate(images):
                fd, temp_path = tempfile.mkstemp(suffix=f'_{i:03d}.jpg', prefix='dust3r_')
                os.close(fd)
                cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                temp_paths.append(temp_path)
            return temp_paths 