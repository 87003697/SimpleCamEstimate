"""
VGGT模型辅助类
处理VGGT模型推理和点云提取
"""
import torch
import cv2
from typing import List, Optional
from dataclasses import dataclass
import sys
import os

# 添加dust3r路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'dust3r'))

@dataclass
class VGGTResult:
    """VGGT推理结果"""
    reference_pc: torch.Tensor           # 参考图像点云
    rendered_pcs: List[torch.Tensor]     # 渲染图像点云列表
    depth_maps: List[torch.Tensor]       # 深度图列表

class VGGTHelper:
    """VGGT模型辅助类"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载VGGT模型"""
        try:
            # 导入VGGT相关模块
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.device import to_numpy
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            
            # 加载模型
            model_path = "third_party/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(self.device)
            self.model.eval()
            
            # 存储相关函数
            self.inference_fn = inference
            self.to_numpy = to_numpy
            self.make_pairs = make_pairs
            self.global_aligner = global_aligner
            self.GlobalAlignerMode = GlobalAlignerMode
            
            print("✅ VGGT model loaded successfully")
            
        except Exception as e:
            print(f"❌ VGGT model loading failed: {e}")
            self.model = None
    
    def _preprocess_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """预处理图像数据"""
        processed_images = []
        
        for img in images:
            # 确保是torch.Tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            
            # 转换数据类型
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            
            # 调整到512x512
            if img.shape[:2] != (512, 512):
                # 转换为numpy进行resize，因为cv2需要numpy
                img_np = img.cpu().numpy()
                img_np = cv2.resize(img_np, (512, 512))
                img = torch.from_numpy(img_np).to(self.device)
            
            # 确保在正确的设备上
            img = img.to(self.device)
            processed_images.append(img)
        
        return processed_images
    
    def _extract_point_clouds(self, predictions: dict, images: torch.Tensor) -> List[torch.Tensor]:
        """从预测结果中提取点云"""
        point_clouds = []
        
        try:
            # 使用全局对齐器
            scene = self.global_aligner(predictions, device=self.device, mode=self.GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
            
            # 获取点云
            imgs = scene.imgs
            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()
            
            for i in range(len(imgs)):
                if i < len(pts3d):
                    pts = pts3d[i]
                    # 转换为torch tensor
                    if not isinstance(pts, torch.Tensor):
                        pts = torch.from_numpy(pts)
                    
                    # 重塑点云
                    H, W = imgs[i].shape[:2]
                    pts = pts.reshape(H, W, 3)
                    
                    # 选择有效点云
                    valid_mask = torch.isfinite(pts).all(dim=2)
                    valid_pts = pts[valid_mask]
                    
                    if len(valid_pts) > 0:
                        point_clouds.append(valid_pts)
                    else:
                        point_clouds.append(torch.empty(0, 3))
                else:
                    point_clouds.append(torch.empty(0, 3))
            
        except Exception as e:
            print(f"⚠️ Point cloud extraction failed: {e}")
            # 返回空点云
            for _ in range(len(images)):
                point_clouds.append(torch.empty(0, 3))
        
        return point_clouds
    
    def inference(self, reference_image: torch.Tensor, rendered_views: List[torch.Tensor]) -> VGGTResult:
        """执行VGGT推理"""
        if self.model is None:
            print("❌ VGGT model not loaded")
            return VGGTResult(
                reference_pc=torch.empty(0, 3),
                rendered_pcs=[torch.empty(0, 3) for _ in rendered_views],
                depth_maps=[torch.empty(0, 0) for _ in rendered_views]
            )
        
        try:
            # 预处理图像
            all_images = [reference_image] + rendered_views
            processed_images = self._preprocess_images(all_images)
            
            # 创建图像对
            pairs = self.make_pairs(processed_images, scene_graph='complete', prefilter=None, symmetrize=True)
            
            # 执行推理
            predictions = self.inference_fn(pairs, self.model, self.device, batch_size=1)
            
            # 提取点云
            point_clouds = self._extract_point_clouds(predictions, processed_images)
            
            # 分离参考点云和渲染点云
            reference_pc = point_clouds[0] if len(point_clouds) > 0 else torch.empty(0, 3)
            rendered_pcs = point_clouds[1:] if len(point_clouds) > 1 else []
            
            # 生成空的深度图（VGGT不直接提供深度图）
            depth_maps = [torch.empty(0, 0) for _ in rendered_views]
            
            return VGGTResult(
                reference_pc=reference_pc,
                rendered_pcs=rendered_pcs,
                depth_maps=depth_maps
            )
            
        except Exception as e:
            print(f"❌ VGGT inference failed: {e}")
            return VGGTResult(
                reference_pc=torch.empty(0, 3),
                rendered_pcs=[torch.empty(0, 3) for _ in rendered_views],
                depth_maps=[torch.empty(0, 0) for _ in rendered_views]
            ) 