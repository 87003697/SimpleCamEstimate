"""
核心V2M4相机搜索算法实现
包含数据结构、几何工具和主算法类
"""

import os
import sys
import tempfile
import atexit
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import random

# 类型导入
if TYPE_CHECKING:
    from kiui.mesh import Mesh as KiuiMesh

# 导入GPU性能监控
from .gpu_profiler import profile_stage

# Import the functions directly
from .utils import compute_image_similarity_torch, compute_batch_similarity_torch

@dataclass
class CameraPose:
    """简化的相机姿态参数 - 与KiuiKit兼容"""
    elevation: float    # 仰角 (度)，正值表示相机在物体上方  
    azimuth: float      # 方位角 (度)，绕垂直轴旋转
    radius: float       # 相机到目标点的距离
    center_x: float = 0.0    # 目标点x坐标 (相机观察的中心点)
    center_y: float = 0.0    # 目标点y坐标
    center_z: float = 0.0    # 目标点z坐标
    
    @property
    def target_point(self) -> Tuple[float, float, float]:
        """获取目标点坐标，与kiui_mesh_renderer.py兼容"""
        return (self.center_x, self.center_y, self.center_z)
    
    def apply_to_kiui_camera(self, camera) -> None:
        """直接应用到KiuiKit OrbitCamera"""
        camera.from_angle(elevation=self.elevation, azimuth=self.azimuth, is_degree=True)
        camera.radius = self.radius
        camera.center = torch.tensor([self.center_x, self.center_y, self.center_z], dtype=torch.float32)
    
    def get_kiui_render_params(self) -> Dict[str, Any]:
        """获取kiui_mesh_renderer.render_single_view()的参数"""
        return {
            'elevation': self.elevation,
            'azimuth': self.azimuth, 
            'distance': self.radius,
            'target_point': self.target_point
        }
    
    def to_matrix(self) -> torch.Tensor:
        """转换为4x4变换矩阵"""
        # 球坐标到笛卡尔坐标
        elev_rad = torch.deg2rad(torch.tensor(self.elevation))
        azim_rad = torch.deg2rad(torch.tensor(self.azimuth))
        
        x = self.radius * torch.cos(elev_rad) * torch.cos(azim_rad)
        y = self.radius * torch.cos(elev_rad) * torch.sin(azim_rad)
        z = self.radius * torch.sin(elev_rad)
        
        # 构造视图矩阵
        camera_pos = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.tensor([self.center_x, self.center_y, self.center_z], dtype=torch.float32)
        up = torch.tensor([0, 0, 1], dtype=torch.float32)
        
        # Look-at矩阵
        forward = target - camera_pos
        forward = forward / (torch.linalg.norm(forward) + 1e-8)
        
        right = torch.linalg.cross(forward, up)
        right = right / (torch.linalg.norm(right) + 1e-8)
        
        up = torch.linalg.cross(right, forward)
        
        view_matrix = torch.eye(4, dtype=torch.float32)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = camera_pos
        
        return view_matrix
    
    def __str__(self) -> str:
        return f"CameraPose(elev={self.elevation:.1f}°, azim={self.azimuth:.1f}°, r={self.radius:.2f}, center=({self.center_x:.2f}, {self.center_y:.2f}, {self.center_z:.2f}))"

@dataclass
class DataPair:
    """数据对结构 - 适配简化的数据格式"""
    scene_name: str              # 场景名称
    mesh_path: str              # Mesh文件路径: data/meshes/{scene_name}_textured_frame_000000.glb
    image_path: str             # 图像文件路径: data/images/{scene_name}.png
    
    @classmethod
    def from_scene_name(cls, scene_name: str, data_dir: str = "data"):
        """根据场景名称创建数据对"""
        mesh_path = f"{data_dir}/meshes/{scene_name}_textured_frame_000000.glb"
        image_path = f"{data_dir}/images/{scene_name}.png"
        return cls(scene_name, mesh_path, image_path)
    
    def exists(self) -> bool:
        """检查数据对是否存在"""
        return Path(self.mesh_path).exists() and Path(self.image_path).exists()
    
    def load_mesh(self) -> 'KiuiMesh':
        """加载mesh对象 - 使用kiui.Mesh保持原始几何"""
        if not Path(self.mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        
        from kiui.mesh import Mesh as KiuiMesh
        return KiuiMesh.load(
            self.mesh_path,
            resize=False,    # 保持原始尺寸
            renormal=False   # 保持原始法线
        )

class DataManager:
    """数据发现和验证管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.meshes_dir = self.data_dir / "meshes"
        self.images_dir = self.data_dir / "images"
    
    def discover_data_pairs(self) -> List[DataPair]:
        """发现所有可用的数据对"""
        data_pairs = []
        
        if not self.meshes_dir.exists() or not self.images_dir.exists():
            return data_pairs
        
        # 从mesh文件推断场景名称
        for mesh_file in self.meshes_dir.glob("*_textured_frame_000000.glb"):
            scene_name = mesh_file.stem.replace("_textured_frame_000000", "")
            data_pair = DataPair.from_scene_name(scene_name, str(self.data_dir))
            
            if data_pair.exists():
                data_pairs.append(data_pair)
        
        return data_pairs
    
    def validate_data_structure(self) -> Dict:
        """验证数据结构"""
        validation = {
            'total_mesh_files': 0,
            'total_image_files': 0,
            'valid_data_pairs': 0,
            'data_completeness': 0.0,
            'missing_scenes': []
        }
        
        if self.meshes_dir.exists():
            validation['total_mesh_files'] = len(list(self.meshes_dir.glob("*.glb")))
        
        if self.images_dir.exists():
            validation['total_image_files'] = len(list(self.images_dir.glob("*.png")))
        
        data_pairs = self.discover_data_pairs()
        validation['valid_data_pairs'] = len(data_pairs)
        
        if validation['total_mesh_files'] > 0:
            validation['data_completeness'] = (validation['valid_data_pairs'] / validation['total_mesh_files']) * 100
        
        return validation

class GeometryUtils:
    """几何工具 - 只保留核心功能"""
    
    @staticmethod
    def sample_sphere_poses(num_samples: int) -> List[CameraPose]:
        """球面等面积采样"""
        poses = []
        for _ in range(num_samples):
            # 等面积采样
            u, v = torch.rand(2)
            elevation = torch.rad2deg(torch.asin(2 * u - 1))  # [-90°, 90°]
            azimuth = 360 * v  # [0°, 360°]
            
            # 距离采样 (平方根分布)
            radius = 1.0 + 4.0 * torch.sqrt(torch.rand(1))
            
            # 轻微的中心点随机化
            center_x = torch.rand(1) * 1.0 - 0.5  # [-0.5, 0.5]
            center_y = torch.rand(1) * 1.0 - 0.5
            center_z = torch.rand(1) * 1.0 - 0.5
            
            poses.append(CameraPose(
                elevation=elevation.item(),
                azimuth=azimuth.item(),
                radius=radius.item(),
                center_x=center_x.item(),
                center_y=center_y.item(),
                center_z=center_z.item()
            ))
        
        return poses
    
    @staticmethod
    def align_pointclouds_simple(reference_pc: torch.Tensor, 
                               rendered_pcs: List[torch.Tensor],
                               poses: List[CameraPose]) -> Optional[CameraPose]:
        """简化的点云对齐方法"""
        if len(rendered_pcs) == 0:
            return None
        
        reference_pc = GeometryUtils._clean_pointcloud(reference_pc)
        if reference_pc is None:
            return None
        
        best_pose = None
        best_score = float('inf')
        
        for pc, pose in zip(rendered_pcs, poses):
            pc = GeometryUtils._clean_pointcloud(pc)
            if pc is None:
                continue
            
            score = GeometryUtils._chamfer_distance(reference_pc, pc)
            if score < best_score:
                best_score = score
                best_pose = pose
        
        return best_pose
    
    @staticmethod
    def _clean_pointcloud(pc: torch.Tensor) -> Optional[torch.Tensor]:
        """清理点云数据，移除无效点"""
        if pc is None:
            return None
        
        # 统一转换为tensor
        if not isinstance(pc, torch.Tensor):
            if hasattr(pc, 'shape') and hasattr(pc, 'astype'):
                # numpy array
                pc = torch.from_numpy(pc.astype('float32'))
            else:
                return None
        
        # 检查形状
        if pc.dim() != 2 or pc.shape[1] != 3:
            return None
        
        # 检查大小
        if pc.shape[0] < 10:
            return None
        
        # 移除无效值
        valid_mask = torch.isfinite(pc).all(dim=1)
        pc = pc[valid_mask]
        
        if pc.shape[0] < 10:
            return None
        
        return pc
    
    @staticmethod
    def _chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
        """计算倒角距离"""
        if pc1.shape[0] == 0 or pc2.shape[0] == 0:
            return float('inf')
        
        # 确保在同一设备上
        device = pc1.device
        pc2 = pc2.to(device)
        
        # 采样以提高效率
        if pc1.shape[0] > 1000:
            indices = torch.randperm(pc1.shape[0])[:1000]
            pc1 = pc1[indices]
        
        if pc2.shape[0] > 1000:
            indices = torch.randperm(pc2.shape[0])[:1000]
            pc2 = pc2[indices]
        
        # 计算距离矩阵
        # pc1: (N, 3), pc2: (M, 3)
        # dist_matrix: (N, M)
        dist_matrix = torch.cdist(pc1, pc2)
        
        # 倒角距离
        forward = torch.mean(torch.min(dist_matrix, dim=1)[0])
        backward = torch.mean(torch.min(dist_matrix, dim=0)[0])
        
        return (forward + backward).item() / 2.0

class MeshRenderer:
    """Mesh渲染器 - 直接使用kiui.Mesh对象"""
    
    def __init__(self, device: str = "cuda", render_mode: str = "lambertian"):
        self.device = device
        self.render_mode = render_mode
        self._renderer = None
        
        # 验证渲染模式
        valid_modes = ['lambertian', 'normal', 'textured', 'depth']
        if self.render_mode not in valid_modes:
            print(f"⚠️ Warning: Invalid render mode '{self.render_mode}', using 'lambertian'")
            self.render_mode = 'lambertian'
        
        print(f"🎨 MeshRenderer initialized with render_mode: '{self.render_mode}'")
        
        # 检查依赖
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查必要的依赖"""
        # 检查KiuiKit
        from kiui.mesh import Mesh as KiuiMesh
        from kiui.cam import OrbitCamera
        self.kiui_available = True
        
        # 检查nvdiffrast
        import nvdiffrast.torch as dr
        self.nvdiffrast_available = True
        
        # 检查StandardKiuiRenderer
        sys.path.append(str(Path(__file__).parent.parent / "_reference" / "MeshSeriesGen" / "tools" / "visualization"))
        from kiui_mesh_renderer import StandardKiuiRenderer
        self.renderer_available = True
    
    @property
    def renderer(self):
        """延迟初始化渲染器"""
        if self._renderer is None:
            from kiui_mesh_renderer import StandardKiuiRenderer
            self._renderer = StandardKiuiRenderer(
                width=512, 
                height=512, 
                background_color=(255, 255, 255),
                device=self.device
            )
        return self._renderer
    
    @profile_stage("Single_View_Render")
    def render_single_view(self, mesh: 'KiuiMesh', pose: CameraPose) -> torch.Tensor:
        """渲染单个视图 - 直接使用kiui.Mesh对象"""
        rendered_img = self.renderer.render_single_view(
            loaded_mesh_obj=mesh,
            elevation=pose.elevation,
            azimuth=pose.azimuth,
            distance=pose.radius,
            target_point=pose.target_point,
            render_mode=self.render_mode
        )
        
        if rendered_img is None:
            raise RuntimeError(f"Rendering failed: {pose}")
        
        return rendered_img
    
    @profile_stage("Batch_View_Render")
    def render_batch_views(self, mesh: 'KiuiMesh', poses: List[CameraPose], 
                          max_batch_size: int = 8) -> List[torch.Tensor]:
        """批量渲染多个视图 - 直接使用kiui.Mesh对象"""
        # 准备批量渲染参数
        camera_params = []
        for pose in poses:
            camera_params.append({
                'elevation': pose.elevation,
                'azimuth': pose.azimuth,
                'distance': pose.radius,
                'target_point': pose.target_point
            })
        
        # 执行批量渲染
        rendered_images = self.renderer.render_batch_views(
            loaded_mesh_obj=mesh,
            camera_params=camera_params,
            render_mode=self.render_mode,
            max_batch_size=max_batch_size
        )
        
        # 检查结果并添加有效图像
        valid_images = []
        for i, img in enumerate(rendered_images):
            if img is None:
                raise RuntimeError(f"Batch rendering failed at index {i}")
            valid_images.append(img)
        
        return valid_images

class CleanV2M4CameraSearch:
    """清理版V2M4相机搜索算法 - 专注于核心功能"""
    
    def __init__(self, dust3r_model_path: str, device: str = "cuda", enable_visualization: bool = True):
        self.dust3r_model_path = dust3r_model_path
        self.device = device
        self.enable_visualization = enable_visualization
        
        # 优化后的配置参数 - 恢复到更快的设置
        self.config = {
            'initial_samples': 128,       # 初始采样数 (海选阶段) - 恢复: 512→128 (提升速度)
            'top_n': 7,                   # 候选数 (几何解密)
            'pso_particles': 50,          # PSO粒子数 (全局优化) - 恢复: 80→50 (提升速度)
            'pso_iterations': 20,         # PSO迭代数 (全局优化)
            'grad_iterations': 100,       # 梯度下降迭代数 (精细调整) - 恢复: 200→100 (提升速度)
            'image_size': 512,            # 图像尺寸
            'render_batch_size': 16,      # 批量渲染大小 - 恢复: 32→16 (平衡性能和内存)
            'max_batch_size': 8,          # 渲染器最大批量大小 (用于Top-N选择和PSO搜索)
            
            # 新增的优化参数
            'dust3r_alignment_iterations': 300,  # DUSt3R对齐迭代数 - 恢复: 1000→300 (提升速度)
            'pso_w': 0.6,                        # PSO惯性权重 - 优化: 0.7→0.6
            'pso_c1': 1.0,                       # PSO个体学习因子 - 优化: 1.5→1.0
            'top_k_for_pso': 100,                # PSO选择的top-k候选
            'point_cloud_sample_ratio': 0.05,    # 点云采样比例
            'min_confidence': 0.3,               # 最小置信度阈值
            
            # 模型选择配置
            'use_dust3r': True,                  # 是否使用DUSt3R模型
            'model_name': 'dust3r',              # 模型名称标识
            'skip_model_step': False,            # 是否跳过模型估计步骤
            
            # 性能优化配置
            'use_batch_optimization': True,      # 是否使用批量优化（PSO和梯度下降）
            
            # 渲染配置
            'render_mode': 'lambertian'          # 渲染模式: 'lambertian', 'normal', 'textured', 'depth'
        }
        
        # 延迟初始化组件
        self._dust3r_helper = None
        self._renderer = None
        self._optimizer = None
        self._visualizer = None
        self._normal_predictor = None
        
        # 根据配置更新模型名称
        if self.config['use_dust3r']:
            self.config['model_name'] = 'dust3r'
        else:
            self.config['model_name'] = 'dust3r' # 确保默认使用dust3r
        
        # 可视化数据收集
        self.visualization_data = {
            'progression': [],
            'mesh_info': {},
            'algorithm_stats': {}
        }
    
    @property
    def dust3r_helper(self):
        """延迟初始化DUSt3R助手"""
        if self._dust3r_helper is None:
            from .dust3r_helper import DUSt3RHelper
            self._dust3r_helper = DUSt3RHelper(self.dust3r_model_path, self.device)
        return self._dust3r_helper

    @property
    def renderer(self):
        """延迟初始化渲染器"""
        if self._renderer is None:
            self._renderer = MeshRenderer(self.device, self.config['render_mode'])
        return self._renderer
    
    @property
    def normal_predictor(self):
        """延迟初始化Normal预测器"""
        if self._normal_predictor is None:
            from .normal_predictor import SimpleNormalPredictor
            self._normal_predictor = SimpleNormalPredictor(self.device)
        return self._normal_predictor
    
    @property
    def optimizer(self):
        """优化器 - 懒加载"""
        if self._optimizer is None:
            from .optimizer import OptimizerManager
            
            # 使用优化后的PSO参数
            pso_params = {
                'num_particles': self.config['pso_particles'],
                'max_iterations': self.config['pso_iterations'], 
                'w': 0.6,      # 优化：惯性权重 0.7→0.6
                'c1': 1.0,     # 优化：个体学习因子 1.5→1.0
                'c2': 1.5      # 保持：社会学习因子
            }
            
            gd_params = {
                'learning_rate': 0.01,
                'max_iterations': self.config['grad_iterations'],
                'tolerance': 1e-6
            }
            
            self._optimizer = OptimizerManager(pso_params=pso_params, gd_params=gd_params)
        
        return self._optimizer
    
    @property
    def visualizer(self):
        """延迟初始化可视化器"""
        if self._visualizer is None and self.enable_visualization:
            from .visualization import V2M4Visualizer
            self._visualizer = V2M4Visualizer(output_dir="outputs/visualization")
        return self._visualizer
    
    @profile_stage("V2M4_Algorithm")
    def search_camera_pose(self, data_pair: DataPair, save_visualization: bool = True, use_normal: bool = False) -> CameraPose:
        """主算法入口 - V2M4的9个核心步骤"""
        
        import time
        start_time = time.time()
        
        print(f"🎬 Starting V2M4 camera search algorithm...")
        print(f"   Scene: {data_pair.scene_name}")
        print(f"   Mesh: {data_pair.mesh_path}")
        print(f"   Image: {data_pair.image_path}")
        
        # 如果使用normal模式，自动设置渲染器为normal模式
        if use_normal:
            print("🎨 Using Normal prediction mode")
            self.config['render_mode'] = 'normal'
            # 重置渲染器以使用新的模式
            self._renderer = None
        
        # 验证数据存在
        if not data_pair.exists():
            raise FileNotFoundError(f"Data pair incomplete: {data_pair.scene_name}")
        
        # 加载数据
        reference_image = self._load_image(data_pair.image_path, use_normal=use_normal)
        mesh = data_pair.load_mesh()
        
        # 收集mesh信息用于可视化
        if self.enable_visualization:
            # 获取mesh的aabb边界框 (返回tuple: (min_bounds, max_bounds))
            aabb_bounds = mesh.aabb()
            aabb_min, aabb_max = aabb_bounds
            mesh_center = (aabb_min + aabb_max) / 2
            mesh_scale = torch.linalg.norm(aabb_max - aabb_min)
            
            self.visualization_data['mesh_info'] = {
                'vertices_count': len(mesh.v),
                'faces_count': len(mesh.f),
                'bounds': [aabb_min.tolist(), aabb_max.tolist()],
                'center': mesh_center.tolist(),
                'scale': float(mesh_scale)
            }
        
        # 步骤1: 采样初始相机pose
        print("📐 Step 1: Sphere sampling camera poses...")
        initial_poses = self._sample_sphere_poses()
        
        # 步骤2: 渲染并选择top-n
        print("🎯 Step 2: Selecting top-n candidate poses...")
        top_poses = self._select_top_poses(mesh, reference_image, initial_poses)
        
        # 可视化：记录top-1姿态
        if self.enable_visualization and top_poses:
            rendered_top1 = self.renderer.render_single_view(mesh, top_poses[0])
            similarity_top1 = compute_image_similarity_torch(reference_image, rendered_top1)
            self.visualization_data['progression'].append({
                'step_name': 'Top-1 Selection',
                'pose': top_poses[0],
                'rendered_image': rendered_top1,
                'similarity': similarity_top1,
                'score': similarity_top1
            })
        
        # 步骤3-4: 模型估计 (几何约束 - DUSt3R或VGGT)
        if self.config.get('skip_model_step', False):
            print(f"🔄 Step 3-4: Skipping model estimation step...")
        else:
            print(f"🔍 Step 3-4: {self.config['model_name'].upper()} geometric constraint estimation...")
        model_pose = self._model_estimation(mesh, reference_image, top_poses)
        
        # 可视化：记录模型结果
        if self.enable_visualization and model_pose:
            rendered_model = self.renderer.render_single_view(mesh, model_pose)
            similarity_model = compute_image_similarity_torch(reference_image, rendered_model)
            self.visualization_data['progression'].append({
                'step_name': f'{self.config["model_name"].upper()} Align',
                'pose': model_pose,
                'rendered_image': rendered_model,
                'similarity': similarity_model,
                'score': similarity_model
            })
        
        # 步骤5-6: PSO搜索
        print("🔍 Step 5-6: PSO particle swarm optimization...")
        pso_pose = self._pso_search(mesh, reference_image, model_pose, top_poses)
        
        # 可视化：记录PSO结果
        if self.enable_visualization:
            rendered_pso = self.renderer.render_single_view(mesh, pso_pose)
            similarity_pso = compute_image_similarity_torch(reference_image, rendered_pso)
            self.visualization_data['progression'].append({
                'step_name': 'PSO Optimize',
                'pose': pso_pose,
                'rendered_image': rendered_pso,
                'similarity': similarity_pso,
                'score': similarity_pso
            })
        
        # 步骤7-8: 梯度下降精化
        print("🎯 Step 7-8: Gradient descent refinement...")
        final_pose = self._gradient_refinement(mesh, reference_image, pso_pose)
        
        # 🚀 算法执行完成，计算纯算法执行时间
        algorithm_execution_time = time.time() - start_time
        
        # 收集算法统计信息
        if self.enable_visualization:
            final_rendered = self.renderer.render_single_view(mesh, final_pose)
            final_similarity = compute_image_similarity_torch(reference_image, final_rendered)
            
            self.visualization_data['progression'].append({
                'step_name': 'Final Result',
                'pose': final_pose,
                'rendered_image': final_rendered,
                'similarity': final_similarity,
                'score': final_similarity
            })
            
            self.visualization_data['algorithm_stats'] = {
                'initial_samples': self.config['initial_samples'],
                'top_n': self.config['top_n'],
                'pso_iterations': self.config['pso_iterations'],
                'final_score': final_similarity
            }
        
        # 生成可视化结果（不计入算法执行时间）
        if self.enable_visualization and save_visualization and self.visualizer:
            print("🎨 Generating visualization results...")
            # 创建结果对比图
            final_rendered = self.renderer.render_single_view(mesh, final_pose)
            
            # 转换tensor为numpy数组供可视化使用
            ref_img_np = self._tensor_to_numpy(reference_image)
            rendered_img_np = self._tensor_to_numpy(final_rendered)
            
            comparison_path = self.visualizer.create_result_comparison(
                data_pair=data_pair,
                reference_image=ref_img_np,
                rendered_result=rendered_img_np,
                final_pose=final_pose,
                mesh_info=self.visualization_data['mesh_info'],
                algorithm_stats=self.visualization_data['algorithm_stats'],
                execution_time=algorithm_execution_time
            )
            
            # 创建优化过程可视化
            if self.visualization_data['progression']:
                # 转换progression数据中的图像
                progression_data_np = []
                for step_data in self.visualization_data['progression']:
                    step_data_np = step_data.copy()
                    if 'rendered_image' in step_data_np and step_data_np['rendered_image'] is not None:
                        step_data_np['rendered_image'] = self._tensor_to_numpy(step_data_np['rendered_image'])
                    progression_data_np.append(step_data_np)
                
                progression_path = self.visualizer.create_pose_progression_visualization(
                    data_pair=data_pair,
                    reference_image=ref_img_np,
                    progression_data=progression_data_np,
                    final_pose=final_pose
                )
            
            # 保存单独的结果图像
            individual_paths = self.visualizer.save_individual_results(
                data_pair=data_pair,
                reference_image=ref_img_np,
                rendered_result=rendered_img_np
            )
        
        print("✅ V2M4 camera search completed!")
        print(f"⚡ Algorithm execution time: {algorithm_execution_time:.2f} seconds")
        
        return final_pose
    
    def _load_image(self, image_path: str, use_normal: bool = False) -> torch.Tensor:
        """加载图像文件，可选择转换为法线图"""
        # 统一使用PIL加载图像
        image_pil = Image.open(image_path).convert('RGB')

        if use_normal:
            # 使用Normal predictor生成法线图
            print("   🎨 Converting to normal map...")
            image_pil = self.normal_predictor.predict(image_pil)
        
        # 🚀 优化：直接PIL→torch转换，避免numpy
        import torchvision.transforms as transforms
        pil_to_tensor = transforms.PILToTensor()
        image_tensor = pil_to_tensor(image_pil).float()
        
        # 转换为HWC格式
        image_tensor = image_tensor.permute(1, 2, 0)
        
        return image_tensor
    
    def _sample_sphere_poses(self) -> List[CameraPose]:
        """步骤1: 球面等面积采样"""
        return GeometryUtils.sample_sphere_poses(self.config['initial_samples'])
    
    @profile_stage("Top_N_Selection")
    def _select_top_poses(self, mesh: 'KiuiMesh', reference_image: torch.Tensor, 
                         poses: List[CameraPose]) -> List[CameraPose]:
        """步骤2: 基于相似度选择top-n - 优化批量渲染以避免nvdiffrast卡住"""
        
        print(f"   Evaluating {len(poses)} candidate poses...")
        print(f"   Using max batch size: {self.config.get('max_batch_size', 8)}")
        
        scores = []
        batch_size = self.config['render_batch_size']  # 使用配置的批量渲染大小
        max_batch_size = self.config.get('max_batch_size', 8)  # 获取最大批量大小
        
        # 使用no_grad优化性能，因为top-n选择不需要梯度
        with torch.no_grad():
            # 尝试批量渲染
            for i in range(0, len(poses), batch_size):
                batch_poses = poses[i:i+batch_size]
                print(f"   Batch rendering {i+1}-{min(i+batch_size, len(poses))}/{len(poses)}...")
                
                # 批量渲染这一组，使用配置的max_batch_size
                rendered_images = self.renderer.render_batch_views(mesh, batch_poses, max_batch_size)
                
                # 🚀 批量计算相似度分数 - 性能优化！
                batch_scores = compute_batch_similarity_torch(reference_image, rendered_images)
                scores.extend(batch_scores)
                
                # 清理GPU内存
                from .utils import cleanup_gpu_memory
                cleanup_gpu_memory()
        
        # 选择top-n
        if len(scores) != len(poses):
            print(f"   ⚠️ Score count ({len(scores)}) does not match pose count ({len(poses)})")
            scores = scores[:len(poses)] + [float('inf')] * (len(poses) - len(scores))
        
        top_indices = torch.argsort(torch.tensor(scores))[:self.config['top_n']].tolist()
        selected_poses = [poses[i] for i in top_indices]
        
        print(f"   ✅ Selected {len(selected_poses)} best poses")
        print(f"   Best score: {min(scores):.4f}")
        
        return selected_poses
    
    @profile_stage("DUSt3R_Estimation")
    def _model_estimation(self, mesh: 'KiuiMesh', reference_image: torch.Tensor, 
                         top_poses: List[CameraPose]) -> Optional[CameraPose]:
        """步骤3-4: 模型估计 - 核心几何约束 (DUSt3R或VGGT)"""
        
        # 检查是否跳过模型估计步骤
        if self.config.get('skip_model_step', False):
            print("   🔄 Skipping model estimation step (as configured)")
            # 返回最佳的top pose作为模型估计结果
            return top_poses[0] if top_poses else None
        
        # 1. 渲染top poses
        rendered_views = [self.renderer.render_single_view(mesh, pose) for pose in top_poses]
        
        # 2. 根据配置选择模型进行推理
        if self.config['use_dust3r']:
            # 使用DUSt3R模型
            model_result = self.dust3r_helper.inference(reference_image, rendered_views)
        else:
            # 使用DUSt3R模型 (默认)
            model_result = self.dust3r_helper.inference(reference_image, rendered_views)
        
        # 3. 点云对齐 (简化版)
        best_pose = GeometryUtils.align_pointclouds_simple(
            model_result.reference_pc,
            model_result.rendered_pcs, 
            top_poses[:len(model_result.rendered_pcs)]
        )
        
        return best_pose
    
    @profile_stage("PSO_Search")
    def _pso_search(self, mesh: 'KiuiMesh', reference_image: torch.Tensor,
                   model_pose: Optional[CameraPose], top_poses: List[CameraPose]) -> CameraPose:
        """步骤5-6: PSO搜索 - 支持批量和传统优化"""
        
        # 准备初始候选
        candidates = top_poses[:self.config['pso_particles']]
        if model_pose is not None:
            candidates.append(model_pose)
        
        if not candidates:
            return CameraPose(elevation=0, azimuth=0, radius=2.5)
        
        # 参数边界
        bounds = {
            'elevation': (-90, 90),
            'azimuth': (0, 360),
            'radius': (1.0, 5.0),
            'center_x': (-1.0, 1.0),
            'center_y': (-1.0, 1.0),
            'center_z': (-1.0, 1.0)
        }
        
        # 根据配置选择优化方式
        if self.config.get('use_batch_optimization', True):
            # 使用批量优化
            def batch_objective(poses: List[CameraPose]) -> List[float]:
                with torch.no_grad():  # PSO优化不需要梯度，添加no_grad提升性能
                    max_batch_size = self.config.get('max_batch_size', 8)
                    rendered_images = self.renderer.render_batch_views(mesh, poses, max_batch_size)
                    return compute_batch_similarity_torch(reference_image, rendered_images)
            
            return self.optimizer.pso_optimize_batch(batch_objective, candidates, bounds)
        else:
            # 使用传统单次优化
            def objective(pose: CameraPose) -> float:
                with torch.no_grad():  # PSO优化不需要梯度，添加no_grad提升性能
                    rendered = self.renderer.render_single_view(mesh, pose)
                    return compute_image_similarity_torch(reference_image, rendered)
            
            return self.optimizer.pso_optimize(objective, candidates, bounds)
    
    @profile_stage("Gradient_Refinement")
    def _gradient_refinement(self, mesh: 'KiuiMesh', reference_image: torch.Tensor,
                           initial_pose: CameraPose) -> CameraPose:
        """步骤7-8: 梯度下降精化 - 支持批量和传统优化"""
        
        # 根据配置选择优化方式
        if self.config.get('use_batch_optimization', True):
            # 使用批量优化
            def batch_objective(poses: List[CameraPose]) -> List[float]:
                max_batch_size = self.config.get('max_batch_size', 8)
                rendered_images = self.renderer.render_batch_views(mesh, poses, max_batch_size)
                return compute_batch_similarity_torch(reference_image, rendered_images)
            
            return self.optimizer.gradient_descent_batch(batch_objective, initial_pose)
        else:
            # 使用传统单次优化
            def objective(pose: CameraPose) -> float:
                rendered = self.renderer.render_single_view(mesh, pose)
                return compute_image_similarity_torch(reference_image, rendered)
            
            return self.optimizer.gradient_descent(objective, initial_pose)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将torch.Tensor转换为numpy数组供可视化使用"""
        if tensor is None:
            return None
        
        if isinstance(tensor, torch.Tensor):
            img_np = tensor.detach().cpu().numpy()
        else:
            img_np = np.array(tensor)
        
        # 确保是HWC格式
        if img_np.ndim == 4:
            img_np = img_np.squeeze(0)
        
        if img_np.shape[0] == 3:  # CHW -> HWC
            img_np = img_np.transpose(1, 2, 0)
        
        # 智能处理数据类型和数值范围
        if img_np.dtype == np.uint8:
            # 已经是uint8，直接使用
            return img_np
        else:
            # 检查数值范围来决定如何转换
            img_min, img_max = img_np.min(), img_np.max()
            
            if img_max <= 1.0:
                # 数值在[0,1]范围，需要乘以255
                img_np = img_np * 255.0
            elif img_max <= 255.0:
                # 数值在[0,255]范围，直接使用
                pass
            else:
                # 数值超出255，需要归一化到[0,255]
                img_np = (img_np - img_min) / (img_max - img_min) * 255.0
            
            # 确保严格的[0,255]范围并转换为uint8
            # 使用更严格的舍入和clipping来避免精度问题
            img_np = np.clip(np.round(img_np), 0, 255).astype(np.uint8)
            
            # 额外检查：确保没有超出范围的值
            if img_np.min() < 0 or img_np.max() > 255:
                # 如果仍然超出范围，进行强制归一化
                img_np = np.clip(img_np, 0, 255)
        
        return img_np 