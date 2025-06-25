"""
核心V2M4相机搜索算法实现
包含数据结构、几何工具和主算法类
"""

import os
import sys
import tempfile
import atexit
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import trimesh
from pathlib import Path
from scipy.spatial.distance import cdist
import random

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
        camera.center = np.array([self.center_x, self.center_y, self.center_z], dtype=np.float32)
    
    def get_kiui_render_params(self) -> Dict[str, Any]:
        """获取kiui_mesh_renderer.render_single_view()的参数"""
        return {
            'elevation': self.elevation,
            'azimuth': self.azimuth, 
            'distance': self.radius,
            'target_point': self.target_point
        }
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4变换矩阵"""
        # 球坐标到笛卡尔坐标
        elev_rad = np.radians(self.elevation)
        azim_rad = np.radians(self.azimuth)
        
        x = self.radius * np.cos(elev_rad) * np.cos(azim_rad)
        y = self.radius * np.cos(elev_rad) * np.sin(azim_rad)
        z = self.radius * np.sin(elev_rad)
        
        # 构造视图矩阵
        camera_pos = np.array([x, y, z])
        target = np.array([self.center_x, self.center_y, self.center_z])
        up = np.array([0, 0, 1])
        
        # Look-at矩阵
        forward = target - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = camera_pos
        
        return view_matrix.astype(np.float32)
    
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
    
    def load_mesh(self) -> trimesh.Trimesh:
        """加载mesh对象"""
        if not Path(self.mesh_path).exists():
            raise FileNotFoundError(f"Mesh文件不存在: {self.mesh_path}")
        
        loaded = trimesh.load(self.mesh_path)
        
        # 处理GLB文件可能返回Scene对象的情况
        if hasattr(loaded, 'geometry') and loaded.geometry:
            # Scene对象，提取第一个几何体
            mesh_name = list(loaded.geometry.keys())[0]
            return loaded.geometry[mesh_name]
        elif hasattr(loaded, 'vertices'):
            # 直接是Mesh对象
            return loaded
        else:
            raise ValueError(f"无法从文件中提取mesh: {self.mesh_path}")

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
            u, v = np.random.random(2)
            elevation = np.degrees(np.arcsin(2 * u - 1))  # [-90°, 90°]
            azimuth = 360 * v  # [0°, 360°]
            
            # 距离采样 (平方根分布)
            radius = 1.0 + 4.0 * np.sqrt(np.random.random())
            
            # 轻微的中心点随机化
            center_x = np.random.uniform(-0.5, 0.5)
            center_y = np.random.uniform(-0.5, 0.5)
            center_z = np.random.uniform(-0.5, 0.5)
            
            poses.append(CameraPose(
                elevation=elevation,
                azimuth=azimuth,
                radius=radius,
                center_x=center_x,
                center_y=center_y,
                center_z=center_z
            ))
        
        return poses
    
    @staticmethod
    def align_pointclouds_simple(reference_pc: np.ndarray, 
                               rendered_pcs: List[np.ndarray],
                               poses: List[CameraPose]) -> Optional[CameraPose]:
        """简化点云对齐"""
        if not rendered_pcs or not poses:
            return None
            
        best_pose = None
        best_score = float('inf')
        
        # 预处理参考点云
        ref_pc_clean = GeometryUtils._clean_pointcloud(reference_pc)
        if ref_pc_clean is None or len(ref_pc_clean) == 0:
            return poses[0] if poses else None
        
        for rendered_pc, pose in zip(rendered_pcs, poses):
            if rendered_pc is None:
                continue
                
            # 清理渲染点云
            rend_pc_clean = GeometryUtils._clean_pointcloud(rendered_pc)
            if rend_pc_clean is None or len(rend_pc_clean) == 0:
                continue
                
            # 简单的Chamfer距离对齐
            score = GeometryUtils._chamfer_distance(ref_pc_clean, rend_pc_clean)
            if score < best_score:
                best_score = score
                best_pose = pose
        
        return best_pose if best_pose else (poses[0] if poses else None)
    
    @staticmethod
    def _clean_pointcloud(pc: np.ndarray) -> Optional[np.ndarray]:
        """清理点云数据，确保正确的格式"""
        if pc is None:
            return None
            
        # 确保是numpy数组
        if not isinstance(pc, np.ndarray):
            return None
        
        # 处理不同的维度情况
        if len(pc.shape) == 4:  # (1, H, W, 3)
            pc = pc[0]  # 去掉batch维度
        
        if len(pc.shape) == 3:  # (H, W, 3)
            # 重塑为 (H*W, 3)
            pc = pc.reshape(-1, 3)
        
        if len(pc.shape) != 2 or pc.shape[1] != 3:
            return None
        
        # 移除无效点 (NaN, inf等)
        valid_mask = np.isfinite(pc).all(axis=1)
        pc_clean = pc[valid_mask]
        
        if len(pc_clean) == 0:
            return None
            
        return pc_clean
    
    @staticmethod
    def _chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Chamfer距离计算"""
        if pc1 is None or pc2 is None or len(pc1) == 0 or len(pc2) == 0:
            return float('inf')
        
        # 确保都是2D数组
        if len(pc1.shape) != 2 or len(pc2.shape) != 2:
            return float('inf')
        
        if pc1.shape[1] != 3 or pc2.shape[1] != 3:
            return float('inf')
        
        # 采样以提高效率
        if len(pc1) > 1000:
            indices = np.random.choice(len(pc1), 1000, replace=False)
            pc1 = pc1[indices]
        if len(pc2) > 1000:
            indices = np.random.choice(len(pc2), 1000, replace=False)
            pc2 = pc2[indices]
        
        # Chamfer距离
        try:
            dist_matrix = cdist(pc1, pc2)
            forward = np.mean(np.min(dist_matrix, axis=1))
            backward = np.mean(np.min(dist_matrix, axis=0))
            return forward + backward
        except Exception as e:
            print(f"Chamfer距离计算失败: {e}")
            return float('inf')

class MeshRenderer:
    """Mesh渲染器 - 基于StandardKiuiRenderer"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._renderer = None
        self._cached_mesh_path = None
        self._mesh_loaded = False
        
        # 检查依赖
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查必要的依赖"""
        try:
            # 检查KiuiKit
            from kiui.mesh import Mesh as KiuiMesh
            from kiui.cam import OrbitCamera
            self.kiui_available = True
        except ImportError:
            raise ImportError("KiuiKit不可用，请安装kiui包")
        
        try:
            # 检查nvdiffrast
            import nvdiffrast.torch as dr
            self.nvdiffrast_available = True
        except ImportError:
            raise ImportError("nvdiffrast不可用，请安装nvdiffrast包")
        
        try:
            # 检查StandardKiuiRenderer
            sys.path.append(str(Path(__file__).parent.parent / "_reference" / "MeshSeriesGen" / "tools" / "visualization"))
            from kiui_mesh_renderer import StandardKiuiRenderer
            self.renderer_available = True
        except ImportError:
            raise ImportError("StandardKiuiRenderer不可用，请检查路径设置")
    
    @property
    def renderer(self):
        """延迟初始化渲染器"""
        if self._renderer is None:
            from kiui_mesh_renderer import StandardKiuiRenderer
            self._renderer = StandardKiuiRenderer(
                width=512, 
                height=512, 
                background_color=(1.0, 1.0, 1.0),
                device=self.device
            )
        return self._renderer
    
    def prepare_mesh(self, mesh: trimesh.Trimesh) -> str:
        """准备mesh用于渲染 - 导出为临时文件"""
        if self._cached_mesh_path is None:
            # 创建持久的临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix='.obj', prefix='v2m4_mesh_')
            os.close(temp_fd)
            
            # 导出mesh
            mesh.export(temp_path)
            self._cached_mesh_path = temp_path
            
            # 注册清理函数
            def cleanup_cached_mesh():
                if self._cached_mesh_path and os.path.exists(self._cached_mesh_path):
                    os.unlink(self._cached_mesh_path)
            atexit.register(cleanup_cached_mesh)
        
        return self._cached_mesh_path
    
    def load_mesh_to_renderer(self, mesh_path: str):
        """加载mesh到渲染器"""
        if not self._mesh_loaded or self.renderer.mesh_path_loaded != mesh_path:
            loaded_mesh = self.renderer.load_mesh(mesh_path)
            if loaded_mesh is None:
                raise RuntimeError(f"无法加载mesh到渲染器: {mesh_path}")
            self._mesh_loaded = True
    
    def render_single_view(self, mesh: trimesh.Trimesh, pose: CameraPose) -> np.ndarray:
        """渲染单个视图"""
        # 准备mesh
        mesh_path = self.prepare_mesh(mesh)
        self.load_mesh_to_renderer(mesh_path)
        
        # 渲染
        rendered_img = self.renderer.render_single_view(
            elevation=pose.elevation,
            azimuth=pose.azimuth,
            distance=pose.radius,
            target_point=pose.target_point,
            render_mode='lambertian'
        )
        
        if rendered_img is None:
            raise RuntimeError(f"渲染失败: {pose}")
        
        return rendered_img
    
    def render_batch_views(self, mesh: trimesh.Trimesh, poses: List[CameraPose]) -> List[np.ndarray]:
        """批量渲染多个视图"""
        # 准备mesh
        mesh_path = self.prepare_mesh(mesh)
        self.load_mesh_to_renderer(mesh_path)
        
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
            camera_params=camera_params,
            render_mode='lambertian',
            max_batch_size=8
        )
        
        # 检查结果
        valid_images = []
        for i, img in enumerate(rendered_images):
            if img is None:
                raise RuntimeError(f"批量渲染失败: pose {i}")
            valid_images.append(img)
        
        return valid_images

class CleanV2M4CameraSearch:
    """简化版V2M4相机搜索算法"""
    
    def __init__(self, dust3r_model_path: str, device: str = "cuda", enable_visualization: bool = True):
        self.device = device
        self.dust3r_model_path = dust3r_model_path
        self.enable_visualization = enable_visualization
        
        # 简化配置：只保留核心参数
        self.config = {
            'initial_samples': 2000,      # 初始采样数
            'top_n': 7,                   # DUSt3R候选数
            'pso_particles': 50,          # PSO粒子数
            'pso_iterations': 20,         # PSO迭代数
            'grad_iterations': 100,       # 梯度下降迭代数
            'image_size': 512,            # 图像尺寸
            'render_batch_size': 128       # 批量渲染大小 (可调整以平衡速度和内存)
        }
        
        # 延迟初始化组件
        self._dust3r_helper = None
        self._renderer = None
        self._optimizer = None
        self._visualizer = None
        
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
            self._renderer = MeshRenderer(self.device)
        return self._renderer
    
    @property
    def optimizer(self):
        """延迟初始化优化器"""
        if self._optimizer is None:
            from .optimizer import PSO_GD_Optimizer
            self._optimizer = PSO_GD_Optimizer()
        return self._optimizer
    
    @property
    def visualizer(self):
        """延迟初始化可视化器"""
        if self._visualizer is None and self.enable_visualization:
            from .visualization import V2M4Visualizer
            self._visualizer = V2M4Visualizer()
        return self._visualizer
    
    def search_camera_pose(self, data_pair: DataPair, save_visualization: bool = True) -> CameraPose:
        """主算法入口 - V2M4的9个核心步骤"""
        
        import time
        start_time = time.time()
        
        print(f"🎬 开始V2M4相机搜索算法...")
        print(f"   场景: {data_pair.scene_name}")
        print(f"   Mesh: {data_pair.mesh_path}")
        print(f"   图像: {data_pair.image_path}")
        
        # 验证数据存在
        if not data_pair.exists():
            raise FileNotFoundError(f"数据对不完整: {data_pair.scene_name}")
        
        # 加载数据
        reference_image = self._load_image(data_pair.image_path)
        mesh = data_pair.load_mesh()
        
        # 收集mesh信息用于可视化
        if self.enable_visualization:
            bounds = mesh.bounds
            self.visualization_data['mesh_info'] = {
                'vertices_count': len(mesh.vertices),
                'faces_count': len(mesh.faces),
                'bounds': bounds.tolist(),
                'center': mesh.centroid.tolist(),
                'scale': float(np.linalg.norm(bounds[1] - bounds[0]))
            }
        
        # 步骤1: 采样初始相机pose
        print("📐 步骤1: 球面采样相机姿态...")
        initial_poses = self._sample_sphere_poses()
        
        # 步骤2: 渲染并选择top-n
        print("🎯 步骤2: 选择top-n候选姿态...")
        top_poses = self._select_top_poses(mesh, reference_image, initial_poses)
        
        # 可视化：记录top-1姿态
        if self.enable_visualization and top_poses:
            rendered_top1 = self.renderer.render_single_view(mesh, top_poses[0])
            similarity_top1 = self._compute_similarity(reference_image, rendered_top1)
            self.visualization_data['progression'].append({
                'step_name': 'Top-1 Selection',
                'pose': top_poses[0],
                'rendered_image': rendered_top1,
                'similarity': similarity_top1,
                'score': similarity_top1
            })
        
        # 步骤3-4: DUSt3R估计 (核心几何约束)
        print("🔍 步骤3-4: DUSt3R几何约束估计...")
        dust3r_pose = self._dust3r_estimation(mesh, reference_image, top_poses)
        
        # 可视化：记录DUSt3R结果
        if self.enable_visualization and dust3r_pose:
            rendered_dust3r = self.renderer.render_single_view(mesh, dust3r_pose)
            similarity_dust3r = self._compute_similarity(reference_image, rendered_dust3r)
            self.visualization_data['progression'].append({
                'step_name': 'DUSt3R Align',
                'pose': dust3r_pose,
                'rendered_image': rendered_dust3r,
                'similarity': similarity_dust3r,
                'score': similarity_dust3r
            })
        
        # 步骤5-6: PSO搜索
        print("🔍 步骤5-6: PSO粒子群优化...")
        pso_pose = self._pso_search(mesh, reference_image, dust3r_pose, top_poses)
        
        # 可视化：记录PSO结果
        if self.enable_visualization:
            rendered_pso = self.renderer.render_single_view(mesh, pso_pose)
            similarity_pso = self._compute_similarity(reference_image, rendered_pso)
            self.visualization_data['progression'].append({
                'step_name': 'PSO Optimize',
                'pose': pso_pose,
                'rendered_image': rendered_pso,
                'similarity': similarity_pso,
                'score': similarity_pso
            })
        
        # 步骤7-8: 梯度下降精化
        print("🎯 步骤7-8: 梯度下降精化...")
        final_pose = self._gradient_refinement(mesh, reference_image, pso_pose)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 收集算法统计信息
        if self.enable_visualization:
            final_rendered = self.renderer.render_single_view(mesh, final_pose)
            final_similarity = self._compute_similarity(reference_image, final_rendered)
            
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
        
        # 生成可视化结果
        if self.enable_visualization and save_visualization and self.visualizer:
            try:
                # 创建结果对比图
                final_rendered = self.renderer.render_single_view(mesh, final_pose)
                comparison_path = self.visualizer.create_result_comparison(
                    data_pair=data_pair,
                    reference_image=reference_image,
                    rendered_result=final_rendered,
                    final_pose=final_pose,
                    mesh_info=self.visualization_data['mesh_info'],
                    algorithm_stats=self.visualization_data['algorithm_stats'],
                    execution_time=execution_time
                )
                
                # 创建优化过程可视化
                if self.visualization_data['progression']:
                    progression_path = self.visualizer.create_pose_progression_visualization(
                        data_pair=data_pair,
                        reference_image=reference_image,
                        progression_data=self.visualization_data['progression'],
                        final_pose=final_pose
                    )
                
                # 保存单独的结果图像
                individual_paths = self.visualizer.save_individual_results(
                    data_pair=data_pair,
                    reference_image=reference_image,
                    rendered_result=final_rendered
                )
                
            except Exception as e:
                print(f"⚠️ 可视化生成失败: {e}")
        
        print("✅ V2M4相机搜索完成!")
        print(f"⏱️ 总耗时: {execution_time:.2f}秒")
        
        return final_pose
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像文件"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _sample_sphere_poses(self) -> List[CameraPose]:
        """步骤1: 球面等面积采样"""
        return GeometryUtils.sample_sphere_poses(self.config['initial_samples'])
    
    def _select_top_poses(self, mesh: trimesh.Trimesh, reference_image: np.ndarray, 
                         poses: List[CameraPose]) -> List[CameraPose]:
        """步骤2: 基于相似度选择top-n - 优化批量渲染以避免nvdiffrast卡住"""
        
        print(f"   正在评估 {len(poses)} 个候选姿态...")
        
        scores = []
        batch_size = self.config['render_batch_size']  # 使用配置的批量渲染大小
        
        try:
            # 尝试批量渲染
            for i in range(0, len(poses), batch_size):
                batch_poses = poses[i:i+batch_size]
                print(f"   批量渲染 {i+1}-{min(i+batch_size, len(poses))}/{len(poses)}...")
                
                try:
                    # 批量渲染这一组
                    rendered_images = self.renderer.render_batch_views(mesh, batch_poses)
                    
                    # 计算相似度分数
                    for rendered_img in rendered_images:
                        if rendered_img is not None:
                            score = self._compute_similarity(reference_image, rendered_img)
                            scores.append(score)
                        else:
                            scores.append(float('inf'))  # 渲染失败，给最差分数
                            
                except Exception as e:
                    print(f"   ⚠️ 批量渲染失败，切换到逐个渲染: {e}")
                    # 批量渲染失败，逐个渲染这一组
                    for pose in batch_poses:
                        try:
                            rendered_img = self.renderer.render_single_view(mesh, pose)
                            score = self._compute_similarity(reference_image, rendered_img)
                            scores.append(score)
                        except Exception as e2:
                            print(f"   ⚠️ 单个渲染也失败: {e2}")
                            scores.append(float('inf'))
                
                # 清理GPU内存
                from .utils import cleanup_gpu_memory
                cleanup_gpu_memory()
                
        except Exception as e:
            print(f"   ❌ 批量渲染完全失败，使用顺序渲染: {e}")
            # 完全回退到顺序渲染
            scores = []
            for i, pose in enumerate(poses):
                if i % 100 == 0:
                    print(f"   顺序渲染进度: {i+1}/{len(poses)}")
                try:
                    rendered_img = self.renderer.render_single_view(mesh, pose)
                    score = self._compute_similarity(reference_image, rendered_img)
                    scores.append(score)
                except Exception as e2:
                    print(f"   ⚠️ 渲染姿态 {i} 失败: {e2}")
                    scores.append(float('inf'))
                
                # 每50个姿态清理一次内存
                if i % 50 == 0:
                    from .utils import cleanup_gpu_memory
                    cleanup_gpu_memory()
        
        # 选择top-n
        if len(scores) != len(poses):
            print(f"   ⚠️ 分数数量({len(scores)})与姿态数量({len(poses)})不匹配")
            scores = scores[:len(poses)] + [float('inf')] * (len(poses) - len(scores))
        
        top_indices = np.argsort(scores)[:self.config['top_n']]
        selected_poses = [poses[i] for i in top_indices]
        
        print(f"   ✅ 选择了 {len(selected_poses)} 个最佳姿态")
        print(f"   最佳分数: {min(scores):.4f}")
        
        return selected_poses
    
    def _dust3r_estimation(self, mesh: trimesh.Trimesh, reference_image: np.ndarray, 
                          top_poses: List[CameraPose]) -> Optional[CameraPose]:
        """步骤3-4: DUSt3R估计 - 核心几何约束"""
        
        # 1. 渲染top poses
        rendered_views = [self.renderer.render_single_view(mesh, pose) for pose in top_poses]
        
        # 2. DUSt3R推理
        dust3r_result = self.dust3r_helper.inference(reference_image, rendered_views)
        
        # 3. 点云对齐 (简化版)
        best_pose = GeometryUtils.align_pointclouds_simple(
            dust3r_result.reference_pc,
            dust3r_result.rendered_pcs, 
            top_poses[:len(dust3r_result.rendered_pcs)]
        )
        
        return best_pose
    
    def _pso_search(self, mesh: trimesh.Trimesh, reference_image: np.ndarray,
                   dust3r_pose: Optional[CameraPose], top_poses: List[CameraPose]) -> CameraPose:
        """步骤5-6: PSO搜索"""
        
        # 准备初始候选
        candidates = top_poses[:self.config['pso_particles']]
        if dust3r_pose is not None:
            candidates.append(dust3r_pose)
        
        if not candidates:
            return CameraPose(elevation=0, azimuth=0, radius=2.5)
        
        # 定义目标函数
        def objective(pose: CameraPose) -> float:
            rendered = self.renderer.render_single_view(mesh, pose)
            return self._compute_similarity(reference_image, rendered)
        
        # 参数边界
        bounds = {
            'elevation': (-90, 90),
            'azimuth': (0, 360),
            'radius': (1.0, 5.0),
            'center_x': (-1.0, 1.0),
            'center_y': (-1.0, 1.0),
            'center_z': (-1.0, 1.0)
        }
        
        return self.optimizer.pso_optimize(objective, candidates, bounds)
    
    def _gradient_refinement(self, mesh: trimesh.Trimesh, reference_image: np.ndarray,
                           initial_pose: CameraPose) -> CameraPose:
        """步骤7-8: 梯度下降精化"""
        
        def objective(pose: CameraPose) -> float:
            rendered = self.renderer.render_single_view(mesh, pose)
            return self._compute_similarity(reference_image, rendered)
        
        return self.optimizer.gradient_descent(objective, initial_pose)
    
    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算图像相似度 - 使用超时保护版本"""
        from .utils import compute_image_similarity
        return compute_image_similarity(img1, img2, timeout=10)  # 10秒超时 