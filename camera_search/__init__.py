"""
SimpleCamEstimate - 相机姿态估计库
基于V2M4算法的高效相机姿态搜索
"""

# 导入配置
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import setup_environment, validate_environment
    # 在导入时设置环境
    setup_environment()
except ImportError:
    print("⚠️ config.py not found, using default settings")

from .core import (
    CameraPose,
    DataPair, 
    DataManager,
    GeometryUtils,
    MeshRenderer,
    CleanV2M4CameraSearch
)

from .optimizer import OptimizerManager
from .utils import preprocess_image, compute_image_similarity, cleanup_gpu_memory
from .visualization import V2M4Visualizer
from .dust3r_helper import DUSt3RHelper, DUSt3RResult

__version__ = "0.2.0"
__author__ = "SimpleCamEstimate Team"

def validate_data_integrity(data_dir: str = "data") -> dict:
    """验证数据完整性"""
    data_manager = DataManager(data_dir)
    return data_manager.validate_data_structure()

# 便捷函数
def search_camera_pose(data_pair: DataPair, dust3r_model_path: str = None, 
                      device: str = "cuda", use_normal: bool = False) -> CameraPose:
    """便捷的相机姿态搜索函数"""
    try:
        from config import DUST3R_MODEL_PATH
        if dust3r_model_path is None:
            dust3r_model_path = DUST3R_MODEL_PATH
    except ImportError:
        if dust3r_model_path is None:
            raise ValueError("dust3r_model_path必须指定，或者创建config.py文件")
    
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=dust3r_model_path,
        device=device,
        enable_visualization=False
    )
    
    return searcher.search_camera_pose(data_pair, save_visualization=False, use_normal=use_normal)

def batch_search_all_scenes(data_dir: str = "data", dust3r_model_path: str = None, 
                           device: str = "cuda", use_normal: bool = False) -> dict:
    """批量搜索所有场景"""
    try:
        from config import DUST3R_MODEL_PATH
        if dust3r_model_path is None:
            dust3r_model_path = DUST3R_MODEL_PATH
    except ImportError:
        if dust3r_model_path is None:
            raise ValueError("dust3r_model_path必须指定，或者创建config.py文件")
    
    data_manager = DataManager(data_dir)
    data_pairs = data_manager.discover_data_pairs()
    
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=dust3r_model_path,
        device=device,
        enable_visualization=False
    )
    
    results = {}
    for data_pair in data_pairs:
        try:
            pose = searcher.search_camera_pose(data_pair, save_visualization=False, use_normal=use_normal)
            results[data_pair.scene_name] = pose
        except Exception as e:
            print(f"❌ 场景 {data_pair.scene_name} 搜索失败: {e}")
            results[data_pair.scene_name] = None
    
    return results

def create_visualization_summary(results: dict, execution_times: dict, 
                               output_dir: str = "outputs/visualization") -> str:
    """创建可视化总结"""
    from .visualization import V2M4Visualizer
    
    visualizer = V2M4Visualizer(output_dir)
    
    # 统计成功的结果
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("⚠️ 没有成功的结果可以生成总结")
        return ""
    
    # 创建批量结果总结
    summary_info = {
        'total_scenes': len(results),
        'successful_scenes': len(successful_results),
        'success_rate': len(successful_results) / len(results) * 100,
        'average_execution_time': sum(execution_times.values()) / len(execution_times) if execution_times else 0,
        'results': successful_results
    }
    
    return visualizer.create_batch_results_summary(summary_info, execution_times)

__all__ = [
    # 核心类
    "CameraPose",
    "DataPair", 
    "DataManager",
    "GeometryUtils", 
    "MeshRenderer",
    "CleanV2M4CameraSearch",
    
    # 优化器
    "OptimizerManager",
    
    # 工具函数
    "preprocess_image",
    "compute_image_similarity", 
    "cleanup_gpu_memory",
    "validate_data_integrity",
    "create_visualization_summary",
    
    # 可视化
    "V2M4Visualizer",
    
    # 模型助手
    "DUSt3RHelper",
    "DUSt3RResult",
] 