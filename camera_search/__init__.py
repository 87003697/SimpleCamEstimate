"""
SimpleCamEstimate - 相机姿态估计库
基于V2M4算法的高效相机姿态搜索
"""

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