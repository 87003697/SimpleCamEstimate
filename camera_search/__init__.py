"""
Clean V2M4 Camera Search Algorithm

简化版V2M4相机搜索算法，保留DUSt3R核心功能
适配简化的数据结构：
- Mesh: data/meshes/{scene_name}_textured_frame_000000.glb
- Image: data/images/{scene_name}.png
"""

import time
from .core import CleanV2M4CameraSearch, CameraPose, DataPair, DataManager
from .dust3r_helper import DUSt3RHelper, DUSt3RResult
from .optimizer import PSO_GD_Optimizer
from .utils import preprocess_image, compute_image_similarity, cleanup_gpu_memory
from .visualization import V2M4Visualizer

__version__ = "1.0.0"
__author__ = "V2M4 Team"

# 主要接口
def search_camera_pose(dust3r_model_path: str, 
                      scene_name: str,
                      data_dir: str = "data",
                      device: str = "cuda",
                      enable_visualization: bool = True,
                      save_visualization: bool = True,
                      render_batch_size: int = 128) -> CameraPose:
    """
    一行调用接口：搜索单个场景的最佳相机姿态
    
    Args:
        dust3r_model_path: DUSt3R模型路径
        scene_name: 场景名称 (如 "dancing_spiderman", "trump", "1" 等)
        data_dir: 数据目录路径
        device: 计算设备 ("cuda" 或 "cpu")
        enable_visualization: 是否启用可视化
        save_visualization: 是否保存可视化结果
        render_batch_size: 批量渲染大小 (默认16，GPU内存不足时可减小至4-8)
        
    Returns:
        CameraPose: 最佳相机姿态
        
    Example:
        >>> # 默认批量大小
        >>> best_pose = search_camera_pose(
        ...     dust3r_model_path="models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        ...     scene_name="dancing_spiderman"
        ... )
        >>> 
        >>> # 大GPU内存，加速渲染
        >>> best_pose = search_camera_pose(
        ...     dust3r_model_path="models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        ...     scene_name="dancing_spiderman",
        ...     render_batch_size=32  # 更大批量，更快速度
        ... )
        >>>
        >>> # 小GPU内存，避免OOM
        >>> best_pose = search_camera_pose(
        ...     dust3r_model_path="models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        ...     scene_name="dancing_spiderman", 
        ...     render_batch_size=4   # 更小批量，更稳定
        ... )
    """
    
    # 创建数据对
    data_pair = DataPair.from_scene_name(scene_name, data_dir)
    
    # 验证数据存在
    if not data_pair.exists():
        raise FileNotFoundError(f"场景数据不存在: {scene_name}")
    
    # 创建搜索器
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=dust3r_model_path,
        device=device,
        enable_visualization=enable_visualization
    )
    
    # 调整批量渲染大小
    searcher.config['render_batch_size'] = render_batch_size
    
    # 执行搜索
    return searcher.search_camera_pose(data_pair, save_visualization=save_visualization)

def batch_search_all_scenes(dust3r_model_path: str,
                           data_dir: str = "data", 
                           device: str = "cuda",
                           enable_visualization: bool = True,
                           save_individual_visualizations: bool = True,
                           create_batch_summary: bool = True,
                           render_batch_size: int = 16) -> dict:
    """
    批量处理所有可用场景
    
    Args:
        dust3r_model_path: DUSt3R模型路径
        data_dir: 数据目录路径
        device: 计算设备
        enable_visualization: 是否启用可视化
        save_individual_visualizations: 是否保存每个场景的可视化
        create_batch_summary: 是否创建批量处理总结
        render_batch_size: 批量渲染大小 (默认16，GPU内存不足时可减小至4-8)
        
    Returns:
        场景名称到最佳姿态的映射
    """
    
    print("🚀 开始批量V2M4相机搜索...")
    
    # 发现所有数据对
    data_manager = DataManager(data_dir)
    data_pairs = data_manager.discover_data_pairs()
    
    if not data_pairs:
        print("❌ 未找到任何有效数据对")
        return {}
    
    print(f"📊 发现 {len(data_pairs)} 个场景")
    
    # 创建搜索器
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=dust3r_model_path,
        device=device,
        enable_visualization=enable_visualization
    )
    
    # 调整批量渲染大小
    searcher.config['render_batch_size'] = render_batch_size
    
    # 批量处理
    results = {}
    execution_times = {}
    successful_count = 0
    
    for i, data_pair in enumerate(data_pairs, 1):
        print(f"\n🎯 处理场景 {i}/{len(data_pairs)}: {data_pair.scene_name}")
        
        try:
            start_time = time.time()
            
            # 执行搜索
            pose = searcher.search_camera_pose(
                data_pair, 
                save_visualization=save_individual_visualizations
            )
            
            execution_time = time.time() - start_time
            
            results[data_pair.scene_name] = pose
            execution_times[data_pair.scene_name] = execution_time
            successful_count += 1
            
            print(f"✅ {data_pair.scene_name} 成功 (耗时: {execution_time:.1f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            results[data_pair.scene_name] = None
            execution_times[data_pair.scene_name] = execution_time
            print(f"❌ {data_pair.scene_name} 失败: {e}")
    
    # 创建批量处理总结
    if create_batch_summary and enable_visualization:
        try:
            visualizer = V2M4Visualizer()
            summary_path = visualizer.create_batch_results_summary(
                batch_results=results,
                execution_times=execution_times
            )
        except Exception as e:
            print(f"⚠️ 批量总结生成失败: {e}")
    
    # 输出总结
    success_rate = (successful_count / len(data_pairs)) * 100
    total_time = sum(execution_times.values())
    avg_time = total_time / len(data_pairs) if data_pairs else 0
    
    print(f"\n🎉 批量处理完成!")
    print(f"📊 成功率: {successful_count}/{len(data_pairs)} ({success_rate:.1f}%)")
    print(f"⏱️ 总耗时: {total_time:.1f}秒 (平均: {avg_time:.1f}秒/场景)")
    
    return results

def discover_available_scenes(data_dir: str = "data") -> list:
    """
    发现所有可用的场景名称
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        List[str]: 可用场景名称列表
        
    Example:
        >>> scenes = discover_available_scenes()
        >>> print(f"可用场景: {scenes}")
    """
    
    data_manager = DataManager(data_dir)
    data_pairs = data_manager.discover_data_pairs()
    return [pair.scene_name for pair in data_pairs]

def validate_data_integrity(data_dir: str = "data") -> dict:
    """
    验证数据完整性
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        Dict: 验证结果统计
        
    Example:
        >>> validation = validate_data_integrity()
        >>> print(f"数据完整性: {validation['data_completeness']:.1f}%")
    """
    
    data_manager = DataManager(data_dir)
    return data_manager.validate_data_structure()

def create_visualization_summary(scene_results: dict,
                               execution_times: dict,
                               output_dir: str = "outputs/visualization") -> str:
    """
    为已有结果创建可视化总结
    
    Args:
        scene_results: 场景结果字典
        execution_times: 执行时间字典
        output_dir: 输出目录
        
    Returns:
        str: 总结文件路径
    """
    
    visualizer = V2M4Visualizer(output_dir)
    return visualizer.create_batch_results_summary(scene_results, execution_times)

__all__ = [
    'CleanV2M4CameraSearch',
    'CameraPose', 
    'DataPair',
    'DataManager',
    'DUSt3RHelper',
    'DUSt3RResult',
    'PSO_GD_Optimizer',
    'search_camera_pose',
    'batch_search_all_scenes',
    'discover_available_scenes',
    'validate_data_integrity',
    'create_visualization_summary'
] 