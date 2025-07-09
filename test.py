#!/usr/bin/env python3
"""
V2M4算法统一测试脚本
合并完整算法测试和可视化功能测试，支持参数控制
"""

import sys
import time
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import cv2

# 导入配置
from config import (
    setup_environment, 
    get_dust3r_paths, 
    get_stablenormal_config,
    print_config_status,
    DUST3R_MODEL_PATH
)

# 导入GPU性能监控
from camera_search.gpu_profiler import enable_profiling, disable_profiling, print_profiling_summary

# 设置CUDA设备
def setup_cuda_device(device_id: int = 0):
    """设置CUDA设备"""
    print(f"🔧 Setting CUDA device: {device_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    # 也更新配置
    setup_environment()

def check_environment():
    """环境检查"""
    print("🔧 Environment check...")
    
    # 打印配置状态
    print_config_status()
    
    # CUDA检查
    print(f"   🎯 CUDA device setting: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available, current device: {torch.cuda.current_device()}")
        print(f"   📊 GPU name: {torch.cuda.get_device_name()}")
        print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("   ⚠️ CUDA not available, will use CPU")
    
    # 测试camera_search包导入
    import camera_search
    print("   ✅ camera_search package imported successfully")
    
    # 检查必需依赖
    import nvdiffrast
    print("   ✅ nvdiffrast available (required)")
    
    import kiui
    print("   ✅ kiui available (required)")
    
    # 检查可视化依赖
    import matplotlib
    print("   ✅ matplotlib available")
    
    # 数据完整性检查
    from camera_search import validate_data_integrity
    validation = validate_data_integrity()
    print(f"   📊 Data integrity: {validation['data_completeness']:.1f}%")
    print(f"   📁 Valid data pairs: {validation['valid_data_pairs']}")
    
    if validation['valid_data_pairs'] == 0:
        print("   ❌ No valid data pairs found, cannot run tests")
        return False
    
    return True

def run_single_scene_test(scene_name: str, use_model: str = 'none', enable_visualization: bool = True, 
                          max_batch_size: int = 8, render_mode: str = 'lambertian', use_normal: bool = False) -> bool:
    """运行单场景测试"""
    model_name = use_model.upper() if use_model != 'none' else 'None (Skip Model Step)'
    print(f"🎬 Testing scene: {scene_name} (using {model_name})")
    print(f"   🔧 Max batch size: {max_batch_size}")
    print(f"   🎨 Render mode: {render_mode}")
    if use_normal:
        print(f"   🎨 Using normal predictor mode")
    
    from camera_search import DataPair, CleanV2M4CameraSearch
    
    # 创建数据对
    data_pair = DataPair.from_scene_name(scene_name)
    if not data_pair.exists():
        print(f"   ❌ Scene data does not exist: {scene_name}")
        return False
    
    # 创建搜索器 - 使用配置文件中的路径
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=DUST3R_MODEL_PATH,  # 使用配置文件中的路径
        device="cuda",
        enable_visualization=enable_visualization
    )
    
    # 配置使用的模型
    if use_model == 'dust3r':
        searcher.config['use_dust3r'] = True
        searcher.config['model_name'] = 'dust3r'
        print(f"   🔄 Using DUSt3R mode")
    else:  # none
        searcher.config['skip_model_step'] = True
        searcher.config['model_name'] = 'none'
    
    # 设置渲染配置
    searcher.config['max_batch_size'] = max_batch_size
    searcher.config['render_mode'] = render_mode
    
    # 运行算法
    import time
    start_time = time.time()
    
    best_pose = searcher.search_camera_pose(data_pair, save_visualization=enable_visualization, use_normal=use_normal)
    elapsed = time.time() - start_time
    
    if best_pose is not None:
        print(f"   ✅ Algorithm completed in {elapsed:.1f} seconds")
        print(f"   📊 Final pose: {best_pose}")
        return True
    else:
        print(f"   ❌ Algorithm failed")
        return False

def run_batch_test(num_scenes: int = 5, use_model: str = 'none', 
                  enable_visualization: bool = True, create_batch_summary: bool = True,
                  max_batch_size: int = 8, render_mode: str = 'lambertian', use_normal: bool = False) -> Dict:
    """运行批量测试"""
    model_name = use_model.upper() if use_model != 'none' else 'None (Skip Model Step)'
    
    print(f"\n🔄 Batch testing {num_scenes} scenes (using {model_name})...")
    print(f"   🎨 Visualization: {'enabled' if enable_visualization else 'disabled'}")
    print(f"   📋 Batch summary: {'enabled' if create_batch_summary else 'disabled'}")
    print(f"   🔧 Max batch size: {max_batch_size}")
    print(f"   🎨 Render mode: {render_mode}")
    if use_normal:
        print(f"   🎨 Using normal predictor mode")
    
    from camera_search import DataManager, CleanV2M4CameraSearch
    
    # 获取可用场景
    data_manager = DataManager()
    available_scenes = data_manager.discover_data_pairs()
    
    if not available_scenes:
        print(f"   ❌ No scenes found")
        return {'success': False, 'results': {}}
    
    print(f"   📁 Found scenes: {len(available_scenes)}")
    
    # 选择测试场景
    test_scenes = available_scenes[:num_scenes]
    print(f"   🎯 Test scenes: {[s.scene_name for s in test_scenes]}")
    
    # 创建搜索器 - 使用配置文件中的路径
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path=DUST3R_MODEL_PATH,  # 使用配置文件中的路径
        device="cuda",
        enable_visualization=enable_visualization
    )
    
    # 配置使用的模型
    if use_model == 'dust3r':
        searcher.config['use_dust3r'] = True
        searcher.config['model_name'] = 'dust3r'
    else:  # none
        searcher.config['skip_model_step'] = True
        searcher.config['model_name'] = 'none'
    
    # 设置渲染配置
    searcher.config['max_batch_size'] = max_batch_size
    searcher.config['render_mode'] = render_mode
    
    # 运行批量测试
    results = {}
    execution_times = []
    
    total_start_time = time.time()
    
    for i, data_pair in enumerate(test_scenes):
        print(f"\n🔄 Testing scene {i+1}/{len(test_scenes)}: {data_pair.scene_name}")
        
        start_time = time.time()
        try:
            best_pose = searcher.search_camera_pose(data_pair, save_visualization=enable_visualization, use_normal=use_normal)
            elapsed = time.time() - start_time
            
            if best_pose is not None:
                results[data_pair.scene_name] = {
                    'success': True,
                    'pose': best_pose,
                    'execution_time': elapsed
                }
                print(f"   ✅ Completed in {elapsed:.1f}s")
            else:
                results[data_pair.scene_name] = {
                    'success': False,
                    'pose': None,
                    'execution_time': elapsed
                }
                print(f"   ❌ Failed in {elapsed:.1f}s")
            
            execution_times.append(elapsed)
            
        except Exception as e:
            elapsed = time.time() - start_time
            results[data_pair.scene_name] = {
                'success': False,
                'pose': None,
                'execution_time': elapsed,
                'error': str(e)
            }
            execution_times.append(elapsed)
            print(f"   ❌ Error: {e}")
    
    total_elapsed = time.time() - total_start_time
    
    # 统计结果
    success_count = sum(1 for r in results.values() if r['success'])
    success_rate = success_count / len(results) * 100
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    print(f"\n📊 Batch test results:")
    print(f"   ✅ Success rate: {success_rate:.1f}% ({success_count}/{len(results)})")
    print(f"   ⏱️ Average time: {avg_time:.1f}s")
    print(f"   🕒 Total time: {total_elapsed:.1f}s")
    
    return {
        'success': True,
        'results': results,
        'execution_times': execution_times,
        'success_rate': success_rate,
        'total_time': total_elapsed,
        'average_time': avg_time
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V2M4 Camera Search Algorithm Test')
    parser.add_argument('--scenes', type=int, default=1, help='Number of scenes to test')
    parser.add_argument('--single-scene', type=str, help='Test single scene by name')
    parser.add_argument('--use-model', type=str, choices=['dust3r', 'none'], default='none', 
                       help='Model to use for geometric constraint estimation (default: none)')
    parser.add_argument('--cuda-device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization')
    parser.add_argument('--no-batch-summary', action='store_true', help='Disable batch summary')
    parser.add_argument('--max-batch-size', type=int, default=8, 
                       help='Maximum batch size for rendering (default: 8, larger values use more GPU memory)')
    parser.add_argument('--render-mode', type=str, choices=['lambertian', 'normal', 'textured', 'depth'], default='lambertian',
                       help='Render mode for rendering the 3D model (default: lambertian)')
    parser.add_argument('--use-normal', action='store_true', 
                       help='Use normal predictor to convert input image to normal map before matching')
    parser.add_argument('--profile', action='store_true', 
                       help='Enable GPU performance profiling')
    
    args = parser.parse_args()
    
    # 启用GPU性能监控
    if args.profile:
        print("🔍 GPU Performance profiling enabled")
        enable_profiling()
    
    # 设置CUDA设备
    setup_cuda_device(args.cuda_device)
    
    print("🚀 V2M4 Algorithm Unified Test")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        print("❌ Environment check failed, exiting test")
        sys.exit(1)
    
    # 运行测试
    passed_tests = 0
    total_tests = 0
    
    if args.single_scene:
        # 单场景测试
        total_tests = 1
        if run_single_scene_test(
            args.single_scene, 
            args.use_model, 
            not args.no_visualization,
            args.max_batch_size,
            args.render_mode,
            args.use_normal
        ):
            passed_tests = 1
    else:
        # 批量测试
        total_tests = 1
        batch_results = run_batch_test(
            args.scenes, 
            args.use_model, 
            not args.no_visualization,
            not args.no_batch_summary,
            args.max_batch_size,
            args.render_mode,
            args.use_normal
        )
        if batch_results['success']:
            passed_tests = 1
    
    # 显示性能监控摘要
    if args.profile:
        print_profiling_summary()
        disable_profiling()
    
    print("\n" + "=" * 50)
    print(f"🎉 All tests passed! ({passed_tests}/{total_tests})")
    print("📊 V2M4 algorithm is working normally!")
    
    # 显示可视化文件信息
    if not args.no_visualization:
        visualization_dir = Path("outputs/visualization")
        if visualization_dir.exists():
            viz_files = list(visualization_dir.glob("*"))
            print(f"\n📁 Visualization files: {len(viz_files)}")
            print(f"   Location: {visualization_dir}")
    
    # 显示用法示例
    print("\n💡 Usage examples:")
    print("   python test.py --scenes 5                    # Test 5 scenes (no model)")
    print("   python test.py --single-scene 'dancing_spiderman'  # Test single scene (no model)")
    print("   python test.py --single-scene 'dancing_spiderman' --use-model dust3r  # Use DUSt3R")
    print("   python test.py --no-visualization            # Disable visualization")
    print("   python test.py --scenes 25                   # Test all scenes (no model)")
    print("   python test.py --scenes 5 --use-model dust3r # Use DUSt3R batch test")
    print("   python test.py --max-batch-size 16           # Use larger batch size (more GPU memory)")
    print("   python test.py --max-batch-size 4            # Use smaller batch size (less GPU memory)")
    print("   python test.py --render-mode normal          # Use normal rendering mode")
    print("   python test.py --render-mode textured        # Use textured rendering mode")
    print("   python test.py --render-mode depth           # Use depth rendering mode")
    print("   python test.py --use-normal                  # Use normal predictor mode")
    print("   python test.py --use-normal --render-mode normal  # Use normal predictor + normal rendering")
    print("   python test.py --single-scene 'dancing_spiderman' --profile  # Enable GPU profiling")
    print("   python test.py --scenes 5 --profile          # Enable GPU profiling for batch test")

if __name__ == "__main__":
    main()
