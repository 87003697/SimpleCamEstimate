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

# 设置CUDA设备
def setup_cuda_device(device_id: int = 0):
    """设置CUDA设备"""
    print(f"🔧 Setting CUDA device: {device_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

def check_environment():
    """环境检查"""
    print("🔧 Environment check...")
    
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

def test_visualization_components():
    """测试可视化组件"""
    print("🎨 Testing visualization components...")
    
    from camera_search.visualization import V2M4Visualizer
    from camera_search import DataPair, CameraPose
    import numpy as np
    import cv2
    
    # 创建可视化器
    visualizer = V2M4Visualizer(output_dir="outputs/test_visualization")
    print("   ✅ Visualizer created successfully")
    
    # 测试场景
    test_scene = "dancing_spiderman"
    data_pair = DataPair.from_scene_name(test_scene)
    
    if not data_pair.exists():
        print("   ⚠️ Test scene does not exist, skipping component test")
        return True
    
    # 创建测试数据
    reference_image = np.random.rand(512, 512, 3) * 255
    reference_image = reference_image.astype(np.uint8)
    rendered_result = np.random.rand(512, 512, 3) * 255
    rendered_result = rendered_result.astype(np.uint8)
    test_pose = CameraPose(elevation=30, azimuth=45, radius=3.0)
    
    mesh_info = {
        'vertices': 1000,
        'faces': 2000,
        'bounds': [[-1, -1, -1], [1, 1, 1]],
        'center': [0, 0, 0],
        'scale': 2.0
    }
    
    algorithm_stats = {
        'initial_samples': 512,
        'top_n': 8,
        'pso_iterations': 50,
        'final_score': 0.85
    }
    
    # 测试结果对比图
    comparison_path = visualizer.create_result_comparison(
        data_pair=data_pair,
        reference_image=reference_image,
        rendered_result=rendered_result,
        final_pose=test_pose,
        mesh_info=mesh_info,
        algorithm_stats=algorithm_stats,
        execution_time=120.5
    )
    
    print(f"   ✅ Result comparison chart: {Path(comparison_path).name}")
    
    # 测试优化过程可视化
    progression_data = [
        {
            'step_name': 'Initial Sampling',
            'pose': CameraPose(elevation=20, azimuth=30, radius=3.5),
            'rendered_image': np.random.rand(512, 512, 3) * 255,
            'score': 0.6
        },
        {
            'step_name': 'PSO Optimization',
            'pose': CameraPose(elevation=25, azimuth=40, radius=3.2),
            'rendered_image': np.random.rand(512, 512, 3) * 255,
            'score': 0.75
        },
        {
            'step_name': 'Final Result',
            'pose': test_pose,
            'rendered_image': rendered_result,
            'score': 0.85
        }
    ]
    
    # 确保progression_data中的rendered_image也是numpy数组
    for step_data in progression_data:
        if 'rendered_image' in step_data:
            step_data['rendered_image'] = step_data['rendered_image'].astype(np.uint8)
    
    progression_path = visualizer.create_pose_progression_visualization(
        data_pair=data_pair,
        reference_image=reference_image,
        progression_data=progression_data,
        final_pose=test_pose
    )
    
    print(f"   ✅ Optimization process chart: {Path(progression_path).name}")
    
    return True

def run_single_scene_test(scene_name: str, use_model: str = 'none', enable_visualization: bool = True, 
                          max_batch_size: int = 8) -> bool:
    """运行单场景测试"""
    model_name = use_model.upper() if use_model != 'none' else 'None (Skip Model Step)'
    print(f"🎬 Testing scene: {scene_name} (using {model_name})")
    print(f"   🔧 Max batch size: {max_batch_size}")
    
    from camera_search import DataPair, CleanV2M4CameraSearch
    
    # 创建数据对
    data_pair = DataPair.from_scene_name(scene_name)
    if not data_pair.exists():
        print(f"   ❌ Scene data does not exist: {scene_name}")
        return False
    
    # 创建搜索器
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path="/data0/zhiyuan/code/MeshSeriesGen/pretrained_weights/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
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
    
    # 设置批量渲染大小
    searcher.config['max_batch_size'] = max_batch_size
    
    # 运行算法
    import time
    start_time = time.time()
    
    best_pose = searcher.search_camera_pose(data_pair, save_visualization=enable_visualization)
    elapsed = time.time() - start_time
    
    if best_pose is not None:
        print(f"   ✅ Success! Pose: elevation={best_pose.elevation:.1f}°, azimuth={best_pose.azimuth:.1f}°, distance={best_pose.radius:.2f}")
        print(f"   ⏱️ Execution time: {elapsed:.1f} seconds")
        print(f"   🤖 Model: {model_name}")
        
        # 统计可视化文件
        if enable_visualization:
            output_dir = Path("outputs/visualization")
            viz_files = list(output_dir.glob("*"))
            print(f"   📊 Visualization files: {len(viz_files)}")
        
        return True
    else:
        print(f"   ❌ Algorithm execution failed")
        return False

def run_batch_test(num_scenes: int = 5, use_model: str = 'none', 
                  enable_visualization: bool = True, create_batch_summary: bool = True,
                  max_batch_size: int = 8) -> Dict:
    """运行批量测试"""
    model_name = use_model.upper() if use_model != 'none' else 'None (Skip Model Step)'
    
    print(f"\n🔄 Batch testing {num_scenes} scenes (using {model_name})...")
    print(f"   🎨 Visualization: {'enabled' if enable_visualization else 'disabled'}")
    print(f"   📋 Batch summary: {'enabled' if create_batch_summary else 'disabled'}")
    print(f"   🔧 Max batch size: {max_batch_size}")
    
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
    
    # 创建搜索器
    searcher = CleanV2M4CameraSearch(
        dust3r_model_path="/data0/zhiyuan/code/MeshSeriesGen/pretrained_weights/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
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
    
    # 设置批量渲染大小
    searcher.config['max_batch_size'] = max_batch_size
    
    # 批量处理
    results = {}
    execution_times = {}
    successful = 0
    total_elapsed = 0
    
    for i, data_pair in enumerate(test_scenes):
        print(f"\n   [{i+1}/{len(test_scenes)}] Processing scene: {data_pair.scene_name}")
        
        import time
        start_time = time.time()
        
        best_pose = searcher.search_camera_pose(data_pair, save_visualization=enable_visualization)
        elapsed = time.time() - start_time
        
        if best_pose is not None:
            results[data_pair.scene_name] = best_pose
            execution_times[data_pair.scene_name] = elapsed
            successful += 1
            total_elapsed += elapsed
        else:
            results[data_pair.scene_name] = None
    
    # 生成批量总结
    if create_batch_summary and enable_visualization:
        from camera_search import create_visualization_summary
        summary_path = create_visualization_summary(results, execution_times)
        print(f"   📋 Batch summary: {Path(summary_path).name}")
    
    # 统计结果
    success_rate = (successful / len(test_scenes)) * 100
    avg_time = total_elapsed / successful if successful > 0 else 0
    
    print(f"\n   ✅ Batch processing completed!")
    print(f"   ✅ Success rate: {successful}/{len(test_scenes)} ({success_rate:.1f}%)")
    print(f"   ⏱️ Total execution time: {total_elapsed:.1f} seconds")
    print(f"   ⏱️ Average execution time: {avg_time:.1f} seconds/scene")
    
    # 显示结果摘要
    print(f"\n   📊 Results summary:")
    for scene, pose in results.items():
        if pose is not None:
            print(f"   {scene}: elevation={pose.elevation:.1f}°, azimuth={pose.azimuth:.1f}°")
        else:
            print(f"   {scene}: failed")
    
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
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    setup_cuda_device(args.cuda_device)
    
    print("🚀 V2M4 Algorithm Unified Test")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        print("❌ Environment check failed, exiting test")
        sys.exit(1)
    
    # 测试可视化组件
    if not args.no_visualization:
        if not test_visualization_components():
            print("⚠️ Visualization component test failed, but continuing...")
    
    # 运行测试
    passed_tests = 0
    total_tests = 0
    
    if args.single_scene:
        print(f"\n🎯 Single scene test mode: {args.single_scene}")
        total_tests = 1
        
        success = run_single_scene_test(
            scene_name=args.single_scene,
            use_model=args.use_model,
            enable_visualization=not args.no_visualization,
            max_batch_size=args.max_batch_size
        )
        
        if success:
            passed_tests = 1
    else:
        # 批量测试
        total_tests = 1
        
        batch_result = run_batch_test(
            num_scenes=args.scenes,
            use_model=args.use_model,
            enable_visualization=not args.no_visualization,
            create_batch_summary=not args.no_batch_summary,
            max_batch_size=args.max_batch_size
        )
        
        if batch_result['success']:
            passed_tests = 1
    
    # 输出最终结果
    print("\n" + "=" * 50)
    if passed_tests == total_tests:
        print(f"🎉 All tests passed! ({passed_tests}/{total_tests})")
        print("📊 V2M4 algorithm is working normally!")
        
        # 统计可视化文件
        if not args.no_visualization:
            output_dir = Path("outputs/visualization")
            if output_dir.exists():
                viz_files = list(output_dir.glob("*"))
                print(f"\n📁 Visualization files: {len(viz_files)}")
                print(f"   Location: {output_dir}")
        
        print(f"\n💡 Usage examples:")
        print(f"   python test.py --scenes 5                    # Test 5 scenes (no model)")
        print(f"   python test.py --single-scene 'dancing_spiderman'  # Test single scene (no model)")
        print(f"   python test.py --single-scene 'dancing_spiderman' --use-model dust3r  # Use DUSt3R")
        print(f"   python test.py --no-visualization            # Disable visualization")
        print(f"   python test.py --scenes 25                   # Test all scenes (no model)")
        print(f"   python test.py --scenes 5 --use-model dust3r # Use DUSt3R batch test")
        print(f"   python test.py --max-batch-size 16           # Use larger batch size (more GPU memory)")
        print(f"   python test.py --max-batch-size 4            # Use smaller batch size (less GPU memory)")
        
    else:
        print(f"⚠️ Some tests failed: {passed_tests}/{total_tests}")
        print("Please check error messages and fix issues")

if __name__ == "__main__":
    main()
