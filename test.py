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

# 设置CUDA设备
def set_cuda_device(device_id: int = 2):
    """设置CUDA设备"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    print(f"🔧 设置CUDA设备: {device_id}")

def test_environment():
    """测试环境检查"""
    print("🔧 环境检查...")
    
    # CUDA检查
    print(f"   🎯 CUDA设备设置: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA可用，当前设备: {torch.cuda.current_device()}")
            print(f"   📊 GPU名称: {torch.cuda.get_device_name()}")
            print(f"   💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("   ⚠️ CUDA不可用，将使用CPU")
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False
    
    # 包导入检查
    try:
        import camera_search
        print("   ✅ camera_search包导入成功")
    except ImportError as e:
        print(f"   ❌ camera_search包导入失败: {e}")
        return False
    
    # 必需依赖检查
    try:
        import nvdiffrast.torch as dr
        print("   ✅ nvdiffrast可用 (必需)")
    except ImportError:
        print("   ❌ nvdiffrast不可用 (必需依赖)")
        return False
    
    try:
        import kiui
        print("   ✅ kiui可用 (必需)")
    except ImportError:
        print("   ❌ kiui不可用 (必需依赖)")
        return False
    
    # 可视化检查
    try:
        import matplotlib.pyplot as plt
        print("   ✅ matplotlib可用")
    except ImportError:
        print("   ⚠️ matplotlib不可用，可视化功能将受限")
    
    # 数据完整性检查
    try:
        from camera_search import validate_data_integrity
        validation = validate_data_integrity()
        print(f"   📊 数据完整性: {validation['data_completeness']:.1f}%")
        print(f"   📁 有效数据对: {validation['valid_data_pairs']}个")
        
        if validation['valid_data_pairs'] == 0:
            print("   ❌ 没有有效数据对，无法进行测试")
            return False
            
    except Exception as e:
        print(f"   ⚠️ 数据检查失败: {e}")
    
    return True

def test_visualization_components():
    """测试可视化组件功能（使用模拟数据）"""
    print("🎨 测试可视化组件...")
    
    try:
        from camera_search.visualization import V2M4Visualizer
        from camera_search import DataPair, CameraPose
        import numpy as np
        import cv2
        
        # 创建可视化器
        visualizer = V2M4Visualizer("outputs/test_visualization")
        print("   ✅ 可视化器创建成功")
        
        # 创建测试数据
        test_scene = "1"
        data_pair = DataPair.from_scene_name(test_scene)
        
        if not data_pair.exists():
            print("   ⚠️ 测试场景不存在，跳过组件测试")
            return True
        
        # 加载测试图像
        reference_image = cv2.imread(data_pair.image_path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        
        # 创建模拟渲染结果
        rendered_result = reference_image.copy()
        noise = np.random.randint(-20, 20, rendered_result.shape, dtype=np.int16)
        rendered_result = np.clip(rendered_result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 创建测试姿态
        test_pose = CameraPose(elevation=15.0, azimuth=45.0, radius=2.5)
        
        # 测试结果对比图
        comparison_path = visualizer.create_result_comparison(
            data_pair=data_pair,
            reference_image=reference_image,
            rendered_result=rendered_result,
            final_pose=test_pose,
            mesh_info={'vertices_count': 1000, 'faces_count': 2000, 'scale': 1.5},
            algorithm_stats={'initial_samples': 128, 'top_n': 7, 'final_score': 0.85},
            execution_time=120.5
        )
        print(f"   ✅ 结果对比图: {Path(comparison_path).name}")
        
        # 测试优化过程可视化
        progression_data = [
            {
                'step_name': 'Initial',
                'pose': CameraPose(elevation=0, azimuth=0, radius=3.0),
                'rendered_image': rendered_result,
                'similarity': 0.6,
                'score': 0.6
            },
            {
                'step_name': 'Final',
                'pose': test_pose,
                'rendered_image': rendered_result,
                'similarity': 0.85,
                'score': 0.85
            }
        ]
        
        progression_path = visualizer.create_pose_progression_visualization(
            data_pair=data_pair,
            reference_image=reference_image,
            progression_data=progression_data,
            final_pose=test_pose
        )
        print(f"   ✅ 优化过程图: {Path(progression_path).name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 可视化组件测试失败: {e}")
        return False

def test_single_scene(scene_name: str, enable_visualization: bool = True, device: str = "cuda", use_vggt: bool = False) -> Optional[Dict]:
    """测试单个场景的完整V2M4算法"""
    model_name = "VGGT" if use_vggt else "DUSt3R"
    print(f"🎬 测试场景: {scene_name} (使用{model_name})")
    
    try:
        from camera_search.core import CleanV2M4CameraSearch, DataPair
        
        # 创建数据对
        data_pair = DataPair.from_scene_name(scene_name)
        if not data_pair.exists():
            print(f"   ❌ 场景数据不存在: {scene_name}")
            return None
        
        # 创建搜索器
        searcher = CleanV2M4CameraSearch(
            dust3r_model_path="models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
            device=device,
            enable_visualization=enable_visualization
        )
        
        # 配置模型
        if use_vggt:
            searcher.config['use_vggt'] = True
            searcher.config['model_name'] = 'vggt'
            print(f"   🔄 切换到VGGT模式")
        
        # 运行完整的V2M4算法
        start_time = time.time()
        
        best_pose = searcher.search_camera_pose(
            data_pair=data_pair,
            save_visualization=enable_visualization
        )
        
        elapsed = time.time() - start_time
        
        if best_pose is not None:
            result = {
                'scene_name': scene_name,
                'pose': best_pose,
                'execution_time': elapsed,
                'success': True,
                'model': model_name
            }
            
            print(f"   ✅ 成功! 姿态: 仰角={best_pose.elevation:.1f}°, 方位角={best_pose.azimuth:.1f}°, 距离={best_pose.radius:.2f}")
            print(f"   ⏱️ 耗时: {elapsed:.1f}秒")
            print(f"   🤖 模型: {model_name}")
            
            if enable_visualization:
                # 检查可视化文件
                output_dir = Path("outputs/visualization")
                if output_dir.exists():
                    viz_files = list(output_dir.glob(f"*{scene_name}*"))
                    print(f"   📊 可视化文件: {len(viz_files)}个")
            
            return result
        else:
            print(f"   ❌ 算法执行失败")
            return {
                'scene_name': scene_name,
                'pose': None,
                'execution_time': elapsed,
                'success': False,
                'model': model_name
            }
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return {
            'scene_name': scene_name,
            'pose': None,
            'execution_time': 0,
            'success': False,
            'error': str(e),
            'model': model_name
        }

def test_multiple_scenes(
    num_scenes: int = 3, 
    enable_visualization: bool = True,
    create_batch_summary: bool = True,
    device: str = "cuda",
    use_vggt: bool = False
) -> Dict:
    """测试多个场景的批量处理"""
    model_name = "VGGT" if use_vggt else "DUSt3R"
    print(f"\n🔄 批量测试 {num_scenes} 个场景 (使用{model_name})...")
    print(f"   🎨 可视化: {'启用' if enable_visualization else '禁用'}")
    print(f"   📋 批量总结: {'启用' if create_batch_summary else '禁用'}")
    
    try:
        from camera_search.core import DataManager
        
        # 发现可用场景
        data_manager = DataManager()
        available_data_pairs = data_manager.discover_data_pairs()
        available_scenes = [dp.scene_name for dp in available_data_pairs]
        print(f"   📁 发现场景: {len(available_scenes)}个")
        
        # 选择测试场景
        test_scenes = available_scenes[:num_scenes]
        print(f"   🎯 测试场景: {test_scenes}")
        
        # 批量处理
        start_time = time.time()
        results = {}
        execution_times = {}
        
        for i, scene in enumerate(test_scenes, 1):
            print(f"\n   [{i}/{len(test_scenes)}] 处理场景: {scene}")
            
            # 批量处理时可以选择性禁用单个可视化
            scene_visualization = enable_visualization and not create_batch_summary
            
            result = test_single_scene(
                scene_name=scene,
                enable_visualization=scene_visualization,
                device=device,
                use_vggt=use_vggt
            )
            
            if result and result['success']:
                results[scene] = result['pose']
                execution_times[scene] = result['execution_time']
            else:
                results[scene] = None
                execution_times[scene] = 0
        
        total_elapsed = time.time() - start_time
        
        # 创建批量总结
        if create_batch_summary and enable_visualization:
            try:
                from camera_search import create_visualization_summary
                summary_path = create_visualization_summary(results, execution_times)
                print(f"   📋 批量总结: {Path(summary_path).name}")
            except Exception as e:
                print(f"   ⚠️ 批量总结生成失败: {e}")
        
        # 统计结果
        successful = sum(1 for result in results.values() if result is not None)
        success_rate = (successful / len(test_scenes)) * 100
        avg_time = total_elapsed / len(test_scenes)
        
        print(f"\n   ✅ 批量处理完成!")
        print(f"   ✅ 成功率: {successful}/{len(test_scenes)} ({success_rate:.1f}%)")
        print(f"   ⏱️ 总耗时: {total_elapsed:.1f}秒")
        print(f"   ⏱️ 平均耗时: {avg_time:.1f}秒/场景")
        
        # 显示结果摘要
        for scene, pose in results.items():
            if pose is not None:
                print(f"   {scene}: 仰角={pose.elevation:.1f}°, 方位角={pose.azimuth:.1f}°")
            else:
                print(f"   {scene}: 失败")
        
        return {
            'results': results,
            'execution_times': execution_times,
            'success_rate': success_rate,
            'total_time': total_elapsed,
            'average_time': avg_time
        }
        
    except Exception as e:
        print(f"   ❌ 批量测试失败: {e}")
        return {'success_rate': 0, 'error': str(e)}

def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='V2M4算法统一测试脚本')
    
    # 基础参数
    parser.add_argument('--scenes', '-n', type=int, default=3, 
                       help='测试场景数量 (默认: 3)')
    parser.add_argument('--cuda-device', type=int, default=2,
                       help='CUDA设备ID (默认: 2)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='计算设备 (默认: cuda)')
    
    # 功能控制
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用可视化功能')
    parser.add_argument('--no-batch-summary', action='store_true',
                       help='禁用批量总结')
    parser.add_argument('--test-components', action='store_true',
                       help='测试可视化组件 (使用模拟数据)')
    parser.add_argument('--single-scene', type=str,
                       help='只测试指定场景')
    parser.add_argument('--use-vggt', action='store_true',
                       help='使用VGGT模型进行测试')
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    if args.device == 'cuda':
        set_cuda_device(args.cuda_device)
    
    # 标题
    print("🚀 V2M4算法统一测试")
    print("=" * 50)
    
    # 环境检查
    if not test_environment():
        print("❌ 环境检查失败，退出测试")
        return
    
    # 运行测试
    test_results = []
    
    # 测试1: 可视化组件测试 (可选)
    if args.test_components:
        test_results.append(test_visualization_components())
    
    # 测试2: 单场景测试
    if args.single_scene:
        print(f"\n🎯 单场景测试模式: {args.single_scene}")
        result = test_single_scene(
            scene_name=args.single_scene,
            enable_visualization=not args.no_visualization,
            device=args.device,
            use_vggt=args.use_vggt
        )
        test_results.append(result['success'] if result else False)
    else:
        # 测试3: 多场景批量测试
        batch_result = test_multiple_scenes(
            num_scenes=args.scenes,
            enable_visualization=not args.no_visualization,
            create_batch_summary=not args.no_batch_summary,
            device=args.device,
            use_vggt=args.use_vggt
        )
        test_results.append(batch_result['success_rate'] > 0)
    
    # 总结
    print("\n" + "=" * 50)
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"🎉 所有测试通过! ({passed_tests}/{total_tests})")
        print("📊 V2M4算法运行正常!")
        
        # 显示输出目录
        output_dir = Path("outputs/visualization")
        if output_dir.exists():
            viz_files = list(output_dir.glob("*.png"))
            if viz_files:
                print(f"\n📁 可视化文件: {len(viz_files)}个")
                print(f"   位置: {output_dir}")
        
        print(f"\n💡 使用示例:")
        print(f"   python test.py --scenes 5                    # 测试5个场景")
        print(f"   python test.py --single-scene 'dancing_spiderman'  # 测试单个场景")
        print(f"   python test.py --single-scene 'dancing_spiderman' --use-vggt  # 使用VGGT测试")
        print(f"   python test.py --no-visualization            # 禁用可视化")
        print(f"   python test.py --scenes 25                   # 测试所有场景")
        print(f"   python test.py --scenes 5 --use-vggt         # 使用VGGT批量测试")
        
    else:
        print(f"⚠️ 部分测试失败: {passed_tests}/{total_tests}")
        print("请检查错误信息并修复问题")

if __name__ == "__main__":
    main()
