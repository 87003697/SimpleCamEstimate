#!/usr/bin/env python3
"""
快速VGGT调试测试
"""

import numpy as np
import torch
import sys
import os
sys.path.append('.')

def test_vggt_inference():
    """快速测试VGGT推理"""
    print("🔍 快速VGGT推理测试...")
    
    try:
        from camera_search.vggt_helper import VGGTHelper
        
        # 创建VGGT助手
        helper = VGGTHelper(device="cuda")
        helper.load_model()
        
        if not helper.is_loaded:
            print("❌ VGGT模型加载失败")
            return
            
        # 创建测试图像
        print("📸 创建测试图像...")
        reference_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        rendered_views = [
            np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8),
            np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8),
        ]
        
        print(f"   参考图像形状: {reference_image.shape}")
        print(f"   渲染视图数量: {len(rendered_views)}")
        for i, view in enumerate(rendered_views):
            print(f"   视图{i}形状: {view.shape}")
        
        # 测试图像预处理
        print("🔄 测试图像预处理...")
        all_images = [reference_image] + rendered_views
        try:
            images_tensor = helper._preprocess_images(all_images)
            print(f"   ✅ 预处理成功，张量形状: {images_tensor.shape}")
        except Exception as e:
            print(f"   ❌ 预处理失败: {e}")
            return
        
        # 测试模型推理
        print("🤖 测试模型推理...")
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=helper.dtype):
                    predictions = helper.model(images_tensor)
            print(f"   ✅ 模型推理成功")
            print(f"   预测结果键: {list(predictions.keys())}")
            
            # 检查各个输出的形状
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                    
        except Exception as e:
            print(f"   ❌ 模型推理失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 测试点云提取
        print("☁️ 测试点云提取...")
        try:
            point_clouds = helper._extract_point_clouds(predictions, images_tensor)
            print(f"   ✅ 点云提取成功，数量: {len(point_clouds)}")
            for i, pc in enumerate(point_clouds):
                print(f"   点云{i}形状: {pc.shape}")
        except Exception as e:
            print(f"   ❌ 点云提取失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 测试完整推理
        print("🎯 测试完整推理...")
        try:
            result = helper.inference(reference_image, rendered_views)
            print(f"   ✅ 完整推理成功")
            print(f"   参考点云形状: {result.reference_pc.shape}")
            print(f"   渲染点云数量: {len(result.rendered_pcs)}")
            print(f"   深度图数量: {len(result.depth_maps)}")
            print(f"   置信度分数: {result.confidence_scores}")
        except Exception as e:
            print(f"   ❌ 完整推理失败: {e}")
            import traceback
            traceback.print_exc()
            return
            
        print("🎉 VGGT快速测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vggt_inference() 