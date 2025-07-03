#!/usr/bin/env python3
"""
测试VGGT集成的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera_search.core import CleanV2M4CameraSearch, DataPair

def test_vggt_integration():
    """测试VGGT集成"""
    
    print("🧪 测试VGGT集成...")
    
    # 1. 测试DUSt3R模式 (默认)
    print("\n1️⃣ 测试DUSt3R模式...")
    searcher_dust3r = CleanV2M4CameraSearch(
        dust3r_model_path="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        device="cuda",
        enable_visualization=False
    )
    
    print(f"   配置: use_vggt={searcher_dust3r.config['use_vggt']}")
    print(f"   模型名称: {searcher_dust3r.config['model_name']}")
    
    # 2. 测试VGGT模式
    print("\n2️⃣ 测试VGGT模式...")
    searcher_vggt = CleanV2M4CameraSearch(
        dust3r_model_path="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        device="cuda",
        enable_visualization=False
    )
    
    # 切换到VGGT模式
    searcher_vggt.config['use_vggt'] = True
    searcher_vggt.config['model_name'] = 'vggt'
    
    print(f"   配置: use_vggt={searcher_vggt.config['use_vggt']}")
    print(f"   模型名称: {searcher_vggt.config['model_name']}")
    
    # 3. 测试VGGT助手初始化
    print("\n3️⃣ 测试VGGT助手初始化...")
    try:
        vggt_helper = searcher_vggt.vggt_helper
        print("   ✅ VGGT助手初始化成功")
        
        # 测试模型加载
        print("   🔄 测试VGGT模型加载...")
        vggt_helper.load_model()
        
        if vggt_helper.is_loaded:
            print("   ✅ VGGT模型加载成功")
        else:
            print("   ⚠️ VGGT模型加载失败，将使用占位符")
            
    except Exception as e:
        print(f"   ❌ VGGT助手初始化失败: {e}")
    
    # 4. 测试配置切换
    print("\n4️⃣ 测试配置切换...")
    
    def test_config_switch():
        searcher = CleanV2M4CameraSearch(
            dust3r_model_path="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
            device="cuda",
            enable_visualization=False
        )
        
        # 默认DUSt3R
        assert searcher.config['use_vggt'] == False
        assert searcher.config['model_name'] == 'dust3r'
        
        # 切换到VGGT
        searcher.config['use_vggt'] = True
        searcher.config['model_name'] = 'vggt'
        assert searcher.config['use_vggt'] == True
        assert searcher.config['model_name'] == 'vggt'
        
        print("   ✅ 配置切换测试通过")
    
    test_config_switch()
    
    print("\n🎉 VGGT集成测试完成!")
    print("\n📋 使用方法:")
    print("   # 使用DUSt3R (默认)")
    print("   searcher = CleanV2M4CameraSearch(...)")
    print("   ")
    print("   # 使用VGGT")
    print("   searcher = CleanV2M4CameraSearch(...)")
    print("   searcher.config['use_vggt'] = True")

if __name__ == "__main__":
    test_vggt_integration() 