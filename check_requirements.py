#!/usr/bin/env python3
"""
简化版V2M4相机搜索算法测试脚本
验证基本功能和依赖
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试导入"""
    print("🔍 测试导入...")
    
    try:
        from camera_search import (
            CleanV2M4CameraSearch, 
            CameraPose, 
            DataPair, 
            DataManager,
            search_camera_pose,
            batch_search_all_scenes
        )
        print("  ✅ 基础导入成功")
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False
    
    return True

def test_data_structures():
    """测试数据结构"""
    print("🔍 测试数据结构...")
    
    try:
        from camera_search import CameraPose, DataPair
        
        # 测试CameraPose
        pose = CameraPose(elevation=30, azimuth=45, radius=2.5)
        print(f"  ✅ CameraPose创建成功: {pose}")
        
        # 测试DataPair
        data_pair = DataPair.from_scene_name("test_scene", "data")
        print(f"  ✅ DataPair创建成功: {data_pair.scene_name}")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据结构测试失败: {e}")
        return False

def test_data_discovery():
    """测试数据发现"""
    print("🔍 测试数据发现...")
    
    try:
        from camera_search import DataManager
        
        data_manager = DataManager("data")
        validation = data_manager.validate_data_structure()
        
        print(f"  📊 数据验证结果:")
        print(f"     Mesh文件数: {validation['total_mesh_files']}")
        print(f"     图像文件数: {validation['total_image_files']}")
        print(f"     有效数据对: {validation['valid_data_pairs']}")
        print(f"     数据完整性: {validation['data_completeness']:.1f}%")
        
        if validation['valid_data_pairs'] > 0:
            print("  ✅ 数据发现成功")
            
            # 显示前几个场景
            data_pairs = data_manager.discover_data_pairs()
            print(f"  📋 可用场景 (前5个):")
            for i, pair in enumerate(data_pairs[:5]):
                print(f"     {i+1}. {pair.scene_name}")
        else:
            print("  ⚠️  未发现有效数据对")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据发现测试失败: {e}")
        return False

def check_requirements():
    """检查项目依赖 - 整合版本信息和详细检查"""
    print("🔍 检查项目依赖...")
    
    # 必需依赖列表：(pip包名, import名)
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'), 
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('scikit-image', 'skimage'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('trimesh', 'trimesh'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('kiui', 'kiui'),           # 高质量渲染 (必需)
        ('nvdiffrast', 'nvdiffrast'), # GPU加速渲染 (必需)
    ]
    
    # 可选依赖列表 (目前为空，所有依赖都是必需的)
    optional_packages = [
        # ('package_name', 'import_name'),  # 示例
    ]
    
    def check_package(package_name, import_name):
        """检查单个包并返回版本信息"""
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    print("  📋 必需依赖:")
    required_success = 0
    for package_name, import_name in required_packages:
        success, version = check_package(package_name, import_name)
        if success:
            print(f"    ✅ {package_name}: {version}")
            required_success += 1
        else:
            print(f"    ❌ {package_name}: 未安装")
    
    # 只有在有可选依赖时才显示这个部分
    if optional_packages:
        print(f"\n  📋 可选依赖:")
        optional_success = 0
        for package_name, import_name in optional_packages:
            success, version = check_package(package_name, import_name)
            if success:
                print(f"    ✅ {package_name}: {version}")
                optional_success += 1
            else:
                print(f"    ⚠️  {package_name}: 未安装 (可选)")
    else:
        optional_success = 0
    
    # 结果总结
    print(f"\n  📊 依赖统计:")
    print(f"    必需依赖: {required_success}/{len(required_packages)} ✅")
    if optional_packages:
        print(f"    可选依赖: {optional_success}/{len(optional_packages)} ✅")
    
    if required_success == len(required_packages):
        print(f"  ✅ 所有必要依赖都可用")
        return True
    else:
        missing_count = len(required_packages) - required_success
        print(f"  ❌ 缺少 {missing_count} 个必需依赖")
        print(f"     请运行: pip install -r requirements.txt")
        return False

def test_dust3r_model():
    """测试DUSt3R模型路径"""
    print("🔍 测试DUSt3R模型...")
    
    model_paths = [
        "models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        "models/dust3r",
        "_reference/MeshSeriesGen/pretrained_weights/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"  ✅ 找到模型路径: {model_path}")
            
            # 检查模型文件
            model_files = list(Path(model_path).glob("*"))
            if model_files:
                print(f"     包含文件: {len(model_files)} 个")
                for f in model_files[:3]:  # 显示前3个文件
                    print(f"       - {f.name}")
                if len(model_files) > 3:
                    print(f"       ... 还有 {len(model_files)-3} 个文件")
            return True
    
    print("  ❌ 未找到DUSt3R模型")
    return False

def test_mesh_loading():
    """测试mesh加载"""
    print("🔍 测试mesh加载...")
    
    try:
        from camera_search import DataManager
        
        data_manager = DataManager("data")
        data_pairs = data_manager.discover_data_pairs()
        
        if not data_pairs:
            print("  ⚠️  没有可用的数据对")
            return True
        
        # 测试加载第一个mesh
        test_pair = data_pairs[0]
        print(f"  📦 测试加载: {test_pair.scene_name}")
        
        mesh = test_pair.load_mesh()
        print(f"  ✅ Mesh加载成功:")
        print(f"     顶点数: {len(mesh.vertices)}")
        print(f"     面数: {len(mesh.faces)}")
        print(f"     边界框: {mesh.bounds}")
        
        return True
    except Exception as e:
        print(f"  ❌ Mesh加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 简化版V2M4相机搜索算法 - 功能测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("数据结构测试", test_data_structures),
        ("数据发现测试", test_data_discovery),
        ("依赖检查", check_requirements),
        ("DUSt3R模型测试", test_dust3r_model),
        ("Mesh加载测试", test_mesh_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！可以开始使用相机搜索算法。")
    elif passed >= len(results) * 0.7:
        print("⚠️  大部分测试通过，可能存在一些依赖问题。")
    else:
        print("❌ 多个测试失败，需要检查环境配置。")

if __name__ == "__main__":
    main() 