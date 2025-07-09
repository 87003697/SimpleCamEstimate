"""
V2M4相机搜索算法 - 环境配置文件
根据你的conda环境修改以下配置
"""

import os
from pathlib import Path

# =============================================================================
# 🔧 用户配置区域 - 请根据你的环境修改以下配置
# =============================================================================

# 1. 基础路径配置
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

# 2. 模型路径配置
# DUSt3R模型路径 - 请修改为你的实际路径
DUST3R_MODEL_PATH = "/data0/zhiyuan/code/MeshSeriesGen/pretrained_weights/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# StableNormal权重目录 - 可以使用默认路径或自定义
STABLENORMAL_WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# 3. 引用路径配置
# _reference目录配置 - 请确保路径正确
REFERENCE_DIR = PROJECT_ROOT / "_reference"
MESH_SERIES_GEN_DIR = REFERENCE_DIR / "MeshSeriesGen"
DUST3R_CORE_DIR = MESH_SERIES_GEN_DIR / "models" / "dust3r" / "dust3r_core"
KIUI_RENDERER_DIR = MESH_SERIES_GEN_DIR / "tools" / "visualization"

# 4. 缓存目录配置
TORCH_HUB_CACHE_DIR = None  # None表示使用默认torch.hub缓存目录
HUGGINGFACE_CACHE_DIR = None  # None表示使用默认HuggingFace缓存目录

# 5. CUDA配置
DEFAULT_CUDA_DEVICE = "cuda"
DEFAULT_CUDA_DEVICE_ID = 0

# 6. 输出目录配置
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualization"
TEMP_DIR = PROJECT_ROOT / "temp"

# =============================================================================
# 🚀 自动配置和验证
# =============================================================================

def validate_environment():
    """验证环境配置"""
    issues = []
    
    # 检查必需的目录
    required_dirs = {
        "数据目录": DATA_DIR,
        "引用目录": REFERENCE_DIR,
        "DUSt3R核心目录": DUST3R_CORE_DIR,
        "KiuiRenderer目录": KIUI_RENDERER_DIR,
    }
    
    for name, path in required_dirs.items():
        if not path.exists():
            issues.append(f"❌ {name}不存在: {path}")
    
    # 检查模型文件
    if not Path(DUST3R_MODEL_PATH).exists():
        issues.append(f"❌ DUSt3R模型不存在: {DUST3R_MODEL_PATH}")
    
    # 检查权重目录
    Path(STABLENORMAL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # 检查输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    return issues

def setup_environment():
    """设置环境变量和路径"""
    # 设置HuggingFace缓存目录
    if HUGGINGFACE_CACHE_DIR:
        os.environ['HF_HOME'] = str(HUGGINGFACE_CACHE_DIR)
    
    # 设置torch.hub缓存目录
    if TORCH_HUB_CACHE_DIR:
        os.environ['TORCH_HOME'] = str(TORCH_HUB_CACHE_DIR)
    
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEFAULT_CUDA_DEVICE_ID)

def get_dust3r_paths():
    """获取DUSt3R相关路径"""
    return {
        'core_path': DUST3R_CORE_DIR,
        'lib_path': DUST3R_CORE_DIR / "dust3r",
        'croco_path': DUST3R_CORE_DIR / "croco",
        'model_path': DUST3R_MODEL_PATH
    }

def get_stablenormal_config():
    """获取StableNormal配置"""
    return {
        'weights_dir': STABLENORMAL_WEIGHTS_DIR,
        'yoso_version': 'yoso-normal-v1-8-1',
        'device': DEFAULT_CUDA_DEVICE
    }

def get_output_paths():
    """获取输出路径配置"""
    return {
        'output_dir': OUTPUT_DIR,
        'visualization_dir': VISUALIZATION_DIR,
        'temp_dir': TEMP_DIR
    }

# =============================================================================
# 🔍 配置检查和修复建议
# =============================================================================

def print_config_status():
    """打印配置状态"""
    print("🔧 V2M4环境配置状态")
    print("=" * 50)
    
    # 验证环境
    issues = validate_environment()
    
    if not issues:
        print("✅ 所有配置都正确!")
    else:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n🛠️  修复建议:")
        print("   1. 检查_reference目录是否存在")
        print("   2. 确认DUSt3R模型路径正确")
        print("   3. 运行 python -c \"import config; config.setup_environment()\"")
    
    print(f"\n📋 当前配置:")
    print(f"   项目根目录: {PROJECT_ROOT}")
    print(f"   数据目录: {DATA_DIR}")
    print(f"   DUSt3R模型: {DUST3R_MODEL_PATH}")
    print(f"   StableNormal权重: {STABLENORMAL_WEIGHTS_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   CUDA设备: {DEFAULT_CUDA_DEVICE_ID}")

if __name__ == "__main__":
    print_config_status() 