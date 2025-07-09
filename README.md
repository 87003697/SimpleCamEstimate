# SimpleCamEstimate - V2M4相机搜索算法

简化且高效的V2M4相机姿态估计算法实现，专注于核心功能和性能优化。

## 🎯 **核心特性**

- **🚀 高性能优化**: 完全移除trimesh依赖，使用kiui.mesh直接渲染
- **⚡ 批量渲染**: 支持GPU批量处理，大幅提升速度
- **🎨 多渲染模式**: 支持lambertian、normal、textured、depth四种渲染模式
- **🖼️ StableNormal**: ✨ **真正的StableNormal模型支持**，大幅提升几何匹配精度
- **📊 完整测试**: 25个场景全面测试，确保算法稳定性
- **🔧 灵活配置**: 支持批量大小、渲染模式等多种参数配置

## 🔥 **最新更新**

### **🎨 StableNormal 功能 (v2.1)** ✨
- **🆕 真正的StableNormal模型**：使用官方StableNormal_turbo模型
- **🚀 全面依赖升级**：升级到最新版本的diffusers、transformers等
- **⚡ 高性能**：相比标准normal渲染提升**71%**准确率
- **🔄 智能降级**：模型失败时自动使用基本处理
- **✅ 完全兼容**：完全向后兼容，可选功能

**性能对比**：
```
Normal渲染模式:           0.4435 (基线)
Normal + StableNormal:   0.7582 (+71% 提升) ✨
Textured模式:            0.2839 (几何简单场景最佳)
```

## 📦 **安装**

### **系统要求**
- Python 3.8+
- CUDA 11.8+ (推荐)
- 至少4GB GPU内存

### **快速安装**
```bash
# 克隆仓库
git clone https://github.com/your-repo/SimpleCamEstimate.git
cd SimpleCamEstimate

# 创建虚拟环境
conda create -n camestimate python=3.10
conda activate camestimate

# 安装依赖（最新版本）
pip install -r requirements.txt

# 验证安装
python test.py --single-scene dancing_spiderman --no-visualization
```

### **✨ 升级后的依赖版本**
```bash
# 核心深度学习库 (v2.1)
torch==2.7.1+cu118
torchvision==0.22.1
diffusers==0.34.0          # ⬆️ 升级自 0.25.1
transformers==4.53.1       # ⬆️ 升级自 4.36.0
huggingface_hub==0.33.2    # ⬆️ 升级自 0.23.0
tokenizers==0.21.2         # ⬆️ 升级自 0.15.2
peft==0.16.0               # ⬆️ 升级自 0.13.2
accelerate==1.8.1          # ⬆️ 升级自 1.1.1
xformers==0.0.31.post1     # ⬆️ 升级自 0.0.28.post1

# 渲染和可视化
kiui>=0.2.0
nvdiffrast
matplotlib>=3.5.0
```

## 🚀 **使用方法**

### **基本用法**
```bash
# 单场景测试
python test.py --single-scene dancing_spiderman

# ✨ 使用StableNormal模型（推荐）
python test.py --single-scene dancing_spiderman --use-normal --render-mode normal

# 批量测试（5个场景）
python test.py --scenes 5 --use-normal

# 使用不同渲染模式
python test.py --single-scene dancing_spiderman --render-mode textured
python test.py --single-scene dancing_spiderman --render-mode normal
```

### **高级参数**
```bash
# 调整批量大小（根据GPU内存）
python test.py --max-batch-size 16    # 更大批量（需要更多GPU内存）
python test.py --max-batch-size 4     # 较小批量（节省GPU内存）

# 禁用可视化（更快）
python test.py --scenes 25 --no-visualization --use-normal
```

## 🎨 **渲染模式性能对比**

| 渲染模式 | 使用StableNormal | 平均分数 | 推荐场景 | 状态 |
|----------|----------------|----------|----------|------|
| **normal + stablenormal** | ✅ | **0.7582** | 几何复杂的物体 | ✨ **推荐** |
| **textured** | ❌ | **0.2839** | 纹理丰富的物体 | ✅ 稳定 |
| normal | ❌ | 0.4435 | 几何匹配 | ✅ 基准 |
| lambertian | ❌ | 0.3806 | 光照敏感场景 | ✅ 稳定 |
| depth | ❌ | 0.4200 | 深度信息重要 | ✅ 稳定 |

## 🔧 **技术架构**

### **核心优化**
1. **Trimesh移除**: 完全移除trimesh依赖，使用kiui.mesh直接渲染
2. **批量渲染**: GPU批量处理，避免单次渲染开销
3. **内存管理**: 智能GPU内存清理，避免内存泄漏
4. **渲染模式**: 支持多种渲染模式，适应不同场景需求

### **✨ StableNormal 架构**
```python
# 基本使用
from camera_search import CleanV2M4CameraSearch, DataPair

# 创建搜索器
searcher = CleanV2M4CameraSearch(
    dust3r_model_path="path/to/dust3r",  # 可选，不影响StableNormal
    device="cuda"
)

# 搜索相机姿态（使用StableNormal）
data_pair = DataPair.from_scene_name("dancing_spiderman")
pose = searcher.search_camera_pose(data_pair, use_normal=True)
```

### **算法步骤**
1. **球面采样**: 等面积采样512个候选姿态
2. **Top-N选择**: 批量渲染选择最佳7个姿态
3. **✨ StableNormal预处理**: 将输入图像转换为高质量法线图
4. **模型估计**: 使用几何约束（可选）
5. **PSO优化**: 粒子群优化搜索
6. **梯度精化**: 梯度下降最终优化

## 📊 **性能基准**

### **测试环境**
- GPU: NVIDIA L40 (45GB)
- CUDA: 11.8
- 测试场景: 25个多样化场景

### **性能统计**
```
平均执行时间: 23-47秒
算法成功率: 100%
内存使用: 2-6GB GPU内存 (StableNormal需要额外2GB)
批量渲染: 支持1-32并行
StableNormal加载: 首次约30秒，后续即时
```

### **最佳实践**
- **几何复杂物体**: ✨ 使用`--use-normal --render-mode normal` (最佳效果)
- **纹理丰富物体**: 使用`--render-mode textured`
- **高端GPU**: 使用`--max-batch-size 16`获得最快速度
- **中端GPU**: 使用`--max-batch-size 8`（默认）平衡性能和内存
- **低端GPU**: 使用`--max-batch-size 4`节省内存

## 🐛 **故障排除**

### **常见问题**
1. **GPU内存不足**: 降低`--max-batch-size`参数
2. **StableNormal加载失败**: 
   ```bash
   # 升级依赖
   pip install --upgrade diffusers transformers huggingface_hub
   ```
3. **渲染失败**: 检查mesh文件格式，确保使用`.glb`格式
4. **xFormers警告**: 
   ```bash
   pip install --upgrade xformers
   ```

### **✨ StableNormal特定问题**
```bash
# 如果StableNormal模型下载失败
rm -rf ~/.cache/torch/hub/hugoycj_StableNormal_main
python test.py --use-normal  # 重新下载

# 如果依赖冲突
pip install --upgrade diffusers==0.34.0 transformers==4.53.1 tokenizers==0.21.2

# 检查StableNormal状态
python test_stablenormal_standalone.py
```

### **依赖版本检查**
```bash
# 检查关键依赖版本
pip list | grep -E "(diffusers|transformers|torch|huggingface)"

# 预期输出:
# diffusers                 0.34.0
# transformers              4.53.1  
# torch                     2.7.1+cu118
# huggingface-hub           0.33.2
```

## 🔄 **更新日志**

### **v2.1.0 (当前版本)** ✨
- ✅ **重大更新**: 真正的StableNormal模型支持
- ✅ **依赖升级**: 升级所有核心依赖到最新稳定版本
- ✅ **性能提升**: StableNormal带来71%性能提升
- ✅ **稳定性**: 解决所有依赖冲突问题
- ✅ **兼容性**: 向后兼容，可选择使用StableNormal

### **v2.0.0**
- ✅ **新增**: Normal Predictor功能框架
- ✅ **新增**: `--use-normal`参数支持
- ✅ **优化**: 智能降级机制

### **v1.5.0**
- ✅ **新增**: 四种渲染模式支持
- ✅ **优化**: 批量渲染性能进一步提升

### **v1.0.0**
- ✅ **重构**: 完全移除trimesh依赖
- ✅ **新增**: 批量渲染支持
- ✅ **新增**: 25个场景完整测试套件

## 🤝 **贡献**

我们欢迎所有形式的贡献！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 **许可证**

本项目使用MIT许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 **致谢**

- **StableNormal**: 高质量法线图预测模型 ✨
- **kiui**: 高质量渲染引擎
- **nvdiffrast**: GPU加速差分渲染
- **HuggingFace**: 模型托管和推理框架

---

**💡 提示**: 对于几何复杂的物体，强烈推荐使用 ✨ `--use-normal --render-mode normal` 组合，可获得最佳匹配效果！
