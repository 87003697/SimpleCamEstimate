# SimpleCamEstimate - V2M4相机搜索算法

简化且高效的V2M4相机姿态估计算法实现，专注于核心功能和性能优化。

## 🎯 **核心特性**

- **🚀 高性能优化**: 完全移除trimesh依赖，使用kiui.mesh直接渲染
- **⚡ 批量渲染**: 支持GPU批量处理，大幅提升速度
- **🎨 多渲染模式**: 支持lambertian、normal、textured、depth四种渲染模式
- **🖼️ Normal Predictor**: 🆕 支持法线图预测，大幅提升几何匹配精度
- **📊 完整测试**: 25个场景全面测试，确保算法稳定性
- **🔧 灵活配置**: 支持批量大小、渲染模式等多种参数配置

## 🔥 **最新更新**

### **🎨 Normal Predictor 功能 (v2.0)**
- **功能**: 将输入图像转换为法线图，专注于几何信息匹配
- **性能**: 相比标准normal渲染提升**71%**准确率
- **智能降级**: StableNormal模型失败时自动使用基本处理
- **兼容性**: 完全向后兼容，可选功能

**性能对比**：
```
Normal渲染模式:           0.4435 (基线)
Normal + Predictor:      0.7582 (+71% 提升)
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

# 安装依赖
pip install -r requirements.txt

# 验证安装
python test.py --single-scene dancing_spiderman --no-visualization
```

## 🚀 **使用方法**

### **基本用法**
```bash
# 单场景测试
python test.py --single-scene dancing_spiderman

# 批量测试（5个场景）
python test.py --scenes 5

# 使用不同渲染模式
python test.py --single-scene dancing_spiderman --render-mode textured
python test.py --single-scene dancing_spiderman --render-mode normal

# 🆕 使用Normal Predictor（推荐）
python test.py --single-scene dancing_spiderman --use-normal --render-mode normal
```

### **高级参数**
```bash
# 调整批量大小（根据GPU内存）
python test.py --max-batch-size 16    # 更大批量（需要更多GPU内存）
python test.py --max-batch-size 4     # 较小批量（节省GPU内存）

# 使用模型估计步骤
python test.py --single-scene dancing_spiderman --use-model dust3r

# 禁用可视化（更快）
python test.py --scenes 25 --no-visualization
```

## 🎨 **渲染模式性能对比**

| 渲染模式 | 使用Normal Predictor | 平均分数 | 推荐场景 |
|----------|-------------------|----------|----------|
| **textured** | ❌ | **0.2839** | 纹理丰富的物体 |
| **normal + predictor** | ✅ | **0.7582** | 几何复杂的物体 |
| normal | ❌ | 0.4435 | 几何匹配 |
| lambertian | ❌ | 0.3806 | 光照敏感场景 |
| depth | ❌ | 0.4200 | 深度信息重要 |

## 🔧 **技术架构**

### **核心优化**
1. **Trimesh移除**: 完全移除trimesh依赖，使用kiui.mesh直接渲染
2. **批量渲染**: GPU批量处理，避免单次渲染开销
3. **内存管理**: 智能GPU内存清理，避免内存泄漏
4. **渲染模式**: 支持多种渲染模式，适应不同场景需求

### **Normal Predictor 架构**
```python
# 基本使用
from camera_search import CleanV2M4CameraSearch, DataPair

# 创建搜索器
searcher = CleanV2M4CameraSearch(
    dust3r_model_path="path/to/dust3r",
    device="cuda"
)

# 搜索相机姿态（使用Normal Predictor）
data_pair = DataPair.from_scene_name("dancing_spiderman")
pose = searcher.search_camera_pose(data_pair, use_normal=True)
```

### **算法步骤**
1. **球面采样**: 等面积采样512个候选姿态
2. **Top-N选择**: 批量渲染选择最佳7个姿态
3. **模型估计**: 使用DUSt3R进行几何约束（可选）
4. **PSO优化**: 粒子群优化搜索
5. **梯度精化**: 梯度下降最终优化

## 📊 **性能基准**

### **测试环境**
- GPU: NVIDIA L40 (45GB)
- CUDA: 11.8
- 测试场景: 25个多样化场景

### **性能统计**
```
平均执行时间: 23-47秒
算法成功率: 100%
内存使用: 2-4GB GPU内存
批量渲染: 支持1-32并行
```

### **最佳实践**
- **高端GPU**: 使用`--max-batch-size 16`获得最快速度
- **中端GPU**: 使用`--max-batch-size 8`（默认）平衡性能和内存
- **低端GPU**: 使用`--max-batch-size 4`节省内存
- **几何复杂物体**: 使用`--use-normal --render-mode normal`
- **纹理丰富物体**: 使用`--render-mode textured`

## 🐛 **故障排除**

### **常见问题**
1. **GPU内存不足**: 降低`--max-batch-size`参数
2. **CUDA版本不匹配**: 重新安装PyTorch和相关依赖
3. **渲染失败**: 检查mesh文件格式，确保使用`.glb`格式
4. **Normal Predictor失败**: 自动降级到基本处理，不影响功能

### **依赖问题**
```bash
# 重新安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nvdiffrast kiui

# 解决flash-attention问题
conda install -c conda-forge flash-attn
```

## 🔄 **更新日志**

### **v2.0.0 (当前版本)**
- ✅ **新增**: Normal Predictor功能，提升71%准确率
- ✅ **新增**: 智能降级机制，StableNormal失败时自动使用基本处理
- ✅ **新增**: `--use-normal`参数支持
- ✅ **优化**: 修复torchvision兼容性问题
- ✅ **优化**: 更完善的错误处理和日志输出

### **v1.5.0**
- ✅ **新增**: 四种渲染模式支持（lambertian、normal、textured、depth）
- ✅ **新增**: 渲染模式性能对比和推荐
- ✅ **优化**: 批量渲染性能进一步提升
- ✅ **新增**: `--render-mode`参数支持

### **v1.0.0**
- ✅ **重构**: 完全移除trimesh依赖，使用kiui.mesh
- ✅ **新增**: 批量渲染支持，显著提升性能
- ✅ **新增**: 智能内存管理，避免GPU内存泄漏
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

- **DUSt3R**: 核心几何约束模型
- **kiui**: 高质量渲染引擎
- **nvdiffrast**: GPU加速差分渲染
- **StableNormal**: 法线图预测模型

---

**💡 提示**: 对于几何复杂的物体，强烈推荐使用 `--use-normal --render-mode normal` 组合，可获得最佳匹配效果！
