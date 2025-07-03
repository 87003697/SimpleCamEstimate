# DUSt3R模型设置

## 模型下载

请从以下位置下载DUSt3R模型：

1. **官方仓库**: https://github.com/naver/dust3r
2. **Hugging Face**: https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt

### 使用 Hugging Face Hub 下载

首先安装 huggingface_hub：

```bash
pip install huggingface_hub
```

然后使用以下 Python 代码下载模型：

```python
from huggingface_hub import snapshot_download
import os

# 创建模型目录
model_dir = "models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
os.makedirs(model_dir, exist_ok=True)

# 下载模型
snapshot_download(
    repo_id="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)

print(f"模型已下载到: {model_dir}")
```

或者使用命令行工具：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt --local-dir models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt --local-dir-use-symlinks False
```

## 模型放置

将下载的模型文件放置在以下目录结构中：

```
models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt/
├── config.json
├── model.safetensors
├── README.md
└── ... (其他模型文件)
```

## 或者使用符号链接

如果你已经在其他位置有DUSt3R模型，可以创建符号链接：

```bash
ln -s /path/to/your/dust3r/model models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt
```

## 验证安装

运行以下命令验证模型是否正确安装：

```bash
python check_requirements.py
```

应该看到：
```
✅ 找到模型路径: models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt
``` 