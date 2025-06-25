# DUSt3R模型设置

## 模型下载

请从以下位置下载DUSt3R模型：

1. **官方仓库**: https://github.com/naver/dust3r
2. **Hugging Face**: https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt

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