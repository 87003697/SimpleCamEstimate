"""
工具函数模块
包含图像处理、相似度计算和内存管理
"""

import torch
import cv2
import gc
import signal
from typing import Optional

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("计算超时")

def preprocess_image(image: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """图像预处理"""
    # 确保图像是HWC格式
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # 如果是CHW格式，转换为HWC
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    
    # 调整大小
    if image.shape[:2] != (target_size, target_size):
        # 转换为numpy进行resize，因为cv2需要numpy
        image_np = image.cpu().numpy()
        image_np = cv2.resize(image_np, (target_size, target_size))
        image = torch.from_numpy(image_np).to(image.device)
    
    # 确保数据类型正确
    if image.dtype != torch.uint8:
        image = torch.clamp(image * 255, 0, 255).to(torch.uint8)
    
    return image

def compute_image_similarity(img1: torch.Tensor, img2: torch.Tensor, timeout: int = 30) -> float:
    """
    计算图像相似度 - 基于原始实现，修复SSIM参数问题并增加超时保护
    
    Args:
        img1: 第一张图像 (torch.Tensor)
        img2: 第二张图像 (torch.Tensor)
        timeout: 超时时间（秒）
        
    Returns:
        float: 相似度分数（越小越相似）
    """
    
    # 设置超时保护
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    # 转换为numpy数组进行处理，因为skimage需要numpy
    if isinstance(img1, torch.Tensor):
        img1_np = img1.cpu().numpy()
    else:
        img1_np = img1
        
    if isinstance(img2, torch.Tensor):
        img2_np = img2.cpu().numpy()
    else:
        img2_np = img2
    
    # 确保图像尺寸一致
    if img1_np.shape != img2_np.shape:
        img2_np = cv2.resize(img2_np, img1_np.shape[:2][::-1])
    
    # 检查图像尺寸，确保足够大
    min_size = min(img1_np.shape[:2])
    if min_size < 7:
        # 图像太小，只使用MSE
        mse_score = torch.mean((torch.from_numpy(img1_np).float() - torch.from_numpy(img2_np).float()) ** 2)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return (mse_score / 1000.0).item()
    
    # 使用SSIM + MSE组合
    from skimage.metrics import structural_similarity as ssim
    
    # 计算合适的win_size
    win_size = min(7, min_size)
    if win_size % 2 == 0:
        win_size -= 1  # 确保是奇数
    
    # 限制图像大小以提高计算速度
    if max(img1_np.shape[:2]) > 512:
        scale_factor = 512 / max(img1_np.shape[:2])
        new_height = int(img1_np.shape[0] * scale_factor)
        new_width = int(img1_np.shape[1] * scale_factor)
        img1_resized = cv2.resize(img1_np, (new_width, new_height))
        img2_resized = cv2.resize(img2_np, (new_width, new_height))
    else:
        img1_resized = img1_np
        img2_resized = img2_np
    
    # SSIM相似度 - 修复参数
    if len(img1_resized.shape) == 3:
        # 彩色图像
        ssim_score = ssim(img1_resized, img2_resized, 
                        win_size=win_size,
                        channel_axis=2,  # 新版本skimage参数
                        data_range=255)
    else:
        # 灰度图像
        ssim_score = ssim(img1_resized, img2_resized, 
                        win_size=win_size,
                        data_range=255)
    
    # MSE距离 - 使用torch计算
    mse_score = torch.mean((torch.from_numpy(img1_resized).float() - torch.from_numpy(img2_resized).float()) ** 2)
    
    # 恢复原始信号处理器
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)
    
    # 组合分数 (越小越好)
    return 1.0 - ssim_score + (mse_score / 1000.0).item()

def compute_image_similarity_torch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    纯PyTorch版本的图像相似度计算 - 避免numpy转换
    
    Args:
        img1: 第一张图像 (torch.Tensor)
        img2: 第二张图像 (torch.Tensor)
        
    Returns:
        float: 相似度分数（越小越相似）
    """
    
    # 确保都是torch.Tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
    
    # 确保在同一设备上
    device = img1.device
    img2 = img2.to(device)
    
    # 确保数据类型一致
    if img1.dtype != img2.dtype:
        img1 = img1.float()
        img2 = img2.float()
    
    # 归一化到[0,1]范围
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # 确保尺寸一致
    if img1.shape != img2.shape:
        # 使用PyTorch的interpolate进行resize
        import torch.nn.functional as F
        img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> BCHW
        img2 = F.interpolate(img2, size=img1.shape[:2], mode='bilinear', align_corners=False)
        img2 = img2.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
    
    # 计算MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # 计算简化版SSIM（基于方差和协方差）
    # 转换为灰度图进行SSIM计算
    if img1.dim() == 3 and img1.shape[2] == 3:
        # RGB转灰度
        gray1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
        gray2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
    else:
        gray1 = img1.squeeze() if img1.dim() == 3 else img1
        gray2 = img2.squeeze() if img2.dim() == 3 else img2
    
    # 简化SSIM计算
    mu1 = torch.mean(gray1)
    mu2 = torch.mean(gray2)
    
    sigma1_sq = torch.var(gray1)
    sigma2_sq = torch.var(gray2)
    sigma12 = torch.mean((gray1 - mu1) * (gray2 - mu2))
    
    # SSIM常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # SSIM计算
    ssim_score = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                 ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 组合分数 (越小越好)
    return (1.0 - ssim_score + mse).item()

def cleanup_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 