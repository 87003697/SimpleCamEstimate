"""
工具函数模块
包含图像处理、相似度计算和内存管理
"""

import torch
import cv2
import gc
import signal
from typing import Optional, List

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

def compute_image_similarity_torch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    纯PyTorch版本的图像相似度计算 - 避免numpy转换
    
    Args:
        img1: 第一张图像 (torch.Tensor)
        img2: 第二张图像 (torch.Tensor)
        
    Returns:
        float: 相似度分数（越小越相似）
    """
    
    # 使用no_grad优化性能，因为相似度计算不需要梯度
    with torch.no_grad():
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
        
        # 组合分数 (越小越好) - 修复MSE系数，与原始实现保持一致
        return (1.0 - ssim_score + mse / 10.0).item()

def compute_batch_similarity_torch(reference_image: torch.Tensor, 
                                  rendered_images: List[torch.Tensor]) -> List[float]:
    """
    批量计算图像相似度 - 纯PyTorch版本，真正的批量优化
    
    Args:
        reference_image: 参考图像 (torch.Tensor, HWC格式)
        rendered_images: 渲染图像列表 (List[torch.Tensor], 每个都是HWC格式)
        
    Returns:
        List[float]: 相似度分数列表（越小越相似）
    """
    
    with torch.no_grad():
        # 确保参考图像是tensor
        if not isinstance(reference_image, torch.Tensor):
            reference_image = torch.from_numpy(reference_image).float()
        
        # 预处理参考图像
        ref_img = reference_image.clone()
        if ref_img.max() > 1.0:
            ref_img = ref_img / 255.0
        
        # 处理渲染图像列表
        batch_images = []
        for rendered_img in rendered_images:
            if rendered_img is None:
                # 创建一个零图像作为占位符
                batch_images.append(torch.zeros_like(ref_img))
            else:
                # 确保转换为tensor
                if isinstance(rendered_img, torch.Tensor):
                    img = rendered_img.clone()
                else:
                    # 处理numpy数组
                    img = torch.from_numpy(rendered_img).float()
                
                # 确保在同一设备上
                img = img.to(ref_img.device)
                
                # 归一化
                if img.max() > 1.0:
                    img = img / 255.0
                    
                # 确保尺寸一致
                if img.shape != ref_img.shape:
                    import torch.nn.functional as F
                    img = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> BCHW
                    img = F.interpolate(img, size=ref_img.shape[:2], mode='bilinear', align_corners=False)
                    img = img.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
                
                batch_images.append(img)
        
        # 堆叠成批量tensor: [B, H, W, C]
        batch_tensor = torch.stack(batch_images, dim=0)
        B, H, W, C = batch_tensor.shape
        
        # 扩展参考图像以匹配批量大小: [B, H, W, C]
        ref_batch = ref_img.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 批量计算MSE: [B]
        mse_batch = torch.mean((ref_batch - batch_tensor) ** 2, dim=(1, 2, 3))
        
        # 批量计算简化SSIM
        # 转换为灰度图进行SSIM计算
        if C == 3:
            # 批量RGB转灰度: [B, H, W]
            gray_ref = 0.299 * ref_batch[:, :, :, 0] + 0.587 * ref_batch[:, :, :, 1] + 0.114 * ref_batch[:, :, :, 2]
            gray_batch = 0.299 * batch_tensor[:, :, :, 0] + 0.587 * batch_tensor[:, :, :, 1] + 0.114 * batch_tensor[:, :, :, 2]
        else:
            gray_ref = ref_batch.squeeze(-1) if ref_batch.dim() == 4 else ref_batch
            gray_batch = batch_tensor.squeeze(-1) if batch_tensor.dim() == 4 else batch_tensor
        
        # 批量计算SSIM统计量
        mu1 = torch.mean(gray_ref, dim=(1, 2))  # [B]
        mu2 = torch.mean(gray_batch, dim=(1, 2))  # [B]
        
        # 计算方差和协方差
        gray_ref_centered = gray_ref - mu1.unsqueeze(1).unsqueeze(2)
        gray_batch_centered = gray_batch - mu2.unsqueeze(1).unsqueeze(2)
        
        sigma1_sq = torch.mean(gray_ref_centered ** 2, dim=(1, 2))  # [B]
        sigma2_sq = torch.mean(gray_batch_centered ** 2, dim=(1, 2))  # [B]
        sigma12 = torch.mean(gray_ref_centered * gray_batch_centered, dim=(1, 2))  # [B]
        
        # SSIM常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 批量SSIM计算: [B]
        ssim_batch = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 组合分数 (越小越好): [B]
        scores = 1.0 - ssim_batch + mse_batch / 10.0
        
        # 处理渲染失败的情况
        results = []
        for i, (score, rendered_img) in enumerate(zip(scores, rendered_images)):
            if rendered_img is None:
                results.append(float('inf'))  # 渲染失败，给最差分数
            else:
                results.append(score.item())
        
        return results

def cleanup_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 