"""
工具函数模块
包含图像处理、相似度计算和内存管理
"""

import numpy as np
import cv2
import torch
import gc
import signal
from typing import Optional

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("计算超时")

def preprocess_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """图像预处理"""
    if image.shape[:2] != (target_size, target_size):
        image = cv2.resize(image, (target_size, target_size))
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    return image

def compute_image_similarity(img1: np.ndarray, img2: np.ndarray, timeout: int = 30) -> float:
    """
    计算图像相似度 - 基于原始实现，修复SSIM参数问题并增加超时保护
    
    Args:
        img1: 第一张图像
        img2: 第二张图像  
        timeout: 超时时间（秒）
        
    Returns:
        float: 相似度分数（越小越相似）
    """
    
    # 设置超时保护
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # 确保图像尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, img1.shape[:2][::-1])
        
        # 检查图像尺寸，确保足够大
        min_size = min(img1.shape[:2])
        if min_size < 7:
            # 图像太小，只使用MSE
            mse_score = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            return mse_score / 1000.0
        
        # 使用SSIM + MSE组合
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # 计算合适的win_size
            win_size = min(7, min_size)
            if win_size % 2 == 0:
                win_size -= 1  # 确保是奇数
            
            # 限制图像大小以提高计算速度
            if max(img1.shape[:2]) > 512:
                scale_factor = 512 / max(img1.shape[:2])
                new_height = int(img1.shape[0] * scale_factor)
                new_width = int(img1.shape[1] * scale_factor)
                img1_resized = cv2.resize(img1, (new_width, new_height))
                img2_resized = cv2.resize(img2, (new_width, new_height))
            else:
                img1_resized = img1
                img2_resized = img2
            
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
            
            # MSE距离
            mse_score = np.mean((img1_resized.astype(float) - img2_resized.astype(float)) ** 2)
            
            # 组合分数 (越小越好)
            return 1.0 - ssim_score + mse_score / 1000.0
            
        except ImportError:
            # 如果没有scikit-image，使用简单MSE
            mse_score = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            return mse_score / 1000.0
        except Exception as e:
            # SSIM计算失败，回退到MSE
            print(f"SSIM计算失败，使用MSE: {e}")
            mse_score = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            return mse_score / 1000.0
    
    except TimeoutException:
        print(f"相似度计算超时({timeout}s)，使用快速MSE")
        # 超时时使用快速MSE计算
        mse_score = np.mean((img1[::4, ::4].astype(float) - img2[::4, ::4].astype(float)) ** 2)
        return mse_score / 1000.0
    
    except Exception as e:
        print(f"相似度计算出错: {e}")
        # 出错时返回一个中等分数
        return 0.5
    
    finally:
        # 恢复原始信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def cleanup_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 