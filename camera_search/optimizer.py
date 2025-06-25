"""
优化器模块
包含PSO粒子群优化和梯度下降优化
"""

import numpy as np
import torch
import torch.optim as optim
import random
from typing import List, Callable, Dict, Tuple
from .core import CameraPose

class PSO_GD_Optimizer:
    """PSO + 梯度下降优化器 - 基于原始v2m4实现"""
    
    def __init__(self):
        # PSO配置 - 与原始代码保持一致
        self.pso_config = {
            'particles': 50,      # 减少从199到50
            'iterations': 20,     # 减少从25到20 
            'w': 0.7,            # 惯性权重
            'c1': 1.5,           # 个体学习因子
            'c2': 1.5            # 社会学习因子
        }
        
        # 梯度下降配置
        self.gd_config = {
            'iterations': 100,
            'lr': 0.01
        }
    
    def pso_optimize(self, 
                    objective_func: Callable[[CameraPose], float],
                    initial_poses: List[CameraPose], 
                    bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """PSO优化相机pose - 基于原始实现"""
        
        # 初始化粒子
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        # 使用initial_poses初始化部分粒子
        for pose in initial_poses[:self.pso_config['particles']]:
            particle = [pose.elevation, pose.azimuth, pose.radius, 
                       pose.center_x, pose.center_y, pose.center_z]
            particles.append(particle)
            velocities.append([0.0] * 6)
            personal_best.append(particle[:])
            personal_best_fitness.append(float('inf'))
        
        # 补充随机粒子
        while len(particles) < self.pso_config['particles']:
            particle = []
            for param, (min_val, max_val) in bounds.items():
                particle.append(random.uniform(min_val, max_val))
            particles.append(particle)
            velocities.append([0.0] * 6)
            personal_best.append(particles[-1][:])
            personal_best_fitness.append(float('inf'))
        
        global_best = None
        global_best_fitness = float('inf')
        
        # PSO主循环
        for iteration in range(self.pso_config['iterations']):
            # 添加进度日志
            if iteration % 5 == 0 or iteration == 0:
                print(f"      PSO迭代 {iteration+1}/{self.pso_config['iterations']}...")
            
            for i in range(len(particles)):
                # 评估适应度
                current_particle = particles[i]
                current_pose = self._particle_to_pose(current_particle)
                
                fitness = objective_func(current_pose)
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = current_particle[:]
                    
                    # 更新全局最优
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best = current_particle[:]
            
            # 每次迭代后显示最佳分数
            if iteration % 5 == 0 or iteration == 0:
                print(f"        当前最佳分数: {global_best_fitness:.4f}")
            
            # 更新速度和位置
            for i in range(len(particles)):
                for j in range(6):  # 6个参数: elevation, azimuth, radius, center_x, center_y, center_z
                    r1, r2 = random.random(), random.random()
                    
                    velocities[i][j] = (
                        self.pso_config['w'] * velocities[i][j] + 
                        self.pso_config['c1'] * r1 * (personal_best[i][j] - particles[i][j]) +
                        self.pso_config['c2'] * r2 * (global_best[j] - particles[i][j])
                    )
                    
                    particles[i][j] += velocities[i][j]
                    
                    # 边界约束
                    param_names = ['elevation', 'azimuth', 'radius', 'center_x', 'center_y', 'center_z']
                    param_name = param_names[j]
                    if param_name in bounds:
                        min_val, max_val = bounds[param_name]
                        particles[i][j] = max(min_val, min(max_val, particles[i][j]))
        
        # 返回最佳pose
        if global_best is None:
            raise RuntimeError("PSO优化失败，未找到有效解")
        
        return self._particle_to_pose(global_best)
    
    def gradient_descent(self, 
                        objective_func: Callable[[CameraPose], float],
                        initial_pose: CameraPose) -> CameraPose:
        """梯度下降优化 - 基于原始实现"""
        
        # 转换为tensor (角度转换为弧度)
        pose_params = torch.tensor([
            np.radians(initial_pose.elevation), 
            np.radians(initial_pose.azimuth), 
            initial_pose.radius,
            initial_pose.center_x, 
            initial_pose.center_y, 
            initial_pose.center_z
        ], requires_grad=True)
        
        optimizer = optim.Adam([pose_params], lr=self.gd_config['lr'])
        
        best_loss = float('inf')
        best_params = pose_params.clone().detach()
        no_improvement_count = 0
        
        for iteration in range(self.gd_config['iterations']):
            optimizer.zero_grad()
            
            # 从参数构造pose (转换回角度) - 修复tensor转换
            current_pose = CameraPose(
                elevation=float(pose_params[0].detach().cpu().numpy()) * 180.0 / np.pi,
                azimuth=float(pose_params[1].detach().cpu().numpy()) * 180.0 / np.pi,
                radius=float(pose_params[2].detach().cpu().numpy()),
                center_x=float(pose_params[3].detach().cpu().numpy()),
                center_y=float(pose_params[4].detach().cpu().numpy()),
                center_z=float(pose_params[5].detach().cpu().numpy())
            )
            
            # 计算损失
            loss = objective_func(current_pose)
            loss_tensor = torch.tensor(loss, requires_grad=True)
            
            # 反向传播
            loss_tensor.backward()
            optimizer.step()
            
            # 边界约束 (弧度空间)
            with torch.no_grad():
                pose_params[0] = torch.clamp(pose_params[0], np.radians(-90), np.radians(90))  # elevation
                pose_params[1] = torch.clamp(pose_params[1], 0, np.radians(360))  # azimuth
                pose_params[2] = torch.clamp(pose_params[2], 1.0, 5.0)  # radius
                pose_params[3:] = torch.clamp(pose_params[3:], -1.0, 1.0)  # center
            
            # 检查改进
            if loss < best_loss:
                best_loss = loss
                best_params = pose_params.clone().detach()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 早停机制
            if no_improvement_count > 30:
                print(f"      早停: {iteration}次迭代后无改进")
                break
            
            if iteration % 20 == 0:
                print(f"      迭代 {iteration}, loss: {loss:.6f}")
        
        # 返回最佳pose (转换回角度) - 修复tensor转换
        final_pose = CameraPose(
            elevation=float(best_params[0].detach().cpu().numpy()) * 180.0 / np.pi,
            azimuth=float(best_params[1].detach().cpu().numpy()) * 180.0 / np.pi,
            radius=float(best_params[2].detach().cpu().numpy()),
            center_x=float(best_params[3].detach().cpu().numpy()),
            center_y=float(best_params[4].detach().cpu().numpy()),
            center_z=float(best_params[5].detach().cpu().numpy())
        )
        
        print(f"    ✅ 梯度下降完成，最终loss: {best_loss:.6f}")
        return final_pose
    
    def _particle_to_pose(self, particle: List[float]) -> CameraPose:
        """粒子转换为姿态"""
        return CameraPose(
            elevation=particle[0],
            azimuth=particle[1],
            radius=particle[2],
            center_x=particle[3],
            center_y=particle[4],
            center_z=particle[5]
        ) 