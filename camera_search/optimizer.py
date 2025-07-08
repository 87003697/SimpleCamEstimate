"""
优化器模块
包含PSO和梯度下降优化器
"""

import torch
import random
from typing import List, Callable, Tuple, Dict, Any
from .core import CameraPose

class PSOOptimizer:
    """粒子群优化器"""
    
    def __init__(self, num_particles: int = 30, max_iterations: int = 50, 
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        
    def optimize(self, objective_func: Callable[[CameraPose], float], 
                candidates: List[CameraPose], 
                bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """PSO优化 - 传统单次渲染版本"""
        
        print(f"   🐌 Using traditional PSO optimization ({self.num_particles} particles, {self.max_iterations} iterations)")
        
        # 从候选姿态中选择初始种群
        if len(candidates) >= self.num_particles:
            particles = candidates[:self.num_particles]
        else:
            particles = candidates + self._generate_random_particles(
                self.num_particles - len(candidates), bounds)
        
        # 初始化速度和个体最优
        velocities = [self._initialize_velocity() for _ in particles]
        personal_best = particles.copy()
        personal_best_scores = [objective_func(p) for p in personal_best]
        
        # 全局最优
        global_best_idx = torch.argmin(torch.tensor(personal_best_scores)).item()
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        print(f"   📊 Initial best score: {global_best_score:.4f}")
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = torch.rand(2)
                
                # 计算速度更新
                inertia = self._multiply_velocity(velocities[i], self.w)
                cognitive = self._multiply_velocity(
                    self._subtract_poses(personal_best[i], particles[i]), 
                    self.c1 * r1)
                social = self._multiply_velocity(
                    self._subtract_poses(global_best, particles[i]), 
                    self.c2 * r2)
                
                velocities[i] = self._add_velocities([inertia, cognitive, social])
                
                # 更新位置
                particles[i] = self._add_pose_velocity(particles[i], velocities[i])
                particles[i] = self._clamp_pose(particles[i], bounds)
                
                # 评估新位置
                score = objective_func(particles[i])
                
                # 更新个体最优
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score
                    
                    # 更新全局最优
                    if score < global_best_score:
                        global_best = particles[i]
                        global_best_score = score
            
            # 进度报告
            if (iteration + 1) % 5 == 0:
                print(f"   📈 Iteration {iteration + 1}/{self.max_iterations}: best score = {global_best_score:.4f}")
        
        print(f"   ✅ Traditional PSO completed. Final best score: {global_best_score:.4f}")
        return global_best
    
    def optimize_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                      candidates: List[CameraPose], 
                      bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """PSO优化 - 批量渲染版本（性能优化）"""
        
        print(f"   🚀 Using batch PSO optimization ({self.num_particles} particles, {self.max_iterations} iterations)")
        
        # 从候选姿态中选择初始种群
        if len(candidates) >= self.num_particles:
            particles = candidates[:self.num_particles]
        else:
            particles = candidates + self._generate_random_particles(
                self.num_particles - len(candidates), bounds)
        
        # 初始化速度和个体最优
        velocities = [self._initialize_velocity() for _ in particles]
        personal_best = particles.copy()
        
        # 批量评估初始粒子
        personal_best_scores = batch_objective_func(personal_best)
        
        # 全局最优
        global_best_idx = torch.argmin(torch.tensor(personal_best_scores)).item()
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        print(f"   📊 Initial best score: {global_best_score:.4f}")
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 更新所有粒子的速度和位置
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = torch.rand(2)
                
                # 计算速度更新
                inertia = self._multiply_velocity(velocities[i], self.w)
                cognitive = self._multiply_velocity(
                    self._subtract_poses(personal_best[i], particles[i]), 
                    self.c1 * r1)
                social = self._multiply_velocity(
                    self._subtract_poses(global_best, particles[i]), 
                    self.c2 * r2)
                
                velocities[i] = self._add_velocities([inertia, cognitive, social])
                
                # 更新位置
                particles[i] = self._add_pose_velocity(particles[i], velocities[i])
                particles[i] = self._clamp_pose(particles[i], bounds)
            
            # 关键优化：批量评估所有粒子
            current_scores = batch_objective_func(particles)
            
            # 更新个体最优和全局最优
            for i, score in enumerate(current_scores):
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score
                    
                    # 更新全局最优
                    if score < global_best_score:
                        global_best = particles[i]
                        global_best_score = score
            
            # 进度报告
            if (iteration + 1) % 5 == 0:
                print(f"   📈 Iteration {iteration + 1}/{self.max_iterations}: best score = {global_best_score:.4f}")
        
        print(f"   ✅ PSO completed. Final best score: {global_best_score:.4f}")
        return global_best

    def _generate_random_particles(self, num: int, bounds: Dict[str, Tuple[float, float]]) -> List[CameraPose]:
        """生成随机粒子"""
        particles = []
        for _ in range(num):
            pose = CameraPose(
                elevation=torch.rand(1).item() * (bounds['elevation'][1] - bounds['elevation'][0]) + bounds['elevation'][0],
                azimuth=torch.rand(1).item() * (bounds['azimuth'][1] - bounds['azimuth'][0]) + bounds['azimuth'][0],
                radius=torch.rand(1).item() * (bounds['radius'][1] - bounds['radius'][0]) + bounds['radius'][0]
            )
            particles.append(pose)
        return particles
    
    def _initialize_velocity(self) -> Dict[str, float]:
        """初始化速度"""
        return {
            'elevation': torch.randn(1).item() * 2.0,
            'azimuth': torch.randn(1).item() * 5.0,
            'radius': torch.randn(1).item() * 0.5
        }
    
    def _multiply_velocity(self, velocity: Dict[str, float], factor: float) -> Dict[str, float]:
        """速度乘以因子"""
        return {k: v * factor for k, v in velocity.items()}
    
    def _subtract_poses(self, pose1: CameraPose, pose2: CameraPose) -> Dict[str, float]:
        """计算姿态差异 - 正确处理方位角周期性"""
        
        # 处理仰角和半径 - 直接相减
        elevation_diff = pose1.elevation - pose2.elevation
        radius_diff = pose1.radius - pose2.radius
        
        # 处理方位角 - 考虑周期性，选择最短路径
        azimuth_diff = pose1.azimuth - pose2.azimuth
        
        # 将差异限制在[-180, 180]范围内
        if azimuth_diff > 180:
            azimuth_diff -= 360
        elif azimuth_diff < -180:
            azimuth_diff += 360
        
        return {
            'elevation': elevation_diff,
            'azimuth': azimuth_diff,
            'radius': radius_diff
        }
    
    def _add_velocities(self, velocities: List[Dict[str, float]]) -> Dict[str, float]:
        """速度相加"""
        result = {'elevation': 0.0, 'azimuth': 0.0, 'radius': 0.0}
        for vel in velocities:
            for k in result:
                result[k] += vel[k]
        return result
    
    def _add_pose_velocity(self, pose: CameraPose, velocity: Dict[str, float]) -> CameraPose:
        """姿态加速度"""
        return CameraPose(
            elevation=pose.elevation + velocity['elevation'],
            azimuth=pose.azimuth + velocity['azimuth'],
            radius=pose.radius + velocity['radius']
        )
    
    def _clamp_pose(self, pose: CameraPose, bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """限制姿态范围 - 正确处理方位角周期性"""
        
        # 处理仰角和半径 - 使用简单的截断
        elevation = max(bounds['elevation'][0], min(bounds['elevation'][1], pose.elevation))
        radius = max(bounds['radius'][0], min(bounds['radius'][1], pose.radius))
        
        # 处理方位角 - 使用模运算处理周期性
        azimuth = pose.azimuth % 360.0
        if azimuth < 0:
            azimuth += 360.0
        
        return CameraPose(
            elevation=elevation,
            azimuth=azimuth,
            radius=radius,
            center_x=pose.center_x,
            center_y=pose.center_y,
            center_z=pose.center_z
        )

class GradientDescentOptimizer:
    """梯度下降优化器"""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 100, 
                 tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def optimize(self, objective_func: Callable[[CameraPose], float], 
                initial_pose: CameraPose) -> CameraPose:
        """梯度下降优化 - 传统单次渲染版本"""
        
        print(f"   🐌 Using traditional gradient descent optimization (max {self.max_iterations} iterations)")
        
        current_pose = initial_pose
        
        for iteration in range(self.max_iterations):
            # 计算梯度
            gradient = self._compute_gradient(objective_func, current_pose)
            
            # 计算梯度范数
            grad_norm = torch.sqrt(torch.tensor(gradient['elevation']**2 + 
                                              gradient['azimuth']**2 + 
                                              gradient['radius']**2))
            
            # 检查收敛
            if grad_norm < self.tolerance:
                print(f"   ✅ Traditional gradient descent converged at iteration {iteration + 1} (grad_norm: {grad_norm:.6f})")
                break
                
            # 更新参数
            current_pose = CameraPose(
                elevation=current_pose.elevation - self.learning_rate * gradient['elevation'],
                azimuth=current_pose.azimuth - self.learning_rate * gradient['azimuth'],
                radius=current_pose.radius - self.learning_rate * gradient['radius']
            )
            
            # 进度报告
            if (iteration + 1) % 50 == 0:
                print(f"   📊 Iteration {iteration + 1}/{self.max_iterations}: grad_norm = {grad_norm:.6f}")
        
        return current_pose
    
    def optimize_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                      initial_pose: CameraPose) -> CameraPose:
        """梯度下降优化 - 批量渲染版本（性能优化）"""
        
        print(f"   🎯 Using batch gradient descent optimization (max {self.max_iterations} iterations)")
        
        current_pose = initial_pose
        
        for iteration in range(self.max_iterations):
            # 批量计算梯度
            gradient = self._compute_gradient_batch(batch_objective_func, current_pose)
            
            # 计算梯度范数
            grad_norm = torch.sqrt(torch.tensor(gradient['elevation']**2 + 
                                              gradient['azimuth']**2 + 
                                              gradient['radius']**2))
            
            # 检查收敛
            if grad_norm < self.tolerance:
                print(f"   ✅ Gradient descent converged at iteration {iteration + 1} (grad_norm: {grad_norm:.6f})")
                break
                
            # 更新参数
            current_pose = CameraPose(
                elevation=current_pose.elevation - self.learning_rate * gradient['elevation'],
                azimuth=current_pose.azimuth - self.learning_rate * gradient['azimuth'],
                radius=current_pose.radius - self.learning_rate * gradient['radius']
            )
            
            # 进度报告
            if (iteration + 1) % 50 == 0:
                print(f"   📊 Iteration {iteration + 1}/{self.max_iterations}: grad_norm = {grad_norm:.6f}")
        
        return current_pose
    
    def _compute_gradient(self, objective_func: Callable[[CameraPose], float], 
                         pose: CameraPose) -> Dict[str, float]:
        """数值梯度计算 - 传统单次渲染版本"""
        epsilon = 1e-5
        
        # 基准分数
        base_score = objective_func(pose)
        
        # 计算各参数的梯度
        gradients = {}
        
        # elevation梯度
        pose_plus = CameraPose(pose.elevation + epsilon, pose.azimuth, pose.radius)
        gradients['elevation'] = (objective_func(pose_plus) - base_score) / epsilon
        
        # azimuth梯度
        pose_plus = CameraPose(pose.elevation, pose.azimuth + epsilon, pose.radius)
        gradients['azimuth'] = (objective_func(pose_plus) - base_score) / epsilon
        
        # radius梯度
        pose_plus = CameraPose(pose.elevation, pose.azimuth, pose.radius + epsilon)
        gradients['radius'] = (objective_func(pose_plus) - base_score) / epsilon
        
        return gradients
    
    def _compute_gradient_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                               pose: CameraPose) -> Dict[str, float]:
        """数值梯度计算 - 批量渲染版本（性能优化）"""
        epsilon = 1e-5
        
        # 构造扰动姿态：[原始, +ε_elev, +ε_azim, +ε_radius]
        poses = [
            pose,  # 基准
            CameraPose(pose.elevation + epsilon, pose.azimuth, pose.radius, 
                      pose.center_x, pose.center_y, pose.center_z),
            CameraPose(pose.elevation, pose.azimuth + epsilon, pose.radius,
                      pose.center_x, pose.center_y, pose.center_z),
            CameraPose(pose.elevation, pose.azimuth, pose.radius + epsilon,
                      pose.center_x, pose.center_y, pose.center_z)
        ]
        
        # 批量渲染评估
        scores = batch_objective_func(poses)
        base_score = scores[0]
        
        # 计算梯度
        gradients = {
            'elevation': (scores[1] - base_score) / epsilon,
            'azimuth': (scores[2] - base_score) / epsilon,
            'radius': (scores[3] - base_score) / epsilon
        }
        
        return gradients

class OptimizerManager:
    """优化器管理器"""
    
    def __init__(self, pso_params: Dict[str, Any] = None, 
                 gd_params: Dict[str, Any] = None):
        
        # PSO参数
        pso_defaults = {
            'num_particles': 30,
            'max_iterations': 50,
            'w': 0.9,
            'c1': 2.0,
            'c2': 2.0
        }
        pso_config = {**pso_defaults, **(pso_params or {})}
        self.pso = PSOOptimizer(**pso_config)
        
        # 梯度下降参数  
        gd_defaults = {
            'learning_rate': 0.01,
            'max_iterations': 100,
            'tolerance': 1e-6
        }
        gd_config = {**gd_defaults, **(gd_params or {})}
        self.gd = GradientDescentOptimizer(**gd_config)
    
    def pso_optimize(self, objective_func: Callable[[CameraPose], float], 
                    candidates: List[CameraPose], 
                    bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """PSO优化 - 传统单次渲染版本"""
        return self.pso.optimize(objective_func, candidates, bounds)
    
    def pso_optimize_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                          candidates: List[CameraPose], 
                          bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """PSO优化 - 批量渲染版本（性能优化）"""
        return self.pso.optimize_batch(batch_objective_func, candidates, bounds)
    
    def gradient_descent(self, objective_func: Callable[[CameraPose], float], 
                        initial_pose: CameraPose) -> CameraPose:
        """梯度下降优化 - 传统单次渲染版本"""
        return self.gd.optimize(objective_func, initial_pose)
    
    def gradient_descent_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                              initial_pose: CameraPose) -> CameraPose:
        """梯度下降优化 - 批量渲染版本（性能优化）"""
        return self.gd.optimize_batch(batch_objective_func, initial_pose) 