import time
import functools
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import GPUtil

@dataclass
class StageMetrics:
    """单个阶段的性能指标"""
    name: str
    duration: float
    gpu_memory_start: float
    gpu_memory_peak: float
    gpu_memory_end: float
    gpu_utilization_avg: float
    gpu_utilization_peak: float

class GPUProfiler:
    """GPU性能监控器"""
    
    def __init__(self):
        self.enabled = False
        self.metrics: List[StageMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.current_stage_data: Dict = {}
        
    def enable(self):
        """启用性能监控"""
        self.enabled = True
        self.metrics.clear()
        
    def disable(self):
        """禁用性能监控"""
        self.enabled = False
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
            
    def get_gpu_memory_usage(self) -> float:
        """获取GPU内存使用量（MB）"""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0
        return gpus[0].memoryUsed
    
    def get_gpu_utilization(self) -> float:
        """获取GPU利用率（%）"""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0
        return gpus[0].load * 100
    
    def start_monitoring_stage(self, stage_name: str):
        """开始监控阶段"""
        if not self.enabled:
            return
            
        self.current_stage_data[stage_name] = {
            'start_time': time.time(),
            'gpu_memory_start': self.get_gpu_memory_usage(),
            'gpu_memory_peak': self.get_gpu_memory_usage(),
            'gpu_utilization_samples': [],
            'gpu_utilization_peak': 0.0
        }
        
        # 启动监控线程
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_gpu_usage,
            args=(stage_name,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def end_monitoring_stage(self, stage_name: str):
        """结束监控阶段"""
        if not self.enabled or stage_name not in self.current_stage_data:
            return
            
        # 停止监控线程
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
        
        stage_data = self.current_stage_data[stage_name]
        end_time = time.time()
        gpu_memory_end = self.get_gpu_memory_usage()
        
        # 计算平均GPU利用率
        utilization_samples = stage_data['gpu_utilization_samples']
        avg_utilization = sum(utilization_samples) / len(utilization_samples) if utilization_samples else 0.0
        
        # 创建指标记录
        metrics = StageMetrics(
            name=stage_name,
            duration=end_time - stage_data['start_time'],
            gpu_memory_start=stage_data['gpu_memory_start'],
            gpu_memory_peak=stage_data['gpu_memory_peak'],
            gpu_memory_end=gpu_memory_end,
            gpu_utilization_avg=avg_utilization,
            gpu_utilization_peak=stage_data['gpu_utilization_peak']
        )
        
        self.metrics.append(metrics)
        del self.current_stage_data[stage_name]
    
    def _monitor_gpu_usage(self, stage_name: str):
        """在后台监控GPU使用情况"""
        while not self.stop_monitoring.is_set():
            current_memory = self.get_gpu_memory_usage()
            current_utilization = self.get_gpu_utilization()
            
            # 更新峰值内存
            if current_memory > self.current_stage_data[stage_name]['gpu_memory_peak']:
                self.current_stage_data[stage_name]['gpu_memory_peak'] = current_memory
            
            # 更新峰值利用率
            if current_utilization > self.current_stage_data[stage_name]['gpu_utilization_peak']:
                self.current_stage_data[stage_name]['gpu_utilization_peak'] = current_utilization
            
            # 记录利用率样本
            self.current_stage_data[stage_name]['gpu_utilization_samples'].append(current_utilization)
            
            # 每100ms采样一次
            time.sleep(0.1)
    
    def print_summary(self):
        """打印性能监控摘要"""
        if not self.metrics:
            print("No profiling data available.")
            return
            
        print("\n" + "="*80)
        print("GPU PERFORMANCE PROFILING SUMMARY")
        print("="*80)
        
        # 定义主要stage和子stage
        main_stages = ['V2M4_Algorithm', 'Normal_Prediction', 'Top_N_Selection', 'PSO_Search', 'Gradient_Refinement', 'DUSt3R_Estimation']
        
        # 表头
        print(f"{'Stage':<25} {'Duration':<12} {'GPU Memory (MB)':<20} {'GPU Util (%)':<15}")
        print(f"{'Name':<25} {'(seconds)':<12} {'Start/Peak/End':<20} {'Avg/Peak':<15}")
        print("-" * 80)
        
        # 显示主要阶段
        print("MAIN STAGES:")
        main_total = 0
        for metrics in self.metrics:
            if metrics.name in main_stages:
                # 只对非V2M4_Algorithm的主要stage计算总时间，避免重复计算
                if metrics.name != 'V2M4_Algorithm':
                    main_total += metrics.duration
                    
                memory_info = f"{metrics.gpu_memory_start:.0f}/{metrics.gpu_memory_peak:.0f}/{metrics.gpu_memory_end:.0f}"
                util_info = f"{metrics.gpu_utilization_avg:.1f}/{metrics.gpu_utilization_peak:.1f}"
                
                print(f"{metrics.name:<25} {metrics.duration:<12.2f} {memory_info:<20} {util_info:<15}")
        
        # 显示子阶段
        sub_stages = [m for m in self.metrics if m.name not in main_stages]
        if sub_stages:
            print("\nSUB STAGES:")
            for metrics in sub_stages:
                memory_info = f"{metrics.gpu_memory_start:.0f}/{metrics.gpu_memory_peak:.0f}/{metrics.gpu_memory_end:.0f}"
                util_info = f"{metrics.gpu_utilization_avg:.1f}/{metrics.gpu_utilization_peak:.1f}"
                
                print(f"  {metrics.name:<23} {metrics.duration:<12.2f} {memory_info:<20} {util_info:<15}")
        
        print("-" * 80)
        
        # 显示总时间信息
        v2m4_total = next((m.duration for m in self.metrics if m.name == 'V2M4_Algorithm'), 0)
        if v2m4_total > 0:
            print(f"{'ACTUAL TOTAL':<25} {v2m4_total:<12.2f} (V2M4_Algorithm)")
        
        if main_total > 0:
            print(f"{'MAIN STAGES TOTAL':<25} {main_total:<12.2f} (excludes V2M4_Algorithm)")
        
        print("="*80)

# 全局性能监控器实例
_profiler = GPUProfiler()

def enable_profiling():
    """启用性能监控"""
    _profiler.enable()

def disable_profiling():
    """禁用性能监控"""
    _profiler.disable()

def print_profiling_summary():
    """打印性能监控摘要"""
    _profiler.print_summary()

def profile_stage(stage_name: str):
    """装饰器：监控函数/方法的GPU性能"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _profiler.start_monitoring_stage(stage_name)
            result = func(*args, **kwargs)
            _profiler.end_monitoring_stage(stage_name)
            return result
        return wrapper
    return decorator 