"""
可视化模块
为V2M4相机搜索算法提供结果可视化功能
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from PIL import Image
import trimesh

from .core import CameraPose, DataPair
from .utils import compute_image_similarity

class V2M4Visualizer:
    """V2M4算法结果可视化器"""
    
    def __init__(self, output_dir: str = "outputs/visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_result_comparison(self, 
                               data_pair: DataPair,
                               reference_image: np.ndarray,
                               rendered_result: Optional[np.ndarray],
                               final_pose: CameraPose,
                               mesh_info: Optional[Dict] = None,
                               algorithm_stats: Optional[Dict] = None,
                               execution_time: float = 0.0) -> str:
        """创建结果对比图"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = self.output_dir / f"v2m4_result_{data_pair.scene_name}_{timestamp}.png"
        
        # 创建3列布局的对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 第1列：参考图像
        axes[0].imshow(reference_image)
        axes[0].set_title("Reference Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 第2列：渲染结果
        if rendered_result is not None:
            axes[1].imshow(rendered_result)
            
            # 计算相似性指标
            similarity = compute_image_similarity(reference_image, rendered_result)
            
            title = f"V2M4 Estimated Pose\n(Similarity: {similarity:.3f})"
            title_color = 'green' if similarity > 0.7 else 'orange' if similarity > 0.5 else 'red'
        else:
            placeholder = np.full_like(reference_image, 128, dtype=np.uint8)
            axes[1].imshow(placeholder)
            title = "V2M4 Estimated Pose\n(Rendering Failed)"
            title_color = 'red'
        
        axes[1].set_title(title, fontsize=14, fontweight='bold', color=title_color)
        axes[1].axis('off')
        
        # 第3列：详细信息
        info_text = self._generate_info_text(
            data_pair, final_pose, mesh_info, algorithm_stats, execution_time
        )
        
        axes[2].text(0.05, 0.95, info_text, fontsize=10, 
                    transform=axes[2].transAxes, verticalalignment='top',
                    fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightgray", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        axes[2].set_title("Algorithm Details", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 可视化结果已保存: {comparison_path}")
        return str(comparison_path)
    
    def create_pose_progression_visualization(self,
                                            data_pair: DataPair,
                                            reference_image: np.ndarray,
                                            progression_data: List[Dict],
                                            final_pose: CameraPose) -> str:
        """创建姿态优化过程可视化"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        progression_path = self.output_dir / f"v2m4_progression_{data_pair.scene_name}_{timestamp}.png"
        
        # 准备数据
        num_steps = len(progression_data)
        if num_steps == 0:
            return ""
        
        # 创建多行布局
        rows = 2
        cols = min(4, num_steps + 1)  # 最多4列，包括参考图像
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 第一张图：参考图像
        axes[0, 0].imshow(reference_image)
        axes[0, 0].set_title("Reference Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 显示优化步骤
        for i, step_data in enumerate(progression_data[:cols-1]):
            col_idx = i + 1
            
            # 上排：渲染结果
            if 'rendered_image' in step_data and step_data['rendered_image'] is not None:
                axes[0, col_idx].imshow(step_data['rendered_image'])
            else:
                placeholder = np.full_like(reference_image, 128, dtype=np.uint8)
                axes[0, col_idx].imshow(placeholder)
            
            step_name = step_data.get('step_name', f'Step {i+1}')
            similarity = step_data.get('similarity', 0.0)
            axes[0, col_idx].set_title(f"{step_name}\nSim: {similarity:.3f}", 
                                      fontsize=10, fontweight='bold')
            axes[0, col_idx].axis('off')
            
            # 下排：姿态参数
            pose = step_data.get('pose')
            if pose:
                param_text = f"""Pose Parameters:
Elevation: {pose.elevation:.1f}°
Azimuth: {pose.azimuth:.1f}°
Distance: {pose.radius:.2f}
Center: ({pose.center_x:.2f}, {pose.center_y:.2f}, {pose.center_z:.2f})

Score: {step_data.get('score', 'N/A')}"""
            else:
                param_text = "No pose data available"
            
            axes[1, col_idx].text(0.1, 0.9, param_text, fontsize=8,
                                 transform=axes[1, col_idx].transAxes, 
                                 verticalalignment='top', fontfamily='monospace')
            axes[1, col_idx].set_xlim(0, 1)
            axes[1, col_idx].set_ylim(0, 1)
            axes[1, col_idx].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(progression_data) + 1, cols):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        # 下排第一个位置：算法总结
        summary_text = f"""V2M4 Algorithm Summary:
Scene: {data_pair.scene_name}
Total Steps: {num_steps}

Final Pose:
• Elevation: {final_pose.elevation:.2f}°
• Azimuth: {final_pose.azimuth:.2f}°
• Distance: {final_pose.radius:.3f}

Optimization Progress:
{' → '.join([step.get('step_name', f'S{i+1}') for i, step in enumerate(progression_data)])}"""
        
        axes[1, 0].text(0.1, 0.9, summary_text, fontsize=9,
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="lightblue", alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.savefig(progression_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 优化过程可视化已保存: {progression_path}")
        return str(progression_path)
    
    def create_batch_results_summary(self, 
                                   batch_results: Dict[str, Any],
                                   execution_times: Dict[str, float]) -> str:
        """创建批量处理结果总结"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.output_dir / f"v2m4_batch_summary_{timestamp}.png"
        
        # 统计数据
        total_scenes = len(batch_results)
        successful_scenes = sum(1 for result in batch_results.values() if result is not None)
        success_rate = (successful_scenes / total_scenes) * 100 if total_scenes > 0 else 0
        
        avg_time = np.mean(list(execution_times.values())) if execution_times else 0
        total_time = sum(execution_times.values()) if execution_times else 0
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 成功率饼图
        success_data = [successful_scenes, total_scenes - successful_scenes]
        success_labels = ['Successful', 'Failed']
        success_colors = ['lightgreen', 'lightcoral']
        
        ax1.pie(success_data, labels=success_labels, colors=success_colors, autopct='%1.1f%%')
        ax1.set_title(f'Success Rate: {success_rate:.1f}%\n({successful_scenes}/{total_scenes} scenes)', 
                     fontsize=14, fontweight='bold')
        
        # 2. 执行时间分布
        if execution_times:
            times = list(execution_times.values())
            ax2.hist(times, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Execution Time (seconds)')
            ax2.set_ylabel('Number of Scenes')
            ax2.set_title(f'Execution Time Distribution\nAvg: {avg_time:.1f}s, Total: {total_time:.1f}s', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. 场景结果列表
        scene_names = list(batch_results.keys())
        scene_statuses = ['✅' if batch_results[name] is not None else '❌' for name in scene_names]
        scene_times = [execution_times.get(name, 0) for name in scene_names]
        
        # 创建表格
        table_data = []
        for i, (name, status, time_val) in enumerate(zip(scene_names, scene_statuses, scene_times)):
            table_data.append([f"{i+1:2d}", name, status, f"{time_val:.1f}s"])
        
        table = ax3.table(cellText=table_data, 
                         colLabels=['#', 'Scene Name', 'Status', 'Time'],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == 2:  # 状态列
                if i > 0:
                    cell.set_facecolor('#E8F5E8' if '✅' in cell.get_text().get_text() else '#FFE8E8')
        
        ax3.axis('off')
        ax3.set_title('Individual Scene Results', fontsize=14, fontweight='bold')
        
        # 4. 总结信息
        summary_info = f"""V2M4 Batch Processing Summary
{'='*40}

📊 Overall Statistics:
• Total Scenes: {total_scenes}
• Successful: {successful_scenes} ({success_rate:.1f}%)
• Failed: {total_scenes - successful_scenes}

⏱️ Performance:
• Total Time: {total_time:.1f} seconds
• Average Time: {avg_time:.1f} seconds per scene
• Fastest: {min(execution_times.values()) if execution_times else 0:.1f}s
• Slowest: {max(execution_times.values()) if execution_times else 0:.1f}s

🎯 Algorithm Configuration:
• Initial Samples: 2000
• Top-N Selection: 7
• PSO Particles: 50
• PSO Iterations: 20
• Gradient Descent: 100 iterations

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        ax4.text(0.05, 0.95, summary_info, fontsize=10,
                transform=ax4.transAxes, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightyellow", alpha=0.9))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📋 批量处理总结已保存: {summary_path}")
        return str(summary_path)
    
    def save_individual_results(self,
                               data_pair: DataPair,
                               reference_image: np.ndarray,
                               rendered_result: Optional[np.ndarray]) -> List[str]:
        """保存单独的结果图像"""
        
        saved_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存参考图像
        ref_path = self.output_dir / f"reference_{data_pair.scene_name}_{timestamp}.png"
        Image.fromarray(reference_image).save(ref_path)
        saved_paths.append(str(ref_path))
        
        # 保存渲染结果
        if rendered_result is not None:
            render_path = self.output_dir / f"rendered_{data_pair.scene_name}_{timestamp}.png"
            Image.fromarray(rendered_result).save(render_path)
            saved_paths.append(str(render_path))
        
        return saved_paths
    
    def _generate_info_text(self, 
                           data_pair: DataPair,
                           final_pose: CameraPose,
                           mesh_info: Optional[Dict],
                           algorithm_stats: Optional[Dict],
                           execution_time: float) -> str:
        """生成详细信息文本"""
        
        info_lines = [
            "🎯 V2M4 Camera Search Results",
            "=" * 35,
            "",
            f"📁 Scene: {data_pair.scene_name}",
            f"🕒 Time: {execution_time:.1f}s",
            "",
            "📐 Final Camera Pose:",
            f"• Elevation: {final_pose.elevation:.2f}°",
            f"• Azimuth: {final_pose.azimuth:.2f}°", 
            f"• Distance: {final_pose.radius:.3f}",
            f"• Center: ({final_pose.center_x:.2f}, {final_pose.center_y:.2f}, {final_pose.center_z:.2f})",
            ""
        ]
        
        # 添加mesh信息
        if mesh_info:
            info_lines.extend([
                "🔺 Mesh Information:",
                f"• Vertices: {mesh_info.get('vertices_count', 'N/A')}",
                f"• Faces: {mesh_info.get('faces_count', 'N/A')}",
                f"• Scale: {mesh_info.get('scale', 0):.2f}",
                ""
            ])
        
        # 添加算法统计
        if algorithm_stats:
            info_lines.extend([
                "⚙️ Algorithm Stats:",
                f"• Initial Samples: {algorithm_stats.get('initial_samples', 'N/A')}",
                f"• Top-N Selected: {algorithm_stats.get('top_n', 'N/A')}",
                f"• PSO Iterations: {algorithm_stats.get('pso_iterations', 'N/A')}",
                f"• Final Score: {algorithm_stats.get('final_score', 'N/A'):.4f}",
                ""
            ])
        
        # 添加系统信息
        import torch
        info_lines.extend([
            "💻 System:",
            f"• GPU: {'Available' if torch.cuda.is_available() else 'Not Available'}",
            f"• Generated: {datetime.now().strftime('%H:%M:%S')}"
        ])
        
        return "\n".join(info_lines) 