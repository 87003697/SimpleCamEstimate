# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
真实的VGGT模型导入 - 基于V2M4实现
"""

try:
    # 使用真实的VGGT模型
    from vggt.models.vggt import VGGT
    print("✅ 成功导入真实的VGGT模型")
except ImportError as e:
    print(f"❌ 无法导入真实的VGGT模型: {e}")
    print("请确保已安装VGGT包：pip install vggt")
    
    # 如果无法导入真实模型，创建一个错误提示类
    class VGGT:
        @classmethod
        def from_pretrained(cls, model_name):
            raise ImportError(
                f"无法加载VGGT模型 '{model_name}'。\n"
                "请确保已正确安装VGGT包：\n"
                "pip install vggt\n"
                "或者从源码安装：\n"
                "git clone https://github.com/facebookresearch/vggt.git\n"
                "cd vggt && pip install -e ."
            )

# 导出VGGT类
__all__ = ['VGGT']
