#!/usr/bin/env python
"""
HamGNN 推理模型封装

将 representation 和 output_module 组合成适合 TorchScript 编译的推理模型
"""

import torch
import torch.nn as nn
from typing import Dict


class HamGNNInference(nn.Module):
    """
    HamGNN 推理模型
    
    将训练时分离的 representation 和 output_module 组合成单一模型，
    用于 TorchScript 编译和部署。
    
    Args:
        representation: HamGNN 表示学习模块 (如 HamGNNConvE3)
        output_module: HamGNN 输出模块 (如 HamGNNPlusPlusOut)
    """
    
    def __init__(self, representation: nn.Module, output_module: nn.Module):
        super().__init__()
        self.representation = representation
        self.output_module = output_module
        
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推理前向传播
        
        Args:
            data: 包含图数据的字典，必须包含以下键：
                - pos: 原子位置 [num_atoms, 3]
                - cell: 晶胞参数 [batch_size, 3, 3]
                - node_attrs: 节点特征
                - edge_index: 边索引
                - edge_feats: 边特征
                - batch: 批次索引
                
        Returns:
            包含预测结果的字典，通常包含：
                - hamiltonian: 预测的哈密顿量矩阵
                - 其他可能的输出（取决于 output_module）
        """
        # 步骤1: 计算原子表示
        representation = self.representation(data)
        
        # 步骤2: 基于表示计算输出
        predictions = self.output_module(data, representation)
        
        return predictions
    
    # TorchScript兼容: 移除device属性，避免生成器相关问题
    # 在推理模式下，设备信息应该在模型加载时确定
    
    def to(self, *args, **kwargs):
        """重写 to 方法以确保子模块正确迁移"""
        super().to(*args, **kwargs)
        return self