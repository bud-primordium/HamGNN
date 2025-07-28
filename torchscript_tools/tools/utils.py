#!/usr/bin/env python
"""
HamGNN TorchScript 工具函数集合

提供统一的数据处理和转换功能
"""

import torch
from typing import Dict, Union, Optional
from torch_geometric.data import Data, Batch


def batch_to_input_dict(batch: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
    """
    将 PyTorch Geometric 的 Data 或 Batch 对象转换为模型输入字典
    
    这是所有 HamGNN TorchScript 工具的核心数据转换函数，确保数据格式的一致性。
    根据实际HamGNN模型的需求，包含所有必需的属性。
    
    Args:
        batch: PyTorch Geometric 的 Data 或 Batch 对象
        
    Returns:
        包含模型所需所有字段的字典
        
    Note:
        这个函数基于HamGNN v2.0的实际需求，确保包含所有在模型forward中访问的属性
    """
    data_dict = {}
    
    # === 必需的基础字段 ===
    required_fields = [
        'pos',               # 原子位置
        'cell',              # 晶胞参数
        'z',                 # 原子序数
        'edge_index',        # 边索引
    ]
    
    for field in required_fields:
        if hasattr(batch, field):
            data_dict[field] = getattr(batch, field)
        else:
            raise ValueError(f"Required field '{field}' not found in batch")
    
    # === 批次相关字段 ===
    if hasattr(batch, 'batch'):
        data_dict['batch'] = batch.batch
    else:
        # 单个图的情况，创建全零批次索引
        data_dict['batch'] = torch.zeros(batch.pos.size(0), dtype=torch.long, device=batch.pos.device)
    
    # node_counts - 必需字段
    if hasattr(batch, 'node_counts'):
        data_dict['node_counts'] = batch.node_counts
    else:
        # 如果没有node_counts，从batch推断
        unique_batch = torch.unique(data_dict['batch'])
        node_counts = torch.bincount(data_dict['batch'])
        data_dict['node_counts'] = node_counts
    
    # === 图结构字段 ===
    graph_structure_fields = [
        'inv_edge_idx',      # 反向边索引
        'nbr_shift',         # 邻居位移（必需）
        'cell_shift',        # 晶胞位移
    ]
    
    for field in graph_structure_fields:
        if hasattr(batch, field):
            data_dict[field] = getattr(batch, field)
    
    # === 哈密顿量和重叠矩阵相关字段 ===
    hamiltonian_fields = [
        # 基本哈密顿量和重叠矩阵
        'Hon', 'Hoff',       # 哈密顿量 onsite/offsite
        'Son', 'Soff',       # 重叠矩阵 onsite/offsite
        'Hon0', 'Hoff0',     # 零阶哈密顿量 onsite/offsite (SOC相关)
        
        # 原始分量（如果存在）
        'H_u', 'H_d',        # 自旋上/下哈密顿量分量
        'H0_u', 'H0_d',      # 零阶自旋上/下哈密顿量分量
        
        # SOC相关（虚部）
        'iHon0', 'iHoff0',   # 虚部哈密顿量 onsite/offsite
        
        # 非SOC版本
        'Hon_nonsoc', 'Hoff_nonsoc',
        
        # 导数（如果存在）
        'dSon', 'dSoff',     # 重叠矩阵导数
        
        # 完整矩阵（用于验证）
        'hamiltonian',       # 完整哈密顿量矩阵
        'overlap',           # 完整重叠矩阵
        'hamiltonian_real',  # 实部哈密顿量（SOC情况）
    ]
    
    for field in hamiltonian_fields:
        if hasattr(batch, field):
            data_dict[field] = getattr(batch, field)
    
    # === k点相关字段 ===
    k_point_fields = [
        'k_vecs',            # k点向量（可能由模型生成）
    ]
    
    for field in k_point_fields:
        if hasattr(batch, field):
            data_dict[field] = getattr(batch, field)
    
    # === 图构建相关字段 ===
    graph_construction_fields = [
        'unique_cell_shift',     # 唯一晶胞位移
        'cell_shift_indices',    # 晶胞位移索引
        'zero_shift_idx',        # 零位移索引
    ]
    
    for field in graph_construction_fields:
        if hasattr(batch, field):
            data_dict[field] = getattr(batch, field)
    
    # === 其他属性字段 ===
    other_fields = [
        'node_attrs',        # 节点属性
        'edge_feats',        # 边特征
        'edge_attrs',        # 边属性
        'total_energy',      # 总能量（用于验证）
        'ptr',               # 批次指针
        'num_nodes',         # 节点数
        'num_edges',         # 边数
    ]
    
    for field in other_fields:
        if hasattr(batch, field):
            value = getattr(batch, field)
            # TorchScript要求Dict[str, Tensor]，只包含tensor
            if torch.is_tensor(value):
                data_dict[field] = value
    
    return data_dict


def load_single_graph_from_npz(npz_path: str, 
                              graph_idx: int = 0,
                              device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    从 HamGNN npz 文件加载单个图并转换为输入字典
    
    Args:
        npz_path: graph_data.npz 文件路径
        graph_idx: 要加载的图索引
        device: 目标设备 ('cpu' 或 'cuda')
        
    Returns:
        准备好的输入字典
        
    Note:
        这个函数主要用于测试和演示，生产环境建议使用 DataLoader
    """
    import numpy as np
    
    # 加载 npz 文件
    npz_data = np.load(npz_path, allow_pickle=True)
    graph_dict = npz_data['graph'].item()
    
    # 检查索引有效性
    if graph_idx not in graph_dict:
        available_indices = list(graph_dict.keys())
        raise ValueError(f"图索引 {graph_idx} 不存在。可用索引: {available_indices[:10]}...")
    
    # 获取 Data 对象
    graph_data = graph_dict[graph_idx]
    
    # 使用核心转换函数
    data_dict = batch_to_input_dict(graph_data)
    
    # 移动到指定设备
    if device is not None:
        device_obj = torch.device(device)
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device_obj)
    
    return data_dict


def compare_model_outputs(output1: Dict[str, torch.Tensor], 
                         output2: Dict[str, torch.Tensor],
                         rtol: float = 1e-7,
                         atol: float = 1e-7) -> Dict[str, Dict[str, float]]:
    """
    比较两个模型输出的数值差异
    
    用于验证编译前后模型的一致性
    
    Args:
        output1: 第一个模型的输出
        output2: 第二个模型的输出
        rtol: 相对容差
        atol: 绝对容差
        
    Returns:
        包含每个输出键的差异统计信息
    """
    results = {}
    
    # 找出共同的输出键
    common_keys = set(output1.keys()) & set(output2.keys())
    
    for key in common_keys:
        tensor1 = output1[key]
        tensor2 = output2[key]
        
        # 确保形状一致
        if tensor1.shape != tensor2.shape:
            results[key] = {
                'error': f'形状不匹配: {tensor1.shape} vs {tensor2.shape}'
            }
            continue
        
        # 计算差异
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (torch.abs(tensor1) + 1e-10)
        
        results[key] = {
            'max_abs_diff': abs_diff.max().item(),
            'mean_abs_diff': abs_diff.mean().item(),
            'max_rel_diff': rel_diff.max().item(),
            'mean_rel_diff': rel_diff.mean().item(),
            'is_close': torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        }
    
    # 检查是否有不匹配的键
    only_in_output1 = set(output1.keys()) - set(output2.keys())
    only_in_output2 = set(output2.keys()) - set(output1.keys())
    
    if only_in_output1:
        results['_missing_in_output2'] = list(only_in_output1)
    if only_in_output2:
        results['_missing_in_output1'] = list(only_in_output2)
    
    return results