"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-11-29 22:13:49 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-29 22:26:42
 */
"""
"""该模块提供了一系列工具函数，用于模型的构建和训练过程。

功能包括：
- 激活函数的实现（swish）。
- 构建包含线性层、批归一化和激活函数的网络块。
- 特殊激活函数（SSP, SWISH）的类实现。
- 根据名称获取激活函数。
- 绘制预测值与目标值的散点图。
- 自定义损失函数的实现。
- 解析配置文件中的度量函数列表。
- 根据配置获取超参数字典。
- 计算三元组（triplet）信息。
"""
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import numpy as np
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, BatchNorm1d as BN)
from typing import Callable, Union
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
from easydict import EasyDict
from scipy.stats import gaussian_kde

def swish(x):
    """Swish 激活函数"""
    return x * x.sigmoid()

def linear_bn_act(in_features: int, out_features: int, lbias: bool = False, activation: Callable = None, use_batch_norm: bool = False):
    """创建一个包含线性层、批归一化层和激活函数的序列模块。

    Args:
        in_features (int): 输入特征维度。
        out_features (int): 输出特征维度。
        lbias (bool, optional): 线性层是否使用偏置。默认为 False。
        activation (Callable, optional): 激活函数模块。默认为 None。
        use_batch_norm (bool, optional): 是否使用批归一化。默认为 False。

    Returns:
        torch.nn.Sequential: 组装好的序列模块。
    """
    if use_batch_norm:
        if activation is None:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features))
        else:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features), activation)
    else:
        if activation is None:
            return Linear(in_features, out_features, lbias)
        else:
            return Sequential(Linear(in_features, out_features, lbias), activation)

class SSP(nn.Module):
    r"""应用逐元素的 Shifted SoftPlus (SSP) 激活函数。

    SSP 的计算公式为: :math:`\text{SSP}(x)=\text{Softplus}(x)-\text{Softplus}(0)`。
    这确保了 :math:`\text{SSP}(0)=0`。

    Args:
        beta: Softplus 公式中的 :math:`\beta` 值。默认为 1。
        threshold: 当输入值高于此阈值时，Softplus 将退化为线性函数。默认为 20。

    Shape:
        - 输入: :math:`(N, *)`，其中 `*` 表示任意数量的附加维度。
        - 输出: :math:`(N, *)`，形状与输入相同。
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class SWISH(nn.Module):
    """SWISH 激活函数模块"""
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, input):
        return swish(input)

def get_activation(name):
    """根据字符串名称返回对应的激活函数实例。

    Args:
        name (str): 激活函数的名称，支持带有参数，如 "elu(1.0)"。

    Returns:
        torch.nn.Module: 激活函数模块的实例。
    
    Raises:
        NameError: 如果输入的名称不被支持。
    """
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'swish':
        return SWISH()
    elif act_name == 'tanh':
        return Tanh()
    elif act_name == 'silu':
        return SiLU()
    elif act_name == 'celu':
        return CELU(alpha)
    else:
        raise NameError("Not supported activation: {}".format(name))

def scatter_plot(pred: np.ndarray = None, target: np.ndarray = None):
    """绘制预测值与目标值的散点图。

    Args:
        pred (np.ndarray, optional): 预测值数组。
        target (np.ndarray, optional): 目标值数组。

    Returns:
        matplotlib.figure.Figure: 绘制好的图表对象。
    """
    fig, ax = plt.subplots()
    """
        try:
        # Calculate the point density
        xy = np.vstack([pred, target])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        pred, target, z = pred[idx], target[idx], z[idx]
        # scatter plot
        ax.scatter(x=pred, y=target, s=25, c=z, marker=".")
    except:
        ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    """
    ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    ax.set_title('Prediction VS Target')
    ax.set_aspect('equal')
    min_val, max_val = np.min([target, pred]), np.max([target, pred])
    ax.plot([min_val, max_val], [min_val, max_val],
            ls="--", linewidth=1, c='r')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig

class cosine_similarity_loss(nn.Module):
    """计算 1 - 余弦相似度的损失函数。"""
    def __init__(self):
        super(cosine_similarity_loss, self).__init__()

    def forward(self, pred, target):
        vec_product = torch.sum(pred*target, dim=-1)
        pred_norm = torch.norm(pred, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        loss = torch.tensor(1.0).type_as(
            pred) - vec_product/(pred_norm*target_norm)
        loss = torch.mean(loss)
        return loss

class sum_zero_loss(nn.Module):
    """计算预测值总和的 L2 范数，用于约束总和为零。"""
    def __init__(self):
        super(sum_zero_loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.sum(pred, dim=0).pow(2).sum(dim=-1).sqrt()
        return loss

class Euclidean_loss(nn.Module):
    """计算预测值和目标值之间的平均欧几里得距离。"""
    def __init__(self):
        super(Euclidean_loss, self).__init__()

    def forward(self, pred, target):
        dist = (pred - target).pow(2).sum(dim=-1).sqrt()
        loss = torch.mean(dist)
        return loss

class RMSELoss(nn.Module):
    """均方根误差损失。"""
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

def parse_metric_func(losses_list: Union[list, tuple] = None):
    """解析配置文件中的度量函数列表，将字符串名称转换为函数实例。

    Args:
        losses_list (Union[list, tuple], optional): 包含度量函数配置的列表。

    Returns:
        list or tuple: 更新后的列表，其中 'metric' 的值被替换为函数实例。
    """
    for loss_dict in losses_list:
        if loss_dict['metric'].lower() == 'mse':
            loss_dict['metric'] = nn.MSELoss()
        elif loss_dict['metric'].lower() == 'mae':
            loss_dict['metric'] = nn.L1Loss()
        elif loss_dict['metric'].lower() == 'cosine_similarity':
            loss_dict['metric'] = cosine_similarity_loss()
        elif loss_dict['metric'].lower() == 'sum_zero':
            loss_dict['metric'] = sum_zero_loss()
        elif loss_dict['metric'].lower() == 'euclidean_loss':
            loss_dict['metric'] = Euclidean_loss()
        elif loss_dict['metric'].lower() == 'rmse':
            loss_dict['metric'] = RMSELoss()
        else:
            print(f'This metric function is not supported!')
    return losses_list

def get_hparam_dict(config: dict = None):
    """根据配置文件提取并组织用于日志记录的超参数字典。

    Args:
        config (dict, optional): 项目的全局配置对象。

    Returns:
        dict: 包含 GNN 名称和相关超参数的字典。
    """
    if config.setup.GNN_Net.lower() == 'dimnet':
        hparam_dict = config.representation_nets.dimnet_params
    elif config.setup.GNN_Net.lower() == 'edge_gnn':
        hparam_dict = config.representation_nets.Edge_GNN
    elif config.setup.GNN_Net.lower() == 'schnet':
        hparam_dict = config.representation_nets.SchNet
    elif config.setup.GNN_Net.lower() == 'cgcnn':
        hparam_dict = config.representation_nets.cgcnn
    elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
        hparam_dict = config.representation_nets.cgcnn_edge
    elif config.setup.GNN_Net.lower() == 'painn':
        hparam_dict = config.representation_nets.painn
    elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
        hparam_dict = config.representation_nets.cgcnn_triplet
    elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
        hparam_dict = config.representation_nets.dimenet_triplet
    elif config.setup.GNN_Net.lower() == 'dimeham':
        hparam_dict = config.representation_nets.dimeham
    elif config.setup.GNN_Net.lower() == 'dimeorb':
        hparam_dict = config.representation_nets.dimeorb
    elif config.setup.GNN_Net.lower() == 'schnorb':
        hparam_dict = config.representation_nets.schnorb
    elif config.setup.GNN_Net.lower() == 'nequip':
        hparam_dict = config.representation_nets.nequip
    elif config.setup.GNN_Net.lower() == 'hamgnn_pre':
        hparam_dict = config.representation_nets.HamGNN_pre
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()
    for key in hparam_dict:
        if type(hparam_dict[key]) not in [str, float, int, bool, None]:
            hparam_dict[key] = type(hparam_dict[key]).__name__.split(".")[-1]
    out = {'GNN_Name': config.setup.GNN_Net}
    out.update(dict(hparam_dict))
    return out

def triplets(edge_index, nbr_shift, nbr_counts):
    """计算图中的所有三元组 (i, j, k)，其中 j 是中心原子。

    Args:
        edge_index (torch.Tensor): 边索引，形状为 [2, N_edges]。
        nbr_shift (torch.Tensor): 边的周期性偏移向量。
        nbr_counts (torch.Tensor): 每个节点的近邻数量。

    Returns:
        tuple: 包含三元组信息的元组 (col_i, row_j, idx_i, idx_j, idx_k, idx_kj, idx_ji)。
    """
    row_k, col_j = edge_index  # k->j
    row_j, col_i = edge_index  # j->i
    idx_k, idx_j, idx_i, idx_kj, idx_ji = [], [], [], [], []

    nbr_counts_cumsum = [0] + torch.cumsum(nbr_counts, dim=0).tolist()

    for edge_kj, j in enumerate(col_j):
        j_j = torch.arange(nbr_counts_cumsum[j], nbr_counts_cumsum[j+1]).tolist()
        idx_kj += [edge_kj]*len(j_j)
        idx_ji += j_j
        idx_k += [row_k[edge_kj]]*len(j_j)
        idx_j += [j]*len(j_j)
        idx_i += col_i[j_j].tolist()      

    idx_k = torch.LongTensor(idx_k).type_as(edge_index)
    idx_j = torch.LongTensor(idx_j).type_as(edge_index)
    idx_i = torch.LongTensor(idx_i).type_as(edge_index)
    idx_kj = torch.LongTensor(idx_kj).type_as(edge_index)
    idx_ji = torch.LongTensor(idx_ji).type_as(edge_index)

    # 移除 i == k 的三元组，除非它们在不同的周期性晶胞中
    mask = (idx_i != idx_k) | ((nbr_shift[idx_kj]+nbr_shift[idx_ji]).pow(2).sum(dim=-1).sqrt() > 1.0e-3)
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # 对应三元组的边索引 (k->j, j->i)
    idx_kj = idx_kj[mask]
    idx_ji = idx_ji[mask]

    """
    idx_i -> pos[idx_i]
    idx_j -> pos[idx_j] - nbr_shift[idx_ji]
    idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj] 
    """
    
    return col_i, row_j, idx_i, idx_j, idx_k, idx_kj, idx_ji