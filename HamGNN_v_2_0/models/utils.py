"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-11-29 22:13:49 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-29 22:26:42
 */
"""
"""该模块提供了一系列在模型构建和训练过程中使用的工具函数和辅助类。

内容包括：
- 自定义激活函数 (SSP, SWISH) 和激活函数获取器。
- 各种损失函数 (cosine_similarity_loss, sum_zero_loss, Euclidean_loss, RMSELoss)。
- 绘图工具 (scatter_plot)。
- 配置解析和度量函数解析工具。
- 图算法辅助函数 (triplets)。
- 基于 e3nn 的张量操作类 (Expansion)。
- 其他张量操作工具函数。
"""
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import numpy as np
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, BatchNorm1d as BN)
from typing import Callable, Union, Optional
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
from easydict import EasyDict
from scipy.stats import gaussian_kde
from e3nn import o3
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish 激活函数。

    计算公式为 `x * sigmoid(x)`。

    Args:
        x (torch.Tensor): 输入张量。

    Returns:
        torch.Tensor: 经过 Swish 激活后的张量。
    """
    return x * x.sigmoid()

def linear_bn_act(in_features: int, out_features: int, lbias: bool = False, activation: Optional[Callable] = None, use_batch_norm: bool = False) -> Union[nn.Module, nn.Sequential]:
    """根据输入参数灵活构建一个线性层、批量归一化和激活函数的组合模块。

    该函数根据 `use_batch_norm` 和 `activation` 是否提供，来构造不同组合的模块。

    .. warning::
       该函数的返回值类型不是固定的。当 `use_batch_norm` 为 `False` 且 `activation`
       为 `None` 时，它返回一个独立的 `nn.Linear` 模块；在其他所有情况下，
       它返回一个 `nn.Sequential` 容器。调用者需要注意处理这个差异。

    Args:
        in_features (int): 线性层的输入特征维度。
        out_features (int): 线性层的输出特征维度。
        lbias (bool, optional): 线性层是否使用偏置。默认为 `False`。
        activation (Optional[Callable], optional): 要应用的激活函数实例。如果为 `None`，则不添加激活函数。默认为 `None`。
        use_batch_norm (bool, optional): 是否在激活之前应用批量归一化。默认为 `False`。

    Returns:
        Union[nn.Module, nn.Sequential]: 组建好的 Pytorch 模块。
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
    def __init__(self, beta: float = 1, threshold: float = 20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold
        # 预计算 softplus(0) 的值以提高效率
        self.sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。"""
        return F.softplus(input, self.beta, self.threshold) - self.sp0

    def extra_repr(self) -> str:
        """返回模块的额外表示信息，用于打印。"""
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class SWISH(nn.Module):
    """Swish 激活函数的模块封装。"""
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。"""
        return swish(input)

def get_activation(name: str) -> nn.Module:
    """根据名称字符串获取并实例化一个激活函数模块。

    支持的激活函数包括 'softplus', 'ssp', 'elu', 'relu', 'selu', 'swish',
    'tanh', 'silu', 'celu'。对于 'elu' 和 'celu'，可以指定 alpha 参数，
    例如 "elu(0.5)"。

    Args:
        name (str): 激活函数的名称。

    Returns:
        nn.Module: 对应的激活函数实例。

    Raises:
        NameError: 如果请求的激活函数不受支持。
    """
    act_name = name.lower()
    # 使用正则表达式解析可能带参数的激活函数名称，例如 "elu(0.5)"
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0  # 默认 alpha 值
        
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
        raise NameError("不支持的激活函数: {}".format(name))

def scatter_plot(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    """生成一个预测值 vs. 目标值的散点图。

    图中会画出 y=x 的虚线作为参考。可选地，可以使用核密度估计
    为散点着色，但当前版本为简化实现，使用了固定的绿色。

    Args:
        pred (np.ndarray): 预测值的一维数组。
        target (np.ndarray): 目标值的一维数组。

    Returns:
        plt.Figure: 生成的 matplotlib Figure 对象。
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
    # 简单的散点图实现
    ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    ax.set_title('Prediction VS Target')
    ax.set_aspect('equal') # 保证 x,y 轴刻度相同
    
    # 绘制 y=x 参考线
    min_val, max_val = np.min([target, pred]), np.max([target, pred])
    ax.plot([min_val, max_val], [min_val, max_val],
            ls="--", linewidth=1, c='r')
            
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig

class cosine_similarity_loss(nn.Module):
    """计算两个向量之间的余弦相似度损失。

    损失定义为 `1 - cos(theta)`，其中 `theta` 是两个向量的夹角。
    该损失鼓励两个向量指向相同的方向。
    """
    def __init__(self):
        super(cosine_similarity_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。"""
        # 逐元素点积
        vec_product = torch.sum(pred*target, dim=-1)
        # 计算各自的 L2 范数
        pred_norm = torch.norm(pred, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        # 计算损失，并取批次平均
        loss = torch.tensor(1.0).type_as(pred) - vec_product/(pred_norm*target_norm)
        loss = torch.mean(loss)
        return loss

class sum_zero_loss(nn.Module):
    """一个约束预测向量总和为零的损失。

    该损失计算预测向量在批次维度上求和后的 L2 范数。
    这在需要满足某些物理守恒定律（如总力为零）时非常有用。
    """
    def __init__(self):
        super(sum_zero_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。 target 在此损失中未使用。"""
        loss = torch.sum(pred, dim=0).pow(2).sum(dim=-1).sqrt()
        return loss

class Euclidean_loss(nn.Module):
    """计算预测值和目标值之间的平均欧几里得距离。"""
    def __init__(self):
        super(Euclidean_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。"""
        dist = (pred - target).pow(2).sum(dim=-1).sqrt()
        loss = torch.mean(dist)
        return loss

class RMSELoss(nn.Module):
    """计算均方根误差 (RMSE) 损失。"""
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。"""
        return torch.sqrt(self.mse(pred, target))

def parse_metric_func(losses_list: list) -> list:
    """解析一个包含损失函数信息的列表，并将字符串名称替换为实际的损失函数实例。

    Args:
        losses_list (list): 一个字典列表，每个字典包含 'metric' (str) 和其他参数。

    Returns:
        list: 更新后的列表，其中 'metric' 的值被替换为 nn.Module 实例。
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
            raise ValueError(f'不支持的度量函数: {loss_dict["metric"]}')
    return losses_list

def get_hparam_dict(config: EasyDict) -> dict:
    """从配置对象中提取并格式化用于日志记录的超参数字典。

    它根据 `config.setup.GNN_Net` 的值从 `config.representation_nets`
    中选择对应的参数字典，并进行一些格式化处理。

    Args:
        config (EasyDict): 全局配置对象。

    Returns:
        dict: 格式化后的超参数字典，适用于 TensorBoard 等日志工具。
    """
    # 查找与GNN网络名称匹配的参数配置
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
    elif config.setup.GNN_Net.lower()[:6] == 'hamgnn':
        hparam_dict = config.representation_nets.HamGNN_pre
    else:
        print(f"不支持的网络 {config.setup.GNN_Net}")
        quit()
    for key in hparam_dict:
        if type(hparam_dict[key]) not in [str, float, int, bool, None]:
            hparam_dict[key] = type(hparam_dict[key]).__name__.split(".")[-1]
    out = {'GNN_Name': config.setup.GNN_Net}
    out.update(dict(hparam_dict))
    return out

def triplets(edge_index: torch.Tensor, num_nodes: int, cell_shift: torch.Tensor) -> tuple:
    """从边列表计算原子三元组 (k -> j -> i)。

    这个函数对于构建需要三体相互作用（如键角）的图模型至关重要。
    它首先将边列表转换为稀疏邻接矩阵，然后通过邻接矩阵的乘积思想
    有效地找到所有通过一个中间节点 `j` 连接的原子对 `(k, i)`。

    Args:
        edge_index (torch.Tensor): 形状为 `(2, N_edges)` 的边索引张量，表示 `j -> i` 的边。
        num_nodes (int): 图中的节点总数。
        cell_shift (torch.Tensor): 形状为 `(N_edges, 3)` 的张量，表示每条边跨越的晶胞偏移。

    Returns:
        tuple: 包含三元组信息的元组:
            - col, row (torch.Tensor): 原始的边索引。
            - idx_i, idx_j, idx_k (torch.Tensor): 三元组中 `i`, `j`, `k` 的原子索引。
            - idx_kj, idx_ji (torch.Tensor): 构成三元组的两条边 `k->j` 和 `j->i` 的原始边索引。
    """
    row, col = edge_index  # j->i
            
    value = torch.arange(row.size(0), device=row.device)
    # 构建稀疏邻接矩阵，其中值为边的索引
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    # 对于每个 j->i 的边 (由 row 表示)，找到所有指向 j 的邻居 k
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
                    
    # --- 构建三元组的节点索引 (k->j->i) ---
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
                   
    # --- 构建三元组的边索引 (k->j, j->i) ---
    idx_kj = adj_t_row.storage.value() # 边 k->j 的索引
    idx_ji = adj_t_row.storage.row() # 边 j->i 的索引

    """
    idx_i -> pos[idx_i]
    idx_j -> pos[idx_j] - nbr_shift[idx_ji]
    idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
    """           
    # --- 过滤掉无效的三元组 ---
    # 一个原子不能通过两个周期性边界的像成为自己的邻居，形成 k=i 的情况
    # 除非这两个像的相对晶胞位移不为零
    relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
    mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
    idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]
               
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

def prod(x: list) -> float:
    """计算序列中所有元素的乘积。"""
    out = 1
    for a in x:
        out *= a
    return out

class Expansion(nn.Module):
    """一个使用 e3nn 库实现的等变特征扩展模块。

    该模块的核心功能是将一个输入特征（表示为一个 `e3nn` 的不可约表示 `irrep_in`），
    通过张量积分解，映射到一个由两个其他不可约表示 (`irrep_out_1` 和 `irrep_out_2`)
    构成的二维特征空间中。这在构建等变神经网络的交互块时非常关键，因为它允许
    信息在不同阶数的张量特征之间进行混合和传递。

    例如，一个向量特征 (`1o`) 与另一个向量特征 (`1o`) 交互，可以产生标量 (`0e`)、
    反对称矩阵/伪向量 (`1o`) 和对称无迹矩阵 (`2e`) 特征。这个模块就是实现这种
    分解和映射的计算单元。

    计算的核心是利用 Wigner 3-j 符号 (`o3.wigner_3j`)，它描述了三个角动量
    如何耦合。

    Attributes:
        irrep_in (o3.Irreps): 输入特征的不可约表示。
        irrep_out_1 (o3.Irreps): 输出特征的第一个维度（行）的不可约表示。
        irrep_out_2 (o3.Irreps): 输出特征的第二个维度（列）的不可约表示。
        internal_weights (bool): 控制权重生成方式。如果为 `True`，权重是模块内部
            固定的 `nn.Parameter`；如果为 `False`，权重由一个 `o3.Linear` 层从
            输入特征动态生成。
    """
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2, internal_weights: Optional[bool] = False):
        """构造 Expansion 类的实例。

        Args:
            irrep_in (o3.Irreps): 输入特征的 e3nn 不可约表示。
            irrep_out_1 (o3.Irreps): 输出特征的第一维（行）的 e3nn 不可约表示。
            irrep_out_2 (o3.Irreps): 输出特征的第二维（列）的 e3nn 不可约表示。
            internal_weights (bool, optional): 是否使用内部权重。默认为 `False`。
        """
        super().__init__()
        self.irrep_in = o3.Irreps(irrep_in)
        self.irrep_out_1 = o3.Irreps(irrep_out_1)
        self.irrep_out_2 = o3.Irreps(irrep_out_2)
        
        # --- 步骤 1: 查找所有可能的分解路径 ---
        # 根据群论规则 (ir_in 必须存在于 ir_out1 和 ir_out2 的张量积中)，
        # 确定所有合法的从 `irrep_in`到 `(irrep_out_1, irrep_out_2)` 对的映射路径。
        self.instructions = self.get_expansion_path(self.irrep_in, self.irrep_out_1, self.irrep_out_2)
        
        # --- 步骤 2: 计算所需权重的数量 ---
        # 路径权重：每个合法的分解路径都需要一组可学习的权重。
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        # 偏置权重：仅当输入是标量 (l=0) 时，才存在偏置项。
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias

        # --- 步骤 3: 初始化权重 ---
        self.internal_weights = internal_weights
        if self.internal_weights:
            # 将所有权重（路径权重+偏置）创建为模块内部的一个可训练参数。
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        else:
            # 创建一个 e3nn 线性层，用于从输入特征动态地预测所有权重。
            # 输出是一个标量类型(0e)，通道数为 num_weights。
            self.linear_weight_bias = o3.Linear(self.irrep_in, o3.Irreps([(self.num_weights, (0, 1))]))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """定义前向传播逻辑。

        Args:
            x_in (torch.Tensor): 输入特征张量，其维度应与 `irrep_in` 匹配。

        Returns:
            torch.Tensor: 输出的二维特征张量，其形状为 `(N_batch, irrep_out_1.dim, irrep_out_2.dim)`。
        """
        if self.internal_weights:
            # 如果使用内部权重，则直接使用 `self.weights`，不依赖输入。
            weights, bias_weights = None, None # Placeholder，实际在循环中使用 self.weights
        else:
            # 如果使用动态权重，则通过线性层从输入 `x_in` 生成权重。
            weight_bias = self.linear_weight_bias(x_in)
            weights, bias_weights = torch.split(weight_bias, 
                                               split_size_or_sections=[self.num_path_weight, self.num_bias], dim=-1)
        
        batch_num = x_in.shape[0]
        # 将输入的扁平化张量根据 irrep_in 的定义，切分成不同不可约表示对应的块。
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        # --- 核心计算：遍历所有合法的分解路径 ---
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            
            # 获取 Wigner 3-j 符号，这是进行等变张量积的核心。
            w3j_matrix = o3.wigner_3j(
                mul_ir_out1.ir.l, mul_ir_out2.ir.l, mul_ir_in.ir.l).type_as(x_in)
            
            # 如果该路径需要权重 (ins[3] is True)
            if ins[3] is True or weights is not None:
                if weights is None: # 对应 internal_weights=True 的情况
                    # 从内部参数 self.weights 中切片出当前路径所需的权重
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    # `einsum` 执行带权重的张量积
                    result = torch.einsum(
                        f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else: # 对应 internal_weights=False 的情况
                    # 从动态生成的权重中切片
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    # 先将权重与输入特征作用
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    # 如果输入是标量 (l=0)，则添加偏置项
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
                            reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    # 再与 3j 符号作用，完成张量积
                    result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else: # 如果路径不需要权重，则权重视为全 1
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
                )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            # 将同一目标块 (key) 的结果累加起来
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
        
        # --- 步骤 4: 组装最终的输出张量 ---
        # 将所有计算出的块按照 (irrep_out_1, irrep_out_2) 的网格结构拼接起来。
        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    # 如果某个块没有合法的分解路径，则用零填充。
                    blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                           device=x_in.device).type(x_in.type())]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2).reshape(batch_num, -1)
        return output

    def get_expansion_path(self, irrep_in: o3.Irreps, irrep_out_1: o3.Irreps, irrep_out_2: o3.Irreps) -> list:
        """计算所有可能的从输入 irrep 到输出 irrep 对的分解路径。

        路径存在的条件是，根据群论的耦合规则，`ir_in` 必须包含在
        `ir_out1` 和 `ir_out2` 的张量积中。

        Args:
            irrep_in (o3.Irreps): 输入的不可约表示。
            irrep_out_1 (o3.Irreps): 输出的第一个不可约表示。
            irrep_out_2 (o3.Irreps): 输出的第二个不可约表示。

        Returns:
            list: 一个包含指令的列表。每个指令是一个列表，格式为
                  `[输入irrep索引, 输出1 irrep索引, 输出2 irrep索引, 是否需要权重, 1.0, [multiplicities]]`。
        """
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    # 这是核心的群论选择规则
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'


def blockwise_2x2_concat(
    top_left: torch.Tensor,
    top_right: torch.Tensor,
    bottom_left: torch.Tensor,
    bottom_right: torch.Tensor
) -> torch.Tensor:
    """将四个张量以 2x2 的块状模式拼接成一个双倍大小的张量。

    拼接模式如下:
    [top_left | top_right]
    ----------------------
    [bottom_left | bottom_right]

    Args:
        top_left (torch.Tensor): 形状为 `[N, H, W]` 的张量。
        top_right (torch.Tensor): 与 `top_left` 形状相同的张量。
        bottom_left (torch.Tensor): 与 `top_left` 形状相同的张量。
        bottom_right (torch.Tensor): 与 `top_left` 形状相同的张量。

    Returns:
        torch.Tensor: 拼接后的张量，形状为 `[N, 2*H, 2*W]`。

    Raises:
        ValueError: 如果输入的张量形状不匹配。

    Example:
        >>> a = torch.ones(2, 3, 3)
        >>> b = torch.zeros(2, 3, 3)
        >>> result = blockwise_2x2_concat(a, b, b, a)
        >>> result.shape
        torch.Size([2, 6, 6])
    """
    # 验证输入张量的维度
    expected_shape = top_left.shape
    for i, tensor in enumerate([top_right, bottom_left, bottom_right], start=2):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"张量 {i} 的形状 {tensor.shape} 与第一个张量的形状 {expected_shape} 不匹配。"
            )

    # 首先进行水平拼接 (维度 W)
    top_row = torch.cat([top_left, top_right], dim=-1)
    bottom_row = torch.cat([bottom_left, bottom_right], dim=-1)

    # 然后进行垂直拼接 (维度 H)
    return torch.cat([top_row, bottom_row], dim=-2)


def extract_elements_above_threshold(
    condition_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    threshold: float = 0.0
) -> torch.Tensor:
    """根据条件张量中的值是否超过阈值，从源张量中提取元素。
    
    Args:
        condition_tensor (torch.Tensor): 用于与阈值比较的张量，形状为 `[N_batch, N, N]`。
        source_tensor (torch.Tensor): 从中提取值的源张量，形状为 `[N_batch, N, N]`。
        threshold (float): `condition_tensor` 中元素的最小阈值，超过该值则触发提取。
        
    Returns:
        torch.Tensor: 从 `source_tensor` 中提取的一维值张量。
        
    Raises:
        ValueError: 如果输入张量的形状不匹配。
        
    Example:
        >>> S = torch.randn(2, 3, 3)
        >>> H = torch.randn(2, 3, 3)
        >>> result = extract_elements_above_threshold(S, H, 0.5)
    """
    # 验证输入形状
    if condition_tensor.shape != source_tensor.shape:
        raise ValueError(f"形状不匹配: {condition_tensor.shape} vs {source_tensor.shape}")

    # 创建布尔掩码
    threshold_mask = condition_tensor > threshold
    
    # 提取相应的元素
    extracted_values = source_tensor[threshold_mask]
    
    return extracted_values

def upgrade_tensor_precision(tensor_dict: dict):
    """升级给定字典中特定类型张量的精度。
    
    该函数遍历字典，将 `torch.float32` 张量转换为 `torch.float64` (double)，
    并将 `torch.complex64` 张量转换为 `torch.complex128`。
    所有其他类型的张量保持不变。转换过程中会保留张量的原始设备。
    
    Args:
        tensor_dict (dict): 包含 PyTorch 张量的字典。
    
    Returns:
        None: 该函数直接在原地修改字典。
    Notes:
        对于 `float32` 类型的张量，可以使用 `.to(dtype=torch.float64)` 或 `.double()` 两种方法来将其转换为 `float64` 类型。为了与复数张量的转换方式保持一致，此函数中使用了 `.to()` 方法。
    Example:
        >>> data = {'float_tensor': torch.tensor([1.0, 2.0], dtype=torch.float32)}
        >>> upgrade_tensor_precision(data)
        >>> print(data['float_tensor'].dtype)
        torch.float64
    """
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                tensor_dict[key] = value.to(dtype=torch.float64)
            elif value.dtype == torch.complex64:
                tensor_dict[key] = value.to(dtype=torch.complex128)
