"""
/*
* @Author: Yang Zhong 
* @Date: 2021-10-08 22:38:15 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 10:54:51
*/
"""
"""该模块定义了多种输出层，用于从学习到的图表示中预测各种物理属性。

这些输出层将图神经网络（GNN）编码的原子和边的特征作为输入，
并计算出诸如力、Born有效电荷、压电张量、总能量等物理量。
每个类对应一个特定的物理属性预测任务。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, batch
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh)
from .utils import linear_bn_act
from .layers import MLPRegression, denseRegression
from typing import Callable
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class Force(nn.Module):
    """基于边特征计算原子受力的输出模块。

    该模块使用一个密集回归模型（`denseRegression`）来处理每条边的特征，
    并将其预测的标量值沿着边的方向矢量投影，从而得到力矢量。
    最后，将所有作用到同一个原子上的力矢量进行汇总（scatter-add），得到每个原子的总受力。
    """
    def __init__(self, num_edge_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """构造 Force 类的实例。

        Args:
            num_edge_features (int, optional): 输入的边特征维度。
            activation (callable, optional): 回归模型中使用的激活函数。默认为 `Softplus()`。
            use_bath_norm (bool, optional): 是否在回归模型中使用批量归一化。默认为 `True`。
            bias (bool, optional): 回归模型中的线性层是否使用偏置。默认为 `True`。
            n_h (int, optional): 回归模型中的隐藏层数量。默认为 `3`。
        """
        super(Force, self).__init__()
        self.num_edge_features = num_edge_features
        self.regression_edge = denseRegression(in_features=num_edge_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象，包含原子位置 `pos` 和边索引 `edge_index` 等信息。
            graph_representation (dict, optional): 包含图表示的字典，需要 `edge_attr` 键。

        Returns:
            dict: 包含计算出的原子受力 `force` 的字典。
        """
        edge_attr = graph_representation['edge_attr']  # 边特征张量mji
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        pos = data['pos']
        # 计算从原子 j 指向原子 i 的方向矢量 (考虑周期性边界条件)
        edge_dir = (pos[i]+nbr_shift) - pos[j] # j->i: ri - rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        # 归一化方向矢量
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji, 形状为 (N_edges, 3)
        
        # 将回归模型输出的标量力大小乘以方向矢量，得到每条边上的力
        force = self.regression_edge(edge_attr) * edge_dir  # mji * eji
        # 将所有作用到目标原子 i 上的力进行求和
        force = scatter(force, i, dim=0)
        return {'force': force} # 形状为 (N_nodes, 3)

class Force_node_vec(nn.Module):
    """基于节点标量和矢量特征计算原子受力的输出模块。

    这个模块假设原子受力可以表示为一个标量部分和一个矢量部分的乘积。
    如果节点特征维度大于1，则使用一个回归模型来计算标量部分；
    否则，直接使用节点标量特征。
    """
    def __init__(self, num_node_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """构造 Force_node_vec 类的实例。

        Args:
            num_node_features (int, optional): 输入的节点特征维度。
            activation (callable, optional): 回归模型中使用的激活函数。默认为 `Softplus()`。
            use_bath_norm (bool, optional): 是否在回归模型中使用批量归一化。默认为 `True`。
            bias (bool, optional): 回归模型中的线性层是否使用偏置。默认为 `True`。
            n_h (int, optional): 回归模型中的隐藏层数量。默认为 `3`。
        """
        super(Force_node_vec, self).__init__()
        self.num_node_features = num_node_features
        if self.num_node_features > 1:
            self.regression_node = denseRegression(in_features=num_node_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data: Data, graph_representation: dict = None) -> torch.Tensor:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典，需要 `node_attr` 和 `node_vec_attr`。

        Returns:
            torch.Tensor: 计算得到的原子受力。
        """
        node_attr = graph_representation['node_attr'] # 节点标量特征
        node_vec_attr = graph_representation['node_vec_attr'] # 节点矢量特征, 形状: (N_nodes, 1, 3)
        basis = node_vec_attr.view(-1,3) # 基矢量, 形状: (N_nodes, 3)
        
        if self.num_node_features == 1:
            # 如果节点特征只有一维，直接将其作为力的标量部分
            force = node_attr * basis
        else:
            # 否则，通过回归模型计算力的标量部分
            force = self.regression_node(node_attr) * basis
        return force     

class Born(nn.Module):
    """计算 Born 有效电荷张量的输出模块。

    该模块通过组合二体（边）和三体（角）相互作用来计算每个原子的 Born 有效电荷张量。
    二体项由边特征和边方向矢量的外积（张量积）构成。
    三体项（可选）由三元组特征和两个相关边的方向矢量外积构成。
    """
    def __init__(self, include_triplet:bool=True, num_node_features:int=None, num_edge_features:int=None, num_triplet_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3, cutoff_triplet:float=6.0, l_minus_mean: bool=False):
        """构造 Born 类的实例。

        Args:
            include_triplet (bool, optional): 是否包含三体相互作用。默认为 `True`。
            num_node_features (int, optional): 节点特征维度 (当前未使用)。
            num_edge_features (int, optional): 边特征维度。
            num_triplet_features (int, optional): 三元组特征维度。
            activation (callable, optional): 回归模型激活函数。默认为 `Softplus()`。
            use_bath_norm (bool, optional): 是否使用批量归一化。默认为 `True`。
            bias (bool, optional): 线性层是否使用偏置。默认为 `True`。
            n_h (int, optional): 隐藏层数量。默认为 `3`。
            cutoff_triplet (float, optional): 计算三体项时边的距离截断半径。默认为 `6.0`。
            l_minus_mean (bool, optional): 是否从最终的 Born 张量中减去批次均值。默认为 `False`。
        """
        super(Born, self).__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.include_triplet = include_triplet
        self.cutoff_triplet = cutoff_triplet
        self.l_minus_mean = l_minus_mean
        self.regression_edge = denseRegression(in_features=num_edge_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)
        if self.include_triplet:
            self.num_triplet_features = num_triplet_features
            self.regression_triplet = denseRegression(in_features=num_triplet_features, out_features=1, bias=bias, 
                                                    use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data: Data, graph_representation: dict = None) -> torch.Tensor:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            torch.Tensor: 每个原子的 Born 张量，形状为 (N_nodes, 9)。
        """
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        triplet_attr = graph_representation['triplet_attr']
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        # (idx_i, idx_j, idx_k, idx_kj, idx_ji)
        if self.include_triplet:
            idx_i, idx_j, idx_k, idx_kj, idx_ji = graph_representation['triplet_index']
        pos = data['pos']
        edge_dir = (pos[i]+nbr_shift) - pos[j] # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji 形状(N_edges, 3)
        
        # --- 对称部分 (二体项) ---
        # 计算方向矢量的二阶张量积 (dyad product)
        dyad_ji_ji = edge_dir.unsqueeze(-1)@edge_dir.unsqueeze(1) # e_ji ⊗ e_ji
        dyad_ji_ji = dyad_ji_ji.view(-1, 9)
        temp_sym = self.regression_edge(edge_attr) * dyad_ji_ji  # m_ji * (e_ji ⊗ e_ji)
        # 将贡献累加到中心原子 i
        born_tensor_sym = scatter(temp_sym, i, dim=0)

        if self.include_triplet:
            # --- 交叉部分 (三体项) ---
            # 计算 e_kj 和 e_ji 的二阶张量积
            dyad_kj_ji = edge_dir[idx_kj].unsqueeze(-1)@edge_dir[idx_ji].unsqueeze(1) # e_kj ⊗ e_ji
            dyad_kj_ji = dyad_kj_ji.view(-1,9)
            # 应用距离截断
            mask = (edge_length[idx_kj] < self.cutoff_triplet) & (edge_length[idx_ji] < self.cutoff_triplet)
            mask = mask.float().unsqueeze(-1)
            temp_cross = self.regression_triplet(triplet_attr) * mask * dyad_kj_ji  # m_kji * (e_kj ⊗ e_ji)
            # 将贡献累加到中心原子 j
            born_tensor_cross = scatter(temp_cross, idx_j, dim=0)
            born_tensor = born_tensor_sym + born_tensor_cross
        else:
            born_tensor = born_tensor_sym
        
        # 可选：减去批次均值以满足某些约束
        if self.l_minus_mean:
            born_tensor = born_tensor - global_mean_pool(born_tensor, data['batch'])[data['batch']]
        return born_tensor # 形状 (N_nodes, 9)

class Born_node_vec(nn.Module):
    """基于节点标量和双矢量特征计算 Born 张量的输出模块。

    此模块假设 Born 张量可以由节点上的两个基矢量（`node_vec_attr`）的
    外积（张量积）与一个标量系数相乘得到。
    """
    def __init__(self, num_node_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """构造 Born_node_vec 类的实例。

        Args:
            num_node_features (int, optional): 输入的节点特征维度。
            activation (callable, optional): 回归模型中使用的激活函数。默认为 `Softplus()`。
            use_bath_norm (bool, optional): 是否在回归模型中使用批量归一化。默认为 `True`。
            bias (bool, optional): 回归模型中的线性层是否使用偏置。默认为 `True`。
            n_h (int, optional): 回归模型中的隐藏层数量。默认为 `3`。
        """
        super(Born_node_vec, self).__init__()
        self.num_node_features = num_node_features
        if self.num_node_features > 1:
            self.regression_node = denseRegression(in_features=num_node_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data: Data, graph_representation: dict = None) -> torch.Tensor:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典，需要 `node_attr` 和 `node_vec_attr`。

        Returns:
            torch.Tensor: 计算得到的 Born 张量，形状为 (N_nodes, 9)。
        """
        node_attr = graph_representation['node_attr'] # 节点标量特征
        node_vec_attr = graph_representation['node_vec_attr'] # 形状: (N_nodes, 2, 3)
        # 计算两个基矢量的外积
        basis = node_vec_attr[:,0,:].unsqueeze(-1)@node_vec_attr[:,1,:].unsqueeze(1) # 形状: (N_nodes, 3, 3)
        basis = basis.view(-1,9) # 形状: (N_nodes, 9)
        
        if self.num_node_features == 1:
            born = node_attr*basis
        else:
            born = self.regression_node(node_attr)*basis
        return born      

"""
class piezoelectric(nn.Module):
    def __init__(self, num_node_features: int = None, num_edge_features: int = None, activation: callable = Softplus(),
                 use_bath_norm: bool = True, bias: bool = True, n_h: int = 3):
        super(piezoelectric, self).__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.regression_edge = denseRegression(in_features=num_edge_features, out_features=1, bias=bias,
                                               use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data, graph_representation: dict = None):
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        pos = data['pos']
        edge_dir = (pos[i]+nbr_shift) - pos[j]  # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji Shape(Nedges, 3)

        dyad_ji_ji_ji = torch.einsum(
            'ij,ik,il->ijkl', [edge_dir, edge_dir, edge_dir])  # Shape(Nedges, 3, 3, 3)
        dyad_ji_ji_ji = dyad_ji_ji_ji.view(-1, 27)
        temp_sym = self.regression_edge(
            edge_attr) * dyad_ji_ji_ji  # mji*eji@eji@eji
        pz_tensor_atom = scatter(temp_sym, i, dim=0)

        pz_tensor = global_mean_pool(pz_tensor_atom, data['batch'])
        return pz_tensor  # shape (N, 27)
"""

class piezoelectric(nn.Module):
    """计算压电张量的输出模块。

    该模块通过组合二体和三体相互作用来计算压电张量。压电张量是三阶张量，
    描述了材料在应变下产生极化的能力。计算涉及三阶的矢量张量积。
    """
    def __init__(self, include_triplet: bool = True, num_node_features: int = None, num_edge_features: int = None, num_triplet_features: int = None, activation: callable = Softplus(),
                 use_bath_norm: bool = True, bias: bool = True, n_h: int = 3, cutoff_triplet: float = 6.0):
        """构造 piezoelectric 类的实例。

        Args:
            include_triplet (bool, optional): 是否包含三体相互作用。默认为 `True`。
            num_node_features (int, optional): 节点特征维度 (当前未使用)。
            num_edge_features (int, optional): 边特征维度。
            num_triplet_features (int, optional): 三元组特征维度。
            activation (callable, optional): 回归模型激活函数。默认为 `Softplus()`。
            use_bath_norm (bool, optional): 是否使用批量归一化。默认为 `True`。
            bias (bool, optional): 线性层是否使用偏置。默认为 `True`。
            n_h (int, optional): 隐藏层数量。默认为 `3`。
            cutoff_triplet (float, optional): 计算三体项时边的距离截断半径。默认为 `6.0`。
        """
        super(piezoelectric, self).__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.include_triplet = include_triplet
        self.cutoff_triplet = cutoff_triplet
        self.regression_edge = denseRegression(in_features=num_edge_features, out_features=1, bias=bias,
                                               use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)
        if self.include_triplet:
            self.num_triplet_features = num_triplet_features
            self.regression_triplet = denseRegression(in_features=num_triplet_features, out_features=1, bias=bias,
                                                      use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            dict: 包含计算出的压电张量 `piezoelectric` 的字典。
        """
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # m_ji
        triplet_attr = graph_representation['triplet_attr']
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        # (idx_i, idx_j, idx_k, idx_kj, idx_ji)
        if self.include_triplet:
            idx_i, idx_j, idx_k, idx_kj, idx_ji = graph_representation['triplet_index']
        pos = data['pos']
        edge_dir = (pos[i]+nbr_shift) - pos[j]  # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # e_ji 形状(N_edges, 3)

        # --- 对称部分 (二体项) ---
        # 计算方向矢量的三阶张量积: e_ji ⊗ e_ji ⊗ e_ji
        dyad_ji_ji_ji = torch.einsum(
            'ij,ik,il->ijkl', [edge_dir, edge_dir, edge_dir]) # 形状 (N_edges, 3, 3, 3)
        dyad_ji_ji_ji = dyad_ji_ji_ji.view(-1, 27)

        temp_sym = self.regression_edge(edge_attr) * dyad_ji_ji_ji # m_ji * (e_ji ⊗ e_ji ⊗ e_ji)
        # 将贡献累加到中心原子 i
        pzt_sym = scatter(temp_sym, i, dim=0)

        if self.include_triplet:
            # --- 交叉部分 (三体项) ---
            # 计算三阶张量积: e_kj ⊗ e_ji ⊗ e_ji
            dyad_kj_ji_ji = torch.einsum(
            'ij,ik,il->ijkl', [edge_dir[idx_kj], edge_dir[idx_ji], edge_dir[idx_ji]]) # 形状 (N_triplet, 3, 3, 3)
            dyad_kj_ji_ji = dyad_kj_ji_ji.view(-1, 27)
            # 应用距离截断
            mask = (edge_length[idx_kj] < self.cutoff_triplet) & (
                edge_length[idx_ji] < self.cutoff_triplet)
            mask = mask.float().unsqueeze(-1)
            temp_cross = self.regression_triplet(
                triplet_attr) * mask * dyad_kj_ji_ji  # m_kji * (e_kj ⊗ e_ji ⊗ e_ji)
            # 将贡献累加到中心原子 j
            pzt_cross = scatter(temp_cross, idx_j, dim=0)
            pzt = pzt_sym + pzt_cross
        else:
            pzt = pzt_sym
        # 对一个晶格内的所有原子贡献进行平均池化，得到晶体的压电张量
        pzt = global_mean_pool(pzt, data['batch'])
        return {'piezoelectric': pzt}  # 形状 (N_batch, 27)

class trivial_scalar(nn.Module):
    """一个简单的标量预测模块。

    该模块直接对节点特征进行全局池化（平均、求和或最大值）来预测一个标量属性。
    它不包含任何可学习的参数，主要用于基线模型或简单任务。
    """
    def __init__(self, aggr:str = 'mean'):
        """构造 trivial_scalar 类的实例。

        Args:
            aggr (str, optional): 池化操作的类型 ('mean', 'sum'/'add', 'max')。默认为 'mean'。
        """
        super(trivial_scalar, self).__init__()
        self.aggr = aggr

    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典，需要 `node_attr`。

        Returns:
            dict: 包含预测标量 `scalar` 的字典。
        """
        if self.aggr == 'mean':
            x = global_mean_pool(graph_representation['node_attr'], data['batch'])
        elif self.aggr == 'sum' or self.aggr == 'add':
            x = global_add_pool(graph_representation['node_attr'], data['batch'])
        elif self.aggr == 'max':
            x = global_max_pool(graph_representation['node_attr'], data['batch'])
        else:
            raise ValueError(f"不支持的聚合类型: {self.aggr}")
        return {'scalar': x.view(-1)}

class scalar(nn.Module):
    """一个更复杂的标量预测模块，包含一个 MLP。

    该模块首先对节点特征进行全局池化，然后将得到的图级别特征
    送入一个多层感知机（MLP）进行回归或分类。
    """
    def __init__(self, aggr:str = 'mean', classification:bool=False, num_node_features:int=None, n_h:int=3, activation:callable=nn.Softplus()):
        """构造 scalar 类的实例。

        Args:
            aggr (str, optional): 池化操作类型。默认为 'mean'。
            classification (bool, optional): 是否为分类任务。默认为 `False` (回归任务)。
            num_node_features (int, optional): 节点特征维度。
            n_h (int, optional): MLP 的隐藏层数量。默认为 `3`。
            activation (callable, optional): MLP 中使用的激活函数。默认为 `nn.Softplus()`。
        """
        super().__init__()
        self.aggr = aggr
        self.classification = classification
        self.activation = activation
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(num_node_features, num_node_features)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([self.activation
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(num_node_features, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(num_node_features, 1)

    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典，需要 `node_attr`。

        Returns:
            dict: 包含预测标量 `scalar` 的字典。
        """
        # 步骤 1: 全局池化
        if self.aggr.lower() == 'mean':
            crys_fea = global_mean_pool(graph_representation['node_attr'], data['batch'])
        elif self.aggr.lower() == 'sum':
            crys_fea = global_add_pool(graph_representation['node_attr'], data['batch'])
        elif self.aggr.lower() == 'max':
            # 对于 'max' 池化，MLP 在池化之前应用
            crys_fea = graph_representation['node_attr']
        else:
            raise ValueError(f"不支持的聚合类型: {self.aggr}")
        
        if self.classification:
            crys_fea = self.dropout(crys_fea)

        # 步骤 2: MLP
        try:
            fcs = self.fcs
            softpluses = self.softpluses
            for fc, softplus in zip(fcs, softpluses):
                crys_fea = softplus(fc(crys_fea))
        except AttributeError:
            pass

        # 步骤 3: 输出层
        out = self.fc_out(crys_fea)
        if self.aggr.lower() == 'max':
            # 在 MLP 之后应用最大池化
            out = global_max_pool(out, data['batch'])
        
        if self.classification:
            out = self.logsoftmax(out)
        else:
            out = out.view(-1)
        return {'scalar': out}

class crystal_tensor(nn.Module):
    """计算晶体级别张量的输出模块。

    该模块包裹了 `Born` 模块，用于计算原子级别的张量。
    然后，它可以选择直接返回原子级别的张量，或者通过平均池化
    得到晶体级别的张量。
    """
    def __init__(self, l_pred_atomwise_tensor: bool=True, include_triplet:bool=True, num_node_features:int=None, num_edge_features:int=None, num_triplet_features:int=None, activation:callable=Softplus(),
                 use_bath_norm: bool = True, bias: bool = True, n_h: int = 3, cutoff_triplet: float = 6.0, l_minus_mean: bool = False):
        """构造 crystal_tensor 类的实例。

        Args:
            l_pred_atomwise_tensor (bool, optional): 如果为 True, 返回原子级张量。否则返回晶体级张量。默认为 `True`。
            其他参数 (All other args): 传递给 `Born` 模块的构造函数。
        """
        super(crystal_tensor, self).__init__()
        self.l_pred_atomwise_tensor = l_pred_atomwise_tensor
        self.atom_tensor_output = Born(include_triplet, num_node_features, num_edge_features, num_triplet_features, activation, use_bath_norm, bias, n_h, cutoff_triplet, l_minus_mean)
    
    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            dict: 包含 `atomic_tensor` 或 `crystal_tensor` 的字典。
        """
        atom_tensors = self.atom_tensor_output(data, graph_representation)
        if self.l_pred_atomwise_tensor:
            return {'atomic_tensor': atom_tensors}
        else:
            output = global_mean_pool(atom_tensors, data['batch'])
            return {'crystal_tensor': output}

class total_energy_and_atomic_forces(nn.Module):
    """同时预测总能量和原子受力的模块。

    能量被计算为所有原子能量贡献的总和。力是通过对总能量关于原子位置
    求负梯度得到的（依据海尔曼-费曼定理）。
    """
    def __init__(self, num_node_features:int=None, n_h:int=3, activation:callable=nn.Softplus(), derivative:bool=False):
        """构造 total_energy_and_atomic_forces 类的实例。

        Args:
            num_node_features (int, optional): 节点特征维度。
            n_h (int, optional): 回归模型隐藏层数量。默认为 `3`。
            activation (callable, optional): 激活函数。默认为 `nn.Softplus()`。
            derivative (bool, optional): 是否计算力的解析导数。默认为 `False`。
        """
        super().__init__()
        self.derivative = derivative # 在模型中设置 data['pos'] 的梯度
        #self.energy = scalar(aggr='sum', classification=False, num_node_features=num_node_features, n_h=n_h, activation=activation)

        self.atom_regression = denseRegression(in_features=num_node_features, out_features=1, bias=True,
                                               use_batch_norm=False, activation=activation, n_h=n_h)
    
    def forward(self, data: Data, graph_representation: dict = None) -> dict:
        """定义前向传播逻辑。

        Args:
            data (Data): PyG 图数据对象。
            graph_representation (dict, optional): 包含图表示的字典，需要 `node_attr`。

        Returns:
            dict: 包含 `forces` 和 `total_energy` 的字典。
        """
        #energy = self.energy(data, graph_representation)['scalar']
        # 计算每个原子的能量贡献
        atomic_energy = self.atom_regression(graph_representation['node_attr'])
        # 对一个晶格内的所有原子能量求和，得到总能量
        energy = global_add_pool(atomic_energy, data['batch']).reshape(-1)
        if self.derivative:
            # 通过自动微分计算力
            forces = -torch.autograd.grad(energy, data['pos'],
                                        grad_outputs=torch.ones_like(energy),
                                        create_graph=self.training)[0]
        else:
            forces = None
        return {'forces':forces, 'total_energy':energy}

class EPC_output:
    """计算电子-声子耦合 (Electron-Phonon Coupling, EPC) 矩阵的模块。

    这个类不是一个标准的 `nn.Module`，而是一个可调用的计算流程封装。
    它接收一个图表示模型 (representation) 和一个哈密顿量输出模型 (output)，
    通过自动微分计算哈密顿量对原子位移的导数，并结合波函数，最终计算出 EPC 矩阵。
    EPC 矩阵描述了电子与晶格振动（声子）之间的相互作用强度，是计算材料
    电导率、超导电性等关键物理量的核心。

    计算的核心是利用 `torch.autograd.functional.jacobian` 来获得雅可比矩阵
    `nabla_HK` (∂H(k)/∂R)，即 k 点哈密顿量相对于原子坐标 R 的梯度。

    .. note::
       此类要求批处理中所有晶体的原子数量必须相等。

    Attributes:
        representation (Callable): GNN 模型，用于从原子结构数据计算图表示。
        output (Callable): 基于图表示计算哈密顿量 `HK`、重叠矩阵 `SK` 等物理量的模型。
                           该模型还需提供 `basis_def` 属性以定义原子轨道基组。
        band_win_min (int): 计算 EPC 矩阵时所考虑的能带窗口的起始索引 (从 1 开始)。
        band_win_max (int): 计算 EPC 矩阵时所考虑的能带窗口的结束索引。
    """
    def __init__(self, representation:Callable=None, output:Callable=None, band_win_min:int=None, band_win_max:int=None):
        """构造 EPC_output 类的实例。

        Args:
            representation (Callable, optional): GNN 模型实例。
            output (Callable, optional): 哈密顿量输出模型实例。
            band_win_min (int, optional): 能带窗口起始索引。
            band_win_max (int, optional): 能带窗口结束索引。
        """
        self.representation = representation
        self.output = output
        self.band_win_min = band_win_min
        self.band_win_max = band_win_max        
        
    def  __call__(self, data):
        """使类的实例可被调用。"""
        out = self.forward(data)
        return out

    def forward(self, data):
        """执行 EPC 矩阵的前向计算。

        Args:
            data (Data): PyG 图数据对象，包含原子坐标 `pos`、原子类型 `z`、晶胞 `cell` 等信息。

        Returns:
            dict: 一个包含哈密顿量 `hamiltonian` 和 EPC 矩阵 `epc_mat` 的字典。
                  `epc_mat` 的形状为 [N_batch, n_k, n_bands, n_bands, n_atoms, 3]。
        """
        Nbatch = data['cell'].shape[0]
        # 约束：批处理中每个晶体的原子数必须相同
        natoms = int(len(data['z'])/Nbatch)
        
        # --- 步骤 1: 构建轨道到原子的索引映射 (orb2atom_idx) ---
        # 这个索引将每个原子轨道映射到其所属原子的索引。
        atomic_nums = data['z'].view(-1, natoms) # shape: [Nbatch, natoms]
        orb2atom_idx  = []
        for ib in range(Nbatch):
            # 根据每个原子的类型，从基组定义中获取其轨道数量
            repeats = []
            for ia in range(natoms):
                repeats.append(len(self.output.basis_def[atomic_nums[ib][ia].item()]))
            repeats = torch.LongTensor(repeats)
            # 例如，如果前两个原子各有3个轨道，则映射为 [0, 0, 0, 1, 1, 1, ...]
            orb2atom_idx.append(torch.repeat_interleave(torch.arange(natoms), repeats, dim=0).type_as(atomic_nums))
        
        # --- 步骤 2: 计算哈密顿量关于原子坐标的雅可比矩阵 (nabla_HK) ---
        # 定义一个包装函数，使其仅接受原子坐标 `pos` 作为输入，并返回哈密顿量 `HK`。
        # 这是 `torch.autograd.functional.jacobian` 所要求的函数签名。
        # 使用 `nonlocal` 关键字来捕获和更新外部作用域中的变量。
        HK, SK, wavefunction, hamiltonian, dSK = None,None,None,None,None
        
        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            nonlocal data, HK, SK, wavefunction, hamiltonian, dSK
            data['pos'] = pos
            graph_representation = self.representation(data)
            out = self.output(data, graph_representation)
            # 在前向传播过程中，保存所有需要的中间结果
            HK, SK, wavefunction, hamiltonian, dSK = out['HK'], out['SK'], out['wavefunction'], out['hamiltonian'], out['dSK']
            return HK
        
        # `detect_anomaly` 用于调试，可以定位在反向传播中导致 NaN 的操作
        with torch.autograd.detect_anomaly():          
            # 核心计算：自动微分得到雅可比矩阵 d(HK)/d(pos)
            # nabla_HK 形状: [Nbatch, num_k, norbs, norbs, natoms, 3]
            nabla_HK = torch.autograd.functional.jacobian(func=wrapper, inputs=data['pos'], create_graph=False, vectorize=False)

        # --- 步骤 3: 计算 EPC 矩阵元素 ---
        # EPC 矩阵 g_{mn} = <ψ_m| dH/dR |ψ_n>
        # 在非正交基下，公式会更复杂，需要考虑重叠矩阵 S 的导数 dS/dR。
        norbs = HK.shape[-1]
        m = torch.arange(0, norbs)
           
        # 根据指定的窗口选择能带（波函数）
        wavefunction = wavefunction[:,:,self.band_win_min-1:self.band_win_max,:]
        wavefunction_conj = torch.conj(wavefunction)
        
        # --- 方法 1: 使用 `einsum` (内存消耗大，但可能更快) ---
        # 这个方法被注释掉了，因为它构建了巨大的中间张量，容易导致内存溢出。
        # method 1 for faster speed
        """
        epc_mat = []
        for idx in range(Nbatch):        
            #nabla_SK1 = nabla_SK[idx,:,:,m,orb2atom_idx[idx][m],:].type_as(HK) # shape:[num_k, norbs, norbs, 3]
            #nabla_SK2 = nabla_SK[idx,:,n,:,orb2atom_idx[idx][n],:].type_as(HK) # shape:[norbs, num_k, norbs, 3]
            #nabla_SK2 = torch.swapaxes(nabla_SK2, axis0=0, axis1=1) # shape:[num_k, norbs, norbs, 3]
            
            nabla_SK1 = torch.zeros_like(nabla_HK, dtype=HK.dtype)
            nabla_SK1[idx,:,:,m,orb2atom_idx[idx][m],:] = nabla_HK[idx,:,:,m,orb2atom_idx[idx][m],:].type_as(HK)
            
            nabla_SK2 = torch.zeros_like(nabla_HK, dtype=HK.dtype)
            nabla_SK2[idx,:,n,:,orb2atom_idx[idx][n],:] = nabla_HK[idx,:,n,:,orb2atom_idx[idx][n],:].type_as(HK)
            
            sum1 = 'abd, ace, afghi, adf, age -> abchi'
            part1 = torch.einsum(sum1, torch.conj(wavefunction[idx]), wavefunction[idx], nabla_HK[idx], SK[idx], SK[idx])
            
            sum2 = 'abd, ace, afg, adfhi, age -> abchi'
            part2 = torch.einsum(sum2, torch.conj(wavefunction[idx]), wavefunction[idx], HK[idx], nabla_SK1[idx], SK[idx])
            
            sum3 = 'abd, ace, afg, adf, agehi -> abchi'
            part3 = torch.einsum(sum3, torch.conj(wavefunction[idx]), wavefunction[idx], HK[idx], SK[idx], nabla_SK2[idx])
            
            epc_mat.append(part1 + part2 + part3)
        
        epc_mat = torch.cat(epc_mat, dim=0)
        """
        # --- 方法 2: 使用循环和 `einsum` (内存优化) ---
        # 这是当前使用的方法。通过显式循环遍历能带和轨道，避免一次性构造
        # 过大的张量，以空间换时间。
        epc_mat_batch = []
        for idx in range(Nbatch): 
            epc_mat = []     
                  
            # 构造 dS/dR 张量，注意这里 dSK 的原始形状可能与 nabla_HK 不同
            nabla_SK = torch.zeros_like(nabla_HK, dtype=HK.dtype)
            nabla_SK[idx,:,:,m,orb2atom_idx[idx][m],:] = dSK[idx]
            
            # 循环遍历所有能带对 (b, c)
            for b in range(wavefunction.shape[-2]):
                for c in range(wavefunction.shape[-2]):
                    temp_sum = []
                    # 循环遍历所有轨道对 (d, e)
                    for d in range(norbs):
                        for e in range(norbs):
                            # 以下是 EPC 矩阵在非正交基下的三个组成部分
                            # sum1: <ψ_b| (dH/dR) |ψ_c> 项
                            sum1 = 'a, a, afghi, af, ag -> ahi'
                            part1 = torch.einsum(sum1, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], nabla_HK[idx], SK[idx,:,d,:], SK[idx,:,:,e])
            
                            # sum2: <ψ_b| H (dS/dR) |ψ_c> 项 (部分)
                            sum2 = 'a, a, afg, afhi, ag -> ahi'
                            part2 = torch.einsum(sum2, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], HK[idx], nabla_SK[idx,:,d,:,:,:], SK[idx,:,:,e])
            
                            # sum3: <ψ_b| H (dS/dR) |ψ_c> 项 (另一部分)
                            sum3 = 'a, a, afg, af, aghi -> ahi'
                            part3 = torch.einsum(sum3, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], HK[idx], SK[idx,:,d,:], nabla_SK[idx,:,e,:,:,:])
            
                            temp_sum.append(part1 + part2 + part3)
                    # 对轨道 d 和 e 的贡献求和       
                    temp_sum = torch.sum(torch.stack(temp_sum, dim=0), dim=0)
                    epc_mat.append(temp_sum) # 形状: [num_k, natoms, 3]
                    
            # 重新组织形状以匹配 [n_k, n_bands_b, n_bands_c, n_atoms, 3]
            epc_mat = torch.stack(epc_mat, dim=1).reshape(-1, wavefunction.shape[-2], wavefunction.shape[-2], natoms, 3)
            epc_mat_batch.append(epc_mat)
        # 将批次结果堆叠起来
        epc_mat = torch.stack(epc_mat_batch, dim=0) # 最终形状: [Nbatch, num_k, n_bands, n_bands, natoms, 3]
        
        
        return {'hamiltonian':hamiltonian, 'epc_mat': epc_mat}

        