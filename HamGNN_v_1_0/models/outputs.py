"""
/*
* @Author: Yang Zhong 
* @Date: 2021-10-08 22:38:15 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 10:54:51
*/
"""
"""该模块定义了用于预测各种物理属性的输出层。

这些模块接收来自表示层网络的图表示（节点、边、三元组特征），
并将其转换为具体的物理量，如力、Born有效电荷、压电张量、总能量等。
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
    """基于边特征计算原子受力的模块。"""
    def __init__(self, num_edge_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """
        Args:
            num_edge_features (int, optional): 边特征的维度。
            activation (callable, optional): 激活函数。默认为 Softplus()。
            use_bath_norm (bool, optional): 是否使用批归一化。默认为 True。
            bias (bool, optional): 线性层是否使用偏置。默认为 True。
            n_h (int, optional): 回归网络中的隐藏层数量。默认为 3。
        """
        super(Force, self).__init__()
        self.num_edge_features = num_edge_features
        self.regression_edge = denseRegression(in_features=num_edge_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典，需要 'edge_attr'。

        Returns:
            dict: 包含预测力 'force' 的字典。
        """
        edge_attr = graph_representation['edge_attr']  # mji
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        pos = data['pos']
        edge_dir = (pos[i]+nbr_shift) - pos[j] # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji Shape(Nedges, 3)
        
        # 回归出一个标量，然后乘以方向向量得到力的矢量
        force = self.regression_edge(edge_attr) * edge_dir  # mji*eji
        # 将所有指向原子i的力加和
        force = scatter(force, i, dim=0)
        return {'force': force} # shape (N_nodes, 3)

class Force_node_vec(nn.Module):
    """基于节点标量和矢量特征计算原子受力的模块。"""
    def __init__(self, num_node_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """
        Args:
            num_node_features (int, optional): 节点标量特征的维度。
            activation (callable, optional): 激活函数。默认为 Softplus()。
            use_bath_norm (bool, optional): 是否使用批归一化。默认为 True。
            bias (bool, optional): 线性层是否使用偏置。默认为 True。
            n_h (int, optional): 回归网络中的隐藏层数量。默认为 3。
        """
        super(Force_node_vec, self).__init__()
        self.num_node_features = num_node_features
        if self.num_node_features > 1:
            self.regression_node = denseRegression(in_features=num_node_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典，需要 'node_attr' 和 'node_vec_attr'。

        Returns:
            torch.Tensor: 预测的力。
        """
        node_attr = graph_representation['node_attr']
        node_vec_attr = graph_representation['node_vec_attr'] # shape: (N_nodes, 1, 3)
        basis = node_vec_attr.view(-1,3) # shape: (N_nodes, 3)
        
        if self.num_node_features == 1:
            force = node_attr*basis
        else:
            force = self.regression_node(node_attr)*basis
        return force      

class Born(nn.Module):
    """计算 Born 有效电荷张量的模块。"""
    def __init__(self, include_triplet:bool=True, num_node_features:int=None, num_edge_features:int=None, num_triplet_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3, cutoff_triplet:float=6.0, l_minus_mean: bool=False):
        """
        Args:
            include_triplet (bool, optional): 是否包含三元组信息。默认为 True。
            num_node_features (int, optional): 节点特征维度。
            num_edge_features (int, optional): 边特征维度。
            num_triplet_features (int, optional): 三元组特征维度。
            activation (callable, optional): 激活函数。默认为 Softplus()。
            use_bath_norm (bool, optional): 是否使用批归一化。默认为 True。
            bias (bool, optional): 线性层是否使用偏置。默认为 True。
            n_h (int, optional): 回归网络中的隐藏层数量。默认为 3。
            cutoff_triplet (float, optional): 三元组计算的截断半径。默认为 6.0。
            l_minus_mean (bool, optional): 是否减去均值。默认为 False。
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

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            torch.Tensor: 预测的 Born 张量，形状为 (N_nodes, 9)。
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
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji Shape(Nedges, 3)
        
        # 计算二体项贡献
        dyad_ji_ji = edge_dir.unsqueeze(-1)@edge_dir.unsqueeze(1)
        dyad_ji_ji = dyad_ji_ji.view(-1, 9)
        temp_sym = self.regression_edge(edge_attr) * dyad_ji_ji  # mji*eji@eji
        born_tensor_sym = scatter(temp_sym, i, dim=0)

        # 计算三体项贡献
        if self.include_triplet:
            dyad_kj_ji = edge_dir[idx_kj].unsqueeze(-1)@edge_dir[idx_ji].unsqueeze(1)
            dyad_kj_ji = dyad_kj_ji.view(-1,9)
            mask = (edge_length[idx_kj] < self.cutoff_triplet) & (edge_length[idx_ji] < self.cutoff_triplet)
            mask = mask.float().unsqueeze(-1)
            temp_cross = self.regression_triplet(triplet_attr) * mask * dyad_kj_ji  # mkji*ekj@eji
            born_tensor_cross = scatter(temp_cross, idx_j, dim=0)
            born_tensor = born_tensor_sym + born_tensor_cross
        else:
            born_tensor = born_tensor_sym
        if self.l_minus_mean:
            born_tensor = born_tensor - global_mean_pool(born_tensor, data['batch'])[data['batch']]
        return born_tensor # shape (N_nodes, 9)

class Born_node_vec(nn.Module):
    """基于节点标量和矢量特征计算 Born 张量的模块。"""
    def __init__(self, num_node_features:int=None, activation:callable=Softplus(),
                    use_bath_norm:bool=True, bias:bool=True, n_h:int=3):
        """
        Args:
            num_node_features (int, optional): 节点标量特征的维度。
            activation (callable, optional): 激活函数。默认为 Softplus()。
            use_bath_norm (bool, optional): 是否使用批归一化。默认为 True。
            bias (bool, optional): 线性层是否使用偏置。默认为 True。
            n_h (int, optional): 回归网络中的隐藏层数量。默认为 3。
        """
        super(Born_node_vec, self).__init__()
        self.num_node_features = num_node_features
        if self.num_node_features > 1:
            self.regression_node = denseRegression(in_features=num_node_features, out_features=1, bias=bias, 
                                                use_batch_norm=use_bath_norm, activation=activation, n_h=n_h)

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典，需要 'node_attr' 和 'node_vec_attr'。

        Returns:
            torch.Tensor: 预测的 Born 张量。
        """
        node_attr = graph_representation['node_attr']
        node_vec_attr = graph_representation['node_vec_attr'] # shape: (N_nodes, 2, 3)
        basis = node_vec_attr[:,0,:].unsqueeze(-1)@node_vec_attr[:,1,:].unsqueeze(1) # shape: (N_nodes, 3, 3)
        basis = basis.view(-1,9) # shape: (N_nodes, 9)
        
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
    """计算压电张量的模块。"""
    def __init__(self, include_triplet: bool = True, num_node_features: int = None, num_edge_features: int = None, num_triplet_features: int = None, activation: callable = Softplus(),
                 use_bath_norm: bool = True, bias: bool = True, n_h: int = 3, cutoff_triplet: float = 6.0):
        """
        Args:
            include_triplet (bool, optional): 是否包含三元组信息。默认为 True。
            num_node_features (int, optional): 节点特征维度。
            num_edge_features (int, optional): 边特征维度。
            num_triplet_features (int, optional): 三元组特征维度。
            activation (callable, optional): 激活函数。默认为 Softplus()。
            use_bath_norm (bool, optional): 是否使用批归一化。默认为 True。
            bias (bool, optional): 线性层是否使用偏置。默认为 True。
            n_h (int, optional): 回归网络中的隐藏层数量。默认为 3。
            cutoff_triplet (float, optional): 三元组计算的截断半径。默认为 6.0。
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

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            dict: 包含预测压电张量 'piezoelectric' 的字典。
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
        edge_dir = (pos[i]+nbr_shift) - pos[j]  # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        edge_dir = edge_dir/edge_length.unsqueeze(-1)  # eji Shape(Nedges, 3)

        # 计算二体项贡献
        dyad_ji_ji_ji = torch.einsum(
            'ij,ik,il->ijkl', [edge_dir, edge_dir, edge_dir])  # Shape(Nedges, 3, 3, 3)
        dyad_ji_ji_ji = dyad_ji_ji_ji.view(-1, 27)

        temp_sym = self.regression_edge(edge_attr) * dyad_ji_ji_ji # mji*eji@eji@eji
        pzt_sym = scatter(temp_sym, i, dim=0)

        # 计算三体项贡献
        if self.include_triplet:
            dyad_kj_ji_ji = torch.einsum(
            'ij,ik,il->ijkl', [edge_dir[idx_kj], edge_dir[idx_ji], edge_dir[idx_ji]])  # Shape(Ntriplet, 3, 3, 3)
            dyad_kj_ji_ji = dyad_kj_ji_ji.view(-1, 27)

            mask = (edge_length[idx_kj] < self.cutoff_triplet) & (
                edge_length[idx_ji] < self.cutoff_triplet)
            mask = mask.float().unsqueeze(-1)
            temp_cross = self.regression_triplet(
                triplet_attr) * mask * dyad_kj_ji_ji  # mkji*ekj@eji@eji
            pzt_cross = scatter(temp_cross, idx_j, dim=0)
            pzt = pzt_sym + pzt_cross
        else:
            pzt = pzt_sym
        # 对原子级别的张量进行全局池化得到晶体级别的张量
        pzt = global_mean_pool(pzt, data['batch'])
        return {'piezoelectric': pzt}  # shape (N, 27)

class trivial_scalar(nn.Module):
    """一个简单的标量预测模块，仅对节点特征进行池化。"""
    def __init__(self, aggr:str = 'mean'):
        """
        Args:
            aggr (str, optional): 池化操作类型 ('mean', 'sum', 'add', 'max')。默认为 'mean'。
        """
        super(trivial_scalar, self).__init__()
        self.aggr = aggr

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含 'node_attr' 的字典。

        Returns:
            dict: 包含预测标量 'scalar' 的字典。
        """
        if self.aggr == 'mean':
            x = global_mean_pool(graph_representation.node_attr, data['batch'])
        elif self.aggr == 'sum' or 'add':
            x = global_add_pool(graph_representation.node_attr, data['batch'])
        elif self.aggr == 'max':
            x = global_max_pool(graph_representation.node_attr, data['batch'])
        else:
            print(f"Wrong parameter 'aggr': {self.aggr}")
            exit()
        return {'scalar': x.view(-1)}

class scalar(nn.Module):
    """一个更复杂的标量预测模块，包含一个 MLP 回归层。"""
    def __init__(self, aggr:str = 'mean', classification:bool=False, num_node_features:int=None, n_h:int=3, activation:callable=nn.Softplus()):
        """
        Args:
            aggr (str, optional): 池化操作类型。默认为 'mean'。
            classification (bool, optional): 是否为分类任务。默认为 False。
            num_node_features (int, optional): 节点特征维度。
            n_h (int, optional): MLP 的隐藏层数量。默认为 3。
            activation (callable, optional): 激活函数。默认为 nn.Softplus()。
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

    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含 'node_attr' 的字典。

        Returns:
            dict: 包含预测标量 'scalar' 的字典。
        """
        # 池化操作
        if self.aggr.lower() == 'mean':
            crys_fea = global_mean_pool(graph_representation.node_attr, data['batch'])
        elif self.aggr.lower() == 'sum':
            crys_fea = global_add_pool(graph_representation.node_attr, data['batch'])
        elif self.aggr.lower() == 'max':
            crys_fea = graph_representation.node_attr
        else:
            print(f"Wrong parameter 'aggr': {self.aggr}")
            exit()
        if self.classification:
            crys_fea = self.dropout(crys_fea)

        # MLP 回归
        try:
            fcs = self.fcs
            softpluses = self.softpluses
            for fc, softplus in zip(fcs, softpluses):
                crys_fea = softplus(fc(crys_fea))
        except AttributeError:
            pass

        out = self.fc_out(crys_fea)
        if self.aggr.lower() == 'max':
            out = global_max_pool(out, data['batch'])
        if self.classification:
            out = self.logsoftmax(out)
        else:
            out = out.view(-1)
        return {'scalar': out}

class crystal_tensor(nn.Module):
    """预测晶体级别张量的模块。"""
    def __init__(self, l_pred_atomwise_tensor: bool=True, include_triplet:bool=True, num_node_features:int=None, num_edge_features:int=None, num_triplet_features:int=None, activation:callable=Softplus(),
                 use_bath_norm: bool = True, bias: bool = True, n_h: int = 3, cutoff_triplet: float = 6.0, l_minus_mean: bool = False):
        """
        Args:
            l_pred_atomwise_tensor (bool, optional): 是否同时预测原子级别的张量。默认为 True。
            ... (其他参数与 Born 类相同)
        """
        super(crystal_tensor, self).__init__()
        self.l_pred_atomwise_tensor = l_pred_atomwise_tensor
        self.atom_tensor_output = Born(include_triplet, num_node_features, num_edge_features, num_triplet_features, activation, use_bath_norm, bias, n_h, cutoff_triplet, l_minus_mean)
    
    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含图表示的字典。

        Returns:
            dict: 包含 'atomic_tensor' 或 'crystal_tensor' 的字典。
        """
        atom_tensors = self.atom_tensor_output(data, graph_representation)
        if self.l_pred_atomwise_tensor:
            return {'atomic_tensor': atom_tensors}
        else:
            output = global_mean_pool(atom_tensors, data['batch'])
            return {'crystal_tensor': output}

class total_energy_and_atomic_forces(nn.Module):
    """同时预测总能量和原子受力的模块。"""
    def __init__(self, num_node_features:int=None, n_h:int=3, activation:callable=nn.Softplus(), derivative:bool=False):
        """
        Args:
            num_node_features (int, optional): 节点特征维度。
            n_h (int, optional): 回归网络隐藏层数。
            activation (callable, optional): 激活函数。
            derivative (bool, optional): 是否通过对总能量求导来计算力。默认为 False。
        """
        super().__init__()
        self.derivative = derivative # 在 Model 中设置 data['pos'] 的梯度
        #self.energy = scalar(aggr='sum', classification=False, num_node_features=num_node_features, n_h=n_h, activation=activation)
        
        self.atom_regression = denseRegression(in_features=num_node_features, out_features=1, bias=True,
                                               use_batch_norm=False, activation=activation, n_h=n_h)
    
    def forward(self, data, graph_representation: dict = None):
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据。
            graph_representation (dict, optional): 包含 'node_attr' 的字典。

        Returns:
            dict: 包含 'forces' 和 'total_energy' 的字典。
        """
        #energy = self.energy(data, graph_representation)['scalar']
        atomic_energy = self.atom_regression(graph_representation.node_attr)
        energy = global_add_pool(atomic_energy, data['batch']).reshape(-1)
        # 通过对总能量关于原子位置求导来计算力
        if self.derivative:
            forces = -torch.autograd.grad(energy, data['pos'],
                                        grad_outputs=torch.ones_like(energy),
                                        create_graph=self.training)[0]
        else:
            forces = None
        return {'forces':forces, 'total_energy':energy}

class EPC_output():
    """计算电子-声子耦合（EPC）矩阵的模块。"""
    def __init__(self, representation:Callable=None, output:Callable=None, band_win_min:int=None, band_win_max:int=None):
        """
        Args:
            representation (Callable, optional): 表示层网络。
            output (Callable, optional): 输出层网络 (通常是 HamGNN_out)。
            band_win_min (int, optional): 计算 EPC 的能带窗口最小值。
            band_win_max (int, optional): 计算 EPC 的能带窗口最大值。
        """
        self.representation = representation
        self.output = output
        self.band_win_min = band_win_min
        self.band_win_max = band_win_max        
        
    def  __call__(self, data):
        out = self.forward(data)
        return out

    def forward(self, data):
        Nbatch = data['cell'].shape[0]
        natoms = int(len(data['z'])/Nbatch) # 每个晶胞中的原子数必须相等，否则batch_size只能为1
        
        # 初始化 orb2atom_idx，即每个原子轨道到其所属原子的索引映射
        atomic_nums = data['z'].view(-1, natoms) # shape: [Nbatch, natoms]
        orb2atom_idx  = []
        for ib in range(Nbatch):
            repeats = []
            for ia in range(natoms):
                repeats.append(len(self.output.basis_def[atomic_nums[ib][ia].item()]))
            repeats = torch.LongTensor(repeats)
            orb2atom_idx.append(torch.repeat_interleave(torch.arange(natoms), repeats, dim=0).type_as(atomic_nums))
        
        # 计算哈密顿量对原子位移的导数 nabla_{R}(HK)
        data = data
        HK, SK, wavefunction, hamiltonian, dSK = None,None,None,None,None
        
        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            nonlocal data, HK, SK, wavefunction, hamiltonian, dSK
            data['pos'] = pos
            graph_representation = self.representation(data)
            out = self.output(data, graph_representation)
            HK, SK, wavefunction, hamiltonian, dSK = out['HK'], out['SK'], out['wavefunction'], out['hamiltonian'], out['dSK']
            return HK
        
        with torch.autograd.detect_anomaly():          
            # shape: [Nbatch, num_k, norbs, norbs, natoms, 3]
            nabla_HK = torch.autograd.functional.jacobian(func=wrapper, inputs=data['pos'], create_graph=False, vectorize=False)

        norbs = HK.shape[-1]
        m = torch.arange(0, norbs)
           
        # 选取指定的能带窗口
        wavefunction = wavefunction[:,:,self.band_win_min-1:self.band_win_max,:]
        wavefunction_conj = torch.conj(wavefunction)
        # 方法一，使用大矩阵计算 EPC 矩阵以提高速度
        """
        epc_mat = []
        for idx in range(Nbatch):        
            #nabla_SK1 = nabla_SK[idx,:,:,m,orb2atom_idx[idx][m],:].type_as(HK) # shape:[num_k, norbs, norbs, 3]
            #nabla_SK2 = nabla_SK[idx,:,n,:,orb2atom_idx[idx][n],:].type_as(HK) # shape:[norbs, num_k, norbs, 3]
            #nabla_SK2 = torch.swapaxes(nabla_SK2, axis0=0, axis1=1) # shape:[num_k, norbs, norbs, 3]
            
            nabla_SK1 = torch.zeros_like(nabla_SK, dtype=HK.dtype)
            nabla_SK1[idx,:,:,m,orb2atom_idx[idx][m],:] = nabla_SK[idx,:,:,m,orb2atom_idx[idx][m],:].type_as(HK)
            
            nabla_SK2 = torch.zeros_like(nabla_SK, dtype=HK.dtype)
            nabla_SK2[idx,:,n,:,orb2atom_idx[idx][n],:] = nabla_SK[idx,:,n,:,orb2atom_idx[idx][n],:].type_as(HK)
            
            sum1 = 'abd, ace, afghi, adf, age -> abchi'
            part1 = torch.einsum(sum1, torch.conj(wavefunction[idx]), wavefunction[idx], nabla_HK[idx], SK[idx], SK[idx])
            
            sum2 = 'abd, ace, afg, adfhi, age -> abchi'
            part2 = torch.einsum(sum2, torch.conj(wavefunction[idx]), wavefunction[idx], HK[idx], nabla_SK1[idx], SK[idx])
            
            sum3 = 'abd, ace, afg, adf, agehi -> abchi'
            part3 = torch.einsum(sum3, torch.conj(wavefunction[idx]), wavefunction[idx], HK[idx], SK[idx], nabla_SK2[idx])
            
            epc_mat.append(part1 + part2 + part3)
        
        epc_mat = torch.cat(epc_mat, dim=0)
        """

        # 方法2，使用循环计算 EPC 矩阵以减少内存开销
        epc_mat_batch = []
        for idx in range(Nbatch): 
            epc_mat = []     
                  
            nabla_SK = torch.zeros_like(nabla_HK, dtype=HK.dtype)
            nabla_SK[idx,:,:,m,orb2atom_idx[idx][m],:] = dSK[idx]
            
            for b in range(wavefunction.shape[-2]):
                for c in range(wavefunction.shape[-2]):
                    temp_sum = []
                    for d in range(norbs):
                        for e in range(norbs):
                            sum1 = 'a, a, afghi, af, ag -> ahi'
                            part1 = torch.einsum(sum1, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], nabla_HK[idx], SK[idx,:,d,:], SK[idx,:,:,e])
            
                            sum2 = 'a, a, afg, afhi, ag -> ahi'
                            part2 = torch.einsum(sum2, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], HK[idx], nabla_SK[idx,:,d,:,:,:], SK[idx,:,:,e])
            
                            sum3 = 'a, a, afg, af, aghi -> ahi'
                            part3 = torch.einsum(sum3, torch.conj(wavefunction_conj[idx,:,b,d]), wavefunction[idx,:,c,e], HK[idx], SK[idx,:,d,:], nabla_SK[idx,:,e,:,:,:])
            
                            temp_sum.append(part1 + part2 + part3)
                    # 对 d 和 e 求和       
                    temp_sum = torch.sum(torch.stack(temp_sum, dim=0), dim=0)
                    epc_mat.append(temp_sum) # shape: [num_k, natoms, 3]
                    
            epc_mat = torch.stack(epc_mat, dim=1).reshape(-1, wavefunction.shape[-2], wavefunction.shape[-2], natoms, 3)
            epc_mat_batch.append(epc_mat)
        epc_mat = torch.stack(epc_mat_batch, dim=0) # shape: [Nbatch, num_k, norbs, norbs, natoms, 3]
        
        
        return {'hamiltonian':hamiltonian, 'epc_mat': epc_mat}
