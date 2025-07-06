'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-08-24 14:22:19
LastEditors: Yang Zhong
LastEditTime: 2024-11-28 22:14:26
'''

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter
from easydict import EasyDict

import warnings
from ase import geometry, neighborlist
import numpy as np
from pymatgen.core.periodic_table import Element
from typing import List, Union

"""
HamGNN 模型的基类定义

本模块提供了 `BaseModel` 类，作为所有 HamGNN 图表示学习模型的基类。
它主要负责实现一个核心功能：在模型内部动态地构建计算图（邻居列表）。
这与在数据预处理阶段固定图结构的方法不同，它允许模型根据原子种类和可调的缩放因子，
为每个结构即时地、灵活地确定原子间的相互作用范围。

该动态图构建功能对于处理多样化的化学环境和实现更具适应性的模型至关重要。
"""

# 不同 DFT 软件使用的原子半径 (单位：Angstrom，abacus 除外)
# 这些半径用于在动态图构建时确定初始的邻居搜索范围
ATOMIC_RADII = {
    'openmx': {
        'H': 6.0, 'He': 8.0, 'Li': 8.0, 'Be': 7.0, 'B': 7.0, 'C': 6.0,
        'N': 6.0, 'O': 6.0, 'F': 6.0, 'Ne': 9.0, 'Na': 9.0, 'Mg': 9.0,
        'Al': 7.0, 'Si': 7.0, 'P': 7.0, 'S': 7.0, 'Cl': 7.0, 'Ar': 9.0,
        'K': 10.0, 'Ca': 9.0, 'Sc': 9.0, 'Ti': 7.0, 'V': 6.0, 'Cr': 6.0,
        'Mn': 6.0, 'Fe': 5.5, 'Co': 6.0, 'Ni': 6.0, 'Cu': 6.0, 'Zn': 6.0,
        'Ga': 7.0, 'Ge': 7.0, 'As': 7.0, 'Se': 7.0, 'Br': 7.0, 'Kr': 10.0,
        'Rb': 11.0, 'Sr': 10.0, 'Y': 10.0, 'Zr': 7.0, 'Nb': 7.0, 'Mo': 7.0,
        'Tc': 7.0, 'Ru': 7.0, 'Rh': 7.0, 'Pd': 7.0, 'Ag': 7.0, 'Cd': 7.0,
        'In': 7.0, 'Sn': 7.0, 'Sb': 7.0, 'Te': 7.0, 'I': 7.0, 'Xe': 11.0,
        'Cs': 12.0, 'Ba': 10.0, 'La': 8.0, 'Ce': 8.0, 'Pr': 8.0, 'Nd': 8.0,
        'Pm': 8.0, 'Sm': 8.0, 'Dy': 8.0, 'Ho': 8.0, 'Lu': 8.0, 'Hf': 9.0,
        'Ta': 7.0, 'W': 7.0, 'Re': 7.0, 'Os': 7.0, 'Ir': 7.0, 'Pt': 7.0,
        'Au': 7.0, 'Hg': 8.0, 'Tl': 8.0, 'Pb': 8.0, 'Bi': 8.0
    },
    'siesta':{},
    'abacus': { # unit: au
    'Ag':7,  'Cu':8,  'Mo':7,  'Sc':8,
    'Al':7,  'Fe':8,  'Na':8,  'Se':8,
    'Ar':7,  'F' :7,  'Nb':8,  'S' :7,
    'As':7,  'Ga':8,  'Ne':6,  'Si':7,
    'Au':7,  'Ge':8,  'N' :7,  'Sn':7,
    'Ba':10, 'He':6,  'Ni':8,  'Sr':9,
    'Be':7,  'Hf':7,  'O' :7,  'Ta':8,
    'B' :8,  'H' :6,  'Os':7,  'Tc':7,
    'Bi':7,  'Hg':9,  'Pb':7,  'Te':7,
    'Br':7,  'I' :7,  'Pd':7,  'Ti':8,
    'Ca':9,  'In':7,  'P' :7,  'Tl':7,
    'Cd':7,  'Ir':7,  'Pt':7,  'V' :8,
    'C' :7,  'K' :9,  'Rb':10, 'W' :8,
    'Cl':7,  'Kr':7,  'Re':7,  'Xe':8,
    'Co':8,  'Li':7,  'Rh':7,  'Y' :8,
    'Cr':8,  'Mg':8,  'Ru':7,  'Zn':8,
    'Cs':10, 'Mn':8,  'Sb':7,  'Zr':8
}
}

DEFAULT_RADIUS = 10.0 # 默认原子半径

def get_radii_from_atomic_numbers(atomic_numbers: Union[torch.Tensor, List[int]], 
                                  radius_scale: float = 1.5, radius_type: str = 'openmx') -> List[float]:
    """
    根据原子序数列表检索缩放后的原子半径。

    Args:
        atomic_numbers (Union[torch.Tensor, List[int]]): 包含原子序数的列表或张量。
        radius_scale (float): 原子半径的缩放因子，默认为 1.5。
        radius_type (str): 所用原子半径的来源软件，默认为 'openmx'。

    Returns:
        List[float]: 与输入原子序数对应的缩放后的原子半径列表。
    """
    
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.tolist()

    # 将原子序数转换为元素符号，然后获取缩放后的半径。
    # 如果在字典中找不到元素，则使用默认值 0.0。
    return [radius_scale * ATOMIC_RADII[radius_type].get(Element.from_Z(z).symbol, DEFAULT_RADIUS) for z in atomic_numbers]


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """基于径向截断距离创建邻居列表和相对向量。

    边的约定如下:
    - `edge_index[0]` 是 *源* (卷积中心)。
    - `edge_index[1]` 是 *目标* (邻居)。

    Args:
        pos (shape [N, 3]): 原子位置坐标；可以是 Tensor 或 numpy 数组。
        r_max (float): 用于寻找邻居的径向截断距离。
        cell (numpy shape [3, 3]): 周期性边界条件的晶胞矩阵。
        pbc (bool or 3-tuple of bool): 三个维度上的周期性。
        self_interaction (bool): 是否包括相同周期性镜像的自相互作用边。
        strict_self_interaction (bool): 是否包括任何自相互作用边。

    Returns:
        edge_index (torch.Tensor [2, num_edges]): 边的列表。
        shifts (torch.Tensor [num_edges, 3]): 相对晶胞平移向量。
        cell_tensor (torch.Tensor [3, 3]): 晶胞张量。
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # 处理位置数据
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    if out_device.type != "cpu":
        warnings.warn(
            "当前，邻居列表计算需要 CPU 数据。如果可能，请传递 CPU 张量。"
        )

    # 处理晶胞数据
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    temp_cell = geometry.complete_cell(temp_cell)

    # 生成邻居列表
    first_index, second_index, shifts = neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=r_max,
        self_interaction=strict_self_interaction,
        use_scaled_positions=False,
    )

    # 过滤自相互作用的边
    if not self_interaction:
        bad_edge = first_index == second_index
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError("消除自相互作用的边后，没有边剩下。")
        first_index = first_index[keep_edge]
        second_index = second_index[keep_edge]
        shifts = shifts[keep_edge]

    # 构建输出
    edge_index = torch.vstack(
        (torch.LongTensor(first_index), torch.LongTensor(second_index))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=torch.long,
        device=out_device,
    )

    return edge_index, shifts, cell_tensor

def find_matching_columns_of_A_in_B(A, B):
    """
    在矩阵 B 中查找与矩阵 A 匹配的列。

    Args:
        A (torch.Tensor): 第一个矩阵。
        B (torch.Tensor): 第二个矩阵。

    Returns:
        torch.Tensor: 在 B 中匹配列的索引。
    """
    assert A.shape[0] == B.shape[0], "A 和 B 的行数必须相同。"
    assert A.shape[-1] <= B.shape[-1], "请增 `radius_scale` 因子！"

    # 转置 A 和 B，将列视为行进行比较
    A_rows = A.T.unsqueeze(1)  # 形状: (num_cols_A, 1, num_rows)
    B_rows = B.T.unsqueeze(0)  # 形状: (1, num_cols_B, num_rows)

    # 比较 A 的每一行与 B 的每一行
    matches = torch.all(A_rows == B_rows, dim=-1)  # 形状: (num_cols_A, num_cols_B)

    # 找到匹配的索引
    matching_indices = matches.nonzero(as_tuple=True)[1]  # 取元组的第二个元素

    return matching_indices

class BaseModel(nn.Module):
    """所有 HamGNN 图学习模型的基类。

    它封装了动态图构建的核心逻辑。

    Attributes:
        radius_type (str): 用于确定原子半径的来源类型 (e.g., 'openmx')。
        radius_scale (float): 原子半径的缩放因子。
    """
    def __init__(self, radius_type: str = 'openmx', radius_scale: float = 1.5) -> None:
        """
        Args:
            radius_type (str): 原子半径的类型，默认为 'openmx'。
            radius_scale (float): 原子半径的缩放因子，默认为 1.5。
        """
        super().__init__()
        self.radius_type = radius_type
        self.radius_scale = radius_scale

    def forward(self, data):
        """前向传播的占位符，应在子类中实现。"""
        raise NotImplementedError

    def generate_graph(
        self,
        data,
    ):
        """
        根据原子位置和种类动态生成计算图（邻居列表）。

        此方法会忽略 `data` 中预先计算的 `edge_index`，并根据每种原子的
        化学性质（半径）和 `radius_scale` 重新计算邻居关系。
        它还会找到新生成的图的边与原始图的边的对应关系。

        Args:
            data (Data): 输入的图数据对象，必须包含 `pos`, `z`, `cell`, `batch` 等信息。

        Returns:
            EasyDict: 
                一个包含新生成的图信息的字典。
                - 'z': 原子序数。
                - 'pos': 原子位置。
                - 'edge_index': 新的边索引。
                - 'cell_shift': 边的晶胞平移向量。
                - 'nbr_shift': 邻居的相对位置向量。
                - 'batch': 批处理索引。
                - 'matching_edges': 新图的边在原始图中的匹配索引。
        """
        graph = EasyDict()

        node_counts = scatter(torch.ones_like(data.batch), data.batch, dim=0).detach()

        latt_batch = data.cell.detach().reshape(-1, 3, 3)
        pos_batch = data.pos.detach()

        pos_batch = torch.split(pos_batch, node_counts.tolist(), dim=0)
        z_batch = torch.split(data.z.detach(), node_counts.tolist(), dim=0)
        
        nbr_shift = []
        edge_index = []
        cell_shift = []

        # 遍历批处理中的每个晶体
        for idx_xtal, pos in enumerate(pos_batch):
            # 基于原子半径动态计算邻居列表
            edge_index_temp, shifts_tmp, _ = neighbor_list_and_relative_vec(
                pos,
                r_max=get_radii_from_atomic_numbers(z_batch[idx_xtal], radius_scale=self.radius_scale, radius_type=self.radius_type),
                self_interaction=False,
                strict_self_interaction=True,
                cell=latt_batch[idx_xtal],
                pbc=True,
            )
            # 计算邻居的真实位移向量
            nbr_shift_temp = torch.einsum('ni, ij -> nj',  shifts_tmp.type_as(pos), latt_batch[idx_xtal])
            
            # 调整边索引以适应批处理
            if idx_xtal > 0:
                edge_index_temp += node_counts[idx_xtal - 1]

            edge_index.append(edge_index_temp)
            cell_shift.append(shifts_tmp)
            nbr_shift.append(nbr_shift_temp)

        # 拼接所有晶体的图信息
        edge_index = torch.cat(edge_index, dim=-1).type_as(data.edge_index)
        cell_shift = torch.cat(cell_shift, dim=0).type_as(data.cell_shift)
        nbr_shift = torch.cat(nbr_shift, dim=0).type_as(data.nbr_shift)

        # 找到新生成的边与原始数据中边的对应关系
        matching_edges = find_matching_columns_of_A_in_B(torch.cat([data.edge_index, data.cell_shift.t()], dim=0), 
                                                      torch.cat([edge_index, cell_shift.t()], dim=0))

        # 填充图字典
        graph['z'] = data.z
        graph['pos'] = data.pos
        graph['edge_index'] = edge_index
        graph['cell_shift'] = cell_shift
        graph['nbr_shift'] = nbr_shift
        graph['batch'] = data.batch
        graph['matching_edges'] = matching_edges

        return graph

    @property
    def num_params(self) -> int:
        """计算模型的总参数数量。"""
        return sum(p.numel() for p in self.parameters())
