'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-08-24 20:42:41
LastEditors: Yang Zhong
LastEditTime: 2024-10-10 17:43:34
'''
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import torch
from torch import nn
import math
from collections import OrderedDict
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_unit_step
from torch_scatter import scatter
from e3nn.util.jit import compile_mode
from ..Toolbox.nequip.data import AtomicDataDict
import torch.nn.functional as F 
from ..Toolbox.mace.modules.blocks import EquivariantProductBasisBlock

from ..Toolbox.mace.modules.irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from torch_geometric.utils import softmax as edge_softmax
from ..Toolbox.efficient_kan import KAN
from ..Toolbox.nequip.nn.nonlinearities import ShiftedSoftPlus
from e3nn.nn import Gate, NormActivation
from ..Toolbox.nequip.nn import GraphModuleMixin
from ..layers import cuttoff_envelope, CosineCutoff

GRID_SIZE = 3
GRID_RANGE = [-1, 1]

class TensorExpansion(nn.Module):
    r"""将哈密顿量或重叠矩阵从原子轨道基组展开为球谐函数基组。

    该模块的核心操作是执行以下基组变换，将原子轨道 :math:`|l_i, m_i \rangle` 和 :math:`|l_j, m_j \rangle`
    耦合为总角动量 :math:`|L, M \rangle` 的不可约表示：

    .. math::

       O_{LM} = \frac{1}{N} \sum_{m_i, m_j} C_{l_i m_i, l_j m_j}^{L M} \cdot H_{l_i m_i, l_j m_j}

    其中 :math:`C` 是 Clebsch-Gordan 系数。

    该模块还能处理不同DFT软件（如 'openmx', 'siesta', 'abacus'）的原子轨道排序约定，
    并将它们统一转换为 e3nn 库兼容的不可约表示 (irreps) 形式。

    Attributes:
        ham_type (str): 哈密顿矩阵类型 ('openmx', 'siesta', 'abacus', 'pasp')。
        nao_max (int): 原子轨道的最大数量。
        irreps_out (o3.Irreps): 展开后的输出不可约表示。
    """
    def __init__(self, ham_type, nao_max):
        """
        初始化张量展开模块。

        Args:
            ham_type (str): 哈密顿矩阵类型 ('openmx', 'siesta', 'abacus', 'pasp')。
            nao_max (int): 原子轨道的最大数量。
        """
        super().__init__()
        self.ham_type = ham_type
        self.nao_max = nao_max
        self.index_change = None
        self.minus_index = None
        self.row = None
        self.col = None
        self._set_basis_info()
        
        # 计算Clebsch-Gordan系数所需的最大l值
        max_l = self.row.lmax + self.col.lmax
        self.cg_calculator = ClebschGordanCoefficients(max_l=max_l)

        irreps_combined = self._combine_irreps()
        self.irreps_out, self.permute_indices, self.inverse_permute_indices = o3.Irreps(irreps_combined).sort()
        self.irreps_out = self.irreps_out.simplify()

    def _combine_irreps(self):
        """

        组合行和列的不可约表示(irreps)以确定输出的不可约表示。

        Returns:
            o3.Irreps: 组合后的不可约表示列表。
        """
        combined_irreps = []
        for _, li in self.row:
            for _, lj in self.col:
                for L in range(abs(li.l - lj.l), li.l + lj.l + 1):
                    combined_irreps.append(o3.Irrep(L, (-1) ** (li.l + lj.l)))
        return o3.Irreps(combined_irreps)

    def _get_index_change_inv(self, index_change):
        """
        获取索引变换张量的逆变换。

        Args:
            index_change (torch.Tensor): 表示索引变换的张量。
            
        Returns:
            torch.Tensor: 表示逆索引变换的张量。
        """
        index_change_inv = torch.zeros_like(index_change)
        
        for i in range(len(index_change)):
            index_change_inv[index_change[i]] = i
        
        return index_change_inv

    def _set_basis_info(self):
        """
        根据哈密顿矩阵类型和原子轨道数量设置基组信息。
        """
        if self.ham_type == 'openmx':
            self._set_openmx_basis()
        elif self.ham_type == 'siesta':
            self._set_siesta_basis()
        elif self.ham_type == 'abacus':
            self._set_abacus_basis()
        elif self.ham_type == 'pasp':
            self.row = self.col = o3.Irreps("1x1o")
        else:
            raise NotImplementedError(f"Hamiltonian type '{self.ham_type}' is not supported.")

    def _set_openmx_basis(self):
        """
        为 'openmx' 哈密顿矩阵设置基组信息。
        """
        if self.nao_max == 14:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e")
        elif self.nao_max == 13:
            self.index_change = torch.LongTensor([0, 1, 4, 2, 3, 7, 5, 6, 10, 12, 8, 11, 9])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
        elif self.nao_max == 19:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10, 16, 18, 14, 17, 15])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
        elif self.nao_max == 26:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10, 16, 18, 14, 17, 15, 22, 23, 21, 24, 20, 25, 19])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'openmx'.")

    def _set_siesta_basis(self):
        """
        为 'siesta' 哈密顿矩阵设置基组信息。
        """
        if self.nao_max == 13:
            self.index_change = None
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([2, 4, 5, 7, 9, 11])
        elif self.nao_max == 19:
            self.index_change = None
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
            self.minus_index = torch.LongTensor([3, 5, 6, 8, 10, 12, 15, 17])
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'siesta'.")

    def _set_abacus_basis(self):
        """
        为 'abacus' 哈密顿矩阵设置基组信息。
        """
        if self.nao_max == 13:
            self.index_change = torch.LongTensor([0, 1, 3, 4, 2, 6, 7, 5, 10, 11, 9, 12, 8])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([3, 4, 6, 7, 9, 10])
        elif self.nao_max == 27:
            self.index_change = torch.LongTensor([0, 1, 2, 3, 5, 6, 4, 8, 9, 7, 12, 13, 11, 14, 10, 17, 18, 16, 19, 15, 23, 24, 22, 25, 21, 26, 20])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
            self.minus_index = torch.LongTensor([5, 6, 8, 9, 11, 12, 16, 17, 21, 22, 25, 26])
        elif self.nao_max == 40:
            self.index_change = torch.LongTensor([0, 1, 2, 3, 5, 6, 4, 8, 9, 7, 11, 12, 10, 14, 15, 13, 18, 19, 17, 20, 16, 23, 24, 22, 25, 21, 29, 30, 28, 31, 27, 32, 26, 36, 37, 35, 38, 34, 39, 33])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x3o+1x3o")
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'abacus'.")

    def _change_index(self, hamiltonian):
        """
        根据 `index_change` 和 `minus_index` 调整哈密顿矩阵的索引和符号。
        
        Args:
            hamiltonian (torch.Tensor): 需要调整索引的哈密顿矩阵。
            
        Returns:
            torch.Tensor: 调整索引后的哈密顿矩阵。
        """
        has_minus_index = False
        try:
            _ = self.minus_index
            has_minus_index = True
        except AttributeError:
            pass
        
        if self.index_change is not None or has_minus_index:
            hamiltonian = hamiltonian.reshape(-1, self.nao_max, self.nao_max)   
            if self.index_change is not None:
                hamiltonian = hamiltonian[:, self.index_change[:,None], self.index_change[None,:]] 
            if has_minus_index:
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]                
        return hamiltonian

    def _change_index_inv(self, hamiltonian):
        """
        根据 `index_change` 和 `minus_index` 逆向调整哈密顿矩阵的索引和符号。
        
        Args:
            hamiltonian (torch.Tensor): 需要逆向调整索引的哈密顿矩阵。
            
        Returns:
            torch.Tensor: 逆向调整索引后的哈密顿矩阵。
        """
        has_minus_index = False
        try:
            _ = self.minus_index
            has_minus_index = True
        except AttributeError:
            pass
        
        if self.index_change is not None or has_minus_index:
            hamiltonian = hamiltonian.reshape(-1, self.nao_max, self.nao_max) 
            if has_minus_index:
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]  
            if self.index_change is not None:
                index_change_inv = self._get_index_change_inv(self.index_change)
                hamiltonian = hamiltonian[:, index_change_inv[:,None], index_change_inv[None,:]]               
        return hamiltonian

    def forward(self, x):
        """
        前向传播，将输入矩阵展开为球谐函数基组。

        Args:
            x (torch.Tensor): 输入张量，形状为 (\*, row.dim, col.dim)。

        Returns:
            torch.Tensor: 展开后的张量，其不可约表示由 `self.irreps_out` 定义。
        """
        x = x.reshape(-1, self.row.dim, self.col.dim)
        x = self._change_index_inv(x)
        
        output_blocks = []

        row_start = 0
        for _, li in self.row:
            num_rows = 2 * li.l + 1
            col_start = 0
            for _, lj in self.col:
                num_cols = 2 * lj.l + 1
                for L in range(abs(li.l - lj.l), li.l + lj.l + 1):
                    # 计算Clebsch-Gordan系数
                    cg_coeffs = self.cg_calculator(L, li.l, lj.l)
                    block = x.narrow(-2, row_start, num_rows).narrow(-1, col_start, num_cols)
                    output_blocks.append(torch.einsum('nij, kij -> nk', block, cg_coeffs))

                col_start += num_cols
            row_start += num_rows

        # 连接输出并应用逆置换
        expanded_output = torch.cat([output_blocks[idx] for idx in self.inverse_permute_indices], dim=-1)
        return expanded_output

class OverlapExpand(nn.Module):
    """
    一个封装了 `TensorExpansion` 的模块，用于展开重叠矩阵。
    """
    def __init__(self, ham_type, nao_max) -> None:
        """
        初始化 OverlapExpand 模块。

        Args:
            ham_type (str): 哈密顿矩阵类型 ('openmx', 'siesta', 'abacus', 'pasp')。
            nao_max (int): 原子轨道的最大数量。
        """
        super().__init__()
        self.tensor_expansion = TensorExpansion(ham_type=ham_type, nao_max=nao_max)
        self.irreps_overlap = self.tensor_expansion.irreps_out

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播，展开在位(on-site)和异位(off-site)的重叠矩阵。

        Args:
            data (dict): 包含 'Son' 和 'Soff' 张量的数据对象，待被展开。

        Returns:
            dict: 更新后的数据对象，包含展开后的 'Son_expand' 和 'Soff_expand'。
        """
        data['Son_expand'] = self.tensor_expansion(data['Son'])
        data['Soff_expand'] = self.tensor_expansion(data['Soff'])
        return data

@compile_mode("script")
class TensorWrapper(nn.Module):
    """包装张量以便在ModuleDict中使用，确保TorchScript兼容性"""
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        # 将张量注册为buffer，自动处理设备移动
        self.register_buffer('data', tensor)

@compile_mode("script")
class ClebschGordanCoefficients(nn.Module):
    r"""
    预计算和存储 Clebsch-Gordan (CG) 系数的模块。

    该模块利用 `e3nn.o3.wigner_3j` 来计算 Wigner 3j 符号，并通过它们得到 CG 系数。
    Wigner 3j 符号与 CG 系数通过以下关系相关联：

    .. math::

       \langle l_1 m_1, l_2 m_2 | l_3 m_3 \rangle =
       (-1)^{l_1 - l_2 + m_3} \sqrt{2l_3 + 1}
       \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & -m_3 \end{pmatrix}
       
    此版本经过重构，以实现完整的JIT兼容性和新旧模型双向兼容。
    所有兼容性逻辑都被封装在此类内部，对外部代码完全透明。
    """

    def __init__(self, max_l=8):
        """
        初始化模块并预计算 Clebsch-Gordan 系数，直到指定的最大角动量。

        Args:
            max_l (int): 计算系数的最大角动量(l)。
        """
        super().__init__()
        
        self.max_l = max_l
        
        # 为TorchScript兼容性，使用tensor存储而非ModuleDict
        # 创建一个6D张量来存储所有的CG系数（因为wigner_3j返回3D张量）
        # 索引方式: coeffs[l1, l2, l3-abs(l1-l2), :d1, :d2, :d3]
        # 最大可能的l3是l1+l2，所以最大可能的维度是2*(l1+l2)+1 = 2*(max_l+max_l)+1
        max_dim_l1_l2 = 2 * max_l + 1  # l1和l2的最大维度
        max_dim_l3 = 2 * (2 * max_l) + 1  # l3的最大维度（当l3=l1+l2时）
        # 第三个维度需要能容纳所有可能的l3值
        max_l3_range = 2 * max_l + 1  # 从0到2*max_l的范围  
        self.register_buffer('_cg_tensor', torch.zeros((max_l + 1, max_l + 1, max_l3_range, max_dim_l1_l2, max_dim_l1_l2, max_dim_l3)))
        self.register_buffer('_cg_mask', torch.zeros((max_l + 1, max_l + 1, max_l3_range), dtype=torch.bool))
        
        # 预计算并存储所有必需的Clebsch-Gordan系数
        for l1 in range(max_l + 1):
            for l2 in range(max_l + 1):
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    idx = l3 - abs(l1 - l2)
                    cg_coeffs = o3.wigner_3j(l1, l2, l3)
                    # wigner_3j返回的是3D张量
                    d1, d2, d3 = cg_coeffs.shape
                    self._cg_tensor[l1, l2, idx, :d1, :d2, :d3] = cg_coeffs
                    self._cg_mask[l1, l2, idx] = True

    def forward(self, l1: int, l2: int, l3: int) -> torch.Tensor:
        """
        获取给定角动量的预计算 Clebsch-Gordan 系数。

        Args:
            l1 (int): 第一个角动量。
            l2 (int): 第二个角动量。
            l3 (int): 第三个角动量。
            
        Returns:
            torch.Tensor: 对应的 Clebsch-Gordan 系数张量。
        """
        # TorchScript兼容: 使用张量索引而非字符串索引
        idx = l3 - abs(l1 - l2)
        d1 = 2 * l1 + 1
        d2 = 2 * l2 + 1
        d3 = 2 * l3 + 1
        return self._cg_tensor[l1, l2, idx, :d1, :d2, :d3]

    def state_dict(self, *args, **kwargs):
        """
        保持向后兼容性，state_dict保持标准格式
        """
        return super().state_dict(*args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        处理旧checkpoint的兼容性加载
        旧格式: prefix + 'cg_l1_l2_l3' (直接存储的tensor)
        新格式: prefix + '_cg_tensor', prefix + '_cg_mask' (统一的tensor存储)
        """
        # 检查是否是旧格式的checkpoint
        old_format_keys = [k for k in state_dict if k.startswith(prefix + 'cg_')]
        
        if old_format_keys:
            # 创建新格式的张量
            max_l = getattr(self, 'max_l', 8)
            max_dim_l1_l2 = 2 * max_l + 1  # l1和l2的最大维度
            max_dim_l3 = 2 * (2 * max_l) + 1  # l3的最大维度（当l3=l1+l2时）
            max_l3_range = 2 * max_l + 1  # 从0到2*max_l的范围
            cg_tensor = torch.zeros((max_l + 1, max_l + 1, max_l3_range, max_dim_l1_l2, max_dim_l1_l2, max_dim_l3))
            cg_mask = torch.zeros((max_l + 1, max_l + 1, max_l3_range), dtype=torch.bool)
            
            # 从旧格式转换
            for key in old_format_keys:
                cg_name = key[len(prefix):]  # 'cg_0_1_1'
                parts = cg_name.split('_')
                if len(parts) == 4 and parts[0] == 'cg':
                    l1, l2, l3 = int(parts[1]), int(parts[2]), int(parts[3])
                    idx = l3 - abs(l1 - l2)
                    cg_coeffs = state_dict.pop(key)
                    
                    # 旧checkpoint的CG系数直接使用，不需要重构
                    # 获取实际的CG系数形状
                    if cg_coeffs.dim() == 3:
                        d1, d2, d3 = cg_coeffs.shape
                    elif cg_coeffs.dim() == 2:
                        # 如果是2D，假设是(d1*d2, d3)格式
                        expected_d1 = 2 * l1 + 1
                        expected_d2 = 2 * l2 + 1
                        h, d3 = cg_coeffs.shape
                        if h == expected_d1 * expected_d2:
                            d1, d2 = expected_d1, expected_d2
                            cg_coeffs = cg_coeffs.reshape(d1, d2, d3)
                        else:
                            # 使用实际形状
                            d1, d2 = cg_coeffs.shape
                            d3 = 1
                            cg_coeffs = cg_coeffs.unsqueeze(-1)
                    else:
                        # 1D情况，需要推断
                        expected_d1 = 2 * l1 + 1
                        expected_d2 = 2 * l2 + 1
                        expected_d3 = 2 * l3 + 1
                        if cg_coeffs.numel() == expected_d1 * expected_d2 * expected_d3:
                            d1, d2, d3 = expected_d1, expected_d2, expected_d3
                            cg_coeffs = cg_coeffs.reshape(d1, d2, d3)
                        else:
                            # 保守处理：使用平方根估算
                            numel = cg_coeffs.numel()
                            d1 = expected_d1
                            d2 = expected_d2
                            d3 = numel // (d1 * d2)
                            cg_coeffs = cg_coeffs.reshape(d1, d2, d3)
                    
                    try:
                        cg_tensor[l1, l2, idx, :d1, :d2, :d3] = cg_coeffs
                        cg_mask[l1, l2, idx] = True
                    except RuntimeError as e:
                        print(f" 在设置CG({l1},{l2},{l3})时出错：")
                        print(f"  idx={idx}, cg_coeffs.shape={cg_coeffs.shape}, 期望放置位置=[{l1}, {l2}, {idx}, :{d1}, :{d2}, :{d3}]")
                        print(f"  cg_tensor.shape={cg_tensor.shape}")
                        raise e
            
            # 添加新格式的键
            state_dict[prefix + '_cg_tensor'] = cg_tensor
            state_dict[prefix + '_cg_mask'] = cg_mask
        
        # 调用父类方法继续加载
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

@compile_mode("script")
class LinearScaleWithWeights(nn.Module):
    """
    一个线性缩放模块，其权重由外部提供。
    
    该模块首先通过张量积(tensor product)将输入特征与一个伪标量(pseudo-scalar)相乘，
    然后应用一个外部提供的权重向量。最后，通过一个线性层将结果映射到输出的不可约表示。
    """
    def __init__(self, irreps_in, irreps_out):
        """
        初始化带权重的线性缩放模块。
        
        Args:
            irreps_in (o3.Irreps): 输入的不可约表示。
            irreps_out (o3.Irreps): 输出的不可约表示。
        """
        super().__init__()
        
        instructions =  [(i, 0, i, "uvu", True) for i in range(len(irreps_in))]
        
        self.tp = o3.TensorProduct(
            irreps_in,
            o3.Irreps('1x0e'),
            irreps_in,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.weight_numel = self.tp.weight_numel
        
        self.linear_out = o3.Linear(irreps_in, irreps_out, internal_weights=True, shared_weights=True)
        
    def forward(self, x, weight):
        """
        带权重的线性变换前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。
            weight (torch.Tensor): 权重张量。
            
        Returns:
            torch.Tensor: 变换后的输出张量。
        """
        y = torch.ones_like(x[:, 0:1])
        out = self.tp(x, y, weight)
        out = self.linear_out(out)
        return out

@compile_mode("script")
class SoftUnitStepCutoff(nn.Module):
    """
    一个应用带截断(cutoff)的软单位阶跃函数的模块。
    
    该函数在接近截断半径时平滑地将值过渡到零。
    
    Attributes:
        cutoff (float): 应用截断的距离。
        cut_param (nn.Parameter): 影响阶跃函数平滑度的可学习参数。
    """
    def __init__(self, cutoff):
        """
        初始化 SoftUnitStepCutoff 模块。
        
        Args:
            cutoff (float): 阶跃函数的截断距离。
        """
        super(SoftUnitStepCutoff, self).__init__()
        self.cutoff = cutoff
        self.cut_param = nn.Parameter(torch.tensor(10.0, dtype=torch.get_default_dtype()))

    def forward(self, edge_distance):
        """
        模块的前向传播。
        
        对输入的边距离应用软单位阶跃函数。
        
        Args:
            edge_distance (torch.Tensor): 包含边距离的张量。
        
        Returns:
            torch.Tensor: 应用截断后计算得到的边权重张量。
        """
        # 计算缩放后的差异并应用软单位阶跃函数
        scaled_diff = self.cut_param * (1.0 - edge_distance / self.cutoff)
        edge_weight_cutoff = soft_unit_step(scaled_diff)
        
        return edge_weight_cutoff

def count_neighbors_per_node(source_nodes):
    """
    计算图中每个节点的邻居数量。

    Args:
        source_nodes (torch.Tensor): 包含源节点索引的一维张量 (通常是 `edge_index[0]`)。

    Returns:
        torch.Tensor: 一个张量，其索引对应节点ID，值为该节点的邻居数量。
    """
    # 识别唯一节点并计算它们的出现次数
    unique_nodes, counts = torch.unique(source_nodes, return_counts=True)

    # 确定图中的总节点数
    total_nodes = source_nodes.max().item() + 1

    # 初始化一个张量来存储每个节点的邻居计数
    neighbor_counts = torch.zeros((total_nodes,)).type_as(source_nodes)

    # 将计数分配给它们各自的节点
    neighbor_counts[unique_nodes] = counts

    # 确保输出张量与输入具有相同的类型
    return neighbor_counts

@compile_mode("script")
class TensorProductWithMemoryOptimizationWithWeight(nn.Module):
    """
    一个带有内存优化和动态权重的张量积模块。

    该模块通过一个径向多层感知机(MLP)或KAN网络从标量输入动态生成权重，
    然后将这些权重应用于两个输入不可约表示(irreps)的张量积结果。
    """
    def __init__(self, irreps_input_1, irreps_input_2, irreps_out, irreps_scalar, radial_MLP, use_kan):
        """
        初始化 TensorProductWithMemoryOptimizationWithWeight 模块。

        Args:
            irreps_input_1 (str): 第一个输入的不可约表示。
            irreps_input_2 (str): 第二个输入的不可约表示。
            irreps_out (str): 输出的不可约表示。
            irreps_scalar (str): 标量输入的不可约表示，用于生成权重。
            radial_MLP (List[int]): 径向MLP的隐藏层维度列表。
            use_kan (bool): 是否使用KAN代替传统的MLP。
        """
        super().__init__()

        # 初始化不可约表示
        self.irreps_input_1 = o3.Irreps(irreps_input_1)
        self.irreps_input_2 = o3.Irreps(irreps_input_2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_scalar = o3.Irreps(irreps_scalar)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        # 计算中间不可约表示和指令
        self.irreps_mid, self.instructions = self._tp_out_irreps_with_instructions(
            self.irreps_input_1,
            self.irreps_input_2,
            self.irreps_out,
        )

        # 初始化张量积
        self.tensor_product = o3.TensorProduct(
            self.irreps_input_1,
            self.irreps_input_2,
            self.irreps_mid,
            instructions=self.instructions,
            internal_weights=True, 
            shared_weights=True
        )

        # 初始化带权重的线性缩放层
        self.linear_scaler = LinearScaleWithWeights(
            irreps_in=self.irreps_mid.simplify(),
            irreps_out=self.irreps_out
        )

        # 初始化权重生成器
        input_dim = self.irreps_scalar.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.linear_scaler.weight_numel)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        """
        计算张量积的输出不可约表示和指令。
        """
        trainable = True

        # 收集可能的不可约表示和它们的指令
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # 对张量积的输出不可约表示进行排序，以便在提供给第二个o3.Linear时可以简化它们
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # 置换指令的输出索引以匹配排序后的不可约表示
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器模块。

        Args:
            input_dim (int): 权重生成器的输入维度。
            weight_numel (int): 权重向量中的元素数量。

        Returns:
            nn.Module: 初始化后的权重生成器模块。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, x, y, scalars):
        """
        TensorProductWithMemoryOptimizationWithWeight 模块的前向传播。

        Args:
            x (torch.Tensor): 第一个不可约表示的输入张量。
            y (torch.Tensor): 第二个不可约表示的输入张量。
            scalars (torch.Tensor): 用于生成权重的标量输入张量。

        Returns:
            torch.Tensor: 应用张量积和缩放后的输出张量。
        """
        # 使用标量MLP生成权重
        weights = self.weight_generator(scalars)

        # 计算张量积
        output = self.tensor_product(x, y)
        output = self.linear_scaler(output, weights)

        return output

@compile_mode("script")
class TensorProductWithScalarComponents(nn.Module):
    """
    一个带内存优化的张量积模块，专门处理至少一个输入是标量的情况。
    """

    def __init__(self, irreps_input_1, irreps_input_2, irreps_out):
        """
        初始化带标量组件的张量积模块。
        
        Args:
            irreps_input_1 (str): 第一个输入的不可约表示。
            irreps_input_2 (str): 第二个输入的不可约表示。
            irreps_out (str): 输出的不可约表示。
        """
        super().__init__()

        # 初始化不可约表示
        self.irreps_input_1 = o3.Irreps(irreps_input_1)
        self.irreps_input_2 = o3.Irreps(irreps_input_2)
        self.irreps_out = o3.Irreps(irreps_out)

        # 计算中间不可约表示和指令
        irreps_mid_list = []
        instructions = []
        for i, (mul_1, ir_1) in enumerate(self.irreps_input_1):
            for j, (mul_2, ir_2) in enumerate(self.irreps_input_2):
                for _, (mul_o, ir_out) in enumerate(self.irreps_out):                  
                    if (ir_out in ir_1 * ir_2) and ((ir_1.l, ir_1.p) == (0, 1) or (ir_2.l, ir_2.p) == (0, 1)):
                        k = len(irreps_mid_list)
                        instructions += [(i, j, k, "uvw", True)]
                        irreps_mid_list.append((mul_o, ir_out))

        irreps_mid = o3.Irreps(irreps_mid_list)
        irreps_mid, permut, _ = irreps_mid.sort()

        # 置换指令的输出索引以匹配排序后的不可约表示
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
    
        instructions = sorted(instructions, key=lambda x: x[2])

        # 初始化张量积
        self.tensor_product = o3.TensorProduct(
            self.irreps_input_1,
            self.irreps_input_2,
            irreps_mid,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True,
        )

        # 初始化线性层
        self.linear_out = o3.Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out,
            internal_weights=True, 
            shared_weights=True
        )

    def forward(self, x, y):
        """
        模块的前向传播。

        Args:
            x (torch.Tensor): 第一个不可约表示的输入张量。
            y (torch.Tensor): 第二个不可约表示的输入张量。

        Returns:
            torch.Tensor: 应用张量积和缩放后的输出张量。
        """
        # 计算张量积
        output = self.tensor_product(x, y)
        output = self.linear_out(output)

        return output

def extract_scalar_irreps(irreps: o3.Irreps) -> o3.Irreps:
    """
    从给定的不可约表示中提取并返回标量不可约表示。

    标量不可约表示定义为 l=0 且 p=1 (即 '0e') 的表示。
    该函数计算此类标量不可约表示的总多重度，并构造一个仅包含这些表示的新 Irreps 对象。

    Args:
        irreps (o3.Irreps): 要从中提取标量分量的输入不可约表示。

    Returns:
        o3.Irreps: 仅包含标量分量的新 Irreps 对象。
    """
    scalar_multiplicity = sum(
        multiplicity for multiplicity, irrep in irreps if irrep.l == 0 and irrep.p == 1
    )
    return o3.Irreps(f"{scalar_multiplicity}x0e")

@compile_mode("script")
class EdgeScalarEmbedding(nn.Module):
    """
    一个从源节点属性、目标节点属性和边嵌入计算边标量的层。
    """
    def __init__(self, irreps_node_attrs, irreps_edge_embed, irreps_edge_scalars):
        """
        初始化边标量嵌入模块。
        
        Args:
            irreps_node_attrs (o3.Irreps): 节点属性的不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入的不可约表示。
            irreps_edge_scalars (o3.Irreps): 边标量的不可约表示。
        """
        super().__init__()
        self.linear_out = o3.Linear(
            irreps_node_attrs + irreps_node_attrs + irreps_edge_embed, irreps_edge_scalars
        )
        
    def forward(self, node_attr_src, node_attr_dst, edge_embed):
        """
        前向传播，计算边标量。

        Args:
            node_attr_src (torch.Tensor): 源节点属性。
            node_attr_dst (torch.Tensor): 目标节点属性。
            edge_embed (torch.Tensor): 边嵌入。

        Returns:
            torch.Tensor: 计算出的边标量。
        """
        combined_features = torch.cat([node_attr_src, node_attr_dst, edge_embed], dim=-1)
        return self.linear_out(combined_features)

@compile_mode("script")
class LocalEnvironmentEmbedding(nn.Module):
    """
    使用节点和边属性、边嵌入和球谐函数来嵌入局部环境。
    """
    def __init__(self, irreps_edge_attrs, irreps_edge_embed, irreps_node_attrs,
                 irreps_edge_scalars, irreps_env_sh, radial_MLP=[64, 64], use_kan=False):
        """
        初始化局部环境嵌入模块。
        
        Args:
            irreps_edge_attrs (o3.Irreps): 边属性的不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入的不可约表示。
            irreps_node_attrs (o3.Irreps): 节点属性的不可约表示。
            irreps_edge_scalars (o3.Irreps): 边标量的不可约表示。
            irreps_env_sh (o3.Irreps): 环境球谐函数的不可约表示。
            radial_MLP (List[int]): 径向MLP的维度列表。
            use_kan (bool): 是否使用KAN模型。
        """
        super().__init__()

        self.edge_scalar_layer = EdgeScalarEmbedding(irreps_node_attrs, irreps_edge_embed, irreps_edge_scalars)
        
        instructions = [(i, 0, i, "uvw", True) for i in range(len(irreps_edge_attrs))]
        
        self.tensor_product = o3.TensorProduct(
            irreps_edge_attrs,
            o3.Irreps('1x0e'),
            irreps_env_sh,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        
        self.weight_numel = self.tensor_product.weight_numel

        input_dim = irreps_edge_embed.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.weight_numel, radial_MLP, use_kan)

    def _initialize_weight_generator(self, input_dim, weight_numel, radial_MLP, use_kan):
        """
        初始化权重生成器。

        Args:
            input_dim (int): 生成器的输入维度。
            weight_numel (int): 权重中的元素数量。
            radial_MLP (list[int]): 径向MLP的维度。
            use_kan (bool): 是否使用KAN模型。

        Returns:
            nn.Module: 权重生成器模型。
        """
        if use_kan:
            return KAN([input_dim] + radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )
        
    def forward(self, edge_index, node_attr, edge_attr, edge_embed):
        """
        前向传播，计算局部环境嵌入。

        Args:
            edge_index (torch.Tensor): 边的索引。
            node_attr (torch.Tensor): 节点属性。
            edge_attr (torch.Tensor): 边属性。
            edge_embed (torch.Tensor): 边嵌入。

        Returns:
            torch.Tensor: 局部环境嵌入。
        """
        src = edge_index[0]
        dst = edge_index[1]
        pseudo_scalar = torch.ones_like(edge_embed[:, :1])
        
        edge_scalars = self.edge_scalar_layer(node_attr[src], node_attr[dst], edge_embed)
        weights = self.weight_generator(edge_scalars)
        local_env_edge = self.tensor_product(edge_attr, pseudo_scalar, weights)
        
        return local_env_edge

@compile_mode("script")
class ConcatenatedIrrepsTensorProduct(nn.Module):
    """
    一个张量积模块，它首先将第一个输入的多个张量连接起来，
    然后再与第二个输入进行张量积。
    """
    def __init__(self, irreps_in1, irreps_in2, num_tensors_in1, irreps_out, irreps_edge_scalars, radial_MLP, use_kan):
        """
        初始化连接不可约表示的张量积模块。

        Args:
            irreps_in1 (o3.Irreps): 第一个输入张量的不可约表示。
            irreps_in2 (o3.Irreps): 第二个输入张量的不可约表示。
            num_tensors_in1 (int): 第一个输入的张量数量。
            irreps_out (o3.Irreps): 期望的输出不可约表示。
            irreps_edge_scalars (o3.Irreps): 边标量的不可约表示。
            radial_MLP (List[int]): 径向多层感知机的维度。
            use_kan (bool): 是否使用KAN生成权重。
        """
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan
        self.num_tensors_in1 = num_tensors_in1
        self.irreps_in1_combined = scale_irreps(self.irreps_in1, self.num_tensors_in1)

        self.fuse_in = AttentionHeadsToVector(self.irreps_in1)
        
        # 计算中间不可约表示和指令
        self.irreps_mid, self.instructions = self. _tp_out_irreps_with_instructions(
            self.irreps_in1_combined,
            self.irreps_in2,
            self.irreps_out,
        )

        # 初始化张量积
        self.tensor_product = o3.TensorProduct(
            self.irreps_in1_combined,
            self.irreps_in2,
            self.irreps_mid,
            instructions=self.instructions,
            internal_weights=True,
            shared_weights=True
        )

        # 初始化带权重的线性缩放层
        self.linear_scaler = LinearScaleWithWeights(
            irreps_in=self.irreps_mid.simplify(),
            irreps_out=self.irreps_out
        )

        # 初始化权重生成器
        input_dim = self.irreps_edge_scalars.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.linear_scaler.weight_numel)

        # 线性组合
        self.linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        """
        计算张量积的输出不可约表示和指令。
        """
        trainable = True

        # 收集可能的不可约表示和它们的指令
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # 对张量积的输出不可约表示进行排序
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # 置换指令的输出索引以匹配排序后的不可约表示
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器模块。

        Args:
            input_dim (int): 权重生成器的输入维度。
            weight_numel (int): 权重向量中的元素数量。

        Returns:
            nn.Module: 初始化后的权重生成器模块。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, input_tensors1_list: List[torch.Tensor], input_tensor2: torch.Tensor, scalars: torch.Tensor):
        """
        ConcatenatedIrrepsTensorProduct 模块的前向传播。

        Args:
            input_tensors1_list (List[torch.Tensor]): 第一个输入的张量列表。
            input_tensor2 (torch.Tensor): 第二个输入的张量。
            scalars (torch.Tensor): 用于生成权重的标量输入。

        Returns:
            torch.Tensor: 处理后的输出张量。
        """
        input_tensor1 = self.fuse_in(torch.stack(input_tensors1_list, dim=-2))

        # 使用标量MLP生成权重
        weights = self.weight_generator(scalars)

        # 计算张量积
        output = self.tensor_product(input_tensor1, input_tensor2)
        output = self.linear_scaler(output, weights)

        # 输出
        output = self.linear_out(output)

        return output

@compile_mode("script")
class MessagePackBlock(nn.Module):
    """
    一个消息打包模块，它将节点特征、边特征和局部环境信息组合起来生成消息。
    """
    def __init__(
        self,
        irreps_node_feats: str,
        irreps_edge_feats: str,
        irreps_local_env_edge: str,
        irreps_out: str,
        irreps_edge_scalars: str,
        radial_MLP: List[int] = [64, 64],
        use_kan: bool = False
    ):
        """
        初始化 MessagePackBlock 模块。

        Args:
            irreps_node_feats (str): 节点特征的不可约表示。
            irreps_edge_feats (str): 边特征的不可约表示。
            irreps_local_env_edge (str): 局部环境边的不可约表示。
            irreps_out (str): 输出的不可约表示。
            irreps_edge_scalars (str): 边标量的不可约表示。
            radial_MLP (List[int]): 径向多层感知机的层维度。
            use_kan (bool): 是否使用KAN生成权重的标志。
        """
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_local_env_edge = o3.Irreps(irreps_local_env_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        self.combined_node_irreps = scale_irreps(self.irreps_node_feats, 2)
        self.fuse_node = AttentionHeadsToVector(self.irreps_node_feats)

        # 计算中间不可约表示和指令
        self.mid_node_irreps, self.node_instructions = self._tp_out_irreps_with_instructions(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_edge_irreps, self.edge_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.irreps_out,
        )

        # 初始化张量积
        self.node_tensor_product = o3.TensorProduct(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.mid_node_irreps,
            instructions=self.node_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.edge_tensor_product = o3.TensorProduct(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.mid_edge_irreps,
            instructions=self.edge_instructions,
            internal_weights=True,
            shared_weights=True
        )

        # 初始化带权重的线性缩放层
        self.node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.edge_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_edge_irreps.simplify(),
            irreps_out=self.irreps_out
        )

        # 初始化权重生成器
        input_dim = self.irreps_edge_scalars.num_irreps
        self.node_weight_generator = self._initialize_weight_generator(input_dim, self.node_linear_scaler.weight_numel)
        self.edge_weight_generator = self._initialize_weight_generator(input_dim, self.edge_linear_scaler.weight_numel)

        # 线性输出层
        self.node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.edge_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        """
        计算张量积的输出不可约表示和指令。
        """
        trainable = True

        # 收集可能的不可约表示和它们的指令
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # 对张量积的输出不可约表示进行排序
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # 置换指令的输出索引以匹配排序后的不可约表示
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器模块。

        Args:
            input_dim (int): 权重生成器的输入维度。
            weight_numel (int): 权重向量中的元素数量。

        Returns:
            nn.Module: 初始化后的权重生成器模块。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, node_feats_src: torch.Tensor, 
                node_feats_dst: torch.Tensor, 
                edge_feats: torch.Tensor, 
                local_env_edge: torch.Tensor,
                edge_scalars: torch.Tensor):
        """
        前向传播。

        Args:
            node_feats_src (torch.Tensor): 源节点特征。
            node_feats_dst (torch.Tensor): 目标节点特征。
            edge_feats (torch.Tensor): 边特征。
            local_env_edge (torch.Tensor): 局部环境边特征。
            edge_scalars (torch.Tensor): 边标量。

        Returns:
            torch.Tensor: 生成的消息。
        """
        # 计算节点交互的张量积
        node_inter = self.fuse_node(torch.stack([node_feats_src, node_feats_dst], dim=-2))
        weights_node = self.node_weight_generator(edge_scalars)
        node_inter_up = self.node_tensor_product(node_inter, local_env_edge)
        node_inter_dn = self.node_linear_scaler(node_inter_up, weights_node)
        
        # 计算边特征的张量积
        weights_edge = self.edge_weight_generator(edge_scalars)
        edge_feats_up = self.edge_tensor_product(edge_feats, local_env_edge)
        edge_feats_dn = self.edge_linear_scaler(edge_feats_up, weights_edge)        

        # 输出
        output = self.node_linear_out(node_inter_dn) + self.edge_linear_out(edge_feats_dn)

        return output

@compile_mode("script")
class MessagePackBlockV2(nn.Module):
    """
    MessagePackBlock 的一个变体，增加了节点-节点交互项。
    """
    def __init__(
        self,
        irreps_node_feats: str,
        irreps_edge_feats: str,
        irreps_local_env_edge: str,
        irreps_out: str,
        irreps_edge_scalars: str,
        radial_MLP: List[int] = [64, 64],
        use_kan: bool = False
    ):
        """
        初始化 MessagePackBlockV2 模块。

        Args:
            irreps_node_feats (str): 节点特征的不可约表示。
            irreps_edge_feats (str): 边特征的不可约表示。
            irreps_local_env_edge (str): 局部环境边的不可约表示。
            irreps_out (str): 输出的不可约表示。
            irreps_edge_scalars (str): 边标量的不可约表示。
            radial_MLP (List[int]): 径向多层感知机的层维度。
            use_kan (bool): 是否使用KAN生成权重的标志。
        """
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_local_env_edge = o3.Irreps(irreps_local_env_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        self.combined_node_irreps = scale_irreps(self.irreps_node_feats, 2)
        self.fuse_node = AttentionHeadsToVector(self.irreps_node_feats)

        # 计算中间不可约表示和指令
        self.mid_node_irreps, self.node_instructions = self._tp_out_irreps_with_instructions(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_edge_irreps, self.edge_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_node_node_irreps, self.node_node_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_node_feats,
            self.irreps_node_feats,
            self.irreps_out,
            mode='uvu'
        )

        # 初始化张量积
        self.node_tensor_product = o3.TensorProduct(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.mid_node_irreps,
            instructions=self.node_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.edge_tensor_product = o3.TensorProduct(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.mid_edge_irreps,
            instructions=self.edge_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.node_node_tensor_product = o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_node_feats,
            self.mid_node_node_irreps,
            instructions=self.node_node_instructions,
            internal_weights=True,
            shared_weights=True
        )

        # 初始化带权重的线性缩放层
        self.node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.edge_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_edge_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.node_node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )

        # 初始化权重生成器
        input_dim = self.irreps_edge_scalars.num_irreps
        self.node_weight_generator = self._initialize_weight_generator(input_dim, self.node_linear_scaler.weight_numel)
        self.edge_weight_generator = self._initialize_weight_generator(input_dim, self.edge_linear_scaler.weight_numel)
        self.node_node_weight_generator = self._initialize_weight_generator(input_dim, self.node_node_linear_scaler.weight_numel)

        # 线性输出层
        self.node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.edge_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.node_node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps, mode: str='uvw'
    ) -> Tuple[o3.Irreps, List]:
        """
        计算张量积的输出不可约表示和指令。
        """
        trainable = True

        # 收集可能的不可约表示和它们的指令
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (mul_i, ir_in) in enumerate(irreps1):
            for j, (mul_j, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        if mode=='uvw':
                            irreps_out_list.append((mul, ir_out))
                        elif mode=='uvu':
                            irreps_out_list.append((mul_i, ir_out))
                        else:
                            raise NotImplementedError
                        instructions.append((i, j, k, mode, trainable))

        # 对张量积的输出不可约表示进行排序
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # 置换指令的输出索引以匹配排序后的不可约表示
        instructions = [
            (i_in1, i_in2, permut[i_out], m, train)
            for i_in1, i_in2, i_out, m, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器模块。

        Args:
            input_dim (int): 权重生成器的输入维度。
            weight_numel (int): 权重向量中的元素数量。

        Returns:
            nn.Module: 初始化后的权重生成器模块。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, node_feats_src: torch.Tensor, 
                node_feats_dst: torch.Tensor, 
                edge_feats: torch.Tensor, 
                local_env_edge: torch.Tensor,
                edge_scalars: torch.Tensor):
        """
        前向传播。

        Args:
            node_feats_src (torch.Tensor): 源节点特征。
            node_feats_dst (torch.Tensor): 目标节点特征。
            edge_feats (torch.Tensor): 边特征。
            local_env_edge (torch.Tensor): 局部环境边特征。
            edge_scalars (torch.Tensor): 边标量。

        Returns:
            torch.Tensor: 生成的消息。
        """
        # 计算节点交互的张量积
        node_inter = self.fuse_node(torch.stack([node_feats_src, node_feats_dst], dim=-2))
        weights_node = self.node_weight_generator(edge_scalars)
        node_inter_up = self.node_tensor_product(node_inter, local_env_edge)
        node_inter_dn = self.node_linear_scaler(node_inter_up, weights_node)
        
        # 节点-节点张量积
        weights_node_node = self.node_node_weight_generator(edge_scalars)
        node_node_inter_up = self.node_node_tensor_product(node_feats_dst, node_feats_src)
        node_node_inter_dn = self.node_node_linear_scaler(node_node_inter_up, weights_node_node)
        
        # 计算边特征的张量积
        weights_edge = self.edge_weight_generator(edge_scalars)
        edge_feats_up = self.edge_tensor_product(edge_feats, local_env_edge)
        edge_feats_dn = self.edge_linear_scaler(edge_feats_up, weights_edge)        

        # 输出
        output = self.node_linear_out(node_inter_dn) + self.edge_linear_out(edge_feats_dn) + self.node_node_linear_out(node_node_inter_dn)

        return output


@torch.jit.script
def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """JIT-compatible version of ShiftedSoftPlus."""
    # Use torch.log and torch.tensor instead of math.log
    return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))


@torch.jit.script
def abs_activation(x: torch.Tensor) -> torch.Tensor:
    """JIT-compatible wrapper for torch.abs."""
    return torch.abs(x)


acts = {
    "abs": abs_activation,
    "tanh": torch.nn.functional.tanh,
    "ssp": shifted_softplus,
    "silu": torch.nn.functional.silu,
}

def irreps2gate(
    irreps: o3.Irreps,
    nonlinearity_scalars: Dict[int, str] = {1: "ssp", -1: "tanh"},
    nonlinearity_gates: Dict[int, str] = {1: "ssp", -1: "abs"},
) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps, List[Callable], List[Callable]]:
    """
    将不可约表示(irreps)分解为标量和门控(gated)组件，并关联相应的激活函数。

    Args:
        irreps (o3.Irreps): 输入的不可约表示。
        nonlinearity_scalars (Dict[int, str]): 标量组件的激活函数字典，键为宇称(parity)。
        nonlinearity_gates (Dict[int, str]): 门控组件的激活函数字典，键为宇称(parity)。

    Returns:
        Tuple[o3.Irreps, o3.Irreps, o3.Irreps, List[Callable], List[Callable]]:
            一个元组，包含：
            - ``irreps_scalars`` (o3.Irreps): 标量不可约表示。
            - ``irreps_gates`` (o3.Irreps): 门控不可约表示。
            - ``irreps_gated`` (o3.Irreps): 门控后的不可约表示。
            - ``act_scalars`` (List[Callable]): 标量激活函数列表。
            - ``act_gates`` (List[Callable]): 门控激活函数列表。
    """
    # 将irreps分解为标量和门控组件
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0]).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l != 0]).simplify()

    # 根据门控组件的存在确定门(gate)的不可约表示
    irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify() if irreps_gated.dim > 0 else o3.Irreps([])

    # 获取标量和门的激活函数
    act_scalars = [acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars]
    act_gates = [acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates]

    return irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates

def scale_irreps(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    """
    按给定因子缩放不可约表示的多重度，确保多重度至少为1。

    Args:
        irreps (o3.Irreps): 输入的不可约表示。
        factor (float): 缩放因子。

    Returns:
        o3.Irreps: 缩放后的不可约表示。
    """
    return o3.Irreps([(max(1, int(mul * factor)), ir) for mul, ir in irreps])

def filter_and_split_irreps(irreps: o3.Irreps, num_channels: int, min_l: int, max_l: int) -> o3.Irreps:
    """
    根据指定的角动量(l)范围过滤和分割不可约表示。

    Args:
        irreps (o3.Irreps): 输入的不可约表示。
        num_channels (int): 用于分割多重度的通道数。
        min_l (int): 最小角动量（包含）。
        max_l (int): 最大角动量（包含）。

    Returns:
        o3.Irreps: 过滤和分割后的不可约表示。
    """
    result_irreps = o3.Irreps()
    for multiplicity, irrep in irreps:
        if irrep.l < min_l or irrep.l > max_l:
            # 保留指定l范围之外的irreps
            result_irreps += o3.Irreps([(multiplicity, irrep)])
        else:
            # 对在范围内的irreps按num_channels分割多重度
            split_multiplicity = multiplicity // num_channels
            if split_multiplicity > 0:
                result_irreps += split_multiplicity * o3.Irreps([(num_channels, irrep)])
    
    return result_irreps

@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    """
    使用指定的径向基组对边长度进行编码。

    Attributes:
        out_field (str): 存储编码后边特征的字典键。
    """

    def __init__(
        self,
        basis=None,
        cutoff=None,
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        """
        初始化 RadialBasisEdgeEncoding 模块。

        Args:
            basis: 用于编码的径向基函数。
            cutoff: 截断函数。
            out_field (str): 编码边的输出字段键。
            irreps_in: 输入的不可约表示。
        """
        super().__init__()
        self.basis = basis
        self.cutoff = cutoff
        self.out_field = out_field

        # 根据基函数类型确定基函数的数量
        basis_type = type(basis).__name__.split(".")[-1]
        if basis_type in {'BesselBasis', 'GaussianSmearing'}:
            num_basis = basis.freqs.size(0) if basis_type == 'BesselBasis' else basis.offset.size(0)
        elif basis_type in {
            'ExponentialGaussianRadialBasisFunctions',
            'ExponentialBernsteinRadialBasisFunctions',
            'GaussianRadialBasisFunctions',
            'BernsteinRadialBasisFunctions'
        }:
            num_basis = basis.num_basis_functions
        else:
            raise NotImplementedError(f"Basis type {basis_type} is not supported.")

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(num_basis, (0, 1))])},
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算边编码并更新数据字典。

        Args:
            data (AtomicDataDict.Type): 包含图数据的字典。
            
        Returns:
            AtomicDataDict.Type: 带有编码后边特征的更新图数据。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        nbr_shift = data['nbr_shift']
        pos = data['pos']

        # 计算边向量和边长
        edge_dir = (pos[i] + nbr_shift) - pos[j]
        edge_length = torch.norm(edge_dir, p=2, dim=-1)

        # 更新数据字典中的边向量和边长
        data["edge_vectors"] = edge_dir/edge_length[:,None]
        data["edge_lengths"] = edge_length

        # 将径向基应用于边长
        edge_length_embedded = self.basis(edge_length)
        
        if self.cutoff is not None:
            edge_length_embedded = edge_length_embedded*self.cutoff(edge_length)[:, None]
            
        data[self.out_field] = edge_length_embedded

        return data

@compile_mode("script")
class VectorToAttentionHeads(nn.Module):
    """将形状为 :math:`[N, D_{mid}]` 的向量重塑为 :math:`[N, H, D_{head}]` 的多头形式。

    其中 :math:`H` 是注意力头的数量 (num_heads)，:math:`D_{mid}` 是中间特征维度，
    :math:`D_{head}` 是每个头的特征维度。

    Attributes:
        num_heads (int): 注意力头的数量。
        irreps_head (o3.Irreps): 每个头的不可约表示。
        irreps_mid_in (o3.Irreps): 中间输入的不可约表示。
        mid_in_indices (List[Tuple[int, int]]): 用于重塑的索引。
    """

    def __init__(self, irreps_head: o3.Irreps, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = o3.Irreps([(mul * num_heads, ir) for mul, ir in irreps_head])
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 重塑后的张量。
        """
        N, _ = x.shape
        reshaped_tensors = [
            x.narrow(1, start_idx, end_idx - start_idx).view(N, self.num_heads, -1)
            for start_idx, end_idx in self.mid_in_indices
        ]
        return torch.cat(reshaped_tensors, dim=2)

    def __repr__(self):
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head}, num_heads={self.num_heads})'

@compile_mode("script")
class AttentionHeadsToVector(nn.Module):
    """将 :math:`[N, H, D_{head}]` 的多头注意力向量转换回 :math:`[N, D_{flat}]` 的扁平向量。

    其中 :math:`H` 是注意力头的数量, :math:`D_{head}` 是每个头的维度, 
    :math:`D_{flat}` 是扁平化后的总维度。
    
    Attributes:
        irreps_head (o3.Irreps): 定义注意力头结构的不可约表示列表。
        head_sizes (List[int]): 从不可约表示推导出的每个注意力头的大小列表。
    """

    def __init__(self, irreps_head: o3.Irreps):
        """
        初始化 AttentionHeadsToVector 模块。

        Args:
            irreps_head (o3.Irreps): 用于定义注意力头结构的不可约表示列表。
        """
        super().__init__()
        self.irreps_head = irreps_head

        # 根据irreps定义计算每个注意力头的大小
        self.head_sizes = [multiplicity * irrep.dim for multiplicity, irrep in self.irreps_head]

    def __repr__(self):
        """
        提供模块的字符串表示形式，用于调试。

        Returns:
            str: AttentionHeadsToVector 实例的字符串表示。
        """
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，处理注意力头并将其扁平化为单个向量。

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, num_heads, input_dim)，其中N为批量大小，num_heads为注意力头数，input_dim为所有头的总维度。

        Returns:
            torch.Tensor: 输出张量，形状为 (N, flattened_dim)，其中flattened_dim为所有头的总维度。

        Raises:
            ValueError: 如果 `head_sizes` 的总和与输入张量的 `input_dim` 不匹配。
        """
        # 提取输入张量的维度
        batch_size, num_heads, input_dim = x.shape

        # 确保所有注意力头的总大小与输入张量的最后一个维度匹配
        if sum(self.head_sizes) != input_dim:
            raise ValueError(
                f"The sum of head_sizes ({sum(self.head_sizes)}) does not match the input_dim ({input_dim}) "
                "of the input tensor."
            )

        # 根据 head_sizes 沿最后一个维度分割输入张量
        split_tensors = torch.split(x, self.head_sizes, dim=2)

        # 重塑每个分割后的张量，将注意力头扁平化为每个批次的单个向量，使用contiguous()确保内存连续性
        flattened_tensors = [sub_tensor.contiguous().view(batch_size, -1) for sub_tensor in split_tensors]

        # 沿最后一个维度连接扁平化的张量以产生输出
        return torch.cat(flattened_tensors, dim=1)

@compile_mode("script")
class ConvBlockE3(nn.Module):
    """
    使用张量积处理节点特征的等变卷积块，支持跳跃连接(skip-connections)。
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        radial_MLP: Optional[list] = None,
        use_skip_connections: bool = True,
        use_kan: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: dict = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: dict = {"e": "ssp", "o": "abs"},
    ):
        """
        初始化 ConvBlockE3 模块。

        Args:
            irreps_in (o3.Irreps): 输入不可约表示。
            irreps_out (o3.Irreps): 输出不可约表示。
            irreps_node_attrs (o3.Irreps): 节点属性不可约表示。
            irreps_edge_attrs (o3.Irreps): 边属性不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入不可约表示。
            radial_MLP (Optional[List[int]]): 径向嵌入的多层感知机架构。
            use_skip_connections (bool): 是否使用跳跃连接。
            use_kan (bool): 是否使用 KAN 模块生成权重。
            nonlinearity_type (str): 使用的非线性类型 ("gate" 或 "norm")。
            nonlinearity_scalars (Dict[str, str]): 标量通道的非线性函数名。
            nonlinearity_gates (Dict[str, str]): 门控通道的非线性函数名。
        """
        super().__init__()

        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan
        self.use_skip_connections = use_skip_connections

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # 转换非线性映射
        scalar_nonlinearities = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        gate_nonlinearities = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)

        # 用于处理特征的残差块
        self.residual = ResidualBlock(self.irreps_in, self.irreps_out)

        # 卷积层       
        self.conv_tp = MessagePackBlock(
            irreps_node_feats=self.irreps_in,
            irreps_edge_feats=self.irreps_in,
            irreps_local_env_edge=self.irreps_edge_attrs,
            irreps_out=self.irreps_out,
            irreps_edge_scalars=self.irreps_edge_embed, 
            radial_MLP=self.radial_MLP, 
            use_kan=self.use_kan
            )
        
        # 跳跃连接层
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(self.irreps_in, self.irreps_out)

    def create_linear(self, irreps_in, irreps_out=None):
        """
        创建线性层。

        Args:
            irreps_in (o3.Irreps): 线性层的输入不可约表示。
            irreps_out (o3.Irreps, optional): 线性层的输出不可约表示。

        Returns:
            o3.Linear: 线性变换层。
        """
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        卷积块的前向传播。

        Args:
            data (dict): 包含图数据的字典。

        Returns:
            torch.Tensor: 更新后的节点特征。
        """
        edge_index = data["edge_index"]
        sender = edge_index[0]
        receiver = edge_index[1]
        node_features = data["node_features"]
        edge_embedding = data["edge_embedding"]
        edge_attributes = data["edge_attrs"]
        num_nodes = len(data["node_features"])

        # 跳跃连接
        skip_connection = self.skip_linear(node_features) if self.use_skip_connections else None
        
        # 消息        
        messages = self.conv_tp(
            node_features[sender], 
            node_features[receiver],  
            data["edge_features"], 
            edge_attributes,
            edge_embedding
        )

        # 聚合消息
        aggregated_messages = scatter(
            src=messages, index=receiver, dim=0, dim_size=num_nodes
        )
        
        # 应用残差块
        output_features = self.residual(aggregated_messages)

        # 如果使用，则应用跳跃连接
        if self.use_skip_connections and skip_connection is not None:
            output_features = output_features + skip_connection

        data["node_features"] = output_features
        
        return output_features

@compile_mode("script")
class AttentionAggregationV2(nn.Module):
    """
    一个等变注意力(equivariant attention)聚合模块。
    它根据外部计算的注意力权重来聚合值(value)向量。
    """

    def __init__(
        self,
        num_heads: int, 
        irreps_value: o3.Irreps, 
    ):
        """
        初始化 AttentionAggregationV2 模块。

        Args:
            num_heads (int): 注意力头的数量。
            irreps_value (o3.Irreps): 值(value)向量的不可约表示。
        """
        super().__init__()
        self.num_heads = num_heads
        irreps_value = o3.Irreps(irreps_value)
        
        self.value_irreps_head = scale_irreps(irreps_value, 1/num_heads)
        self.unfuse_value = VectorToAttentionHeads(self.value_irreps_head, num_heads)
        self.fuse_value = AttentionHeadsToVector(self.value_irreps_head)
    
    def forward(
        self, 
        value,
        edge_weights: torch.Tensor,  # (num_edges, num_heads)
        edge_weights_cutoff: torch.Tensor, # (num_edges,)
        edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """
        注意力机制的前向传播。

        Args:
            value (torch.Tensor): 值向量。
            edge_weights (torch.Tensor): 边的注意力权重，形状为 (num_edges, num_heads)。
            edge_weights_cutoff (torch.Tensor): 边的截断权重，形状为 (num_edges, )。
            edge_index (torch.LongTensor): 边索引。

        Returns:
            torch.Tensor: 注意力聚合后的输出向量。
        """
        value = self.unfuse_value(value)
        
        edgr_src = edge_index[0]
        edge_dst = edge_index[1]
        
        # 计算每个边的注意力权重
        if edge_weights_cutoff is not None:
            edge_weights = edge_weights_cutoff[:, None] * edge_weights  # (num_edges, num_heads)
        edge_weights = edge_softmax(edge_weights, edge_dst)  # (num_edges, num_heads)
        edge_weights = edge_weights.unsqueeze(-1)  # (num_edges, num_heads, 1)

        # Compute the attended outputs per node
        f_out = scatter(edge_weights * value, edge_dst, dim=0)  # (num_nodes, num_heads, irreps_head)
        f_out = self.fuse_value(f_out)  # Merge heads
        return f_out

@compile_mode("script")
class AttentionAggregation(nn.Module):
    r"""处理键(key)、值(value)和查询(query)向量的等变注意力机制。

    该模块在图的边上应用缩放点积注意力机制。注意力权重 :math:`\alpha_{ij}` 的计算方式如下：

    .. math::

        \alpha_{ij} = \frac{(Q_i \cdot K_j)}{\sqrt{d_k}}

    其中 :math:`Q_i` 是目标节点的查询向量，:math:`K_j` 是源节点的键向量，:math:`d_k` 是键向量的维度。
    然后使用 softmax 对权重进行归一化，并用于加权聚合值向量 :math:`V_j`。
    """

    def __init__(
        self, 
        num_heads: int, 
        irreps_key: o3.Irreps, 
        irreps_value: o3.Irreps, 
        irreps_query: o3.Irreps
    ):
        """
        初始化 AttentionAggregation 模块。

        Args:
            num_heads (int): 注意力头的数量。
            irreps_key (o3.Irreps): 键(key)向量的不可约表示。
            irreps_value (o3.Irreps): 值(value)向量的不可约表示。
            irreps_query (o3.Irreps): 查询(query)向量的不可约表示。
        """
        super().__init__()
        self.num_heads = num_heads
        self.irreps_key = o3.Irreps(irreps_key)
        irreps_value = o3.Irreps(irreps_value)
        irreps_query = o3.Irreps(irreps_query)
        
        self.key_irreps_head = scale_irreps(irreps_key, 1/num_heads)
        self.value_irreps_head = scale_irreps(irreps_value, 1/num_heads)
        self.query_irreps_head = scale_irreps(irreps_query, 1/num_heads)
        
        # Pre-compute dim for TorchScript compatibility
        self.key_irreps_head_dim = self.key_irreps_head.dim
        
        self.unfuse_key = VectorToAttentionHeads(self.key_irreps_head, num_heads)
        self.unfuse_value = VectorToAttentionHeads(self.value_irreps_head, num_heads)
        self.unfuse_query = VectorToAttentionHeads(self.query_irreps_head, num_heads)
        
        self.fuse_value = AttentionHeadsToVector(self.value_irreps_head)
    
    def forward(
        self, 
        key: torch.Tensor,  
        value: torch.Tensor, 
        query: torch.Tensor,  
        edge_weight_cutoff: torch.Tensor, 
        edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """
        注意力机制的前向传播。

        Args:
            key (torch.Tensor): 键向量，形状为 :math:`(N_{edges}, d_{hidden})`。
            value (torch.Tensor): 值向量，形状为 :math:`(N_{edges}, d_{hidden})`。
            query (torch.Tensor): 查询向量，形状为 :math:`(N_{edges}, d_{hidden})`。
            edge_weight_cutoff (torch.Tensor): 边的截断权重，形状为 :math:`(N_{edges},)`。
            edge_index (torch.LongTensor): 边索引。

        Returns:
            torch.Tensor: 注意力输出向量。
        """
        key = self.unfuse_key(key)
        value = self.unfuse_value(value)
        query = self.unfuse_query(query)
        
        edgr_src = edge_index[0]
        edge_dst = edge_index[1]
        
        # 计算每个边的注意力权重
        edge_weights = (query * key).sum(-1)  # (num_edges, num_heads)
        if edge_weight_cutoff is not None:
            edge_weights = edge_weight_cutoff[:, None] * edge_weights  # (num_edges, num_heads)
        # TorchScript workaround: use pre-computed dim value
        edge_weights = edge_weights / math.sqrt(self.key_irreps_head_dim)
        edge_weights = edge_softmax(edge_weights, edge_dst)  # (num_edges, num_heads)
        edge_weights = edge_weights.unsqueeze(-1)  # (num_edges, num_heads, 1)

        # 计算每个节点的加权输出
        f_out = scatter(edge_weights * value, edge_dst, dim=0)  # (num_nodes, num_heads, irreps_head)
        f_out = self.fuse_value(f_out)  # 合并多头
        return f_out

@compile_mode("script")
class AttentionBlockE3(nn.Module):
    """
    使用注意力机制(attention mechanism)处理图数据的等变注意力块。
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        num_heads: int,
        max_radius: float,
        radial_MLP: Optional[List[int]] = None,
        use_skip_connections: bool = True,
        use_kan: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        """
        初始化 AttentionBlockE3 模块。

        Args:
            irreps_in (o3.Irreps): 输入不可约表示。
            irreps_out (o3.Irreps): 输出不可约表示。
            irreps_node_attrs (o3.Irreps): 节点属性不可约表示。
            irreps_edge_feats (o3.Irreps): 边特征不可约表示。
            irreps_edge_attrs (o3.Irreps): 边属性不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入不可约表示。
            num_heads (int): 注意力头的数量。
            max_radius (float): 边截断的最大半径。
            radial_MLP (Optional[List[int]]): 径向多层感知机的架构。
            use_skip_connections (bool): 是否使用跳跃连接。
            use_kan (bool): 是否在径向多层感知机中使用 KAN。
            nonlinearity_type (str): 非线性类型 ('gate' 或 'norm')。
            nonlinearity_scalars (Dict[int, Callable]): 标量非线性函数名。
            nonlinearity_gates (Dict[int, Callable]): 门控非线性函数名。
        """
        super().__init__()
        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan
        self.use_skip_connections = use_skip_connections

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # 转换非线性映射
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # 分配不可约表示
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        self.register_buffer(
            "max_radius", torch.tensor(max_radius, dtype=torch.get_default_dtype())
        )
        self.cutoff_func = SoftUnitStepCutoff(cutoff=max_radius)
        
        # 线性变换
        self.linear_up_src = self.create_linear(self.irreps_in)
        self.linear_up_tar = self.create_linear(self.irreps_in)
        self.linear_up_edge = self.create_linear(self.irreps_in)

        # 非线性
        self.residual = ResidualBlock(self.irreps_in, self.irreps_out)

        # 为值(value)创建张量积      
        self.conv_tp_value = MessagePackBlock(irreps_node_feats=self.irreps_in,
                                            irreps_edge_feats=self.irreps_edge_feats,
                                            irreps_local_env_edge=self.irreps_edge_attrs,
                                            irreps_out=self.irreps_out,
                                            irreps_edge_scalars=self.irreps_edge_embed,
                                            radial_MLP=self.radial_MLP,
                                            use_kan=self.use_kan)
        
        # 键、查询和值的线性层
        self.linear_key = self.create_linear(self.irreps_in, self.irreps_in)
        self.linear_query = self.create_linear(self.irreps_in, self.irreps_in)

        # 注意力机制
        self.attention = AttentionAggregation(
            num_heads=num_heads,
            irreps_key=self.irreps_in,
            irreps_value=self.irreps_in,
            irreps_query=self.irreps_in,
        )
        
        # 跳跃连接
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(self.irreps_in, self.irreps_out)

    def create_linear(self, irreps_in, irreps_out=None):
        """创建线性层。"""
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def create_tensor_product(self, irreps_mid, instructions):
        """创建张量积层。"""
        return o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge_attrs,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    def init_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def create_nonlinearity(self, nonlinearity_type, nonlinearity_scalars, nonlinearity_gates):
        """
        创建非线性模块。
        """
        if nonlinearity_type == "gate":
            irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates = irreps2gate(
                self.irreps_in, nonlinearity_scalars, nonlinearity_gates
            )
            return Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
        return NormActivation(
            irreps_in=self.irreps_in,
            scalar_nonlinearity=acts[nonlinearity_scalars[1]],
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        注意力块的前向传播。

        Args:
            data (Dict[str, torch.Tensor]): 包含图数据的字典。

        Returns:
        - torch.Tensor: 更新后的节点特征。
        """
        edge_index = data["edge_index"]
        sender = edge_index[0]
        receiver = edge_index[1]
        node_feats = data["node_features"]
        edge_embed = data["edge_embedding"]
        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_features"]
        
        # 跳跃连接
        sc = self.skip_linear(node_feats) if self.use_skip_connections else None

        # 处理键、查询和值
        key = self.linear_key(node_feats)[sender]
        query = self.linear_key(node_feats)[receiver]
        
        value = self.conv_tp_value(self.linear_up_src(node_feats)[sender], 
                                   self.linear_up_tar(node_feats)[receiver],  
                                   self.linear_up_edge(edge_feats),
                                   edge_attrs, 
                                   edge_embed)

        # 注意力机制 
        edge_weight_cutoff = self.cutoff_func(data["edge_lengths"])
        node_feats = self.attention(key, value, query, edge_weight_cutoff, edge_index=data["edge_index"])

        # 应用残差块
        node_feats = self.residual(node_feats)

        # 如果使用，则应用跳跃连接
        if self.use_skip_connections and sc is not None:
            node_feats = node_feats + sc  

        data["node_features"] = node_feats

        return node_feats

@compile_mode("script")
class PairInteractionEmbeddingBlock(nn.Module):
    """
    基于节点特征和边属性更新边特征的对交互嵌入块。
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        use_kan: bool = False,
        radial_MLP: Optional[List[int]] = None,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[str, str] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[str, str] = {"e": "ssp", "o": "abs"},
    ) -> None:
        """
        初始化 PairInteractionEmbeddingBlock 模块。

        Args:
            irreps_node_feats (o3.Irreps): 节点特征的不可约表示。
            irreps_edge_attrs (o3.Irreps): 边属性的不可约表示。
            irreps_node_attrs (o3.Irreps): 节点属性的不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入的不可约表示。
            irreps_edge_feats (o3.Irreps): 边特征的不可约表示。
            use_kan (bool): 是否在径向MLP中使用KAN。
            radial_MLP (Optional[List[int]]): 径向MLP的架构。
            nonlinearity_type (str): 使用的非线性类型 ("gate" 或 "norm")。
            nonlinearity_scalars (Dict[str, str]): 标量通道的非线性函数名。
            nonlinearity_gates (Dict[str, str]): 门控通道的非线性函数名。
        """
        super().__init__()
        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan

        # 分配不可约表示
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # 转换非线性映射
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # 用于提升(lifting)节点特征的线性层
        self.linear_up_src = self.create_linear(self.irreps_node_feats)
        self.linear_up_dst = self.create_linear(self.irreps_node_feats)

        # 用于边特征混合的张量积层
        self.conv_tp = TensorProductWithMemoryOptimizationWithWeight(irreps_input_1=self.irreps_node_feats, 
                                                                      irreps_input_2=self.irreps_edge_attrs, 
                                                                      irreps_out=self.irreps_edge_feats, 
                                                                      irreps_scalar=self.irreps_edge_embed, 
                                                                      radial_MLP=self.radial_MLP, 
                                                                      use_kan=self.use_kan)

    def create_linear(self, irreps_in, irreps_out=None):
        """创建线性层。"""
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def create_tensor_product(self, irreps_mid, instructions):
        """创建张量积层。"""
        return o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_edge_attrs,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    def init_weight_generator(self, input_dim, weight_numel):
        """
        初始化权重生成器。
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        对交互块的前向传播。

        Args:
            data (Dict[str, torch.Tensor]): 包含图数据的字典。

        Returns:
            torch.Tensor: 更新后的边特征。
        """
        edge_index = data["edge_index"]
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        node_feats = data["node_features"]
        edge_embed = data["edge_embedding"]
        edge_attributes = data["edge_attrs"]
        
        node_feats_src = self.linear_up_src(node_feats[edge_src])
        node_feats_dst = self.linear_up_dst(node_feats[edge_dst])

        # 混合节点特征以生成边特征
        edge_feats_mix_tp = self.conv_tp(
            node_feats_src + node_feats_dst, edge_attributes, edge_embed
        )

        data["edge_features"] = edge_feats_mix_tp
        return edge_feats_mix_tp

@compile_mode("script")
class PairInteractionBlock(nn.Module):
    """
    一个基于节点特征和边属性更新边特征的对交互块。
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        use_skip_connections: bool = False,
        use_kan: bool = False,
        radial_MLP: Optional[List[int]] = None,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[str, str] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[str, str] = {"e": "ssp", "o": "abs"},
    ) -> None:
        """
        初始化 PairInteractionBlock 模块。

        Args:
            irreps_node_feats (o3.Irreps): 节点特征的不可约表示。
            irreps_node_attrs (o3.Irreps): 节点属性的不可约表示。
            irreps_edge_attrs (o3.Irreps): 边属性的不可约表示。
            irreps_edge_embed (o3.Irreps): 边嵌入的不可约表示。
            irreps_edge_feats (o3.Irreps): 边特征的不可约表示。
            use_skip_connections (bool): 是否使用跳跃连接。
            use_kan (bool): 是否在径向MLP中使用KAN。
            radial_MLP (Optional[List[int]]): 径向MLP的架构。
            nonlinearity_type (str): 使用的非线性类型 ("gate" 或 "norm")。
            nonlinearity_scalars (Dict[str, str]): 标量通道的非线性函数名。
            nonlinearity_gates (Dict[str, str]): 门控通道的非线性函数名。
        """
        super().__init__()

        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_skip_connections = use_skip_connections
        self.use_kan = use_kan

        # 分配不可约表示
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # 转换非线性映射
        scalar_nonlinearities = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        gate_nonlinearities = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # 线性变换
        self.linear_up_src = self.create_linear(self.irreps_node_feats)
        self.linear_up_tar = self.create_linear(self.irreps_node_feats)

        # 用于边特征混合的张量积层
        self.conv_tp = MessagePackBlock(
            irreps_node_feats=self.irreps_node_feats,
            irreps_edge_feats=self.irreps_edge_feats,
            irreps_local_env_edge=self.irreps_edge_attrs,
            irreps_out=self.irreps_edge_feats,
            irreps_edge_scalars=self.irreps_edge_embed, 
            radial_MLP=self.radial_MLP, 
            use_kan=self.use_kan
            )

        # 跳跃连接
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(irreps_edge_feats, irreps_edge_feats)

    def create_linear(self, irreps_in, irreps_out=None):
        """
        创建线性层。

        Args:
            irreps_in (o3.Irreps): 线性层的输入不可约表示。
            irreps_out (o3.Irreps, optional): 线性层的输出不可约表示。

        Returns:
            o3.Linear: 一个线性变换层。
        """
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        对交互块的前向传播。

        Args:
            data (Dict[str, torch.Tensor]): 包含图数据的字典。

        Returns:
            torch.Tensor: 更新后的边特征。
        """
        edge_index = data["edge_index"]
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        node_feats = data["node_features"]
        edge_embed = data["edge_embedding"]
        edge_feats = data["edge_features"]

        # 混合节点特征以生成边特征       
        edge_feats_mix = self.conv_tp(
            self.linear_up_src(node_feats)[edge_src], 
            self.linear_up_tar(node_feats)[edge_dst], 
            edge_feats, 
            data["edge_attrs"], 
            edge_embed
        )
        
        if self.use_skip_connections and hasattr(self, 'skip_linear'):
            skip_feats = self.skip_linear(edge_feats)  
            edge_feats = edge_feats_mix + skip_feats
        # else:
        #     edge_feats = edge_feats_mix

        data["edge_features"] = edge_feats
        
        return edge_feats
    

@compile_mode("script")
class CorrProductBlock(nn.Module):
    """
    一个使用等变乘积操作更新节点特征的相关性乘积块。
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        num_hidden_features: int,
        correlation: int,
        use_skip_connections: bool = True,
        num_elements: Optional[int] = None
    ) -> None:
        """
        初始化 CorrProductBlock 模块。

        Args:
            irreps_node_feats (o3.Irreps): 节点特征的不可约表示。
            num_hidden_features (int): 隐藏特征的数量。
            correlation (int): 乘积操作的相关性阶数。
            use_skip_connections (bool): 是否使用跳跃连接。
            num_elements (Optional[int]): 用于乘积操作的元素数量。
        """
        super().__init__()

        self.irreps_node_feats = o3.Irreps(irreps_node_feats).simplify()
        self.num_hidden_features = num_hidden_features
        self.correlation = correlation
        self.use_skip_connections = use_skip_connections
        self.num_elements = num_elements

        self.irreps_hidden_features = o3.Irreps(
            [(self.num_hidden_features, irrep.ir) for irrep in self.irreps_node_feats]
        )

        # 用于提升和跳跃连接的线性层
        self.linear_pre = o3.Linear(
            self.irreps_node_feats,
            self.irreps_hidden_features,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear_sc = o3.Linear(
            self.irreps_node_feats,
            self.irreps_node_feats,
            internal_weights=True,
            shared_weights=True,
        )

        # 等变乘积操作
        self.prod = EquivariantProductBasisBlock(
            node_feats_irreps=self.irreps_hidden_features,
            target_irreps=self.irreps_hidden_features,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=False,
        )

        # 用于输出的线性层
        self.linear_out = o3.Linear(
            self.irreps_hidden_features,
            self.irreps_node_feats,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.reshape = reshape_irreps(self.irreps_hidden_features)

    def forward(
        self,
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        相关性乘积块的前向传播。

        Args:
            data (Dict[str, torch.Tensor]): 包含图数据的字典。

        Returns:
            torch.Tensor: 更新后的节点特征。
        """
        node_feats = self.linear_pre(data["node_features"])
        node_feats = self.reshape(node_feats) # [n_nodes, channels, (l + 1)**2]

        out = self.prod(node_feats, None, data["node_attrs"])
        out = self.linear_out(out)

        if self.use_skip_connections:
            sc = self.linear_sc(data["node_features"])
            data["node_features"] = out + sc
        else:
            data["node_features"] = out

        return out

@compile_mode("script")
class ResidualBlock(nn.Module):
    """
    一个在等变神经网络中使用的残差块。
    """

    def __init__(
        self,
        irreps_in: str,
        feature_irreps_hidden: str,
        resnet: bool = True,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[str, str] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[str, str] = {"e": "ssp", "o": "abs"},
    ):
        """
        初始化 ResidualBlock 模块。
        
        Args:
            irreps_in (str): 输入的不可约表示 (irreps)。
            feature_irreps_hidden (str): 隐藏特征的不可约表示。
            resnet (bool): 如果为True，则应用残差连接。
            nonlinearity_type (str): 应用的非线性类型 ('gate' 或 'norm')。
            nonlinearity_scalars (Dict[str, str]): 用于标量特征的非线性函数字典，键为宇称。
            nonlinearity_gates (Dict[str, str]): 用于门控特征的非线性函数字典，键为宇称。
        """
        super().__init__()
        
        # 确保非线性类型有效
        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity_type. Choose either 'gate' or 'norm'."

        # 根据宇称转换标量和门控非线性
        nonlinearity_scalars = {1: nonlinearity_scalars["e"], -1: nonlinearity_scalars["o"]}
        nonlinearity_gates = {1: nonlinearity_gates["e"], -1: nonlinearity_gates["o"]}

        self.irreps_in = o3.Irreps(irreps_in)
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        
        self.equivariant_nonlin = self.create_nonlinearity(nonlinearity_type, self.feature_irreps_hidden, nonlinearity_scalars, nonlinearity_gates)
        
        # 定义线性层
        self.linear1 = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.equivariant_nonlin.irreps_in)
        self.linear2 = o3.Linear(irreps_in=self.equivariant_nonlin.irreps_out, irreps_out=irreps_in)

    def create_nonlinearity(self, nonlinearity_type, irreps_mid, nonlinearity_scalars, nonlinearity_gates):
        """创建非线性模块。"""
        if nonlinearity_type == "gate":
            irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates = irreps2gate(
                irreps_mid, nonlinearity_scalars, nonlinearity_gates
            )
            return Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
        return NormActivation(
            irreps_in=irreps_mid,
            scalar_nonlinearity=acts[nonlinearity_scalars[1]],
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

    def forward(self, x):
        """
        残差块的前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，其形状匹配 `irreps_in`。
        
        Returns:
            torch.Tensor: 输出张量，其形状匹配 `irreps_in`。
        """
        # 如果适用，存储旧输入以用于残差连接
        old_x = x
        
        # 应用第一个线性变换
        x = self.linear1(x)
        
        # 应用非线性
        x = self.equivariant_nonlin(x)
        
        # 应用第二个线性变换
        x = self.linear2(x)
        
        # 如果启用resnet，则应用残差连接
        if self.resnet:
            x = old_x + x
            
        return x

@compile_mode("script")
class HamLayer(nn.Module):
    """
    一个哈密顿层，由一个残差块和一个最终的线性变换组成。
    """
    def __init__(self, irreps_in, feature_irreps_hidden, irreps_out, nonlinearity_type: str = "gate", resnet: bool = True):
        """
        初始化 HamLayer 模块。

        Args:
            irreps_in (o3.Irreps): 输入的不可约表示。
            feature_irreps_hidden (o3.Irreps): 残差块中隐藏特征的不可约表示。
            irreps_out (o3.Irreps): 输出的不可约表示。
            nonlinearity_type (str): 要使用的非线性类型 ('gate' 或 'norm')。
            resnet (bool): 是否在残差块中使用残差连接。
        """
        super().__init__()
        
        # 定义残差块
        self.residual_block = ResidualBlock(irreps_in=irreps_in, 
                                            feature_irreps_hidden=feature_irreps_hidden, 
                                            nonlinearity_type=nonlinearity_type, 
                                            resnet=resnet)
        
        # 定义线性变换
        self.linear_transform = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    
    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 应用残差块
        x = self.residual_block(x)
        
        # 应用线性变换
        x = self.linear_transform(x)
        
        return x
