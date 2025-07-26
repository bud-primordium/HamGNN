"""这个模块定义了 HamGNN 的核心网络架构。

主要包含三个部分：
1.  `HamGNN_pre` / `HamGNN_pre2` / `HamGNN_pre_charge`: 这一系列是特征提取网络（或称为表示网络），
    负责从输入的原子结构（节点、边）中学习具有 O(3) 等变性的特征表示。它们基于 `nequip` 框架，
    通过多层消息传递（卷积）来构建节点和边的特征。
2.  `HamGNN_out`: 这是输出网络，它接收来自表示网络的等变特征，并利用这些特征构建
    等变的哈密顿量 (Hamiltonian) 和重叠矩阵 (Overlap Matrix)。
3.  辅助模块和类: 如 `residual_block`, `Edge_builder`, `Triplet_builder` 等，
    这些是构成上述核心网络的积木块。
"""

from numpy import zeros
import torch
import torch.nn as nn
from torch_geometric.data import Data, batch
from torch.nn import (Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, Identity)
from ..utils import linear_bn_act
from ..layers import denseRegression
from torch_scatter import scatter
import sympy as sym
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate, NormActivation
from easydict import EasyDict
from typing import Union
from ..layers import GaussianSmearing, cuttoff_envelope, CosineCutoff, BesselBasis, sph_harm_layer
from .nequip.data import AtomicDataDict, AtomicDataset
import math
from .clebsch_gordan import ClebschGordan
# from e3nn.o3._wigner import _so3_clebsch_gordan
import copy
from typing import Dict, Callable
from .nequip.nn.nonlinearities import ShiftedSoftPlus
from .kpoint_gen import kpoints_generator
import numpy as np
from pymatgen.core.periodic_table import Element
from torch_sparse import SparseTensor
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from ..e3_layers import e3TensorDecomp

au2ang = 0.5291772083 # 玻尔半径到埃的转换因子

# 激活函数的字典，方便按名称调用
acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

# 从 nequip 库导入核心模块
from .nequip.nn import (
    GraphModuleMixin,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from .nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    Embedding_block,
    Embedding_block_q
)


class residual_block(torch.nn.Module):
    """一个带有残差连接的等变模块。

    它包含两个线性层和一个非线性激活层。
    可以作为网络中的一个基本构建块。

    Attributes:
        irreps_in (o3.Irreps): 输入的不可约表示。
        feature_irreps_hidden (o3.Irreps): 隐藏层的特征的不可约表示。
        resnet (bool): 是否使用残差连接。
        equivariant_nonlin (torch.nn.Module): 等变非线性激活模块。
        linear1 (Linear): 第一个线性层。
        linear2 (Linear): 第二个线性层。

    """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        """构造函数。

        Args:
            irreps_in (o3.Irreps): 输入的不可约表示。
            feature_irreps_hidden (o3.Irreps): 隐藏层的特征的不可约表示。
            resnet (bool, optional): 是否使用残差连接。默认为 False。
            nonlinearity_type (str, optional): 非线性激活的类型，可以是 "gate" 或 "norm"。默认为 "gate"。
            nonlinearity_scalars (Dict[str, Callable], optional): 用于标量部分的激活函数字典，键为'e'(偶)和'o'(奇)。
            nonlinearity_gates (Dict[str, Callable], optional): 用于门控激活中的门控标量的激活函数字典。
        """
        super().__init__()
        # 初始化
        assert nonlinearity_type in ("gate", "norm")
        # 将非线性激活函数的字典键从字符串改为宇称整数 (1 for even, -1 for odd)
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.irreps_in = o3.Irreps(irreps_in)
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet

        # 分离出标量和需要门控的张量部分
        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0 and ir in self.irreps_in
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0 and ir in self.irreps_in
            ]
        )

        self.irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            # 确定门控标量的类型 (偶或奇)
            ir = (
                "0e"
                if o3.Irrep("0e") in self.feature_irreps_hidden
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TODO: 直接使用字典可能不够安全，未来可以改进
            # 定义门控激活函数
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            linear_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else: # "norm"
            linear_irreps_out = self.irreps_layer_out.simplify()

            # 定义范数激活函数
            equivariant_nonlin = NormActivation(
                irreps_in=linear_irreps_out,
                # 范数是一个偶标量, 所以使用 nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin
        
        # 定义两个线性层
        self.linear1 = Linear(
            irreps_in=self.irreps_in, irreps_out=linear_irreps_out
        )
        
        self.linear2 = Linear(
            irreps_in=self.equivariant_nonlin.irreps_out, irreps_out=irreps_in
        )

    def forward(self, x):
        """前向传播。"""
        # 为残差连接保存原始特征
        old_x = x
        x = self.linear1(x)
        # 应用非线性激活
        x = self.equivariant_nonlin(x)
        x = self.linear2(x)
        # 执行残差连接
        if self.resnet:
            x = old_x + x
        return x

class Edge_builder(GraphModuleMixin, torch.nn.Module):
    """通过组合节点特征和边属性来构建边特征的模块。

    该模块的核心思想是使用一个可学习的张量积（Tensor Product）来融合来自
    源节点、目标节点以及它们之间连线的几何信息（球谐函数表示的边属性）。
    张量积的权重不是固定的，而是通过一个全连接网络（FCN）根据边的径向
    距离嵌入动态生成的，这使得模型能够根据距离调整相互作用的强度和形式。

    Attributes:
        linear_node_src (Linear): 用于处理源节点特征的线性层。
        linear_node_dst (Linear): 用于处理目标节点特征的线性层。
        tp (TensorProduct): 核心的张量积操作模块。
        fc (FullyConnectedNet): 根据边嵌入生成张量积权重的全连接网络。
        linear_edge (Linear): 对张量积结果进行最终线性变换的层。
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:
        """构造函数。

        Args:
            irreps_in (dict): 输入特征的不可约表示字典。
            irreps_out (dict): 输出特征的不可约表示字典。
            invariant_layers (int, optional): FCN中的隐藏层数量。默认为1。
            invariant_neurons (int, optional): FCN中隐藏层的神经元数量。默认为8。
            nonlinearity_scalars (Dict[str, Callable], optional): FCN中使用的非线性激活函数。
        """
        super().__init__()

        # 初始化并校验模块所需的输入输出Irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1), # (l=0, p=1) 偶标量
                        )
                    ]  # 强制要求边的嵌入是标量，以便可以使用标准的全连接网络处理
                )
            },
            irreps_out={AtomicDataDict.EDGE_FEATURES_KEY: irreps_out},
        )
        
        irreps_node_fea = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]   
        feature_irreps_out = self.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY]     

        # - 构建模块 -
        # 线性层，用于处理源节点和目标节点的特征
        self.linear_node_src = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.linear_node_dst = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )

        # 准备张量积的指令 (instructions)
        # 这部分定义了哪些输入Irreps如何组合以生成输出Irreps
        irreps_mid = []
        instructions = []
        # 遍历节点特征的每个Irrep
        for i, (mul, ir_in1) in enumerate(irreps_node_fea):
            # 遍历边属性的每个Irrep
            for j, (_, ir_in2) in enumerate(irreps_edge_attr):
                # 遍历两个Irrep张量积后可能产生的所有输出Irrep
                for ir_out in ir_in1 * ir_in2:
                    # 如果这个输出Irrep是我们想要的，就记录下来
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True)) # uvu表示权重是可训练的

        # 对张量积的输出irreps进行排序和简化，这对于后续线性层的效率至关重要
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # 调整指令中的输出索引以匹配排序后的irreps
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # 定义张量积操作
        self.tp = TensorProduct(
            irreps_node_fea,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # 全连接网络，用于从边的径向基函数嵌入中生成张量积的权重
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        # 最终的线性层，将张量积的输出映射到最终的边特征irreps
        self.linear_edge = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """前向传播。"""
        # 1. 从径向基函数嵌入生成张量积权重
        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        # 2. 准备输入特征
        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1] # 源节点 i
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0] # 目标节点 j

        # 3. 线性变换源节点和目标节点的特征，然后相加
        x_ij = self.linear_node_src(x[edge_src]) + self.linear_node_dst(x[edge_dst])
        
        # 4. 执行张量积
        edge_features = self.tp(
            x_ij, data[AtomicDataDict.EDGE_ATTRS_KEY], weight
        )

        # 5. 应用最终的线性层
        edge_features = self.linear_edge(edge_features)
        
        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features
        return data

class Edge_builder_tp(GraphModuleMixin, torch.nn.Module):
    """`Edge_builder` 的一个变体，使用两个不同节点特征的张量积。

    这个模块将源节点特征和目标节点特征分别进行线性变换后，直接进行张量积，
    而不是像 `Edge_builder` 那样先将它们相加。这允许模型学习更复杂的、
    非对称的相互作用。最终计算出的特征会累加到已有的边特征上。

    Attributes:
        linear_node_src (Linear): 用于处理源节点特征的线性层。
        linear_node_dst (Linear): 用于处理目标节点特征的线性层。
        tp (TensorProduct): 核心的张量积操作模块。
        fc (FullyConnectedNet): 根据边嵌入生成张量积权重的全连接网络。
        linear_edge (Linear): 对张量积结果进行最终线性变换的层。
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
        irreps_node_prev = None
    ) -> None:
        """构造函数。

        Args:
            irreps_in (dict): 输入特征的不可约表示字典。
            irreps_out (dict): 输出特征的不可约表示字典。
            invariant_layers (int, optional): FCN中的隐藏层数量。默认为1。
            invariant_neurons (int, optional): FCN中隐藏层的神经元数量。默认为8。
            nonlinearity_scalars (Dict[str, Callable], optional): FCN中使用的非线性激活函数。
            irreps_node_prev (o3.Irreps or str, optional): 目标节点特征经过线性变换后的Irreps。
        """
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.EDGE_FEATURES_KEY: irreps_out},
        )
        
        irreps_node_fea = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY]     

        # - 构建模块 -
        self.linear_node_src = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        if isinstance(irreps_node_prev, str):
            self.irreps_node_prev = o3.Irreps(irreps_node_prev)
        else:
            self.irreps_node_prev = irreps_node_prev
        
        self.linear_node_dst = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=self.irreps_node_prev,
            internal_weights=True,
            shared_weights=True,
        )

        # 准备张量积指令
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_node_fea):
            for j, (_, ir_in2) in enumerate(self.irreps_node_prev):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # 排序和简化
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # 调整指令索引
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_node_fea,
            self.irreps_node_prev,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_edge = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """前向传播。"""
        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1] # 源节点 i
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0] # 目标节点 j

        x_i = self.linear_node_src(x[edge_src]) 
        x_j = self.linear_node_dst(x[edge_dst])
        
        edge_features = self.tp(
            x_i, x_j, weight
        )

        edge_features = self.linear_edge(edge_features)
          
        # 将新计算的边特征与已有的特征相加（累积效应）
        data[AtomicDataDict.EDGE_FEATURES_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY] + edge_features
        return data

class Triplet_builder(GraphModuleMixin, torch.nn.Module):
    """构建三元组（triplet, k-j-i）特征的模块。

    此模块旨在捕捉原子间的三体相互作用信息。它通过张量积组合两条
    相连边（k->j 和 j->i）的特征来创建三元组特征。张量积的权重由
    这三点形成的夹角嵌入（通过球谐函数计算）来动态调节。

    Attributes:
        ang_emb (sph_harm_layer): 用于角度嵌入的球谐函数层。
        linear_edge_kj (Linear): 边 k->j 特征的线性变换层。
        linear_edge_ji (Linear): 边 j->i 特征的线性变换层。
        tp (TensorProduct): 核心的张量积操作模块。
        fc (FullyConnectedNet): 根据角度嵌入生成张量积权重的全连接网络。
        linear_triplet (Linear): 对张量积结果进行最终线性变换的层。
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:
        """构造函数。

        Args:
            irreps_in (dict): 输入特征的不可约表示字典。
            irreps_out (dict): 输出特征的不可约表示字典。
            invariant_layers (int, optional): FCN中的隐藏层数量。默认为1。
            invariant_neurons (int, optional): FCN中隐藏层的神经元数量。默认为8。
            nonlinearity_scalars (Dict[str, Callable], optional): FCN中使用的非线性激活函数。
        """
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_FEATURES_KEY,
                AtomicDataDict.ANGLE_EMBEDDING_KEY
            ],
            my_irreps_in={
                AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # 强制要求角度嵌入是标量
                )
            },
            irreps_out={AtomicDataDict.TRIPLET_FEATURES_KEY: irreps_out},
        )
        
        # 球谐函数层，用于角度嵌入
        self.ang_emb = sph_harm_layer(self.irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps)
        
        irreps_edge_fea = self.irreps_in[AtomicDataDict.EDGE_FEATURES_KEY]   
        feature_irreps_out = self.irreps_out[AtomicDataDict.TRIPLET_FEATURES_KEY]  

        # - 构建模块 -
        self.linear_edge_kj = Linear(
            irreps_in=irreps_edge_fea,
            irreps_out=irreps_edge_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.linear_edge_ji = Linear(
            irreps_in=irreps_edge_fea,
            irreps_out=irreps_edge_fea,
            internal_weights=True,
            shared_weights=True,
        )   
        
        # 准备张量积指令
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_edge_fea):
            for j, (_, ir_in2) in enumerate(irreps_edge_fea):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # 排序和简化
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # 调整指令索引
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_edge_fea,
            irreps_edge_fea,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # 全连接网络，用于从角度嵌入中生成张量积的权重
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_triplet = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        
    def triplets(self, edge_index, num_nodes, cell_shift):
        """计算图中的所有三元组 (k->j->i)。

        利用 `torch_sparse` 的 `SparseTensor` 高效地寻找相连的边。
        一个三元组由中心原子 j，以及两个邻居 i 和 k 构成。

        Args:
            edge_index (torch.Tensor): 边索引, 形状为 [2, num_edges]。
            num_nodes (int): 节点总数。
            cell_shift (torch.Tensor): 边的周期性晶胞偏移, 形状为 [num_edges, 3]。

        Returns:
            tuple: 包含三元组信息的元组 (col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji)。
                   其中 idx_i, idx_j, idx_k 是三元组中原子的索引。
                   idx_kj, idx_ji 是构成三元组的两条边的索引。
        """
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        # 构建邻接矩阵的稀疏表示
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # 三元组的节点索引 (k->j->i)
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # 三元组的边索引 (k-j, j->i)
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()
        """
        idx_i -> pos[idx_i]
        idx_j -> pos[idx_j] - nbr_shift[idx_ji]
        idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
        """
        # 移除 i == k 的三元组 (即 A->B->A 形式的路径)
        # 但需要注意周期性边界条件：如果晶胞偏移不为零，即使 i==k，也应保留，因为它们是不同的原子映象
        relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
        mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
        idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """前向传播。"""
        z = data['z']
        pos = data['pos']
        edge_index = data['edge_index']
        nbr_shift = data['nbr_shift']
        cell_shift = data['cell_shift'] # shape(Nedges, 3)

        # 1. 找出所有的三元组
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, z.size(0), cell_shift)
        
        # 2. 计算三元组的角度
        # 考虑周期性边界条件，获取原子的真实位置
        pos_i = pos[idx_i]
        pos_j = pos[idx_j] - nbr_shift[idx_ji]
        pos_k = pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj] 

        pos_ji = pos_j - pos_i
        pos_kj = pos_k - pos_j

        # 利用点积和叉积计算夹角，以保证数值稳定性
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)
        
        # 3. 将角度嵌入为球谐函数特征
        ang_emb = self.ang_emb(angle)
        
        # 4. 线性变换两条边的特征
        edge_kj = self.linear_edge_kj(data[AtomicDataDict.EDGE_FEATURES_KEY])[idx_kj]
        edge_ji = self.linear_edge_ji(data[AtomicDataDict.EDGE_FEATURES_KEY])[idx_ji]
        
        # 5. 从角度嵌入生成张量积权重
        weight = self.fc(ang_emb)
        
        # 6. 执行张量积
        triplet_fea = self.tp(
            edge_kj, edge_ji, weight
        )

        # 7. 应用最终的线性层
        triplet_fea = self.linear_triplet(triplet_fea)
        
        # 8. 将结果存回data字典
        data[AtomicDataDict.TRIPLET_FEATURES_KEY] = triplet_fea
        data[AtomicDataDict.TRIPLET_INDEX_KEY] = (idx_i, idx_j, idx_k, idx_kj, idx_ji)
        return data

class Ham_layer(nn.Module):
    """一个简单的哈密顿量预测层。

    该层由一个残差块和一个最终的线性层组成，用于将学习到的特征
    映射到最终的输出空间（例如，哈密顿量矩阵的元素）。

    Attributes:
        residual (residual_block): 一个残差连接块，用于深化网络和稳定训练。
        linear (Linear): 一个等变线性层，用于输出最终结果。
    """
    def __init__(self, irreps_in, feature_irreps_hidden, irreps_out, nonlinearity_type: str = "gate", resnet: bool = True):
        """构造函数。

        Args:
            irreps_in (o3.Irreps): 输入特征的不可约表示。
            feature_irreps_hidden (o3.Irreps): 残差块中隐藏层的不可约表示。
            irreps_out (o3.Irreps): 输出特征的不可约表示。
            nonlinearity_type (str, optional): 非线性激活的类型。默认为 "gate"。
            resnet (bool, optional): 是否在残差块中使用残差连接。默认为 True。
        """
        super().__init__()
        self.residual = residual_block(irreps_in=irreps_in, feature_irreps_hidden=feature_irreps_hidden, 
                                                 nonlinearity_type = nonlinearity_type, resnet=resnet) 
        self.linear = Linear(irreps_in=irreps_in, irreps_out=irreps_out) 
        
    def forward(self, x):
        """前向传播。"""
        x = self.residual(x)
        x = self.linear(x)
        return x

class HamGNN_pre(nn.Module):
    """HamGNN 的特征提取网络（也称为表示网络）。

    这个网络基于 `nequip` 架构，通过一系列精心设计的嵌入和等变卷积层，
    从原子结构（节点、边、角度）中学习丰富的、考虑了对称性的特征表示。
    这些特征随后可以被用于预测各种物理属性。

    Attributes:
        one_hot (OneHotAtomEncoding): 将原子类型编码为 one-hot 向量。
        spharm_edges (SphericalHarmonicEdgeAttrs): 将边的方向向量编码为球谐函数特征。
        radial_basis (RadialBasisEdgeEncoding): 将边的距离编码为径向基函数特征。
        chemical_embedding (AtomwiseLinear): 将 one-hot 编码线性映射到初始节点特征。
        convnet (nn.ModuleList): 包含多个等变卷积层的列表。
        conv_to_output_node (AtomwiseLinear): 将卷积后的节点特征映射到最终输出。
        conv_to_output_edge (Edge_builder): 从节点特征和边属性构建最终的边特征。
        conv_to_output_triplet (Triplet_builder, optional): 如果启用，构建三元组特征。
    """
    def __init__(self, config):
        """构造函数。

        Args:
            config (EasyDict): 包含所有模型超参数的配置对象。
        """
        super(HamGNN_pre, self).__init__()
        
        self.num_types = config.HamGNN_pre.num_types # 数据集中的原子类型数量
        self.set_features = config.HamGNN_pre.set_features # 是否将 one_hot 编码设置为节点特征
        
        self.export_triplet = config.HamGNN_pre.export_triplet # 是否导出三元组的特征
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh # 边方向的球谐函数表示
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization # 球谐函数的归一化方法
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize # 是否对球谐函数进行归一化
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output # 最终输出的节点特征的不可约表示
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output # 最终输出的边特征的不可约表示
        
        # 径向基函数和截断函数
        self.cutoff = config.HamGNN_pre.cutoff # 截断半径
        self.cutoff_func = config.HamGNN_pre.cutoff_func # 截断函数类型
        if 'p' == self.cutoff_func.lower()[0]:  # "Ddimnet 中使用的包络函数"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "余弦截断函数"
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func # 径向基函数类型
        self.num_radial = config.HamGNN_pre.num_radial # 径向基函数的数量
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # 相互作用层（卷积层）的数量
        self.resnet = config.HamGNN_pre.resnet # 是否在卷积层中使用残差连接      
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # 节点特征的不可约表示
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # 卷积层中隐藏特征的不可约表示
        self.invariant_layers = config.HamGNN_pre.invariant_layers # 卷积层中不变部分的网络层数
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons # 卷积层中不变部分的神经元数量
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # 卷积层的额外初始化参数
        
        # 节点原子类型的一次性编码
        self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # 将节点的原子序数映射为 "num_types*0e" 的 one-hot 编码
        
        # 将边的方向向量嵌入为球谐函数特征
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # 将边的距离嵌入为径向基函数特征
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        # 将 one-hot 编码线性映射到初始的节点特征
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.one_hot.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        # 相互作用层（卷积层）列表
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.one_hot.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        # 将卷积后的节点特征映射到最终的输出节点特征
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        # 构建最终的边特征
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        # 如果需要，构建三元组特征
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
    
    def forward(self, data, batch=None):
        """前向传播。

        Args:
            data (AtomicDataDict.Type): 包含原子图信息的字典。
            batch (torch.Tensor, optional): 批处理索引。默认为 None。

        Returns:
            EasyDict: 包含计算出的节点、边和（可选）三元组特征的字典。
        """
        # 1. 初始嵌入
        self.one_hot(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # 2. 相互作用（卷积）
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        # 3. 输出特征构建
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)    
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_pre2(nn.Module):
    """`HamGNN_pre` 的一个变体。

    主要区别在于：
    - 使用 `Embedding_block` 代替 `OneHotAtomEncoding`，允许更灵活的初始节点嵌入，
      而不仅仅是原子类型的 one-hot 编码。
    - 增加了一个可选的 `Edge_builder_tp` 层，用于引入额外的、通过不同节点特征
      张量积计算的边特征，以增强模型的表达能力。
    """
    def __init__(self, config):
        """构造函数。

        Args:
            config (EasyDict): 包含所有模型超参数的配置对象。
        """
        super(HamGNN_pre2, self).__init__()
        
        #self.num_types = config.HamGNN_pre.num_types # 数据集中的原子种类数量
        self.set_features = config.HamGNN_pre.set_features # 是否将 one_hot 编码设置为数据中的节点特征
        
        self.export_triplet = config.HamGNN_pre.export_triplet # 是否导出三元组的特征
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output
        
        # 余弦基函数展开层
        self.cutoff = config.HamGNN_pre.cutoff
        self.cutoff_func = config.HamGNN_pre.cutoff_func
        if 'e' == self.cutoff_func.lower()[0]:  # "Ddimnet 中使用的包络函数"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "余弦截断函数"
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # 相互作用层数量
        self.resnet = config.HamGNN_pre.resnet # 是否添加残差层       
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # 节点的不可约表示
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # 卷积层中节点的隐藏不可约表示
        self.invariant_layers = config.HamGNN_pre.invariant_layers 
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # 卷积层的额外初始化参数
        
        #self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # 将节点的原子序数映射为 "num_types*0e" 的 one-hot 编码
        
        self.num_node_attr_feas = config.HamGNN_pre.num_node_attr_feas
        self.emb = Embedding_block(num_node_attr_feas = self.num_node_attr_feas, set_features=self.set_features)
        
        # 将边的方向嵌入为球谐函数特征, irreps_edge_sh 是边方向的不可约表示
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # 将边的距离嵌入为 'num_basis*0e' 的特征
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.emb.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.emb.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        self.add_edge_tp = config.HamGNN_pre.add_edge_tp
        if self.add_edge_tp:
            self.irreps_node_prev = config.HamGNN_pre.irreps_node_prev
            self.conv_to_output_edge_tp = Edge_builder_tp(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                            irreps_node_prev=self.irreps_node_prev,
                                                            irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
        #"""
        
        #self.conv_to_output_edge = AtomwiseLinear(field=AtomicDataDict.EDGE_FEATURES_KEY, out_field=AtomicDataDict.EDGE_FEATURES_KEY, 
                                                  #irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.convnet[-1].conv.linear_2.irreps_in}, irreps_out= self.irreps_edge_output)
    
    def forward(self, data, batch=None):
        """前向传播。"""
        #self.one_hot(data)
        self.emb(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # 轨道卷积
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)  
        if self.add_edge_tp:
            self.conv_to_output_edge_tp(data)  
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_pre_charge(nn.Module):
    """`HamGNN_pre` 的另一个变体，增加了对电荷掺杂的支持。

    它使用 `Embedding_block_q`，该模块可以接收每个节点的电荷信息作为额外的
    节点属性，并将其与原子类型嵌入相结合，生成考虑了电荷状态的初始节点特征。
    这使得模型能够学习电荷对体系性质的影响。
    """
    def __init__(self, config):
        """构造函数。

        Args:
            config (EasyDict): 包含所有模型超参数的配置对象。
        """
        super(HamGNN_pre_charge, self).__init__()
        
        #self.num_types = config.HamGNN_pre.num_types # 数据集中的原子种类数量
        self.set_features = config.HamGNN_pre.set_features # 是否将 one_hot 编码设置为数据中的节点特征
        
        self.export_triplet = config.HamGNN_pre.export_triplet # 是否导出三元组的特征
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output
        
        # 余弦基函数展开层
        self.cutoff = config.HamGNN_pre.cutoff
        self.cutoff_func = config.HamGNN_pre.cutoff_func
        if 'e' == self.cutoff_func.lower()[0]:  # "Ddimnet 中使用的包络函数"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "余弦截断函数"
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # 相互作用层数量
        self.resnet = config.HamGNN_pre.resnet # 是否添加残差层       
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # 节点的不可约表示
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # 卷积层中节点的隐藏不可约表示
        self.invariant_layers = config.HamGNN_pre.invariant_layers 
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # 卷积层的额外初始化参数
        
        #self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # 将节点的原子序数映射为 "num_types*0e" 的 one-hot 编码

        self.num_node_attr_feas = config.HamGNN_pre.num_node_attr_feas
        self.apply_charge_doping = True 
        self.num_charge_attr_feas = config.HamGNN_pre.num_charge_attr_feas
        self.emb = Embedding_block_q(num_node_attr_feas = self.num_node_attr_feas, apply_charge_doping=self.apply_charge_doping, 
                                   num_charge_attr_feas = self.num_charge_attr_feas, set_features=self.set_features)
        
        # 将边的方向嵌入为球谐函数特征, irreps_edge_sh 是边方向的不可约表示
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # 将边的距离嵌入为 'num_basis*0e' 的特征
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.emb.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.emb.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        self.add_edge_tp = config.HamGNN_pre.add_edge_tp
        if self.add_edge_tp:
            self.irreps_node_prev = config.HamGNN_pre.irreps_node_prev
            self.conv_to_output_edge_tp = Edge_builder_tp(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                            irreps_node_prev=self.irreps_node_prev,
                                                            irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
        #"""
        
        #self.conv_to_output_edge = AtomwiseLinear(field=AtomicDataDict.EDGE_FEATURES_KEY, out_field=AtomicDataDict.EDGE_FEATURES_KEY, 
                                                  #irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.convnet[-1].conv.linear_2.irreps_in}, irreps_out= self.irreps_edge_output)
    
    def forward(self, data, batch=None):
        """前向传播。"""
        #self.one_hot(data)
        self.emb(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # 轨道卷积
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)  
        if self.add_edge_tp:
            self.conv_to_output_edge_tp(data)  
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_out(nn.Module):
    """HamGNN 的输出模块，用于从学习到的特征构建哈密顿量和重叠矩阵。

    这个模块是 HamGNN 的核心物理构建模块。它接收由表示网络（如 `HamGNN_pre`）
    生成的节点和边特征，然后通过几个专门的网络（onsitenet, offsitenet）将这些
    特征映射到局域的哈密顿量和重SAO（Symmetrized Atomic Orbitals）基矢下的
    重叠矩阵元素。

    该模块内置了对多种 DFT 软件（OpenMX, SIESTA, ABACUS）基组定义的兼容性，
    能够处理自旋轨道耦合（SOC），并能计算能带结构和原子受力。

    Attributes:
        derivative (bool): 是否计算梯度（用于力）。
        create_graph (bool): 是否创建计算图（用于高阶导数）。
        nao_max (int): 原子轨道基组的最大数量。
        ham_type (str): 使用的 DFT 软件类型 ('openmx', 'siesta', 'abacus')。
        ham_only (bool): 是否只预测哈密顿量（不预测重叠矩阵）。
        symmetrize (bool): 是否对生成的矩阵进行对称化处理。
        include_triplet (bool): 是否在计算中使用三元组特征。
        soc_switch (bool): 是否开启自旋轨道耦合（SOC）。
        soc_basis (str): SOC计算使用的基组 ('so3' 或 'su2')。
        onsitenet_residual/linear: 用于预测在位（on-site）哈密顿量元素的网络层。
        offsitenet_residual/linear: 用于预测异位（off-site）哈密顿量元素的网络层。
        onsitenet_s/offsitenet_s: 用于预测重叠矩阵元素的网络层。
        cg_cal (ClebschGordan): 用于计算 Clebsch-Gordan 系数的工具。
    """
    
    def __init__(self, irreps_in_node: Union[int, str, o3.Irreps]=None, irreps_in_edge: Union[int, str, o3.Irreps]=None, irreps_in_triplet: Union[int, str, o3.Irreps]=None, nao_max: int = 14, return_forces=False, create_graph=False, 
                 ham_type: str='openmx', ham_only: bool = False, symmetrize: bool=True, include_triplet: bool = False, calculate_band_energy: bool = False, num_k: int = 8, 
                 k_path:Union[list, np.array, tuple]=None, band_num_control:dict=None, soc_switch:bool=True, nonlinearity_type:str='norm', export_reciprocal_values: bool = False, add_H0:bool= False, soc_basis: str='so3'):
        """构造函数。

        Args:
            irreps_in_node (o3.Irreps, optional): 输入节点特征的不可约表示。
            irreps_in_edge (o3.Irreps, optional): 输入边特征的不可约表示。
            irreps_in_triplet (o3.Irreps, optional): 输入三元组特征的不可约表示。
            nao_max (int, optional): 原子轨道基组的最大数量。默认为 14。
            return_forces (bool, optional): 是否返回力。默认为 False。
            create_graph (bool, optional): 是否创建计算图。默认为 False。
            ham_type (str, optional): DFT软件类型。默认为 'openmx'。
            ham_only (bool, optional): 是否只预测哈密顿量。默认为 False。
            symmetrize (bool, optional): 是否对称化矩阵。默认为 True。
            include_triplet (bool, optional): 是否包含三元组信息。默认为 False。
            calculate_band_energy (bool, optional): 是否计算能带。默认为 False。
            num_k (int, optional): k点路径上的采样点数。默认为 8。
            k_path (list/np.array/tuple, optional): 定义能带计算的k点路径。
            band_num_control (dict, optional): 控制每个元素计算的能带数量。
            soc_switch (bool, optional): 是否开启SOC。默认为 True。
            nonlinearity_type (str, optional): 使用的非线性激活函数类型。默认为 'norm'。
            export_reciprocal_values (bool, optional): 是否导出倒空间相关值。默认为 False。
            add_H0 (bool, optional): 是否添加一个零阶哈密顿量项。默认为 False。
            soc_basis (str, optional): SOC计算的基组。默认为 'so3'。
        """
        super(HamGNN_out, self).__init__()
        if return_forces:
            self.derivative = True
        else:
            self.derivative = False

        self.create_graph = create_graph

        self.nao_max = nao_max
        self.ham_type = ham_type.lower()
        self.ham_only = ham_only
        self.symmetrize = symmetrize
        self.include_triplet = include_triplet
        self.soc_switch = soc_switch
        self.nonlinearity_type = nonlinearity_type
        self.export_reciprocal_values = export_reciprocal_values
        self.add_H0 = add_H0
        # 能带相关参数
        self.calculate_band_energy = calculate_band_energy
        self.num_k = num_k
        self.k_path = k_path
        
        self.soc_basis = soc_basis.lower()
        
        # 能带数量控制
        if (band_num_control is not None) and (not self.export_reciprocal_values) and (isinstance(band_num_control, dict)):      
            self.band_num_control = {Element[k].Z: band_num_control[k] for k in band_num_control.keys()}
        elif isinstance(band_num_control, int):
            self.band_num_control = band_num_control
        else:
            self.band_num_control = None
        
        # OpenMX 基组定义
        if self.ham_type == 'openmx':
            # 每种元素的价电子数
            self.num_valence = {Element['H'].Z: 1, Element['He'].Z: 2, Element['Li'].Z: 3, Element['Be'].Z: 2, Element['B'].Z: 3,
                                Element['C'].Z: 4, Element['N'].Z: 5,  Element['O'].Z: 6,  Element['F'].Z: 7,  Element['Ne'].Z: 8,
                                Element['Na'].Z: 9, Element['Mg'].Z: 8, Element['Al'].Z: 3, Element['Si'].Z: 4, Element['P'].Z: 5,
                                Element['S'].Z: 6,  Element['Cl'].Z: 7, Element['Ar'].Z: 8, Element['K'].Z: 9,  Element['Ca'].Z: 10,
                                Element['Sc'].Z: 11, Element['Ti'].Z: 12, Element['V'].Z: 13, Element['Cr'].Z: 14, Element['Mn'].Z: 15,
                                Element['Fe'].Z: 16, Element['Co'].Z: 17, Element['Ni'].Z: 18, Element['Cu'].Z: 19, Element['Zn'].Z: 20,
                                Element['Ga'].Z: 13, Element['Ge'].Z: 4,  Element['As'].Z: 15, Element['Se'].Z: 6,  Element['Br'].Z: 7,
                                Element['Kr'].Z: 8,  Element['Rb'].Z: 9,  Element['Sr'].Z: 10, Element['Y'].Z: 11, Element['Zr'].Z: 12,
                                Element['Nb'].Z: 13, Element['Mo'].Z: 14, Element['Tc'].Z: 15, Element['Ru'].Z: 14, Element['Rh'].Z: 15,
                                Element['Pd'].Z: 16, Element['Ag'].Z: 17, Element['Cd'].Z: 12, Element['In'].Z: 13, Element['Sn'].Z: 14,
                                Element['Sb'].Z: 15, Element['Te'].Z: 16, Element['I'].Z: 7, Element['Xe'].Z: 8, Element['Cs'].Z: 9,
                                Element['Ba'].Z: 10, Element['La'].Z: 11, Element['Ce'].Z: 12, Element['Pr'].Z: 13, Element['Nd'].Z: 14,
                                Element['Pm'].Z: 15, Element['Sm'].Z: 16, Element['Dy'].Z: 20, Element['Ho'].Z: 21, Element['Lu'].Z: 11,
                                Element['Hf'].Z: 12, Element['Ta'].Z: 13, Element['W'].Z: 12,  Element['Re'].Z: 15, Element['Os'].Z: 14,
                                Element['Ir'].Z: 15, Element['Pt'].Z: 16, Element['Au'].Z: 17, Element['Hg'].Z: 18, Element['Tl'].Z: 19,
                                Element['Pb'].Z: 14, Element['Bi'].Z: 15
                            }
            
            if self.nao_max == 14:
                # OpenMX基组到e3nn标准顺序的索引映射
                self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e")
                # 定义每种元素使用的轨道基函数
                self.basis_def = {  1:[0,1,3,4,5], # H
                                    2:[0,1,3,4,5], # He
                                    3:[0,1,2,3,4,5,6,7,8], # Li
                                    4:[0,1,3,4,5,6,7,8], # Be
                                    5:[0,1,3,4,5,6,7,8,9,10,11,12,13], # B
                                    6:[0,1,3,4,5,6,7,8,9,10,11,12,13], # C
                                    7:[0,1,3,4,5,6,7,8,9,10,11,12,13], # N
                                    8:[0,1,3,4,5,6,7,8,9,10,11,12,13], # O
                                    9:[0,1,3,4,5,6,7,8,9,10,11,12,13], # F
                                    10:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ne
                                    11:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Na
                                    12:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mg
                                    13:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Al
                                    14:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Si
                                    15:[0,1,3,4,5,6,7,8,9,10,11,12,13], # p
                                    16:[0,1,3,4,5,6,7,8,9,10,11,12,13], # S
                                    17:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Cl
                                    18:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ar
                                    19:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # K
                                    20:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ca
                                    35:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Br  
                                    Element['V'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # V
                                }
            
            elif self.nao_max == 13:
                self.basis_def = {  1:[0,1,2,3,4], # H
                                    5:[0,1,2,3,4,5,6,7,8,9,10,11,12], # B
                                    6:[0,1,2,3,4,5,6,7,8,9,10,11,12], # C
                                    7:[0,1,2,3,4,5,6,7,8,9,10,11,12], # N
                                    8:[0,1,2,3,4,5,6,7,8,9,10,11,12] # O
                                }
                self.index_change = torch.LongTensor([0,1,4,2,3,7,5,6,10,12,8,11,9])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            
            elif self.nao_max == 19:
                self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10,16,18,14,17,15])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
                self.basis_def = {  1:[0,1,3,4,5], # H
                                    2:[0,1,3,4,5], # He
                                    3:[0,1,2,3,4,5,6,7,8], # Li
                                    4:[0,1,3,4,5,6,7,8], # Be
                                    5:[0,1,3,4,5,6,7,8,9,10,11,12,13], # B
                                    6:[0,1,3,4,5,6,7,8,9,10,11,12,13], # C
                                    7:[0,1,3,4,5,6,7,8,9,10,11,12,13], # N
                                    8:[0,1,3,4,5,6,7,8,9,10,11,12,13], # O
                                    9:[0,1,3,4,5,6,7,8,9,10,11,12,13], # F
                                    10:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ne
                                    11:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Na
                                    12:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mg
                                    13:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Al
                                    14:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Si
                                    15:[0,1,3,4,5,6,7,8,9,10,11,12,13], # p
                                    16:[0,1,3,4,5,6,7,8,9,10,11,12,13], # S
                                    17:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Cl
                                    18:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ar
                                    19:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # K
                                    20:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ca
                                    42:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Mo   
                                    83:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Bi  
                                    34:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Se
                                    24:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Cr 
                                    53:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # I  
                                    82:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # pb
                                    55:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Cs
                                    33:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # As
                                    31:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ga  
                                    32:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ge
                                    Element['V'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # V
                                    Element['Sb'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Sb
                                }
            
            elif self.nao_max == 26:
                self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10,16,18,14,17,15,22,23,21,24,20,25,19])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
                self.basis_def = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18],f1=[19,20,21,22,23,24,25]: {
                    Element['H'].Z : s1+s2+p1,  # H6.0-s2p1
                    Element['He'].Z : s1+s2+p1,  # He8.0-s2p1
                    Element['Li'].Z : s1+s2+s3+p1+p2,  # Li8.0-s3p2
                    Element['Be'].Z : s1+s2+p1+p2,  # Be7.0-s2p2
                    Element['B'].Z : s1+s2+p1+p2+d1,  # B7.0-s2p2d1
                    Element['C'].Z : s1+s2+p1+p2+d1,  # C6.0-s2p2d1
                    Element['N'].Z : s1+s2+p1+p2+d1,  # N6.0-s2p2d1
                    Element['O'].Z : s1+s2+p1+p2+d1,  # O6.0-s2p2d1
                    Element['F'].Z : s1+s2+p1+p2+d1,  # F6.0-s2p2d1
                    Element['Ne'].Z: s1+s2+p1+p2+d1,  # Ne9.0-s2p2d1
                    Element['Na'].Z: s1+s2+s3+p1+p2+d1,  # Na9.0-s3p2d1
                    Element['Mg'].Z: s1+s2+s3+p1+p2+d1,  # Mg9.0-s3p2d1
                    Element['Al'].Z: s1+s2+p1+p2+d1,  # Al7.0-s2p2d1
                    Element['Si'].Z: s1+s2+p1+p2+d1,  # Si7.0-s2p2d1
                    Element['P'].Z: s1+s2+p1+p2+d1,  # P7.0-s2p2d1
                    Element['S'].Z: s1+s2+p1+p2+d1,  # S7.0-s2p2d1
                    Element['Cl'].Z: s1+s2+p1+p2+d1,  # Cl7.0-s2p2d1
                    Element['Ar'].Z: s1+s2+p1+p2+d1,  # Ar9.0-s2p2d1
                    Element['K'].Z: s1+s2+s3+p1+p2+d1,  # K10.0-s3p2d1
                    Element['Ca'].Z: s1+s2+s3+p1+p2+d1,  # Ca9.0-s3p2d1
                    Element['Sc'].Z: s1+s2+s3+p1+p2+d1,  # Sc9.0-s3p2d1
                    Element['Ti'].Z: s1+s2+s3+p1+p2+d1,  # Ti7.0-s3p2d1
                    Element['V'].Z: s1+s2+s3+p1+p2+d1,  # V6.0-s3p2d1
                    Element['Cr'].Z: s1+s2+s3+p1+p2+d1,  # Cr6.0-s3p2d1
                    Element['Mn'].Z: s1+s2+s3+p1+p2+d1,  # Mn6.0-s3p2d1
                    Element['Fe'].Z: s1+s2+s3+p1+p2+d1,  # Fe5.5H-s3p2d1
                    Element['Co'].Z: s1+s2+s3+p1+p2+d1,  # Co6.0H-s3p2d1
                    Element['Ni'].Z: s1+s2+s3+p1+p2+d1,  # Ni6.0H-s3p2d1
                    Element['Cu'].Z: s1+s2+s3+p1+p2+d1,  # Cu6.0H-s3p2d1
                    Element['Zn'].Z: s1+s2+s3+p1+p2+d1,  # Zn6.0H-s3p2d1
                    Element['Ga'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ga7.0-s3p2d2
                    Element['Ge'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ge7.0-s3p2d2
                    Element['As'].Z: s1+s2+s3+p1+p2+d1+d2,  # As7.0-s3p2d2
                    Element['Se'].Z: s1+s2+s3+p1+p2+d1+d2,  # Se7.0-s3p2d2
                    Element['Br'].Z: s1+s2+s3+p1+p2+d1+d2,  # Br7.0-s3p2d2
                    Element['Kr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Kr10.0-s3p2d2
                    Element['Rb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Rb11.0-s3p2d2
                    Element['Sr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sr10.0-s3p2d2
                    Element['Y'].Z: s1+s2+s3+p1+p2+d1+d2,  # Y10.0-s3p2d2
                    Element['Zr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Zr7.0-s3p2d2
                    Element['Nb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Nb7.0-s3p2d2
                    Element['Mo'].Z: s1+s2+s3+p1+p2+d1+d2,  # Mo7.0-s3p2d2
                    Element['Tc'].Z: s1+s2+s3+p1+p2+d1+d2,  # Tc7.0-s3p2d2
                    Element['Ru'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ru7.0-s3p2d2
                    Element['Rh'].Z: s1+s2+s3+p1+p2+d1+d2,  # Rh7.0-s3p2d2
                    Element['Pd'].Z: s1+s2+s3+p1+p2+d1+d2,  # Pd7.0-s3p2d2
                    Element['Ag'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ag7.0-s3p2d2
                    Element['Cd'].Z: s1+s2+s3+p1+p2+d1+d2,  # Cd7.0-s3p2d2
                    Element['In'].Z: s1+s2+s3+p1+p2+d1+d2,  # In7.0-s3p2d2
                    Element['Sn'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sn7.0-s3p2d2
                    Element['Sb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sb7.0-s3p2d2
                    Element['Te'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Te7.0-s3p2d2f1
                    Element['I'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # I7.0-s3p2d2f1
                    Element['Xe'].Z: s1+s2+s3+p1+p2+d1+d2,  # Xe11.0-s3p2d2
                    Element['Cs'].Z: s1+s2+s3+p1+p2+d1+d2,  # Cs12.0-s3p2d2
                    Element['Ba'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ba10.0-s3p2d2
                    Element['La'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # La8.0-s3p2d2f1
                    Element['Ce'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ce8.0-s3p2d2f1
                    Element['Pr'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pr8.0-s3p2d2f1
                    Element['Nd'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Nd8.0-s3p2d2f1
                    Element['Pm'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pm8.0-s3p2d2f1
                    Element['Sm'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Sm8.0-s3p2d2f1
                    Element['Dy'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Dy8.0-s3p2d2f1
                    Element['Ho'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ho8.0-s3p2d2f1
                    Element['Lu'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Lu8.0-s3p2d2f1
                    Element['Hf'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Hf9.0-s3p2d2f1
                    Element['Ta'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ta7.0-s3p2d2f1
                    Element['W'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # W7.0-s3p2d2f1
                    Element['Re'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Re7.0-s3p2d2f1
                    Element['Os'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Os7.0-s3p2d2f1
                    Element['Ir'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ir7.0-s3p2d2f1
                    Element['Pt'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pt7.0-s3p2d2f1
                    Element['Au'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Au7.0-s3p2d2f1
                    Element['Hg'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Hg8.0-s3p2d2f1
                    Element['Tl'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Tl8.0-s3p2d2f1
                    Element['Pb'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pb8.0-s3p2d2f1
                    Element['Bi'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Bi8.0-s3p2d2f1 
                })()
            else:
                raise NotImplementedError
        
        # SIESTA 基组定义
        elif self.ham_type == 'siesta':
            self.num_valence = {
                1:1,2:2,
                3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
                11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
                19:1,20:2,22:12
            }
            if self.nao_max == 13:
                self.index_change = None       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
                # 这个列表应该遵循 siesta 中 spher_harm.f 的顺序
                self.minus_index = torch.LongTensor([2,4,5,7,9,11]) 
                self.basis_def = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
                    1 : s1+s2+p1, # H
                    2 : s1+s2+p1, # He
                    3 : s1+s2+p1, # Li
                    4 : s1+s2+p1, # Be
                    5 : s1+s2+p1+p2+d1, # B
                    6 : s1+s2+p1+p2+d1, # C
                    7 : s1+s2+p1+p2+d1, # N
                    8 : s1+s2+p1+p2+d1, # O
                    9 : s1+s2+p1+p2+d1, # F
                    10: s1+s2+p1+p2+d1, # Ne
                    11: s1+s2+p1, # Na
                    12: s1+s2+p1, # Mg
                    13: s1+s2+p1+p2+d1, # Al
                    14: s1+s2+p1+p2+d1, # Si
                    15: s1+s2+p1+p2+d1, # P
                    16: s1+s2+p1+p2+d1, # S
                    17: s1+s2+p1+p2+d1, # Cl
                    18: s1+s2+p1+p2+d1, # Ar
                    19: s1+s2+p1, # K
                    20: s1+s2+p1, # Cl
                })()
            elif self.nao_max == 19:
                self.index_change = None
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
                # 这个列表应该遵循 siesta 中 spher_harm.f 的顺序
                self.minus_index = torch.LongTensor([3,5,6,8,10,12,15,17]) 
                self.basis_def = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18]: {
                    1 : s1+s2+p1, # H
                    2 : s1+s2+p1, # He
                    3 : s1+s2+p1, # Li
                    4 : s1+s2+p1, # Be
                    5 : s1+s2+p1+p2+d1, # B
                    6 : s1+s2+p1+p2+d1, # C
                    7 : s1+s2+p1+p2+d1, # N
                    8 : s1+s2+p1+p2+d1, # O
                    9 : s1+s2+p1+p2+d1, # F
                    10: s1+s2+p1+p2+d1, # Ne
                    11: s1+s2+p1, # Na
                    12: s1+s2+p1, # Mg
                    13: s1+s2+p1+p2+d1, # Al
                    14: s1+s2+p1+p2+d1, # Si
                    15: s1+s2+p1+p2+d1, # P
                    16: s1+s2+p1+p2+d1, # S
                    17: s1+s2+p1+p2+d1, # Cl
                    18: s1+s2+p1+p2+d1, # Ar
                    19: s1+s2+p1, # K
                    20: s1+s2+p1, # Cl
                    22: s1+s2+s3+p1+p2+d1+d2, # Ti, 由 Qin 创建
                })()
            else:
                raise NotImplementedError
        # ABACUS 基组定义
        elif self.ham_type == 'abacus':
            # 这个字典用于 abacus 计算
            self.num_valence = {1: 1,  2: 2,
                            3: 3,  4: 4,
                            5: 3,  6: 4,
                            7: 5,  8: 6,
                            9: 7,  10: 8,
                            11: 9, 12: 10,
                            13: 11, 14: 4,
                            15: 5,  16: 6,
                            17: 7,  18: 8,
                            19: 9,  20: 10,
                            21: 11, 22: 12,
                            23: 13, 24: 14,
                            25: 15, 26: 16,
                            27: 17, 28: 18,
                            29: 19, 30: 20,
                            31: 13, 32: 14,
                            33: 5,  34: 6,
                            35: 7,  36: 8,
                            37: 9,  38: 10,
                            39: 11, 40: 12,
                            41: 13, 42: 14,
                            43: 15, 44: 16,
                            45: 17, 46: 18,
                            47: 19, 48: 20,
                            49: 13, 50: 14,
                            51: 15, 52: 16,
                            53: 17, 54: 18,
                            55: 9, 56: 10,
                            57: 11, 72: 26,
                            73: 27, 74: 28,
                            75: 15, 76: 16,
                            77: 17, 78: 18,
                            79: 19, 80: 20,
                            81: 13, 82: 14,
                            83: 15}
            
            if self.nao_max == 13:
                self.index_change = torch.LongTensor([0,1,3,4,2,6,7,5,10,11,9,12,8])
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
                self.minus_index = torch.LongTensor([3,4,6,7,9,10])
                self.basis_def = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
                    1 : np.array(s1+s2+p1, dtype=int), # H
                    2 : np.array(s1+s2+p1, dtype=int), # He
                    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
                    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
                    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
                    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
                    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
                    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
                    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
                    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
                    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
                    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
                    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
                })()           
            
            elif self.nao_max == 27:
                self.index_change = torch.LongTensor([0,1,2,3,5,6,4,8,9,7,12,13,11,14,10,17,18,16,19,15,23,24,22,25,21,26,20])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
                # 这个列表应该遵循 abacus 的顺序
                self.minus_index = torch.LongTensor([5,6,8,9,11,12,16,17,21,22,25,26]) 
                self.basis_def = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14],d2=[15,16,17,18,19],f1=[20,21,22,23,24,25,26]: {
                1 : s1+s2+p1, # H
                2 : s1+s2+p1, # He
                3 : s1+s2+s3+s4+p1, # Li
                4 : s1+s2+s3+s4+p1, # Bi
                5 : s1+s2+p1+p2+d1, # B
                6 : s1+s2+p1+p2+d1, # C
                7 : s1+s2+p1+p2+d1, # N
                8 : s1+s2+p1+p2+d1, # O
                9 : s1+s2+p1+p2+d1, # F
                10: s1+s2+p1+p2+d1, # Ne
                11: s1+s2+s3+s4+p1+p2+d1, # Na
                12: s1+s2+s3+s4+p1+p2+d1, # Mg
                # 13: Al
                14: s1+s2+p1+p2+d1, # Si
                15: s1+s2+p1+p2+d1, # P
                16: s1+s2+p1+p2+d1, # S
                17: s1+s2+p1+p2+d1, # Cl
                18: s1+s2+p1+p2+d1, # Ar
                19: s1+s2+s3+s4+p1+p2+d1, # K
                20: s1+s2+s3+s4+p1+p2+d1, # Ca
                21: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Sc
                22: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ti
                23: s1+s2+s3+s4+p1+p2+d1+d2+f1, # V
                24: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cr
                25: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Mn
                26: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Fe
                27: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Co
                28: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ni
                29: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cu
                30: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Zn
                31: s1+s2+p1+p2+d1+d2+f1, # Ga
                32: s1+s2+p1+p2+d1+d2+f1, # Ge
                33: s1+s2+p1+p2+d1, # As
                34: s1+s2+p1+p2+d1, # Se
                35: s1+s2+p1+p2+d1, # Br
                36: s1+s2+p1+p2+d1, # Kr
                37: s1+s2+s3+s4+p1+p2+d1, # Rb
                38: s1+s2+s3+s4+p1+p2+d1, # Sr
                39: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Y
                40: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Zr
                41: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Nb
                42: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Mo
                43: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Tc
                44: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ru
                45: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Rh
                46: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Pd
                47: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ag
                48: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cd
                49: s1+s2+p1+p2+d1+d2+f1, # In
                50: s1+s2+p1+p2+d1+d2+f1, # Sn
                51: s1+s2+p1+p2+d1+d2+f1, # Sb
                52: s1+s2+p1+p2+d1+d2+f1, # Te
                53: s1+s2+p1+p2+d1+d2+f1, # I
                54: s1+s2+p1+p2+d1+d2+f1, # Xe
                55: s1+s2+s3+s4+p1+p2+d1, # Cs
                56: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ba
                #
                79: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Au
                80: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Hg
                81: s1+s2+p1+p2+d1+d2+f1, # Tl
                82: s1+s2+p1+p2+d1+d2+f1, # Pb
                83: s1+s2+p1+p2+d1+d2+f1, # Bi
            })()

            elif self.nao_max == 40:
                self.index_change = torch.LongTensor([0,1,2,3,5,6,4,8,9,7,11,12,10,14,15,13,18,19,17,20,16,23,24,22,25,21,29,30,28,31,27,32,26,36,37,35,38,34,39,33])       
                self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x3o+1x3o")
                # 这个列表应该遵循 abacus 的顺序
                self.minus_index = torch.LongTensor([5,6,8,9,11,12,14,15,17,18,22,23,27,28,31,32,34,35,38,39]) 
                self.basis_def = (lambda s1=[0],
                       s2=[1],
                       s3=[2],
                       s4=[3],
                       p1=[4,5,6],
                       p2=[7,8,9],
                       p3=[10,11,12],
                       p4=[13,14,15],
                       d1=[16,17,18,19,20],
                       d2=[21,22,23,24,25],
                       f1=[26,27,28,29,30,31,32],
                       f2=[33,34,35,36,37,38,39]: {
                    Element('Ag').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Al').Z: s1+s2+s3+s4+p1+p2+p3+p4+d1, 
                    Element('Ar').Z: s1+s2+p1+p2+d1, 
                    Element('As').Z: s1+s2+p1+p2+d1, 
                    Element('Au').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Ba').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Be').Z: s1+s2+s3+s4+p1, 
                    Element('B').Z: s1+s2+p1+p2+d1, 
                    Element('Bi').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Br').Z: s1+s2+p1+p2+d1, 
                    Element('Ca').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Cd').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('C').Z: s1+s2+p1+p2+d1, 
                    Element('Cl').Z: s1+s2+p1+p2+d1, 
                    Element('Co').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Cr').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Cs').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Cu').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Fe').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('F').Z: s1+s2+p1+p2+d1, 
                    Element('Ga').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Ge').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('He').Z: s1+s2+p1, 
                    Element('Hf').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # Hf_gga_10au_100Ry_4s2p2d2f.orb
                    Element('H').Z: s1+s2+p1, 
                    Element('Hg').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('I').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('In').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Ir').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('K').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Kr').Z: s1+s2+p1+p2+d1, 
                    Element('Li').Z: s1+s2+s3+s4+p1, 
                    Element('Mg').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Mn').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Mo').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Na').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Nb').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Ne').Z: s1+s2+p1+p2+d1, 
                    Element('N').Z: s1+s2+p1+p2+d1, 
                    Element('Ni').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('O').Z: s1+s2+p1+p2+d1, 
                    Element('Os').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Pb').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Pd').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('P').Z: s1+s2+p1+p2+d1, 
                    Element('Pt').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Rb').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Re').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Rh').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Ru').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Sb').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Sc').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Se').Z: s1+s2+p1+p2+d1, 
                    Element('S').Z: s1+s2+p1+p2+d1, 
                    Element('Si').Z: s1+s2+p1+p2+d1, 
                    Element('Sn').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Sr').Z: s1+s2+s3+s4+p1+p2+d1, 
                    Element('Ta').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # Ta_gga_10au_100Ry_4s2p2d2f.orb
                    Element('Tc').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Te').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Ti').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Tl').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('V').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('W').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # W_gga_10au_100Ry_4s2p2d2f.orb
                    Element('Xe').Z: s1+s2+p1+p2+d1+d2+f1, 
                    Element('Y').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Zn').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                    Element('Zr').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1,
                    })()
            else:
                raise NotImplementedError
        # PASP 基组定义
        elif self.ham_type == 'pasp':   
            self.row = self.col = o3.Irreps("1x1o")
        else:
            raise NotImplementedError
        
        self._init_irreps()
        self.cg_cal = ClebschGordan()

        # -- 哈密顿量预测网络 --
        if soc_switch:
            if self.ham_type is not 'openmx':
                # 如果不是 openmx，强制使用 su2 基组
                self.soc_basis == 'su2'
            
            if self.soc_basis == 'su2':
                # 在位项网络 (SU2 基组)
                self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=2*self.ham_irreps)

                # 异位项网络 (SU2 基组)
                self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)  
                self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=2*self.ham_irreps)
            
            elif self.soc_basis == 'so3':
                # 在位项网络 (SO3 基组)
                self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)

                # 异位项网络 (SO3 基组)
                self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)  
                self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
                
                # 用于预测 SOC 强度的标量网络 (在位)
                self.onsitenet_residual_ksi = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.ksi_on_scalar = Linear(irreps_in=irreps_in_node, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify())
    
                # 用于预测 SOC 强度的标量网络 (异位)
                self.offsitenet_residual_ksi = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)
                self.ksi_off_scalar = Linear(irreps_in=irreps_in_edge, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify()) 
            
            else:
                raise NotImplementedError(f"{soc_basis} not supportted!")
                                
        else: # 非 SOC 情况
            # 在位项网络
            self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                     nonlinearity_type = self.nonlinearity_type, resnet=True) 
            self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)
               
            # 异位项网络
            self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                     nonlinearity_type = self.nonlinearity_type, resnet=True)  
            self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
        
        # -- 重叠矩阵预测网络 --
        if not self.ham_only:            
            self.onsitenet_s = Ham_layer(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node,irreps_out=self.ham_irreps, 
                                                 nonlinearity_type = self.nonlinearity_type, resnet=True)
            self.offsitenet_s = Ham_layer(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, irreps_out=self.ham_irreps, 
                                                 nonlinearity_type = self.nonlinearity_type, resnet=True)
                 
    def _init_irreps(self):
        """初始化哈密顿量矩阵的不可约表示 (Irreps)。

        根据是否启用自旋轨道耦合（SOC）以及所选的基组（'so3' 或 'su2'），
        此方法确定描述哈密顿量矩阵元素所需的 e3nn 不可约表示。
        """
        self.ham_irreps_dim = []
        self.ham_irreps = o3.Irreps()

        if self.soc_switch and (self.soc_basis == 'su2'): 
            # 对于 SU(2) 基组的 SOC 计算，使用 e3TensorDecomp 工具来分解哈密顿量
            out_js_list = []
            for _, li in self.row:
                for _, lj in self.col:
                    out_js_list.append((li.l, lj.l))

            self.hamDecomp = e3TensorDecomp(None, out_js_list, default_dtype_torch=torch.float32, nao_max=self.nao_max, spinful=True)
            self.ham_irreps = self.hamDecomp.required_irreps_out
        else:
            # 对于非 SOC 或 SO(3) 基组的 SOC，通过 Clebsch-Gordan 系数手动构建 Irreps
            for _, li in self.row:
                for _, lj in self.col:
                    for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                        # 宇称为 (-1)^(l_i + l_j)
                        self.ham_irreps += o3.Irrep(L, (-1)**(li.l+lj.l))

        for irs in self.ham_irreps:
            self.ham_irreps_dim.append(irs.dim)
        self.ham_irreps_dim = torch.LongTensor(self.ham_irreps_dim)

    def matrix_merge(self, sph_split):   
        """将以不可约表示形式存在的哈密顿量元素合并成矩阵块。

        此函数执行逆球谐张量积操作，将网络输出的等变特征（sph_split）
        通过 Clebsch-Gordan 系数重新组合成标准的矩阵形式。

        Args:
            sph_split (list of torch.Tensor): 包含不同不可约表示的特征张量列表。

        Returns:
            torch.Tensor: 合并后的矩阵块，形状为 [N, nao_max * nao_max]。
        """
        block = torch.zeros(sph_split[0].shape[0], self.nao_max, self.nao_max).type_as(sph_split[0])
        
        idx = 0 # 用于访问正确的 irreps 的索引
        start_i = 0
        for _, li in self.row:
            n_i = 2*li.l+1
            start_j = 0
            for _, lj in self.col:
                n_j = 2*lj.l+1
                for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                    # 计算逆球谐张量积
                    cg = math.sqrt(2*L+1)*self.cg_cal(li.l, lj.l, L).unsqueeze(0)
                    product = (cg*sph_split[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    # 将乘积添加到矩阵块的适当部分
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
            
        return block.reshape(-1, self.nao_max*self.nao_max)
    
    def change_index(self, hamiltonian):
        """根据不同 DFT 软件的原子轨道顺序，调整输出矩阵元素的顺序。

        Args:
            hamiltonian (torch.Tensor): 输入的哈密顿量矩阵。

        Returns:
            torch.Tensor: 调整顺序后的哈密顿量矩阵。
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
                # 对于某些基组（如 SIESTA），一些轨道需要乘以 -1
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]
            hamiltonian = hamiltonian.reshape(-1, self.nao_max**2)                
        return hamiltonian
    
    def convert_to_mole_Ham(self, data, Hon, Hoff):
        """将局域的在位和异位哈密顿量转换为完整的分子/晶胞哈密顿量。

        Args:
            data (AtomicDataDict.Type): 包含图结构和原子信息的字典。
            Hon (torch.Tensor): 在位（on-site）哈密顿量块。
            Hoff (torch.Tensor): 异位（off-site）哈密顿量块。

        Returns:
            torch.Tensor: 构建的完整哈密顿量矩阵。
        """
        # 获取每个晶格中的原子数
        max_atoms = torch.max(data['node_counts']).item()
                
        # 解析原子轨道基组
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        basis_def_temp = copy.deepcopy(self.basis_def)
        # key 是原子序数, value 是占据的轨道
        for k in self.basis_def.keys():
            basis_def_temp[k] = [num-1 for num in self.basis_def[k]]
            basis_definition[k][basis_def_temp[k]] = 1
            
        orb_mask = basis_definition[data['z']].view(-1, max_atoms*self.nao_max) # shape: [Nbatch, max_atoms*nao_max]  
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # shape: [Nbatch, max_atoms*nao_max, max_atoms*nao_max]
        orb_mask = orb_mask.view(-1, max_atoms*self.nao_max) # shape: [Natoms*nao_max, max_atoms*nao_max]
        
        atom_idx = torch.arange(data['z'].shape[0]).type_as(data['z'])
        H = torch.zeros([data['z'].shape[0], max_atoms, self.nao_max**2]).type_as(Hon) # shape: [Natoms, max_atoms, nao_max**2]
        H[atom_idx, atom_idx%max_atoms] = Hon
        H[data['edge_index'][0], data['edge_index'][1]%max_atoms] = Hoff
        H = H.reshape(
            data['z'].shape[0], max_atoms, self.nao_max, self.nao_max) # shape: [Natoms, max_atoms, nao_max, nao_max]

        # 调整哈密顿量的维度
        H = H.permute((0, 2, 1, 3))
        H = H.reshape(data['z'].shape[0] * self.nao_max, max_atoms * self.nao_max)

        # 掩码填充的轨道
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / (data['z'].shape[0]/max_atoms)))
        H = H.reshape(-1, orbs)              
        return H
    
    def cat_onsite_and_offsite(self, data, Hon, Hoff):
        """将批处理的在位和异位哈密顿量块连接成一个张量。

        Args:
            data (AtomicDataDict.Type): 包含图结构和原子信息的字典。
            Hon (torch.Tensor): 在位（on-site）哈密顿量块。
            Hoff (torch.Tensor): 异位（off-site）哈密顿量块。

        Returns:
            torch.Tensor: 连接后的哈密顿量张量。
        """
        # 获取每个晶格中的原子数
        node_counts = data['node_counts']
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        #
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0)
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        #
        H = []
        for i in range(len(node_counts)):
            H.append(Hon_split[i])
            H.append(Hoff_split[i])
        H = torch.cat(H, dim=0)
        return H 
    
    def symmetrize_Hon(self, Hon, sign:str='+'):
        """对称化在位哈密顿量/重叠矩阵。

        Args:
            Hon (torch.Tensor): 在位矩阵块。
            sign (str, optional): 对称化方式, '+' 表示 H, '-' 表示 H-H.T。默认为 '+'。

        Returns:
            torch.Tensor: 对称化后的矩阵块。
        """
        if self.symmetrize:
            Hon = Hon.reshape(-1, self.nao_max, self.nao_max)
            if sign == '+':
                Hon = 0.5*(Hon + Hon.permute((0,2,1)))
            else:
                Hon = 0.5*(Hon - Hon.permute((0,2,1)))
            Hon = Hon.reshape(-1, self.nao_max**2)
            return Hon
        else:
            return Hon
    
    def symmetrize_Hoff(self, Hoff, inv_edge_idx, sign:str='+'):
        """对称化异位哈密顿量/重叠矩阵。

        利用 H_ij = H_ji.T 的性质。

        Args:
            Hoff (torch.Tensor): 异位矩阵块。
            inv_edge_idx (torch.Tensor): 反向边的索引。
            sign (str, optional): 对称化方式。默认为 '+'。

        Returns:
            torch.Tensor: 对称化后的矩阵块。
        """
        if self.symmetrize:
            Hoff = Hoff.reshape(-1, self.nao_max, self.nao_max)
            if sign == '+':
                Hoff = 0.5*(Hoff + Hoff[inv_edge_idx].permute((0,2,1)))
            else:
                Hoff = 0.5*(Hoff - Hoff[inv_edge_idx].permute((0,2,1)))
            Hoff = Hoff.reshape(-1, self.nao_max**2)
            return Hoff
        else:
            return Hoff
    
    def cal_band_energy_debug(self, Hon, Hoff, Son, Soff, data, export_reciprocal_values:bool=False):
        """计算能带结构（调试版本）。

        此函数通过在倒空间中构建哈密顿量和重叠矩阵，然后求解广义本征值问题
        来计算能带。它包含了详细的步骤，用于调试和验证。

        Args:
            Hon, Hoff (torch.Tensor): 预测的在位和异位哈密顿量。
            Son, Soff (torch.Tensor): 预测的在位和异位重叠矩阵。
            data (AtomicDataDict.Type): 输入数据。
            export_reciprocal_values (bool, optional): 是否导出倒空间矩阵。

        Returns:
        """
        # 目前此函数只能用于计算 openmx 哈密顿量的能带
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        # 解析原子轨道基组
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data['z']] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data['node_counts'].tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # 设置价电子数
        num_val = torch.zeros((99,)).type_as(data['z'])
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data['z']] # shape: [Natoms]
        num_val = scatter(num_val, data['batch'], dim=0) # shape: [Nbatch]
                
        # 初始化 band_num_win
        if self.band_num_control is not None:
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # shape: (Nbatch,)
             
        # 按批次分离 Hon 和 Hoff
        node_counts = data['node_counts']
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'], node_counts.tolist(), dim=0)
        Son_pred_split = torch.split(Son, node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0) # shape: (Nbatch,)
        edge_num_shift = torch.cumsum(edge_num, dim=0) - edge_num
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data['Soff'], edge_num.tolist(), dim=0)
        Soff_pred_split = torch.split(Soff, edge_num.tolist(), dim=0)
        if export_reciprocal_values:
            dSon_split = torch.split(data['dSon'], node_counts.tolist(), dim=0)
            dSoff_split = torch.split(data['dSoff'], edge_num.tolist(), dim=0)
        
        band_energy = []
        wavefunction = []
        H_reciprocal = []
        H_sym = []
        S_reciprocal = []
        dS_reciprocal = []
        gap = []
        for idx in range(Nbatch):
            k_vec = data['k_vecs'][idx]   
            natoms = data['node_counts'][idx]
            
            # 初始化 HK 和 SK   
            coe = torch.exp(2j*torch.pi*torch.sum(data['nbr_shift'][edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (nedges, 1, 3)*(1, num_k, 3) -> (nedges, num_k)     
            
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))  
            SK_pred = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))           
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            na = torch.arange(natoms).type_as(j)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # shape (num_k, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            SK_pred[:,na,na,:,:] +=  Son_pred_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK_pred)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            
            for iedge in range(edge_num[idx]):
                # shape (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
                j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                HK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Hoff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK_pred[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_pred_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
            
            if export_reciprocal_values:
                for iedge in range(edge_num[idx]):
                    j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    dSK[:,j_idx,i_idx,:,:,:] += coe[iedge,:,None,None,None] * dSoff_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,iedge,:,:,:]

            HK = torch.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK_pred = torch.swapaxes(SK_pred,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK_pred = SK_pred.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, natoms, nao_max, natoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # 掩码 HK 和 SK
            HK = torch.masked_select(HK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(HK.numel()/self.num_k))
            HK = HK.reshape(self.num_k, norbs, norbs)
            
            SK = torch.masked_select(SK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)

            SK_pred = torch.masked_select(SK_pred, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK_pred.numel()/self.num_k))
            SK_pred = SK_pred.reshape(self.num_k, norbs, norbs)
            
            if export_reciprocal_values:
                dSK = torch.masked_select(dSK, orb_mask_batch[idx].unsqueeze(-1).repeat(self.num_k,1,1,3) > 0)
                dSK = dSK.reshape(self.num_k, norbs, norbs, 3)            
            
            # 计算能带能量
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # 将波函数系数转换回原始基组
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy 实现（用于对比验证）
            """
            HK_t = HK.detach().cpu().numpy()
            SK_t = SK.detach().cpu().numpy()
            from scipy.linalg import eigh
            eigen = []
            eigen_vecs = []
            for ik in range(self.num_k):
                w, v = eigh(a=HK_t[ik], b=SK_t[ik])
                eigen.append(w)
                eigen_vecs.append(v)
            eigen_vecs = np.array(eigen_vecs) # (nk, nbands, nbands)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK_t, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (numk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]  
            orbital_energies, orbital_coefficients = torch.Tensor(eigen).type_as(data['pos']), torch.complex(torch.Tensor(eigen_vecs.real), torch.Tensor(eigen_vecs.imag)).type_as(HK)
            """
            
            if export_reciprocal_values:
                # 归一化波函数
                lamda = torch.einsum('nai, nij, naj -> na', torch.conj(orbital_coefficients), SK, orbital_coefficients).real
                lamda = 1/torch.sqrt(lamda) # shape: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK_pred)
                dS_reciprocal.append(dSK)
            
            if self.band_num_control is not None:
                orbital_energies = orbital_energies[:,:band_num_win[idx]]
                orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]                
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)  
            H_sym.append(Hs.view(-1)) 
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))    
            
        band_energy = torch.cat(band_energy, dim=0) # [shape:(Nbands, num_k)]
        
        gap = torch.cat(gap, dim=0)
        
        if export_reciprocal_values:
            wavefunction = torch.stack(wavefunction, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            HK = torch.stack(H_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            SK = torch.stack(S_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]  
            dSK = torch.stack(dS_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs, 3]   
            return band_energy, wavefunction, HK, SK, dSK, gap
        else:
            wavefunction = [wavefunction[idx].reshape(-1) for idx in range(Nbatch)]
            wavefunction = torch.cat(wavefunction, dim=0) # shape:[Nbatch*num_k*norbs*norbs]
            H_sym = torch.cat(H_sym, dim=0) # shape:(Nbatch*num_k*norbs*norbs)
            return band_energy, wavefunction, gap, H_sym   
    
    def cal_band_energy(self, Hon, Hoff, data, export_reciprocal_values:bool=False):
        """
        计算能带结构（优化版本）。

        此函数是 `cal_band_energy_debug` 的优化版本，使用更高效的张量操作
        来构建倒空间矩阵，以提高计算速度。

        Args:
            Hon, Hoff (torch.Tensor): 预测的在位和异位哈密顿量。
            data (AtomicDataDict.Type): 输入数据。
            export_reciprocal_values (bool, optional): 是否导出倒空间矩阵。

        Returns:
            tuple: 包含能带能量、波函数、带隙等信息的元组。
        """
        # 目前此函数只能用于计算 openmx 哈密顿量的能带
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        # 解析原子轨道基组
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data['z']] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data['node_counts'].tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # 设置价电子数
        num_val = torch.zeros((99,)).type_as(data['z'])
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data['z']] # shape: [Natoms]
        num_val = scatter(num_val, data['batch'], dim=0) # shape: [Nbatch]
                
        # 初始化 band_num_win
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # shape: (Nbatch,)   
             
        # 按批次分离 Hon 和 Hoff
        node_counts = data['node_counts']
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'], node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0) # shape: (Nbatch,)
        edge_num_shift = torch.cumsum(edge_num, dim=0) - edge_num
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data['Soff'], edge_num.tolist(), dim=0)
        if export_reciprocal_values:
            dSon_split = torch.split(data['dSon'], node_counts.tolist(), dim=0)
            dSoff_split = torch.split(data['dSoff'], edge_num.tolist(), dim=0)
        
        band_energy = []
        wavefunction = []
        H_reciprocal = []
        H_sym = []
        S_reciprocal = []
        dS_reciprocal = []
        gap = []
        for idx in range(Nbatch):
            k_vec = data['k_vecs'][idx]   
            natoms = data['node_counts'][idx]
            
            # 初始化 HK 和 SK       
            coe = torch.exp(2j*torch.pi*torch.sum(data['nbr_shift'][edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (nedges, 1, 3)*(1, num_k, 3) -> (nedges, num_k)     
            
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))            
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            na = torch.arange(natoms).type_as(j)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # shape (num_k, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            # 计算所有边的索引
            edge_indices = torch.arange(edge_num[idx], device=j.device)
            j_indices = j[edge_num_shift[idx] + edge_indices] - node_counts_shift[idx]
            i_indices = i[edge_num_shift[idx] + edge_indices] - node_counts_shift[idx]

            # 重塑系数和矩阵以方便矢量化操作
            Hoff_reshaped = Hoff_split[idx].reshape(edge_num[idx], self.nao_max, self.nao_max)
            Soff_reshaped = Soff_split[idx].reshape(edge_num[idx], self.nao_max, self.nao_max)
            
            for k_idx in range(self.num_k):
                # 预先计算当前 k 点的所有值
                coe_k = coe[:edge_num[idx], k_idx].unsqueeze(-1).unsqueeze(-1)  # shape: (edge_num, 1, 1)
                HK_values = coe_k * Hoff_reshaped  # shape: (edge_num, nao_max, nao_max)
                SK_values = coe_k * Soff_reshaped  # shape: (edge_num, nao_max, nao_max)

                # 使用 index_put 进行累加
                HK[k_idx] = torch.index_put(HK[k_idx], (j_indices, i_indices), HK_values, accumulate=True)
                SK[k_idx] = torch.index_put(SK[k_idx], (j_indices, i_indices), SK_values, accumulate=True)

            if export_reciprocal_values:
                dSoff_reshaped = dSoff_split[idx].reshape(edge_num[idx], self.nao_max, self.nao_max, 3)

                for k_idx in range(self.num_k):
                    # 估算当前 k 点的所有值
                    coe_k = coe[:edge_num[idx], k_idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: (edge_num, 1, 1, 1)
                    dSK_values = coe_k * dSoff_reshaped  # shape: (edge_num, nao_max, nao_max, 3)

                    # 使用 index_put 进行累加
                    dSK[k_idx] = torch.index_put(dSK[k_idx], (j_indices, i_indices), dSK_values, accumulate=True)

            HK = torch.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, natoms, nao_max, natoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # 掩码 HK 和 SK
            HK = torch.masked_select(HK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(HK.numel()/self.num_k))
            HK = HK.reshape(self.num_k, norbs, norbs)
            
            SK = torch.masked_select(SK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            if export_reciprocal_values:
                dSK = torch.masked_select(dSK, orb_mask_batch[idx].unsqueeze(-1).repeat(self.num_k,1,1,3) > 0)
                dSK = dSK.reshape(self.num_k, norbs, norbs, 3)            
            
            # 计算能带能量
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # 将波函数系数转换回原始基组
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy 实现（用于对比验证）
            """
            HK_t = HK.detach().cpu().numpy()
            SK_t = SK.detach().cpu().numpy()
            from scipy.linalg import eigh
            eigen = []
            eigen_vecs = []
            for ik in range(self.num_k):
                w, v = eigh(a=HK_t[ik], b=SK_t[ik])
                eigen.append(w)
                eigen_vecs.append(v)
            eigen_vecs = np.array(eigen_vecs) # (nk, nbands, nbands)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK_t, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (numk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]  
            orbital_energies, orbital_coefficients = torch.Tensor(eigen).type_as(data['pos']), torch.complex(torch.Tensor(eigen_vecs.real), torch.Tensor(eigen_vecs.imag)).type_as(HK)
            """
            
            if export_reciprocal_values:
                # 归一化波函数
                lamda = torch.einsum('nai, nij, naj -> na', torch.conj(orbital_coefficients), SK, orbital_coefficients).real
                lamda = 1/torch.sqrt(lamda) # shape: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK)
                dS_reciprocal.append(dSK)
            
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    orbital_energies = orbital_energies[:,:band_num_win[idx]]   
                    orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]
                else:
                    if isinstance(self.band_num_control, float):
                        self.band_num_control = max([1, int(self.band_num_control*numc)])
                    else:
                        self.band_num_control = min([self.band_num_control, numc])
                    orbital_energies = orbital_energies[:,numc-self.band_num_control:numc+self.band_num_control]   
                    orbital_coefficients = orbital_coefficients[:,numc-self.band_num_control:numc+self.band_num_control,:]               
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)  
            H_sym.append(Hs.view(-1))   
            
        band_energy = torch.cat(band_energy, dim=0) # [shape:(Nbands, num_k)]
        
        gap = torch.cat(gap, dim=0)
        
        if export_reciprocal_values:
            wavefunction = torch.stack(wavefunction, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            HK = torch.stack(H_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            SK = torch.stack(S_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]  
            dSK = torch.stack(dS_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs, 3]   
            return band_energy, wavefunction, HK, SK, dSK, gap
        else:
            wavefunction = [wavefunction[idx].reshape(-1) for idx in range(Nbatch)]
            wavefunction = torch.cat(wavefunction, dim=0) # shape:[Nbatch*num_k*norbs*norbs]
            H_sym = torch.cat(H_sym, dim=0) # shape:(Nbatch*num_k*norbs*norbs)
            return band_energy, wavefunction, gap, H_sym
    
    def cal_band_energy_soc(self, Hsoc_on_real, Hsoc_on_imag, Hsoc_off_real, Hsoc_off_imag, data):
        """计算包含自旋轨道耦合（SOC）的能带结构。

        此函数处理包含 SOC 效应的哈密顿量，该哈密顿量通常是复数且非对角
        （在自旋空间中）。它分别构建哈密顿量的四个自旋分量（up-up, up-down,
        down-up, down-down），然后在倒空间中组合它们，最后求解本征值问题。

        Args:
            Hsoc_on_real, Hsoc_on_imag (torch.Tensor): 在位SOC哈密顿量的实部和虚部。
            Hsoc_off_real, Hsoc_off_imag (torch.Tensor): 异位SOC哈密顿量的实部和虚部。
            data (AtomicDataDict.Type): 输入数据。

        Returns:
            tuple: 包含能带能量和波函数的元组。
        """
        # 目前此函数只能用于计算 openmx 哈密顿量的能带
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        Hsoc_on_real = Hsoc_on_real.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_on_imag = Hsoc_on_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_off_real = Hsoc_off_real.reshape(-1, 2*self.nao_max, 2*self.nao_max) 
        Hsoc_off_imag = Hsoc_off_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        
        # 解析原子轨道基组
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data['z']] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data['node_counts'].tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # 设置价电子数
        num_val = torch.zeros((99,)).type_as(data['z'])
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data['z']] # shape: [Natoms]
        num_val = scatter(num_val, data['batch'], dim=0) # shape: [Nbatch]
                
        # 初始化 band_num_win
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # shape: (Nbatch,)       
            
        # 按批次分离 Hon 和 Hoff
        node_counts = data['node_counts']
        Hon_split = torch.split(Hsoc_on_real, node_counts.tolist(), dim=0)
        iHon_split = torch.split(Hsoc_on_imag, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'].reshape(-1, self.nao_max, self.nao_max), node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0)
        Hoff_split = torch.split(Hsoc_off_real, edge_num.tolist(), dim=0)
        iHoff_split = torch.split(Hsoc_off_imag, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data['Soff'].reshape(-1, self.nao_max, self.nao_max), edge_num.tolist(), dim=0)
        
        cell_shift_split = torch.split(data['cell_shift'], edge_num.tolist(), dim=0)
        nbr_shift_split = torch.split(data['nbr_shift'], edge_num.tolist(), dim=0)
        edge_index_split = torch.split(data['edge_index'], edge_num.tolist(), dim=1)
        node_num = torch.cumsum(node_counts, dim=0) - node_counts
        edge_index_split = [edge_index_split[idx]-node_num[idx] for idx in range(len(node_num))]
        
        band_energy = []
        wavefunction = []
        for idx in range(Nbatch):
            k_vec = data['k_vecs'][idx]   
            natoms = data['node_counts'][idx].item() 
            
            # 初始化晶胞索引
            cell_shift_tuple = [tuple(c) for c in cell_shift_split[idx].detach().cpu().tolist()]
            cell_shift_set = set(cell_shift_tuple)
            cell_shift_list = list(cell_shift_set)
            cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple]
            cell_index = torch.LongTensor(cell_index).type_as(data['edge_index'])
            ncells = len(cell_shift_set)
            
            # 初始化 SK
            phase = torch.view_as_complex(torch.zeros((self.num_k, ncells, 2)).type_as(data['Son']))
            phase[:, cell_index] = torch.exp(2j*torch.pi*torch.sum(nbr_shift_split[idx][None,:,:]*k_vec[:,None,:], dim=-1))
            na = torch.arange(natoms).type_as(j)

            S_cell = torch.view_as_complex(torch.zeros((ncells, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(data['Son']))
            S_cell[cell_index, edge_index_split[idx][0], edge_index_split[idx][1], :, :] += Soff_split[idx]

            SK = torch.einsum('ijklm, ni->njklm', S_cell, phase) # (nk, natoms, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx][None,na,:,:]
            SK = torch.swapaxes(SK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(self.num_k, natoms*self.nao_max, natoms*self.nao_max)
            SK = SK[:,orb_mask_batch[idx] > 0]
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            I = torch.eye(2).type_as(data['Hon'])
            SK = torch.kron(I, SK)
            
            # 初始化 Hsoc
            # 在位项
            H11 = Hon_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHon_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hon_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHon_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hon_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHon_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hon_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHon_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hon_soc = [H11, H12, H21, H22]
            # 异位项
            H11 = Hoff_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHoff_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hoff_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHoff_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hoff_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHoff_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hoff_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHoff_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hoff_soc = [H11, H12, H21, H22]
            
            # 初始化 HK
            HK_list = []
            for Hon, Hoff in zip(Hon_soc, Hoff_soc):
                H_cell = torch.view_as_complex(torch.zeros((ncells, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(data['Son']))
                H_cell[cell_index, edge_index_split[idx][0], edge_index_split[idx][1], :, :] += Hoff    

                HK = torch.einsum('ijklm, ni->njklm', H_cell, phase) # (nk, natoms, natoms, nao_max, nao_max)
                HK[:,na,na,:,:] +=  Hon[None,na,:,:] # shape (nk, natoms, nao_max, nao_max)

                HK = torch.swapaxes(HK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
                HK = HK.reshape(self.num_k, natoms*self.nao_max, natoms*self.nao_max)

                # 掩码 HK
                HK = HK[:, orb_mask_batch[idx] > 0]
                norbs = int(math.sqrt(HK.numel()/self.num_k))
                HK = HK.reshape(self.num_k, norbs, norbs)
        
                HK_list.append(HK)

            HK = torch.cat([torch.cat([HK_list[0],HK_list[1]], dim=-1), torch.cat([HK_list[2],HK_list[3]], dim=-1)],dim=-2)
        
            # 计算能带能量
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)   
            # 将波函数系数转换回原始基组
            orbital_coefficients = torch.bmm(L_t_inv, orbital_coefficients) # shape:(num_k, Nbands, Nbands)
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    orbital_energies = orbital_energies[:,:band_num_win[idx]]   
                    orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]
                else:
                    orbital_energies = orbital_energies[:,num_val[idx]-self.band_num_control:num_val[idx]+self.band_num_control]   
                    orbital_coefficients = orbital_coefficients[:,num_val[idx]-self.band_num_control:num_val[idx]+self.band_num_control,:]
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)
        return torch.cat(band_energy, dim=0), torch.cat(wavefunction, dim=0).reshape(-1)
    
    def mask_Ham(self, Hon, Hoff, data):
        """根据每种元素的实际轨道基组，屏蔽哈密顿量/重叠矩阵中不存在的元素。

        由于模型为所有原子预测一个固定大小（nao_max * nao_max）的矩阵块，
        此函数用于将那些对应于该原子实际未使用的轨道的矩阵元素置零。

        Args:
            Hon (torch.Tensor): 在位矩阵块。
            Hoff (torch.Tensor): 异位矩阵块。
            data (AtomicDataDict.Type): 输入数据。

        Returns:
            tuple: 屏蔽后的在位和异位矩阵块 (Hon_mask, Hoff_mask)。
        """
        # 解析原子轨道基组定义
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
        
        # 保存原始形状
        original_shape_on = Hon.shape
        original_shape_off = Hoff.shape
        
        if len(original_shape_on) > 2:
            Hon = Hon.reshape(original_shape_on[0], -1)
        if len(original_shape_off) > 2:
            Hoff = Hoff.reshape(original_shape_off[0], -1)
        
        # 首先屏蔽 Hon   
        orb_mask = basis_definition[data['z']].view(-1, self.nao_max) # shape: [Natoms, nao_max] 
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # shape: [Natoms, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Natoms, nao_max*nao_max]
        
        Hon_mask = torch.zeros_like(Hon)
        Hon_mask[orb_mask>0] = Hon[orb_mask>0]
        
        # 屏蔽 Hoff
        j = data['edge_index'][0]
        i = data['edge_index'][1]       
        orb_mask_j = basis_definition[data['z'][j]].view(-1, self.nao_max) # shape: [Nedges, nao_max]
        orb_mask_i = basis_definition[data['z'][i]].view(-1, self.nao_max) # shape: [Nedges, nao_max] 
        orb_mask = orb_mask_j[:,:,None] * orb_mask_i[:,None,:]       # shape: [Nedges, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Nedges, nao_max*nao_max]
        
        Hoff_mask = torch.zeros_like(Hoff)
        Hoff_mask[orb_mask>0] = Hoff[orb_mask>0]

        # 以原始形状输出结果
        Hon_mask = Hon_mask.reshape(original_shape_on)
        Hoff_mask = Hoff_mask.reshape(original_shape_off)
        
        return Hon_mask, Hoff_mask    
    
    def construct_Hsoc(self, H, iH):
        """将 SOC 哈密顿量的实部和虚部组合成一个复数张量。

        Args:
            H (torch.Tensor): 哈密顿量的实部。
            iH (torch.Tensor): 哈密顿量的虚部。

        Returns:
            torch.Tensor: 复数形式的 SOC 哈密顿量。
        """
        Hsoc = torch.view_as_complex(torch.zeros((H.shape[0], (2*self.nao_max)**2, 2)).type_as(H))
        Hsoc = H + 1.0j*iH
        return Hsoc
    
    def reduce(self, coefficient):
        """对 SOC 耦合系数进行降维/平均化处理。

        在某些基组定义下，属于同一轨道壳层（如 p, d, f）的多个轨道
        被假定共享同一个 SOC 耦合系数。此函数通过对这些轨道的系数
        取平均值来实现这一点。

        Args:
            coefficient (torch.Tensor): 原始的 SOC 耦合系数矩阵。

        Returns:
            torch.Tensor: 处理后的系数矩阵。
        """
        if self.nao_max == 14:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            
        elif self.nao_max == 19:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 14:19] = torch.mean(coefficient[:, 14:19], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 14:19] = torch.mean(coefficient[:, :, 14:19], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)

        elif self.nao_max == 26:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 14:19] = torch.mean(coefficient[:, 14:19], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 19:26] = torch.mean(coefficient[:, 19:26], dim=1, keepdim=True).expand(coefficient.shape[0], 7, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 14:19] = torch.mean(coefficient[:, :, 14:19], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 19:26] = torch.mean(coefficient[:, :, 19:26], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 7)
        return coefficient.view(coefficient.shape[0], -1)
         
    def forward(self, data, graph_representation: dict = None):
        """模型的前向传播。

        Args:
            data (AtomicDataDict.Type): 输入数据字典。
            graph_representation (dict, optional): 由表示网络预先计算的特征字典。

        Returns:
            dict: 包含预测的哈密顿量、重叠矩阵、能带能量等结果的字典。
        """
        # 准备 data['hamiltonian'] 和 ['data.overlap']
        if 'hamiltonian' not in data:
            data['hamiltonian'] = self.cat_onsite_and_offsite(data, data['Hon'], data['Hoff'])
        if 'overlap' not in data:
            data['overlap'] = self.cat_onsite_and_offsite(data, data['Son'], data['Soff'])
        
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        
        # 在批处理中计算 inv_edge_index
        inv_edge_idx = data['inv_edge_idx']
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0)
        edge_num = torch.cumsum(edge_num, dim=0) - edge_num
        inv_edge_idx = inv_edge_idx + edge_num[data['batch'][j]]
        
        # 计算在位哈密顿量
        self.ham_irreps_dim = self.ham_irreps_dim.type_as(j)  
        
        if not self.ham_only:
            # -- 重叠矩阵预测 --
            node_sph = self.onsitenet_s(node_attr)
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Son = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Son = self.change_index(Son)
        
            # 对 Son 强制施加厄米对称性
            Son = self.symmetrize_Hon(Son)

            # 计算异位重叠矩阵
            # 计算边的贡献
            edge_sph = self.offsitenet_s(edge_attr)
            edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
            Soff = self.matrix_merge(edge_sph)
        
            Soff = self.change_index(Soff)        
            # 对 Soff 强制施加厄米对称性
            Soff = self.symmetrize_Hoff(Soff, inv_edge_idx)
        
            if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
                Son, Soff = self.mask_Ham(Son, Soff, data)

        if self.soc_switch:
            # -- SOC 哈密顿量预测 --
            if self.soc_basis == 'so3':
                # -- SO(3) 基组下的 SOC --
                # 计算非相对论部分
                node_sph = self.onsitenet_residual(node_attr)     
                node_sph = self.onsitenet_linear(node_sph) 
                node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
                Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
                
                Hon = self.change_index(Hon)
    
                # Impose Hermitian symmetry for Hon
                Hon = self.symmetrize_Hon(Hon)            
    
                # Calculate the off-site Hamiltonian
                # Calculate the contribution of the edges       
                edge_sph = self.offsitenet_residual(edge_attr)
                edge_sph = self.offsitenet_linear(edge_sph)
                edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
                Hoff = self.matrix_merge(edge_sph)
            
                Hoff = self.change_index(Hoff)        
                Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
                
                Hon, Hoff = self.mask_Ham(Hon, Hoff, data)
                
                # 构建 Hsoc
                # 预测 SOC 耦合强度 ksi
                node_sph = self.onsitenet_residual_ksi(node_attr)
                ksi_on = self.ksi_on_scalar(node_sph)
                ksi_on = self.reduce(ksi_on)
                
                edge_sph = self.offsitenet_residual_ksi(edge_attr)
                ksi_off = self.ksi_off_scalar(edge_sph)
                ksi_off = self.reduce(ksi_off)  
                
                # 组装 SOC 哈密顿量的实部
                Hsoc_on_real = torch.zeros((Hon.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hon)
                Hsoc_on_real[:,:self.nao_max,:self.nao_max] = Hon.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,:self.nao_max,self.nao_max:] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,self.nao_max:,:self.nao_max] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,self.nao_max:,self.nao_max:] = Hon.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real = Hsoc_on_real.reshape(-1, (2*self.nao_max)**2)
                
                # 组装 SOC 哈密顿量的虚部
                Hsoc_on_imag = torch.zeros((Hon.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hon)
                Hsoc_on_imag[:,:self.nao_max,:self.nao_max] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,2]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,:self.nao_max, self.nao_max:] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,0]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,self.nao_max:,:self.nao_max] = -self.symmetrize_Hon((ksi_on*data['Lon'][:,:,0]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,self.nao_max:,self.nao_max:] = -self.symmetrize_Hon((ksi_on*data['Lon'][:,:,2]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag = Hsoc_on_imag.reshape(-1, (2*self.nao_max)**2)
                
                Hsoc_off_real = torch.zeros((Hoff.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hoff)
                Hsoc_off_real[:,:self.nao_max,:self.nao_max] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,:self.nao_max,self.nao_max:] = self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,1]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,self.nao_max:,:self.nao_max] = self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,1]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,self.nao_max:,self.nao_max:] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real = Hsoc_off_real.reshape(-1, (2*self.nao_max)**2)
                
                Hsoc_off_imag = torch.zeros((Hoff.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hoff)
                Hsoc_off_imag[:,:self.nao_max,:self.nao_max] = self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,2]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,:self.nao_max, self.nao_max:] = self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,0]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,self.nao_max:,:self.nao_max] = -self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,0]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,self.nao_max:,self.nao_max:] = -self.symmetrize_Hoff((ksi_off*data['Loff'][:,:,2]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag = Hsoc_off_imag.reshape(-1, (2*self.nao_max)**2)
            
            elif self.soc_basis == 'su2':
                # -- SU(2) 基组下的 SOC --
                node_sph = self.onsitenet_residual(node_attr)
                node_sph = self.onsitenet_linear(node_sph) 
                
                Hon = self.hamDecomp.get_H(node_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                Hon = self.change_index(Hon)
                Hon = Hon.reshape(-1, 2, 2, self.nao_max, self.nao_max)                
                Hon = torch.swapaxes(Hon, 2, 3) # shape (Nnodes, 2, nao_max, 2, nao_max)
    
                # Calculate the off-site Hamiltonian
                # Calculate the contribution of the edges       
                edge_sph = self.offsitenet_residual(edge_attr)
                edge_sph = self.offsitenet_linear(edge_sph)
                
                Hoff = self.hamDecomp.get_H(edge_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                Hoff = self.change_index(Hoff)
                Hoff = Hoff.reshape(-1, 2, 2, self.nao_max, self.nao_max)
                Hoff = torch.swapaxes(Hoff, 2, 3) # shape (Nedges, 2, nao_max, 2, nao_max)    
                
                # 屏蔽零元素
                for i in range(2):
                    for j in range(2):
                        Hon[:,i,:,j,:], Hoff[:,i,:,j,:] = self.mask_Ham(Hon[:,i,:,j,:], Hoff[:,i,:,j,:], data)
                Hon = Hon.reshape(-1, (2*self.nao_max)**2)
                Hoff = Hoff.reshape(-1, (2*self.nao_max)**2)
                # 构建四个部分
                Hsoc_on_real =  Hon.real
                Hsoc_off_real = Hoff.real
                Hsoc_on_imag = Hon.imag
                Hsoc_off_imag = Hoff.imag
                
            else:
                raise NotImplementedError
            
            if self.add_H0:
                Hsoc_on_real =  Hsoc_on_real + data['Hon0']
                Hsoc_off_real = Hsoc_off_real + data['Hoff0']
                Hsoc_on_imag = Hsoc_on_imag + data['iHon0']
                Hsoc_off_imag = Hsoc_off_imag + data['iHoff0']
            
            Hsoc_real = self.cat_onsite_and_offsite(data, Hsoc_on_real, Hsoc_off_real)
            Hsoc_imag = self.cat_onsite_and_offsite(data, Hsoc_on_imag, Hsoc_off_imag)
            
            data['hamiltonian_real'] = self.cat_onsite_and_offsite(data, data['Hon'], data['Hoff'])
            data['hamiltonian_imag'] = self.cat_onsite_and_offsite(data, data['iHon'], data['iHoff'])
            
            #Hsoc = self.construct_Hsoc(Hsoc_real, Hsoc_imag)
            #data['hamiltonian'] = self.construct_Hsoc(data['hamiltonian_real'], data['hamiltonian_imag'])
            
            Hsoc = torch.cat((Hsoc_real, Hsoc_imag), dim=0)
            data['hamiltonian'] = torch.cat((data['hamiltonian_real'], data['hamiltonian_imag']), dim=0)
            
            if self.calculate_band_energy:
                # -- 能带计算 --
                k_vecs = []
                for idx in range(data['batch'][-1]+1):
                    cell = data['cell']
                    # 生成 K 点路径
                    if self.k_path is not None:
                        kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                    else:
                        lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                        k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
                    k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
                    k_vec = torch.Tensor(k_vec).type_as(Hon)
                    k_vecs.append(k_vec)  
                data['k_vecs'] = torch.stack(k_vecs, dim=0)
                band_energy, wavefunction = self.cal_band_energy_soc(Hsoc_on_real, Hsoc_on_imag, Hsoc_off_real, Hsoc_off_imag, data) 
                with torch.no_grad():
                    data['band_energy'], data['wavefunction'] = self.cal_band_energy_soc(data['Hon'], data['iHon'], data['Hoff'], data['iHoff'], data)
            else:
                band_energy = None
                wavefunction = None
        else:
            # -- 非 SOC 哈密顿量预测 --
            node_sph = self.onsitenet_residual(node_attr)
            node_sph = self.onsitenet_linear(node_sph) 
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Hon = self.change_index(Hon)
        
            # Impose Hermitian symmetry for Hon
            Hon = self.symmetrize_Hon(Hon)
            if self.add_H0:
                Hon = Hon + data['Hon0']
               
            # Calculate the off-site Hamiltonian
            # Calculate the contribution of the edges       
            edge_sph = self.offsitenet_residual(edge_attr)
            edge_sph = self.offsitenet_linear(edge_sph)
            edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
            Hoff = self.matrix_merge(edge_sph)
        
            Hoff = self.change_index(Hoff)        
            # Impose Hermitian symmetry for Hoff
            Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
            if self.add_H0:
                Hoff = Hoff + data['Hoff0']
        
            if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
                Hon, Hoff = self.mask_Ham(Hon, Hoff, data)
        
            if self.calculate_band_energy:
                k_vecs = []
                for idx in range(data['batch'][-1]+1):
                    cell = data['cell']
                    # 生成 K 点路径
                    if isinstance(self.k_path, list):
                        kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                    elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                        # 自动生成高对称点路径
                        latt = cell[idx].detach().cpu().numpy()*au2ang
                        pos = torch.split(data['pos'], data['node_counts'].tolist(), dim=0)[idx].detach().cpu().numpy()*au2ang
                        species = torch.split(data['z'], data['node_counts'].tolist(), dim=0)[idx]
                        struct = Structure(lattice=latt, species=[Element.from_Z(k.item()).symbol for k in species], coords=pos, coords_are_cartesian=True)
                        # 初始化 k_path 和标签
                        kpath_seek = KPathSeek(structure = struct)
                        klabels = []
                        for lbs in kpath_seek.kpath['path']:
                            klabels += lbs
                        # 移除相邻的重复点
                        res = [klabels[0]]
                        [res.append(x) for x in klabels[1:] if x != res[-1]]
                        klabels = res
                        k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
                        try:
                            kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(k_path, self.num_k)
                        except:
                            lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                            k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    else:
                        lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                        k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
                    k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
                    k_vec = torch.Tensor(k_vec).type_as(Hon)
                    k_vecs.append(k_vec)  
                data['k_vecs'] = torch.stack(k_vecs, dim=0)
                if self.export_reciprocal_values:
                    if self.ham_only:
                        band_energy, wavefunction, HK, SK, dSK, gap = self.cal_band_energy(Hon, Hoff, data, True)
                        H_sym = None
                    else:
                        band_energy, wavefunction, HK, SK, dSK, gap = self.cal_band_energy_debug(Hon, Hoff, Son, Soff, data, True)
                        H_sym = None
                else:
                    band_energy, wavefunction, gap, H_sym = self.cal_band_energy(Hon, Hoff, data)
                with torch.no_grad():
                    data['band_energy'], data['wavefunction'], data['band_gap'], data['H_sym'] = self.cal_band_energy(data['Hon'], data['Hoff'], data)
            else:
                band_energy = None
                wavefunction = None
                gap = None
                H_sym = None
        
        # -- 组合最终结果 --
        # openmx
        if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
            if self.soc_switch:
                result = {'hamiltonian': Hsoc, 'hamiltonian_real':Hsoc_real, 
                          'hamiltonian_imag':Hsoc_imag, 'band_energy': band_energy, 
                          'wavefunction': wavefunction, 'iHon': Hsoc_on_imag, 'iHoff': Hsoc_off_imag}
            else:
                H = self.cat_onsite_and_offsite(data, Hon, Hoff)
                result = {'hamiltonian': H, 'band_energy': band_energy, 'wavefunction': wavefunction, 'band_gap':gap, 'H_sym': H_sym}
                if self.export_reciprocal_values:
                    result.update({'HK':HK, 'SK':SK, 'dSK': dSK})
        else:
            raise NotImplementedError
        
        if not self.ham_only:                
            # openmx
            if self.ham_type in ['openmx','pasp', 'siesta','abacus']:
                S = self.cat_onsite_and_offsite(data, Son, Soff)
            else:
                raise NotImplementedError
            result.update({'overlap': S})
        
        return result