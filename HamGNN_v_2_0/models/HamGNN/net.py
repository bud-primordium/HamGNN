'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-08-24 16:14:48
LastEditors: Yang Zhong
LastEditTime: 2025-06-09 14:13:10
'''
import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import numpy as np
from .BaseModel import BaseModel
from e3nn import o3
from ..layers import GaussianSmearing, BesselBasis, cuttoff_envelope, CosineCutoff
from ..basis import (
    ExponentialGaussianRadialBasisFunctions, 
    ExponentialBernsteinRadialBasisFunctions,
    GaussianRadialBasisFunctions,
    BernsteinRadialBasisFunctions
)
from ..Toolbox.nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs
)
from ..Toolbox.nequip.nn import AtomwiseLinear
from ..Toolbox.nequip.data import AtomicDataDict
from .Attention_kan import (RadialBasisEdgeEncoding,
                            AttentionBlockE3, 
                            PairInteractionBlock, 
                            PairInteractionEmbeddingBlock,
                            CorrProductBlock, 
                            HamLayer, 
                            ConvBlockE3,  
                            ClebschGordanCoefficients,
                            SoftUnitStepCutoff)
from pymatgen.core.periodic_table import Element
from .clebsch_gordan import ClebschGordan
from ..e3_layers import e3TensorDecomp
import math, copy
from easydict import EasyDict
from torch_scatter import scatter
import opt_einsum as oe
from .kpoint_gen import kpoints_generator
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from e3nn.math import soft_unit_step
from ..utils import blockwise_2x2_concat, extract_elements_above_threshold, upgrade_tensor_precision

au2ang = 0.5291772083

class HamGNNConvE3(BaseModel):
    """基于 E(3) 等变图卷积的 HamGNN 模型。

    该模型通过堆叠多个等变卷积层来学习原子及其相互作用的表示。
    每一层都会更新原子（节点）和原子对（边）的特征，同时严格保持在旋转下的等变性。
    最终输出节点和边的等变特征，用于后续的物理属性预测。

    Attributes:
        num_types (int): 系统中原子种类的数量。
        cutoff (float): 径向截断半径。
        num_layers (int): 等变卷积层的数量。
        irreps_node_features (o3.Irreps): 节点特征的不可约表示。
        radial_basis_functions (nn.Module): 径向基函数模块。
        atomic_embedding (nn.Module): 将原子种类转换为独热编码的模块。
        spharm_edges (nn.Module): 计算边向量的球谐函数的模块。
        radial_basis (nn.Module): 编码边长度的径向基函数模块。
        pair_embedding (nn.Module): 嵌入原子对相互作用的初始特征。
        chemical_embedding (nn.Module): 将独热编码嵌入到初始节点特征中。
        convolutions (nn.ModuleList): 等变卷积层列表。
        corr_products (nn.ModuleList): （可选）相关性乘积层列表。
        pair_interactions (nn.ModuleList): 原子对相互作用层列表。
    """
    def __init__(self, config):
        """
        Args:
            config (EasyDict): 
                包含模型配置参数的对象。关键属性包括：

                - HamGNN_pre.radius_type (str): 用于确定邻居的半径类型。
                - HamGNN_pre.radius_scale (float): 内部图构建时的半径缩放因子，必须大于1.0。
                - HamGNN_pre.num_types (int): 原子种类的总数。
                - HamGNN_pre.irreps_edge_sh (str): 边球谐函数的不可约表示。
                - HamGNN_pre.edge_sh_normalization (str): 球谐函数的归一化方式 ('component' 或 'norm')。
                - HamGNN_pre.edge_sh_normalize (bool): 是否对球谐函数进行归一化。
                - HamGNN_pre.build_internal_graph (bool): 是否在模型内部动态构建图。
                - HamGNN_pre.use_corr_prod (bool): 是否使用相关性乘积块。
                - HamGNN_pre.cutoff (float): 截断半径。
                - HamGNN_pre.rbf_func (str): 径向基函数的类型 (例如, 'gaussian', 'bessel')。
                - HamGNN_pre.num_radial (int): 径向基函数的数量。
                - HamGNN_pre.num_layers (int): 卷积层数。
                - HamGNN_pre.irreps_node_features (str): 节点特征的不可约表示。
                - HamGNN_pre.use_kan (bool): 是否在 MLP 中使用 KAN (Kolmogorov-Arnold Networks) 层。
                - HamGNN_pre.radial_MLP (list): 径向 MLP 的隐藏层维度。
                - HamGNN_pre.correlation (int): 相关性乘积块中的相关性阶数。
                - HamGNN_pre.num_hidden_features (int): 相关性乘积块中的隐藏特征数。
        """
        if 'radius_scale' not in config.HamGNN_pre:
            config.HamGNN_pre.radius_scale = 1.0
        else:
            assert config.HamGNN_pre.radius_scale > 1.0, "半径缩放因子必须大于 1.0。"
        super().__init__(radius_type=config.HamGNN_pre.radius_type, radius_scale=config.HamGNN_pre.radius_scale)
        
        # --- 配置设定 ---
        self.num_types = config.HamGNN_pre.num_types  # 原子种类数量
        self.set_features = True  # 是否将独热编码设置为节点特征
        self.irreps_edge_sh = o3.Irreps(config.HamGNN_pre.irreps_edge_sh)  # 边球谐函数的不可约表示
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        self.build_internal_graph = config.HamGNN_pre.build_internal_graph
        if 'use_corr_prod' not in config.HamGNN_pre:
            self.use_corr_prod = False
        else:
            self.use_corr_prod = config.HamGNN_pre.use_corr_prod
        
        # --- 径向基函数 ---
        self.cutoff = config.HamGNN_pre.cutoff
        self.rbf_func = config.HamGNN_pre.rbf_func.lower()
        self.num_radial = config.HamGNN_pre.num_radial                
        if self.rbf_func == 'gaussian':
            self.radial_basis_functions = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'bessel':
            self.radial_basis_functions = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(self.num_radial, self.cutoff)
        elif self.rbf_func == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.num_radial, self.cutoff)
        elif self.rbf_func == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(self.num_radial, self.cutoff)
        else:
            raise ValueError(f'不支持的径向基函数: {self.rbf_func}')
        
        self.num_layers = config.HamGNN_pre.num_layers  # 卷积层数量
        self.irreps_node_features = o3.Irreps(config.HamGNN_pre.irreps_node_features)  # 节点特征的不可约表示
        
        # --- 原子嵌入模块 ---
        self.atomic_embedding = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features)
        
        # --- 边向量的球谐函数 ---
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, 
                                                       edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize=self.edge_sh_normalize)
        
        # --- 边长度的径向基函数 ---
        self.cutoff_func = CosineCutoff(self.cutoff)
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.radial_basis_functions, 
                                                    cutoff=self.cutoff_func)

       # --- 边特征嵌入模块 ---
        use_kan = config.HamGNN_pre.use_kan
        self.radial_MLP = config.HamGNN_pre.radial_MLP
        self.pair_embedding = PairInteractionEmbeddingBlock(irreps_node_feats=self.atomic_embedding.irreps_out['node_attrs'],
                                        irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],
                                        irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                        irreps_edge_feats=self.irreps_node_features,
                                        irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                        use_kan=use_kan,
                                        radial_MLP=self.radial_MLP)
        
        # --- 原子化学环境嵌入 ---
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.atomic_embedding.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        # --- 定义卷积层、相关性乘积层和原子对相互作用层 ---
        correlation = config.HamGNN_pre.correlation
        num_hidden_features = config.HamGNN_pre.num_hidden_features
        
        self.convolutions = torch.nn.ModuleList()
        if self.use_corr_prod:
            self.corr_products = torch.nn.ModuleList()
        self.pair_interactions = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            conv = ConvBlockE3(irreps_in=self.irreps_node_features,
                                               irreps_out=self.irreps_node_features,
                                               irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                               irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],                      
                                               irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                               radial_MLP=self.radial_MLP,
                                               use_skip_connections=True,
                                               use_kan=use_kan)
            self.convolutions.append(conv)
            
            if self.use_corr_prod:
                corr_product = CorrProductBlock(
                    irreps_node_feats=self.irreps_node_features,
                    num_hidden_features=num_hidden_features,
                    correlation=correlation,
                    num_elements=self.num_types,
                    use_skip_connections=True
                )
                self.corr_products.append(corr_product)

            pair_interaction = PairInteractionBlock(irreps_node_feats=self.irreps_node_features,
                                                    irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                                    irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],
                                                    irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                                    irreps_edge_feats=self.irreps_node_features,
                                                    use_skip_connections=True if i > 0 else False,
                                                    use_kan=use_kan,
                                                    radial_MLP=self.radial_MLP)
            self.pair_interactions.append(pair_interaction)
    
    def forward(self, data):
        """执行模型的前向传播。

        Args:
            data (Data or EasyDict): 
                输入的图数据对象，遵循 PyG 或 Nequip 的数据格式。
                需要包含 `pos`, `z`, `edge_index` 等原子结构信息。

        Returns:
            EasyDict:
                一个包含最终节点和边等变特征的字典。
                - 'node_attr': 节点的等变特征张量。
                - 'edge_attr': 边的等变特征张量。
        """
        if torch.get_default_dtype() == torch.float64:
            upgrade_tensor_precision(data)

        # 图构建现在在数据预处理阶段完成，模型直接使用预处理好的数据
        graph = data
        
        # --- 特征提取与嵌入 ---
        self.atomic_embedding(graph)  # 原子种类独热编码
        self.spharm_edges(graph)      # 边向量的球谐函数
        self.radial_basis(graph)      # 边长度的径向基
        self.pair_embedding(graph)    # 原子对特征嵌入
        self.chemical_embedding(graph)# 初始化学环境特征
        
        # --- 等变卷积层堆叠 ---
        for i in range(self.num_layers):
            self.convolutions[i](graph)
            if self.use_corr_prod:
                self.corr_products[i](graph)
            self.pair_interactions[i](graph)
            
        # --- 整理并返回最终的图表示 ---
        graph_representation = EasyDict()
        graph_representation['node_attr'] = graph[AtomicDataDict.NODE_FEATURES_KEY]
        # 如果数据包含 matching_edges (由 DynamicGraphTransform 生成)，则使用匹配的边
        if 'matching_edges' in graph:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY][graph.matching_edges]
        else:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY]
        return graph_representation


class HamGNNTransformer(BaseModel):
    """基于 E(3) 等变 Transformer 的 HamGNN 模型。

    该模型采用等变自注意力机制来捕捉原子间的相互作用，替代了传统的卷积操作。
    它允许模型在更新节点特征时，动态地权衡来自不同邻居的信息。
    模型的整体结构与 `HamGNNConvE3` 相似，但核心交互层换成了 `AttentionBlockE3`。

    Attributes:
        num_types (int): 系统中原子种类的数量。
        cutoff (float): 径向截断半径。
        num_layers (int): 等变 Transformer 层的数量。
        irreps_node_features (o3.Irreps): 节点特征的不可约表示。
        num_heads (int): 自注意力机制中的头数。
        radial_basis_functions (nn.Module): 径向基函数模块。
        atomic_embedding (nn.Module): 将原子种类转换为独热编码的模块。
        spharm_edges (nn.Module): 计算边向量的球谐函数的模块。
        radial_basis (nn.Module): 编码边长度的径向基函数模块。
        pair_embedding (nn.Module): 嵌入原子对相互作用的初始特征。
        chemical_embedding (nn.Module): 将独热编码嵌入到初始节点特征中。
        orb_transformers (nn.ModuleList): 等变 Transformer 层列表。
        corr_products (nn.ModuleList): 相关性乘积层列表。
        pair_interactions (nn.ModuleList): 原子对相互作用层列表。
    """
    def __init__(self, config):
        """
        Args:
            config (EasyDict): 
                包含模型配置参数的对象。关键属性包括：

                - HamGNN_pre.radius_type (str): 用于确定邻居的半径类型。
                - HamGNN_pre.radius_scale (float): 内部图构建时的半径缩放因子，必须大于1.0。
                - HamGNN_pre.num_types (int): 原子种类的总数。
                - HamGNN_pre.irreps_edge_sh (str): 边球谐函数的不可约表示。
                - HamGNN_pre.edge_sh_normalization (str): 球谐函数的归一化方式 ('component' 或 'norm')。
                - HamGNN_pre.edge_sh_normalize (bool): 是否对球谐函数进行归一化。
                - HamGNN_pre.build_internal_graph (bool): 是否在模型内部动态构建图。
                - HamGNN_pre.cutoff (float): 截断半径。
                - HamGNN_pre.rbf_func (str): 径向基函数的类型 (例如, 'gaussian', 'bessel')。
                - HamGNN_pre.num_radial (int): 径向基函数的数量。
                - HamGNN_pre.num_layers (int): Transformer 层数。
                - HamGNN_pre.irreps_node_features (str): 节点特征的不可约表示。
                - HamGNN_pre.num_heads (int): 注意力头的数量。
                - HamGNN_pre.use_kan (bool): 是否在 MLP 中使用 KAN (Kolmogorov-Arnold Networks) 层。
                - HamGNN_pre.radial_MLP (list): 径向 MLP 的隐藏层维度。
                - HamGNN_pre.correlation (int): 相关性乘积块中的相关性阶数。
                - HamGNN_pre.num_hidden_features (int): 相关性乘积块中的隐藏特征数。
        """
        if 'radius_scale' not in config.HamGNN_pre:
            config.HamGNN_pre.radius_scale = 1.0
        else:
            assert config.HamGNN_pre.radius_scale > 1.0, "半径缩放因子必须大于 1.0。"
        super().__init__(radius_type=config.HamGNN_pre.radius_type, radius_scale=config.HamGNN_pre.radius_scale)
        
        # --- 配置设定 ---
        self.num_types = config.HamGNN_pre.num_types  # 原子种类数量
        self.set_features = True  # 是否将独热编码设置为节点特征
        self.irreps_edge_sh = o3.Irreps(config.HamGNN_pre.irreps_edge_sh)  # 边球谐函数的不可约表示
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        self.build_internal_graph = config.HamGNN_pre.build_internal_graph

        # --- 径向基函数 ---
        self.cutoff = config.HamGNN_pre.cutoff
        self.rbf_func = config.HamGNN_pre.rbf_func.lower()
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func == 'gaussian':
            self.radial_basis_functions = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'bessel':
            self.radial_basis_functions = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(self.num_radial, self.cutoff)
        elif self.rbf_func == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.num_radial, self.cutoff)
        elif self.rbf_func == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(self.num_radial, self.cutoff)
        else:
            raise ValueError(f'不支持的径向基函数: {self.rbf_func}')
        
        self.num_layers = config.HamGNN_pre.num_layers  # Transformer 层数
        self.irreps_node_features = o3.Irreps(config.HamGNN_pre.irreps_node_features)  # 节点特征的不可约表示
        
        # --- 原子嵌入模块 ---
        self.atomic_embedding = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features)
        
        # --- 边向量的球谐函数 ---
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, 
                                                       edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize=self.edge_sh_normalize)
        
        # --- 边长度的径向基函数 ---
        self.cutoff_func = CosineCutoff(self.cutoff)
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.radial_basis_functions, 
                                                    cutoff=self.cutoff_func)
        
        # --- 边特征嵌入模块 ---
        use_kan = config.HamGNN_pre.use_kan
        self.radial_MLP = config.HamGNN_pre.radial_MLP
        self.pair_embedding = PairInteractionEmbeddingBlock(irreps_node_feats=self.atomic_embedding.irreps_out['node_attrs'],
                                        irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],
                                        irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                        irreps_edge_feats=self.irreps_node_features,
                                        irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                        use_kan=use_kan,
                                        radial_MLP=self.radial_MLP)
        
        # --- 原子化学环境嵌入 ---
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.atomic_embedding.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        # --- 定义 Transformer 层 ---
        self.num_heads = config.HamGNN_pre.num_heads
        correlation = config.HamGNN_pre.correlation
        num_hidden_features = config.HamGNN_pre.num_hidden_features
        
        self.orb_transformers = torch.nn.ModuleList()
        self.corr_products = torch.nn.ModuleList()
        self.pair_interactions = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            orb_transformer = AttentionBlockE3(irreps_in=self.irreps_node_features,
                                               irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                               irreps_out=self.irreps_node_features,
                                               irreps_edge_feats=self.irreps_node_features,
                                               irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],                      
                                               irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                               num_heads=self.num_heads, 
                                               max_radius=self.cutoff,
                                               radial_MLP=self.radial_MLP,
                                               use_skip_connections=True,
                                               use_kan=use_kan)
            self.orb_transformers.append(orb_transformer)

            corr_product = CorrProductBlock(
                irreps_node_feats=self.irreps_node_features,
                num_hidden_features=num_hidden_features,
                correlation=correlation,
                num_elements=self.num_types,
                use_skip_connections=True
            )
            self.corr_products.append(corr_product)

            pair_interaction = PairInteractionBlock(irreps_node_feats=self.irreps_node_features,
                                                    irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                                                    irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],
                                                    irreps_edge_embed=self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                                    irreps_edge_feats=self.irreps_node_features,
                                                    use_skip_connections=True,
                                                    use_kan=use_kan,
                                                    radial_MLP=self.radial_MLP)
            self.pair_interactions.append(pair_interaction)
    
    def forward(self, data):
        """执行模型的前向传播。

        Args:
            data (Data or EasyDict): 
                输入的图数据对象，遵循 PyG 或 Nequip 的数据格式。
                需要包含 `pos`, `z`, `edge_index` 等原子结构信息。

        Returns:
            EasyDict:
                一个包含最终节点和边等变特征的字典。
                - 'node_attr': 节点的等变特征张量。
                - 'edge_attr': 边的等变特征张量。
        """
        # 图构建现在在数据预处理阶段完成，模型直接使用预处理好的数据
        graph = data
        
        # --- 特征提取与嵌入 ---
        self.atomic_embedding(graph)
        self.spharm_edges(graph)
        self.radial_basis(graph)
        self.pair_embedding(graph)
        self.chemical_embedding(graph)

        # --- 等变 Transformer 层堆叠 ---
        for i in range(self.num_layers):
            self.orb_transformers[i](graph)
            self.corr_products[i](graph)
            self.pair_interactions[i](graph)
            
        # --- 整理并返回最终的图表示 ---
        graph_representation = EasyDict()
        graph_representation['node_attr'] = graph[AtomicDataDict.NODE_FEATURES_KEY]
        # 如果数据包含 matching_edges (由 DynamicGraphTransform 生成)，则使用匹配的边
        if 'matching_edges' in graph:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY][graph.matching_edges]
        else:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY]
        return graph_representation


class HamGNNPlusPlusOut(nn.Module):
    """HamGNN 的输出模块，用于构建物理哈密顿量并计算相关属性。

    这个模块接收图神经网络编码的等变特征，并利用这些特征作为系数，
    通过 Clebsch-Gordan 张量积将它们组合成完整的哈密顿量 (H) 和重叠矩阵 (S)。
    它能够处理多种 DFT 软件（如 OpenMX, SIESTA, ABACUS）的基组，支持自旋轨道耦合（SOC）、
    自旋约束计算，并能最终求解能带结构。

    .. note::
       这个类的实现非常复杂，因为它紧密地耦合了物理学（量子化学、固体物理）
       和 E(3) 等变神经网络的数学原理。

    Attributes:
        nao_max (int): 预设的最大原子轨道数。
        ham_type (str): 使用的 DFT 基组类型 (例如 'openmx')。
        ham_only (bool): 如果为 True，则只计算哈密顿量，不计算重叠矩阵。
        soc_switch (bool): 是否启用自旋轨道耦合计算。
        spin_constrained (bool): 是否进行自旋约束计算。
        ham_irreps (o3.Irreps): 哈密顿量/重叠矩阵块的不可约表示。
        onsitenet_h (nn.Module): 用于从节点特征预测 onsite 哈密顿量块的网络。
        offsitenet_h (nn.Module): 用于从边特征预测 offsite 哈密顿量块的网络。
        cg_cal (ClebschGordanCoefficients): 用于计算 Clebsch-Gordan 系数的工具。
    """
    def __init__(self, 
                 irreps_in_node: Union[int, str, o3.Irreps] = None, 
                 irreps_in_edge: Union[int, str, o3.Irreps] = None, 
                 nao_max: int = 14, 
                 return_forces: bool = False, 
                 create_graph: bool = False, 
                 ham_type: str = 'openmx', 
                 ham_only: bool = False, 
                 symmetrize: bool = True, 
                 include_triplet: bool = False, 
                 calculate_band_energy: bool = False, 
                 num_k: int = 8, 
                 k_path: Union[list, np.ndarray, tuple] = None, 
                 band_num_control: dict = None, 
                 soc_switch: bool = True, 
                 nonlinearity_type: str = 'gate', 
                 export_reciprocal_values: bool = False, 
                 add_H0: bool = False, 
                 soc_basis: str = 'so3',
                 spin_constrained: bool = False, 
                 use_learned_weight: bool = True, 
                 minMagneticMoment: float = 0.5, 
                 collinear_spin: bool = False,
                 zero_point_shift: bool = False,
                 add_H_nonsoc: bool = False,
                 get_nonzero_mask_tensor: bool = False):
        """
        Args:
            irreps_in_node (o3.Irreps): 输入的节点特征的不可约表示。
            irreps_in_edge (o3.Irreps): 输入的边特征的不可约表示。
            nao_max (int): 系统中单个原子的最大轨道数。
            return_forces (bool): 是否计算并返回力。
            create_graph (bool): 是否在模块内部创建图。
            ham_type (str): 哈密顿量的类型，决定了基组的选择 (e.g., 'openmx', 'siesta', 'abacus')。
            ham_only (bool): 若为 True，则只计算哈密顿量，跳过重叠矩阵。
            symmetrize (bool): 是否对哈密顿量和重叠矩阵进行对称化处理。
            include_triplet (bool): 是否包含三体相互作用 (当前未完全实现)。
            calculate_band_energy (bool): 是否计算能带结构。
            num_k (int): K 点路径上的采样点数量。
            k_path (list/np.ndarray): 定义计算能带的高对称 K 点路径。
            band_num_control (dict): 控制不同原子类型计算的能带数量。
            soc_switch (bool): 是否启用自旋轨道耦合 (SOC)。
            nonlinearity_type (str): 在 HamLayer 中使用的非线性激活函数类型 (e.g., 'gate')。
            export_reciprocal_values (bool): 是否导出倒易空间中的矩阵 (H(k), S(k))。
            add_H0 (bool): 是否添加一个初始的哈密顿量项 H0。
            soc_basis (str): SOC 计算使用的基组 ('so3' 或 'su2')。
            spin_constrained (bool): 是否进行自旋约束计算。
            use_learned_weight (bool): 在自旋约束中是否使用可学习的权重。
            minMagneticMoment (float): 最小磁矩阈值。
            collinear_spin (bool): 是否处理共线自旋。
            zero_point_shift (bool): 是否进行零点能量校正。
            add_H_nonsoc (bool): 是否添加非 SOC 哈密顿量。
            get_nonzero_mask_tensor (bool): 是否获取非零矩阵元素的掩码张量。
        """
        
        super().__init__()

        if return_forces:
            self.derivative = True
        else:
            self.derivative = False

        self.create_graph = create_graph

        # 确定是否计算力
        self.compute_forces = return_forces

        # 是否在前向传播中创建图
        self.create_graph = create_graph

        # 原子轨道的最大总数
        self.nao_max = nao_max

        # 哈密顿量类型
        self.ham_type = ham_type.lower()

        # 是否只计算哈密顿量
        self.ham_only = ham_only

        # 是否对称化哈密顿量
        self.symmetrize = symmetrize

        # 是否包含三体相互作用
        self.include_triplet = include_triplet

        # 是否开启自旋轨道耦合
        self.soc_switch = soc_switch

        # 非线性激活函数类型
        self.nonlinearity_type = nonlinearity_type

        # 是否导出倒易空间值
        self.export_reciprocal_values = export_reciprocal_values

        # 是否添加初始哈密顿项 H0
        self.add_H0 = add_H0

        # 自旋约束
        self.spin_constrained = spin_constrained

        # 是否使用可学习的哈密顿量权重
        self.use_learned_weight = use_learned_weight

        # 最小磁矩
        self.minMagneticMoment = minMagneticMoment

        # 是否考虑共线自旋
        self.collinear_spin = collinear_spin

        # 自旋轨道耦合基组
        self.soc_basis = soc_basis.lower()

        # 能带结构计算相关参数
        self.calculate_band_energy = calculate_band_energy
        self.num_k = num_k
        self.k_path = k_path
        
        # 其他参数
        self.add_quartic = False
        
        # 其他参数
        self.zero_point_shift = zero_point_shift
        self.add_H_nonsoc = add_H_nonsoc
        self.get_nonzero_mask_tensor = get_nonzero_mask_tensor

        # 能带数量控制
        self._set_band_num_control(band_num_control)
        
        # 初始化基组信息和不可约表示
        self._set_basis_info()
        self._init_irreps()
        
        self.cg_cal = ClebschGordanCoefficients(max_l=self.ham_irreps.lmax)

        # --- 哈密顿量预测网络 ---
        self.onsitenet_h = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)
        self.offsitenet_h = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
        
        # --- SOC 相关网络 ---
        if soc_switch:
            if self.ham_type != 'openmx':
                self.soc_basis == 'su2'
            
            # 仅用于测试目的，请谨慎使用！
            if self.soc_basis == 'su2':
                self.onsitenet_h = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=2*self.ham_irreps_su2)
                self.offsitenet_h = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=2*self.ham_irreps_su2)
            
            elif self.soc_basis == 'so3':                
                self.onsitenet_ksi = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify())
                self.offsitenet_ksi = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify())
            
            else:
                raise NotImplementedError(f"不支持的 SOC 基组: {soc_basis}！")

        # --- 自旋约束相关网络 ---
        if self.spin_constrained:
            # 交换相互作用 J
            self.onsitenet_J = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=self.J_irreps)
            self.offsitenet_J = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=self.J_irreps)
            
            # 四阶项 K
            if self.add_quartic:
                self.onsitenet_K = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=self.K_irreps)
                self.offsitenet_K = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=self.K_irreps)
            
            # 权重矩阵
            if self.use_learned_weight:
                self.onsitenet_weight = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)
                self.offsitenet_weight = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
        
        # --- 重叠矩阵预测网络 ---
        if not self.ham_only:            
            self.onsitenet_s = self._create_ham_layer(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)
            self.offsitenet_s = self._create_ham_layer(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps) 
                 
    def _init_irreps(self):
        """初始化哈密顿量所需的不可约表示 (Irreps)。

        该方法根据基组定义（`self.row` 和 `self.col`），通过计算轨道角动量
        `li` 和 `lj` 的所有可能耦合结果 `L`，来构建目标哈密顿量块的 Irreps。
        这个 Irreps 决定了神经网络需要预测哪些球谐系数。
        同时，它也为 SOC 和自旋约束情况下的 Irreps 进行初始化。
        """
        self.ham_irreps_dim = []
        
        self.ham_irreps = o3.Irreps()

        # 遍历所有行列轨道对 (li, lj)
        for _, li in self.row:
            for _, lj in self.col:
                # 根据角动量耦合规则，确定输出的 L
                for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                    # 宇称为 (-1)^(li+lj)
                    self.ham_irreps += o3.Irrep(L, (-1)**(li.l+lj.l))
        
        for irs in self.ham_irreps:
            self.ham_irreps_dim.append(irs.dim)
        
        self.ham_irreps_dim = torch.LongTensor(self.ham_irreps_dim)

        # 如果启用 SOC 且基组为 su2
        if self.soc_switch and (self.soc_basis == 'su2'): 
            out_js_list = []
            for _, li in self.row:
                for _, lj in self.col:
                    out_js_list.append((li.l, lj.l))

            self.hamDecomp = e3TensorDecomp(None, out_js_list, default_dtype_torch=torch.float32, nao_max=self.nao_max, spinful=True)
            self.ham_irreps_su2 = self.hamDecomp.required_irreps_out

        # 如果启用自旋约束
        if self.spin_constrained:
            
            self.J_irreps = o3.Irreps()
            self.K_irreps = o3.Irreps()
            
            self.J_irreps_dim = []
            self.K_irreps_dim = []
            self.Nblocks = 0            
            
            for _, li in self.row:
                for _, lj in self.col:
                    self.Nblocks += 1
                    if self.soc_switch:
                        # 对于 SOC，交换相互作用 J 是一个二阶张量 (L=0,1,2)
                        for L in range(0, 3):
                            self.J_irreps += o3.Irrep(L, 1)   # t=1, p=1
                            self.K_irreps += o3.Irrep(0, 1)  # t=1, p=1
                    else:
                        # 对于非 SOC，J 是一个标量 (L=0)
                        self.J_irreps += o3.Irrep(0, 1)   # t=1, p=1
            
            for irs in self.J_irreps:
                self.J_irreps_dim.append(irs.dim)

            for irs in self.K_irreps:
                self.K_irreps_dim.append(irs.dim)   
            
            self.J_irreps_dim = torch.LongTensor(self.J_irreps_dim)
            self.K_irreps_dim = torch.LongTensor(self.K_irreps_dim)

    def _set_basis_info(self):
        """
        根据哈密顿量类型 (`ham_type`) 和最大原子轨道数 (`nao_max`) 设置基组信息。
        这会调用特定于 `openmx`, `siesta`, `abacus` 等的方法。
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
            raise NotImplementedError(f"不支持的哈密顿量类型: '{self.ham_type}'。")

    def _set_openmx_basis(self):
        """
        为 'openmx' 类型的哈密顿量设置基组信息。
        定义了每种元素的价电子数、轨道构成 (Irreps)，以及轨道在矩阵中的索引重排方式。
        """
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
            self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e")
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
                                Element['Mn'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mn
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
                25:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mn
                42:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Mo  
                83:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Bi  
                34:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Se 
                24:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Cr 
                53:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # I
                28:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ni
                35:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Br 
                26:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Fe
                77:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ir
                52:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Te
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
            raise NotImplementedError(f"不支持的 NAO max '{self.nao_max}' for 'openmx'.")

    def _set_siesta_basis(self):
        """
        为 'siesta' 类型的哈密顿量设置基组信息。
        """
        self.num_valence = {
            1:1,2:2,
            3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
            11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
            19:1,20:2,22:12
        }
        if self.nao_max == 13:
            self.index_change = None       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([2,4,5,7,9,11]) # this list should follow the order in siesta. See spher_harm.f
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
            self.minus_index = torch.LongTensor([3,5,6,8,10,12,15,17]) # this list should follow the order in siesta. See spher_harm.f
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
                22: s1+s2+s3+p1+p2+d1+d2, # Ti, created by Qin.
            })()
        else:
            raise NotImplementedError(f"不支持的 NAO max '{self.nao_max}' for 'siesta'.")

    def _set_abacus_basis(self):
        """
        为 'abacus' 类型的哈密顿量设置基组信息。
        """
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
            self.minus_index = torch.LongTensor([5,6,8,9,11,12,16,17,21,22,25,26]) # this list should follow the order in abacus.
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
            self.minus_index = torch.LongTensor([5,6,8,9,11,12,14,15,17,18,22,23,27,28,31,32,34,35,38,39]) # this list should follow the order in abacus.
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
            raise NotImplementedError(f"不支持的 NAO max '{self.nao_max}' for 'abacus'.")

    def _set_band_num_control(self, band_num_control):
        """设置计算能带时每个原子类型贡献的能带数量。"""
        # 能带数量控制
        if band_num_control is not None and not self.export_reciprocal_values:
            if isinstance(band_num_control, dict):
                # 为保持一致性，将原子序数转换为整数
                self.band_num_control = {int(k): v for k, v in band_num_control.items()}
            elif isinstance(band_num_control, int):
                self.band_num_control = band_num_control
            else:
                self.band_num_control = None
        else:
            self.band_num_control = None

    def _create_ham_layer(self, irreps_in, irreps_out):
        """
        创建一个 `HamLayer` 实例。
        
        Args:
            irreps_in (o3.Irreps): 输入的不可约表示。
            irreps_out (o3.Irreps): 输出的不可约表示。
            
        Returns:
            HamLayer: 一个配置好的 HamLayer 实例。
        """
        return HamLayer(
            irreps_in=irreps_in,
            feature_irreps_hidden=irreps_in,
            irreps_out=irreps_out,
            nonlinearity_type=self.nonlinearity_type,
            resnet=True
        )

    def matrix_merge(self, sph_split):   
        """
        将一系列球谐系数（等变特征）通过 Clebsch-Gordan 积合并成矩阵块。

        这是模型的核心步骤之一，它将神经网络预测的抽象特征（`sph_split`）
        转换为物理上可解释的哈密顿量或重叠矩阵的组成部分。

        Args:
            sph_split (list of torch.Tensor): 
                一个张量列表，每个张量对应 `self.ham_irreps` 中的一个不可约表示的系数。
                形状为 `(N_batch, 2L+1)`。

        Returns:
            torch.Tensor: 
                合并后的矩阵块，形状为 `(N_batch, nao_max * nao_max)`。
        """
        block = torch.zeros(sph_split[0].shape[0], self.nao_max, self.nao_max).type_as(sph_split[0])
        
        idx = 0 # 用于访问正确 irreps 的索引
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

                    # 将乘积添加到块的适当部分
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
            
        return block.reshape(-1, self.nao_max*self.nao_max)

    def matrix_2rank_merge(self, sph_split):   
        """
        将球谐系数合并为二阶张量（3x3矩阵），用于 SOC 计算中的交换项 J。

        Args:
            sph_split (list of torch.Tensor): 
                对应 `self.J_irreps` 的球谐系数列表。

        Returns:
            torch.Tensor: 
                形状为 `(N_batch, N_blocks, 3, 3)` 的张量。
        """
        block = torch.zeros(sph_split[0].shape[0], self.Nblocks, 3, 3).type_as(sph_split[0]) # 形状: (N_batch, N_blocks, 3, 3)
        indices_change = torch.LongTensor([2, 0, 1])
        
        idx = 0 # 用于访问正确 irreps 的索引
        block_idx = 0
        for _, li in self.row:
            for _, lj in self.col:
                for L in range(0, 3):
                    # 计算逆球谐张量积
                    cg = math.sqrt(2*L+1)*self.cg_cal(1, 1, L).unsqueeze(0)
                    product = (cg*sph_split[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1) # 形状: (N_batch, 3, 3)

                    # 将乘积添加到块的适当部分
                    block[:,block_idx,:,:] = block[:,block_idx,:,:] + product

                    idx += 1
                block_idx += 1
    
        block = block[:, :, indices_change[:,None], indices_change[None, :]]
        return block

    def matrix_0rank_merge(self, sph_split):   
        """
        将零阶（标量）球谐系数合并为矩阵。

        这用于非 SOC 情况下的交换项 J，其中 J 是一个标量。

        Args:
            sph_split (list of torch.Tensor): 
                对应零阶 Irreps 的系数列表。

        Returns:
            torch.Tensor: 
                形状为 `(N_batch, nao_max, nao_max)` 的矩阵。
        """
        block = torch.zeros(sph_split[0].shape[0], self.nao_max, self.nao_max).type_as(sph_split[0]) # 形状: (N_batch, nao_max, nao_max)
        
        idx = 0 # 用于访问正确 irreps 的索引
        start_i = 0
        for _, li in self.row:
            n_i = 2*li.l+1
            start_j = 0
            for _, lj in self.col:
                n_j = 2*lj.l+1
                product = sph_split[idx].unsqueeze(-1).expand(-1, n_i, n_j) # 形状: (N_batch, n_i, n_j)
                # 将乘积添加到块的适当部分
                blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                blockpart += product
                idx += 1
                start_j += n_j
            start_i += n_i
        return block # 形状: (N_batch, nao_max, nao_max)
    
    def J_merge(self, lagrange):   
        """
        将拉格朗日乘子（交换场）合并成完整的交换相互作用矩阵 J。

        Args:
            lagrange (torch.Tensor): 神经网络预测的交换场系数。

        Returns:
            torch.Tensor: 
                交换相互作用矩阵 J。
                如果 `soc_switch` 为 True, 形状为 `(N_batch, nao_max, nao_max, 3, 3)`。
                否则，形状为 `(N_batch, nao_max, nao_max)`。
        """
        if self.soc_switch:
            sph_split = torch.split(lagrange, self.J_irreps_dim.tolist(), dim=-1)
            lagrange = self.matrix_2rank_merge(sph_split)  # 形状: (N_batch, N_blocks, 3, 3)

            block = torch.zeros(lagrange.shape[0], self.nao_max, self.nao_max, 3, 3).type_as(lagrange)

            block_idx = 0 # 用于访问正确块的索引
            start_i = 0
            for _, li in self.row:
                n_i = 2*li.l+1
                start_j = 0
                for _, lj in self.col:
                    n_j = 2*lj.l+1
                    block[:, start_i:start_i+n_i, start_j:start_j+n_j] = lagrange[:,block_idx].reshape(-1, 1, 1, 3, 3).expand(lagrange.shape[0], n_i, n_j, 3, 3)
                    block_idx += 1
                    start_j += n_j
                start_i += n_i
        else:
            sph_split = torch.split(lagrange, self.J_irreps_dim.tolist(), dim=-1)
            block = self.matrix_0rank_merge(sph_split)  # 形状: (N_batch, nao_max, nao_max)
            
        return block # 形状: (N_batch, nao_max, nao_max, 3, 3)

    def K_merge(self, lagrange):   
        """
        将拉格朗日乘子合并成四阶相互作用矩阵 K。

        Args:
            lagrange (torch.Tensor): 神经网络预测的四阶项系数。

        Returns:
            torch.Tensor: 四阶相互作用矩阵 K，形状 `(N_batch, nao_max, nao_max)`。
        """
        
        lagrange = lagrange.reshape(lagrange.shape[0], -1) # 形状: (N_atoms/N_edges, N_blocks)
        block = torch.zeros(lagrange.shape[0], self.nao_max, self.nao_max).type_as(lagrange)
        
        idx = 0 # 用于访问正确 irreps 的索引
        start_i = 0
        for _, li in self.row:
            n_i = 2*li.l+1
            start_j = 0
            for _, lj in self.col:
                n_j = 2*lj.l+1
                block[:, start_i:start_i+n_i, start_j:start_j+n_j] = lagrange[:,idx].reshape(-1, 1, 1).expand(lagrange.shape[0], n_i, n_j)
                idx += 1
                start_j += n_j
            start_i += n_i
            
        return block

    def change_index(self, hamiltonian):
        """
        根据 DFT 软件（如 openmx, siesta）的约定，调整输出矩阵元素的顺序。
        这确保了生成的哈密顿量与目标软件的原子轨道顺序相匹配。

        Args:
            hamiltonian (torch.Tensor): 原始哈密顿量矩阵。

        Returns:
            torch.Tensor: 索引调整后的哈密顿量。
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
                # 对某些 m!=0 的轨道乘以 -1，以匹配 siesta 等软件的 Condon-Shortley 相位约定
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]
            hamiltonian = hamiltonian.reshape(-1, self.nao_max**2)                
        return hamiltonian
    
    def convert_to_mole_Ham(self, data, Hon, Hoff):
        """
        将 onsite 和 offsite 矩阵块组装成完整的分子哈密顿量。
        这个函数目前似乎没有被使用。

        Args:
            data (Data): 输入的图数据。
            Hon (torch.Tensor): Onsite 哈密顿量块。
            Hoff (torch.Tensor): Offsite 哈密顿量块。

        Returns:
            torch.Tensor: 组装后的完整哈密顿量。
        """
        # 获取每个晶胞中的原子数
        max_atoms = torch.max(data['node_counts']).item()
                
        # 解析原子轨道基组定义
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        basis_def_temp = copy.deepcopy(self.basis_def)
        # key 是原子序数, value 是占据的轨道
        for k in self.basis_def.keys():
            basis_def_temp[k] = [num-1 for num in self.basis_def[k]]
            basis_definition[k][basis_def_temp[k]] = 1
            
        orb_mask = basis_definition[data['z']].view(-1, max_atoms*self.nao_max) # 形状: [N_batch, max_atoms*nao_max]  
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # 形状: [N_batch, max_atoms*nao_max, max_atoms*nao_max]
        orb_mask = orb_mask.view(-1, max_atoms*self.nao_max) # 形状: [N_atoms*nao_max, max_atoms*nao_max]
        
        atom_idx = torch.arange(data['z'].shape[0]).type_as(data['z'])
        H = torch.zeros([data['z'].shape[0], max_atoms, self.nao_max**2]).type_as(Hon) # 形状: [N_atoms, max_atoms, nao_max**2]
        H[atom_idx, atom_idx%max_atoms] = Hon
        H[data['edge_index'][0], data['edge_index'][1]%max_atoms] = Hoff
        H = H.reshape(
            data['z'].shape[0], max_atoms, self.nao_max, self.nao_max) # 形状: [N_atoms, max_atoms, nao_max, nao_max]

        # 重塑哈密顿量的维度
        H = H.permute((0, 2, 1, 3))
        H = H.reshape(data['z'].shape[0] * self.nao_max, max_atoms * self.nao_max)

        # 掩码填充的轨道
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / (data['z'].shape[0]/max_atoms)))
        H = H.reshape(-1, orbs)              
        return H
    
    def cat_onsite_and_offsite(self, data, Hon, Hoff):
        """
        将批处理中的 onsite 和 offsite 矩阵块按顺序拼接起来。

        Args:
            data (Data): 输入的图数据。
            Hon (torch.Tensor): Onsite 矩阵块。
            Hoff (torch.Tensor): Offsite 矩阵块。

        Returns:
            torch.Tensor: 拼接后的矩阵块。
        """
        # 获取每个晶胞中的节点数
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
        """
        对称化 Onsite 矩阵块。

        Args:
            Hon (torch.Tensor): Onsite 矩阵块。
            sign (str): 对称化方式，'+' 表示 (A + A.T)/2 (厄米)，'-' 表示 (A - A.T)/2 (反厄米)。

        Returns:
            torch.Tensor: 对称化后的矩阵。
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
        """
        对称化 Offsite 矩阵块。
        利用 `inv_edge_idx` (逆向边的索引) 来确保 H_ij = H_ji.T。

        Args:
            Hoff (torch.Tensor): Offsite 矩阵块。
            inv_edge_idx (torch.Tensor): 逆向边的索引。
            sign (str): 对称化方式，'+' 或 '-'。

        Returns:
            torch.Tensor: 对称化后的矩阵。
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

    def symmetrize_Hon_soc(self, Hon, sign:str='+'):
        """对称化包含 SOC 的 Onsite 矩阵块 (尺寸为 2*nao_max)。"""
        if self.symmetrize:
            Hon = Hon.reshape(-1, 2*self.nao_max, 2*self.nao_max)
            if sign == '+':
                Hon = 0.5*(Hon + Hon.permute((0,2,1)))
            else:
                Hon = 0.5*(Hon - Hon.permute((0,2,1)))
            Hon = Hon.reshape(-1, (2*self.nao_max)**2)
            return Hon
        else:
            Hon = Hon.reshape(-1, (2*self.nao_max)**2)
            return Hon    

    def symmetrize_Hoff_soc(self, Hoff, inv_edge_idx, sign:str='+'):
        """对称化包含 SOC 的 Offsite 矩阵块 (尺寸为 2*nao_max)。"""
        if self.symmetrize:
            Hoff = Hoff.reshape(-1, 2*self.nao_max, 2*self.nao_max)
            if sign == '+':
                Hoff = 0.5*(Hoff + Hoff[inv_edge_idx].permute((0,2,1)))
            else:
                Hoff = 0.5*(Hoff - Hoff[inv_edge_idx].permute((0,2,1)))
            Hoff = Hoff.reshape(-1, (2*self.nao_max)**2)
            return Hoff
        else:
            Hoff = Hoff.reshape(-1, (2*self.nao_max)**2)
            return Hoff

    def cal_band_energy_debug(self, Hon, Hoff, Son, Soff, data, export_reciprocal_values:bool=False):
        """
        计算能带结构（调试版本）。
        
        通过傅里叶变换将实空间的哈密顿量 H(R) 和重叠矩阵 S(R) 转换为
        倒易空间的 H(k) 和 S(k)，然后求解广义本征值问题 `H(k)c = ES(k)c` 来得到能带。
        
        Args:
            Hon (torch.Tensor): 预测的 onsite 哈密顿量。
            Hoff (torch.Tensor): 预测的 offsite 哈密顿量。
            Son (torch.Tensor): 预测的 onsite 重叠矩阵。
            Soff (torch.Tensor): 预测的 offsite 重叠矩阵。
            data (Data): 输入数据。
            export_reciprocal_values (bool): 是否返回倒易空间矩阵。

        Returns:
            tuple: 包含能带能量、波函数、带隙等信息的元组。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # 形状:(N_batch, 3, 3)
        Nbatch = cell.shape[0]
        
        # 解析原子轨道基组定义
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引。
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data['z']] # 形状: [N_atoms, nao_max] 
        orb_mask = torch.split(orb_mask, data['node_counts'].tolist(), dim=0) # 形状: [n_atoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # 形状: [n_atoms*nao_max, n_atoms*nao_max]
        
        # 设置价电子数
        num_val = torch.zeros((99,)).type_as(data['z'])
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data['z']] # 形状: [N_atoms]
        num_val = scatter(num_val, data['batch'], dim=0) # 形状: [N_batch]
                
        # 初始化 band_num_win
        if self.band_num_control is not None:
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # 形状: [N_atoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # 形状: (N_batch,)
             
        # 按 batch 分离 Hon 和 Hoff
        node_counts = data['node_counts']
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'], node_counts.tolist(), dim=0)
        Son_pred_split = torch.split(Son, node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0) # 形状: (N_batch,)
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
            
            # 初始化 H(k) 和 S(k)       
            # 傅里叶变换的相位因子
            coe = torch.exp(2j*torch.pi*torch.sum(data['nbr_shift'][edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (n_edges, 1, 3)*(1, num_k, 3) -> (n_edges, num_k)     
            
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))  
            SK_pred = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))           
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            na = torch.arange(natoms).type_as(j)
            # 添加 onsite 部分 (R=0)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # 形状 (num_k, n_atoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            SK_pred[:,na,na,:,:] +=  Son_pred_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK_pred)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            # 添加 offsite 部分 (R!=0)
            for iedge in range(edge_num[idx]):
                # 形状 (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
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

            HK = torch.swapaxes(HK,-2,-3) #(nk, n_atoms, nao_max, n_atoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, n_atoms, nao_max, n_atoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK_pred = torch.swapaxes(SK_pred,-2,-3) #(nk, n_atoms, nao_max, n_atoms, nao_max)
            SK_pred = SK_pred.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, n_atoms, nao_max, n_atoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # 掩码 H(k) 和 S(k)
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
            # 变换到正交基 H' = L^-1 * H * (L^T)^-1
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # 将波函数系数转换回原始基
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy 实现（用于验证）
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
                lamda = 1/torch.sqrt(lamda) # 形状: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK_pred)
                dS_reciprocal.append(dSK)
            
            if self.band_num_control is not None:
                orbital_energies = orbital_energies[:,:band_num_win[idx]]
                orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]                
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [形状:(N_bands, num_k)]
            wavefunction.append(orbital_coefficients)  
            H_sym.append(Hs.view(-1)) 
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))    
            
        band_energy = torch.cat(band_energy, dim=0) # [形状:(N_bands, num_k)]
        
        gap = torch.cat(gap, dim=0)
        
        if export_reciprocal_values:
            wavefunction = torch.stack(wavefunction, dim=0) # 形状:[N_batch, num_k, n_orbs, n_orbs]
            HK = torch.stack(H_reciprocal, dim=0) # 形状:[N_batch, num_k, n_orbs, n_orbs]
            SK = torch.stack(S_reciprocal, dim=0) # 形状:[N_batch, num_k, n_orbs, n_orbs]  
            dSK = torch.stack(dS_reciprocal, dim=0) # 形状:[N_batch, num_k, n_orbs, n_orbs, 3]   
            return band_energy, wavefunction, HK, SK, dSK, gap
        else:
            wavefunction = [wavefunction[idx].reshape(-1) for idx in range(Nbatch)]
            wavefunction = torch.cat(wavefunction, dim=0) # 形状:[N_batch*num_k*n_orbs*n_orbs]
            H_sym = torch.cat(H_sym, dim=0) # 形状:(N_batch*num_k*n_orbs*n_orbs)
            return band_energy, wavefunction, gap, H_sym   
    
    def cal_band_energy(self, Hon: torch.Tensor, Hoff: torch.Tensor, data, export_reciprocal_values: bool = False):
        """计算能带结构（非调试的正式版本）。

        此函数通过给定的 onsite 和 offsite 矩阵块，为一批晶体构建 k 点依赖的哈密顿量
        和重叠矩阵，然后通过求解广义本征值问题来计算能带结构。
        该过程与 `cal_band_energy_debug` 类似，但流程更精简，不涉及对预计算 `Son`, `Soff` 的依赖。
        
        .. warning::
           当前实现主要针对 `openmx` 类型的哈密顿量。

        Args:
            Hon (torch.Tensor): 批处理中所有原子/轨道的 Onsite 矩阵块，形状通常为 `(N_nodes, nao_max*nao_max)`。
            Hoff (torch.Tensor): 批处理中所有边的 Offsite 矩阵块，形状通常为 `(N_edges, nao_max*nao_max)`。
            data (Data): PyG 图数据对象，包含原子结构、k点等信息。
            export_reciprocal_values (bool, optional): 如果为 True，则额外返回倒易空间中的
                哈密顿量 `HK`、重叠矩阵 `SK`、`SK`的导数`dSK` 和波函数。默认为 `False`。

        Returns:
            tuple: 
                如果 `export_reciprocal_values` 为 `False`，返回 `(band_energy, wavefunction, gap, H_sym)`。
                如果 `export_reciprocal_values` 为 `True`，返回 `(band_energy, wavefunction, HK, SK, dSK, gap)`。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # 形状:(N_batch, 3, 3)
        Nbatch = cell.shape[0]
        
        # --- 步骤 1: 解析和准备掩码及参数 ---
        # 解析原子轨道基组定义，用于后续的掩码操作
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引。
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data['z']] # 形状: [N_atoms, nao_max] 
        orb_mask = torch.split(orb_mask, data['node_counts'].tolist(), dim=0) # 形状: [n_atoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            # 为每个晶体创建轨道掩码，用于从 full-nao 矩阵中提取有效轨道
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # 形状: [n_atoms*nao_max, n_atoms*nao_max]
        
        # 设置每个晶体的价电子数
        num_val = torch.zeros((99,)).type_as(data['z'])
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data['z']] # 形状: [N_atoms]
        num_val = scatter(num_val, data['batch'], dim=0) # 形状: [N_batch]
                
        # 初始化能带窗口控制参数
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # 形状: [N_atoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # 形状: (N_batch,)   
             
        # --- 步骤 2: 按批次分离输入张量 ---
        node_counts = data['node_counts']
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'], node_counts.tolist(), dim=0)
        
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0) # 形状: (N_batch,)
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
        
        # --- 步骤 3: 遍历批处理中的每个晶体 ---
        for idx in range(Nbatch):
            k_vec = data['k_vecs'][idx]   
            natoms = data['node_counts'][idx]
            
            # --- 3a: 构建 k 点依赖的矩阵 HK 和 SK ---
            # 计算傅里叶变换的相位因子
            coe = torch.exp(2j*torch.pi*torch.sum(data['nbr_shift'][edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (nedges, 1, 3)*(1, num_k, 3) -> (nedges, num_k)     
            
            # 初始化 k 点矩阵
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))            
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            # 添加 On-site 部分 (k 点无关)
            na = torch.arange(natoms).type_as(j)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # shape (num_k, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            # 添加 Off-site 部分 (傅里叶变换)
            for iedge in range(edge_num[idx]):
                # shape (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
                j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                HK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Hoff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
            
            if export_reciprocal_values:
                for iedge in range(edge_num[idx]):
                    j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    dSK[:,j_idx,i_idx,:,:,:] += coe[iedge,:,None,None,None] * dSoff_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,iedge,:,:,:]

            # --- 3b: 重塑并掩码矩阵 ---
            HK = torch.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, natoms, nao_max, natoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # 使用预先计算的轨道掩码，去除无效的轨道
            HK = torch.masked_select(HK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(HK.numel()/self.num_k))
            HK = HK.reshape(self.num_k, norbs, norbs)
            
            SK = torch.masked_select(SK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            if export_reciprocal_values:
                dSK = torch.masked_select(dSK, orb_mask_batch[idx].unsqueeze(-1).repeat(self.num_k,1,1,3) > 0)
                dSK = dSK.reshape(self.num_k, norbs, norbs, 3)            
            
            # --- 3c: 求解广义本征值问题 ---
            # Calculate band energies
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # Convert the wavefunction coefficients back to the original basis
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy implementation (for verification)
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
            
            # --- 3d: 后处理和存储结果 ---
            if export_reciprocal_values:
                # Normalize wave function
                lamda = torch.einsum('nai, nij, naj -> na', torch.conj(orbital_coefficients), SK, orbital_coefficients).real
                lamda = 1/torch.sqrt(lamda) # shape: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK)
                dS_reciprocal.append(dSK)
            
            # 计算带隙
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))
            
            # 根据配置控制输出的能带数量
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
            
        # --- 步骤 4: 整合批处理结果 ---
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

    def cal_band_energy_soc(self, Hsoc_on_real: torch.Tensor, Hsoc_on_imag: torch.Tensor, 
                              Hsoc_off_real: torch.Tensor, Hsoc_off_imag: torch.Tensor, data) -> tuple:
        """计算包含自旋轨道耦合（SOC）的能带结构。

        此方法通过给定的 onsite 和 offsite SOC 哈密顿量的实部和虚部，构建一个 2x2 
        的块矩阵来表示完整的 SOC 哈密顿量。其中，对角线块是自旋守恒的部分，
        非对角线块是自旋翻转的部分。然后通过求解该复厄米矩阵的本征值问题得到能带。
        
        .. warning::
           当前实现主要针对 `openmx` 类型的哈密顿量。

        Args:
            Hsoc_on_real (torch.Tensor): Onsite SOC 哈密顿量的实部。
            Hsoc_on_imag (torch.Tensor): Onsite SOC 哈密顿量的虚部。
            Hsoc_off_real (torch.Tensor): Offsite SOC 哈密顿量的实部。
            Hsoc_off_imag (torch.Tensor): Offsite SOC 哈密顿量的虚部。
            data (Data): 输入数据，包含原子结构、k点、重叠矩阵等信息。

        Returns:
            tuple: 返回一个包含能带能量 `band_energy` 和扁平化的波函数 `wavefunction` 的元组。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        cell = data['cell'] # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        # --- 步骤 1: 重塑输入并准备掩码 ---
        Hsoc_on_real = Hsoc_on_real.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_on_imag = Hsoc_on_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_off_real = Hsoc_off_real.reshape(-1, 2*self.nao_max, 2*self.nao_max) 
        Hsoc_off_imag = Hsoc_off_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        
        # 解析原子轨道基组
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key 是原子序数, value 是占据轨道的索引。
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
                
        # 初始化能带窗口
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data['z'])
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data['z']] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data['batch'], dim=0) # shape: (Nbatch,)       
            
        # --- 步骤 2: 按批次分离输入张量 ---
        node_counts = data['node_counts']
        Hon_split = torch.split(Hsoc_on_real, node_counts.tolist(), dim=0)
        iHon_split = torch.split(Hsoc_on_imag, node_counts.tolist(), dim=0)
        Son_split = torch.split(data['Son'].reshape(-1, self.nao_max, self.nao_max), node_counts.tolist(), dim=0)
        
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
        # --- 步骤 3: 遍历批处理中的每个晶体 ---
        for idx in range(Nbatch):
            k_vec = data['k_vecs'][idx]   
            natoms = data['node_counts'][idx].item() 
            
            # --- 3a: 构建 k 点依赖的 SOC 重叠矩阵 SK ---
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
            # 掩码并重塑
            SK = SK[:,orb_mask_batch[idx] > 0]
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            # 构建 SOC 基下的重叠矩阵 (对角块)
            I = torch.eye(2).type_as(data['Hon'])
            SK = torch.kron(I, SK)
            
            # --- 3b: 构建 k 点依赖的 SOC 哈密顿量 HK ---
            # on-site term
            H11 = Hon_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHon_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hon_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHon_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hon_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHon_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hon_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHon_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hon_soc = [H11, H12, H21, H22]
            # off-site term
            H11 = Hoff_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHoff_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hoff_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHoff_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hoff_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHoff_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hoff_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHoff_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hoff_soc = [H11, H12, H21, H22]
            
            # 初始化 HK
            HK_list = []
            # 分别为 H11, H12, H21, H22 构建 k 点哈密顿量
            for Hon, Hoff in zip(Hon_soc, Hoff_soc):
                H_cell = torch.view_as_complex(torch.zeros((ncells, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(data['Son']))
                H_cell[cell_index, edge_index_split[idx][0], edge_index_split[idx][1], :, :] += Hoff    

                HK = torch.einsum('ijklm, ni->njklm', H_cell, phase) # (nk, natoms, natoms, nao_max, nao_max)
                HK[:,na,na,:,:] +=  Hon[None,na,:,:] # shape (nk, natoms, nao_max, nao_max)

                HK = torch.swapaxes(HK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
                HK = HK.reshape(self.num_k, natoms*self.nao_max, natoms*self.nao_max)

                # mask HK
                HK = HK[:, orb_mask_batch[idx] > 0]
                norbs = int(math.sqrt(HK.numel()/self.num_k))
                HK = HK.reshape(self.num_k, norbs, norbs)
        
                HK_list.append(HK)
            
            # 将四个块拼接成完整的 SOC 哈密顿量
            HK = torch.cat([torch.cat([HK_list[0],HK_list[1]], dim=-1), torch.cat([HK_list[2],HK_list[3]], dim=-1)],dim=-2)
        
            # --- 3c: 求解广义本征值问题 ---
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)   
            # Convert the wavefunction coefficients back to the original basis
            orbital_coefficients = torch.bmm(L_t_inv, orbital_coefficients) # shape:(num_k, Nbands, Nbands)
            
            # --- 3d: 后处理和存储结果 ---
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

    def mask_Ham(self, Hon: torch.Tensor, Hoff: torch.Tensor, data) -> tuple:
        """根据每种原子的实际轨道数，对哈密顿量和重叠矩阵进行掩码操作。

        此函数确保只有在物理上存在的原子轨道之间的矩阵元才具有非零值，
        而那些由于使用固定大小 `nao_max` 导致的填充项则被置零。

        Args:
            Hon (torch.Tensor): Onsite 矩阵块 (哈密顿量或重叠矩阵)。
            Hoff (torch.Tensor): Offsite 矩阵块 (哈密顿量或重叠矩阵)。
            data (Data): 输入数据，需要 `z` 和 `edge_index`。

        Returns:
            tuple: 返回一个元组，包含掩码后的 `(Hon, Hoff)`。
        """
        # 解析原子轨道基组定义
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data['z'])
        # key is the atomic number, value is the index of the occupied orbits.
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
        
        # 保存原始形状以便最后恢复
        original_shape_on = Hon.shape
        original_shape_off = Hoff.shape
        
        if len(original_shape_on) > 2:
            Hon = Hon.reshape(original_shape_on[0], -1)
        if len(original_shape_off) > 2:
            Hoff = Hoff.reshape(original_shape_off[0], -1)
        
        # 首先掩码 on-site 矩阵
        orb_mask = basis_definition[data['z']].view(-1, self.nao_max) # shape: [Natoms, nao_max] 
        # 通过外积创建 (i, j) 轨道对的掩码
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # shape: [Natoms, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Natoms, nao_max*nao_max]
        
        Hon_mask = torch.zeros_like(Hon)
        Hon_mask[orb_mask>0] = Hon[orb_mask>0]
        
        # 接着掩码 off-site 矩阵
        j = data['edge_index'][0]
        i = data['edge_index'][1]        
        orb_mask_j = basis_definition[data['z'][j]].view(-1, self.nao_max) # shape: [Nedges, nao_max]
        orb_mask_i = basis_definition[data['z'][i]].view(-1, self.nao_max) # shape: [Nedges, nao_max] 
        # 通过外积创建 (i, j) 轨道对的掩码
        orb_mask = orb_mask_j[:,:,None] * orb_mask_i[:,None,:]       # shape: [Nedges, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Nedges, nao_max*nao_max]
        
        Hoff_mask = torch.zeros_like(Hoff)
        Hoff_mask[orb_mask>0] = Hoff[orb_mask>0]

        # 恢复原始形状并返回
        Hon_mask = Hon_mask.reshape(original_shape_on)
        Hoff_mask = Hoff_mask.reshape(original_shape_off)
        
        return Hon_mask, Hoff_mask

    def construct_Hsoc(self, H: torch.Tensor, iH: torch.Tensor) -> torch.Tensor:
        """从自旋对角（实部）和自旋非对角（虚部）块构建复数 SOC 哈密顿量。
        
        Args:
            H (torch.Tensor): SOC 哈密顿量的实部块。
            iH (torch.Tensor): SOC 哈密顿量的虚部块。

        Returns:
            torch.Tensor: 完整的复数 SOC 哈密顿量张量。
        """
        Hsoc = torch.view_as_complex(torch.zeros((H.shape[0], (2*self.nao_max)**2, 2)).type_as(H))
        Hsoc = H + 1.0j*iH
        return Hsoc

    def reduce(self, coefficient: torch.Tensor) -> torch.Tensor:
        """通过 Clebsch-Gordan 分解，将矩阵块分解为球谐系数（等变特征）。

        这是 `matrix_merge` 的逆过程。它通过在轨道组内取平均值来近似地提取
        具有特定角动量 `l` 的特征分量。这主要用于调试或分析目的，以检查
        网络是否学习到了符合物理对称性的特征。

        Args:
            coefficient (torch.Tensor): 形状为 `(N, nao_max, nao_max)` 的矩阵块。

        Returns:
            torch.Tensor: 分解得到的球谐系数，形状被重新扁平化为 `(N, nao_max*nao_max)`。
        """
        # 注意：这里的实现是一种近似，它假设同一 l 壳层内的所有 m 分量贡献相同
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

    def index_cells(self, unique_cell_shift_list: List[Tuple[int, ...]]) -> dict:
        """将每个唯一的晶胞位移向量映射到其在唯一列表中的索引。

        Args:
            unique_cell_shift_list (List[Tuple[int, ...]]): 唯一晶胞位移向量的列表。

        Returns:
            dict: 一个字典，键是晶胞位移元组，值是其索引。
        """
        cell_index_map = {}
        for index, cell_tuple in enumerate(unique_cell_shift_list):
            cell_index_map[tuple(cell_tuple)] = index
        return cell_index_map

    def get_unique_cell_shift_and_cell_shift_indices(self, data):
        """从边的晶胞位移中找到唯一的位移向量，并为每条边提供其对应的唯一位移索引。
        
        Args:
            data (Data): PyG 图数据对象。

        Returns:
            tuple: 返回 `(unique_cell_shift, cell_shift_indices, cell_index_map)`。
        """
        # 匹配 cell_shift 在 unique_cell_shift 中的行索引
        cell_shift = data['cell_shift']
        unique_cell_shift = torch.unique(cell_shift, dim=0)
        
        zero_vector = torch.tensor([[0, 0, 0]]).type_as(unique_cell_shift)
        is_zero_vector_present = (unique_cell_shift == zero_vector).all(dim=1).any()
        
        # 如果 (0, 0, 0) 不存在，则将其添加到 unique_cell_shift 中
        if not is_zero_vector_present:
            unique_cell_shift = torch.cat((zero_vector, unique_cell_shift), dim=0)
        
        # 扩展张量以便进行广播比较
        expanded_cell_shift = cell_shift.unsqueeze(1).expand(-1, unique_cell_shift.size(0), -1)
        expanded_unique_cell_shift = unique_cell_shift.unsqueeze(0).expand(cell_shift.size(0), -1, -1)
        
        # 比较并找到匹配的行
        matches = (expanded_cell_shift == expanded_unique_cell_shift).all(dim=2)
        
        # 获取匹配行的索引
        cell_shift_indices = matches.nonzero(as_tuple=True)[1] # (Nedges,)
        
        # 获取晶胞索引映射字典
        cell_index_map = self.index_cells(unique_cell_shift.tolist())
        
        return unique_cell_shift, cell_shift_indices, cell_index_map

    def edge_hunter(self, data, inv_edge_idx=None):
        # ... (此函数逻辑复杂且高度特化，保持原有注释风格)
        src = data['edge_index'][0]
        tar = data['edge_index'][1]
        unique_cell_shift = data['unique_cell_shift']
        cell_shift_indices = data['cell_shift_indices']
        cell_index_map = data['cell_index_map']

        num_nodes = len(data['z'])
        num_shifts = len(unique_cell_shift)

        edge_matcher_src = [torch.where(src == ia)[0] for ia in range(num_nodes)]
        edge_matcher_tar = [[[] for _ in range(num_shifts)] for _ in range(num_nodes)]

        for ia in range(num_nodes):
            inv_src = inv_edge_idx[edge_matcher_src[ia]]
            cell_shift_inv_src = cell_shift_indices[inv_src]

            for idx_edge, idx_cell in zip(inv_src, cell_shift_inv_src):
                edge_matcher_tar[ia][idx_cell.item()].append(idx_edge)

            for idx_cell in range(num_shifts):
                if edge_matcher_tar[ia][idx_cell]:
                    edge_matcher_tar[ia][idx_cell] = torch.stack(edge_matcher_tar[ia][idx_cell]).type_as(src)
                else:
                    edge_matcher_tar[ia][idx_cell] = torch.tensor([], dtype=torch.long).type_as(src)

        return edge_matcher_src, edge_matcher_tar

    def get_basis_definition(self, z: torch.Tensor) -> torch.Tensor:
        """为掩码计算创建并返回基组定义张量。
        
        Args:
            z (torch.Tensor): 原子序数张量。

        Returns:
            torch.Tensor: 一个 `(99, nao_max)` 的张量，其中 `basis_definition[z, i] = 1`
                          表示原子 `z` 包含轨道 `i`。
        """
        basis_definition = torch.zeros((99, self.nao_max)).type_as(z)
        for k in self.basis_def:
            basis_definition[k][self.basis_def[k]] = 1
        return basis_definition

    def mask_tensor_builder(self, data) -> torch.Tensor:
        """构建掩码张量并返回拼接后的掩码张量。

        Args:
            data (Data): PyG 图数据对象。

        Returns:
            torch.Tensor: 拼接后的 on-site 和 off-site 掩码。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        z = data['z']
        basis_definition = self.get_basis_definition(z)
        # 使用 einsum 计算 on-site 和 off-site 掩码
        mask_on = torch.einsum('ni, nj -> nij', basis_definition[z], basis_definition[z]).bool()
        mask_off = torch.einsum('ni, nj -> nij', basis_definition[z[j]], basis_definition[z[i]]).bool()
        # 拼接并重塑掩码
        mask_all = torch.cat(
            (mask_on.reshape(-1, self.nao_max**2), 
             mask_off.reshape(-1, self.nao_max**2)), 
            dim=0
        )
        return mask_all

    def mask_tensor_builder_col(self, data) -> torch.Tensor:
        """为共线自旋情况构建掩码张量。

        Args:
            data (Data): PyG 图数据对象。

        Returns:
            torch.Tensor: 适用于共线自旋计算的拼接掩码。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        z = data['z']
        basis_definition = self.get_basis_definition(z)
        # 使用 einsum 计算 on-site 和 off-site 掩码
        mask_on = torch.einsum('ni, nj -> nij', basis_definition[z], basis_definition[z]).bool()
        mask_on = torch.stack([mask_on, mask_on], dim=1) # (Nbatchs, 2, nao_max, nao_max)
        mask_off = torch.einsum('ni, nj -> nij', basis_definition[z[j]], basis_definition[z[i]]).bool()
        mask_off = torch.stack([mask_off, mask_off], dim=1) # (Nbatchs, 2, nao_max, nao_max)
        # 拼接并重塑掩码
        mask_all = torch.cat(
            (mask_on.reshape(-1, 2, self.nao_max**2), 
             mask_off.reshape(-1, 2, self.nao_max**2)), 
            dim=0
        )
        return mask_all

    def mask_tensor_builder_soc(self, data) -> tuple:
        """为包含自旋轨道耦合的情况构建掩码张量。

        Args:
            data (Data): PyG 图数据对象。

        Returns:
            tuple: 返回一个元组 `(mask_real_imag, mask_all)`，
                   分别对应 SOC 哈密顿量的实部/虚部掩码和总掩码。
        """
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        z = data['z']
        basis_definition = self.get_basis_definition(z)

        # 计算基础掩码
        mask_on = torch.einsum('ni, nj -> nij', basis_definition[z], basis_definition[z])
        mask_off = torch.einsum('ni, nj -> nij', basis_definition[z[j]], basis_definition[z[i]])

        # 扩展张量以包含自旋分量
        mask_on_expanded = blockwise_2x2_concat(mask_on, mask_on, mask_on, mask_on).reshape(-1, (2*self.nao_max)**2).bool()
        mask_off_expanded = blockwise_2x2_concat(mask_off, mask_off, mask_off, mask_off).reshape(-1, (2*self.nao_max)**2).bool()

        # 处理实部和虚部掩码
        mask_real_imag = self.cat_onsite_and_offsite(data, mask_on_expanded, mask_off_expanded)

        # 拼接所有掩码
        mask_all = torch.cat((mask_real_imag, mask_real_imag), dim=0)

        return mask_real_imag, mask_all

    def forward(self, data, graph_representation: dict = None):
        """模型的核心前向传播逻辑。

        该方法根据 `__init__` 中设置的各种开关（如 `soc_switch`, `spin_constrained` 等），
        执行不同的计算路径来构建哈密顿量和重叠矩阵，并最终计算能带结构等物理属性。

        Args:
            data (Data): PyG 图数据对象，可能包含预计算的矩阵。
            graph_representation (dict, optional): 来自 GNN 表示层的节点和边等变特征字典。

        Returns:
            dict: 一个包含计算结果的字典，可能包括 'hamiltonian', 'overlap', 
                  'band_energy', 'wavefunction' 等键。
        """
        # ... (由于此函数逻辑极为复杂且分支众多，保持原有注释，仅翻译少量关键部分)

        # To be compatible with the format of Hongyu yu
        if 'H0_u' in data:
            Hon_u0 = data['H0_u'][:len(data['z'])].flatten(1)
            Hon_d0 = data['H0_d'][:len(data['z'])].flatten(1)
            Hoff_u0 = data['H0_u'][len(data['z']):].flatten(1)
            Hoff_d0 = data['H0_d'][len(data['z']):].flatten(1)
            data['Hon0'] = torch.stack([Hon_u0, Hon_d0], dim=1)
            data['Hoff0'] = torch.stack([Hoff_u0, Hoff_d0], dim=1)
            data['Hon'] = torch.stack([data['H_u'][:len(data['z'])], data['H_d'][:len(data['z'])]], dim=1).flatten(2)
            data['Hoff'] = torch.stack([data['H_u'][len(data['z']):], data['H_d'][len(data['z']):]], dim=1).flatten(2)
    
        # prepare data['hamiltonian'] & data['overlap']
        if 'hamiltonian' not in data:
            data['hamiltonian'] = self.cat_onsite_and_offsite(data, data['Hon'], data['Hoff'])
        if 'overlap' not in data:
            data['overlap'] = self.cat_onsite_and_offsite(data, data['Son'], data['Soff'])
        
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        j = data['edge_index'][0]
        i = data['edge_index'][1]
        
        # Calculate inv_edge_index in batch
        inv_edge_idx = data['inv_edge_idx']
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data['batch'][j], dim=0)
        edge_num = torch.cumsum(edge_num, dim=0) - edge_num
        inv_edge_idx = inv_edge_idx + edge_num[data['batch'][j]]
        
        # Calculate the on-site Hamiltonian 
        self.ham_irreps_dim = self.ham_irreps_dim.type_as(j)  
        
        if not self.ham_only:
            node_sph = self.onsitenet_s(node_attr)
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Son = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Son = self.change_index(Son)
        
            # Impose Hermitian symmetry for Son
            Son = self.symmetrize_Hon(Son)

            # Calculate the off-site overlap
            # Calculate the contribution of the edges       
            edge_sph = self.offsitenet_s(edge_attr)
            edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
            Soff = self.matrix_merge(edge_sph)
        
            Soff = self.change_index(Soff)        
            # Impose Hermitian symmetry for Soff
            Soff = self.symmetrize_Hoff(Soff, inv_edge_idx)
        
            if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
                Son, Soff = self.mask_Ham(Son, Soff, data)
        
        if self.soc_switch or self.spin_constrained:            
            if self.soc_switch:
                # build Hsoc
                if self.soc_basis == 'so3':
                    if self.add_H_nonsoc:
                        Hon, Hoff = data['Hon_nonsoc'], data['Hoff_nonsoc']
                        
                        # Load the on-site and off-site Hamiltonian matrices
                        Hon0, Hoff0 = data['Hon0'], data['Hoff0']
                    
                        # Reshape `Hon0` and `Hoff0` into 3D tensors for block-wise manipulation
                        Hon0_resized = Hon0.reshape(-1, 2 * self.nao_max, 2 * self.nao_max)
                        Hoff0_resized = Hoff0.reshape(-1, 2 * self.nao_max, 2 * self.nao_max)
                    
                        # Create zero blocks for the submatrices
                        zero_on = torch.zeros_like(data['Son']).reshape(-1, self.nao_max, self.nao_max)
                        zero_off = torch.zeros_like(data['Soff']).reshape(-1, self.nao_max, self.nao_max)
                    
                        # Zero out the upper-left and bottom-right blocks of `Hon0`
                        Hon0_resized[:, :self.nao_max, :self.nao_max] = zero_on
                        Hon0_resized[:, self.nao_max:, self.nao_max:] = zero_on
                    
                        # Zero out the upper-left and bottom-right blocks of `Hoff0`
                        Hoff0_resized[:, :self.nao_max, :self.nao_max] = zero_off
                        Hoff0_resized[:, self.nao_max:, self.nao_max:] = zero_off
                    
                        # Flatten the processed matrices back to their original shape
                        data['Hon0'] = Hon0_resized.reshape(-1, (2 * self.nao_max) ** 2)
                        data['Hoff0'] = Hoff0_resized.reshape(-1, (2 * self.nao_max) ** 2)
                        
                    else:
                        node_sph = self.onsitenet_h(node_attr)     
                        node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
                        Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
    
                        Hon = self.change_index(Hon)
    
                        # Impose Hermitian symmetry for Hon
                        Hon = self.symmetrize_Hon(Hon)            
    
                        # Calculate the off-site Hamiltonian
                        # Calculate the contribution of the edges       
                        edge_sph = self.offsitenet_h(edge_attr)
                        edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
                        Hoff = self.matrix_merge(edge_sph)
    
                        Hoff = self.change_index(Hoff)        
                        # Impose Hermitian symmetry for Hoff
                        Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
    
                        Hon, Hoff = self.mask_Ham(Hon, Hoff, data)

                    # build Hsoc
                    ksi_on = self.onsitenet_ksi(node_attr)
                    ksi_on = self.reduce(ksi_on)

                    ksi_off = self.offsitenet_ksi(edge_attr)
                    ksi_off = self.reduce(ksi_off)  

                    Hsoc_on_real = torch.zeros((Hon.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hon)
                    Hsoc_on_real[:,:self.nao_max,:self.nao_max] = Hon.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real[:,:self.nao_max,self.nao_max:] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real[:,self.nao_max:,:self.nao_max] = self.symmetrize_Hon((ksi_on*data['Lon'][:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real[:,self.nao_max:,self.nao_max:] = Hon.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real = Hsoc_on_real.reshape(-1, (2*self.nao_max)**2)

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
                    node_sph = self.onsitenet_h(node_attr) 

                    Hon = self.hamDecomp.get_H(node_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                    Hon = self.change_index(Hon)
                    Hon = Hon.reshape(-1, 2, 2, self.nao_max, self.nao_max)                
                    Hon = torch.swapaxes(Hon, 2, 3) # shape (Nnodes, 2, nao_max, 2, nao_max)

                    # Calculate the off-site Hamiltonian
                    # Calculate the contribution of the edges       
                    edge_sph = self.offsitenet_h(edge_attr)

                    Hoff = self.hamDecomp.get_H(edge_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                    Hoff = self.change_index(Hoff)
                    Hoff = Hoff.reshape(-1, 2, 2, self.nao_max, self.nao_max)
                    Hoff = torch.swapaxes(Hoff, 2, 3) # shape (Nedges, 2, nao_max, 2, nao_max)    

                    # mask zeros         
                    for i in range(2):
                        for j in range(2):
                            Hon[:,i,:,j,:], Hoff[:,i,:,j,:] = self.mask_Ham(Hon[:,i,:,j,:], Hoff[:,i,:,j,:], data)
                    Hon = Hon.reshape(-1, (2*self.nao_max)**2)
                    Hoff = Hoff.reshape(-1, (2*self.nao_max)**2)
                    # build four parts
                    Hsoc_on_real =  Hon.real
                    Hsoc_off_real = Hoff.real
                    Hsoc_on_imag = Hon.imag
                    Hsoc_off_imag = Hoff.imag

                else:
                    raise NotImplementedError
            else:
                node_sph = self.onsitenet_h(node_attr)     
                node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
                Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
                Hon = self.change_index(Hon)
                # Impose Hermitian symmetry for Hon
                Hon = self.symmetrize_Hon(Hon)            
                # Calculate the off-site Hamiltonian
                # Calculate the contribution of the edges       
                edge_sph = self.offsitenet_h(edge_attr)
                edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
                Hoff = self.matrix_merge(edge_sph)
                Hoff = self.change_index(Hoff)        
                # Impose Hermitian symmetry for Hoff
                Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
                Hon, Hoff = self.mask_Ham(Hon, Hoff, data)
                
                if not self.collinear_spin:
                    Hsoc_on_real = torch.zeros_like(data['Hon']).reshape(Hon.shape[0], 2*self.nao_max, 2*self.nao_max)
                    Hsoc_on_real[:,:self.nao_max,:self.nao_max] = Hon.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real[:,self.nao_max:,self.nao_max:] = Hon.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_on_real = Hsoc_on_real.reshape(Hon.shape[0], (2*self.nao_max)**2)
                    
                    Hsoc_off_real = torch.zeros_like(data['Hoff']).reshape(Hoff.shape[0], 2*self.nao_max, 2*self.nao_max)
                    Hsoc_off_real[:,:self.nao_max,:self.nao_max] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_off_real[:,self.nao_max:,self.nao_max:] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                    Hsoc_off_real = Hsoc_off_real.reshape(Hoff.shape[0], (2*self.nao_max)**2)
                    
                    Hsoc_on_imag = torch.zeros_like(data['iHon']) 
                    Hsoc_off_imag = torch.zeros_like(data['iHoff'])
            
            if self.spin_constrained:
                magnetic_atoms = (data['spin_length'] > self.minMagneticMoment)
                data['unique_cell_shift'], data['cell_shift_indices'], data['cell_index_map'] = self.get_unique_cell_shift_and_cell_shift_indices(data)
                cell_shift_indices = data['cell_shift_indices'].tolist()
                cell_index_map = data['cell_index_map']
                
                # learn a weight matrix
                if self.use_learned_weight:
                    node_sph = self.onsitenet_weight(node_attr)     
                    node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
                    weight_on = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
                    weight_on = self.change_index(weight_on)

                    # Impose Hermitian symmetry for Hon
                    weight_on = self.symmetrize_Hon(weight_on)           

                    # Calculate the off-site Hamiltonian
                    # Calculate the contribution of the edges       
                    edge_sph = self.offsitenet_weight(edge_attr)
                    edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
                    weight_off = self.matrix_merge(edge_sph)
        
                    weight_off = self.change_index(weight_off)        
                    # Impose Hermitian symmetry for Hoff
                    weight_off = self.symmetrize_Hoff(weight_off, inv_edge_idx)
                    
                    weight_on, weight_off = self.mask_Ham(weight_on, weight_off, data)
                    weight_on, weight_off = weight_on.reshape(-1, self.nao_max, self.nao_max), weight_off.reshape(-1, self.nao_max, self.nao_max)
                    data['weight_on'] = weight_on
                    data['weight_off'] = weight_off

                if self.soc_switch:
                    J_on = self.onsitenet_J(node_attr)     
                    J_on = self.J_merge(J_on) # shape: (Natoms, nao_max, nao_max, 3, 3)

                    J_off = self.offsitenet_J(edge_attr) # shape: (Nedges, Nblocks)    
                    J_off = self.J_merge(J_off) # shape: (Nedges, nao_max, nao_max, 3, 3)

                    if self.add_quartic:
                        K_on = self.onsitenet_K(node_attr) # shape: (Natoms, Nblocks)    
                        K_on = self.K_merge(K_on) # shape: (Natoms, nao_max, nao_max)

                        K_off = self.offsitenet_K(edge_attr)  # shape: (Nedges, Nblocks)    
                        K_off = self.K_merge(K_off) # shape: (Nedges, nao_max, nao_max)

                    sigma = torch.view_as_complex(torch.zeros((3,2,2,2)).type_as(J_on))
                    sigma[0] = torch.Tensor([[0.0, 1.0],[1.0, 0.0]]).type_as(sigma)
                    sigma[1] = torch.complex(real=torch.zeros((2,2)), imag=torch.Tensor([[0.0, -1.0],[1.0, 0.0]])).type_as(sigma) 
                    sigma[2] = torch.Tensor([[1.0, 0.0],[0.0, -1.0]]).type_as(sigma) 

                    spin_vec = data['spin_vec']

                    # brodcast shape: (Natoms/Nedges, Natoms/Nedges, 2, nao_max, 2, nao_max)
                    H_heisen_J_on = torch.zeros(len(J_on), 2, self.nao_max, 2, self.nao_max).type_as(sigma)
                    H_heisen_J_off = torch.zeros(len(j), 2, self.nao_max, 2, self.nao_max).type_as(sigma)

                    # Optimize Performance
                    edge_matcher_src, edge_matcher_tar = self.edge_hunter(data, inv_edge_idx)           

                    H_heisen_J_on[magnetic_atoms] += oe.contract('mijkl, mij, kop, ml -> moipj', J_on[magnetic_atoms].type_as(sigma), weight_on[magnetic_atoms].type_as(sigma), sigma, spin_vec[magnetic_atoms].type_as(sigma))    
                    H_heisen_J_on[magnetic_atoms] += oe.contract('mijkl, mij, lop, mk -> moipj', J_on[magnetic_atoms].type_as(sigma), weight_on[magnetic_atoms].type_as(sigma), sigma, spin_vec[magnetic_atoms].type_as(sigma))

                    zero_shift_idx = cell_index_map[(0, 0, 0)]
                    for ia in range(len(J_on)):
                        # src
                        if magnetic_atoms[ia]:
                            zero_shift_edges = edge_matcher_tar[ia][zero_shift_idx]
                            edge_matcher_src_ = torch.cat([edge_matcher_src[ia], zero_shift_edges])
                            Woff = weight_off[edge_matcher_src_]                    
                            H_heisen_J_off[edge_matcher_src_] += oe.contract('ijkl, mij, kop, l -> moipj', J_on[ia].type_as(sigma), Woff.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))
                            H_heisen_J_off[edge_matcher_src_] += oe.contract('ijkl, mij, lop, k -> moipj', J_on[ia].type_as(sigma), Woff.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))
                    
                    for i_edge in range(len(j)):
                        ia = j[i_edge].item()
                        ja = i[i_edge].item()

                        # i
                        if magnetic_atoms[ja]:
                            Won = weight_on[ia]
                            Woff_src = weight_off[edge_matcher_src[ia]]
                            H_heisen_J_on[ia] += oe.contract('ijkl, ij, kop, l -> oipj', J_off[i_edge].type_as(sigma), Won.type_as(sigma), sigma, spin_vec[ja].type_as(sigma))
                            H_heisen_J_off[edge_matcher_src[ia]] += oe.contract('ijkl, mij, kop, l -> moipj', J_off[i_edge].type_as(sigma), Woff_src.type_as(sigma), sigma, spin_vec[ja].type_as(sigma))

                        # j
                        if magnetic_atoms[ia]:
                            Woff_tar = weight_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]]
                            H_heisen_J_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]] += oe.contract('ijkl, mij, lop, k -> moipj', J_off[i_edge].type_as(sigma), Woff_tar.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))
                            if cell_shift_indices[i_edge] == data['cell_index_map'][(0,0,0)]:
                                Won = weight_on[ja]
                                H_heisen_J_on[ja] += oe.contract('ijkl, ij, lop, k -> oipj', J_off[i_edge].type_as(sigma), Won.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))   
                else:
                    J_on = self.onsitenet_J(node_attr) # shape: (Natoms, Nblocks)  
                    J_on = self.J_merge(J_on) # shape: (Natoms, nao_max, nao_max,)
    
                    J_off = self.offsitenet_J(edge_attr) # shape: (Nedges, Nblocks)
                    J_off = self.J_merge(J_off) # shape: (Nedges, nao_max, nao_max,)
    
                    if self.add_quartic:
                        K_on = self.onsitenet_K(node_attr) # shape: (Natoms, Nblocks)    
                        K_on = self.K_merge(K_on) # shape: (Natoms, nao_max, nao_max)
                                        
                        K_off = self.offsitenet_K(edge_attr) # shape: (Nedges, Nblocks)
                        K_off = self.K_merge(K_off) # shape: (Nedges, nao_max, nao_max)               
                    
                    if self.collinear_spin:
                        sigma_z = torch.Tensor([[1.0, 0.0],[0.0, -1.0]]).type_as(J_on) 

                        spin_vec = data['spin_vec']

                        # brodcast shape: (Natoms/Nedges, Natoms/Nedges, 2, nao_max, 2, nao_max)
                        H_heisen_J_on = torch.zeros(len(J_on), 2, self.nao_max, 2, self.nao_max).type_as(J_on) 
                        H_heisen_J_off = torch.zeros(len(j), 2, self.nao_max, 2, self.nao_max).type_as(J_off) 

                        # Optimize Performance
                        edge_matcher_src, edge_matcher_tar = self.edge_hunter(data, inv_edge_idx)           

                        H_heisen_J_on[magnetic_atoms] += oe.contract('mij, mij, op, m -> moipj', J_on[magnetic_atoms], weight_on[magnetic_atoms], sigma_z, spin_vec[magnetic_atoms,2])

                        zero_shift_idx = cell_index_map[(0, 0, 0)]
                        for ia in range(len(J_on)):
                            # src
                            if magnetic_atoms[ia]:
                                zero_shift_edges = edge_matcher_tar[ia][zero_shift_idx]
                                edge_matcher_src_ = torch.cat([edge_matcher_src[ia], zero_shift_edges])
                                Woff = weight_off[edge_matcher_src_]   
                                H_heisen_J_off[edge_matcher_src_] += oe.contract('ij, mij, op-> moipj', J_on[ia], Woff, sigma_z)*spin_vec[ia,2]

                        for i_edge in range(len(j)):
                            ia = j[i_edge].item()
                            ja = i[i_edge].item()

                            # i
                            if magnetic_atoms[ja]:
                                Won = weight_on[ia]
                                Woff_src = weight_off[edge_matcher_src[ia]]
                                H_heisen_J_on[ia] += oe.contract('ij, ij, op-> oipj', J_off[i_edge], Won, sigma_z)*spin_vec[ja,2]
                                H_heisen_J_off[edge_matcher_src[ia]] += oe.contract('ij, mij, op -> moipj', J_off[i_edge], Woff_src, sigma_z)*spin_vec[ja,2]

                            # j
                            if magnetic_atoms[ia]:
                                Woff_tar = weight_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]]
                                H_heisen_J_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]] += oe.contract('ij, mij, op -> moipj', J_off[i_edge], Woff_tar, sigma_z)*spin_vec[ia,2]
                                if cell_shift_indices[i_edge] == data['cell_index_map'][(0,0,0)]:
                                    Won = weight_on[ja]
                                    H_heisen_J_on[ja] += oe.contract('ij, ij, op-> oipj', J_off[i_edge], Won, sigma_z)*spin_vec[ia,2]

                    else:                 
                        sigma = torch.view_as_complex(torch.zeros((3,2,2,2)).type_as(J_on))
                        sigma[0] = torch.Tensor([[0.0, 1.0],[1.0, 0.0]]).type_as(sigma)
                        sigma[1] = torch.complex(real=torch.zeros((2,2)), imag=torch.Tensor([[0.0, -1.0],[1.0, 0.0]])).type_as(sigma) 
                        sigma[2] = torch.complex(real=torch.zeros((2,2)), imag=torch.Tensor([[1.0, 0.0],[0.0, -1.0]])).type_as(sigma) 

                        spin_vec = data['spin_vec']

                        # brodcast shape: (Natoms/Nedges, Natoms/Nedges, 2, nao_max, 2, nao_max)
                        H_heisen_J_on = torch.zeros(len(J_on), 2, self.nao_max, 2, self.nao_max).type_as(sigma)
                        H_heisen_J_off = torch.zeros(len(j), 2, self.nao_max, 2, self.nao_max).type_as(sigma)

                        # Optimize Performance
                        edge_matcher_src, edge_matcher_tar = self.edge_hunter(data, inv_edge_idx)          

                        H_heisen_J_on[magnetic_atoms] += oe.contract('mij, mij, kop, mk -> moipj', J_on[magnetic_atoms].type_as(sigma), weight_on[magnetic_atoms].type_as(sigma), sigma, spin_vec[magnetic_atoms].type_as(sigma))

                        zero_shift_idx = cell_index_map[(0, 0, 0)]
                        for ia in range(len(J_on)):
                            # src
                            if magnetic_atoms[ia]:
                                zero_shift_edges = edge_matcher_tar[ia][zero_shift_idx]
                                edge_matcher_src_ = torch.cat([edge_matcher_src[ia], zero_shift_edges])
                                Woff = weight_off[edge_matcher_src_]  
                                H_heisen_J_off[edge_matcher_src_] += oe.contract('ij, mij, kop, k -> moipj', J_on[ia].type_as(sigma), Woff.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))

                        for i_edge in range(len(j)):
                            ia = j[i_edge].item()
                            ja = i[i_edge].item()

                            # i
                            if magnetic_atoms[ja]:
                                Won = weight_on[ia]
                                Woff_src = weight_off[edge_matcher_src[ia]]
                                H_heisen_J_on[ia] += oe.contract('ij, ij, kop, k -> oipj', J_off[i_edge].type_as(sigma), Won.type_as(sigma), sigma, spin_vec[ja].type_as(sigma))
                                H_heisen_J_off[edge_matcher_src[ia]] += oe.contract('ij, mij, kop, k -> moipj', J_off[i_edge].type_as(sigma), Woff_src.type_as(sigma), sigma, spin_vec[ja].type_as(sigma))

                            # j
                            if magnetic_atoms[ia]:
                                Woff_tar = weight_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]]
                                H_heisen_J_off[edge_matcher_tar[ja][cell_shift_indices[i_edge]]] += oe.contract('ij, mij, kop, k -> moipj', J_off[i_edge].type_as(sigma), Woff_tar.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))
                                if cell_shift_indices[i_edge] == data['cell_index_map'][(0,0,0)]:
                                    Won = weight_on[ja]
                                    H_heisen_J_on[ja] += oe.contract('ij, ij, kop, k -> oipj', J_off[i_edge].type_as(sigma), Won.type_as(sigma), sigma, spin_vec[ia].type_as(sigma))                                

                if not self.collinear_spin:
                    Hsoc_on_real =  Hsoc_on_real + H_heisen_J_on.reshape(-1, (2*self.nao_max)**2).real
                    Hsoc_off_real = Hsoc_off_real + H_heisen_J_off.reshape(-1, (2*self.nao_max)**2).real
                    Hsoc_on_imag = Hsoc_on_imag + H_heisen_J_on.reshape(-1, (2*self.nao_max)**2).imag
                    Hsoc_off_imag = Hsoc_off_imag + H_heisen_J_off.reshape(-1, (2*self.nao_max)**2).imag
                    
                    if self.symmetrize:
                        Hsoc_on_real = self.symmetrize_Hon_soc(Hsoc_on_real, sign='+')
                        Hsoc_off_real = self.symmetrize_Hoff_soc(Hsoc_off_real, inv_edge_idx, sign='+')
                        Hsoc_on_imag = self.symmetrize_Hon_soc(Hsoc_on_imag, sign='-')
                        Hsoc_off_imag = self.symmetrize_Hoff_soc(Hsoc_off_imag, inv_edge_idx, sign='-')
                else:
                    Hcol_on = torch.stack([Hon.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_on[:,0,:,0,:], Hon.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_on[:,1,:,1,:]], dim=1).reshape(-1, 2, (self.nao_max)**2)
                    Hcol_off = torch.stack([Hoff.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_off[:,0,:,0,:], Hoff.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_off[:,1,:,1,:]], dim=1).reshape(-1, 2, (self.nao_max)**2)
                    
            if self.add_H0:
                if not self.collinear_spin:
                    Hsoc_on_real =  Hsoc_on_real + data['Hon0']
                    Hsoc_off_real = Hsoc_off_real + data['Hoff0']
                    Hsoc_on_imag = Hsoc_on_imag + data['iHon0']
                    Hsoc_off_imag = Hsoc_off_imag + data['iHoff0']
                else:
                    Hcol_on =  Hcol_on + data['Hon0']
                    Hcol_off = Hcol_off + data['Hoff0']   

            if not self.collinear_spin:
                Hsoc_real = self.cat_onsite_and_offsite(data, Hsoc_on_real, Hsoc_off_real)
                Hsoc_imag = self.cat_onsite_and_offsite(data, Hsoc_on_imag, Hsoc_off_imag)

                data['hamiltonian_real'] = self.cat_onsite_and_offsite(data, data['Hon'], data['Hoff'])
                data['hamiltonian_imag'] = self.cat_onsite_and_offsite(data, data['iHon'], data['iHoff'])

                Hsoc = torch.cat((Hsoc_real, Hsoc_imag), dim=0)
                data['hamiltonian'] = torch.cat((data['hamiltonian_real'], data['hamiltonian_imag']), dim=0)

                if self.calculate_band_energy:
                    k_vecs = []
                    for idx in range(data['batch'][-1]+1):
                        cell = data['cell']
                        # Generate K point path
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
                Hcol = self.cat_onsite_and_offsite(data, Hcol_on, Hcol_off)
                data['hamiltonian'] = self.cat_onsite_and_offsite(data, data['Hon'], data['Hoff'])
                
                # cal band energy
                if self.calculate_band_energy:
                    k_vecs = []
                    for idx in range(data['batch'][-1]+1):
                        cell = data['cell']
                        # Generate K point path
                        if isinstance(self.k_path, list):
                            kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                        elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                            # build crystal structure
                            latt = cell[idx].detach().cpu().numpy()*au2ang
                            pos = torch.split(data['pos'], data['node_counts'].tolist(), dim=0)[idx].detach().cpu().numpy()*au2ang
                            species = torch.split(data['z'], data['node_counts'].tolist(), dim=0)[idx]
                            struct = Structure(lattice=latt, species=[Element.from_Z(k.item()).symbol for k in species], coords=pos, coords_are_cartesian=True)
                            # Initialize k_path and label
                            kpath_seek = KPathSeek(structure = struct)
                            klabels = []
                            for lbs in kpath_seek.kpath['path']:
                                klabels += lbs
                            # remove adjacent duplicates   
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
                        band_energy_up, wavefunction_up, HK_up, SK_up, dSK_up, gap_up = self.cal_band_energy(Hcol_on[:,0,:], Hcol_off[:,0,:], data, True)
                        band_energy_down, wavefunction_down, HK_down, SK_down, dSK_down, gap_down = self.cal_band_energy(Hcol_on[:,1,:], Hcol_off[:,1,:], data, True)
                        H_sym = None
                        band_energy = torch.cat([band_energy_up, band_energy_down])
                        wavefunction = torch.cat([wavefunction_up, wavefunction_down])
                        HK = torch.cat([HK_up, HK_down])
                        gap = torch.cat([gap_up, gap_down])
                    else:
                        band_energy_up, wavefunction_up, gap_up, H_sym = self.cal_band_energy(Hcol_on[:,0,:], Hcol_off[:,0,:], data)
                        band_energy_down, wavefunction_down, gap_down, H_sym = self.cal_band_energy(Hcol_on[:,1,:], Hcol_off[:,1,:], data)
                        band_energy = torch.cat([band_energy_up, band_energy_down])
                        wavefunction = torch.cat([wavefunction_up, wavefunction_down])
                        gap = torch.cat([gap_up, gap_down])
                    with torch.no_grad():
                        data['band_energy_up'], data['wavefunction'], data['band_gap_up'], data['H_sym'] = self.cal_band_energy(data['Hon'][:,0,:], data['Hoff'][:,0,:], data)
                        data['band_energy_down'], data['wavefunction'], data['band_gap_down'], data['H_sym'] = self.cal_band_energy(data['Hon'][:,1,:], data['Hoff'][:,1,:], data)
                        data['band_energy'] = torch.cat([data['band_energy_up'], data['band_energy_down']])
                        data['band_gap'] = torch.cat([data['band_gap_up'], data['band_gap_down']])
                else:
                    band_energy = None
                    wavefunction = None
                    gap = None
                    H_sym = None        
        
        # non-soc and non-magnetic
        else:                
            node_sph = self.onsitenet_h(node_attr)
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Hon = self.change_index(Hon)
        
            # Impose Hermitian symmetry for Hon
            Hon = self.symmetrize_Hon(Hon)
            if self.add_H0:
                Hon = Hon + data['Hon0']

            # Calculate the off-site Hamiltonian
            # Calculate the contribution of the edges       
            edge_sph = self.offsitenet_h(edge_attr)
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
                    # Generate K point path
                    if isinstance(self.k_path, list):
                        kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                    elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                        # build crystal structure
                        latt = cell[idx].detach().cpu().numpy()*au2ang
                        pos = torch.split(data['pos'], data['node_counts'].tolist(), dim=0)[idx].detach().cpu().numpy()*au2ang
                        species = torch.split(data['z'], data['node_counts'].tolist(), dim=0)[idx]
                        struct = Structure(lattice=latt, species=[Element.from_Z(k.item()).symbol for k in species], coords=pos, coords_are_cartesian=True)
                        # Initialize k_path and label
                        kpath_seek = KPathSeek(structure = struct)
                        klabels = []
                        for lbs in kpath_seek.kpath['path']:
                            klabels += lbs
                        # remove adjacent duplicates   
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
                      
        # Combining on-site and off-site Hamiltonians
        # openmx
        if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:                
            if self.soc_switch or self.spin_constrained:
                if not self.collinear_spin:
                    if self.zero_point_shift:
                        # calculate miu
                        S = data['overlap'].reshape(-1, self.nao_max, self.nao_max)                        
                        S_soc = blockwise_2x2_concat(S, torch.zeros_like(S), torch.zeros_like(S), S).reshape(-1, (2*self.nao_max)**2)
                        sum_S_soc = 2*torch.sum(S[S > 1e-6])                        
                        miu_real = torch.sum(extract_elements_above_threshold(S_soc, Hsoc_real-data['hamiltonian_real'], 1e-6))/sum_S_soc
                        # shift Hamiltonian and band_energy
                        Hsoc_real = Hsoc_real-miu_real*S_soc
                        Hsoc = torch.cat((Hsoc_real, Hsoc_imag), dim=0)
                        band_energy = band_energy-torch.mean(band_energy-data['band_energy']) if band_energy is not None else band_energy
                    
                    result = {'hamiltonian': Hsoc, 'hamiltonian_real':Hsoc_real, 'hamiltonian_imag':Hsoc_imag, 
                              'band_energy': band_energy, 'wavefunction': wavefunction}
                    
                    if self.get_nonzero_mask_tensor:
                        mask_real_imag, mask_all = self.mask_tensor_builder_soc(data)
                        result['mask_real_imag'] = mask_real_imag
                    
                else: # collinear_spin
                    if self.zero_point_shift:
                        # calculate miu
                        S = data['overlap']
                        S_col = torch.stack([S, S], dim=1) # (Nbatchs, 2, nao_max**2)
                        sum_S_col = 2*torch.sum(S[S > 1e-6]) 
                        miu = torch.sum(extract_elements_above_threshold(S_col, Hcol-data['hamiltonian'], 1e-6))/sum_S_col
                        # shift Hamiltonian and band_energy
                        Hcol = Hcol - miu*S_col
                        band_energy = band_energy-torch.mean(band_energy-data['band_energy']) if band_energy is not None else band_energy
                    result = {'hamiltonian': Hcol, 'band_energy': band_energy, 'wavefunction': wavefunction}
                    
                    if self.get_nonzero_mask_tensor:
                        mask_all = self.mask_tensor_builder_col(data)
                        result['mask'] = mask_all             
            else:
                H = self.cat_onsite_and_offsite(data, Hon, Hoff)
                if self.zero_point_shift:
                    # calculate miu
                    S = data['overlap']  
                    sum_S = torch.sum(S[S > 1e-6]) 
                    miu = torch.sum(extract_elements_above_threshold(S, H-data['hamiltonian'], 1e-6))/sum_S
                    # shift Hamiltonian and band_energy
                    H = H-miu*data['overlap']
                    band_energy = band_energy-torch.mean(band_energy-data['band_energy']) if band_energy is not None else band_energy                
                
                result = {'hamiltonian': H, 'band_energy': band_energy, 'wavefunction': wavefunction, 'band_gap':gap, 'H_sym': H_sym}
                if self.export_reciprocal_values:
                    result.update({'HK':HK, 'SK':SK, 'dSK': dSK})
                
                if self.get_nonzero_mask_tensor:
                    mask_all = self.mask_tensor_builder(data)
                    result['mask'] = mask_all
                
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
