'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-06-20 21:33:29
LastEditors: Yang Zhong
LastEditTime: 2024-06-20 21:35:59
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .electron_configurations import *

"""用于将原子序数 Z 转换为特征向量的嵌入层模块。
"""
class Embedding(nn.Module):
    """嵌入层，将标量的原子序数 Z 转换为指定维度的特征向量。

    该模块结合了两种信息来生成每个元素的嵌入表示：
    1. 一个可学习的嵌入矩阵，针对每个原子序数进行优化。
    2. 基于元素电子结构的线性变换，提供物理上有意义的先验信息。
    这两种表示相加，形成最终的元素特征向量。

    Attributes:
        num_features (int): 输出特征向量的维度。
        Zmax (int): 支持的最大原子序数。
        electron_config (torch.Tensor): 存储元素电子结构信息的张量，作为 buffer 注册。
        element_embedding (nn.Parameter): 可学习的元素嵌入矩阵，形状为 `(Zmax, num_features)`。
        config_linear (nn.Linear): 将电子结构向量线性变换到特征空间的层。
    """
    def __init__(self, num_features: int, Zmax: int = 87):
        """构造 Embedding 类的实例。

        Args:
            num_features (int): 输出特征向量的维度。
            Zmax (int): 支持的最大原子序数，默认为 87（钫）。
        """
        super(Embedding, self).__init__()
        self.num_features = num_features
        self.Zmax = Zmax
        # 注册电子结构表，使其成为模型状态的一部分但不参与梯度计算
        self.register_buffer('electron_config', torch.tensor(electron_configurations))
        # 注册可学习的元素嵌入参数
        self.register_parameter('element_embedding', nn.Parameter(torch.Tensor(self.Zmax, self.num_features))) 
        # 用于处理电子结构特征的线性层
        self.config_linear = nn.Linear(self.electron_config.size(1), self.num_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化或重置模型的参数。

        对元素嵌入矩阵使用均匀分布进行初始化，对线性层的权重使用正交初始化。
        """
        # 使用均匀分布初始化可学习的元素嵌入
        nn.init.uniform_(self.element_embedding, -np.sqrt(3), np.sqrt(3))
        # 使用正交初始化线性层的权重，有助于保持特征的独立性
        nn.init.orthogonal_(self.config_linear.weight)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """定义模型的前向传播逻辑。

        Args:
            Z (torch.Tensor): 输入的原子序数张量，形状为 `(N,)`，其中 N 是原子数。

        Returns:
            torch.Tensor:
                返回每个原子对应的嵌入向量，形状为 `(N, num_features)`。
        """
        # 将可学习的嵌入与电子结构变换后的特征相加，得到最终的组合嵌入
        embedding = self.element_embedding + self.config_linear(self.electron_config)
        # 根据输入的原子序数 Z，从组合嵌入矩阵中索引相应的向量
        return embedding[Z]
