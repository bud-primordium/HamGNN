"""该模块提供了计算 Wigner-D 矩阵的函数。

注意：此版本的实现与 `e3nn` 库紧密耦合，并为 l=0, 1, 2 的情况
提供了具体的矩阵表达式。
"""
from numpy import zeros
import torch
import torch.nn as nn
from torch_geometric.data import Data, batch
from torch.nn import (Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh)
from ..utils import linear_bn_act
from ..layers import denseRegression
from torch_scatter import scatter
import sympy as sym
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import Gate, NormActivation
from easydict import EasyDict
from typing import Union
from ..layers import GaussianSmearing, cuttoff_envelope, CosineCutoff, BesselBasis
from .nequip.data import AtomicDataDict, AtomicDataset
import math
# from ..PhiSNet.modules.clebsch_gordan import ClebschGordan  # PhiSNet目录不存在，暂时注释
import copy
from typing import Dict, Callable

def wigner(l, axis, angle):
    """计算指定 l 值的 Wigner-D 矩阵。

    该函数使用 e3nn 库来获取 l=1 的旋转矩阵，并为 l=2 的情况提供了
    一个硬编码的、基于 l=1 旋转矩阵元素的表达式。

    Args:
        l (int): 轨道角动量量子数 (支持 0, 1, 2)。
        axis (torch.Tensor): 旋转轴向量，形状为 (3,)。
        angle (torch.Tensor): 旋转角度 (弧度)，是一个标量。

    Returns:
        torch.Tensor: 对应于 l 值的 Wigner-D 矩阵。

    Raises:
        ValueError: 如果 l 的值不是 0, 1, 或 2。
    """
    if l == 0:
        # l=0 是标量，旋转不变
        w = torch.Tensor([1.0]).type_as(angle)
    elif l == 1:
        # l=1 是矢量，使用 e3nn 计算 3x3 旋转矩阵
        w = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
    elif l == 2:
        # l=2 是二阶张量，使用硬编码的公式从 l=1 的旋转矩阵 R 计算
        R = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
        w = torch.Tensor([[R[0,0]*R[1,1]+R[0,1]*R[1,0], R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,2]*R[1,2], R[0,0]*R[1,2]+R[0,2]*R[1,0], R[0,0]*R[1,0]-R[0,1]*R[1,1]],
                 [R[1,0]*R[2,1]+R[1,1]*R[2,0], R[1,1]*R[2,2]+R[1,2]*R[2,1], R[1,2]*R[2,2], R[1,0]*R[2,2]+R[1,2]*R[2,0], R[1,0]*R[2,0]-R[1,1]*R[2,1]],
                 [2.0*R[2,0]*R[2,1]-R[0,0]*R[0,1]-R[1,0]*R[1,1], 2.0*R[2,1]*R[2,2]-R[0,1]*R[0,2]-R[1,1]*R[1,2], R[2,2]*R[2,2]-0.5*R[0,2]*R[0,2]-0.5*R[1,2]*R[1,2], 2.0*R[2,0]*R[2,2]-R[0,0]*R[0,2]-R[1,0]*R[1,2], R[2,0]*R[2,0]+0.5*R[0,1]*R[0,1]+0.5*R[1,1]*R[1,1]-0.5*R[0,0]*R[0,0]-0.5*R[1,0]*R[1,0]-R[2,1]*R[2,1]],
                 [R[0,0]*R[2,1]+R[0,1]*R[2,0], R[0,1]*R[2,2]+R[0,2]*R[2,1], R[0,2]*R[2,2], R[0,0]*R[2,2]+R[0,2]*R[2,0], R[0,0]*R[2,0]-R[0,1]*R[2,1]],
                 [R[0,0]*R[0,1]-R[1,0]*R[1,1], R[0,1]*R[0,2]-R[1,1]*R[1,2], 0.5*(R[0,2]*R[0,2]-R[1,2]*R[1,2]), R[0,0]*R[0,2]-R[1,0]*R[1,2], 0.5*(R[0,0]*R[0,0]+R[1,1]*R[1,1]-R[1,0]*R[1,0]-R[0,1]*R[0,1])]]).type_as(angle)
    else:
        raise ValueError
    return w