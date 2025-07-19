"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 10:18:50 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-28 20:19:52
 */
"""
"""
本模块定义了 HamGNN 模型中使用的各种基础神经网络层和激活函数。
这些层是构建更复杂的等变网络模块的基础组件，包括自定义的全连接层、
回归网络、径向基函数以及截断函数等。
"""
import torch
import torch.nn as nn
from .utils import linear_bn_act
from torch_geometric.data import Data, batch
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh)
from typing import Callable
import sympy as sym
import math
from torch_geometric.nn.models.dimenet_utils import real_sph_harm
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from math import pi
from functools import partial

# 默认的权重初始化函数，使用零进行初始化
zeros_initializer = partial(constant_, val=0.0)


class denseLayer(nn.Module):
    """一个包含线性层、批量归一化和激活函数的残差连接稠密层。

    结构为: `out = Linear(Act(BN(Linear(x))))`
    """
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True, 
                    use_batch_norm:bool=True, activation:callable=ELU()):
        """
        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            bias (bool): 是否在线性层中使用偏置。
            use_batch_norm (bool): 是否使用批量归一化。
            activation (callable): 激活函数。
        """
        super().__init__()
        self.lba = linear_bn_act(in_features=in_features, out_features=out_features, lbias=bias, 
                        activation=activation, use_batch_norm=use_batch_norm)
        self.linear = Linear(out_features, out_features, bias=bias)
    def forward(self, x):
        """前向传播。"""
        out = self.linear(self.lba(x))
        return out

class denseRegression(nn.Module):
    """一个用于回归任务的多层稠密网络。
    
    包含多个隐藏层和一个最终的线性输出层。
    """
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True, 
                    use_batch_norm:bool=True, activation:callable=Softplus(), n_h:int=3):
        """
        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度（回归目标维度）。
            bias (bool): 是否在线性层中使用偏置。
            use_batch_norm (bool): 是否使用批量归一化。
            activation (callable): 隐藏层使用的激活函数。
            n_h (int): 隐藏层的数量。
        """
        super().__init__()
        if n_h > 1:
            self.fcs = nn.ModuleList([linear_bn_act(in_features=in_features, out_features=in_features, lbias=bias, 
                        activation=activation, use_batch_norm=use_batch_norm) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(in_features, out_features)

    def forward(self, x):
        """前向传播。"""
        for fc in self.fcs:
            x = fc(x)
        out = self.fc_out(x)
        return out

class cuttoff_envelope(nn.Module):
    """多项式形式的截断包络函数。
    
    当距离 `x` 接近 `cutoff` 时，该函数平滑地衰减到零。
    """
    def __init__(self, cutoff, exponent=6):
        """
        Args:
            cutoff (float): 截断半径。
            exponent (int): 多项式的指数。
        """
        super(cuttoff_envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
        self.cutoff = cutoff

    def forward(self, x):
        """前向传播。"""
        p, a, b, c = self.p, self.a, self.b, self.c
        x = x/self.cutoff
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        # 仅对小于 cutoff 的距离计算包络，其余为零
        return (1. + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < self.cutoff).float()

class CosineCutoff(nn.Module):
    r"""Behler-Parrinello 余弦截断函数。源自 SchNetPack。

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): 截断半径。

    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """计算截断值。

        Args:
            distances (torch.Tensor): 原子间距离张量。

        Returns:
            torch.Tensor: 截断函数的值。

        """
        # 计算截断函数的值
        cutoffs = 0.5 * (torch.cos(distances * 3.141592653589793 / self.cutoff) + 1.0)
        # 移除超出截断半径的贡献
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs

class MLPRegression(nn.Module):
    """一个多层感知机（MLP）回归网络，特征维度在层间减半。"""
    def __init__(self, num_in_features: int, num_out_features: int, num_mlp: int=3, lbias: bool = False,
                 activation: Callable = ELU(), use_batch_norm: bool = False):
        """
        Args:
            num_in_features (int): 输入特征维度。
            num_out_features (int): 输出特征维度。
            num_mlp (int): MLP 的层数。
            lbias (bool): 是否在线性层中使用偏置。
            activation (Callable): 激活函数。
            use_batch_norm (bool): 是否使用批量归一化。
        """
        super(MLPRegression, self).__init__()
        self.linear_regression = [linear_bn_act(int(num_in_features/2**(i-1)), int(num_in_features/2**i), 
                                   lbias, activation, use_batch_norm) for i in range(1, num_mlp)]
        self.linear_regression += [linear_bn_act(int(num_in_features/2**(num_mlp-1)), num_out_features, 
                                    lbias, activation, use_batch_norm)]                           
        self.linear_regression = ModuleList(self.linear_regression)

    def forward(self, x):
        """前向传播。"""
        for lr in self.linear_regression:
            x = lr(x)
        return x

class sph_harm_layer(nn.Module):
    """计算实数球谐函数的层。"""
    def __init__(self, num_spherical):
        """
        Args:
            num_spherical (int): 球谐函数的阶数 l。
        """
        super(sph_harm_layer, self).__init__()
        self.num_spherical = num_spherical
        # 使用 PyTorch Geometric 的工具生成符号化的球谐函数表达式
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []

        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                # l=0 时，Y00 是常数
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                # 将符号表达式转换为可计算的 PyTorch 函数
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)

    def forward(self, angle):
        """前向传播。
        
        Args:
            angle (torch.Tensor): 极角 theta。
            
        Returns:
            torch.Tensor: 拼接后的球谐函数值。
        """
        out = torch.cat([f(angle.unsqueeze(-1)) for f in self.sph_funcs], dim=-1)
        return out

class BesselBasis(nn.Module):
    """
    零阶贝塞尔函数作为径向基函数（来自 DimeNet）。
    """
    def __init__(self, cutoff=5.0, n_rbf:int=None, cutoff_func:callable=None):
        """
        Args:
            cutoff (float): 径向截断半径。
            n_rbf (int): 径向基函数的数量。
            cutoff_func (callable, optional): 应用于基函数的截断函数。
        """
        super(BesselBasis, self).__init__()
        # 计算贝塞尔函数的频率
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.cutoff_func = cutoff_func

    def forward(self, dist):
        r"""计算原子间距离的零阶贝塞尔展开。

            Args:
                dist (torch.Tensor): 形状为 (N_edge,) 的原子间距离。

            Returns:
                rbf (torch.Tensor): 形状为 (N_edge, n_rbf) 的零阶贝塞尔展开。
            """
        a = self.freqs[None, :]
        ax = dist.unsqueeze(-1) * a
        # 零阶球贝塞尔函数 j0(x) = sin(x)/x
        rbf = torch.sin(ax) / dist.unsqueeze(-1)
        if self.cutoff_func is not None:
            rbf = rbf * self.cutoff_func(dist.unsqueeze(-1))
        return rbf

class GaussianSmearing(nn.Module):
    """高斯函数作为径向基函数。"""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, cutoff_func=None):
        """
        Args:
            start (float): 第一个高斯函数的中心位置。
            stop (float): 最后一个高斯函数的中心位置。
            num_gaussians (int): 高斯函数的数量。
            cutoff_func (callable, optional): 应用于基函数的截断函数。
        """
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽度系数
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
        self.cutoff_func = cutoff_func
    def forward(self, dist):
        """前向传播。"""
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        expansion = torch.exp(self.coeff * torch.pow(dist, 2))
        if self.cutoff_func is not None:
            expansion = expansion*self.cutoff_func(dist)
        return expansion

class Dense(nn.Linear):
    r"""源自 SchNetPack 的全连接线性层，带有激活函数。

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): 输入特征数 :math:`x`。
        out_features (int): 输出特征数 :math:`y`。
        bias (bool, optional): 如果为 False, 层将不学习偏置 :math:`b`。
        activation (callable, optional): 如果为 None, 则不使用激活函数。
        weight_init (callable, optional): 权重的初始化函数。
        bias_init (callable, optional): 偏置的初始化函数。

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)
        self.activation = activation
        # 初始化线性层 y = xW^T + b

    def reset_parameters(self):
        """重新初始化模型的权重和偏置值。"""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """计算层的输出。

        Args:
            inputs (torch.Tensor): 输入值的批次。

        Returns:
            torch.Tensor: 层的输出。

        """
        # 计算线性层 y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # 添加激活函数
        if self.activation:
            y = self.activation(y)
        return y

