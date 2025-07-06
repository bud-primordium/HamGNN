'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-06-20 21:30:56
LastEditors: Yang Zhong
LastEditTime: 2024-06-20 21:32:28
'''
"""此模块定义了多种用于图神经网络的径向基函数 (Radial Basis Functions, RBFs)。

RBFs 的核心作用是将一个标量（通常是原子间距离）映射到一个高维特征向量。
这使得模型能够以一种平滑且具有良好泛化能力的方式来编码距离信息。
文件中实现了基于伯恩斯坦多项式和高斯函数的多种变体。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
from .functional import cutoff_function, softplus_inverse

"""
computes radial basis functions with Bernstein polynomials
"""
class BernsteinRadialBasisFunctions(nn.Module):
    """使用伯恩斯坦多项式计算径向基函数。

    这种基函数利用伯恩斯坦多项式作为基底，将距离 `r` 映射到 `num_basis_functions` 维
    的特征空间。为了数值稳定性，计算过程在对数空间中进行。

    原始的伯恩斯坦基函数定义为:
    B_v,n(x) = C(n,v) * x^v * (1-x)^(n-v)
    其中 x = r / cutoff。

    为了避免当 x 接近 0 或 1 时可能出现的数值下溢，本实现计算其对数：
    log(B_v,n(x)) = log(C(n,v)) + v*log(x) + (n-v)*log(1-x)
    并利用 `torch.log(-torch.expm1(log(x)))` 来精确计算 `log(1-x)`。

    Attributes:
        num_basis_functions (int): 基函数的数量，即输出向量的维度。
        cutoff (torch.Tensor): 截断半径，超出此距离的相互作用将被忽略。
        logc (torch.Tensor): 预计算的对数化二项式系数 log(C(n,v))。
        n (torch.Tensor): 伯恩斯坦多项式的次数 n = N-1-v，其中 N 是基函数数量。
        v (torch.Tensor): 伯恩斯坦多项式的索引 v。
    """
    def __init__(self, num_basis_functions: int, cutoff: float):
        """
        Args:
            num_basis_functions (int): 要使用的径向基函数的数量。
            cutoff (float): 截断半径。
        """
        super(BernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        # 预计算对数化的二项式系数 log(C(n,v)) 以提高效率
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        # 注册为 buffer，使其成为模型状态的一部分但不是可训练参数
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        """此基函数没有可训练参数，因此该方法为空。"""
        pass

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """计算给定距离 `r` 的径向基函数值。

        Args:
            r (torch.Tensor): 原子间距离张量，形状为 (N_interactions, 1)。

        Returns:
            torch.Tensor: 径向基函数的值，形状为 (N_interactions, num_basis_functions)。
        """
        # 将距离映射到 [0, 1] 区间，并在对数空间中计算
        x = torch.log(r / self.cutoff)
        # log(B) = log(C) + n*log(x) + v*log(1-x)
        # 使用 torch.log(-torch.expm1(x)) 来稳定地计算 log(1-exp(x))
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        # 乘以截断函数并转换回正常空间
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf

"""
computes radial basis functions with exponential Bernstein polynomials
"""
class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    """使用指数缩放的伯恩斯坦多项式计算径向基函数。

    该变体在标准的伯恩斯坦基函数上引入了一个可学习的指数衰减因子 `alpha`。
    这使得基函数能够更好地适应不同尺度上的相互作用。

    基函数的变量从 `x = r / cutoff` 替换为 `x = exp(-alpha * r)`。
    参数 `alpha` 通过 `softplus` 函数保证其为正值。

    Attributes:
        num_basis_functions (int): 基函数的数量。
        cutoff (torch.Tensor): 截断半径。
        _alpha (nn.Parameter): 可训练的衰减系数（softplus之前）。
    """
    def __init__(self, num_basis_functions: int, cutoff: float, ini_alpha: float = 0.5):
        """
        Args:
            num_basis_functions (int): 要使用的径向基函数的数量。
            cutoff (float): 截断半径。
            ini_alpha (float, optional): `alpha` 参数的初始值。默认为 0.5。
        """
        super(ExponentialBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        # 与 BernsteinRadialBasisFunctions 相同的二项式系数预计算
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        # 注册 buffer 和可训练参数
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        """使用 softplus 的逆函数来初始化 `_alpha`，使其 softplus 后的值等于 `ini_alpha`。"""
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """计算给定距离 `r` 的径向基函数值。"""
        alpha = F.softplus(self._alpha)
        # 变量替换为 x = exp(-alpha*r)，并进入对数空间
        x = -alpha * r
        # 同样使用数值稳定的方法计算对数伯恩斯坦多项式
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf


"""
computes radial basis functions with exponential Gaussians
"""
class ExponentialGaussianRadialBasisFunctions(nn.Module):
    """使用指数缩放的高斯函数计算径向基函数。

    这种基函数由一组高斯函数构成，其中心点在指数空间上均匀分布。
    具体来说，中心点位于 `exp(-alpha*r)` 的空间中，而不是直接在 `r` 上。
    这允许基函数在靠近原子核（r较小）的地方分布得更密集。

    Attributes:
        num_basis_functions (int): 基函数的数量。
        cutoff (torch.Tensor): 截断半径。
        center (torch.Tensor): 高斯函数的中心点，在 [1, 0] 区间内线性分布。
        width (torch.Tensor): 高斯函数的宽度。
        _alpha (nn.Parameter): 可训练的指数衰减系数。
    """
    def __init__(self, num_basis_functions: int, cutoff: float, ini_alpha: float = 0.5):
        """
        Args:
            num_basis_functions (int): 要使用的径向基函数的数量。
            cutoff (float): 截断半径。
            ini_alpha (float, optional): `alpha` 参数的初始值。默认为 0.5。
        """
        super(ExponentialGaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        # 中心点在 [1, 0] 区间内线性分布
        self.register_buffer('center', torch.linspace(1, 0, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(1.0 * self.num_basis_functions, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 `_alpha` 参数。"""
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """计算给定距离 `r` 的径向基函数值。"""
        alpha = F.softplus(self._alpha)
        # exp(-width * (exp(-alpha*r) - center)^2)
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width * (torch.exp(-alpha * r) - self.center)**2)
        return rbf


"""
computes radial basis functions with exponential Gaussians
"""
class GaussianRadialBasisFunctions(nn.Module):
    """使用标准高斯函数计算径向基函数。

    这是最常见的一种径向基函数，由一组在距离空间 `r` 上均匀分布的高斯函数构成。

    Attributes:
        num_basis_functions (int): 基函数的数量。
        cutoff (torch.Tensor): 截断半径。
        center (torch.Tensor): 高斯函数的中心点，在 [0, cutoff] 区间内线性分布。
        width (torch.Tensor): 高斯函数的宽度。
        _alpha (nn.Parameter): 一个伪参数，仅为兼容性存在，不起实际作用。
    """
    def __init__(self, num_basis_functions: int, cutoff: float):
        """
        Args:
            num_basis_functions (int): 要使用的径向基函数的数量。
            cutoff (float): 截断半径。
        """
        super(GaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        # 中心点在 [0, cutoff] 之间线性分布
        self.register_buffer('center', torch.linspace(0, cutoff, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(self.num_basis_functions / cutoff, dtype=torch.float64))
        # 为了在 TensorBoard 上与其他基函数兼容而添加，但本身不起作用
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        """此基函数没有可训练参数，因此该方法为空。"""
        pass

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """计算给定距离 `r` 的径向基函数值。"""
        # exp(-width * (r - center)^2)
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width * (r - self.center)**2)
        return rbf


"""
computes radial basis functions with overlap Bernstein polynomials
"""
class OverlapBernsteinRadialBasisFunctions(nn.Module):
    """使用一种重叠变体的伯恩斯坦多项式计算径向基函数。

    与指数伯恩斯坦基函数类似，但也引入了一个可学习的参数 `alpha`，但其
    对距离 `r` 的变换方式不同，旨在更好地描述原子轨道的重叠积分。

    变量替换为 `x = log(1 + alpha*r) - alpha*r`。
    """
    def __init__(self, num_basis_functions: int, cutoff: float, ini_alpha: float = 0.5):
        """
        Args:
            num_basis_functions (int): 要使用的径向基函数的数量。
            cutoff (float): 截断半径。
            ini_alpha (float, optional): `alpha` 参数的初始值。默认为 0.5。
        """
        super(OverlapBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        # 预计算对数化的二项式系数
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        # 注册 buffer 和可训练参数
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 `_alpha` 参数。"""
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """计算给定距离 `r` 的径向基函数值。"""
        alpha_r = F.softplus(self._alpha) * r
        # 使用不同的变量替换 x
        x = torch.log1p(alpha_r) - alpha_r
        # 同样使用数值稳定的方法计算对数伯恩斯坦多项式
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf
