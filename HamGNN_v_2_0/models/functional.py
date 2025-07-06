'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-06-20 21:29:58
LastEditors: Yang Zhong
LastEditTime: 2024-06-21 11:53:21
'''
"""此模块提供了一系列在神经网络模型中常用的、可微的函数。

这些函数包括特殊的激活函数、平滑的截断/开关函数以及用于参数初始化的
反函数。模块中的函数都经过精心设计，以确保在自动微分过程中的数值稳定性。
"""
import math
import torch
import torch.nn.functional as F
from typing import Union

# 重要的数值稳定性说明：
# 下面的截断 (cutoff) 和开关 (switch) 函数在数值上需要一些技巧来处理。
# 在这些函数的分段定义变化的连接点处，形式上会出现 0/0 的除法。
# 这对于函数值本身没有问题，但当使用自动微分时，这种除法会导致梯度计算
# 出现 NaN (Not a Number)。为了规避这个问题，函数的输入也需要被适当地掩码 (mask)。
# （保留此英文注释以强调其重要性）
# IMPORTANT NOTE: The cutoff and the switch function are numerically a bit tricky:
# Right at the "seems" of these functions, i.e. where the piecewise definition changes,
# there is formally a division by 0 (i.e. 0/0). This is of no issue for the function
# itself, but when automatic differentiation is used, this division will lead to NaN
# gradients. In order to circumvent this, the input needs to be masked as well.


_log2 = math.log(2)
def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """计算修正版的 Softplus 激活函数。

    标准的 Softplus(x) = log(1 + exp(x))。此版本将其向下平移 log(2)，
    即 Softplus(x) - log(2)，这使得函数图像恰好经过原点 (0, 0)。

    Args:
        x (torch.Tensor): 输入张量。

    Returns:
        torch.Tensor: 经过修正版 Softplus 计算后的输出张量。
    """
    return F.softplus(x) - _log2


def cutoff_function(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """一个在 [0, cutoff] 区间内从 1 平滑过渡到 0 的截断函数。

    该函数具有无限阶平滑导数，这对于基于梯度的优化至关重要。
    当 x >= cutoff 时，函数值为 0。

    为了避免在 x=cutoff 处出现 0/0 导致梯度为 NaN 的问题，
    函数内部使用 `torch.where` 对输入 `x` 进行了掩码处理。

    函数表达式为: f(x) = exp(-x^2 / (cutoff^2 - x^2)) for x < cutoff

    Args:
        x (torch.Tensor): 输入张量，通常是距离。
        cutoff (float): 截断半径。

    Returns:
        torch.Tensor: 施加截断效果后的输出张量。
    """
    zeros = torch.zeros_like(x)
    # 当 x >= cutoff 时，将 x 替换为 0，避免在分母中出现 (cutoff - x) <= 0
    x_ = torch.where(x < cutoff, x, zeros)
    # 核心计算：exp(-x_**2 / ( (cutoff-x_) * (cutoff+x_) ))
    # 等价于 exp(-x_**2 / (cutoff**2 - x_**2))
    return torch.where(x < cutoff, torch.exp(-x_**2 / ((cutoff - x_) * (cutoff + x_))), zeros)


def _switch_component(x: torch.Tensor, ones: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """switch_function 的辅助计算组件，不应直接调用。

    该实现比简化版本在数值上更稳定，请勿修改。
    """
    # 当 x <= 0 时，将 x 替换为 1，避免在分母中出现 0
    x_ = torch.where(x <= 0, ones, x)
    # 核心计算：exp(-1/x_)
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_))

def switch_function(x: torch.Tensor, cuton: float, cutoff: float) -> torch.Tensor:
    """一个在 [cuton, cutoff] 区间内平滑对称地从 1 过渡到 0 的开关函数。

    该函数同样具有无限阶平滑导数。
    - 当 x <= cuton 时，函数值为 1。
    - 当 x >= cutoff 时，函数值为 0。
    - 如果 cuton > cutoff，过渡方向会反转，从 0 过渡到 1。

    Args:
        x (torch.Tensor): 输入张量。
        cuton (float): 开始过渡的边界。
        cutoff (float): 结束过渡的边界。

    Returns:
        torch.Tensor: 施加开关效果后的输出张量。
    """
    # 将 x 标准化到 [0, 1] 区间
    x = (x - cuton) / (cutoff - cuton)
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1 - x, ones, zeros)
    # 最终的开关函数形式为 f(1-x) / (f(x) + f(1-x))
    # 通过 torch.where 保证在区间端点处的行为正确
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm / (fp + fm)))


def softplus_inverse(x: Union[torch.Tensor, float]) -> torch.Tensor:
    """Softplus 函数的反函数。

    这在初始化需要保证为正值的参数时非常有用。例如，如果希望一个参数 `alpha`
    在通过 `F.softplus(alpha)` 后得到期望的初始值 `alpha_init`，那么
    可以将 `alpha` 初始化为 `softplus_inverse(alpha_init)`。

    反函数表达式为: f^-1(x) = x + log(-expm1(-x))

    Args:
        x (torch.Tensor | float): 输入值，即 softplus 函数的输出。

    Returns:
        torch.Tensor: 对应的 softplus 函数的输入值。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # 使用 torch.log(-torch.expm1(-x)) 来稳定地计算 log(1 - exp(-x))
    return x + torch.log(-torch.expm1(-x))
