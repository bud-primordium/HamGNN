"""此模块提供了计算维格纳D矩阵 (Wigner D-matrix) 的函数。

维格纳D矩阵是三维空间中旋转算符在角动量本征态基矢下的矩阵表示，
在处理具有旋转对称性的物理问题（如分子轨道、张量场等）中至关重要。
此实现目前支持 l=0, 1, 2 的情况。
"""

import torch
from e3nn import o3

def wigner(l: int, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """根据旋转轴和旋转角计算维格纳D矩阵。

    该函数为给定的角动量 `l` 计算一个旋转操作的不可约表示 (Irrep)。
    目前仅实现了 l=0, 1, 2 的情况。

    - l=0: 返回一个标量 1。
    - l=1: 使用 e3nn 库计算标准的 3x3 旋转矩阵。
    - l=2: 首先计算 l=1 的旋转矩阵 R, 然后基于 R 的元素显式构造 5x5 的D矩阵。

    Args:
        l (int):
            角动量量子数，决定了表示的维度 (2l+1)。
        axis (torch.Tensor):
            一个三维向量，表示旋转轴。
        angle (torch.Tensor):
            一个标量张量，表示绕轴旋转的角度（单位：弧度）。

    Returns:
        torch.Tensor:
            计算出的维格纳D矩阵，形状为 (2l+1, 2l+1)。

    Raises:
        ValueError: 如果 `l` 不是 0, 1, 或 2，则抛出此异常。
    """
    if l == 0:
        w = torch.Tensor([1.0]).type_as(angle)
    elif l == 1:
        w = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
    elif l == 2:
        R = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
        w = torch.Tensor([[R[0,0]*R[1,1]+R[0,1]*R[1,0], R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,2]*R[1,2], R[0,0]*R[1,2]+R[0,2]*R[1,0], R[0,0]*R[1,0]-R[0,1]*R[1,1]],
                 [R[1,0]*R[2,1]+R[1,1]*R[2,0], R[1,1]*R[2,2]+R[1,2]*R[2,1], R[1,2]*R[2,2], R[1,0]*R[2,2]+R[1,2]*R[2,0], R[1,0]*R[2,0]-R[1,1]*R[2,1]],
                 [2.0*R[2,0]*R[2,1]-R[0,0]*R[0,1]-R[1,0]*R[1,1], 2.0*R[2,1]*R[2,2]-R[0,1]*R[0,2]-R[1,1]*R[1,2], R[2,2]*R[2,2]-0.5*R[0,2]*R[0,2]-0.5*R[1,2]*R[1,2], 2.0*R[2,0]*R[2,2]-R[0,0]*R[0,2]-R[1,0]*R[1,2], R[2,0]*R[2,0]+0.5*R[0,1]*R[0,1]+0.5*R[1,1]*R[1,1]-0.5*R[0,0]*R[0,0]-0.5*R[1,0]*R[1,0]-R[2,1]*R[2,1]],
                 [R[0,0]*R[2,1]+R[0,1]*R[2,0], R[0,1]*R[2,2]+R[0,2]*R[2,1], R[0,2]*R[2,2], R[0,0]*R[2,2]+R[0,2]*R[2,0], R[0,0]*R[2,0]-R[0,1]*R[2,1]],
                 [R[0,0]*R[0,1]-R[1,0]*R[1,1], R[0,1]*R[0,2]-R[1,1]*R[1,2], 0.5*(R[0,2]*R[0,2]-R[1,2]*R[1,2]), R[0,0]*R[0,2]-R[1,0]*R[1,2], 0.5*(R[0,0]*R[0,0]+R[1,1]*R[1,1]-R[1,0]*R[1,0]-R[0,1]*R[0,1])]]).type_as(angle)
    else:
        raise ValueError("l must be 0, 1, or 2")
    return w
            
        