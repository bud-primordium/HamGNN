import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations

"""
存储CG系数的辅助类
"""

class ClebschGordan(nn.Module):
    """一个用于加载、存储和提供 Clebsch-Gordan (CG) 系数的辅助类。

    CG系数在耦合角动量理论中至关重要，用于将两个角动量的状态分解为总角动量的状态。
    这个类从一个预先计算好的 `.npz` 文件中加载 CG 系数张量。由于原始文件
    只存储了满足 l1 <= l2 <= l3 的系数以减少冗余，该类在初始化时会
    自动生成所有可能的 (l1, l2, l3) 排列组合，并将它们注册为 PyTorch 的
    缓冲区 (buffers)，以便高效访问。

    .. note::
        该类继承自 `torch.nn.Module`，这使得它可以被无缝集成到任何
        PyTorch 模型中，并且通过 `register_buffer` 注册的张量会自动
        随模型移动到正确的设备 (如 CPU 或 GPU)。

    Attributes:
        (动态) cg_l1_l2_l3 (torch.Tensor):
            注册的 CG 系数张量。例如，`self.cg_1_1_2` 存储了耦合 l1=1 和 l2=1
            得到 l3=2 的 CG 系数。
    """
    def __init__(self):
        """初始化 ClebschGordan 类。

        此构造函数执行以下操作：
        1. 定位并加载 `clebsch_gordan_coefficients_L10.npz` 文件。
        2. 从文件中提取存储在 'cg' 键下的系数字典。
        3. 遍历字典中的每一个 (l1, l2, l3) 组合。
        4. 为每种组合生成所有可能的轴排列 (permutations)。
        5. 将原始张量和所有排列后的新张量注册为 PyTorch 缓冲区，
           命名格式为 `cg_l1_l2_l3`。
        """
        super(ClebschGordan, self).__init__()
        # 加载预计算的CG系数，该文件位于当前脚本所在目录
        cg_file = os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz')
        # allow_pickle=True 是为了兼容旧版Numpy创建的.npz文件
        # [()] 用于从0维数组中提取出字典对象
        tmp = np.load(cg_file, allow_pickle=True)['cg'][()]

        # 原始 .npz 文件为了节省空间，只存储了 l1 <= l2 <= l3 的情况。
        # 这里需要生成所有可能的 (l1, l2, l3) 组合的系数。
        for l123 in tmp.keys():
            # (0, 1, 2) 代表 (l1, l2, l3) 的索引
            for a, b, c in permutations((0, 1, 2)):
                # 根据排列组合动态生成缓冲区的名称，例如 'cg_1_2_1'
                name = 'cg_{}_{}_{}'.format(l123[a], l123[b], l123[c])
                # 检查该名称是否已存在，避免重复注册
                if name not in dir(self):
                    # self.register_buffer() 将张量注册为模块的持久状态，但不是模型参数。
                    # 它会被保存到 state_dict 中，并随模型移动到 GPU。
                    # .transpose(a,b,c) 根据索引 (a,b,c) 调整张量的维度顺序。
                    self.register_buffer(name, torch.tensor(tmp[l123].transpose(a, b, c)))

    def forward(self, l1: int, l2: int, l3: int) -> torch.Tensor:
        """获取指定角动量组合的 Clebsch-Gordan 系数张量。

        Args:
            l1 (int): 第一个角动量的量子数。
            l2 (int): 第二个角动量的量子数。
            l3 (int): 耦合后的总角动量的量子数。

        Returns:
            torch.Tensor: 对应于 (l1, l2, l3) 组合的 CG 系数张量。
        """
        # 使用 getattr() 动态地从 self 中获取名为 'cg_l1_l2_l3' 的属性
        return getattr(self, 'cg_{}_{}_{}'.format(l1, l2, l3))