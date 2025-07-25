"""该模块提供了一个用于存储和访问 Clebsch-Gordan (CG) 系数的辅助类。

CG 系数在耦合角动量时使用，是计算张量积所必需的。这个类通过从预先计算好的
.npz 文件中加载系数，并将其注册为 PyTorch 的缓冲区（buffer），从而实现高效访问。
"""
import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations

"""
Helper class that stores Clebsch-Gordan coefficients
"""
class ClebschGordan(nn.Module):
    """一个用于存储和提供 Clebsch-Gordan 系数的辅助类。

    它从一个预先计算好的 .npz 文件加载 CG 系数。为了方便使用，
    它还处理了角动量量子数 (l1, l2, l3) 的排列组合，因为原始文件
    可能只存储了 l1 <= l2 <= l3 的情况。
    """
    def __init__(self):
        """初始化 ClebschGordan 类。
        
        该方法会加载 .npz 文件，并为 (l1, l2, l3) 的所有排列组合
        创建并注册相应的 CG 系数张量作为 PyTorch 的 buffer。
        """
        super(ClebschGordan, self).__init__()
        # 加载包含预计算 CG 系数的 .npz 文件
        tmp = np.load(os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz'), allow_pickle=True)['cg'][()]
        #add permutations (the npz file only stores coefficients for l1 <= l2 <= l3) and register buffers
        # 遍历加载的 CG 系数，并为 l1, l2, l3 的所有排列创建 buffer
        for l123 in tmp.keys():
            for a,b,c in permutations((0,1,2)):
                name = 'cg_{}_{}_{}'.format(l123[a],l123[b],l123[c])
                if name not in dir(self):
                    # 将 CG 系数张量注册为 buffer，这样它会被模型追踪，但不会被视为模型参数
                    self.register_buffer(name, torch.tensor(tmp[l123].transpose(a,b,c)))

    def forward(self, l1, l2, l3):
        """获取指定 (l1, l2, l3) 组合的 Clebsch-Gordan 系数张量。

        Args:
            l1 (int): 第一个角动量量子数。
            l2 (int): 第二个角动量量子数。
            l3 (int): 耦合后的角动量量子数。

        Returns:
            torch.Tensor: 对应的 CG 系数张量。
        """
        return getattr(self, 'cg_{}_{}_{}'.format(l1,l2,l3))
