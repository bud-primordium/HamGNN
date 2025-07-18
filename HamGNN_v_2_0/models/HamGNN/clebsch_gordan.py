import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations

"""
存储CG系数的辅助类
"""

class TensorWrapper(nn.Module):
    """包装张量以便在ModuleDict中使用"""
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer('data', tensor)

class ClebschGordan(nn.Module):
    """一个用于加载、存储和提供 Clebsch-Gordan (CG) 系数的辅助类。

    CG系数在耦合角动量理论中至关重要，用于将两个角动量的状态分解为总角动量的状态。
    这个类从一个预先计算好的 `.npz` 文件中加载 CG 系数张量。
    
    采用"实时排列"策略：
    - 只存储规范形式的CG系数（l1 <= l2 <= l3）
    - 在forward时动态计算所需的排列
    - 完全兼容TorchScript，支持设备自动移动

    .. note::
        该类继承自 `torch.nn.Module`，这使得它可以被无缝集成到任何
        PyTorch 模型中，并且通过 `register_buffer` 注册的张量会自动
        随模型移动到正确的设备 (如 CPU 或 GPU)。

    Attributes:
        cg_storage (nn.ModuleDict): 存储规范形式CG系数的字典
        l_max (int): 最大的角动量量子数
    """
    def __init__(self):
        """初始化 ClebschGordan 类。

        此构造函数执行以下操作：
        1. 定位并加载 `clebsch_gordan_coefficients_L10.npz` 文件。
        2. 从文件中提取存储在 'cg' 键下的系数字典。
        3. 只存储规范形式的CG系数（l1 <= l2 <= l3）。
        4. 使用ModuleDict确保TorchScript兼容性和设备管理。
        """
        super(ClebschGordan, self).__init__()
        
        # 使用ModuleDict存储CG系数，这是JIT友好的
        self.cg_storage = nn.ModuleDict()
        
        # 加载预计算的CG系数
        cg_file = os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz')
        tmp = np.load(cg_file, allow_pickle=True)['cg'][()]
        
        self.l_max = 0
        
        # 只存储规范形式的CG系数（文件中已经是l1 <= l2 <= l3的形式）
        for l123, cg_array in tmp.items():
            self.l_max = max(self.l_max, max(l123))
            # 创建规范键
            key = f'{l123[0]}_{l123[1]}_{l123[2]}'
            # 使用TensorWrapper包装并存储
            self.cg_storage[key] = TensorWrapper(torch.tensor(cg_array))
        
        # 注册l_max为buffer，避免在forward中使用Python属性
        self.register_buffer('_l_max', torch.tensor(self.l_max))

    def forward(self, l1: int, l2: int, l3: int) -> torch.Tensor:
        """获取指定角动量组合的 Clebsch-Gordan 系数张量。

        动态计算从规范形式到请求形式的排列。

        Args:
            l1 (int): 第一个角动量的量子数。
            l2 (int): 第二个角动量的量子数。
            l3 (int): 耦合后的总角动量的量子数。

        Returns:
            torch.Tensor: 对应于 (l1, l2, l3) 组合的 CG 系数张量。
        """
        # 检查所有可能的排列，找到存在的规范形式
        # 这里我们需要尝试所有6种排列，因为原始代码生成了所有排列
        input_tuple = (l1, l2, l3)
        
        # 尝试所有排列找到存在的键
        for perm_indices in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
            # 计算这个排列对应的l值
            perm_l1 = input_tuple[perm_indices[0]]
            perm_l2 = input_tuple[perm_indices[1]]
            perm_l3 = input_tuple[perm_indices[2]]
            
            # 检查是否满足规范顺序（l1 <= l2 <= l3）
            if perm_l1 <= perm_l2 <= perm_l3:
                # 构造键
                key = f'{perm_l1}_{perm_l2}_{perm_l3}'
                
                # 检查键是否存在
                if key in self.cg_storage:
                    # 获取基础张量
                    base_tensor = self.cg_storage[key].data
                    
                    # 计算逆排列以恢复原始顺序
                    # 如果perm_indices将(l1,l2,l3)变为规范形式
                    # 那么逆排列将规范形式变回(l1,l2,l3)
                    inverse_perm = [0, 0, 0]
                    for i in range(3):
                        inverse_perm[perm_indices[i]] = i
                    
                    # 应用逆排列
                    return base_tensor.permute(inverse_perm)
        
        # 如果没找到，抛出错误
        raise ValueError(f"No CG coefficient for combination ({l1}, {l2}, {l3})")