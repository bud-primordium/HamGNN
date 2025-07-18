import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations
from collections import OrderedDict

"""
存储CG系数的辅助类
"""

class _TensorWrapper(nn.Module):
    """内部包装器，用于在ModuleDict中正确注册张量并兼容TorchScript。"""
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer('data', tensor)

class ClebschGordan(nn.Module):
    """一个用于加载、存储和提供 Clebsch-Gordan (CG) 系数的辅助类。

    此版本经过重构，实现了三个关键目标：
    1. **TorchScript兼容**：使用ModuleDict和"实时排列"策略，移除了不兼容的getattr调用。
    2. **向后兼容**：可以无缝加载用旧版代码保存的checkpoint。
    3. **向前兼容**：用此版本代码保存的checkpoint，也可以被旧版代码无缝加载。

    所有兼容性逻辑都被封装在此类内部，对外部代码完全透明。

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
        
        l_max = 0
        
        # 只存储规范形式的CG系数（l1 <= l2 <= l3），以节省内存
        for l123, cg_array in tmp.items():
            l_max = max(l_max, max(l123))
            # 键名不含'cg_'前缀，使其更简洁
            key = f'{l123[0]}_{l123[1]}_{l123[2]}'
            self.cg_storage[key] = _TensorWrapper(torch.tensor(cg_array, dtype=torch.get_default_dtype()))
        
        # 注册l_max为buffer，避免在forward中使用Python属性
        self.register_buffer('_l_max', torch.tensor(l_max))

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
        # 动态计算并返回所需的CG系数张量
        input_tuple = (l1, l2, l3)
        # 尝试所有排列找到存在的键
        
        for perm_indices in permutations((0, 1, 2)):
            perm_l = tuple(input_tuple[i] for i in perm_indices)
            # 检查是否满足规范顺序（l1 <= l2 <= l3）
            if perm_l[0] <= perm_l[1] <= perm_l[2]:
                key = f'{perm_l[0]}_{perm_l[1]}_{perm_l[2]}'
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

    # --- 以下是实现双向兼容性的核心魔法 ---

    def state_dict(self, *args, **kwargs):
        """
        在保存时，将内部的嵌套键名 (e.g., "cg_storage.0_1_1.data")
        伪装成旧版的平铺键名 (e.g., "cg_0_1_1")，以确保向前兼容。
        """
        original_dict = super().state_dict(*args, **kwargs)
        
        prefix = kwargs.get('prefix', '')
        new_dict = OrderedDict()
        
        for key, value in original_dict.items():
            stripped_key = key[len(prefix):]
            if stripped_key.startswith('cg_storage.'):
                # 'cg_storage.0_1_1.data' -> 'cg_0_1_1'
                new_key_part = 'cg_' + stripped_key.replace('cg_storage.', '').replace('.data', '')
                new_dict[prefix + new_key_part] = value
            else:
                # 保留其他键，如 _l_max
                new_dict[key] = value
        return new_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        在加载时，能自动识别旧版的平铺键名 (e.g., "cg_0_1_1")，
        并将其无缝转换为新版内部所需的嵌套格式，以确保向后兼容。
        """
        processed_state_dict = state_dict.copy()
        
        # 查找所有属于此模块的、且是旧格式的键
        keys_to_remap = [
            k for k in processed_state_dict 
            if k.startswith(prefix) and not k.startswith(prefix + 'cg_storage.') and 'cg_' in k
        ]
        
        for old_key in keys_to_remap:
            # old_key = '...cg_cal.cg_0_1_1'
            cg_part = old_key[len(prefix):]  # 'cg_0_1_1'
            
            if cg_part.startswith('cg_'):
                # '0_1_1'
                key_in_dict = cg_part.replace('cg_', '')
                # '...cg_cal.cg_storage.0_1_1.data'
                new_key = prefix + 'cg_storage.' + key_in_dict + '.data'
                
                # 更新字典
                processed_state_dict[new_key] = processed_state_dict.pop(old_key)
        
        # 使用处理后的state_dict调用父类的加载方法
        super()._load_from_state_dict(processed_state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

# 保持TensorWrapper类的向后兼容性
TensorWrapper = _TensorWrapper