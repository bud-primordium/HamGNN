import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations
from collections import OrderedDict

"""
Helper class that stores Clebsch-Gordan coefficients
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
    """
    def __init__(self):
        super(ClebschGordan, self).__init__()
        
        # Use ModuleDict for JIT compatibility
        self.cg_storage = nn.ModuleDict()
        
        # Load CG coefficients
        tmp = np.load(os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz'), allow_pickle=True)['cg'][()]
        
        l_max = 0
        
        # Store only canonical form (l1 <= l2 <= l3)
        for l123, cg_array in tmp.items():
            l_max = max(l_max, max(l123))
            # Create canonical key without cg_ prefix
            key = '{}_{}_{}' .format(l123[0], l123[1], l123[2])
            # Wrap and store
            self.cg_storage[key] = _TensorWrapper(torch.tensor(cg_array, dtype=torch.get_default_dtype()))
        
        # Register l_max as buffer
        self.register_buffer('_l_max', torch.tensor(l_max))

    def forward(self, l1, l2, l3):
        """动态计算并返回所需的CG系数张量。"""
        # Check all possible permutations to find canonical form
        input_tuple = (l1, l2, l3)
        
        for perm_indices in permutations((0, 1, 2)):
            perm_l = tuple(input_tuple[i] for i in perm_indices)
            
            if perm_l[0] <= perm_l[1] <= perm_l[2]:
                key = '{}_{}_{}' .format(perm_l[0], perm_l[1], perm_l[2])
                if key in self.cg_storage:
                    base_tensor = self.cg_storage[key].data
                    inverse_perm = [0, 0, 0]
                    for i in range(3):
                        inverse_perm[perm_indices[i]] = i
                    return base_tensor.permute(inverse_perm)
        
        # If not found, raise error
        raise ValueError('No CG coefficient for combination ({}, {}, {})'.format(l1, l2, l3))

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