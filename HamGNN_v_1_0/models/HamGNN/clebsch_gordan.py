import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations

"""
Helper class that stores Clebsch-Gordan coefficients
"""

class TensorWrapper(nn.Module):
    """Wrapper for tensors to use in ModuleDict"""
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer('data', tensor)

class ClebschGordan(nn.Module):
    def __init__(self):
        super(ClebschGordan, self).__init__()
        
        # Use ModuleDict for JIT compatibility
        self.cg_storage = nn.ModuleDict()
        
        # Load CG coefficients
        tmp = np.load(os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz'), allow_pickle=True)['cg'][()]
        
        self.l_max = 0
        
        # Store only canonical form (l1 <= l2 <= l3)
        for l123, cg_array in tmp.items():
            self.l_max = max(self.l_max, max(l123))
            # Create canonical key
            key = '{}_{}_{}' .format(l123[0], l123[1], l123[2])
            # Wrap and store
            self.cg_storage[key] = TensorWrapper(torch.tensor(cg_array))
        
        # Register l_max as buffer
        self.register_buffer('_l_max', torch.tensor(self.l_max))

    def forward(self, l1, l2, l3):
        # Check all possible permutations to find canonical form
        input_tuple = (l1, l2, l3)
        
        # Try all permutations to find existing key
        for perm_indices in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
            # Calculate l values for this permutation
            perm_l1 = input_tuple[perm_indices[0]]
            perm_l2 = input_tuple[perm_indices[1]]
            perm_l3 = input_tuple[perm_indices[2]]
            
            # Check if this is canonical order (l1 <= l2 <= l3)
            if perm_l1 <= perm_l2 <= perm_l3:
                # Construct key
                key = '{}_{}_{}' .format(perm_l1, perm_l2, perm_l3)
                
                # Check if key exists
                if key in self.cg_storage:
                    # Get base tensor
                    base_tensor = self.cg_storage[key].data
                    
                    # Calculate inverse permutation to restore original order
                    inverse_perm = [0, 0, 0]
                    for i in range(3):
                        inverse_perm[perm_indices[i]] = i
                    
                    # Apply inverse permutation
                    return base_tensor.permute(inverse_perm)
        
        # If not found, raise error
        raise ValueError('No CG coefficient for combination ({}, {}, {})'.format(l1, l2, l3))