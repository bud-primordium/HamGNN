utils
======

.. currentmodule:: torchscript_tools.tools.utils

工具函数模块。

概述
----

``utils.py`` 提供了数据格式转换、模型输出比较等通用工具函数。

函数参考
--------

batch_to_input_dict
^^^^^^^^^^^^^^^^^^^

.. autofunction:: batch_to_input_dict

将 PyTorch Geometric 批次数据转换为 TorchScript 兼容的字典格式。

**参数:**

* ``batch``: PyTorch Geometric Data 对象或批次
* ``device`` (可选): 目标设备，默认为 'cuda'

**返回:**

包含所有必需字段的字典，可直接用于 TorchScript 模型输入。

**示例:**

.. code-block:: python

   from utils import batch_to_input_dict
   
   # 从 DataLoader 获取批次
   for batch in dataloader:
       # 转换为标准输入格式
       input_dict = batch_to_input_dict(batch, device='cuda')
       
       # 用于推理
       output = model(input_dict)

compare_model_outputs
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compare_model_outputs

比较两个模型输出的数值差异，用于验证编译正确性。

**参数:**

* ``output1``: 第一个模型的输出字典
* ``output2``: 第二个模型的输出字典  
* ``rtol`` (可选): 相对误差容限，默认 1e-5
* ``atol`` (可选): 绝对误差容限，默认 1e-8

**返回:**

包含比较结果的字典，包括最大差异、平均差异等统计信息。

支持的数据字段
--------------

``batch_to_input_dict`` 函数处理以下字段类别：

**必需基础字段:**

* ``pos``: 原子位置
* ``cell``: 晶胞参数
* ``z``: 原子序数
* ``edge_index``: 边索引

**图结构字段:**

* ``inv_edge_idx``: 逆边索引
* ``nbr_shift``: 邻居位移
* ``cell_shift``: 晶胞位移

**哈密顿量相关:**

* ``Hon``, ``Hoff``: 哈密顿量矩阵元素
* ``Son``, ``Soff``: 重叠矩阵元素
* ``eigenvalue``, ``eigenvector``: 本征值和本征向量

**批次管理:**

* ``batch``: 批次索引
* ``node_counts``: 每个图的节点数
* ``ptr``: 批次指针

技术说明
--------

* 自动处理可选字段，仅转换存在的属性
* 支持标量、向量和矩阵类型的自动转换
* 处理批次相关的索引和指针
* 确保所有张量在正确的设备上