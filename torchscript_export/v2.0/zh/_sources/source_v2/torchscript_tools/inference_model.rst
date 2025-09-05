inference_model
===============

.. currentmodule:: torchscript_tools.tools.inference_model

推理模型封装模块。

概述
----

``inference_model.py`` 提供了 ``HamGNNInference`` 类，将训练时分离的模型组件封装为统一的推理接口。

类参考
------

HamGNNInference
^^^^^^^^^^^^^^^

.. autoclass:: HamGNNInference
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: to

使用示例
--------

创建推理模型：

.. code-block:: python

    from inference_model import HamGNNInference
    
    # 从训练模型组件创建
    inference_model = HamGNNInference(
        representation=representation_module,
        output_module=output_module
    )
    
    # 编译为 TorchScript
    scripted_model = torch.jit.script(inference_model)

推理使用：

.. code-block:: python

    # 准备输入数据
    data = {
        'pos': positions,
        'z': atomic_numbers,
        'edge_index': edge_indices,
        # ... 其他必需字段
    }
    
    # 执行推理
    with torch.no_grad():
        output = scripted_model(data)
        hamiltonian = output['hamiltonian']

设计理念
--------

1. **模块分离**: 将训练相关逻辑从推理模型中剥离
2. **接口统一**: 提供标准化的字典输入输出接口
3. **TorchScript 友好**: 避免动态特性，确保编译兼容性
4. **设备无关**: 支持灵活的设备迁移

技术特点
--------

* 仅包含推理必需的 ``representation`` 和 ``output_module``
* 重写 ``to()`` 方法确保子模块正确迁移设备
* 使用字典格式输入，兼容 TorchScript 类型系统
* 简化的前向传播逻辑，优化推理性能