=======================
TorchScript 用户指南
=======================

.. warning::
   本文档仅适用于 HamGNN v2.0 版本。TorchScript 工具包目前不支持 v1.0 版本。

概述
====

HamGNN TorchScript 工具包为 HamGNN v2.0 模型提供完整的 TorchScript 编译和部署支持。通过将训练好的 PyTorch Lightning checkpoint 转换为优化的 TorchScript 格式，可以实现更高效的模型推理和生产部署。

主要功能
========

* **模型编译**: 将 HamGNN checkpoint (.ckpt) 转换为 TorchScript (.hamgnn.pt) 格式
* **一致性验证**: 确保编译前后模型输出的数值一致性
* **性能测试**: 比较编译前后的推理速度
* **部署优化**: 生成可移植的模型文件用于生产环境

快速开始
========

1. 编译模型
-----------

使用 ``hamgnn-compile`` 命令行工具将训练好的模型转换为 TorchScript 格式：

.. code-block:: bash

   # 推荐使用命令行工具
   hamgnn-compile \
     --config path/to/config.yaml \
     --checkpoint path/to/checkpoint.ckpt \
     --output compiled_model.hamgnn.pt \
     --device cuda

   # 或直接运行脚本
   python torchscript_tools/tools/compile.py \
     --config path/to/config.yaml \
     --checkpoint path/to/checkpoint.ckpt \
     --output compiled_model.hamgnn.pt \
     --device cuda

参数说明：

* ``--config``: 训练时使用的配置文件路径
* ``--checkpoint``: 训练好的 PyTorch Lightning checkpoint 文件
* ``--output``: 输出的 TorchScript 模型路径（建议使用 .hamgnn.pt 扩展名）
* ``--device``: 编译目标设备（cuda 或 cpu）

2. 使用编译模型
---------------

在 Python 代码中加载和使用编译后的模型：

.. code-block:: python

   import torch
   from torchscript_tools.tools.utils import batch_to_input_dict

   # 加载编译模型
   model = torch.jit.load('compiled_model.hamgnn.pt')
   model.eval()

   # 准备输入数据（从 DataLoader 获取）
   for batch in test_loader:
       # 转换数据格式
       data = batch_to_input_dict(batch)
       
       # 推理
       with torch.no_grad():
           output = model(data)
           hamiltonian = output['hamiltonian']

性能测试
========

使用 ``performance_test.py`` 进行全面的性能评估：

.. code-block:: bash

   python torchscript_tools/examples/performance_test.py

该脚本会自动进行：

* TorchScript 模型与原始模型的性能对比
* 多种 batch_size 下的性能分析
* 详细的性能统计和建议

**注意**: 脚本中的 ``torch.compile`` 测试目前会失败，因为我们尚未完成 torch.fx 改造。这是技术路线图第二阶段的内容。

注意事项：使用前需要修改脚本中的文件路径，将示例路径替换为你的实际文件路径。

部署最佳实践
============

1. 模型文件管理
---------------

* **命名规范**: 使用 ``.hamgnn.pt`` 扩展名区分 TorchScript 模型
* **元数据保存**: 编译过程会自动生成 ``.metadata.json`` 文件，包含配置和环境信息
* **版本控制**: 建议将配置文件纳入版本控制，模型文件使用 Git LFS 管理

2. 性能优化建议
---------------

* **批处理大小**: 根据硬件配置选择合适的 batch_size，通常较大的批次有更好的性能
* **设备选择**: GPU 加速效果通常比 CPU 更明显
* **预热运行**: 首次推理可能较慢，建议进行预热运行排除 JIT 编译开销
* **内存管理**: 大规模推理时注意 GPU 内存使用情况

**重要提示**: 部署环境需要与编译环境的 CUDA、e3nn、NumPy 版本保持一致。

故障排除
========

获取帮助
--------

* 查看详细错误日志和堆栈信息
* 检查 ``metadata.json`` 文件中的环境信息
* 参考开发者文档中的技术细节
* 在项目 Issues 中提交问题报告

技术参考
========

相关文档
--------

* :doc:`torchscript_developer_guide` - 开发者技术指南
* `PyTorch TorchScript 官方文档 <https://pytorch.org/docs/stable/jit.html>`_
* `E3NN TorchScript 支持 <https://docs.e3nn.org/en/stable/api/util/jit.html>`_

模型格式
--------

编译后的模型包含以下组件：

* **主模型文件** (``.hamgnn.pt``): TorchScript 格式的可执行模型
* **元数据文件** (``.metadata.json``): 包含配置、编译信息和环境详情
* **输入格式**: 标准化的字典格式，包含所有必需的图数据字段
* **输出格式**: 包含哈密顿量等预测结果的字典

版本兼容性
----------

.. list-table::
   :header-rows: 1

   * - 组件
     - 最低版本
     - 推荐版本
     - 说明
   * - PyTorch
     - 1.8.0
     - 2.0.0+
     - 目前仅支持 TorchScript script 模式
   * - torch-geometric
     - 2.0.0
     - 最新稳定版
     - 图数据处理
   * - e3nn
     - 0.5.0
     - 0.5.6+
     - 等变神经网络
   * - Python
     - 3.8
     - 3.9+
     - 运行环境