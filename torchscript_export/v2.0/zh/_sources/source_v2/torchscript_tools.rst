TorchScript 工具包
==================

HamGNN v2.0 的 TorchScript 编译和部署工具包。

.. warning::
   TorchScript 工具包目前仅支持 HamGNN v2.0 版本。

概述
----

本工具包提供了将 HamGNN v2.0 模型编译为 TorchScript 格式的完整解决方案，
包括模型编译、推理优化和部署支持。

文档
----

.. toctree::
   :maxdepth: 1
   
   torchscript_user_guide
   torchscript_developer_guide

API 参考
--------

.. toctree::
   :maxdepth: 1
   
   torchscript_tools/hamgnn_compile
   torchscript_tools/inference_model
   torchscript_tools/utils

主要功能
--------

* **模型编译**: 将 PyTorch Lightning checkpoint 转换为优化的 TorchScript 格式
* **推理封装**: 提供专门的推理模型架构，优化部署性能
* **数据处理**: 统一的数据格式转换和批处理支持
* **性能测试**: 完整的性能评估和对比工具

快速开始
--------

.. code-block:: bash

   # 编译模型
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

   # 使用编译模型
   import torch
   model = torch.jit.load('compiled_model.hamgnn.pt')

更多详细信息请参阅 :doc:`torchscript_user_guide`。