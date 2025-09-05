hamgnn-compile
==============

.. currentmodule:: torchscript_tools.tools.compile

HamGNN TorchScript 编译工具。

概述
----

``compile.py`` 是将 HamGNN PyTorch Lightning checkpoint 转换为 TorchScript 格式的命令行工具。

使用方法
--------

.. code-block:: bash

   python compile.py [OPTIONS]

命令行参数
----------

.. list-table::
   :header-rows: 1

   * - 参数
     - 类型
     - 必需
     - 说明
   * - ``--config``
     - str
     - 是
     - HamGNN 配置文件路径
   * - ``--checkpoint``
     - str
     - 是
     - PyTorch Lightning checkpoint 文件路径
   * - ``--output``
     - str
     - 是
     - 输出的 TorchScript 模型路径（建议使用 .hamgnn.pt 后缀）
   * - ``--device``
     - str
     - 否
     - 目标设备 (cuda/cpu)，默认: cuda

示例
----

基本用法::

    python compile.py \
        --config path/to/config.yaml \
        --checkpoint path/to/model.ckpt \
        --output compiled_model.hamgnn.pt \
        --device cuda

CPU 编译::

    python compile.py \
        --config config.yaml \
        --checkpoint model.ckpt \
        --output model_cpu.hamgnn.pt \
        --device cpu

输出文件
--------

编译过程会生成两个文件：

1. **模型文件** (``.hamgnn.pt``): TorchScript 编译后的模型
2. **元数据文件** (``.metadata.json``): 包含配置信息、编译环境等元数据

函数参考
--------

.. autofunction:: load_config

.. autofunction:: load_checkpoint

.. autofunction:: compile_model

.. autofunction:: save_compiled_model

技术细节
--------

* 使用 ``e3nn.util.jit.script()`` 进行混合编译，优化等变操作
* 自动处理设备迁移和 CUDA 可用性检查
* 保存详细的编译元数据用于部署验证

注意事项
--------

* 编译环境与部署环境的 CUDA、e3nn、NumPy 版本需要保持一致
* 当前需要配置文件辅助编译，未来版本将支持直接从 checkpoint 读取配置
* 首次编译可能需要较长时间，请耐心等待