=======================
TorchScript 开发者指南
=======================

.. warning::
   本文档仅适用于 HamGNN v2.0 版本。TorchScript 工具包目前不支持 v1.0 版本。

概述
====

本文档面向 HamGNN TorchScript 工具包的开发者，详细介绍技术实现细节、代码架构、开发改造过程和未来技术路线图。

技术架构
========

核心组件
--------

TorchScript 工具包包含以下核心组件：

1. **模型编译器** (``compile.py``)
   
   * 加载 PyTorch Lightning checkpoint
   * 构建 HamGNN 模型组件
   * 封装为推理模型并编译为 TorchScript

2. **推理模型封装** (``inference_model.py``)
   
   * 将训练时分离的 ``representation`` 和 ``output_module`` 组合
   * 提供 TorchScript 兼容的前向传播接口
   * 处理设备迁移和模块管理

3. **数据转换工具** (``utils.py``)
   
   * 统一的数据格式转换函数
   * 模型输出比较工具
   * 支持复杂图数据结构的处理

编译策略
--------

HamGNN 采用以下 TorchScript 编译策略：

1. **混合编译模式**
   
   使用 ``e3nn.util.jit.script()`` 而非标准 ``torch.jit.script()``：
   
   .. code-block:: python
   
      from e3nn.util.jit import script
      
      with torch.jit.optimized_execution(True):
          scripted_model = script(inference_model)

2. **兼容性修复**
   
   * 字典属性访问：``data.attribute`` → ``data['attribute']``
   * 动态类型检查：``hasattr()`` → ``try-except`` 模式
   * 设备管理优化：避免生成器相关的设备绑定问题

3. **数据格式标准化**
   
   所有输入数据通过 ``batch_to_input_dict()`` 转换为标准字典格式，确保 TorchScript 类型系统的兼容性。

开发改造过程
============

主要历史改造
--------------

从 commit `164c2d2 <https://github.com/bud-primordium/HamGNN/commit/164c2d2d2c40bbea37812d4fda96359f3d313abf>`_ 开始，我们进行了系统性的 TorchScript 兼容性改造（`完整提交历史 <https://github.com/bud-primordium/HamGNN/commits/torchscript_export/>`_）：

1. **字典访问模式改造** (`164c2d2 <https://github.com/bud-primordium/HamGNN/commit/164c2d2d2c40bbea37812d4fda96359f3d313abf>`_)
   
   * 将属性访问 (``data.attribute``) 替换为字典访问 (``data['attribute']``)
   * 解决 PyTorch Geometric 数据对象与 TorchScript 的兼容性问题

2. **hasattr() 替换为 try-except** (`9b69a9d <https://github.com/bud-primordium/HamGNN/commit/9b69a9d>`_)
   
   * TorchScript 不支持 ``hasattr()`` 函数
   * 系统性地替换为 try-except 模式

3. **ClebschGordan 惰性排列策略** (`dd2a830 <https://github.com/bud-primordium/HamGNN/commit/dd2a830>`_)
   
   * 实现惰性加载机制以支持 TorchScript
   * 添加向后兼容性处理 (`829cfa5 <https://github.com/bud-primordium/HamGNN/commit/829cfa5>`_, `95dded3 <https://github.com/bud-primordium/HamGNN/commit/95dded3>`_)

4. **BaseModel 架构重设计与 DynamicGraphTransform 拆分** (`bb3ffb2 <https://github.com/bud-primordium/HamGNN/commit/bb3ffb2>`_)
   
   * 将动态图构建逻辑从 BaseModel 拆分为独立的 DynamicGraphTransform 类
   * 实现数据预处理与模型计算的分离
   * 完全移除 ASE 依赖，支持 TorchScript 编译
   * 修复 DataLoader 兼容性问题

5. **HamGNNPlusPlusOut 模块拆分** (`58e175a <https://github.com/bud-primordium/HamGNN/commit/58e175a>`_)
   
   * 将复杂模块拆分为 wrapper 和 core 部分
   * 优化网络前向传播逻辑 (`616bbd4 <https://github.com/bud-primordium/HamGNN/commit/616bbd4>`_)

6. **其他重要修复**
   
   * 添加 @compile_mode 装饰器 (`96e169c <https://github.com/bud-primordium/HamGNN/commit/96e169c>`_)
   * 解决条件属性初始化问题 (`617bc13 <https://github.com/bud-primordium/HamGNN/commit/617bc13>`_)
   * 确保 tensor 操作的整数参数 (`2c18358 <https://github.com/bud-primordium/HamGNN/commit/2c18358>`_)
   * 解决变量作用域兼容性问题 (`27b0aec <https://github.com/bud-primordium/HamGNN/commit/27b0aec>`_, `6eba7c8 <https://github.com/bud-primordium/HamGNN/commit/6eba7c8>`_)

技术细节
========

核心算法实现
------------

1. **模型重构逻辑**

   .. code-block:: python
   
      class HamGNNInference(nn.Module):
          def __init__(self, representation, output_module):
              super().__init__()
              self.representation = representation
              self.output_module = output_module
              
          def forward(self, data):
              # 计算原子表示
              representation = self.representation(data)
              # 基于表示计算输出
              predictions = self.output_module(data, representation)
              return predictions

2. **数据转换策略**

   ``batch_to_input_dict()`` 函数处理以下字段类别：
   
   * 必需基础字段：``pos``, ``cell``, ``z``, ``edge_index``
   * 图结构字段：``inv_edge_idx``, ``nbr_shift``, ``cell_shift``
   * 哈密顿量相关：``Hon``, ``Hoff``, ``Son``, ``Soff`` 等
   * 批次管理：``batch``, ``node_counts``, ``ptr``

3. **设备管理优化**

   * 避免在模型定义中绑定设备信息
   * 使用模型加载时的设备设置
   * 重写 ``to()`` 方法确保子模块正确迁移

历史问题与教训
--------------

1. **配置文件依赖**

   **问题**: 早期 checkpoint 不包含超参数信息
   
   **原因**: ``Model.py`` 中 ``self.save_hyperparameters()`` 被注释
   
   **影响**: 编译时需要额外提供配置文件
   
   **未来改进**: 启用超参数保存，实现直接从 checkpoint 编译

技术路线图
==========

我们规划了三个阶段的技术演进路径，从当前的 TorchScript 实现逐步向更先进的编译技术迁移。

阶段 1：从 TorchScript Script 转向 Trace 模式
---------------------------------------------

**目标**: 减少编译开销，提高兼容性

**技术方案**:

* **Trace 模式优势**:
  
  * 更好的动态张量形状支持
  * 减少 JIT 编译时的类型推断开销
  * 对复杂控制流的更好处理

* **实现策略**:
  
  .. code-block:: python
  
     # 当前：Script 模式
     scripted_model = torch.jit.script(model)
     
     # 目标：Trace 模式
     example_input = prepare_example_input(config)
     traced_model = torch.jit.trace(model, example_input)

* **挑战与解决**:
  
  * **动态图结构**: HamGNN 的原子数量和邻居关系在不同结构中变化
  * **解决方案**: 设计代表性的示例输入，覆盖常见的图大小范围
  * **验证机制**: 确保 trace 模式下的输出一致性

阶段 2：引入 torch.fx 模型重构，进而集成 PyTorch 2.0 TorchInductor
---------------------------------------------------------------

**目标**: 利用图级别优化，集成新一代编译器

**技术方案**:

* **torch.fx 图变换**:
  
  * 在计算图级别分析模型结构
  * 识别瓶颈操作和优化机会
  * 自动重排和融合操作

* **TorchInductor 集成**:
  
  .. code-block:: python
  
     import torch._dynamo as dynamo
     from torch.fx import symbolic_trace
     
     # 符号化追踪
     traced_graph = symbolic_trace(model)
     
     # 图级别优化
     optimized_graph = optimize_graph(traced_graph)
     
     # TorchInductor 编译
     compiled_model = torch.compile(optimized_graph, backend="inductor")

* **动态张量形状适应**:
  
  TorchInductor 的关键优势是对动态张量形状的原生支持，这对 HamGNN 至关重要：
  
  * **原子数量变化**: 不同分子/晶体结构的原子数量差异很大
  * **邻居对数量**: 根据截断半径和晶体结构，邻居原子对数量动态变化
  * **批次处理**: 同一批次内不同图的大小可能不同

* **编译器后端选择**:
  
  * **GPU**: 利用 Triton 生成优化的 CUDA 内核
  * **CPU**: 生成 C++ 和 OpenMP 代码
  * **硬件无关**: 自动适应不同的硬件特性

阶段 3：利用 Triton 自定义内核优化 e3nn.o3.TensorProduct 的张量积操作
------------------------------------------------------------------

**目标**: 为最昂贵的计算操作开发专用内核

**技术背景**:

HamGNN 的计算瓶颈主要在等变张量积操作：

* ``e3nn.o3.TensorProduct``: 球谐函数之间的张量积
* 不可约表示的变换和组合
* 高维张量的复杂索引和求和

**Triton 内核开发**:

.. code-block:: python

   import triton
   import triton.language as tl
   
   @triton.jit
   def tensor_product_kernel(
       input_ptr, weight_ptr, output_ptr,
       irreps_in, irreps_out,
       BLOCK_SIZE: tl.constexpr
   ):
       """
       专用的张量积内核，针对 HamGNN 的数据模式优化
       """
       # 高效的内存访问模式
       # 向量化的数学运算
       # 针对球谐函数的特化计算

* **内核特性**:
  
  * **内存优化**: 针对 HamGNN 的数据访问模式设计
  * **向量化计算**: 利用现代 GPU 的并行特性
  * **数学优化**: 针对球谐函数和 Clebsch-Gordan 系数的特化
  * **动态形状**: 支持变长的不可约表示

* **集成策略**:
  
  .. code-block:: python
  
     # 自动替换 e3nn TensorProduct
     def optimize_tensor_products(graph_module):
         for node in graph_module.graph.nodes:
             if is_tensor_product_op(node):
                 replace_with_triton_kernel(node)
         return graph_module

实施计划
--------

.. list-table:: 技术路线图
   :header-rows: 1

   * - 阶段
     - 主要任务
     - 技术挑战
   * - 阶段 1
     - Trace 模式编译器
     - 动态图结构处理
   * - 阶段 2
     - torch.fx + TorchInductor
     - 动态张量形状适配
   * - 阶段 3
     - Triton 自定义内核
     - 张量积操作优化

补充方案：hamgnn-package
------------------------

除了上述技术路线图，我们还在考虑一个补充的模型分发方案：

**hamgnn-package**: 模型打包与分发工具

当前的 hamgnn-compile 主要解决从 checkpoint 部署到生产环境的问题，但编译后的 .hamgnn.pt 文件仍然依赖于：

* CUDA、e3nn、NumPy 等核心外部库的版本
* 没有设计与其他软件（如 ASE 计算器）的接口
* 对 HamGNN 版本有一定依赖

未来的 hamgnn-package 将提供更完整的模型分发方案：

* 在训练时或从 checkpoint/.hamgnn.pt 导出完整模型包
* 封装模型权重、运行所需的 Python 源代码快照、元数据
* 真正实现模型的跨平台、跨版本分发

这样形成两个互补的工具：

* **hamgnn-compile**: 负责生产环境的高性能部署推理
* **hamgnn-package**: 负责模型的打包分发

注意：ASE 等软件的集成接口需要额外开发，生产环境的集成仍然基于 .hamgnn.pt 文件。

代码维护
========

开发规范
--------

* 遵循 Google Python 风格指南
* 使用完整的 docstring 文档
* 类型注解覆盖所有公共接口

调试工具
--------

1. **性能分析**
   
   .. code-block:: python
   
      # 使用 PyTorch Profiler 分析性能
      with torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA],
          record_shapes=True,
          with_stack=True
      ) as prof:
          output = model(data)
      
      # 导出详细报告
      prof.export_chrome_trace("trace.json")

2. **模型比较工具**
   
   使用 ``compare_model_outputs()`` 函数进行精确的数值验证。

3. **元数据检查**
   
   编译过程会生成详细的元数据，包含环境信息和编译配置。

参考资源
========

* **高性能深度等变原子间势训练与推理**
  
  Tan, C. W., Descoteaux, M. L., Kotak, M., Nascimento, G. D. M., Kavanagh, S. R., Zichi, L., Wang, M., Saluja, A., Hu, Y. R., Smidt, T., Johansson, A., Witt, W. C., Kozinsky, B., & Musaelian, A. (2025). High-performance training and inference for deep equivariant interatomic potentials. arXiv preprint arXiv:2504.16068. `链接 <https://arxiv.org/abs/2504.16068>`_

技术文档
--------

* `PyTorch 2.0 编译器栈 <https://pytorch.org/docs/stable/torch.compiler.html>`_
* `Triton 开发指南 <https://triton-lang.org/>`_
* `torch.fx 变换文档 <https://pytorch.org/docs/stable/fx.html>`_
* `E3NN TorchScript 集成 <https://docs.e3nn.org/>`_