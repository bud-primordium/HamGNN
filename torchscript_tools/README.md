# HamGNN TorchScript 工具包

> **快速参考**: 本文档提供工具包的快速入门。详细使用说明和技术细节请参阅 Sphinx 文档。

## 概述

HamGNN TorchScript 工具包为 HamGNN v2.0 提供 TorchScript 编译支持，将 PyTorch Lightning checkpoint 转换为优化的 TorchScript 格式。

## 主要功能

- **模型编译**: 将 checkpoint (.ckpt) 转换为 TorchScript (.hamgnn.pt)
- **性能测试**: 比较编译前后的推理速度
- **部署优化**: 生成可移植的模型文件

## 目录结构

```
torchscript_tools/
├── README.md                    # 快速参考
├── __init__.py                  # 包定义
├── tools/                       # 核心工具
│   ├── __init__.py              
│   ├── compile.py               # 模型编译脚本
│   ├── inference_model.py       # 推理模型封装
│   └── utils.py                 # 通用工具函数
├── examples/                    # 使用示例
│   ├── __init__.py              
│   ├── performance_test.py      # 性能测试脚本
│   ├── compile_hamgnn.slurm     # 编译作业示例
│   └── performance_test.slurm   # 性能测试作业示例
├── compiled_models/             # 编译产物（已忽略）
└── tests/                       # 测试文件（已忽略）
```

## 快速开始

### 1. 编译模型

```bash
# 使用命令行工具（推荐）
hamgnn-compile \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.ckpt \
  --output compiled_model.hamgnn.pt \
  --device cuda

# 或直接运行脚本
python tools/compile.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.ckpt \
  --output compiled_model.hamgnn.pt \
  --device cuda
```

### 2. 性能测试

```bash
python examples/performance_test.py
```

**注意**: 使用前需要修改 `performance_test.py` 中的文件路径。

## 使用编译模型

```python
import torch
from tools.utils import batch_to_input_dict

# 加载编译模型
model = torch.jit.load('compiled_model.hamgnn.pt')
model.eval()

# 准备输入数据
for batch in test_loader:
    data = batch_to_input_dict(batch)
    
    # 推理
    with torch.no_grad():
        output = model(data)
        hamiltonian = output['hamiltonian']
```

## 技术特点

- **兼容性**: 解决了 HamGNN v2.0 与 TorchScript 的兼容性问题
- **高性能**: 使用 e3nn 的混合编译策略优化推理速度
- **易部署**: 生成可移植的 .hamgnn.pt 模型文件

**注意**: 部署环境需要与编译环境的 CUDA、e3nn、NumPy 版本保持一致

## 详细文档

完整文档已集成到 HamGNN v2.0 Sphinx 文档系统：

- **用户指南**: `docs/source_v2/torchscript_user_guide.rst`
- **开发者指南**: `docs/source_v2/torchscript_developer_guide.rst`
- **API 参考**: `docs/source_v2/torchscript_api.rst`

构建文档：`cd docs && make html`

## 技术路线图

我们规划了三个阶段的技术演进：

1. **阶段 1**: 从 TorchScript script 转向 trace 模式
2. **阶段 2**: 引入 torch.fx 模型重构，集成 PyTorch 2.0 TorchInductor  
3. **阶段 3**: 利用 Triton 自定义内核优化 e3nn.o3.TensorProduct 的张量积操作

详见开发者指南中的完整技术路线图。

## 问题反馈

如遇到问题，请提供：

- 详细的错误信息和堆栈跟踪
- 环境信息（从 metadata.json 获取）
- 最小化的复现案例

## 贡献

欢迎提交 Issue 和 Pull Request！请确保：

- 遵循现有的代码风格
- 添加适当的测试
- 更新相关文档