#!/usr/bin/env python
"""
HamGNN TorchScript 编译工具

将 HamGNN PyTorch Lightning checkpoint 转换为 TorchScript 格式。

示例用法::

    # 基本用法（推荐）
    hamgnn-compile \\
        --config path/to/config.yaml \\
        --checkpoint path/to/model.ckpt \\
        --output compiled_model.hamgnn.pt \\
        --device cuda

    # 或直接运行脚本
    python compile.py \\
        --config path/to/config.yaml \\
        --checkpoint path/to/model.ckpt \\
        --output compiled_model.hamgnn.pt \\
        --device cuda

    # 使用 CPU 编译
    python compile.py \\
        --config config.yaml \\
        --checkpoint model.ckpt \\
        --output model_cpu.hamgnn.pt \\
        --device cpu

命令行参数:
    --config: HamGNN 配置文件路径 (必需)
    --checkpoint: PyTorch Lightning checkpoint 路径 (必需)
    --output: 输出的 TorchScript 模型路径，建议使用 .hamgnn.pt 后缀 (必需)
    --device: 目标设备，可选 'cuda' 或 'cpu' (默认: cuda)

注意:
    - 编译后的模型文件会同时生成一个 .metadata.json 文件记录编译信息
    - 部署环境需要与编译环境的 CUDA、e3nn、NumPy 版本保持一致
    - 当前需要配置文件辅助编译，未来可能直接从 checkpoint 读取配置
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any
import platform
from importlib.metadata import version
import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """加载并解析配置文件"""
    # 延迟导入以提高启动速度
    from HamGNN_v_2_0.input.config_parsing import read_config
    # 使用 HamGNN 的配置解析器
    config = read_config(config_path)
    return config


def load_checkpoint(checkpoint_path: str, config: Dict[str, Any], device: str) -> tuple:
    """
    加载 checkpoint 并构建模型
    
    Returns:
        (representation, output_module): 模型的两个主要组件
        
    .. note::
        HamGNN 当前的 checkpoint 不包含配置信息，因为 Model.py 中的
        ``self.save_hyperparameters()`` 被注释掉了。未来应该启用该功能，
        这样就可以像 nequip 一样直接从 checkpoint 编译，而不需要额外的配置文件。
        参见: `HamGNN_v_2_0/models/Model.py:100 <https://github.com/bud-primordium/HamGNN/blob/313af4fe09d36b55ec17d9fa7eea143c565465b1/HamGNN_v_2_0/models/Model.py#L100>`_
    """
    # 延迟导入HamGNN特定的重量级模块
    from HamGNN_v_2_0.main import build_model
    from HamGNN_v_2_0.models.Model import Model
    
    print(f"加载 checkpoint: {checkpoint_path}")
    
    # 模拟 main.py 中的配置处理
    # 确保表示网络和输出网络在哈密顿量类型上达成一致
    if hasattr(config, 'representation_nets') and hasattr(config.representation_nets, 'HamGNN_pre'):
        if hasattr(config, 'output_nets') and hasattr(config.output_nets, 'HamGNN_out'):
            config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    
    # 构建模型组件
    representation, output_module, _ = build_model(config)
    
    # 设置精度（模拟 main.py 的行为）
    if config.get('setup', {}).get('precision', 32) == 32:
        dtype = torch.float32
    else:
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    representation.to(dtype)
    output_module.to(dtype)
    
    # 获取损失和指标配置
    losses = config.get('losses_metrics', {}).get('losses', [])
    metrics = config.get('losses_metrics', {}).get('metrics', [])
    
    # 加载完整的 Lightning 模型来获取权重
    # 模拟 main.py 中 Model 的初始化参数
    model = Model(
        representation=representation,
        output=output_module,
        losses=losses,
        validation_metrics=metrics,
        lr=config.get('optim_params', {}).get('lr', 1e-3),
        lr_decay=config.get('optim_params', {}).get('lr_decay', 0.1),
        lr_patience=config.get('optim_params', {}).get('lr_patience', 100),
    )
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从 checkpoint 提取模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重到 Lightning 模型
    model.load_state_dict(state_dict, strict=False)
    
    # 返回内部模块（已加载权重）
    representation = model.representation
    output_module = model.output_module
    
    # 设置为评估模式
    representation.eval()
    output_module.eval()
    
    # 移动到目标设备
    representation = representation.to(device)
    output_module = output_module.to(device)
    
    return representation, output_module


def compile_model(representation, output_module, device: str):
    """
    编译模型为 TorchScript
    
    使用 e3nn.util.jit.script 进行混合编译
    """
    # 延迟导入重量级的e3nn和推理模块
    from e3nn.util.jit import script
    sys.path.insert(0, str(Path(__file__).parent))
    from inference_model import HamGNNInference
    
    print("创建推理模型...")
    inference_model = HamGNNInference(representation, output_module)
    inference_model = inference_model.to(device)
    inference_model.eval()
    
    print("开始 TorchScript 编译...")
    print("使用 e3nn 混合编译策略...")
    
    # 使用 e3nn 的 script 函数进行编译
    # 这会自动处理 @compile_mode 装饰器
    with torch.jit.optimized_execution(True):
        scripted_model = script(inference_model)
    
    print("编译成功！")
    return scripted_model


def save_compiled_model(scripted_model: torch.jit.ScriptModule,
                       output_path: str,
                       config: Dict[str, Any],
                       checkpoint_path: str):
    """保存编译后的模型和元数据"""
    
    # 准备元数据
    metadata = {
        # 保存完整的配置（转换为可序列化的字典格式）
        'config': config if isinstance(config, dict) else dict(config),
        'source_checkpoint': checkpoint_path,
        'hamgnn_version': 'v2.0',
        
        # 编译信息
        'compilation_info': {
            'method': 'e3nn.util.jit.script',
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'compile_timestamp': str(datetime.datetime.now()),
        },
        
        # 环境信息
        'environment': {
            'e3nn_version': version('e3nn') if 'e3nn' in sys.modules else 'unknown',
            'torch_geometric_version': version('torch_geometric') if 'torch_geometric' in sys.modules else 'unknown',
            'numpy_version': version('numpy') if 'numpy' in sys.modules else 'unknown',
        }
    }
    
    # 保存模型
    print(f"保存编译模型到: {output_path}")
    torch.jit.save(scripted_model, output_path)
    
    # 保存元数据
    metadata_path = output_path.replace('.hamgnn.pt', '.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)  # default=str 处理不可序列化的对象
    print(f"保存元数据到: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='HamGNN TorchScript 编译工具')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch Lightning checkpoint 路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出的 TorchScript 模型路径 (.hamgnn.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='目标设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查文件存在性
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"错误：checkpoint 文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 确保输出文件名正确
    if not args.output.endswith('.hamgnn.pt'):
        args.output = args.output.replace('.pt', '.hamgnn.pt')
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载配置
        print("\n=== 步骤 1/4: 加载配置 ===")
        config = load_config(args.config)
        
        # 2. 加载模型
        print("\n=== 步骤 2/4: 加载模型 ===")
        representation, output_module = load_checkpoint(
            args.checkpoint, config, args.device
        )
        
        # 3. 编译模型
        print("\n=== 步骤 3/4: 编译模型 ===")
        scripted_model = compile_model(
            representation, output_module, args.device
        )
        
        # 4. 保存模型
        print("\n=== 步骤 4/4: 保存模型 ===")
        save_compiled_model(
            scripted_model, args.output, config, args.checkpoint
        )
        
        print("\n编译完成！")
        print(f"编译后的模型保存在: {args.output}")
        
    except Exception as e:
        print(f"\n编译失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()