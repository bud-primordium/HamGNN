#!/usr/bin/env python
"""
HamGNN TorchScript 模型使用示例

演示如何加载和使用编译后的 HamGNN 模型进行推理
"""

import torch
import sys
import time
import copy
import os
from pathlib import Path

from HamGNN_v_2_0.GraphData.graph_data import graph_data_module
from HamGNN_v_2_0.input.config_parsing import read_config
from HamGNN_v_2_0.models.HamGNN.BaseModel import DynamicGraphTransform

# 更优雅的导入方式：使用相对导入
# 如果作为脚本运行，需要先将 torchscript_tools 添加到 PYTHONPATH
# 或者从 HamGNN 根目录运行：python -m torchscript_tools.examples.example_usage
try:
    # 尝试使用包内相对导入（推荐）
    from ..tools.utils import batch_to_input_dict
except ImportError:
    # 作为独立脚本运行时的回退方案
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from tools.utils import batch_to_input_dict

def prepare_data_with_batch_size(config, batch_size):
    """创建指定batch_size的数据加载器"""
    # 复制配置避免修改原始配置
    config_copy = copy.deepcopy(config)
    config_copy.dataset_params.batch_size = batch_size
    
    from HamGNN_v_2_0.main import prepare_data
    data_module = prepare_data(config_copy)
    data_module.setup(stage='test')
    return data_module.test_dataloader()


def run_performance_test(config, batch_size):
    """性能测试：比较TorchScript模型与原始checkpoint在指定batch_size下的表现"""

    import os  # 确保可以访问os模块
    
    print(f"\n🔍 HamGNN TorchScript模型性能测试 (batch_size={batch_size})")
    print("=" * 70)
    
    # 从环境变量读取路径（SLURM脚本设置的），如果没有设置则使用默认值
    config_path = os.getenv('CONFIG_PATH', "path/to/your/config.yaml")
    checkpoint_path = os.getenv('CHECKPOINT_PATH', "path/to/your/checkpoint.ckpt")
    compiled_model_path = os.getenv('COMPILED_MODEL_PATH', "path/to/your/compiled_model.hamgnn.pt")
    
    # 检查文件存在性
    for path_name, path in [("配置文件", config_path), 
                           ("Checkpoint", checkpoint_path), 
                           ("编译模型", compiled_model_path)]:
        if not Path(path).exists():
            print(f"❌ {path_name}不存在: {path}")
            return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()
    
    try:
        # 加载配置
        config = read_config(config_path)
        
        # 1. 加载原始模型 - 使用HamGNN标准方式
        print("=== 加载原始checkpoint模型 ===")
        from HamGNN_v_2_0.main import build_model
        from HamGNN_v_2_0.models.Model import Model
        
        representation, output_module, post_utility = build_model(config)
        losses = config.losses_metrics.losses
        metrics = config.losses_metrics.metrics
        
        # 使用Model.load_from_checkpoint - 这是HamGNN标准方式！
        original_model = Model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            representation=representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
        )
        original_model.eval()
        original_model.to(device)
        print(f"✅ 原始模型加载成功（使用标准方式）")
        
        # 2. 加载编译模型
        print("=== 加载编译后的TorchScript模型 ===")
        import torch_scatter  # 必须导入
        compiled_model = torch.jit.load(compiled_model_path, map_location=device)
        compiled_model.eval()
        print(f"✅ TorchScript模型加载成功")
        
        # 3. 创建测试数据
        print(f"=== 创建测试数据 (batch_size={batch_size}) ===")
        test_loader = prepare_data_with_batch_size(config, batch_size)
        
        # 检查是否需要动态图构建
        build_internal_graph = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('build_internal_graph', True)
        dynamic_transform = None
        
        if build_internal_graph:
            print("需要动态图构建（已移至预处理阶段）...")
            from HamGNN_v_2_0.models.HamGNN.BaseModel import DynamicGraphTransform
            radius_type = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('radius_type', 'openmx')
            radius_scale = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('radius_scale', 1.5)
            dynamic_transform = DynamicGraphTransform(radius_type=radius_type, radius_scale=radius_scale)
        else:
            print("使用预构建图数据，无需动态变换")
        
        print(f"测试数据集大小: {len(test_loader)} 批次")
        
        # 4. 充分预热运行（排除JIT编译开销）
        print("=== 充分预热运行（排除JIT编译开销）===")
        warmup_batches = 20  # 增加预热批次数
        warmup_iterator = iter(test_loader)
        
        print(f"  开始预热 {warmup_batches} 个批次...")
        with torch.no_grad():
            for i in range(warmup_batches):
                try:
                    warmup_batch = next(warmup_iterator).to(device)
                    if dynamic_transform is not None:
                        warmup_batch = dynamic_transform(warmup_batch)
                    warmup_data = batch_to_input_dict(warmup_batch)
                    
                    # 预热两个模型
                    _ = compiled_model(warmup_data)
                    _ = original_model(warmup_batch)
                    
                    if i % 5 == 0:
                        print(f"    预热进度: {i+1}/{warmup_batches}")
                except StopIteration:
                    print(f"    数据集只有 {i} 个批次，预热完成")
                    break
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        print("  ✅ 预热完成，开始正式性能测试")
        
        # 重新创建测试迭代器用于正式测试
        test_loader = prepare_data_with_batch_size(config, batch_size)
        
        # 5. 一致性测试（带准确计时）
        print("=== 开始一致性测试（准确计时）===")
        
        tolerance = 1e-5
        # total_batches = min(100, len(test_loader))  # 测试一些批次来看性能趋势
        total_batches = len(test_loader)  # 测试所有批次来看完整性能画像
        all_consistent = True
        
        # 统计信息
        compiled_times = []
        original_times = []
        batch_sizes = []  # 记录每个batch的大小
        edge_counts = []  # 记录每个batch的边数
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= total_batches:
                    break
                
                # 移动数据到设备
                batch = batch.to(device)
                
                # 记录batch信息
                nodes = batch.pos.shape[0]
                edges = batch.edge_index.shape[1]
                batch_sizes.append(nodes)
                edge_counts.append(edges)
                
                # 应用动态图变换（如果需要）
                if dynamic_transform is not None:
                    batch = dynamic_transform(batch)
                    edges = batch.edge_index.shape[1]  # 更新边数
                    edge_counts[-1] = edges
                
                # 转换数据格式给编译模型
                data = batch_to_input_dict(batch)
                
                # 性能测试
                compiled_output = None
                compiled_time = 0.0
                original_output = None
                original_time = 0.0
                
                # 测试编译模型（第一个batch添加profiler分析）
                try:
                    if batch_idx == 0:
                        # 对第一个batch进行详细的性能分析
                        print(f"\n🔍 对第一个batch进行详细性能分析...")
                        
                        # 确保输出目录存在
                        import os
                        profile_dir = "./performance_profiles"
                        os.makedirs(profile_dir, exist_ok=True)
                        
                        # 配置profiler参数，确保跨版本兼容性
                        profiler_kwargs = {
                            'record_shapes': True,
                            'with_stack': True,
                        }
                        
                        # 检查torch.profiler版本兼容性
                        try:
                            # PyTorch 1.8.1+的新版本profiler - 尝试不同的Activity类名
                            activity_cpu = None
                            activity_cuda = None
                            
                            # 先尝试 ProfilerActivity (PyTorch 2.x)
                            if hasattr(torch.profiler, 'ProfilerActivity'):
                                activity_cpu = torch.profiler.ProfilerActivity.CPU
                                activity_cuda = torch.profiler.ProfilerActivity.CUDA
                                print("  ✅ 使用PyTorch 2.x版本的ProfilerActivity")
                            # 再尝试 Activity (较早的PyTorch 2.x或1.x)
                            elif hasattr(torch.profiler, 'Activity'):
                                activity_cpu = torch.profiler.Activity.CPU
                                activity_cuda = torch.profiler.Activity.CUDA
                                print("  ✅ 使用PyTorch 1.8.1+版本的Activity")
                            else:
                                raise AttributeError("No Activity class found")
                            
                            profiler_kwargs['activities'] = [activity_cpu, activity_cuda]
                            profiler_kwargs['profile_memory'] = True
                            
                        except AttributeError:
                            # 旧版本torch.profiler或者使用torch.autograd.profiler
                            print("  ⚠️  torch.profiler.Activity不可用，尝试使用旧版profiler")
                            try:
                                # 尝试使用旧的torch.autograd.profiler
                                import torch.autograd.profiler as old_profiler
                                print("  ✅ 回退到torch.autograd.profiler")
                                
                                # 使用旧版profiler的简化版本
                                with old_profiler.profile(
                                    enabled=True,
                                    use_cuda=True,
                                    record_shapes=True,
                                    with_stack=True
                                ) as prof:
                                    with torch.no_grad():
                                        compiled_output = compiled_model(data)
                                
                                # 打印旧版profiler结果
                                print("\n--- 🚀 TorchScript 模型性能分析 (torch.autograd.profiler) ---")
                                print(prof.table(sort_by="self_cuda_time_total", row_limit=20))
                                
                                # 保存trace
                                trace_file = f"{profile_dir}/hamgnn_profiler_batch{batch_idx+1}_bsize{batch_size}.json"
                                try:
                                    prof.export_chrome_trace(trace_file)
                                    print(f"\n💾 详细性能trace已保存至: {trace_file}")
                                except Exception as trace_error:
                                    print(f"\n⚠️  trace保存失败: {trace_error}")
                                
                                # 跳过新版profiler的其余代码
                                profiler_kwargs = None
                                
                            except Exception as e:
                                print(f"  ❌ 旧版profiler也不可用: {e}")
                                print("  ⚠️  跳过性能分析，继续正常测试")
                                profiler_kwargs = None
                        
                        # 尝试添加experimental_config（在新版本PyTorch中可用）
                        if profiler_kwargs is not None:
                            try:
                                profiler_kwargs['experimental_config'] = torch._C._profiler._ExperimentalConfig(verbose=True)
                            except (AttributeError, ImportError):
                                print("  ⚠️  experimental_config不可用，使用基础profiler配置")
                            
                            with torch.profiler.profile(**profiler_kwargs) as prof:
                                with torch.no_grad():
                                    compiled_output = compiled_model(data)
                            
                            # 打印性能分析结果
                            print("\n--- 🚀 TorchScript 模型性能热点分析 (按CUDA时间排序) ---")
                            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
                            
                            print("\n--- 💾 TorchScript 模型内存使用分析 ---")
                            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
                            
                            print("\n--- ⚡ 关键TensorProduct操作分析 ---")
                            # 专门查找e3nn相关的操作 - 扩大搜索范围
                            e3nn_ops = []
                            for event in prof.key_averages():
                                key_lower = event.key.lower()
                                # 扩大搜索关键词
                                if any(keyword in key_lower for keyword in [
                                    'tensor_product', 'tensorproduct', 'e3nn', 
                                    'linear', 'einsum', 'spherical_harmonics',
                                    'irreps', 'tp', 'fulltp', 'gate'
                                ]):
                                    e3nn_ops.append(event)
                            
                            if e3nn_ops:
                                # 按CUDA时间排序 - 修复属性名
                                e3nn_ops.sort(key=lambda x: getattr(x, 'self_cuda_time_total', getattr(x, 'cuda_time_total', 0)), reverse=True)
                                print("前10个可能的瓶颈操作:")
                                for i, op in enumerate(e3nn_ops[:10]):
                                    # 兼容不同版本的属性名
                                    cuda_time_us = getattr(op, 'self_cuda_time_total', getattr(op, 'cuda_time_total', 0))
                                    cuda_time_ms = cuda_time_us / 1000  # 转换为毫秒
                                    memory_usage = getattr(op, 'self_cuda_memory_usage', getattr(op, 'cuda_memory_usage', 0))
                                    memory_mb = memory_usage / (1024 * 1024) if memory_usage > 0 else 0
                                    print(f"  {i+1:2d}. {op.key[:80]:<80} | CUDA时间: {cuda_time_ms:6.2f}ms | 内存: {memory_mb:6.1f}MB")
                            else:
                                print("  ⚠️  未找到明显的e3nn/TensorProduct相关操作，可能操作名称不同")
                            
                            print("\n--- 📈 性能瓶颈总结 ---")
                            # 分析最耗时的操作类型 - 修复属性访问
                            total_cuda_time = sum(getattr(event, 'self_cuda_time_total', getattr(event, 'cuda_time_total', 0)) for event in prof.key_averages())
                            print(f"  总CUDA计算时间: {total_cuda_time/1000:.2f}ms")
                            
                            # 统计主要操作类型的时间占比
                            op_categories = {
                                'TensorProduct/einsum': 0,
                                'Linear/MatMul': 0,
                                'Memory操作': 0,
                                'Activation': 0,
                                '其他': 0
                            }
                            
                            for event in prof.key_averages():
                                key_lower = event.key.lower()
                                cuda_time = getattr(event, 'self_cuda_time_total', getattr(event, 'cuda_time_total', 0))
                                if any(kw in key_lower for kw in ['tensor_product', 'tensorproduct', 'einsum']):
                                    op_categories['TensorProduct/einsum'] += cuda_time
                                elif any(kw in key_lower for kw in ['linear', 'matmul', 'mm', 'bmm']):
                                    op_categories['Linear/MatMul'] += cuda_time
                                elif any(kw in key_lower for kw in ['copy', 'cat', 'index', 'slice', 'view', 'reshape']):
                                    op_categories['Memory操作'] += cuda_time
                                elif any(kw in key_lower for kw in ['relu', 'gelu', 'silu', 'activation', 'sigmoid']):
                                    op_categories['Activation'] += cuda_time
                                else:
                                    op_categories['其他'] += cuda_time
                            
                            print("  操作类型时间分布:")
                            for category, time_us in op_categories.items():
                                if total_cuda_time > 0:
                                    percentage = time_us / total_cuda_time * 100
                                    print(f"    {category:<20}: {time_us/1000:6.2f}ms ({percentage:5.1f}%)")
                            
                            # 保存详细的profiler trace文件（可选）
                            trace_file = f"{profile_dir}/hamgnn_profiler_batch{batch_idx+1}_bsize{batch_size}.json"
                            try:
                                prof.export_chrome_trace(trace_file)
                                print(f"\n💾 详细性能trace已保存至: {trace_file}")
                                print("   可以在Chrome浏览器中打开 chrome://tracing/ 来查看详细的性能时间线")
                            except Exception as trace_error:
                                print(f"\n⚠️  trace保存失败: {trace_error}")
                            print("=" * 80)
                        else:
                            # 如果profiler不可用，正常运行推理
                            with torch.no_grad():
                                compiled_output = compiled_model(data)
                    
                    # 正常的性能测试
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start_time = time.time()
                    if batch_idx > 0:  # 第一个batch已经在profiler中执行过了
                        compiled_output = compiled_model(data)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    compiled_time = time.time() - start_time
                    compiled_times.append(compiled_time)
                except Exception as e:
                    print(f"❌ 批次{batch_idx + 1}编译模型失败: {e}")
                    all_consistent = False
                    continue
                
                # 测试原始模型
                try:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start_time = time.time()
                    original_output = original_model(batch)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    original_time = time.time() - start_time
                    original_times.append(original_time)
                except Exception as e:
                    print(f"❌ 批次{batch_idx + 1}原始模型失败: {e}")
                    all_consistent = False
                    continue
                
                # 只对前3个batch做详细一致性检查
                if batch_idx < 3:
                    print(f"\n批次 {batch_idx + 1}/{total_batches}:")
                    print(f"  batch信息: nodes={nodes}, edges={edges}")
                    print(f"  ✅ 编译模型推理成功，时间: {compiled_time:.4f}s")
                    print(f"  ✅ 原始模型推理成功，时间: {original_time:.4f}s")
                    
                    batch_consistent = True
                    meaningful_comparisons = 0
                    
                    # 比较输出（跳过空张量）
                    if original_output is not None and compiled_output is not None:
                        for key in original_output.keys():
                            if key in compiled_output and isinstance(original_output[key], torch.Tensor):
                                orig = original_output[key]
                                comp = compiled_output[key]
                                
                                # 跳过空张量
                                if orig.numel() == 0 or comp.numel() == 0:
                                    print(f"  {key}: 空张量，跳过比较 (形状={orig.shape})")
                                    continue
                                    
                                meaningful_comparisons += 1
                                
                                if orig.shape == comp.shape:
                                    diff = torch.abs(orig - comp)
                                    max_diff = torch.max(diff).item()
                                    mean_diff = torch.mean(diff).item()
                                    
                                    # 计算相对差异
                                    orig_magnitude = torch.mean(torch.abs(orig)).item()
                                    relative_max_diff = max_diff / (orig_magnitude + 1e-12)
                                    relative_mean_diff = mean_diff / (orig_magnitude + 1e-12)
                                    
                                    is_consistent = max_diff < tolerance
                                    batch_consistent = batch_consistent and is_consistent
                                    
                                    print(f"  {key}: 形状={orig.shape}, 最大绝对差异={max_diff:.2e} (相对差异={relative_max_diff:.2e}), 平均绝对差异={mean_diff:.2e} (相对差异={relative_mean_diff:.2e}), {'✅' if is_consistent else '❌'}")
                                else:
                                    print(f"  {key}: 形状不匹配 {orig.shape} vs {comp.shape} ❌")
                                    batch_consistent = False
                    
                    # 显示时间对比
                    speedup = original_time / compiled_time if compiled_time > 0 else 0
                    print(f"  推理时间: 原始={original_time:.4f}s, 编译={compiled_time:.4f}s")
                    print(f"  速度比较: {speedup:.2f}x ({'🚀 编译更快' if speedup > 1 else '🐌 编译较慢'})")
                    print(f"  有效比较: {meaningful_comparisons}个输出")
                    print(f"  批次一致性: {'✅' if batch_consistent else '❌'}")
                    
                    all_consistent = all_consistent and batch_consistent
                else:
                    # 其他batch只显示简要性能信息
                    speedup = original_time / compiled_time if compiled_time > 0 else 0
                    status = "🚀" if speedup > 1 else "🐌"
                    # 每10个batch显示一次进度
                    if batch_idx % 10 == 0 or batch_idx < 10:
                        print(f"批次{batch_idx + 1:3d}: nodes={nodes:3d}, edges={edges:4d}, 原始={original_time:.4f}s, 编译={compiled_time:.4f}s, {speedup:.2f}x {status}")
                    elif batch_idx % 50 == 0:
                        print(f"... 已测试 {batch_idx+1} 批次 ...")
        
        print(f"\n⏱️  总共测试了 {len(compiled_times)} 个批次 (共{len(test_loader)}批次)")
        
        # 简单的性能趋势预览
        if len(compiled_times) >= 20:
            print()
            print("📈 性能趋势预览:")
            # 显示每50个batch的平均性能
            chunk_size = 50
            for i in range(0, len(compiled_times), chunk_size):
                chunk_end = min(i + chunk_size, len(compiled_times))
                chunk_speedups = [original_times[j]/compiled_times[j] for j in range(i, chunk_end)]
                avg_speedup = sum(chunk_speedups) / len(chunk_speedups)
                faster_in_chunk = sum(1 for s in chunk_speedups if s > 1)
                print(f"  批次{i+1:3d}-{chunk_end:3d}: 平均{avg_speedup:.2f}x, {faster_in_chunk}/{len(chunk_speedups)}个更快 ({'🚀' if avg_speedup > 1 else '🐌'})")
        
        print()
        print("=" * 60)
        print("🎯 一致性测试结果:")
        print(f"  测试批次: {total_batches}")
        print(f"  容差阈值: {tolerance:.0e}")
        print(f"  整体一致性: {'✅ 通过' if all_consistent else '❌ 失败'}")
        
        # 详细性能统计和分析
        overall_speedup = None
        if compiled_times and original_times:
            import numpy as np
            
            print()
            print("📊 性能统计（已排除JIT编译开销）:")
            
            # 基本统计
            avg_original = sum(original_times) / len(original_times)
            avg_compiled = sum(compiled_times) / len(compiled_times)
            overall_speedup = avg_original / avg_compiled
            
            # 速度比较统计
            speedups = [orig/comp for orig, comp in zip(original_times, compiled_times)]
            
            print(f"  测试批次数: {len(compiled_times)} (batch_size={batch_size})")
            print(f"  原始模型平均时间: {avg_original:.4f}s")
            print(f"  TorchScript模型平均时间: {avg_compiled:.4f}s")
            print(f"  平均速度比较: {overall_speedup:.2f}x ({'🚀 TorchScript更快' if overall_speedup > 1 else '🐌 TorchScript较慢'})")
            
            # 性能分布统计
            speedups_sorted = sorted(speedups)
            print()
            print("📈 速度比较分布:")
            print(f"  最快: {max(speedups):.2f}x")
            print(f"  最慢: {min(speedups):.2f}x") 
            print(f"  中位数: {speedups_sorted[len(speedups_sorted)//2]:.2f}x")
            
            # 计算编译模型更快的batch数量
            faster_count = sum(1 for s in speedups if s > 1)
            print(f"  编译模型更快的批次: {faster_count}/{len(speedups)} ({faster_count/len(speedups)*100:.1f}%)")
            
            # batch大小与性能关系分析
            if batch_sizes and edge_counts:
                print()
                print("🔍 批次规模与性能关系:")
                
                # 找出最快和最慢的几个批次
                speed_indices = sorted(range(len(speedups)), key=lambda i: speedups[i], reverse=True)
                
                print("  编译模型表现最好的批次:")
                for i in speed_indices[:5]:
                    print(f"    批次{i+1}: nodes={batch_sizes[i]}, edges={edge_counts[i]}, {speedups[i]:.2f}x")
                
                print("  编译模型表现最差的批次:")
                for i in speed_indices[-5:]:
                    print(f"    批次{i+1}: nodes={batch_sizes[i]}, edges={edge_counts[i]}, {speedups[i]:.2f}x")
                
                # 规模分析
                avg_nodes = sum(batch_sizes) / len(batch_sizes)
                avg_edges = sum(edge_counts) / len(edge_counts)
                print(f"  平均批次规模: nodes={avg_nodes:.1f}, edges={avg_edges:.1f}")
                
                # 前10个batch vs 后面batch的性能对比
                if len(speedups) > 10:
                    early_speedups = speedups[:10]
                    later_speedups = speedups[10:]
                    
                    avg_early = sum(early_speedups) / len(early_speedups)
                    avg_later = sum(later_speedups) / len(later_speedups)
                    
                    print()
                    print("🚀 预热效果分析:")
                    print(f"  前10个batch平均速度: {avg_early:.2f}x")
                    print(f"  后续batch平均速度: {avg_later:.2f}x")
                    print(f"  预热改善: {avg_later/avg_early:.2f}x ({'✅ 有明显改善' if avg_later > avg_early * 1.1 else '❌ 改善不明显'})")
            
            if overall_speedup < 1:
                print()
                print("  ⚠️  编译模型在当前测试条件下较慢，可能原因:")
                print("     - 模型规模较小，JIT开销相对明显")
                print("     - GPU内存访问模式和硬件特性的影响")
                print("     - 不同batch大小可能触发不同的优化路径")
                print("     - 建议测试更大规模的批次来全面评估性能")
        
        if all_consistent:
            print()
            print("🎉 恭喜！TorchScript模型与原始checkpoint完全一致！")
            print("✅ 模型可以安全用于生产部署")
            
            if overall_speedup is not None and overall_speedup > 1:
                print(f"🚀 性能提升: {overall_speedup:.1f}x 加速")
            elif overall_speedup is not None:
                print(f"📈 性能表现: {overall_speedup:.2f}x (编译模型在大规模推理中可能更快)")
            
            return True
        else:
            print()
            print(f"⚠️  发现一致性问题，需要进一步检查 (batch_size={batch_size})")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_torch_compile_test(config, batch_size):
    """测试torch.compile优化的模型性能"""
    
    print(f"\n🚀 PyTorch torch.compile 模型性能测试 (batch_size={batch_size})")
    print("=" * 70)
    
    # 从环境变量读取路径（SLURM脚本设置的），如果没有设置则使用默认值
    config_path = os.getenv('CONFIG_PATH', "path/to/your/config.yaml")
    checkpoint_path = os.getenv('CHECKPOINT_PATH', "path/to/your/checkpoint.ckpt")
    
    # 检查文件存在性
    for path_name, path in [("Checkpoint", checkpoint_path)]:
        if not Path(path).exists():
            print(f"❌ {path_name}不存在: {path}")
            return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()
    
    try:
        # 1. 加载原始模型
        print("=== 加载原始模型 ===")
        from HamGNN_v_2_0.main import build_model
        from HamGNN_v_2_0.models.Model import Model
        
        representation, output_module, post_utility = build_model(config)
        losses = config.losses_metrics.losses
        metrics = config.losses_metrics.metrics
        
        # 加载原始模型
        original_model = Model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            representation=representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
        )
        original_model.eval()
        original_model.to(device)
        print(f"✅ 原始模型加载成功")
        
        # 2. 使用torch.compile优化
        print("=== 使用torch.compile优化 ===")
        print("⚠️  注意: torch.compile是运行时JIT编译器，不会生成可保存的.pt文件")
        print("    与TorchScript不同，torch.compile在每次运行时都需要重新编译")
        print("    但它可能提供更好的性能优化，特别是在PyTorch 2.0+中")
        print()
        print("⚠️  重要提示: torch.compile 对 HamGNN v2.0 的支持尚未完成!")
        print("    当前会失败，因为我们还没有完成 torch.fx 改造（技术路线图第二阶段）")
        print("    这个测试仅用于演示未来的优化方向")
        print()
        
        # 检查PyTorch版本
        pytorch_version = torch.__version__
        print(f"PyTorch版本: {pytorch_version}")
        
        if not hasattr(torch, 'compile'):
            print("❌ torch.compile不可用，需要PyTorch 2.0+")
            return False
        
        # 编译模型
        compiled_model = torch.compile(original_model, mode="default")
        print(f"✅ torch.compile优化完成 (mode=default)")
        
        # 3. 创建测试数据
        print(f"=== 创建测试数据 (batch_size={batch_size}) ===")
        test_loader = prepare_data_with_batch_size(config, batch_size)
        
        # 检查是否需要动态图构建
        build_internal_graph = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('build_internal_graph', True)
        dynamic_transform = None
        
        if build_internal_graph:
            print("需要动态图构建（已移至预处理阶段）...")
            from HamGNN_v_2_0.models.HamGNN.BaseModel import DynamicGraphTransform
            radius_type = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('radius_type', 'openmx')
            radius_scale = config.get('representation_nets', {}).get('HamGNN_pre', {}).get('radius_scale', 1.5)
            dynamic_transform = DynamicGraphTransform(radius_type=radius_type, radius_scale=radius_scale)
        else:
            print("使用预构建图数据，无需动态变换")
        
        print(f"测试数据集大小: {len(test_loader)} 批次")
        
        # 4. 充分预热运行（torch.compile需要更多预热）
        print("=== 充分预热运行（torch.compile需要更多预热）===")
        warmup_batches = 30  # torch.compile需要更多预热
        warmup_iterator = iter(test_loader)
        
        print(f"  开始预热 {warmup_batches} 个批次...")
        with torch.no_grad():
            for i in range(warmup_batches):
                try:
                    warmup_batch = next(warmup_iterator).to(device)
                    if dynamic_transform is not None:
                        warmup_batch = dynamic_transform(warmup_batch)
                    
                    # 预热两个模型
                    _ = compiled_model(warmup_batch)
                    _ = original_model(warmup_batch)
                    
                    if i % 10 == 0:
                        print(f"    预热进度: {i+1}/{warmup_batches}")
                except StopIteration:
                    print(f"    数据集只有 {i} 个批次，预热完成")
                    break
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        print("  ✅ 预热完成，开始正式性能测试")
        
        # 重新创建测试迭代器用于正式测试
        test_loader = prepare_data_with_batch_size(config, batch_size)
        
        # 5. 性能测试（不做一致性检查，只测速度）
        print("=== 开始性能测试（torch.compile vs 原始模型）===")
        
        total_batches = min(100, len(test_loader))  # 测试适量批次
        
        # 统计信息
        compiled_times = []
        original_times = []
        batch_sizes = []  # 记录每个batch的大小
        edge_counts = []  # 记录每个batch的边数
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= total_batches:
                    break
                
                # 移动数据到设备
                batch = batch.to(device)
                
                # 记录batch信息
                nodes = batch.pos.shape[0]
                edges = batch.edge_index.shape[1]
                batch_sizes.append(nodes)
                edge_counts.append(edges)
                
                # 应用动态图变换（如果需要）
                if dynamic_transform is not None:
                    batch = dynamic_transform(batch)
                    edges = batch.edge_index.shape[1]  # 更新边数
                    edge_counts[-1] = edges
                
                # 性能测试
                compiled_time = 0.0
                original_time = 0.0
                
                # 测试torch.compile模型
                try:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start_time = time.time()
                    compiled_output = compiled_model(batch)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    compiled_time = time.time() - start_time
                    compiled_times.append(compiled_time)
                except Exception as e:
                    print(f"❌ 批次{batch_idx + 1}torch.compile模型失败: {e}")
                    continue
                
                # 测试原始模型
                try:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start_time = time.time()
                    original_output = original_model(batch)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    original_time = time.time() - start_time
                    original_times.append(original_time)
                except Exception as e:
                    print(f"❌ 批次{batch_idx + 1}原始模型失败: {e}")
                    continue
                
                # 显示进度
                if batch_idx < 10:
                    speedup = original_time / compiled_time if compiled_time > 0 else 0
                    status = "🚀" if speedup > 1 else "🐌"
                    print(f"批次{batch_idx + 1:3d}: nodes={nodes:3d}, edges={edges:4d}, 原始={original_time:.4f}s, torch.compile={compiled_time:.4f}s, {speedup:.2f}x {status}")
                elif batch_idx % 20 == 0:
                    speedup = original_time / compiled_time if compiled_time > 0 else 0
                    status = "🚀" if speedup > 1 else "🐌"
                    print(f"批次{batch_idx + 1:3d}: {speedup:.2f}x {status}")
        
        print(f"\n⏱️  总共测试了 {len(compiled_times)} 个批次 (共{len(test_loader)}批次) - batch_size={batch_size}")
        
        # 性能统计和分析
        if compiled_times and original_times:
            import numpy as np
            
            print()
            print("📈 torch.compile性能统计（已排除编译开销）:")
            
            # 基本统计
            avg_original = sum(original_times) / len(original_times)
            avg_compiled = sum(compiled_times) / len(compiled_times)
            overall_speedup = avg_original / avg_compiled
            
            # 速度比较统计
            speedups = [orig/comp for orig, comp in zip(original_times, compiled_times)]
            
            print(f"  测试批次数: {len(compiled_times)} (batch_size={batch_size})")
            print(f"  原始模型平均时间: {avg_original:.4f}s")
            print(f"  torch.compile模型平均时间: {avg_compiled:.4f}s")
            print(f"  平均速度比较: {overall_speedup:.2f}x {'🚀 torch.compile更快' if overall_speedup > 1 else '🐌 torch.compile较慢'}")
            
            # 性能分布统计
            speedups_sorted = sorted(speedups)
            print()
            print("📈 速度比较分布:")
            print(f"  最快: {max(speedups):.2f}x")
            print(f"  最慢: {min(speedups):.2f}x") 
            print(f"  中位数: {speedups_sorted[len(speedups_sorted)//2]:.2f}x")
            
            # 计算torch.compile模型更快的batch数量
            faster_count = sum(1 for s in speedups if s > 1)
            print(f"  torch.compile模型更快的批次: {faster_count}/{len(speedups)} ({faster_count/len(speedups)*100:.1f}%)")
            
            print()
            print("🔍 torch.compile 特点:")
            print("  - 运行时动态优化，无需手动修改代码")
            print("  - 不生成可保存的模型文件，每次启动需重新编译")
            print("  - 适合开发和实验阶段的性能优化")
            
            if overall_speedup > 1:
                print()
                print(f"🎉 torch.compile在batch_size={batch_size}下表现优异！")
                print(f"🚀 性能提升: {overall_speedup:.1f}x 加速 (vs 原始模型)")
            else:
                print()
                print(f"📈 torch.compile在batch_size={batch_size}下表现: {overall_speedup:.2f}x")
                print("  尝试更大的batch_size或不同的compile mode可能会有更好的效果")
            
            return True
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# batch_to_input_dict函数现在从tools.utils导入，这里不再重复定义


def example_dataloader_inference():
    """演示使用 DataLoader 的推理（推荐方式）"""
    
    print("\n=== DataLoader 推理示例 ===")
    
    # 1. 加载模型
    model_path = "path/to/your/compiled_model.hamgnn.pt"
    if not Path(model_path).exists():
        print(f"错误：模型文件不存在: {model_path}")
        return
    
    model = torch.jit.load(model_path)
    model.eval()
    
    # 2. 加载配置文件
    config_path = "config_test.yaml"  # 假设配置文件在当前目录
    if not Path(config_path).exists():
        print(f"错误：配置文件不存在: {config_path}")
        print("请提供正确的配置文件路径")
        return
    
    config = read_config(config_path)
    
    # 3. 创建数据加载器
    print("创建数据加载器...")
    _, _, test_loader = graph_data_module(config)
    
    # 4. 检查是否需要动态图构建
    build_internal_graph = config.get('build_internal_graph', True)
    dynamic_transform = None
    
    if not build_internal_graph:
        print("配置为外部图构建，初始化 DynamicGraphTransform...")
        radius_type = config.get('radius_type', 'openmx')
        radius_scale = config.get('radius_scale', 1.5)
        dynamic_transform = DynamicGraphTransform(radius_type=radius_type, radius_scale=radius_scale)
    
    # 5. 批量推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"开始批量推理，共 {len(test_loader)} 个批次...")
    
    total_samples = 0
    total_mae = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 限制只处理前几个批次作为示例
            if batch_idx >= 3:
                break
            
            # 移动数据到设备
            batch = batch.to(device)
            
            # 如果需要，应用动态图变换
            if dynamic_transform is not None:
                batch = dynamic_transform(batch)
            
            # 使用工具函数转换数据格式
            data = batch_to_input_dict(batch)
            
            # 推理
            output = model(data)
            
            # 计算误差（如果有目标值）
            if 'hamiltonian' in output and hasattr(batch, 'hamiltonian'):
                pred = output['hamiltonian']
                target = batch.hamiltonian
                mae = torch.mean(torch.abs(pred - target)).item()
                total_mae += mae
                total_samples += 1
                
                print(f"  批次 {batch_idx + 1}: 哈密顿量形状 {pred.shape}, MAE: {mae:.6e}")
            else:
                print(f"  批次 {batch_idx + 1}: 输出键 {list(output.keys())}")
    
    if total_samples > 0:
        avg_mae = total_mae / total_samples
        print(f"\n平均 MAE: {avg_mae:.6e}")
    
    print("DataLoader 推理完成")


def example_command_line_usage():
    """演示命令行工具的使用方法"""
    
    print("\n=== 命令行工具使用示例 ===")
    
    print("1. 编译模型:")
    print("   python tools/compile.py \\")
    print("     --config path/to/config.yaml \\")
    print("     --checkpoint path/to/checkpoint.ckpt \\")
    print("     --output compiled_model.hamgnn.pt \\")
    print("     --device cuda")
    
    print("\n2. 验证编译模型（一致性验证，推荐）:")
    print("   python tools/validate_compiled.py \\")
    print("     --model compiled_model.hamgnn.pt \\")
    print("     --config path/to/config.yaml \\")
    print("     --checkpoint path/to/checkpoint.ckpt \\")
    print("     --mode consistency \\")
    print("     --device cuda")
    
    print("\n3. 验证模型精度:")
    print("   python tools/validate_compiled.py \\")
    print("     --model compiled_model.hamgnn.pt \\")
    print("     --config path/to/config.yaml \\")
    print("     --mode accuracy \\")
    print("     --target-mae 2.05e-5 \\")
    print("     --device cuda")
    
    print("\n4. 完整验证（一致性 + 精度）:")
    print("   python tools/validate_compiled.py \\")
    print("     --model compiled_model.hamgnn.pt \\")
    print("     --config path/to/config.yaml \\")
    print("     --checkpoint path/to/checkpoint.ckpt \\")
    print("     --mode both \\")
    print("     --device cuda")


def main():
    """主函数 - 多场景性能测试"""
    
    print("🚀 HamGNN 多场景性能对比测试")
    print("=" * 80)
    print()
    print("📋 测试概述:")
    print("  1. TorchScript性能测试: 比较TorchScript模型与原始模型")
    print("  2. torch.compile性能测试: 比较torch.compile优化模型与原始模型")
    print("  3. 多个batch_size自动测试: 评估不同批次大小下的性能")
    print()
    print("ℹ️  关于torch.compile说明:")
    print("    torch.compile是PyTorch 2.0+的运行时JIT编译器，与生成可保存.pt文件的TorchScript不同")
    print("    它在每次运行时动态优化，但不能生成可移植的模型文件")
    print("    两种方法都有各自的优势，适用于不同的部署场景")
    print()
    
    # 从环境变量读取配置文件路径
    config_path = os.getenv('CONFIG_PATH', "path/to/your/config.yaml")
    config = read_config(config_path)
    
    # 待测试的batch_size列表
    batch_sizes_to_test = [1, 8, 16, 32]  # 可以根据需要调整
    
    print(f"📋 测试计划: 将测试batch_size: {batch_sizes_to_test}")
    print()
    
    all_results = {}
    
    # 对每个batch_size进行测试
    for batch_size in batch_sizes_to_test:
        print(f"\n{'='*80}")
        print(f"📋 开始测试 batch_size = {batch_size}")
        print(f"{'='*80}")
        
        # 1. TorchScript性能测试
        torchscript_success = run_performance_test(config, batch_size)
        
        # 2. torch.compile性能测试
        compile_success = run_torch_compile_test(config, batch_size)
        
        all_results[batch_size] = {
            'torchscript': torchscript_success,
            'compile': compile_success
        }
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("📈 测试结果汇总")
    print(f"{'='*80}")
    
    for batch_size, results in all_results.items():
        torchscript_status = "✅ 成功" if results['torchscript'] else "❌ 失败"
        compile_status = "✅ 成功" if results['compile'] else "❌ 失败"
        print(f"batch_size={batch_size:2d}: TorchScript {torchscript_status}, torch.compile {compile_status}")
    
    print()
    print("📁 使用建议:")
    print("  1. 生产部署：选择TorchScript，生成可移植的.pt模型文件")
    print("  2. 开发实验：可尝试torch.compile进行运行时优化")
    print("  3. 性能测试：建议在实际数据规模下进行全面评估")
    print("  4. 版本要求：torch.compile需要PyTorch 2.0+，TorchScript支持更广泛")
    
    # 返回成功状态（只要有一个测试成功就认为成功）
    any_success = any(results['torchscript'] or results['compile'] for results in all_results.values())
    exit(0 if any_success else 1)


if __name__ == "__main__":
    main()