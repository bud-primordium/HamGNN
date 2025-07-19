"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-12 23:42:11 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 19:15:27
 */
"""
"""HamGNN (v2.0) 的主入口脚本。

该文件负责整个模型的生命周期管理，包括：
1.  **数据准备**: 从配置文件中读取数据集参数，加载图结构数据，并使用 PyTorch Lightning 的 DataModule 进行封装。
2.  **模型构建**: 根据配置动态选择并构建 GNN 表示网络 (如 HamGNNConvE3, HamGNNTransformer) 和相应的输出网络，以适应不同的物理属性预测任务（如总能量、原子力、介电张量等）。
3.  **训练与评估**: 初始化 PyTorch Lightning 的 `Trainer`，配置回调函数（如学习率监控、早停、模型检查点），并启动训练、验证和测试流程。
4.  **命令行接口**: 使用 `argparse` 解析命令行参数，允许用户指定配置文件，并启动整个流程。
"""
import torch
import torch.nn as nn
import numpy as np
import os
import e3nn
from e3nn import o3
from .GraphData.graph_data import graph_data_module
from .input.config_parsing import read_config
from .models.outputs import (Born, Born_node_vec, scalar, trivial_scalar, Force, 
                            Force_node_vec, crystal_tensor, piezoelectric, total_energy_and_atomic_forces, EPC_output)
import pytorch_lightning as pl
from .models.Model import Model
from .models.version import soft_logo
from pytorch_lightning.loggers import TensorBoardLogger
from .models.HamGNN.net import HamGNNTransformer, HamGNNConvE3, HamGNNPlusPlusOut
from torch.nn import functional as F
import pprint
import warnings
import sys
import socket
from .models.utils import get_hparam_dict
import argparse


def prepare_data(config):
    """准备并封装图数据。

    该函数根据配置文件中的参数，加载预处理好的图数据，并将其封装到
    一个 PyTorch Lightning 的 DataModule 中，以便于后续的训练、验证和测试。

    Args:
        config (dict): 
            从 YAML 文件中读取并解析后的配置对象。该对象应包含 `dataset_params` 属性，
            其中包含以下键：
            
            - train_ratio (float): 训练集在总数据集中的比例。
            - val_ratio (float): 验证集在总数据集中的比例。
            - test_ratio (float): 测试集在总数据集中的比例。
            - batch_size (int): 每个批次的样本数量。
            - split_file (str, optional): 用于加载/保存数据集划分索引的文件路径。
            - graph_data_path (str): 包含图数据的 `.npz` 文件路径或其所在目录。

    Returns:
        pytorch_lightning.LightningDataModule: 封装了训练、验证和测试数据集的 DataModule 实例。
    """
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    
    # 如果提供的路径是目录而非文件，则自动附加默认文件名 `graph_data.npz`
    if not os.path.isfile(graph_data_path):
        if not os.path.exists(graph_data_path):
            os.mkdir(graph_data_path)
        graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    
    if os.path.exists(graph_data_path):
        print(f"Loading graph data from {graph_data_path}!")
    else:
        print(f'The graph_data.npz file was not found in {graph_data_path}!')

    # 从 .npz 文件加载图数据字典
    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())

    # 根据配置决定是否需要动态图构建转换
    transform = None
    if hasattr(config, 'representation_nets') and hasattr(config.representation_nets, 'HamGNN_pre'):
        if getattr(config.representation_nets.HamGNN_pre, 'build_internal_graph', False):
            # 导入并创建 DynamicGraphTransform
            from .models.HamGNN.BaseModel import DynamicGraphTransform
            
            # 从配置中获取参数
            radius_type = getattr(config.representation_nets.HamGNN_pre, 'radius_type', 'openmx')
            radius_scale = getattr(config.representation_nets.HamGNN_pre, 'radius_scale', 1.5)
            
            transform = DynamicGraphTransform(radius_type=radius_type, radius_scale=radius_scale)
            print(f"启用动态图构建转换: radius_type={radius_type}, radius_scale={radius_scale}")
    
    # 初始化数据模块,它将处理数据集的划分、加载和批处理
    graph_dataset = graph_data_module(graph_dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, 
                                        batch_size=batch_size, split_file=split_file, transform=transform)
    # 根据当前阶段（如 'fit' 或 'test'）设置数据模块
    graph_dataset.setup(stage=config.setup.stage)

    return graph_dataset

def build_model(config):
    """根据配置构建 GNN 模型。

    此函数动态地构建图表示网络 (GNN) 和特定于任务的输出模块。
    它首先根据配置选择 GNN 架构，然后根据要预测的物理属性（如力、介电张量、哈密顿量等）
    选择并初始化相应的输出层。

    Args:
        config (dict): 
            从 YAML 文件中读取并解析后的配置对象。该对象应包含以下关键属性：
        
            - setup.GNN_Net (str): 要使用的 GNN 网络名称 (例如, 'hamgnnconv')。
            - setup.property (str): 要预测的目标物理属性 (例如, 'hamiltonian')。
            - representation_nets (dict): GNN 表示网络的超参数配置。
            - output_nets (dict): 输出网络的超参数配置。

    Returns:
        tuple: 
            包含三个元素的元组:

            - Gnn_net (torch.nn.Module): 图表示网络实例。
            - output_module (torch.nn.Module): 任务输出网络实例。
            - post_utility (None): 占位符，当前版本中未使用。
    """
    print("Building model")
    # 确保表示网络和输出网络在哈密顿量类型上达成一致
    config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    
    # --- 1. 构建图表示网络 (GNN) ---
    # 根据配置文件选择并实例化核心的图表示学习网络。
    if config.setup.GNN_Net.lower() in ['hamgnnconv', 'hamgnnpre', 'hamgnn_pre']:
        # 为保证旧配置文件的兼容性，如果未指定 `use_corr_prod`，则默认为 True
        if 'use_corr_prod' not in config.representation_nets.HamGNN_pre:
            config.representation_nets.HamGNN_pre.use_corr_prod = True
        Gnn_net = HamGNNConvE3(config.representation_nets)
    elif config.setup.GNN_Net.lower() == 'hamgnntransformer':
        Gnn_net = HamGNNTransformer(config.representation_nets)
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()

    # --- 2. 根据目标属性构建输出网络 ---
    # 不同的物理属性具有不同的数据结构（标量、矢量、张量），因此需要匹配专门的输出模块。
    
    # 任务: 预测二阶张量 (如 Born 有效电荷, 介电常数)
    # Born 有效电荷和介电常数都是二阶张量，因此使用 `crystal_tensor` 输出模块进行预测。
    if config.setup.property.lower() in ['born', 'dielectric']:
        if config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=False, num_node_features=Gnn_net.num_node_pooling_features, num_edge_features=Gnn_net.num_edge_pooling_features, num_triplet_features=Gnn_net.in_features_three_body,
                                           activation=nn.Softplus(), use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'painn':
            #output_module = Born_node_vec(num_node_features=Gnn_net.num_scaler_out, activation=Gnn_net.activation, use_bath_norm=Gnn_net.use_batch_norm, bias=Gnn_net.lnode_bias,n_h=3)
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.luse_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.n_edge_features, 
                                            num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=True, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                           num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels, 
                                           num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet, l_minus_mean=config.setup.l_minus_mean)
        else:
            quit()

    # 任务: 预测原子力
    # 原子力是作用在每个原子上的矢量 (一阶张量)，因此使用 `Force` 输出模块。
    elif config.setup.property.lower() == 'force':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = Force(num_edge_features=Gnn_net.hidden_channels, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3)
        else:
            quit()

    # 任务: 预测压电张量
    # 压电张量是三阶张量，描述应力与电极化之间的关系，使用 `piezoelectric` 模块处理。
    elif config.setup.property.lower() == 'piezoelectric':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = piezoelectric(include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels,
                                          num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet)
        else:
            quit()
            
    # 任务: 预测原子级标量属性 (例如, 形成能), 使用 'mean' 聚合
    # 对于原子级标量属性，模型需要对每个节点的输出进行预测，然后通过 'mean' 池化得到体系的平均值。
    elif config.setup.property.lower() == 'scalar_per_atom':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('mean', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # 任务: 预测原子级标量属性, 使用 'max' 聚合
    # 与 'mean' 类似，但使用 'max' 池化，适用于需要关注最大值的场景。
    elif config.setup.property.lower() == 'scalar_max':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # 任务: 预测体系级标量属性 (例如, 总能量), 使用 'sum' 聚合
    # 对于总能量这类广延量，通常使用 'sum' 池化将所有原子或节点的贡献加和。
    elif config.setup.property.lower() == 'scalar':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=2)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('sum', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
        
    # 任务: 预测哈密顿量矩阵
    # 核心任务：预测哈密顿量矩阵。使用专门设计的 `HamGNNPlusPlusOut` 模块，该模块能够处理复杂的等变性和对称性约束。
    elif config.setup.property.lower() == 'hamiltonian':
        output_params = config.output_nets.HamGNN_out
        # 为保证旧配置文件的兼容性，设置默认参数
        if 'add_H_nonsoc' not in output_params:
            output_params.add_H_nonsoc = False
        if 'get_nonzero_mask_tensor' not in output_params:
            output_params.get_nonzero_mask_tensor = False
        if 'zero_point_shift' not in output_params:
            output_params.zero_point_shift = False
        
        output_module = HamGNNPlusPlusOut(irreps_in_node = Gnn_net.irreps_node_features, irreps_in_edge = Gnn_net.irreps_node_features, nao_max= output_params.nao_max, ham_type= output_params.ham_type,
                                         ham_only= output_params.ham_only, symmetrize=output_params.symmetrize,calculate_band_energy=output_params.calculate_band_energy,num_k=output_params.num_k,k_path=output_params.k_path,
                                         band_num_control=output_params.band_num_control, soc_switch=output_params.soc_switch, nonlinearity_type = output_params.nonlinearity_type, add_H0=output_params.add_H0, 
                                         spin_constrained=output_params.spin_constrained, collinear_spin=output_params.collinear_spin, minMagneticMoment=output_params.minMagneticMoment, add_H_nonsoc=output_params.add_H_nonsoc,
                                         get_nonzero_mask_tensor=output_params.get_nonzero_mask_tensor, zero_point_shift=output_params.zero_point_shift)

    else:
        print('Evaluation of this property is not supported!')
        quit()
    
    # 初始化后处理工具 (当前版本未使用)
    post_utility = None
    
    return Gnn_net, output_module, post_utility

def train_and_eval(config):
    """执行模型的训练、评估和测试流程。

    该函数协调整个工作流程：
    1. 准备数据。
    2. 构建模型。
    3. 设置数值精度。
    4. 根据配置（训练或测试）初始化 `Model` 实例。
    5. 配置并运行 PyTorch Lightning `Trainer`。
    6. 在训练结束后，记录超参数和评估结果。

    Args:
        config (dict): 从 YAML 文件中读取并解析后的配置对象。
    """
    data = prepare_data(config)

    graph_representation, output_module, post_utility = build_model(config)

    # 根据配置设置全局的 PyTorch 数值精度 (float32 或 float64)
    if config.setup.precision == 32:
        dtype = torch.float32
    else:
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    graph_representation.to(dtype)
    output_module.to(dtype)

    # 从配置中获取损失函数和评估指标的定义
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics
    
    # --- 训练阶段 (`stage` == 'fit') ---
    if config.setup.stage == 'fit':
        # 如果指定了检查点但不是断点续训，则从检查点加载模型权重进行微调或评估
        if config.setup.load_from_checkpoint and not config.setup.resume:
            model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            )   
        else:            
            # 否则，初始化一个新模型用于从头训练或断点续训
            model = Model(
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience,
            )

        # 计算并打印模型的可训练参数数量
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("The model you built has %d parameters." % params)

        # 配置 PyTorch Lightning 的回调函数
        callbacks = [
            pl.callbacks.LearningRateMonitor(),  # 监控并记录学习率
            pl.callbacks.EarlyStopping(  # 在验证损失不再改善时提前终止训练
                monitor="training/total_loss",
                patience=config.optim_params.stop_patience, min_delta=1e-6,
            ),
            pl.callbacks.ModelCheckpoint(  # 在训练过程中保存最佳模型
                filename="{epoch}-{val_loss:.6f}",
                save_top_k=1,
                verbose=False,
                monitor='validation/total_loss',
                mode='min',
            )
        ]

        # 配置 TensorBoard 日志记录器
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)    

        # 初始化并配置 PyTorch Lightning Trainer
        trainer = pl.Trainer(
            gpus=config.setup.num_gpus,
            precision=config.setup.precision,
            callbacks=callbacks,
            progress_bar_refresh_rate=1,
            logger=tb_logger,
            gradient_clip_val = config.optim_params.gradient_clip_val,
            max_epochs=config.optim_params.max_epochs,
            default_root_dir=config.profiler_params.train_dir,
            min_epochs=config.optim_params.min_epochs,
            # 如果 `resume` 为 True, 则从指定的检查点路径恢复训练状态
            resume_from_checkpoint = config.setup.checkpoint_path if config.setup.resume else None
        )

        print("Start training.")
        trainer.fit(model, data)
        print("Training done.")

        # --- 评估阶段 (`stage` == 'fit' 结束后) ---
        print("Start eval.")
        results = trainer.test(model, data.test_dataloader())
        
        # 训练和测试结束后，将模型的超参数和对应的性能指标记录到 TensorBoard，便于实验跟踪和比较。
        hparam_dict = get_hparam_dict(config)
        metric_dict = dict() 
        for result_dict in results:
            metric_dict.update(result_dict)
        trainer.logger.experiment.add_hparams(hparam_dict, metric_dict)
        print("Eval done.")
    
    # --- 预测/测试阶段 (`stage` == 'test') ---
    if config.setup.stage == 'test': 
        # 从指定的检查点加载训练好的模型
        model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            ) 
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)

        trainer = pl.Trainer(gpus=config.setup.num_gpus, precision=config.setup.precision, logger=tb_logger)
        trainer.test(model=model, datamodule=data)

def HamGNN():
    """程序主函数。

    解析命令行参数，读取配置，设置全局随机种子以保证可复现性，
    并调用 `train_and_eval` 启动整个训练或评估流程。
    """
    # torch.autograd.set_detect_anomaly(True) # 用于调试，检测并报告反向传播中的数值问题
    
    # 固定全局随机数种子以保证实验的可复现性。
    # `seed_everything` 是 PyTorch Lightning 的一个实用函数。
    pl.seed_everything(666)
    print(soft_logo)
    
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default='config.yaml', type=str, metavar='N', help='Path to the configuration file (default: config.yaml)')
    args = parser.parse_args()

    # --- 配置加载与处理 ---
    configure = read_config(config_file_name=args.config)
    hostname = socket.getfqdn(socket.gethostname())
    configure.setup.hostname = hostname
    pprint.pprint(configure) # 打印最终使用的配置信息，便于调试��记录
    
    # 根据配置决定是否忽略运行时警告
    if configure.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    
    # --- 启动核心流程 ---
    train_and_eval(configure)

if __name__ == '__main__':
    HamGNN()