"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 13:46:53 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 21:09:02
 */
"""
"""定义了核心的 PyTorch Lightning 模型封装。

该文件中的 `Model` 类是一个 `pytorch_lightning.LightningModule`，它将图表示网络 (representation) 
和输出网络 (output) 组合在一起，并实现了标准的训练、验证和测试流程。它负责处理优化器配置、
损失计算、指标记录以及在 TensorBoard 中进行可视化。
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as opt
from typing import List, Dict, Union
from torch.nn import functional as F
from .utils import scatter_plot
import numpy as np
import os
import pandas as pd


class Model(pl.LightningModule):
    """
    核心的 PyTorch Lightning 模型，用于封装和训练 GNN。

    这个类集成了表示网络和输出网络，并定义了完整的训练、验证和测试循环。
    它通过配置文件接收损失函数、评估指标和优化器参数，实现了高度的灵活性和可配置性。

    Attributes:
        representation (nn.Module): 图表示网络，用于从输入图数据中提取特征。
        output_module (nn.Module): 输出网络，用于从表示网络提取的特征中预测目标属性。
        losses (List[Dict]): 一个字典列表，定义了用于训练的损失函数。
        metrics (List[Dict]): 一个字典列表，定义了用于验证和测试的评估指标。
        post_processing (callable, optional): 一个可选的后处理函数，用于在测试阶段计算需要梯度信息的物理量。
    """
    def __init__(
            self,
            representation: nn.Module,
            output: nn.Module,
            losses: List[Dict],
            validation_metrics: List[Dict],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_patience: int = 100,
            lr_monitor="training/total_loss",
            epsilon: float = 1e-8,
            beta1: float = 0.99,
            beta2: float = 0.999,
            amsgrad: bool = True,
            max_points_to_scatter: int = 100000,
            post_processing: callable = None
            ):
        """
        初始化 Model 类。

        Args:
            representation (nn.Module): 图表示网络实例 (例如, HamGNNConvE3)。
            output (nn.Module): 输出网络实例 (例如, HamGNNPlusPlusOut)。
            losses (List[Dict]): 损失函数配置列表。每个字典应包含 'metric', 'prediction', 'target', 'loss_weight' 等键。
            validation_metrics (List[Dict]): 验证指标配置列表。每个字典应包含 'metric', 'prediction', 'target' 等键。
            lr (float, optional): 初始学习率。默认为 1e-3。
            lr_decay (float, optional): 学习率调度器中用于降低学习率的因子。默认为 0.1。
            lr_patience (int, optional): 学习率调度器在降低学习率前的等待轮数。默认为 100。
            lr_monitor (str, optional): 学习率调度器监控的指标。默认为 "training/total_loss"。
            epsilon (float, optional): AdamW 优化器的 epsilon 参数。默认为 1e-8。
            beta1 (float, optional): AdamW 优化器的 beta1 参数。默认为 0.99。
            beta2 (float, optional): AdamW 优化器的 beta2 参数。默认为 0.999。
            amsgrad (bool, optional): 是否在 AdamW 优化器中使用 AMSGrad 变体。默认为 True。
            max_points_to_scatter (int, optional): 在验证/测试结束时，用于生成散点图的最大数据点数。默认为 100000。
            post_processing (callable, optional): 一个可选的后处理模块，用于在测试时进行需要梯度的特殊计算。默认为 None。
        """
        super().__init__()

        self.representation = representation
        self.output_module = output

        self.losses = losses
        self.metrics = validation_metrics

        # --- 优化器和学习率调度器参数 ---
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
        
        self.max_points_to_scatter = max_points_to_scatter
        # post_processing 用于计算某些依赖于梯度反向传播的物理量
        self.post_processing = post_processing

        # self.save_hyperparameters() # 保存超参数，便于从 checkpoint 加载

        # 检查输出模块是否需要计算关于位置的导数（例如，计算力）
        self.requires_dr = self.output_module.derivative

    def calculate_loss(self, batch, result, mode):
        r"""根据 `self.losses` 配置计算总损失。

        例如，对于一个均方误差 (MSE) 损失，其计算方式为：

        .. math::

           L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

        其中 :math:`y_i` 是真实值，:math:`\hat{y}_i` 是预测值。

        Args:
            batch (dict): 当前批次的数据，包含目标值。
            result (dict): 模型的预测输出。
            mode (str): 当前阶段的名称（'training', 'validation', or 'test'），用于日志记录。

        Returns:
            torch.Tensor: 计算得到的加权总损失。
        """
        loss = torch.tensor(0.0, device=self.device)
        # 遍历所有定义的损失函数
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]

            # 如果损失函数需要目标值
            if "target" in loss_dict.keys():
                pred = result[loss_dict["prediction"]]
                target = batch[loss_dict["target"]]
                loss_i = loss_fn(pred, target)
            # 如果损失函数只依赖于预测值（例如，正则化项）
            else:
                loss_i = loss_fn(result[loss_dict["prediction"]])
            
            # 将当前损失乘以其权重并累加到总损失中
            loss += loss_dict["loss_weight"] * loss_i

            # 获取损失函数的名称用于日志记录
            if hasattr(loss_fn, "name"):
                lossname = loss_fn.name
            else:
                lossname = type(loss_fn).__name__.split(".")[-1]

            # 记录单个损失项
            self.log(
                mode
                + "/"
                + lossname
                + "_"
                + loss_dict["prediction"],
                loss_i,
                on_step=False,
                on_epoch=True,
            )

        return loss

    def training_step(self, data, batch_idx):
        """
        执行单个训练步骤。

        Args:
            data (dict): 当前批次的训练数据。
            batch_idx (int): 当前批次的索引。

        Returns:
            torch.Tensor: 该批次的总训练损失。
        """
        self._enable_grads(data) # 如果需要，为位置启用梯度
        pred = self(data) # 前向传播
        loss = self.calculate_loss(data, pred, 'training') # 计算损失
        self.log("training/total_loss", loss, on_step=False, on_epoch=True) # 记录总损失
        # self.check_param() # 用于调试的辅助函数
        return loss

    def validation_step(self, data, batch_idx):
        """
        执行单个验证步骤。

        Args:
            data (dict): 当前批次的验证数据。
            batch_idx (int): 当前批次的索引。

        Returns:
            dict: 包含该批次预测值和目标值的字典，用于 `validation_epoch_end`。
        """
        # 在验证时，如果需要计算导数，需手动开启梯度计算
        if self.requires_dr:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        
        self._enable_grads(data)
        pred = self(data)
        val_loss = self.calculate_loss(
            data, pred, 'validation').detach().item()
        self.log("validation/total_loss", val_loss,
                 on_step=False, on_epoch=True)
        self.log_metrics(data, pred, 'validation')
        
        # 收集预测和目标值，用于 epoch 结束时进行可视化
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            outputs_pred[loss_dict["prediction"]] = pred[loss_dict["prediction"]].detach().cpu().numpy()  
            outputs_target[loss_dict["target"]] = data[loss_dict["target"]].detach().cpu().numpy()      
        return {'pred': outputs_pred, 'target': outputs_target}

    def validation_epoch_end(self, validation_step_outputs):
        """
        在验证的每个 epoch 结束时调用。

        该方法聚合所有验证批次的预测和目标，并生成散点图以可视化模型性能。

        Args:
            validation_step_outputs (List[dict]): 从 `validation_step` 返回的输出列表。
        """
        for loss_dict in self.losses:
            if "target" in loss_dict.keys():
                # 聚合来自所有验证批次的结果
                pred = np.concatenate([out['pred'][loss_dict["prediction"]]
                                 for out in validation_step_outputs])
                target = np.concatenate([out['target'][loss_dict["target"]]
                                    for out in validation_step_outputs])
                
                # 特殊处理复数类型数据，以便绘图
                if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                    if lossname.lower() == 'abs_mae': # 如果是绝对值误差，则取模
                        pred = np.absolute(pred)
                        target = np.absolute(target)
                    else: # 否则，将实部和虚部拼接成一个向量
                        pred = np.concatenate([pred.real, pred.imag], axis=-1)
                        target = np.concatenate([target.real, target.imag], axis=-1)
                
                # 控制要绘制的散点图的点数
                if pred.size > self.max_points_to_scatter:
                    random_state = np.random.RandomState(seed=42)
                    perm = list(random_state.permutation(np.arange(pred.size)))
                    pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
                    target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
                
                # 生成并记录散点图
                figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
                figname = 'PredVSTarget_' + loss_dict['prediction']
                self.logger.experiment.add_figure(
                    'validation/'+figname, figure, global_step=self.global_step)
            else:
                pass

    def test_step(self, data, batch_idx):
        """
        执行单个测试步骤。

        Args:
            data (dict): 当前批次的测试数据。
            batch_idx (int): 当前批次的索引。

        Returns:
            dict: 包含预测、目标和任何后处理值的字典。
        """
        if self.requires_dr:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self._enable_grads(data)
        
        # 如果定义了后处理步骤，则执行它
        if self.post_processing is not None:
            pred = self.post_processing(data)
            if type(self.post_processing).__name__.split(".")[-1].lower() == 'epc_output':
                proessed_values = {'epc_mat': pred['epc_mat'].detach().cpu().numpy()}
            else:
                raise NotImplementedError
        else:
            pred = self(data)
            proessed_values = None
            
        loss = self.calculate_loss(data, pred, 'test').detach().item()
        self.log("test/total_loss", loss, on_step=False, on_epoch=True)
        self.log_metrics(data, pred, "test") 
        
        # 收集预测和目标值
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            outputs_pred[loss_dict["prediction"]] = pred[loss_dict["prediction"]].detach().cpu().numpy()  
            outputs_target[loss_dict["target"]] = data[loss_dict["target"]].detach().cpu().numpy()      
        return {'pred': outputs_pred, 'target': outputs_target, 'processed_values': proessed_values}

    def test_epoch_end(self, test_step_outputs):
        """
        在测试的每个 epoch 结束时调用。

        该方法聚合所有测试批次的结果，将预测和目标保存到 .npy 文件，并生成最终的散点图。

        Args:
            test_step_outputs (List[dict]): 从 `test_step` 返回的输出列表。
        """
        for loss_dict in self.losses:
            if "target" in loss_dict.keys():
                # 聚合所有测试批次的结果
                pred = np.concatenate([out['pred'][loss_dict["prediction"]]
                                 for out in test_step_outputs])
                target = np.concatenate([out['target'][loss_dict["target"]]
                                    for out in test_step_outputs])
                
                if not os.path.exists(self.trainer.logger.log_dir):
                    os.makedirs(self.trainer.logger.log_dir)
                    
                # 将预测和目标数组保存到文件
                np.save(os.path.join(
                    self.trainer.logger.log_dir, 'prediction_'+loss_dict["prediction"]+'.npy'), pred)
                np.save(os.path.join(self.trainer.logger.log_dir,
                        'target_'+loss_dict["target"]+'.npy'), target)
                
                # --- 绘图逻辑 (与 validation_epoch_end 类似) ---
                if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                    if lossname.lower() == 'abs_mae':
                        pred = np.absolute(pred)
                        target = np.absolute(target)
                    else:
                        pred = np.concatenate([pred.real, pred.imag], axis=-1)
                        target = np.concatenate([target.real, target.imag], axis=-1)
                
                # 控制要绘制的散点图的点数
                if pred.size > self.max_points_to_scatter:
                    random_state = np.random.RandomState(seed=42)
                    perm = list(random_state.permutation(np.arange(pred.size)))
                    pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
                    target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
                    
                figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
                figname = 'PredVSTarget_' + loss_dict['prediction']
                self.logger.experiment.add_figure(
                    'test/'+figname, figure, global_step=self.global_step)
            else:
                pass
        
        # 如果有后处理结果，也将其保存
        if self.post_processing is not None:
            if type(self.post_processing).__name__.split(".")[-1].lower() == 'epc_output':
                processed_values = np.concatenate([out['processed_values']["epc_mat"]
                                        for out in test_step_outputs])
                np.save(os.path.join(
                    self.trainer.logger.log_dir, 'processed_values_'+'epc_mat'+'.npy'), processed_values)
            
    def forward(self, data):
        """
        定义模型的前向传播逻辑。

        Args:
            data (dict): 输入的批次数据。

        Returns:
            dict: 包含预测值的字典。
        """
        # torch.set_grad_enabled(True)
        self._enable_grads(data)
        representation = self.representation(data)
        pred = self.output_module(data, representation)
        return pred

    def log_metrics(self, batch, result, mode):
        """
        根据 `self.metrics` 配置计算并记录所有评估指标。

        Args:
            batch (dict): 当前批次的数据，包含目标值。
            result (dict): 模型的预测输出。
            mode (str): 当前阶段的名称（'validation' or 'test'）。
        """
        for metric_dict in self.metrics:
            loss_fn = metric_dict["metric"]

            if "target" in metric_dict.keys():
                pred = result[metric_dict["prediction"]]
                target = batch[metric_dict["target"]]
                loss_i = loss_fn(
                    pred, target
                ).detach().item()
            else:
                loss_i = loss_fn(
                    result[metric_dict["prediction"]]).detach().item()

            if hasattr(loss_fn, "name"):
                lossname = loss_fn.name
            else:
                lossname = type(loss_fn).__name__.split(".")[-1]

            self.log(
                mode
                + "/"
                + lossname
                + "_"
                + metric_dict["prediction"],
                loss_i,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        """
        配置模型的优化器和学习率调度器。

        Returns:
            tuple: 包含优化器列表和调度器配置列表的元组。
        """
        optimizer = opt.AdamW(self.parameters(), lr=self.lr, eps=self.epsilon, betas=(self.beta1, self.beta2), weight_decay=0.0, amsgrad=True)
        scheduler = {
            "scheduler": opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def _enable_grads(self, data):
        """
        一个辅助函数，用于在需要计算导数时，为原子位置启用梯度计算。
        """
        if self.requires_dr:
            data['pos'].requires_grad_()

    def check_param(self):
        """
        一个调试辅助函数，用于打印模型参数的名称和梯度信息。
        """
        for name, parms in self.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                  '-->grad_value:', parms.grad)