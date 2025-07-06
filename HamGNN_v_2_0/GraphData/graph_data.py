"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-07 20:44:01 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:24:33
 */
"""
"""该模块定义了 `graph_data_module` 类，用于处理和封装图数据。

`graph_data_module` 继承自 `pytorch_lightning.LightningDataModule`，
负责数据集的划分（训练、验证、测试集）、以及为每个部分创建对应的数据加载器（DataLoader）。
它支持从预定义的索引文件加载数据集划分，或按比例随机划分。
"""

import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Union, Callable
import numpy as np
from torch.utils.data import random_split, Subset
import os

"""
graph_data_module inherits pl.lightningDatamodule to implement the dataset class,
which divides the dataset and builds the dataset loader.  
"""


class graph_data_module(pl.LightningDataModule):
    """继承自 pl.LightningDataModule，用于管理图数据集。

    该类自动化了数据集的加载、划分和批处理过程，使其与 PyTorch Lightning 的
    训练流程无缝集成。
    """
    def __init__(self, dataset: Union[list, tuple, np.array] = None,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 batch_size: int = 300,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 split_file : str = None):
        """构造函数，初始化数据模块。

        Args:
            dataset (Union[list, tuple, np.array], optional): 
                包含所有数据样本的完整数据集，通常是 `torch_geometric.data.Data` 对象的列表。默认为 None。
            train_ratio (float, optional): 
                训练集所占的比例。默认为 0.6。
            val_ratio (float, optional): 
                验证集所占的比例。默认为 0.2。
            test_ratio (float, optional): 
                测试集所占的比例。默认为 0.2。
            batch_size (int, optional): 
                训练时每个批次的大小。默认为 300。
            val_batch_size (int, optional): 
                验证时每个批次的大小。如果为 None，则使用 `batch_size`。默认为 None。
            test_batch_size (int, optional): 
                测试时每个批次的大小。如果为 None，则使用 `val_batch_size`。默认为 None。
            split_file (str, optional): 
                一个 `.npz` 文件的路径，其中包含预先定义好的 'train_idx', 'val_idx', 'test_idx' 
                数据集划分索引。如果提供此文件，将忽略 `train_ratio`, `val_ratio`, `test_ratio`。
                默认为 None。
        """
        super(graph_data_module, self).__init__()
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.split_file = split_file
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size

    def setup(self, stage=None):
        """划分数据集为训练、验证和测试三部分。

        此方法是 PyTorch Lightning 工作流的一部分，在训练或测试开始时被调用。
        其行为逻辑如下：
        1. 如果 `self.split_file` 存在，则从该文件加载预定义的索引来划分数据集。
           这确保了实验的可复现性。
        2. 如果没有提供划分文件，则根据 `train_ratio`, `val_ratio`, `test_ratio` 
           对数据集进行随机划分。划分是基于固定的随机种子（42），以保证每次运行的划分结果一致。
        3. 根据 `stage` 参数（'fit' 或 'test'）来决定执行哪部分逻辑。

        Args:
            stage (str, optional): 
                当前所处的阶段，由 Lightning 自动传入。可以是 'fit'、'test' 等。默认为 None。
        """
        # 如果提供了预先划分好的索引文件，则直接加载
        if self.split_file is not None and os.path.exists(self.split_file):
            print(f"根据指定的划分文件 {self.split_file} 来划分数据集。")
            S = np.load(self.split_file)
            train_idx = S["train_idx"].tolist()
            val_idx = S["val_idx"].tolist()
            test_idx = S["test_idx"].tolist()
            self.train_data = Subset(self.dataset, indices=train_idx)
            self.val_data = Subset(self.dataset, indices=val_idx)
            self.test_data = Subset(self.dataset, indices=test_idx)
        else:
            # 在 'fit' 阶段或未指定阶段时，执行随机划分
            if stage == 'fit' or stage is None:
                # 使用固定种子以保证划分的可复现性
                random_state = np.random.RandomState(seed=42)
                length = len(self.dataset)
                num_train = round(self.train_ratio * length)
                num_val = round(self.val_ratio * length)
                num_test = round(self.test_ratio * length)
                # 生成随机排列的索引
                perm = list(random_state.permutation(np.arange(length)))
                train_idx = perm[:num_train]
                val_idx = perm[num_train:num_train+num_val]
                test_idx = perm[-num_test:]
                self.train_data = [self.dataset[i] for i in train_idx]
                self.val_data = [self.dataset[i] for i in val_idx]
                self.test_data = [self.dataset[i] for i in test_idx]
            # 在 'test' 阶段，使用整个数据集作为测试集
            if stage == 'test':
                self.test_data = self.dataset

    def train_dataloader(self) -> DataLoader:
        """创建并返回训练数据加载器。

        Returns:
            DataLoader: 用于训练集的数据加载器实例。
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        """创建并返回验证数据加载器。

        Returns:
            DataLoader: 用于验证集的数据加载器实例。
        """
        return DataLoader(self.val_data, batch_size=self.val_batch_size, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        """创建并返回测试数据加载器。

        Returns:
            DataLoader: 用于测试集的数据加载器实例。
        """
        return DataLoader(self.test_data, batch_size=self.test_batch_size, pin_memory=True)

    # ------------------------------------------------------------------
    # 兼容旧版本 PyTorch-Lightning 的变通方案
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> bool:
        """优雅地处理某些旧版 PyTorch-Lightning 试图在 DataModule 实例上
        访问的特殊私有属性。

        例如：``_has_setup_TrainerFn.TESTING`` 或
        ``_has_setup_TrainerFn.FITTING``。

        在标准的 Python 代码中，包含点（.）的属性名称是无效的，因此它们永远
        不会存在于对象上。尝试访问这类属性会导致 ``AttributeError`` 并使程序崩溃。

        此方法捕获这类访问，动态地创建该属性并返回 ``False``，
        这样 Lightning 就会认为相关的 setup 钩子函数尚未运行。
        这可以完全避免程序崩溃，而无需修改 Lightning 库本身或依赖于某个特定版本。

        Args:
            item (str): 尝试访问的属性名称。

        Returns:
            bool: 如果属性是 Lightning 的内部状态标志，则返回 `False`。
        
        Raises:
            AttributeError: 如果访问的不是预期的内部标志属性。
        """
        if item.startswith("_has_setup_TrainerFn.") or item.startswith("_has_teardown_TrainerFn."):
            # 动态创建标志并默认为 False
            super().__setattr__(item, False)
            return False
        # 回退到默认行为
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{item}'")
