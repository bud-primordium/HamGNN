"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-29 15:03:46 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:45:04
 */
"""
"""该模块负责处理项目的配置信息。

它定义了所有参数的默认值，并提供了一个 `read_config` 函数，该函数能够：
1. 从一个指定的 YAML 文件中加载用户配置。
2. 将用户配置与预设的默认配置进行合并。
3. 将最终的配置转换为 `easydict.EasyDict` 对象，方便以属性方式访问。
4. 对损失函数和评估指标的配置进行预处理。
"""
import yaml
from easydict import EasyDict
from ..models.utils import get_activation, parse_metric_func
import pprint

# ==============================================================================
# 默认配置参数
# ==============================================================================
config_default = dict()

# ------------------------------------------------------------------------------
# 训练设置 (setup)
# ------------------------------------------------------------------------------
config_default_setup = dict()
config_default_setup['GNN_Net'] = 'Edge_GNN'  # 使用的 GNN 网络名称
config_default_setup['property'] = 'scalar_per_atom'  # 要预测的属性类型
config_default_setup['num_gpus'] = [1]  # 使用的 GPU 数量
config_default_setup['accelerator'] = None # 加速器类型, e.g., 'dp', 'ddp', 'ddp_cpu'
config_default_setup['precision'] = 32  # 训练精度 (16, 32, 64)
config_default_setup['stage'] = 'fit'  # 当前阶段 ('fit', 'test', 'predict')
config_default_setup['resume'] = False  # 是否从上次的 checkpoint 断点续训
config_default_setup['load_from_checkpoint'] = False  # 是否从指定的 checkpoint 加载模型
config_default_setup['checkpoint_path'] = './'  # checkpoint 文件路径
config_default_setup['ignore_warnings'] = False  # 是否忽略警告信息
config_default_setup['l_minus_mean'] = False  # 是否在计算损失前减去目标均值
config_default['setup'] = config_default_setup

# ------------------------------------------------------------------------------
# 数据集参数 (dataset_params)
# ------------------------------------------------------------------------------
config_default_dataset = dict()
config_default_dataset['database_type'] = 'db'  # 数据源类型 ('db' 或 'csv')
config_default_dataset['train_ratio'] = 0.6  # 训练集比例
config_default_dataset['val_ratio'] = 0.2  # 验证集比例
config_default_dataset['test_ratio'] = 0.2  # 测试集比例
config_default_dataset['batch_size'] = 200  # 批处理大小
config_default_dataset['split_file'] = None  # 数据集划分索引文件的路径
config_default_dataset['radius'] = 6.0  # 构建图时近邻的截断半径
config_default_dataset['max_num_nbr'] = 32  # 每个原子的最大近邻数
config_default_dataset['graph_data_path'] = './graph_data'  # 预处理图数据的保存路径

# --- 数据库类型为 'db' 时的参数 ---
config_default_db_params = dict()
config_default_db_params['db_path'] = './'  # 数据库文件路径
config_default_db_params['property_list'] = ['energy','hamiltonian']  # 需要从数据库中提取的属性列表
config_default_dataset['db_params'] = config_default_db_params

# --- 数据库类型为 'csv' 时的参数 ---
config_default_csv_params = dict()
config_default_csv_params['crystal_path'] = 'crystals'  # 晶体结构文件所在目录
config_default_csv_params['file_type'] = 'poscar'  # 晶体文件类型 ('poscar', 'cif')
config_default_csv_params['id_prop_path'] = './'  # id_prop.csv 文件所在目录
config_default_csv_params['rank_tensor'] = 0  # 预测张量的阶数
config_default_csv_params['l_pred_atomwise_tensor'] = True  # 是否预测原子级别的张量
config_default_csv_params['l_pred_crystal_tensor'] = False  # 是否预测晶体级别的张量
config_default_dataset['csv_params'] = config_default_csv_params

config_default['dataset_params'] = config_default_dataset

# ------------------------------------------------------------------------------
# 优化器参数 (optim_params)
# ------------------------------------------------------------------------------
config_default_optimizer = dict()
config_default_optimizer['lr'] = 0.01  # 初始学习率
config_default_optimizer['lr_decay'] = 0.5  # 学习率衰减因子
config_default_optimizer['lr_patience'] = 5  # 学习率衰减的等待轮数
config_default_optimizer['gradient_clip_val'] = 0.0  # 梯度裁剪阈值 (0表示不裁剪)
config_default_optimizer['stop_patience'] = 30  # 早停的等待轮数
config_default_optimizer['min_epochs'] = 100  # 最小训练轮数
config_default_optimizer['max_epochs'] = 500  # 最大训练轮数
config_default['optim_params'] = config_default_optimizer

# ------------------------------------------------------------------------------
# 损失与评估指标 (losses_metrics)
# ------------------------------------------------------------------------------
config_default_metric = dict()
config_default_metric['losses'] = [{'metric': 'mse', 'prediction': 'energy',  'target': 'energy', 'loss_weight': 1.0}, {
    'metric': 'cosine_similarity', 'prediction': 'energy',  'target': 'energy', 'loss_weight': 0.0}]
config_default_metric['metrics'] = [{'metric': 'mae', 'prediction': 'energy',  'target': 'energy'}, {
    'metric': 'cosine_similarity', 'prediction': 'energy',  'target': 'energy'}]
config_default['losses_metrics'] = config_default_metric

# ------------------------------------------------------------------------------
# 性能分析器参数 (profiler_params)
# ------------------------------------------------------------------------------
config_default_profiler = dict()
config_default_profiler['train_dir'] = 'train_data'  # 训练日志和输出的保存目录
config_default_profiler['progress_bar_refresh_rat'] = 1  # 进度条刷新频率
config_default['profiler_params'] = config_default_profiler

# ------------------------------------------------------------------------------
# 表示层网络 (representation_nets)
# ------------------------------------------------------------------------------
config_default_representation_nets = dict()
# 此处可添加表示层网络的默认参数
config_default['representation_nets'] = config_default_representation_nets

# ------------------------------------------------------------------------------
# 输出层网络 (output_nets)
# ------------------------------------------------------------------------------
config_default_output_nets = dict()
config_default_output_nets['output_module'] = 'HamGNN_out'  # 使用的输出模块名称

# --- HamGNN 输出模块 (HamGNN_out) 的特定参数 ---
config_default_HamGNN_out = dict()
config_default_HamGNN_out['nao_max'] = 14  # 最大原子轨道数
config_default_HamGNN_out['return_forces'] = False  # 是否计算并返回力
config_default_HamGNN_out['create_graph'] = False  # 是否在计算图中创建额外的边
config_default_HamGNN_out['ham_type'] = 'openmx'  # 哈密顿量类型
config_default_HamGNN_out['ham_only'] = True  # 是否只输出哈密顿量
config_default_HamGNN_out['irreps_in_node'] = ''  # 节点输入的不可约表示
config_default_HamGNN_out['irreps_in_edge'] = ''  # 边输入的不可约表示
config_default_HamGNN_out['irreps_in_triplet'] = ''  # 三元组输入的不可约表示
config_default_HamGNN_out['include_triplet'] = False  # 是否包含三元组信息
config_default_HamGNN_out['symmetrize'] = True  # 是否对哈密顿量进行对称化处理
config_default_HamGNN_out['calculate_band_energy'] = False # 是否计算能带能量
config_default_HamGNN_out['num_k'] = 5  # k点数量
config_default_HamGNN_out['soc_switch'] = False  # 是否开启自旋轨道耦合(SOC)
config_default_HamGNN_out['nonlinearity_type'] = 'gate'  # 非线性激活函数类型
config_default_HamGNN_out['band_num_control'] = 6  # 能带数量控制
config_default_HamGNN_out['k_path'] = None  # k点路径
config_default_HamGNN_out['spin_constrained'] = False  # 是否进行自旋约束
config_default_HamGNN_out['collinear_spin'] = False  # 是否为共线自旋计算
config_default_HamGNN_out['minMagneticMoment'] = 0.5  # 最小磁矩
config_default_output_nets['HamGNN_out'] = config_default_HamGNN_out

config_default['output_nets'] = config_default_output_nets


def read_config(config_file_name: str = 'config_default.yaml', config_default=config_default):
    """从 YAML 文件读取配置，并与默认配置合并。

    Args:
        config_file_name (str, optional): 
            用户指定的配置文件路径。默认为 'config_default.yaml'。
        config_default (dict, optional): 
            包含所有默认参数的字典。默认为本模块中定义的 `config_default`。

    Returns:
        easydict.EasyDict: 
            一个 EasyDict 对象，包含了合并后的最终配置，可以通过属性点号 `.`
            方便地访问各级配置项。
    """
    with open(config_file_name, encoding='utf-8') as rstream:
        data = yaml.load(rstream, yaml.SafeLoader)
    # 遍历从文件中加载的顶层键（如 'setup', 'dataset_params'）
    for key in data.keys():
        # 用文件中的配置更新对应的默认配置字典
        config_default[key].update(data[key])
    
    # 将最终的字典转换为 EasyDict 对象
    config = EasyDict(config_default)
    
    # 对损失和指标函数进行特殊解析，将其从字符串转换为实际的函数对象
    config.losses_metrics.losses = parse_metric_func(config.losses_metrics.losses)
    config.losses_metrics.metrics = parse_metric_func(config.losses_metrics.metrics)
    
    return config
