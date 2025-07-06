"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-07 20:30:29 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 15:52:53
 */
"""
"""该模块负责解析晶体结构文件（如 CIF, POSCAR），并将其转换为图数据格式。

主要功能包括：
1.  从晶体文件中读取原子结构信息。
2.  为原子类型构建 one-hot 编码表示。
3.  根据指定的截断半径（cutoff radius）寻找近邻原子，构建图的边。
4.  解析包含目标属性的 CSV 文件。
5.  将所有信息封装成 `torch_geometric.data.Data` 对象，并保存为 `.npz` 文件。
"""
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
import os
import csv
import glob, json


def cal_shfit_vec(image: np.array = None, lattice: np.array = None):
    """计算周期性边界条件下，近邻原子相对于中心原子的偏移向量。

    在晶体中，一个原子的近邻可能位于相邻的晶胞中。这个函数通过晶格矢量
    和周期性偏移镜像（image）来计算这种跨晶胞的实际位移。

    Args:
        image (np.array): 
            形状为 `(3,)` 的整数数组，表示近邻原子所在的晶胞相对于
            中心原子所在晶胞的周期性偏移（例如 `[-1, 0, 1]`）。
        lattice (np.array): 
            形状为 `(3, 3)` 的数组，表示晶胞的三个晶格矢量。

    Returns:
        np.array: 形状为 `(3,)` 的实际偏移向量。
    """
    return np.sum(image[:, None]*lattice, axis=0)

def build_config(config):
    """扫描所有晶体文件，构建原子类型的 one-hot 编码配置。

    该函数会遍历指定路径下的所有晶体结构文件，统计出数据集中出现的所有
    原子类型（以原子序数 Z 表示），然后为每种原子创建一个 one-hot 编码向量。
    最终的配置信息将保存为一个 JSON 文件。

    Args:
        config (easydict.EasyDict): 
            项目的主配置对象，需要包含以下路径信息：
            
            - `dataset_params.csv_params.crystal_path`: 晶体文件所在目录。
            - `dataset_params.csv_params.file_type`: 晶体文件类型 ('poscar' 或 'cif')。
            - `dataset_params.graph_data_path`: one-hot 配置文件 `config_onehot.json` 的保存路径。

    Returns:
        dict: 
            一个包含原子序数列表和对应 one-hot 编码矩阵的字典。
            例如： `{'atomic_numbers': [1, 6, 8], 'node_vectors': [[1,0,0], [0,1,0], [0,0,1]]}`
    """
    crystal_path = config.dataset_params.csv_params.crystal_path
    file_type = config.dataset_params.csv_params.file_type
    if file_type.lower() == 'poscar':
        file_extension = '.vasp'
    elif file_type.lower() == 'cif':
        file_extension = '.cif'
    else:
        # 如果文件类型不被支持，则打印错误信息。
        print(f'文件类型: {file_type} 暂不支持!')
    config_path = config.dataset_params.graph_data_path
    config_path = os.path.join(config_path, 'config_onehot.json')
    atoms=[]
    all_files = sorted(glob.glob(os.path.join(crystal_path,'*'+file_extension)))
    for path in tqdm(all_files):
        crystal = Structure.from_file(path)
        atoms += list(crystal.atomic_numbers)
    # 统计所有出现过的原子序数并去重
    unique_z = np.unique(atoms)
    num_z = len(unique_z)
    print('数据集中的原子种类数量:', num_z)
    print('最小原子序数:', np.min(unique_z))
    print('最大原子序数:', np.max(unique_z))
    # 构建配置文件字典
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot 编码矩阵
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def get_init_atomfea(config:dict=None, crystal:Structure=None):
    """根据 one-hot 配置文件为晶体中的每个原子生成初始特征向量。

    Args:
        config (dict): 
            由 `build_config` 生成的 one-hot 配置字典，包含 'atomic_numbers' 和 'node_vectors'。
        crystal (Structure): 
            一个 `pymatgen.core.structure.Structure` 对象，代表单个晶体结构。

    Returns:
        np.ndarray: 
            形状为 `(N, D)` 的数组，其中 N 是晶体中的原子数，D 是 one-hot 向量的维度。
            每一行是对应原子的 one-hot 特征。
    """
    atoms=crystal.atomic_numbers
    atomnum=config['atomic_numbers']
    # 创建从原子序数到 one-hot 索引的映射
    z_dict = {z:i for i, z in enumerate(atomnum)}
    one_hotvec = np.array(config["node_vectors"])
    # 为晶体中的每个原子提取对应的 one-hot 向量
    atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])
    return atom_fea

def cif_parse(config):
    """解析所有晶体文件和属性文件，生成并保存图数据。

    这是数据预处理的核心函数。它执行以下步骤：

    1.  确定晶体文件的类型和路径。
    2.  加载或创建原子 one-hot 编码配置。
    3.  从 `id_prop.csv` 文件中读取每个晶体ID及其对应的目标属性。
    4.  遍历每个晶体ID，执行以下操作：

        a.  读取晶体结构。
        b.  生成原子特征（node_attr）。
        c.  寻找近邻，构建边列表（edge_index）和周期性偏移（nbr_shift）。
        d.  将所有信息（节点、边、属性等）组装成 `torch_geometric.data.Data` 对象。
    
    5.  将所有图数据存储在一个字典中，并使用 `np.savez` 保存为压缩的 `.npz` 文件。

    Args:
        config (easydict.EasyDict): 
            项目的主配置对象，提供了所有必需的参数，例如文件路径、截断半径、
            最大近邻数以及目标属性的格式等。
    """
    crystal_path = config.dataset_params.csv_params.crystal_path
    id_prop_path = config.dataset_params.csv_params.id_prop_path
    graph_data_path = config.dataset_params.graph_data_path
    radius = config.dataset_params.radius
    max_num_nbr = config.dataset_params.max_num_nbr
    l_pred_atomwise_tensor = config.setup.csv_params.l_pred_atomwise_tensor
    l_pred_crystal_tensor = config.setup.csv_params.l_pred_crystal_tensor
    rank_tensor = config.dataset_params.csv_params.rank_tensor

    file_type = config.dataset_params.csv_params.file_type
    if file_type.lower() == 'poscar':
        file_extension = '.vasp'
    elif file_type.lower() == 'cif':
        file_extension = '.cif'
    else:
        # 如果文件类型不被支持，则打印错误信息。
        print(f'文件类型: {file_type} 暂不支持!')

    # 加载（如果存在）或构建 one-hot 编码配置文件
    # 注意：这个功能在未来版本中可能会被弃用
    config_onehot_file_path = os.path.join(graph_data_path, "config_onehot.json")
    if os.path.exists(config_onehot_file_path):
        config_onehot = json.load(open(config_onehot_file_path))
    else:
        config_onehot = build_config(config)

    assert os.path.exists(id_prop_path), 'id_prop.csv 所在的路径不存在!'
    id_prop_file = os.path.join(id_prop_path, 'id_prop.csv')
    assert os.path.exists(id_prop_file), 'id_prop.csv 文件不存在!'

    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    cif_ids = [id_prop[0] for id_prop in id_prop_data]
    # 解析 id_prop.csv 文件中的目标属性
    if l_pred_atomwise_tensor or l_pred_crystal_tensor:
        # 处理张量类型的目标属性（原子级别或晶体级别）
        targets = []
        length_tensor = 3**rank_tensor
        for idx in range(len(id_prop_data)):
            target_idx = list(map(lambda x: float(x), id_prop_data[idx][1:]))
            target_idx = np.array(target_idx).reshape(-1, length_tensor) # 形状: (N_atom, length)
            targets.append(target_idx) 
    else:
        # 处理标量类型的目标属性
        targets = [float(id_prop[1]) for id_prop in id_prop_data]

    pbar_cif_ids = tqdm(cif_ids)
    graphs = dict()
    for i, cif_id in enumerate(pbar_cif_ids):
        #pbar_cif_ids.set_description("Processing %s" % cif_id) # 设置tqdm的描述
        cif_path = os.path.join(crystal_path, cif_id+file_extension)
        crystal = Structure.from_file(cif_path)

        # 获取原子的 one-hot 特征
        node_attr = get_init_atomfea(config_onehot, crystal)

        # 获取每个原子的所有近邻（在指定半径内），并按距离排序
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        edge_src_index, edge_tar_index, nbr_shift, nbr_counts = [], [], [], []
        for ni, nbr in enumerate(all_nbrs):
            # 如果近邻数小于最大限制，则全部保留
            if len(nbr) < max_num_nbr:
                nbr_counts.append(len(nbr))
                edge_src_index += [ni]*len(nbr)
                edge_tar_index += list(map(lambda x: x[2], nbr))
                nbr_shift += list(map(lambda x: cal_shfit_vec(
                    np.array(x[3], dtype=np.float32), x.lattice.matrix), nbr))
            else:
                # 如果近邻数超过最大限制，则只保留最近的 max_num_nbr 个
                nbr_counts.append(max_num_nbr)
                edge_src_index += [ni]*max_num_nbr
                edge_tar_index += list(map(lambda x: x[2], nbr[:max_num_nbr]))
                nbr_shift += list(map(lambda x: cal_shfit_vec(
                    np.array(x[3], dtype=np.float32), x.lattice.matrix), nbr[:max_num_nbr]))
        
        edge_index = [edge_src_index, edge_tar_index]  # 形状: (2, nedges)
            
        if l_pred_atomwise_tensor or l_pred_crystal_tensor:
            y_label = targets[i]
        else:
            y_label = [targets[i]]

        graphs[cif_id] = Data(z=torch.LongTensor(crystal.atomic_numbers),
                              node_attr=torch.FloatTensor(node_attr),
                              y=torch.FloatTensor(y_label),
                              pos=torch.FloatTensor(crystal.cart_coords),
                              node_counts=torch.LongTensor([len(crystal)]),
                              nbr_counts=torch.LongTensor(nbr_counts),
                              edge_index=torch.LongTensor(edge_index),
                              nbr_shift=torch.FloatTensor(nbr_shift))

    graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    np.savez(graph_data_path, graph=graphs)

if __name__ == '__main__':
    # 当该脚本作为主程序运行时，读取配置文件并执行解析
    from input.config_parsing import read_config

    config = read_config(config_file_name='config.yaml')
    cif_parse(config)