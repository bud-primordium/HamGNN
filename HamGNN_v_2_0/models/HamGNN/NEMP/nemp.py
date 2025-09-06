'''
节点等变消息传递（NEMP）v1 - 最小化实现版本

核心思想：
通过预聚合策略将计算复杂度从O(#edges)降低到O(#nodes)。首先使用径向嵌入通过MLP
生成边系数（Edge Coefficients, EC），然后用EC对源节点特征进行逐维加权，并通过
scatter-sum操作聚合到目标节点形成"虚拟节点"。最后仅需执行一次节点级的等变张量积
（Tensor Product, TP），减少计算量和GPU显存占用。
'''
from typing import Dict, Optional, List
import torch
from torch import nn
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet


class NEMPConvBlock(nn.Module):
    """NEMP最小化卷积块 - 通过节点级张量积实现等变消息传递

    输入数据格式：
        data['edge_index']: torch.LongTensor, 形状 [2, E]
            边的连接关系，第一行为源节点索引，第二行为目标节点索引
        data['node_features']: torch.Tensor, 形状 [N, irreps_in.dim]
            节点特征张量，N为节点数
        data['edge_embedding']: torch.Tensor, 形状 [E, S]
            边的径向标量嵌入，E为边数，S为嵌入维度
        data['edge_lengths']: torch.Tensor, 形状 [E] (可选)
            边长信息，用于截断函数

    输出：
        更新后的节点特征，同时原地修改data['node_features']
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        ec_channels: int = 64,
        irreps_mid: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        use_kan: bool = False,
        gate: str = 'none',
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)

        # 张量积的中间不可约表示，默认与输入相同以保证残差连接的稳定性
        self.irreps_mid = o3.Irreps(irreps_mid) if irreps_mid is not None else self.irreps_in

        self.radial_MLP = radial_MLP or [64, 64]
        self.use_kan = use_kan  # 预留KAN网络接口，当前版本未使用

        # 边系数(Edge Coefficients)生成器：将径向嵌入通过MLP映射到ec_channels维
        in_dim = self.irreps_edge_embed.num_irreps
        layers = [in_dim] + list(self.radial_MLP) + [ec_channels]
        self.ec_mlp = FullyConnectedNet(layers, torch.nn.functional.silu)

        # 线性投影层：将EC映射到节点特征维度，实现逐维门控机制
        self.ec_proj = nn.Linear(ec_channels, self.irreps_in.dim, bias=False)

        # 节点级等变张量积和线性投影层
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_mid,
            internal_weights=True, shared_weights=True
        )
        self.lin = o3.Linear(self.irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        # 可选的非线性门控机制
        gate = (gate or 'none').lower()
        if gate == 'norm':
            self.gate = nn.LayerNorm(self.irreps_out.dim)
        elif gate == 'tanh':
            self.gate = nn.Tanh()
        else:
            self.gate = nn.Identity()

        # 当输入输出不可约表示相同时，启用残差连接
        self.use_residual = (self.irreps_in == self.irreps_out)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = data['edge_index']  # [2, E]
        src, dst = edge_index[0], edge_index[1]
        node_feats = data['node_features']  # [N, D]
        edge_embed = data['edge_embedding']  # [E, S]

        N, D = node_feats.shape
        E = edge_embed.shape[0]

        # 步骤1: 通过径向嵌入生成边系数(EC)
        ec = self.ec_mlp(edge_embed)                 # [E, C_ec]
        gates = self.ec_proj(ec)                     # [E, D]

        # 步骤2: 使用EC对源节点特征加权，并聚合到目标节点形成虚拟节点
        msg = node_feats[src] * gates                # [E, D]
        virtual = scatter(msg, dst, dim=0, dim_size=N, reduce='sum')  # [N, D]

        # 步骤3: 执行节点级张量积和线性变换
        tp_out = self.tp(node_feats, virtual)        # [N, mid_dim]
        upd = self.lin(tp_out)                       # [N, D_out]
        upd = self.gate(upd)

        if self.use_residual:
            out = node_feats + upd
        else:
            out = upd

        data['node_features'] = out
        return out
