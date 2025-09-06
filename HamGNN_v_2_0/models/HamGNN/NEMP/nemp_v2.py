'''
NEMP v2 - 方向环境增强版

核心改进：
在v1的基础上引入方向环境分支，通过双分支架构补偿标量聚合丢失的几何信息。
- 标量分支：与v1相同，使用EC加权机制进行标量消息传递
- 方向分支：利用边的球谐属性(edge_attrs)，通过径向权重s(r)加权后聚合为
  节点级方向环境(node_env)，然后执行节点特征与方向环境的张量积
- 两个分支的输出相加后进行线性投影和残差更新
'''
from typing import Dict, Optional, List
import torch
from torch import nn
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet


class NEMPDirectionalBlock(nn.Module):
    """NEMP方向环境增强块 - 通过双分支架构实现高效等变消息传递
    
    参数：
        irreps_in: 输入节点特征的不可约表示
        irreps_out: 输出节点特征的不可约表示
        irreps_edge_attrs: 边球谐属性的不可约表示
        irreps_edge_embed: 边径向嵌入的不可约表示
        ec_channels: 边系数通道数，默认96
        irreps_mid: 张量积中间表示，默认与输入相同
        radial_MLP: 标量分支的径向MLP层级
        env_scalar_mlp: 方向分支的径向MLP层级
        gate: 门控类型 ('none', 'norm', 'tanh')
        degree_norm: 是否使用度归一化，默认True
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        ec_channels: int = 96,
        irreps_mid: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        env_scalar_mlp: Optional[List[int]] = None,
        gate: str = 'none',
        degree_norm: bool = True,
    ) -> None:
        super().__init__()

        # Irreps
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_mid = o3.Irreps(irreps_mid) if irreps_mid is not None else self.irreps_in

        # Hyper params
        self.radial_MLP = list(radial_MLP or [64, 64])
        self.env_scalar_mlp = list(env_scalar_mlp or [32, 32])
        self.degree_norm = degree_norm

        # -------- 标量分支（与 v1 一致） --------
        in_dim = self.irreps_edge_embed.num_irreps
        self.ec_mlp = FullyConnectedNet([in_dim] + self.radial_MLP + [ec_channels], torch.nn.functional.silu)
        self.ec_proj = nn.Linear(ec_channels, self.irreps_in.dim, bias=False)

        # 节点级 TP（标量分支）
        self.tp_scalar = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_mid,
            internal_weights=True, shared_weights=True
        )

        # -------- 方向环境分支（新增） --------
        # 仅对每条边的 edge_attrs 乘一个径向标量 s(r)（轻量 MLP 仅输出 1 通道），
        # 再按目标节点聚合，得到节点级方向环境 node_env（Irreps 与 edge_attrs 相同）。
        self.env_s_mlp = FullyConnectedNet([in_dim] + self.env_scalar_mlp + [1], torch.nn.functional.silu)

        # 节点级 TP（方向分支）： node ⊗ node_env -> mid
        self.tp_dir = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_edge_attrs, self.irreps_mid,
            internal_weights=True, shared_weights=True
        )

        # -------- 合并与回写 --------
        self.lin = o3.Linear(self.irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        g = (gate or 'none').lower()
        if g == 'norm':
            self.gate = nn.LayerNorm(self.irreps_out.dim)
        elif g == 'tanh':
            self.gate = nn.Tanh()
        else:
            self.gate = nn.Identity()

        self.use_residual = (self.irreps_in == self.irreps_out)

    def _deg_norm(self, x: torch.Tensor, dst: torch.Tensor, N: int, eps: float = 1e-8) -> torch.Tensor:
        if not self.degree_norm:
            return scatter(x, dst, dim=0, dim_size=N, reduce='sum')
        ones = torch.ones_like(dst, dtype=x.dtype)
        deg = scatter(ones, dst, dim=0, dim_size=N, reduce='sum').clamp_min(eps).unsqueeze(-1)
        summed = scatter(x, dst, dim=0, dim_size=N, reduce='sum')
        return summed / deg

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = data['edge_index']
        src, dst = edge_index[0], edge_index[1]
        node_feats = data['node_features']  # [N, D]
        edge_embed = data['edge_embedding']  # [E, S]
        edge_attrs = data['edge_attrs']      # [E, ... Irreps edge]

        N, D = node_feats.shape

        # -------- 标量分支：EC×源节点特征 -> 聚合为虚拟节点 --------
        ec = self.ec_mlp(edge_embed)        # [E, C_ec]
        gates = self.ec_proj(ec)            # [E, D]
        msg_scalar = node_feats[src] * gates
        virtual_scalar = self._deg_norm(msg_scalar, dst, N)  # [N, D]

        # 节点级 TP（标量分支）
        up_scalar = self.tp_scalar(node_feats, virtual_scalar)  # [N, mid]

        # -------- 方向环境分支：s(r)*edge_attrs -> 聚合为 node_env --------
        s = self.env_s_mlp(edge_embed)  # [E, 1]
        # edge_attrs 是 Irreps 张量，逐元素缩放广播到最后维
        weighted_attrs = edge_attrs * s
        node_env = self._deg_norm(weighted_attrs, dst, N)  # [N, Irreps(edge_attrs).dim]

        # 节点级 TP（方向分支）
        up_dir = self.tp_dir(node_feats, node_env)  # [N, mid]

        # -------- 合并与回写 --------
        upd = self.lin(up_scalar + up_dir)
        upd = self.gate(upd)
        out = node_feats + upd if self.use_residual else upd
        data['node_features'] = out
        return out

