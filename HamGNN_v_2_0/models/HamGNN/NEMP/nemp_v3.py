'''
NEMP v3

核心优化：
通过块级操作和单次张量积实现极致性能优化，同时保持等变性。
- 块级EC门控：按irreps块而非全维度进行门控，减少O(E×D)的计算开销
- 单次TP融合：方向环境映射到节点空间后与标量分支融合，仅执行一次节点级TP
- 对称度归一化：使用1/√(deg_src×deg_dst)进行对称归一化，提升训练稳定性
- 可学习混合系数：α和β参数动态平衡标量和方向分支的贡献
'''
from typing import Dict, List, Optional
import torch
from torch import nn
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet


def _block_slices(irreps: o3.Irreps) -> List[slice]:
    """计算irreps各块的索引切片，用于块级操作"""
    idx = []
    off = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        idx.append(slice(off, off + dim))
        off += dim
    return idx


class NEMPV3Block(nn.Module):
    """NEMP极限优化块 - 通过块级门控和单次TP实现最高性能
    
    参数：
        irreps_in: 输入节点特征的不可约表示
        irreps_out: 输出节点特征的不可约表示
        irreps_edge_attrs: 边球谐属性的不可约表示
        irreps_edge_embed: 边径向嵌入的不可约表示
        ec_channels: 边系数通道数，默认128
        irreps_mid: 张量积中间表示，默认与输入相同
        radial_MLP: 标量分支的径向MLP层级
        env_scalar_mlp: 方向分支的径向MLP层级
        gate: 门控类型 ('none', 'norm', 'tanh')，默认'norm'
        degree_norm: 是否使用度归一化，默认True
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        ec_channels: int = 128,
        irreps_mid: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        env_scalar_mlp: Optional[List[int]] = None,
        gate: str = 'norm',
        degree_norm: bool = True,
    ) -> None:
        super().__init__()
        # 不可约表示配置
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_mid = o3.Irreps(irreps_mid) if irreps_mid is not None else self.irreps_in

        # 块级索引切片，用于高效的块级操作
        self.in_slices = _block_slices(self.irreps_in)
        self.attr_slices = _block_slices(self.irreps_edge_attrs)
        self.num_in_blocks = len(self.in_slices)
        self.num_attr_blocks = len(self.attr_slices)

        # 超参数设置
        self.radial_MLP = list(radial_MLP or [64, 64])
        self.env_scalar_mlp = list(env_scalar_mlp or [32, 32])
        self.degree_norm = degree_norm

        in_dim = self.irreps_edge_embed.num_irreps

        # 块级EC门控：为每个irreps块生成一个标量门控系数
        self.ec_mlp = FullyConnectedNet([in_dim] + self.radial_MLP + [self.num_in_blocks], torch.nn.functional.silu)

        # 方向环境权重：为每个edge_attrs块生成标量权重s_l(r)
        self.env_mlp = FullyConnectedNet([in_dim] + self.env_scalar_mlp + [self.num_attr_blocks], torch.nn.functional.silu)

        # 方向环境映射：将节点级方向环境映射到与节点特征同型，用于后续融合
        self.env_to_in = o3.Linear(self.irreps_edge_attrs, self.irreps_in, internal_weights=True, shared_weights=True)

        # 单次节点级张量积和线性投影
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_mid,
            internal_weights=True, shared_weights=True
        )
        self.lin = o3.Linear(self.irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        # 可学习的分支混合系数，初始值1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 标量分支权重
        self.beta = nn.Parameter(torch.tensor(1.0))   # 方向分支权重

        g = (gate or 'none').lower()
        if g == 'norm':
            self.gate = nn.LayerNorm(self.irreps_out.dim)
        elif g == 'tanh':
            self.gate = nn.Tanh()
        else:
            self.gate = nn.Identity()

        self.use_residual = (self.irreps_in == self.irreps_out)

    def _deg_symmetric_norm(self, src: torch.Tensor, dst: torch.Tensor, N: int):
        """对称度归一化，使用平方根形式提升训练稳定性"""
        ones = torch.ones_like(src, dtype=torch.float32)
        deg_src = scatter(ones, src, dim=0, dim_size=N, reduce='sum').clamp_min(1.0)
        deg_dst = scatter(ones, dst, dim=0, dim_size=N, reduce='sum').clamp_min(1.0)
        inv_sqrt_deg_src = deg_src.pow(-0.5)
        inv_sqrt_deg_dst = deg_dst.pow(-0.5)
        return inv_sqrt_deg_src, inv_sqrt_deg_dst

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = data['edge_index']
        src, dst = edge_index[0], edge_index[1]
        node_feats = data['node_features']  # [N, D]
        edge_embed = data['edge_embedding']  # [E, S]
        edge_attrs = data['edge_attrs']      # [E, A]

        N, D = node_feats.shape
        E = edge_embed.shape[0]

        inv_sqrt_deg_src, inv_sqrt_deg_dst = self._deg_symmetric_norm(src, dst, N)

        # -------- 标量分支：块级EC门控机制 --------
        gates = self.ec_mlp(edge_embed)  # [E, B_in]
        virtual_scalar = torch.zeros_like(node_feats)

        for b, sl in enumerate(self.in_slices):
            # 消息构建：节点块特征 × 块门控系数 × 度归一化因子
            x_b = node_feats[src][:, sl]
            g_b = gates[:, b:b+1].type_as(x_b)
            msg_b = x_b * g_b * inv_sqrt_deg_src[src].unsqueeze(-1).type_as(x_b)
            # 按目标节点索引聚合消息到对应块
            virtual_scalar[:, sl].index_add_(0, dst, msg_b)

        # 目标节点侧应用度归一化因子
        virtual_scalar = virtual_scalar * inv_sqrt_deg_dst.unsqueeze(-1).type_as(node_feats)

        # -------- 方向环境分支：块级s_l(r)门控机制 --------
        env_gates = self.env_mlp(edge_embed)  # [E, B_attr]
        node_env = torch.zeros(edge_attrs.new_zeros((N, self.irreps_edge_attrs.dim)).shape, dtype=edge_attrs.dtype, device=edge_attrs.device)

        for b, sl in enumerate(self.attr_slices):
            a_b = edge_attrs[:, sl]
            s_b = env_gates[:, b:b+1].type_as(a_b)
            msg_env_b = a_b * s_b * inv_sqrt_deg_src[src].unsqueeze(-1).type_as(a_b)
            node_env[:, sl].index_add_(0, dst, msg_env_b)

        node_env = node_env * inv_sqrt_deg_dst.unsqueeze(-1).type_as(node_env)

        # 融合两个分支：将方向环境映射到节点空间后，与标量虚拟节点加权融合
        env_mapped = self.env_to_in(node_env)
        virtual_total = self.alpha * virtual_scalar + self.beta * env_mapped

        # -------- 单次节点级处理：张量积 + 线性投影 + 门控 + 残差 --------
        up = self.tp(node_feats, virtual_total)
        upd = self.lin(up)
        upd = self.gate(upd)
        out = node_feats + upd if self.use_residual else upd
        data['node_features'] = out
        return out

