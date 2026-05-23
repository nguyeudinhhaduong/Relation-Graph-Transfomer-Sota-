"""
RelGT++ Model — Novel improvements over the original RelGT SOTA.

Architecture improvements:
1. CrossModalGatedFusion: Learnable per-modality gating instead of naive concat+MLP
2. GumbelSoftCodebook: Differentiable codebook preventing collapse
3. CrossAttentionBridge: Bidirectional local-global information exchange
4. Multi-view readout: Attention + mean + max pooling (in local_module.py)
5. SwiGLU + Stochastic Depth (in local_module.py)
6. Auxiliary losses: codebook entropy + local-global agreement
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP

from codebook import GumbelSoftCodebook
from einops import rearrange
from local_module import LocalModule

from torch_frame.data.stats import StatType
from typing import Dict, Any, List

from encoders import (
    NeighborNodeTypeEncoder,
    NeighborHopEncoder,
    NeighborTimeEncoder,
    NeighborTfsEncoder,
    GNNPEEncoder,
)


# ===================== Novel Module 1: Cross-Modal Gated Fusion ===================== #

class CrossModalGatedFusion(nn.Module):
    """
    Replaces naive concatenation + MLP with learnable per-modality gating.

    Each modality gets a gate score based on its content and the context
    from other modalities. This allows the model to dynamically weight
    the importance of type/hop/time/features/PE per sample and token.

    Formula:
        gate_i = σ(W_self_i(e_i) + W_ctx_i(mean(e_{j≠i})))
        output = LayerNorm(Σ_i gate_i · W_proj_i(e_i))
    """

    def __init__(self, channels: int, num_modalities: int = 5):
        super().__init__()
        self.num_modalities = num_modalities
        self.channels = channels

        # Per-modality: self projection, context projection, gate, value projection
        self.self_projs = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])
        self.ctx_projs = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])
        self.val_projs = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])

        self.layer_norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)

    def reset_parameters(self):
        for mod_list in [self.self_projs, self.ctx_projs, self.val_projs]:
            for m in mod_list:
                m.reset_parameters()
        self.layer_norm.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, embeddings: list):
        """
        Args:
            embeddings: list of N tensors, each [B, K, C]
        Returns:
            [B, K, C] fused representation
        """
        N = len(embeddings)
        assert N == self.num_modalities

        # Compute mean context for each modality (excluding itself)
        total = sum(embeddings)  # [B, K, C]

        fused = torch.zeros_like(embeddings[0])
        for i in range(N):
            ctx = (total - embeddings[i]) / max(N - 1, 1)  # mean of others

            # Gate: how much to attend to this modality
            gate = torch.sigmoid(self.self_projs[i](embeddings[i]) + self.ctx_projs[i](ctx))

            # Value
            val = self.val_projs[i](embeddings[i])

            fused = fused + gate * val

        return self.layer_norm(self.out_proj(fused))


# =================== Novel Module 2: Cross-Attention Bridge =================== #

class CrossAttentionBridge(nn.Module):
    """
    Bidirectional cross-attention between local and global branches.

    Instead of naive concatenation, the local branch attends to global
    context and vice versa, then a learned gate merges both views.

    Formula:
        enriched_l = local + α · V_g  (local attends to global)
        enriched_g = global + β · V_l  (global attends to local)
        gate = σ(W([enriched_l; enriched_g]))
        output = gate ⊙ enriched_l + (1-gate) ⊙ enriched_g
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads

        # Local → Global cross-attention
        self.q_l = nn.Linear(channels, channels)
        self.k_g = nn.Linear(channels, channels)
        self.v_g = nn.Linear(channels, channels)

        # Global → Local cross-attention
        self.q_g = nn.Linear(channels, channels)
        self.k_l = nn.Linear(channels, channels)
        self.v_l = nn.Linear(channels, channels)

        # Gated merge
        self.gate_proj = nn.Linear(2 * channels, channels)

        # Output projection + normalization
        self.out_proj = nn.Linear(channels, channels)
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        for module in [self.q_l, self.k_g, self.v_g, self.q_g, self.k_l, self.v_l,
                        self.gate_proj, self.out_proj]:
            module.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, local_out, global_out):
        """
        Args:
            local_out:  [B, C]  (aggregated local representation)
            global_out: [B, C]  (global centroid attention output)
        Returns:
            [B, C] fused representation
        """
        scale = 1.0 / math.sqrt(self.channels)

        # Local attends to Global
        q1 = self.q_l(local_out)   # [B, C]
        k1 = self.k_g(global_out)  # [B, C]
        v1 = self.v_g(global_out)  # [B, C]
        alpha = torch.sigmoid((q1 * k1).sum(dim=-1, keepdim=True) * scale)  # [B, 1]
        enriched_l = local_out + alpha * v1

        # Global attends to Local
        q2 = self.q_g(global_out)
        k2 = self.k_l(local_out)
        v2 = self.v_l(local_out)
        beta = torch.sigmoid((q2 * k2).sum(dim=-1, keepdim=True) * scale)
        enriched_g = global_out + beta * v2

        # Gated merge
        gate = torch.sigmoid(self.gate_proj(torch.cat([enriched_l, enriched_g], dim=-1)))
        merged = gate * enriched_l + (1 - gate) * enriched_g

        output = self.layer_norm(self.out_proj(self.dropout(merged)) + local_out)
        return output


# ======================== Improved RelGT Layer ======================== #

class RelGTLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        local_num_layers,
        global_dim,
        num_nodes,
        heads=1,
        concat=True,
        ff_dropout=0.0,
        attn_dropout=0.0,
        edge_dim=None,
        conv_type="local",
        num_centroids=None,
        sample_node_len=100,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len

        self.local_module = LocalModule(
            seq_len=self.sample_node_len,
            input_dim=in_channels,
            n_layers=local_num_layers,
            num_heads=heads,
            hidden_dim=out_channels,
            dropout_rate=ff_dropout,
            attention_dropout_rate=attn_dropout,
        )
        self.layer_norm_local = nn.LayerNorm(out_channels)

        # Track auxiliary losses
        self._aux_losses = {}

        if self.conv_type != "local":
            # ★ Use GumbelSoftCodebook instead of VectorQuantizerEMA
            self.vq = GumbelSoftCodebook(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax

            attn_channels = out_channels // heads
            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim, heads * attn_channels)
            self.lin_query_g = Linear(global_dim, heads * attn_channels)
            self.lin_value_g = Linear(global_dim, heads * attn_channels)
            self.layer_norm_global = nn.LayerNorm(out_channels)

            # ★ Cross-Attention Bridge (replaces concat)
            self.cross_bridge = CrossAttentionBridge(out_channels, num_heads=heads)

        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_type != "local":
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            if hasattr(self, 'vq'):
                self.vq.reset_parameters()
            self.cross_bridge.reset_parameters()
            self.layer_norm_global.reset_parameters()

        if hasattr(self.local_module, 'reset_parameters'):
            self.local_module.reset_parameters()

    def forward(self, x_set, x, node_indices):
        if self.conv_type == "local":
            out = self.local_forward(x_set)
            out = self.layer_norm_local(out)

        elif self.conv_type == "global":
            out = self.global_forward(x, node_indices)
            out = self.layer_norm_global(out)

        elif self.conv_type == "full":
            out_local = self.local_forward(x_set)
            out_global = self.global_forward(x, node_indices)
            out_local = self.layer_norm_local(out_local)
            out_global = self.layer_norm_global(out_global)

            # ★ Cross-Attention Bridge instead of concat
            out = self.cross_bridge(out_local, out_global)

            # ★ Local-Global agreement auxiliary loss
            with torch.no_grad():
                cos_sim = F.cosine_similarity(out_local, out_global, dim=-1).mean()
            self._aux_losses['local_global_agreement'] = (1.0 - cos_sim) * 0.01

        else:
            raise NotImplementedError

        return out

    def global_forward(self, x, batch_idx):
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        # ★ Do NOT detach — allow gradients to flow through codebook
        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=h), (q, k, v))
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count
        dots = dots + torch.log(centroid_count.clamp_min(1).view(1, 1, -1).float())

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h n d -> n (h d)")

        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)

            # ★ Codebook entropy loss
            self._aux_losses['codebook_entropy'] = self.vq.codebook_entropy_loss() * 0.1

        return out

    def local_forward(self, x_set, pretrain_token=False):
        return self.local_module(x_set, pretrain_token)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"local_num_layers={self.local_num_layers})"
        )


# ======================== Main RelGT++ Model ======================== #

class RelGT(torch.nn.Module):
    """
    RelGT++ — Improved Relational Graph Transformer.

    Novel improvements over original RelGT:
    1. CrossModalGatedFusion: per-modality gating replaces concat+MLP
    2. GumbelSoftCodebook: differentiable VQ preventing codebook collapse
    3. CrossAttentionBridge: bidirectional local↔global information flow
    4. Multi-view readout: attention + mean + max pooling
    5. SwiGLU FFN + stochastic depth in local transformer
    6. Auxiliary losses: codebook entropy + local-global agreement
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: Dict[str, int],
        col_names_dict: Dict[str, Dict[str, List[str]]],
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        local_num_layers: int,
        channels: int,
        out_channels: int,
        global_dim: int,
        heads: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        conv_type: str = "full",
        ablate: str = "none",
        gnn_pe_dim: int = 0,
        num_centroids: int = 4096,
        sample_node_len: int = 100,
        args: Any = None,
    ):
        super().__init__()

        self.max_neighbor_hop = max_neighbor_hop
        self.node_type_map = node_type_map
        self.conv_type = conv_type
        self.ablate = ablate

        # ─── Token Encoders (unchanged) ───
        self.type_encoder = NeighborNodeTypeEncoder(
            embedding_dim=channels, node_type_map=self.node_type_map
        )
        self.hop_encoder = NeighborHopEncoder(
            embedding_dim=channels, max_neighbor_hop=self.max_neighbor_hop
        )
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder = NeighborTfsEncoder(
            channels=channels,
            node_type_map=self.node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
        self.pe_encoder = GNNPEEncoder(embedding_dim=channels, pe_dim=gnn_pe_dim)

        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_time = nn.LayerNorm(channels)
        self.layer_norm_tfs = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)

        hidden_channels = channels

        ablate_key_dict = {"type": 0, "hop": 1, "time": 2, "tfs": 3, "gnn": 4}
        self.ablate_idx = ablate_key_dict.get(ablate, None)
        num_modalities = 5 if self.ablate_idx is None else 4

        # ★ Novel: Cross-Modal Gated Fusion (replaces concat + MLP)
        self.in_mixture = CrossModalGatedFusion(
            channels=channels,
            num_modalities=num_modalities,
        )

        # ─── RelGT Layers ───
        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        _overall_num_layers = 1
        for _ in range(_overall_num_layers):
            self.convs.append(
                RelGTLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    local_num_layers=local_num_layers,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                    sample_node_len=sample_node_len,
                )
            )

            # ★ Bridge outputs [B, C] not [B, 2C], so h_times=1 always
            self.ffs.append(
                nn.Sequential(
                    nn.BatchNorm1d(hidden_channels),
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Dropout(ff_dropout),
                    nn.BatchNorm1d(hidden_channels),
                )
            )

        # Supervised head
        self.head = MLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            num_layers=2,
        )

        # Store auxiliary losses (accessible after forward)
        self._aux_losses = {}

    def reset_parameters(self):
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.tfs_encoder.reset_parameters()
        self.pe_encoder.reset_parameters()
        self.in_mixture.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            for layer in ff:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        self.head.reset_parameters()

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
    ):
        # ─── Encode all 5 modalities ───
        neighbor_tfs = self.layer_norm_tfs(self.tfs_encoder(grouped_tf_dict, neighbor_types))
        neighbor_types_emb = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
        neighbor_hops_emb = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
        neighbor_times_emb = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
        neighbor_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))

        embeddings = [neighbor_types_emb, neighbor_hops_emb, neighbor_times_emb,
                       neighbor_tfs, neighbor_pe]
        if self.ablate_idx is not None:
            embeddings.pop(self.ablate_idx)

        # ★ Cross-Modal Gated Fusion (novel)
        x_set = self.in_mixture(embeddings)

        # ─── Pass through RelGT layers ───
        x = x_set[:, 0, :]  # seed token
        self._aux_losses = {}

        for i, conv in enumerate(self.convs):
            x = conv(x_set, x, node_indices)
            x = self.ffs[i](x)

            # Collect auxiliary losses from layer
            for k, v in conv._aux_losses.items():
                self._aux_losses[f"layer{i}_{k}"] = v

        x = self.head(x)
        return x

    def get_aux_loss(self):
        """
        Returns total auxiliary loss for the current forward pass.
        Call after forward() and add to task loss during training.
        """
        if not self._aux_losses:
            return torch.tensor(0.0)
        return sum(self._aux_losses.values())

    def global_forward(self, x, pos_enc, node_indices):
        raise NotImplementedError