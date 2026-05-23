"""
RelGT++ Model.

[NOVEL] Improvements over original RelGT:
  1. CrossModalGatedFusion  — per-modality gates replace naive concat+MLP
  2. GumbelSoftCodebook     — differentiable VQ preventing codebook collapse
  3. CrossAttentionBridge   — bidirectional local↔global information exchange
  4. MultiViewReadout       — attention + mean + max pooling  (local_module.py)
  5. SwiGLU + StochasticDepth                                 (local_module.py)
  6. Auxiliary losses       — codebook entropy + local-global cosine agreement
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP
from einops import rearrange

from codebook import GumbelSoftCodebook          # [NOVEL]
from local_module import LocalModule             # [NOVEL]

from torch_frame.data.stats import StatType
from typing import Any, Dict, List

from encoders import (
    NeighborNodeTypeEncoder,
    NeighborHopEncoder,
    NeighborTimeEncoder,
    NeighborTfsEncoder,
    GNNPEEncoder,
)


# ═══════════════════════════════════════════════════════════════════════════════
# [NOVEL] Module 1 — Cross-Modal Gated Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalGatedFusion(nn.Module):
    """
    [NOVEL] Replaces naive concat+MLP with learnable per-modality gating.

    For each modality i:
      ctx_i  = mean of all other modality embeddings
      gate_i = σ( W_self_i(e_i) + W_ctx_i(ctx_i) )
      val_i  = W_val_i(e_i)

    output = LayerNorm( W_out( Σ_i gate_i ⊙ val_i ) )

    Motivation: different modalities (type / hop / time / features / PE)
    have varying relevance per sample.  Static concat treats them equally;
    gated fusion learns task-specific importance weights dynamically.
    """

    def __init__(self, channels: int, num_modalities: int = 5):
        super().__init__()
        self.num_modalities = num_modalities
        self.channels       = channels

        self.self_projs = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])
        self.ctx_projs  = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])
        self.val_projs  = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_modalities)])

        self.out_proj  = nn.Linear(channels, channels)
        self.norm      = nn.LayerNorm(channels)

    def reset_parameters(self):
        for lst in [self.self_projs, self.ctx_projs, self.val_projs]:
            for m in lst:
                m.reset_parameters()
        self.out_proj.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, embeddings: list):
        """
        Args:
            embeddings: list of `num_modalities` tensors each [B, K, C]
        Returns:
            [B, K, C]
        """
        assert len(embeddings) == self.num_modalities
        total  = sum(embeddings)                                # [B, K, C]
        fused  = torch.zeros_like(embeddings[0])

        for i, e_i in enumerate(embeddings):
            ctx    = (total - e_i) / max(self.num_modalities - 1, 1)
            gate   = torch.sigmoid(self.self_projs[i](e_i) + self.ctx_projs[i](ctx))
            fused  = fused + gate * self.val_projs[i](e_i)

        return self.norm(self.out_proj(fused))


# ═══════════════════════════════════════════════════════════════════════════════
# [NOVEL] Module 2 — Cross-Attention Bridge
# ═══════════════════════════════════════════════════════════════════════════════

class CrossAttentionBridge(nn.Module):
    """
    [NOVEL] Bidirectional cross-attention between local and global branches.

    Instead of naive concat([local; global]):
      enriched_l = local  + α · V_g   (local  attends to global)
      enriched_g = global + β · V_l   (global attends to local )
      gate       = σ( W_gate([enriched_l ‖ enriched_g]) )
      merged     = gate ⊙ enriched_l + (1-gate) ⊙ enriched_g
      output     = LayerNorm( W_out(merged) + local )

    Motivation: local and global views are complementary; cross-attention
    lets each branch query the other before gated merging.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads
        scale = 1.0 / math.sqrt(channels)
        self._scale = scale

        # local → global
        self.q_l = nn.Linear(channels, channels)
        self.k_g = nn.Linear(channels, channels)
        self.v_g = nn.Linear(channels, channels)

        # global → local
        self.q_g = nn.Linear(channels, channels)
        self.k_l = nn.Linear(channels, channels)
        self.v_l = nn.Linear(channels, channels)

        self.gate_proj = nn.Linear(2 * channels, channels)
        self.out_proj  = nn.Linear(channels, channels)
        self.norm      = nn.LayerNorm(channels)
        self.dropout   = nn.Dropout(0.1)

    def reset_parameters(self):
        for m in [self.q_l, self.k_g, self.v_g,
                  self.q_g, self.k_l, self.v_l,
                  self.gate_proj, self.out_proj]:
            m.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, local_out, global_out):
        """
        Args:
            local_out  : [B, C]
            global_out : [B, C]
        Returns:
            [B, C]
        """
        scale = self._scale

        # local attends to global
        alpha      = torch.sigmoid((self.q_l(local_out) * self.k_g(global_out)).sum(-1, keepdim=True) * scale)
        enriched_l = local_out  + alpha * self.v_g(global_out)

        # global attends to local
        beta       = torch.sigmoid((self.q_g(global_out) * self.k_l(local_out)).sum(-1, keepdim=True) * scale)
        enriched_g = global_out + beta  * self.v_l(local_out)

        gate   = torch.sigmoid(self.gate_proj(torch.cat([enriched_l, enriched_g], dim=-1)))
        merged = gate * enriched_l + (1 - gate) * enriched_g

        return self.norm(self.out_proj(self.dropout(merged)) + local_out)


# ═══════════════════════════════════════════════════════════════════════════════
# RelGTLayer
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.local_num_layers = local_num_layers
        self.heads           = heads
        self.ff_dropout      = ff_dropout
        self.attn_dropout    = attn_dropout
        self.conv_type       = conv_type
        self.num_centroids   = num_centroids
        self.sample_node_len = sample_node_len
        self._aux_losses: Dict[str, torch.Tensor] = {}

        # ── Local branch ─────────────────────────────────────────────────
        self.local_module  = LocalModule(
            seq_len=sample_node_len,
            input_dim=in_channels,
            n_layers=local_num_layers,
            num_heads=heads,
            hidden_dim=out_channels,
            dropout_rate=ff_dropout,
            attention_dropout_rate=attn_dropout,
        )
        self.norm_local = nn.LayerNorm(out_channels)

        # ── Global branch ────────────────────────────────────────────────
        if conv_type != "local":
            # [NOVEL] GumbelSoftCodebook instead of VQ-EMA
            self.vq = GumbelSoftCodebook(num_centroids, global_dim)
            self.register_buffer("c_idx",
                                  torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long))
            self.attn_fn = F.softmax

            attn_ch = out_channels // heads
            self.lin_proj_g  = Linear(in_channels, global_dim)
            self.lin_key_g   = Linear(global_dim, heads * attn_ch)
            self.lin_query_g = Linear(global_dim, heads * attn_ch)
            self.lin_value_g = Linear(global_dim, heads * attn_ch)
            self.norm_global = nn.LayerNorm(out_channels)

            # [NOVEL] CrossAttentionBridge instead of concat
            self.cross_bridge = CrossAttentionBridge(out_channels, num_heads=heads)

        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_type != "local":
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            self.vq.reset_parameters()
            self.cross_bridge.reset_parameters()
            self.norm_global.reset_parameters()
        self.local_module.reset_parameters()

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x_set, x, node_indices):
        self._aux_losses = {}

        if self.conv_type == "local":
            return self.norm_local(self.local_forward(x_set))

        elif self.conv_type == "global":
            return self.norm_global(self.global_forward(x, node_indices))

        elif self.conv_type == "full":
            out_l = self.norm_local(self.local_forward(x_set))
            out_g = self.norm_global(self.global_forward(x, node_indices))

            # [NOVEL] Cross-Attention Bridge
            out = self.cross_bridge(out_l, out_g)

            # [NOVEL] local-global agreement auxiliary loss
            with torch.no_grad():
                cos = F.cosine_similarity(out_l, out_g, dim=-1).mean()
            self._aux_losses['local_global_agreement'] = (1.0 - cos) * 0.01

            return out
        else:
            raise NotImplementedError(f"conv_type={self.conv_type}")

    def local_forward(self, x_set, pretrain_token=False):
        return self.local_module(x_set, pretrain_token)

    def global_forward(self, x, batch_idx):
        h, d = self.heads, self.out_channels
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        # [NOVEL] gradient flows through learnable codebook
        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = [rearrange(t, "n (h d) -> h n d", h=h) for t in (q, k, v)]
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        # frequency-based logit adjustment
        c_uniq, c_cnt = self.c_idx.unique(return_counts=True)
        cnt = torch.zeros(self.num_centroids, dtype=torch.long, device=x.device)
        cnt[c_uniq.long()] = c_cnt
        dots = dots + torch.log(cnt.clamp_min(1).view(1, 1, -1).float())

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out  = rearrange(torch.einsum("h i j, h j d -> h i d", attn, v), "h n d -> n (h d)")

        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().long()
            # [NOVEL] codebook entropy loss
            self._aux_losses['codebook_entropy'] = self.vq.codebook_entropy_loss() * 0.1

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# RelGT++ — Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class RelGT(torch.nn.Module):
    """
    RelGT++ — Relational Graph Transformer with novel improvements.

    Novel contributions (all in this file / local_module.py / codebook.py):
      1. CrossModalGatedFusion  : dynamic per-modality importance weights
      2. GumbelSoftCodebook     : differentiable VQ, prevents collapse
      3. CrossAttentionBridge   : bidirectional local↔global fusion
      4. MultiViewReadout       : attention + mean + max aggregation
      5. SwiGLU FFN             : gated activation for better gradients
      6. Stochastic Depth       : drop-path regularisation
      7. Auxiliary losses       : entropy + cosine agreement
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: Dict[str, int],
        col_names_dict: Dict[str, Any],
        col_stats_dict: Dict[str, Any],
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
        self.conv_type  = conv_type
        self.ablate     = ablate
        self.node_type_map   = node_type_map
        self.max_neighbor_hop = max_neighbor_hop

        # ── Encoders (unchanged from original) ───────────────────────────
        self.type_encoder = NeighborNodeTypeEncoder(
            node_type_map=node_type_map,
            embedding_dim=channels,
        )
        self.hop_encoder  = NeighborHopEncoder(
            max_neighbor_hop=max_neighbor_hop,
            embedding_dim=channels,
        )
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder  = NeighborTfsEncoder(
            channels=channels,
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
        self.pe_encoder = GNNPEEncoder(channels, pe_dim=gnn_pe_dim)

        self.ln_type = nn.LayerNorm(channels)
        self.ln_hop  = nn.LayerNorm(channels)
        self.ln_time = nn.LayerNorm(channels)
        self.ln_tfs  = nn.LayerNorm(channels)
        self.ln_pe   = nn.LayerNorm(channels)

        # ── Ablation index ────────────────────────────────────────────────
        _ablate_map = {"type": 0, "hop": 1, "time": 2, "tfs": 3, "gnn": 4}
        self.ablate_idx   = _ablate_map.get(ablate, None)
        num_modalities    = 5 if self.ablate_idx is None else 4

        # ── [NOVEL] Cross-Modal Gated Fusion ─────────────────────────────
        self.in_mixture = CrossModalGatedFusion(channels, num_modalities)

        # ── RelGT layers ─────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.ffs   = nn.ModuleList()

        for _ in range(1):   # _overall_num_layers = 1 (same as original)
            self.convs.append(RelGTLayer(
                in_channels=channels,
                out_channels=channels,
                local_num_layers=local_num_layers,
                global_dim=global_dim,
                num_nodes=num_nodes,
                heads=heads,
                ff_dropout=ff_dropout,
                attn_dropout=attn_dropout,
                conv_type=conv_type,
                num_centroids=num_centroids,
                sample_node_len=sample_node_len,
            ))
            # CrossAttentionBridge outputs [B, C] (not [B, 2C]) → no size change
            self.ffs.append(nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.Linear(channels, channels * 2),
                nn.GELU(),
                nn.Dropout(ff_dropout),
                nn.Linear(channels * 2, channels),
                nn.Dropout(ff_dropout),
                nn.BatchNorm1d(channels),
            ))

        self.head = MLP(channels, hidden_channels=channels,
                        out_channels=out_channels, num_layers=2)

        self._aux_losses: Dict[str, torch.Tensor] = {}

    # ── reset_parameters ─────────────────────────────────────────────────

    def reset_parameters(self):
        for enc in [self.type_encoder, self.hop_encoder, self.time_encoder,
                    self.tfs_encoder, self.pe_encoder]:
            enc.reset_parameters()
        self.in_mixture.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            for l in ff:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()
        self.head.reset_parameters()

    # ── forward ──────────────────────────────────────────────────────────

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
        # Encode five modalities
        e_tfs  = self.ln_tfs (self.tfs_encoder(grouped_tf_dict, neighbor_types))
        e_type = self.ln_type(self.type_encoder(neighbor_types.long()))
        e_hop  = self.ln_hop (self.hop_encoder(neighbor_hops.long()))
        e_time = self.ln_time(self.time_encoder(neighbor_times.float()))
        e_pe   = self.ln_pe  (self.pe_encoder(edge_index, batch))

        embeddings = [e_type, e_hop, e_time, e_tfs, e_pe]
        if self.ablate_idx is not None:
            embeddings.pop(self.ablate_idx)

        # [NOVEL] Cross-Modal Gated Fusion → [B, K, C]
        x_set = self.in_mixture(embeddings)

        x = x_set[:, 0, :]   # seed token  [B, C]
        self._aux_losses = {}

        for i, conv in enumerate(self.convs):
            x = conv(x_set, x, node_indices)
            x = self.ffs[i](x)
            for k, v in conv._aux_losses.items():
                self._aux_losses[f"layer{i}_{k}"] = v

        return self.head(x)

    def get_aux_loss(self):
        """Sum of all auxiliary losses (call after forward, add to task loss)."""
        if not self._aux_losses:
            return torch.tensor(0.0)
        return sum(self._aux_losses.values())

    def global_forward(self, x, pos_enc, node_indices):
        raise NotImplementedError
