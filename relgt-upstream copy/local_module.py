"""
Local Module for RelGT++.

[NOVEL] Improvements over original:
  1. StochasticDepth      — drops entire sub-layer during training (regularisation)
  2. GatedFFN (SwiGLU)    — gated activation; better gradient flow
  3. MultiViewReadout     — attention + mean-pool + max-pool combined
  4. Flash-attention      — scaled_dot_product_attention (PyTorch 2.x)

References:
  SwiGLU         : Shazeer 2020
  Stochastic Depth: Huang et al. 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Stochastic Depth ────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """[NOVEL] Drop an entire residual branch with probability drop_prob."""

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        keep = torch.rand(1, device=x.device) < keep_prob
        return x * keep.float() / keep_prob


# ─── Feed-forward networks ───────────────────────────────────────────────────

class GatedFFN(nn.Module):
    """
    [NOVEL] SwiGLU-style Feed-Forward Network.

    FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )

    Advantages over standard GELU-FFN:
      - Gating controls information flow → better gradient signal
      - Wider hidden dimension compensated by gating compression
    """

    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float):
        super().__init__()
        self.bn_in  = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.w_gate = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w_up   = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w_down = nn.Linear(ffn_size,   hidden_size, bias=False)
        self.drop   = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        for l in [self.w_gate, self.w_up, self.w_down]:
            nn.init.xavier_uniform_(l.weight)
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()

    def forward(self, x):
        # x: [B, K, D]  —  BN expects [B, D, K]
        x = self.bn_in(x.permute(0, 2, 1)).permute(0, 2, 1)
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        x    = self.w_down(self.drop(gate * up))
        x    = self.drop(x)
        x    = self.bn_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class FeedForwardNetwork(nn.Module):
    """Original FFN — kept for compatibility / ablation."""

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.bn_in  = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout_rate),
        )

    def reset_parameters(self):
        for l in self.ffn_net:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()

    def forward(self, x):
        x = self.bn_in(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.ffn_net(x)
        x = self.bn_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# ─── Encoder Layer ───────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with:
      [NOVEL] SwiGLU FFN
      [NOVEL] Stochastic Depth on both attention and FFN branches
      Flash attention (scaled_dot_product_attention)
    """

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads, drop_path_rate=0.1):
        super().__init__()
        self.num_heads              = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.q_proj    = nn.Linear(hidden_size, hidden_size)
        self.k_proj    = nn.Linear(hidden_size, hidden_size)
        self.v_proj    = nn.Linear(hidden_size, hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        # GatedFFN with 4x expansion (then halved by gating → effective 2x)
        self.ffn = GatedFFN(hidden_size, ffn_size * 2, dropout_rate)

        dp = drop_path_rate
        self.drop_path_attn = StochasticDepth(dp) if dp > 0 else nn.Identity()
        self.drop_path_ffn  = StochasticDepth(dp) if dp > 0 else nn.Identity()

    def reset_parameters(self):
        self.attn_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        for p in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(p.weight)
            if p.bias is not None:
                nn.init.zeros_(p.bias)
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None):
        B, L, D = x.shape
        head_dim = D // self.num_heads

        # ── Self-attention ────────────────────────────────────────────────
        residual = x
        h = self.attn_norm(x)
        Q = self.q_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.attn_drop(self.out_proj(out))
        x   = residual + self.drop_path_attn(out)

        # ── Feed-forward ─────────────────────────────────────────────────
        residual = x
        x = x + self.drop_path_ffn(self.ffn(self.ffn_norm(x)))
        return x


# ─── Novel: Multi-View Readout ───────────────────────────────────────────────

class MultiViewReadout(nn.Module):
    """
    [NOVEL] Combines three complementary aggregation strategies:
      1. Attention-weighted  — learns which neighbours matter
      2. Mean pooling        — captures overall neighbourhood statistics
      3. Max pooling         — captures most salient features

    Output = LayerNorm( seed + MLP([v_attn ‖ v_mean ‖ v_max]) )

    Motivation: each view captures different inductive biases;
    ensembling all three is more robust than any single readout.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(2 * hidden_dim, 1)
        self.combine    = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def reset_parameters(self):
        self.attn_score.reset_parameters()
        for l in self.combine:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, output, seq_len):
        """
        Args:
            output  : [B, K, D]
            seq_len : K (unused scalar — kept for API compatibility)
        Returns:
            [B, D]
        """
        seed      = output[:, 0, :]        # [B, D]
        neighbors = output[:, 1:, :]       # [B, K-1, D]

        # View 1: attention-weighted
        seed_exp  = seed.unsqueeze(1).expand_as(neighbors)
        scores    = self.attn_score(torch.cat([seed_exp, neighbors], dim=-1))  # [B, K-1, 1]
        weights   = F.softmax(scores, dim=1)
        v_attn    = (neighbors * weights).sum(dim=1)   # [B, D]

        # View 2: mean
        v_mean = neighbors.mean(dim=1)                 # [B, D]

        # View 3: max
        v_max  = neighbors.max(dim=1).values           # [B, D]

        fused  = self.combine(torch.cat([v_attn, v_mean, v_max], dim=-1))  # [B, D]
        return self.norm(seed + fused)


# ─── LocalModule ─────────────────────────────────────────────────────────────

class LocalModule(nn.Module):
    """
    Local Transformer for RelGT++.

    Improvements:
      [NOVEL] SwiGLU gated FFN in each EncoderLayer
      [NOVEL] Stochastic Depth (linearly increasing rate across layers)
      [NOVEL] Multi-View Readout (attention + mean + max)
    """

    def __init__(
        self,
        seq_len,
        input_dim,
        node_only_readout=False,
        n_layers=1,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.3,
        attention_dropout_rate=0,
    ):
        super().__init__()
        self.seq_len              = seq_len
        self.node_only_readout    = node_only_readout
        self.input_dim            = input_dim
        self.hidden_dim           = hidden_dim
        self.ffn_dim              = 2 * hidden_dim
        self.num_heads            = num_heads
        self.n_layers             = n_layers
        self.dropout_rate         = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(input_dim, hidden_dim)

        # Linearly increasing stochastic-depth rates
        drop_rates = [0.05 * (i + 1) / max(n_layers, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_dim, self.ffn_dim,
                dropout_rate, attention_dropout_rate,
                num_heads, drop_path_rate=drop_rates[i],
            )
            for i in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.readout  = MultiViewReadout(hidden_dim)          # [NOVEL]
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)       # kept for compat

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.attn_layer.reset_parameters()
        self.final_ln.reset_parameters()
        self.readout.reset_parameters()
        for l in self.layers:
            l.reset_parameters()

    def forward(self, batched_data, pretrain_token=False):
        x = self.att_embeddings_nope(batched_data)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        if pretrain_token:
            return x
        return self.readout(x, self.seq_len)
