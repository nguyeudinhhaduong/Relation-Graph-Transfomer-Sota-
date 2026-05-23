"""
Local Module for RelGT++.

Improvements over original:
1. StochasticDepth regularization in EncoderLayer
2. SwiGLU-style gated FFN (wider + gated activation)
3. Multi-View Readout: attention + mean-pool + max-pool combined
4. Rotary-style relative position bias in attention (optional)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """Drop entire layer with probability `drop_prob` during training."""

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(1, device=x.device) > self.drop_prob
        return x * keep.float() / (1.0 - self.drop_prob)


class GatedFFN(nn.Module):
    """
    SwiGLU-style Feed-Forward Network.

    Replaces standard MLP with gated linear unit for better gradient flow
    and expressiveness. Uses wider expansion (4x) with gating.

    Formula: FFN(x) = (W_up(x) ⊙ SiLU(W_gate(x))) @ W_down
    """

    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)

        # SwiGLU: two parallel projections, one gated
        self.w_gate = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w_down = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        for layer in [self.w_gate, self.w_up, self.w_down]:
            nn.init.xavier_uniform_(layer.weight)
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()

    def forward(self, x):
        # x: [B, K, D]
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        # SwiGLU gating
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = self.w_down(self.dropout(gate * up))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        return x


class FeedForwardNetwork(nn.Module):
    """Original FFN kept for compatibility."""

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout_rate),
        )

    def reset_parameters(self):
        for layer in self.ffn_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        x = self.ffn_net(x)
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        return x


class EncoderLayer(nn.Module):
    """
    Improved Transformer Encoder Layer.

    Improvements:
    - SwiGLU FFN (4x expansion with gating)
    - Stochastic depth for regularization
    - Flash attention via scaled_dot_product_attention
    """

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads, drop_path_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        # Use gated FFN with 4x expansion
        self.ffn = GatedFFN(hidden_size, ffn_size * 2, dropout_rate)

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None):
        # Self-attention with stochastic depth
        residual = x
        x_norm = self.self_attention_norm(x)

        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        B, L, D = Q.shape
        head_dim = D // self.num_heads

        Q = Q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.self_attention_dropout(attn_output)

        x = residual + self.drop_path(attn_output)

        # FFN with stochastic depth
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.drop_path(ffn_output)
        return x


class MultiViewReadout(nn.Module):
    """
    Novel multi-view readout that combines three aggregation strategies:
    1. Attention-weighted: learns which neighbors matter (original)
    2. Mean pooling: captures overall neighborhood statistics
    3. Max pooling: captures most salient features

    Output = MLP([v_attn; v_mean; v_max])
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)
        self.combine = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def reset_parameters(self):
        self.attn_layer.reset_parameters()
        for layer in self.combine:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, output, seq_len):
        """
        Args:
            output: [B, K, D] full transformer output
            seq_len: total sequence length K
        Returns:
            [B, D] aggregated representation
        """
        node_tensor = output[:, 0, :]  # [B, D] seed node
        neighbor_tensor = output[:, 1:, :]  # [B, K-1, D]

        # View 1: Attention-weighted (original style)
        target = node_tensor.unsqueeze(1).expand_as(neighbor_tensor)
        attn_scores = self.attn_layer(torch.cat([target, neighbor_tensor], dim=-1))
        attn_weights = F.softmax(attn_scores, dim=1)
        v_attn = (neighbor_tensor * attn_weights).sum(dim=1)  # [B, D]

        # View 2: Mean pooling
        v_mean = neighbor_tensor.mean(dim=1)  # [B, D]

        # View 3: Max pooling
        v_max = neighbor_tensor.max(dim=1)[0]  # [B, D]

        # Combine all views
        combined = torch.cat([v_attn, v_mean, v_max], dim=-1)  # [B, 3D]
        fused = self.combine(combined)  # [B, D]

        # Residual with seed node
        output = self.layer_norm(node_tensor + fused)
        return output


class LocalModule(nn.Module):
    """
    Improved Local Transformer Module for RelGT++.

    Improvements:
    - SwiGLU gated FFN in each encoder layer
    - Stochastic depth for regularization
    - Multi-view readout (attention + mean + max)
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
        self.seq_len = seq_len
        self.node_only_readout = node_only_readout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        # Stochastic depth with linearly increasing drop rate
        drop_rates = [0.05 * (i + 1) / max(n_layers, 1) for i in range(n_layers)]

        self.layers = nn.ModuleList([
            EncoderLayer(
                self.hidden_dim,
                self.ffn_dim,
                self.dropout_rate,
                self.attention_dropout_rate,
                self.num_heads,
                drop_path_rate=drop_rates[i],
            )
            for i in range(self.n_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Multi-view readout (replaces simple attention readout)
        self.readout = MultiViewReadout(hidden_dim)

        # Keep old attn_layer for compatibility
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.attn_layer.reset_parameters()
        self.final_ln.reset_parameters()
        self.readout.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data, pretrain_token=False):
        tensor = self.att_embeddings_nope(batched_data)

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)

        if pretrain_token:
            return output

        # Use multi-view readout
        return self.readout(output, self.seq_len)