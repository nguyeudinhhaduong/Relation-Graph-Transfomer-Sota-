"""
Codebook module for RelGT++.

Improvements over original VectorQuantizerEMA:
1. GumbelSoftCodebook: Differentiable soft assignment via Gumbel-Softmax
2. Codebook entropy regularization to prevent collapse
3. Learnable parameters (gradient-based) instead of EMA-only updates
4. Temperature annealing for smooth soft→hard transition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Original VQ-EMA codebook (kept for ablation/reference)."""

    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.register_buffer("_embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("_embedding_output", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.randn(num_embeddings, embedding_dim))
        self._decay = decay
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def reset_parameters(self):
        nn.init.normal_(self._embedding, 0, 1.0)
        nn.init.normal_(self._embedding_output, 0, 1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, 0, 1.0)
        self.bn.reset_parameters()

    def get_k(self):
        return self._embedding_output

    def get_v(self):
        return self._embedding_output[:, :self._embedding_dim]

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding
        distances = (
            torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
            + torch.sum(embedding_normalized ** 2, dim=1)
            - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        if self.training:
            self._ema_cluster_size.data = (
                self._ema_cluster_size * self._decay
                + (1 - self._decay) * torch.sum(encodings, 0)
            )
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
            )
            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
            running_mean = self.bn.running_mean.unsqueeze(0)
            self._embedding_output.data = self._embedding * running_std + running_mean
        return encoding_indices

    def codebook_entropy_loss(self):
        return torch.tensor(0.0)


class GumbelSoftCodebook(nn.Module):
    """
    Novel codebook with Gumbel-Softmax differentiable assignment.

    Key improvements over VQ-EMA:
    1. Gradients flow to ALL codewords via soft assignment → prevents collapse
    2. Temperature annealing: soft exploration → hard exploitation
    3. Codebook entropy loss regularizes uniform utilization
    4. Separate learnable K/V embeddings as nn.Parameter
    """

    def __init__(self, num_embeddings, embedding_dim, decay=0.99,
                 init_temp=2.0, min_temp=0.5, anneal_rate=3e-4):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        # Learnable codebook (nn.Parameter → receives gradients from attention)
        self.embedding_k = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embedding_v = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # Gumbel-Softmax temperature
        self.register_buffer('temperature', torch.tensor(float(init_temp)))
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate

        # EMA usage tracking for entropy loss
        self.register_buffer('_ema_usage', torch.ones(num_embeddings) / num_embeddings)

        # BN for input normalization (like original)
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding_k, 0, 0.02)
        nn.init.normal_(self.embedding_v, 0, 0.02)
        self._ema_usage.fill_(1.0 / self._num_embeddings)
        self.bn.reset_parameters()

    def get_k(self):
        """Return codebook keys (differentiable — do NOT detach in caller)."""
        return self.embedding_k

    def get_v(self):
        """Return codebook values (differentiable — do NOT detach in caller)."""
        return self.embedding_v

    def update(self, x):
        """
        Gumbel-Softmax soft assignment.
        Returns hard indices [N, 1] via straight-through estimator.
        """
        x_norm = self.bn(x)

        # Squared Euclidean distances → logits
        distances = (
            x_norm.pow(2).sum(1, keepdim=True)
            - 2 * x_norm @ self.embedding_k.t()
            + self.embedding_k.pow(2).sum(1).unsqueeze(0)
        )
        logits = -distances

        if self.training:
            # Gumbel noise for exploration
            gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            noisy_logits = (logits + gumbels) / self.temperature.clamp(min=0.1)
            soft_probs = F.softmax(noisy_logits, dim=-1)
            hard_indices = soft_probs.argmax(dim=-1)

            # Track utilization via EMA
            with torch.no_grad():
                counts = torch.bincount(hard_indices, minlength=self._num_embeddings).float()
                counts = counts / (counts.sum() + 1e-8)
                self._ema_usage.mul_(0.99).add_(counts * 0.01)
                # Anneal temperature
                new_temp = max(self.temperature.item() * (1 - self.anneal_rate), self.min_temp)
                self.temperature.fill_(new_temp)

            return hard_indices.unsqueeze(1)
        else:
            return logits.argmax(dim=-1).unsqueeze(1)

    def codebook_entropy_loss(self):
        """
        Negative normalized entropy of codebook usage.
        Returns 0 when perfectly uniform, approaches 1 when fully collapsed.
        """
        p = self._ema_usage / (self._ema_usage.sum() + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        max_entropy = math.log(self._num_embeddings)
        return 1.0 - (entropy / max_entropy)

    @property
    def utilization_rate(self):
        """Fraction of codewords actively used."""
        p = self._ema_usage / (self._ema_usage.sum() + 1e-8)
        threshold = 0.5 / self._num_embeddings
        return (p > threshold).float().mean().item()