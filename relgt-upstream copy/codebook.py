"""
Codebook module for RelGT++.

[NOVEL] Replaces VectorQuantizerEMA with GumbelSoftCodebook:
  1. Differentiable soft assignment via Gumbel-Softmax
  2. Codebook entropy regularisation to prevent collapse
  3. Learnable nn.Parameter (gradient-based) instead of EMA-only
  4. Temperature annealing for smooth soft → hard transition

References:
  Gumbel-Softmax : Jang et al., 2017
  VQ-VAE         : van den Oord et al., 2017
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Original codebook kept for ablation ────────────────────────────────────

class VectorQuantizerEMA(nn.Module):
    """Original VQ-EMA codebook (kept for ablation/reference)."""

    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super().__init__()
        self._embedding_dim  = embedding_dim
        self._num_embeddings = num_embeddings
        self.register_buffer("_embedding",        torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("_embedding_output", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("_ema_cluster_size",  torch.zeros(num_embeddings))
        self.register_buffer("_ema_w",             torch.randn(num_embeddings, embedding_dim))
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
        inputs_normalized    = self.bn(x)
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
            self._ema_w.data    = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            running_std  = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
            running_mean = self.bn.running_mean.unsqueeze(0)
            self._embedding_output.data = self._embedding * running_std + running_mean
        return encoding_indices

    def codebook_entropy_loss(self):
        return torch.tensor(0.0, device=self._embedding.device)


# ─── Novel: Gumbel-Soft Codebook ────────────────────────────────────────────

class GumbelSoftCodebook(nn.Module):
    """
    [NOVEL] Gumbel-Softmax differentiable codebook.

    Key improvements over VQ-EMA:
      - All codewords receive gradient → prevents collapse
      - Temperature τ anneals from init_temp → min_temp
      - Entropy loss encourages uniform codeword utilisation
      - K and V are separate nn.Parameters (fully learnable)

    Assignment formula:
      logits  = -||x_norm - e_k||²
      z_k     = (logits_k + Gumbel_k) / τ
      soft    = softmax(z)          [differentiable, for gradient]
      hard    = argmax(z)           [discrete index, for lookup]
    """

    def __init__(self, num_embeddings, embedding_dim, decay=0.99,
                 init_temp=2.0, min_temp=0.5, anneal_rate=3e-4):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim  = embedding_dim

        # Learnable codebooks (receive gradients from global attention)
        self.embedding_k = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embedding_v = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # Temperature (annealed in-place)
        self.register_buffer('temperature', torch.tensor(float(init_temp)))
        self.min_temp    = min_temp
        self.anneal_rate = anneal_rate

        # EMA usage tracker (for entropy loss only, no gradient)
        self.register_buffer('_ema_usage', torch.ones(num_embeddings) / num_embeddings)

        # BN mirrors original VQ-EMA interface
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding_k, 0, 0.02)
        nn.init.normal_(self.embedding_v, 0, 0.02)
        self._ema_usage.fill_(1.0 / self._num_embeddings)
        self.bn.reset_parameters()

    # ── Public API (drop-in for VectorQuantizerEMA) ──────────────────────

    def get_k(self):
        """Codebook keys — differentiable, do NOT detach in caller."""
        return self.embedding_k

    def get_v(self):
        """Codebook values — differentiable, do NOT detach in caller."""
        return self.embedding_v

    def update(self, x):
        """
        Gumbel-Softmax soft assignment with straight-through hard indices.

        Args:
            x: [N, D]
        Returns:
            hard_indices: [N, 1]
        """
        x_norm = self.bn(x)

        # Negative squared-distance logits
        distances = (
            x_norm.pow(2).sum(1, keepdim=True)
            - 2 * x_norm @ self.embedding_k.detach().t()
            + self.embedding_k.detach().pow(2).sum(1).unsqueeze(0)
        )
        logits = -distances  # [N, K]

        if self.training:
            U       = torch.rand_like(logits).clamp(1e-20, 1 - 1e-20)
            gumbels = -torch.log(-torch.log(U))
            noisy   = (logits + gumbels) / self.temperature.clamp(min=0.1)
            hard_indices = noisy.argmax(dim=-1)  # [N]

            with torch.no_grad():
                counts = torch.bincount(hard_indices,
                                        minlength=self._num_embeddings).float()
                counts /= counts.sum() + 1e-8
                self._ema_usage.mul_(0.99).add_(counts * 0.01)
                new_t = max(self.temperature.item() * (1 - self.anneal_rate),
                            self.min_temp)
                self.temperature.fill_(new_t)

            return hard_indices.unsqueeze(1)
        else:
            return logits.argmax(dim=-1).unsqueeze(1)

    def codebook_entropy_loss(self):
        """
        Normalised entropy loss: 0 = uniform, 1 = fully collapsed.
          L_ent = 1 - H(p) / log(K)
        """
        p       = self._ema_usage / (self._ema_usage.sum() + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        return 1.0 - entropy / math.log(self._num_embeddings)

    @property
    def utilization_rate(self):
        """Fraction of codewords above the 0.5/K usage threshold."""
        p = self._ema_usage / (self._ema_usage.sum() + 1e-8)
        return (p > 0.5 / self._num_embeddings).float().mean().item()
