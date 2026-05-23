# Phân tích kỹ thuật: RelGT → RelGT++

> Tài liệu này phân tích **từng thay đổi cụ thể** trong code, lý do kỹ thuật, công thức toán học, và tác động dự kiến đến performance.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Thay đổi 1: VQ-EMA → GumbelSoftCodebook](#2-thay-đổi-1-vq-ema--gumbelsoftcodebook)
3. [Thay đổi 2: Concat+MLP → CrossModalGatedFusion](#3-thay-đổi-2-concatmlp--crossmodalgatedfusion)
4. [Thay đổi 3: Concat Fusion → CrossAttentionBridge](#4-thay-đổi-3-concat-fusion--crossattentionbridge)
5. [Thay đổi 4: Attention Readout → MultiViewReadout](#5-thay-đổi-4-attention-readout--multiviewreadout)
6. [Thay đổi 5: GELU FFN → SwiGLU GatedFFN](#6-thay-đổi-5-gelu-ffn--swiglu-gatedffn)
7. [Thay đổi 6: Dropout → Stochastic Depth](#7-thay-đổi-6-dropout--stochastic-depth)
8. [Thay đổi 7: Auxiliary Losses](#8-thay-đổi-7-auxiliary-losses)
9. [So sánh số tham số](#9-so-sánh-số-tham-số)
10. [Tóm tắt](#10-tóm-tắt)

---

## 1. Tổng quan kiến trúc

### Luồng dữ liệu — RelGT gốc

```
[type, hop, time, features, PE]  ← 5 modalities [B, K, C] mỗi cái
         ↓
  Concat → [B, K, 5C]
  Linear(5C → C)
  LayerNorm            ← in_mixture (naive)
         ↓
     LocalModule       ← Transformer trên K neighbors
         ↓             ← output [B, C]
  VectorQuantizerEMA   ← EMA-updated centroids (no gradient)
  GlobalAttn           ← attention to codebook [B, C]
         ↓
  Concat([local; global]) → [B, 2C]
  Linear(2C → C)           ← simple concat fusion
         ↓
     MLP Head → prediction
```

### Luồng dữ liệu — RelGT++

```
[type, hop, time, features, PE]  ← 5 modalities [B, K, C] mỗi cái
         ↓
  CrossModalGatedFusion           ← [NOVEL] per-modality gating
  → [B, K, C]
         ↓
  LocalModule (SwiGLU + StochDepth) ← [NOVEL] improved transformer
  → [B, C]
         ↓
  GumbelSoftCodebook               ← [NOVEL] differentiable centroids
  GlobalAttn                       ← attention to learnable codebook [B, C]
         ↓
  CrossAttentionBridge             ← [NOVEL] bidirectional fusion
  → [B, C]
         ↓
     MLP Head → prediction
     + get_aux_loss()              ← [NOVEL] entropy + agreement
```

---

## 2. Thay đổi 1: VQ-EMA → GumbelSoftCodebook

**File:** `codebook.py`

### Vấn đề của VQ-EMA

VQ-EMA (Vector Quantization with Exponential Moving Average) là kỹ thuật từ VQ-VAE (van den Oord 2017). Codewords được update theo EMA:

```python
# VQ-EMA — KHÔNG có gradient
self._ema_cluster_size = decay * cluster_size + (1 - decay) * counts
self._ema_w = decay * ema_w + (1 - decay) * dw
self._embedding = ema_w / cluster_size   # ← buffer, không phải Parameter
```

**Vấn đề cụ thể:**

| Vấn đề | Giải thích |
|--------|-----------|
| **Codebook collapse** | Nhiều codewords không bao giờ được assign → chỉ vài codewords active, phần còn lại "chết" |
| **No gradient flow** | `_embedding` là buffer → optimizer không cập nhật → chỉ EMA update |
| **Hard assignment** | `argmin(distance)` là non-differentiable → gradient bị block tại bước quantization |
| **Shared K=V** | `get_k()` và `get_v()` trả về cùng tensor `_embedding_output` |

### Giải pháp: GumbelSoftCodebook

**Ý tưởng cốt lõi:** Dùng Gumbel-Softmax để tạo assignment **differentiable** trong khi vẫn lấy discrete index khi cần.

#### Công thức đầy đủ

**Bước 1 — Tính logits từ khoảng cách:**
```
distances_{ik} = ||x_i - e_k||² 
               = ||x_i||² - 2·x_i·e_k^T + ||e_k||²

logits_{ik} = -distances_{ik}
```

**Bước 2 — Thêm Gumbel noise:**
```
u ~ Uniform(0, 1)
g_k = -log(-log(u_k))   ← Gumbel(0,1) sample

noisy_k = (logits_k + g_k) / τ   ← τ là temperature
```

**Bước 3 — Discrete index (straight-through):**
```
hard_index = argmax_k(noisy_k)   ← discrete, dùng để update c_idx
```

**Tại sao gradient flows?** Trong `global_forward`, codebook được dùng qua attention:
```python
k_x = self.vq.get_k()   # embedding_k là nn.Parameter → có gradient
v_x = self.vq.get_v()   # embedding_v là nn.Parameter → có gradient
k = self.lin_key_g(k_x)
v = self.lin_value_g(v_x)
# Attention output phụ thuộc vào embedding_k và embedding_v
# → gradient flows back qua attention weights
```

#### Code so sánh

```python
# TRƯỚC — VQ-EMA (codebook.py gốc)
self.register_buffer("_embedding", torch.randn(num_embeddings, embedding_dim))
# _embedding là buffer → không có gradient
# update chỉ qua EMA:
self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

# SAU — GumbelSoftCodebook (codebook.py mới)
self.embedding_k = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
self.embedding_v = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
# nn.Parameter → optimizer cập nhật qua backprop
# Gradient đến từ attention output
```

#### Temperature Annealing

```python
# Mỗi training step, temperature giảm dần:
new_t = max(temperature * (1 - anneal_rate), min_temp)
# anneal_rate = 3e-4
# τ: 2.0 → 1.94 → ... → 0.5 (min)
```

**Tại sao cần annealing?**
- Khi τ cao (2.0): soft assignment → nhiều codewords nhận gradient → exploration
- Khi τ thấp (0.5): near-hard assignment → gần với discrete VQ → exploitation
- Annealing cho phép model học dần từ soft sang hard

#### Entropy Loss để chống collapse

```python
def codebook_entropy_loss(self):
    p = self._ema_usage / (self._ema_usage.sum() + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum()
    # Normalized: 0 = perfectly uniform, 1 = fully collapsed
    return 1.0 - entropy / math.log(self._num_embeddings)
```

Nếu chỉ 1 codeword được dùng: `p = [1, 0, 0, ..., 0]` → `H = 0` → loss = 1.0 (phạt tối đa)

Nếu tất cả dùng đều nhau: `p = [1/K, ..., 1/K]` → `H = log(K)` → loss = 0.0 (không phạt)

---

## 3. Thay đổi 2: Concat+MLP → CrossModalGatedFusion

**File:** `model.py`, class `CrossModalGatedFusion`

### Vấn đề của Concat+MLP

```python
# RelGT gốc — model.py
# 5 modalities [B, K, C] → concat → [B, K, 5C] → Linear → [B, K, C]
x = torch.cat([e_type, e_hop, e_time, e_tfs, e_pe], dim=-1)  # [B, K, 5C]
x = self.lin_in(x)   # Linear(5C, C) — treats all modalities equally
```

**Vấn đề cụ thể:**

| Vấn đề | Giải thích |
|--------|-----------|
| **Equal weighting** | Linear layer xử lý tất cả modalities như nhau, không phân biệt importance |
| **No context** | Không biết context của các modalities khác khi xử lý modality hiện tại |
| **Static weights** | Weights cố định sau training, không adaptive theo từng sample/token |
| **Dimension explosion** | 5C → projection cost tăng tuyến tính với số modalities |

### Giải pháp: CrossModalGatedFusion

**Ý tưởng:** Mỗi modality tự quyết định mình quan trọng bao nhiêu dựa trên nội dung của chính nó **và** context từ các modalities khác.

#### Công thức

```
Cho N modalities, e_1, ..., e_N ∈ [B, K, C]:

ctx_i = (Σ_{j≠i} e_j) / (N-1)        ← mean context từ các modality khác

gate_i = σ( W_self_i(e_i) + W_ctx_i(ctx_i) )   ← sigmoid gate ∈ (0,1)

val_i  = W_val_i(e_i)                  ← project value

fused  = Σ_i gate_i ⊙ val_i           ← element-wise weighted sum

output = LayerNorm( W_out(fused) )
```

#### Code chi tiết

```python
# CrossModalGatedFusion.forward()
total = sum(embeddings)   # [B, K, C] — tổng tất cả modalities

for i, e_i in enumerate(embeddings):
    # Tính context: trung bình của tất cả modalities trừ modality i
    ctx  = (total - e_i) / max(N - 1, 1)       # [B, K, C]
    
    # Gate: học từ cả nội dung của e_i lẫn context
    gate = torch.sigmoid(
        self.self_projs[i](e_i)    # "tôi là ai?"
        + self.ctx_projs[i](ctx)   # "các modalities khác muốn gì?"
    )                                           # [B, K, C] ∈ (0,1)
    
    # Weighted value
    fused += gate * self.val_projs[i](e_i)      # [B, K, C]
```

#### Ví dụ minh họa

Giả sử sample là node có **thời gian rất gần** với seed:
- `e_time` mang tín hiệu thời gian mạnh → gate_time cao
- Các modalities khác biết điều này qua context → điều chỉnh gates tương ứng
- Kết quả: time embedding đóng góp nhiều hơn vào `fused`

Trong khi đó, với node chỉ có **structural information** (type và hop quan trọng):
- gate_type và gate_hop cao
- gate_time thấp hơn

#### Số tham số

```
Mỗi modality: 3 Linear(C, C) → 3C² params
N = 5 modalities: 15C² params
+ out_proj(C, C): C² params
+ LayerNorm: 2C params
Tổng: 16C² + 2C

Vs. Concat+MLP: Linear(5C, C) = 5C² + C params
→ CrossModalGatedFusion tốn ~3x params hơn nhưng adaptive
```

---

## 4. Thay đổi 3: Concat Fusion → CrossAttentionBridge

**File:** `model.py`, class `CrossAttentionBridge`

### Vấn đề của Concat Fusion

Trong RelGT gốc (file `model.py` gốc), sau khi có local và global output:

```python
# RelGT gốc
out = torch.cat([out_local, out_global], dim=-1)  # [B, 2C]
out = self.lin_out(out)   # Linear(2C, C)
```

**Vấn đề:**

| Vấn đề | Giải thích |
|--------|-----------|
| **Independent branches** | Local và global không "giao tiếp" trước khi merge |
| **Passive combination** | Linear projection không biết **cái gì** trong local cần được enriched bởi global và ngược lại |
| **Output size tăng gấp đôi** | Cần Linear(2C → C) → parameter overhead |
| **No selective reading** | Phải đọc toàn bộ cả 2 vectors, không filter được |

### Giải pháp: CrossAttentionBridge

**Ý tưởng:** Local "hỏi" global về những gì mình chưa biết, và ngược lại. Sau đó dùng gate để quyết định balance.

#### Công thức đầy đủ

```
Cho local ∈ [B, C] và global ∈ [B, C]:

scale = 1 / √C

# Local queries global (local muốn học gì từ global?)
q1 = W_q_l(local)                          # [B, C]
k1 = W_k_g(global)                         # [B, C]
v1 = W_v_g(global)                         # [B, C]
α  = σ( (q1 · k1) * scale )                # [B, 1] scalar attention weight
enriched_l = local + α * v1                # local được "bổ sung" từ global

# Global queries local (global muốn biết gì từ local?)
q2 = W_q_g(global)
k2 = W_k_l(local)
v2 = W_v_l(local)
β  = σ( (q2 · k2) * scale )               # [B, 1]
enriched_g = global + β * v2

# Gated merge
gate   = σ( W_gate([enriched_l ‖ enriched_g]) )   # [B, C] ∈ (0,1)
merged = gate ⊙ enriched_l + (1-gate) ⊙ enriched_g

# Residual connection + LayerNorm
output = LayerNorm( W_out(Dropout(merged)) + local )
```

#### Tại sao dùng scalar attention (dot product) thay vì full attention?

Vì local và global đều là vector 1D `[B, C]` (đã được aggregate rồi), không phải sequence. Full multi-head attention không apply được. Scalar dot product đủ để học similarity.

```python
# Scalar attention gate
alpha = torch.sigmoid(
    (self.q_l(local_out) * self.k_g(global_out)).sum(-1, keepdim=True) * scale
)  # shape: [B, 1]
enriched_l = local_out + alpha * self.v_g(global_out)
```

#### Ưu điểm so với concat

| Aspect | Concat+Linear | CrossAttentionBridge |
|--------|---------------|---------------------|
| Thông tin flow | Unidirectional | Bidirectional |
| Output dim | Cần Linear(2C→C) | Giữ nguyên C |
| Selectivity | Không (toàn bộ vectors) | Có (α và β gates) |
| Residual | Không có | Có (+ local_out) |
| Expressiveness | Thấp | Cao hơn |

---

## 5. Thay đổi 4: Attention Readout → MultiViewReadout

**File:** `local_module.py`, class `MultiViewReadout`

### Vấn đề của single-view readout

Trong RelGT gốc, sau khi local transformer chạy xong:

```python
# local_module.py gốc
node_tensor     = output[:, 0, :]       # seed node [B, D]
neighbor_tensor = output[:, 1:, :]      # neighbors [B, K-1, D]

# Chỉ 1 aggregation: attention-weighted
target      = node_tensor.unsqueeze(1).expand_as(neighbor_tensor)
attn_scores = self.attn_layer(torch.cat([target, neighbor_tensor], dim=-1))
attn_weights = F.softmax(attn_scores, dim=1)
v = (neighbor_tensor * attn_weights).sum(dim=1)   # [B, D]
```

**Vấn đề:**

| Vấn đề | Giải thích |
|--------|-----------|
| **Inductive bias mạnh** | Attention giả định rằng chỉ vài neighbor quan trọng → bỏ qua global statistics |
| **Sensitive to outliers** | Nếu attention tập trung vào neighbor "nhiễu", output bị ảnh hưởng lớn |
| **Không capture statistics** | Mean của neighborhood là thông tin hữu ích (e.g., average activity level) |
| **Bỏ qua extreme values** | Max pooling bắt được features đặc biệt nổi bật nhất |

### Giải pháp: MultiViewReadout

**Ý tưởng:** Kết hợp 3 aggregation strategies để capture complementary information.

#### Công thức

```
Cho transformer output O ∈ [B, K, D]:
  seed      = O[:, 0, :]      [B, D]   — seed node
  neighbors = O[:, 1:, :]     [B, K-1, D]

# View 1: Attention-weighted (quan hệ với seed)
scores    = W_score([seed_expanded ‖ neighbors])  → [B, K-1, 1]
weights   = softmax(scores, dim=1)                 → [B, K-1, 1]
v_attn    = Σ_k weights_k * neighbors_k           → [B, D]

# View 2: Mean (thống kê toàn cục)
v_mean = (1/(K-1)) * Σ_k neighbors_k              → [B, D]

# View 3: Max (feature nổi bật nhất)
v_max  = max_k(neighbors_k)                        → [B, D]

# Combine
combined = [v_attn ‖ v_mean ‖ v_max]              → [B, 3D]
fused    = Linear(2D, D)(GELU(Linear(3D, 2D)(combined)))  → [B, D]

# Residual với seed
output = LayerNorm(seed + fused)                   → [B, D]
```

#### Mỗi view bắt được gì?

| View | Bắt được | Inductive Bias |
|------|----------|----------------|
| **v_attn** | Những neighbor liên quan nhất đến seed | "Neighbor nào giống tôi nhất?" |
| **v_mean** | Đặc điểm trung bình của neighborhood | "Neighborhood này thường như thế nào?" |
| **v_max** | Feature nổi bật/cực trị nhất | "Feature đặc biệt nhất trong neighborhood là gì?" |

#### Ví dụ: Fraud detection

- `v_attn`: neighbors có behavior pattern giống user → xác định "peer group"
- `v_mean`: mức độ activity trung bình của neighborhood → baseline normal
- `v_max`: transaction amount lớn nhất trong neighborhood → anomaly signal

Kết hợp cả 3 → richer representation hơn chỉ dùng attention.

---

## 6. Thay đổi 5: GELU FFN → SwiGLU GatedFFN

**File:** `local_module.py`, class `GatedFFN`

### Vấn đề của standard GELU FFN

```python
# local_module.py gốc — FeedForwardNetwork
self.ffn_net = nn.Sequential(
    nn.Linear(hidden_size, ffn_size),   # expand: C → 2C
    nn.GELU(),
    nn.Dropout(dropout_rate),
    nn.Linear(ffn_size, hidden_size),   # project: 2C → C
    nn.Dropout(dropout_rate),
)
```

**GELU activation:** `GELU(x) = x · Φ(x)` trong đó `Φ` là CDF của Normal distribution.

**Vấn đề:**
- Activation không có "control mechanism" — toàn bộ hidden dimension đi qua
- Không có learnable gating → model không thể chọn "bỏ qua" thông tin không cần thiết
- Gradient có thể bị saturate ở vùng GELU gần 0

### Giải pháp: SwiGLU GatedFFN

**Công thức SwiGLU (Shazeer 2020):**

```
FFN_SwiGLU(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )
```

Trong đó:
- `SiLU(x) = x · σ(x)` (Swish activation)
- `⊙` là element-wise multiplication
- `W_gate, W_up: C → ffn_size`
- `W_down: ffn_size → C`

#### Code so sánh

```python
# TRƯỚC — Standard GELU FFN
x = Linear(C, ffn_size)(x)   # expand
x = GELU(x)                  # activate (all neurons)
x = Linear(ffn_size, C)(x)   # project

# SAU — SwiGLU GatedFFN
gate = SiLU(W_gate(x))        # learned gate signal
up   = W_up(x)                # content
x    = W_down(gate * up)      # gated content
```

#### Tại sao tốt hơn?

**1. Gating mechanism:**
- `gate` học cách "mở/đóng" từng dimension của hidden state
- Nếu `gate_i ≈ 0` → dimension `i` bị suppress → model có thể bỏ qua noise
- Nếu `gate_i ≈ 1` → dimension `i` pass through nguyên vẹn

**2. Gradient flow:**
- GELU có gradient gần 0 khi `x << 0` → vanishing gradient
- SiLU có gradient tốt hơn trong khoảng rộng hơn
- Multiplication `gate * up` tạo thêm paths cho gradient

**3. Effective expansion:**
- Standard FFN: `C → 2C → C` (2x expansion)
- GatedFFN: `C → (ffn_size + ffn_size) → C` nhưng chỉ 1 tensor đi vào `W_down`
- Trong code: `GatedFFN(hidden_size, ffn_size * 2, ...)` → `ffn_size * 2` là kích thước của cả gate và up
- Effective: `C → 4C` (gate) + `C → 4C` (up) → sau gating → `C → 2C → C` ≈ 4x effective expansion

**Empirical evidence:** PaLM, LLaMA, Mistral, Gemma đều dùng SwiGLU → proven effectiveness.

---

## 7. Thay đổi 6: Dropout → Stochastic Depth

**File:** `local_module.py`, class `StochasticDepth`

### Dropout vs Stochastic Depth

| | Dropout | Stochastic Depth |
|---|---------|-----------------|
| Drop unit | Individual neurons | Toàn bộ residual branch |
| Granularity | Fine-grained | Coarse-grained |
| Effect khi training | Random neurons → model học redundant features | Random layers → model học shorter paths |
| Effect khi inference | Được "turn off" (scale instead) | Layer luôn active |
| Invented for | Fully connected networks | Deep residual networks |

### Công thức Stochastic Depth

```python
def forward(self, x):
    if not self.training or self.drop_prob == 0.0:
        return x   # inference: không drop
    keep_prob = 1.0 - self.drop_prob
    keep = torch.rand(1, device=x.device) < keep_prob
    return x * keep.float() / keep_prob   # unbiased rescaling
```

Được áp dụng như residual regularization:
```python
# Trước (standard residual):
x = residual + sub_layer(x)

# Sau (stochastic depth):
x = residual + StochasticDepth(p)(sub_layer(x))
# Khi training: sub_layer bị drop với prob p
# Khi inference: sub_layer luôn được dùng
```

### Linearly increasing drop rates

```python
# local_module.py
drop_rates = [0.05 * (i + 1) / max(n_layers, 1) for i in range(n_layers)]
# Layer 0: 0.05, Layer 1: 0.10, Layer 2: 0.15, ...
```

**Lý do tăng dần:** Layers đầu học low-level features quan trọng → drop rate thấp.
Layers cuối học high-level abstraction → drop rate cao hơn để regularize.

### Tác động kỳ vọng

Khi model có `n_layers=1` (mặc định trong RelGT), stochastic depth có `drop_prob=0.05`.
- 95% training steps: sub_layer được dùng bình thường
- 5% training steps: sub_layer bị skip → gradient không đi qua → implicit regularization

---

## 8. Thay đổi 7: Auxiliary Losses

**File:** `model.py`, `codebook.py`

### Codebook Entropy Loss

**Được tính trong:** `GumbelSoftCodebook.codebook_entropy_loss()`
**Được emit từ:** `RelGTLayer.global_forward()` với weight `0.1`

```python
# codebook.py
def codebook_entropy_loss(self):
    p       = self._ema_usage / (self._ema_usage.sum() + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum()
    # H_max = log(K) khi uniform distribution
    return 1.0 - entropy / math.log(self._num_embeddings)
```

**Ý nghĩa hình học:**

```
Codebook collapse scenario (xấu):
  p = [0.9, 0.05, 0.03, 0.02, 0.0, ..., 0.0]
  entropy ≈ 0.6 nats
  L_ent = 1 - 0.6 / log(4096) ≈ 1 - 0.073 = 0.927  ← loss cao, penalize mạnh

Codebook uniform scenario (tốt):
  p = [1/K, 1/K, ..., 1/K]
  entropy = log(K) nats
  L_ent = 1 - log(K) / log(K) = 0.0  ← không penalize
```

### Local-Global Agreement Loss

**Được tính trong:** `RelGTLayer.forward()` khi `conv_type == "full"`

```python
# model.py
with torch.no_grad():
    cos = F.cosine_similarity(out_l, out_g, dim=-1).mean()
self._aux_losses['local_global_agreement'] = (1.0 - cos) * 0.01
```

**Tại sao cosine similarity?**
- Measure alignment của hướng, không phải magnitude
- Nếu local và global "đồng ý" về direction → cos ≈ 1 → loss ≈ 0
- Nếu hoàn toàn ngược chiều → cos ≈ -1 → loss ≈ 0.02

**Weight `0.01` nhỏ — lý do:**
- Loss này chỉ là soft constraint, không phải objective chính
- Không muốn force local và global phải giống nhau quá → giảm diversity
- Chỉ muốn "không quá mâu thuẫn"

**Chú ý `torch.no_grad()`:**
- Không muốn gradient từ loss này ảnh hưởng đến local và global outputs trực tiếp
- Chỉ dùng cos_sim như một "monitoring signal" để regularize thông qua loss value

### Tổng hợp auxiliary losses trong training

```python
# main_node_ddp.py
task_loss  = loss_fn(pred.float(), labels)

# model.get_aux_loss() = Σ conv._aux_losses
# _aux_losses contains:
#   "layer0_codebook_entropy"      = entropy_loss * 0.1
#   "layer0_local_global_agreement" = (1 - cos_sim) * 0.01

aux_loss   = model.module.get_aux_loss() * args.aux_loss_weight
total_loss = task_loss + aux_loss
```

---

## 9. So sánh số tham số

Với `channels = C = 512`, `num_centroids = K = 4096`, `num_modalities = 5`:

### Codebook

| Component | RelGT gốc | RelGT++ |
|-----------|-----------|---------|
| Codebook | buffer `[K, C]` = `4096×512` (không train) | `nn.Parameter [K, C] × 2` = `2×4096×512` (có gradient) |
| BN | `nn.BatchNorm1d(C)` | `nn.BatchNorm1d(C)` |
| **Trainable params** | `~1K` (BN only) | `~4.2M` |

### Modality Fusion

| Component | RelGT gốc | RelGT++ |
|-----------|-----------|---------|
| Fusion | `Linear(5C, C)` = `5×512×512` | `CrossModalGatedFusion` = `16C² + 2C` |
| **Params** | `1.31M` | `4.20M` |

### Local Module (per layer)

| Component | RelGT gốc | RelGT++ |
|-----------|-----------|---------|
| FFN | `Linear(C, 2C)` + `Linear(2C, C)` | `Linear(C, 4C) × 2` + `Linear(4C, C)` |
| **FFN params** | `2C²` | `4C²` (gate) + `4C²` (up) + `4C²` (down) = `12C²` |
| Readout | `Linear(2C, 1)` | `Linear(2C, 1)` + `Linear(3C, 2C)` + `Linear(2C, C)` |
| **Readout params** | `2C` | `8C² + 4C` |

### Fusion

| Component | RelGT gốc | RelGT++ |
|-----------|-----------|---------|
| Fusion | `Linear(2C, C)` | `CrossAttentionBridge`: `8×C²` + `LayerNorm` |
| **Params** | `C²` | `8C²` |

### Tổng hợp (C=512)

| Module | RelGT gốc | RelGT++ | Tỉ lệ |
|--------|-----------|---------|--------|
| Codebook | ~1K | ~4.2M | 4000× |
| Modality Fusion | 1.31M | 4.20M | 3.2× |
| FFN | 0.52M | 3.14M | 6× |
| Readout | 1K | 2.1M | 2100× |
| Fusion | 0.26M | 2.1M | 8× |
| **Tổng tăng thêm** | baseline | **~10M** | ~2–3× overall |

---

## 10. Tóm tắt

### Bảng tổng hợp toàn bộ thay đổi

| # | Thay đổi | File | Dòng code | Vấn đề giải quyết | Kỹ thuật |
|---|----------|------|-----------|-------------------|----------|
| 1 | VQ-EMA → GumbelSoftCodebook | `codebook.py` | L84–191 | Codebook collapse, no gradient | Gumbel-Softmax, temperature annealing, nn.Parameter |
| 2 | Concat+MLP → CrossModalGatedFusion | `model.py` | L42–93 | Static equal-weight fusion | Per-modality sigmoid gates, context-aware |
| 3 | Concat → CrossAttentionBridge | `model.py` | L100–165 | Independent local/global branches | Bidirectional dot-product attention, gated merge |
| 4 | Attention → MultiViewReadout | `local_module.py` | L174–228 | Single inductive bias | Ensemble: attn + mean + max |
| 5 | GELU FFN → SwiGLU GatedFFN | `local_module.py` | L39–73 | No gating, gradient issues | SwiGLU: SiLU gate × up content |
| 6 | Dropout → Stochastic Depth | `local_module.py` | L22–34 | Overfitting deep layers | Layer-level drop, linearly increasing rates |
| 7 | No aux loss → Entropy + Agreement | `codebook.py`, `model.py` | L178–185, L264–267 | Codebook collapse, branch divergence | Entropy regularization, cosine similarity |

### Thay đổi nào quan trọng nhất?

**Rank theo mức độ impact kỳ vọng:**

1. 🥇 **GumbelSoftCodebook** — Trực tiếp giải quyết fundamental problem của VQ (collapse). Codebook là "memory" của toàn bộ graph, nếu collapse thì global branch mất tác dụng.

2. 🥈 **CrossAttentionBridge** — Local và global carry complementary information. Bidirectional exchange khai thác điều này tốt hơn passive concat.

3. 🥉 **CrossModalGatedFusion** — 5 modalities không bao giờ quan trọng đều nhau. Dynamic gating cho phép model adapt theo từng sample.

4. **MultiViewReadout** — Mean và max là free information bị bỏ qua trong original. Kết hợp 3 views = ensemble effect.

5. **SwiGLU FFN** — Proven improvement trên nhiều architectures, consistent gain ~0.5–1%.

6. **Stochastic Depth** — Regularization effect, đặc biệt hữu ích khi overfitting.

7. **Auxiliary Losses** — Soft constraints, giúp training ổn định hơn, không phải objective chính.

### Điểm cần lưu ý khi chạy

```
1. Codebook utilization_rate < 30% sau 5 epochs → tăng entropy loss weight (0.1 → 0.2)
2. aux_loss >> task_loss → giảm aux_loss_weight argument
3. Training loss oscillate → giảm drop_path_rate (0.05 → 0.02)
4. Memory OOM → CrossModalGatedFusion là module nặng nhất, thử reduce channels
```

---

> **References:**
> - Jang et al. 2017 — Categorical Reparameterization with Gumbel-Softmax
> - van den Oord et al. 2017 — Neural Discrete Representation Learning (VQ-VAE)
> - Shazeer 2020 — GLU Variants Improve Transformer
> - Huang et al. 2016 — Deep Networks with Stochastic Depth
> - Dwivedi et al. 2025 — Relational Graph Transformer (arXiv:2505.10960)
