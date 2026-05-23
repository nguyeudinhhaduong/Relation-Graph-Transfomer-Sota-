# RelGT++

Cải tiến của [Relational Graph Transformer](https://arxiv.org/abs/2505.10960) với **6 novel contributions**, không cần thay đổi training procedure.

---

## Novel Contributions

| # | Module | File | Cải tiến |
|---|--------|------|----------|
| 1 | **CrossModalGatedFusion** | `model.py` | Dynamic per-modality gating thay concat+MLP |
| 2 | **GumbelSoftCodebook** | `codebook.py` | Differentiable VQ, ngăn codebook collapse |
| 3 | **CrossAttentionBridge** | `model.py` | Bidirectional local↔global information flow |
| 4 | **MultiViewReadout** | `local_module.py` | Attention + mean + max aggregation |
| 5 | **SwiGLU Gated FFN** | `local_module.py` | Gated activation, gradient flow tốt hơn |
| 6 | **Stochastic Depth** | `local_module.py` | Drop-path regularization |

---

## Cài đặt

### Linux / Mac
```bash
micromamba create -n relgt python=3.12
micromamba activate relgt
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Windows (Conda)
```powershell
conda create -y -n relgt_tf python=3.10 pip
conda activate relgt_tf
pip install torch torchvision torchaudio
pip install torch-geometric relbench sentence-transformers h5py pynvml wandb einops tqdm pandas scikit-network ogb pytorch-frame
```

---

## Quick Start — Smoke Check (single process)

```bash
python main_node_single_check.py \
    --dataset rel-f1 \
    --task driver-top3 \
    --epochs 1 \
    --batch_size 16 \
    --channels 64 \
    --num_neighbors 64 \
    --num_centroids 256 \
    --num_workers 1 \
    --max_steps_per_epoch 10 \
    --out_dir results/smoke_check
```

Kết quả: `results/smoke_check/rel-f1/driver-top3/42.json`

---

## Full Training — DDP

```bash
# Single GPU
python main_node_ddp.py \
    --dataset rel-f1 --task driver-top3 \
    --epochs 10 --batch_size 512 --channels 512 \
    --num_heads 4 --num_neighbors 300 --num_centroids 4096 \
    --ff_dropout 0.1 --attn_dropout 0.1 --lr 1e-4 \
    --max_steps_per_epoch 3000 --num_workers 2 --seed 42

# Multi-GPU
torchrun --nproc_per_node=8 main_node_ddp.py [same args]
```

### Windows DDP
```powershell
$env:WANDB_MODE="offline"; $env:MASTER_ADDR="127.0.0.1"
$env:MASTER_PORT="29631"; $env:LOCAL_RANK="0"
$env:RANK="0"; $env:WORLD_SIZE="1"

python main_node_ddp.py `
    --dataset rel-f1 --task driver-top3 `
    --epochs 10 --batch_size 16 --channels 64 `
    --num_neighbors 64 --num_centroids 256 `
    --num_workers 0 --max_steps_per_epoch 10 `
    --out_dir results/debug
```

---

## Auxiliary Losses (thay đổi duy nhất cần trong training)

```python
# Thêm 2 dòng vào training loop (đã có sẵn trong main_node_ddp.py):
task_loss  = loss_fn(pred, labels)
aux_loss   = model.get_aux_loss()      # codebook entropy + local-global agreement
total_loss = task_loss + aux_loss
total_loss.backward()
```

---

## File Structure

```
relgt_plus_plus/
├── codebook.py                  # [NOVEL] GumbelSoftCodebook
├── local_module.py              # [NOVEL] MultiViewReadout, SwiGLU, StochasticDepth
├── model.py                     # [NOVEL] CrossModalGatedFusion, CrossAttentionBridge
├── encoders.py                  # (unchanged from original)
├── utils.py                     # (unchanged from original)
├── main_node_ddp.py             # DDP training — aux_loss integrated
├── main_node_single_check.py    # Single-process smoke test
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@article{dwivedi2025relational,
  title={Relational Graph Transformer},
  author={Dwivedi, Vijay Prakash and Jaladi, Sri and Shen, Yangyi and
          L{\'o}pez, Federico and Kanatsoulis, Charilaos I and Puri, Rishi
          and Fey, Matthias and Leskovec, Jure},
  journal={arXiv preprint arXiv:2505.10960},
  year={2025}
}
```
