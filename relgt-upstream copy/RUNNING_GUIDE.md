# RelGT Running Guide (Windows-first)

This guide documents reliable commands to run this repository from terminal.

## 1) Go to repo folder

```powershell
cd "E:\Relation Graph Transformer\relgt-upstream"
```

## 2) Use conda env

If your env already exists:

```powershell
conda activate relgt_tf
```

If not, create one quickly:

```powershell
conda create -y -n relgt_tf python=3.10 pip
conda activate relgt_tf
pip install torch torchvision torchaudio
pip install torch-geometric relbench sentence-transformers h5py pynvml wandb einops tqdm pandas scikit-network ogb pytorch-frame
```

## 3) Quick sanity run (recommended first)

Run single-process smoke check with [main_node_single_check.py](main_node_single_check.py):

```powershell
conda run --no-capture-output -n relgt_tf python main_node_single_check.py --dataset rel-f1 --task driver-top3 --epochs 1 --batch_size 16 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 1 --max_steps_per_epoch 10 --precompute 1 --out_dir results/smoke_check
```

Output JSON:

- [results/smoke_check/rel-f1/driver-top3/42.json](results/smoke_check/rel-f1/driver-top3/42.json)

## 4) 10-epoch run (single-check)

```powershell
conda run --no-capture-output -n relgt_tf python main_node_single_check.py --dataset rel-f1 --task driver-top3 --epochs 10 --batch_size 16 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 1 --max_steps_per_epoch 10 --precompute 1 --out_dir results/smoke_check
```

## 5) Run original repo entrypoint (DDP script)

Use [main_node_ddp.py](main_node_ddp.py) with single-process distributed env on Windows:

```powershell
$env:WANDB_MODE="offline"
$env:MASTER_ADDR="127.0.0.1"
$env:MASTER_PORT="29631"
$env:LOCAL_RANK="0"
$env:RANK="0"
$env:WORLD_SIZE="1"

conda run --no-capture-output -n relgt_tf python main_node_ddp.py --dataset rel-f1 --task driver-top3 --epochs 10 --batch_size 16 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 0 --max_steps_per_epoch 10 --out_dir results/smoke_check_ddp --run_name smoke_ddp
```

Why num_workers=0 for this script on Windows:

- Worker spawn can re-import top-level code and re-trigger distributed initialization.
- This can cause socket bind errors on MASTER_PORT.

Output JSON:

- [results/smoke_check_ddp/rel-f1/driver-top3/42.json](results/smoke_check_ddp/rel-f1/driver-top3/42.json)

## 6) Common errors and fixes

### Error: use_libuv was requested but PyTorch was built without libuv support

Typical when using torchrun on Windows.

Fix:

- Prefer `python main_node_ddp.py` with explicit env vars (section 5).
- Avoid torchrun for this local Windows setup.

### Error: Distributed package does not have NCCL built in

Fix:

- Use the patched [main_node_ddp.py](main_node_ddp.py) that falls back to gloo on Windows/CPU.

### Error: server socket failed to bind / port already in use

Fix:

- Change MASTER_PORT, for example to 29632, 29633, etc.
- Keep WORLD_SIZE=1 for local run.

### Warning: profile.ps1 is not digitally signed

This warning is from PowerShell profile loading and does not block training.

### Warning: test labels hidden or missing

For some RelBench tasks, test labels are intentionally hidden.
This is expected. Validation metrics are still used for model selection.

## 7) Where logs are stored

- Wandb offline runs: [wandb](wandb)
- Smoke outputs:
  - [results/smoke_check](results/smoke_check)
  - [results/smoke_check_ddp](results/smoke_check_ddp)

## 8) Repro tips

- Keep seed fixed (`--seed 42`) when comparing runs.
- Change only one hyperparameter at a time.
- Start with small `--max_steps_per_epoch` before large runs.
