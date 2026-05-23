"""
Single-process smoke-check for RelGT++.

Usage:
  python main_node_single_check.py \
      --dataset rel-f1 --task driver-top3 \
      --epochs 1 --batch_size 16 --channels 64 \
      --num_neighbors 64 --num_centroids 256 \
      --num_workers 1 --max_steps_per_epoch 10 \
      --out_dir results/smoke_check
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens

# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",            type=str,   default="rel-f1")
parser.add_argument("--task",               type=str,   default="driver-top3")
parser.add_argument("--precompute",         type=int,   default=1)
parser.add_argument("--lr",                 type=float, default=1e-4)
parser.add_argument("--epochs",             type=int,   default=10)
parser.add_argument("--batch_size",         type=int,   default=16)
parser.add_argument("--channels",           type=int,   default=64)
parser.add_argument("--num_layers",         type=int,   default=1)
parser.add_argument("--num_heads",          type=int,   default=4)
parser.add_argument("--gt_conv_type",       type=str,   default="full")
parser.add_argument("--ablate",             type=str,   default="none")
parser.add_argument("--gnn_pe_dim",         type=int,   default=0)
parser.add_argument("--num_neighbors",      type=int,   default=64)
parser.add_argument("--num_centroids",      type=int,   default=256)
parser.add_argument("--ff_dropout",         type=float, default=0.1)
parser.add_argument("--attn_dropout",       type=float, default=0.1)
parser.add_argument("--weight_decay",       type=float, default=1e-5)
parser.add_argument("--max_steps_per_epoch",type=int,   default=3000)
parser.add_argument("--num_workers",        type=int,   default=1)
parser.add_argument("--seed",               type=int,   default=42)
parser.add_argument("--out_dir",            type=str,   default="results/smoke_check")
parser.add_argument("--aux_loss_weight",    type=float, default=1.0,
                    help="[NOVEL] Weight for auxiliary losses")
parser.add_argument(
    "--cache_dir", type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()
args.precompute = bool(args.precompute)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
dataset: Dataset    = get_dataset(args.dataset, download=True)
task:    EntityTask = get_task(args.dataset, args.task, download=True)

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path) as f:
        col_to_stype_dict = json.load(f)
    for table, c2s in col_to_stype_dict.items():
        for col, s in c2s.items():
            c2s[col] = stype(s)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

data = {
    split: RelGTTokens(
        data=data, task=task, K=args.num_neighbors, split=split,
        undirected=True, precompute=args.precompute,
        precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
        num_workers=args.num_workers, train_stage="finetune",
    )
    for split in ["train", "val", "test"]
}

# ─────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────
loader_train = DataLoader(
    data["train"], batch_size=args.batch_size, shuffle=True,
    collate_fn=data["train"].collate, num_workers=args.num_workers,
)
loader_val  = DataLoader(
    data["val"],   batch_size=args.batch_size, shuffle=False,
    collate_fn=data["val"].collate, num_workers=args.num_workers,
)
loader_test = DataLoader(
    data["test"],  batch_size=args.batch_size, shuffle=False,
    collate_fn=data["test"].collate, num_workers=args.num_workers,
)
loader_dict = {"train": loader_train, "val": loader_val, "test": loader_test}

# ─────────────────────────────────────────────────────────────
# Task settings
# ─────────────────────────────────────────────────────────────
clamp_min = clamp_max = None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn      = BCEWithLogitsLoss()
    tune_metric  = "roc_auc"
    higher       = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn      = L1Loss()
    tune_metric  = "mae"
    higher       = False
    train_table  = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn      = BCEWithLogitsLoss()
    tune_metric  = "multilabel_auprc_macro"
    higher       = True
else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
model = RelGT(
    num_nodes=data["train"].data.num_nodes,
    max_neighbor_hop=data["train"].max_neighbor_hop,
    node_type_map=data["train"].node_type_to_index,
    col_names_dict={
        nt: data["train"].data[nt].tf.col_names_dict
        for nt in data["train"].data.node_types
    },
    col_stats_dict=col_stats_dict,
    local_num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    global_dim=args.channels // 2,
    heads=args.num_heads,
    ff_dropout=args.ff_dropout,
    attn_dropout=args.attn_dropout,
    conv_type=args.gt_conv_type,
    ablate=args.ablate,
    gnn_pe_dim=args.gnn_pe_dim,
    num_centroids=args.num_centroids,
    sample_node_len=args.num_neighbors,
    args=args,
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,}")

optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────
def train(epoch) -> float:
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)

    for step, batch in enumerate(
        tqdm(loader_dict["train"], total=total_steps, desc=f"Epoch {epoch}"), start=1
    ):
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices   = batch["node_indices"].to(device)
        neighbor_hops  = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index     = batch["edge_index"].to(device)
        batch_vec      = batch["batch"].to(device)
        labels         = batch["labels"].to(device)

        grouped_tf_dict = {
            "grouped_tfs":    batch["grouped_tfs"],
            "grouped_indices":batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx":   batch["flat_nbr_idx"],
        }

        optimizer.zero_grad()
        pred = model(
            neighbor_types, node_indices, neighbor_hops, neighbor_times,
            grouped_tf_dict, edge_index=edge_index, batch=batch_vec,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        # [NOVEL] Auxiliary loss
        task_loss  = loss_fn(pred.float(), labels)
        aux_loss   = model.get_aux_loss() * args.aux_loss_weight
        total_loss = task_loss + aux_loss

        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_accum  += total_loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        if step >= args.max_steps_per_epoch:
            break

    return loss_accum / count_accum if count_accum > 0 else float("inf")


@torch.no_grad()
def evaluate(loader, desc) -> np.ndarray:
    model.eval()
    pred_list, idx_list = [], []
    for batch in tqdm(loader, desc=desc):
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices   = batch["node_indices"].to(device)
        neighbor_hops  = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index     = batch["edge_index"].to(device)
        batch_vec      = batch["batch"].to(device)
        grouped_tf_dict = {
            "grouped_tfs":    batch["grouped_tfs"],
            "grouped_indices":batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx":   batch["flat_nbr_idx"],
        }
        pred = model(
            neighbor_types, node_indices, neighbor_hops, neighbor_times,
            grouped_tf_dict, edge_index=edge_index, batch=batch_vec,
        )
        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION,
                               TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())

    all_preds = np.full((len(loader.dataset),), -100.0)
    for idxs, preds in zip(idx_list, pred_list):
        for i, p in zip(idxs, preds):
            all_preds[i] = p
    return all_preds


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
best_val  = -math.inf if higher else math.inf
best_state = None

for epoch in range(1, args.epochs + 1):
    train_loss   = train(epoch)
    val_preds    = evaluate(loader_val, "Val")
    val_metrics  = task.evaluate(val_preds, task.get_table("val"))

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val={val_metrics}")

    # [NOVEL] codebook stats
    for i, conv in enumerate(model.convs):
        if hasattr(conv, "vq"):
            util = conv.vq.utilization_rate
            temp = conv.vq.temperature.item()
            print(f"  Layer {i} codebook util={util:.1%}  τ={temp:.4f}")

    improved = (
        (higher  and val_metrics[tune_metric] >= best_val) or
        (not higher and val_metrics[tune_metric] <= best_val)
    )
    if improved:
        best_val   = val_metrics[tune_metric]
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, os.path.join(output_path, "best_model.pt"))

if best_state is not None:
    model.load_state_dict(best_state)

# Final evaluation
val_preds    = evaluate(loader_val,  "Final Val")
test_preds   = evaluate(loader_test, "Final Test")
val_metrics  = task.evaluate(val_preds,  task.get_table("val"))
test_metrics = task.evaluate(test_preds)

print(f"\nBest Val  : {val_metrics}")
print(f"Best Test : {test_metrics}")

results = {"val_metrics": val_metrics, "test_metrics": test_metrics}
with open(os.path.join(output_path, f"{args.seed}.json"), "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_path}/{args.seed}.json")
