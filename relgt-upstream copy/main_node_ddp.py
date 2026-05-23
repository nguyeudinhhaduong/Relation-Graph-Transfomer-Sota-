import argparse
import copy
import json
import math
import os
import platform
from pathlib import Path
from typing import Dict
import wandb

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import pynvml
except Exception:
    pynvml = None

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
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

torch.autograd.set_detect_anomaly(True)

############################
# 1. Parse arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",            type=str,   default="rel-f1")
parser.add_argument("--task",               type=str,   default="driver-top3")
parser.add_argument("--precompute",         action="store_true", default=True)
parser.add_argument("--lr",                 type=float, default=0.0001)
parser.add_argument("--warmup_steps",       type=int,   default=1000)
parser.add_argument("--epochs",             type=int,   default=10)
parser.add_argument("--batch_size",         type=int,   default=512)
parser.add_argument("--channels",           type=int,   default=512)
parser.add_argument("--aggr",               type=str,   default="sum")
parser.add_argument("--num_layers",         type=int,   default=1)
parser.add_argument("--num_heads",          type=int,   default=4)
parser.add_argument("--gt_conv_type",       type=str,   default="full")
parser.add_argument("--ablate",             type=str,   default="none")
parser.add_argument("--gnn_pe_dim",         type=int,   default=0)
parser.add_argument("--num_neighbors",      type=int,   default=300)
parser.add_argument("--num_centroids",      type=int,   default=4096)
parser.add_argument("--ff_dropout",         type=float, default=0.1)
parser.add_argument("--attn_dropout",       type=float, default=0.1)
parser.add_argument("--weight_decay",       type=float, default=0.00001)
parser.add_argument("--temporal_strategy",  type=str,   default="uniform")
parser.add_argument("--pos_enc",            type=str,   default="none")
parser.add_argument("--max_degree",         type=int,   default=10000)
parser.add_argument("--pos_enc_dim",        type=int,   default=128)
parser.add_argument("--max_steps_per_epoch",type=int,   default=3000)
parser.add_argument("--num_workers",        type=int,   default=2)
parser.add_argument("--seed",               type=int,   default=42)
parser.add_argument("--out_dir",            type=str,   default="results/debug")
parser.add_argument("--run_name",           type=str,   default="debug")
parser.add_argument("--model_parameters",   type=int,   default=0)
parser.add_argument("--aux_loss_weight",    type=float, default=1.0,
                    help="[NOVEL] Scale factor for auxiliary losses (codebook entropy + agreement)")
parser.add_argument(
    "--cache_dir", type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--train_stage", type=str, default="finetune",
                    choices=["finetune"])

args = parser.parse_args()

############################
# 2. DDP initialisation
############################
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK",       "0")
os.environ.setdefault("WORLD_SIZE", "1")
if int(os.environ.get("WORLD_SIZE", "1")) == 1:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")
else:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank       = int(os.environ.get("RANK",       "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

ddp_backend = (
    "nccl"
    if use_cuda and dist.is_nccl_available() and platform.system().lower() != "windows"
    else "gloo"
)
dist.init_process_group(backend=ddp_backend, init_method="env://")

if rank == 0:
    args.run_name = f"{args.dataset}-{args.task}-{args.run_name}"


def init_gpu_utilization(device_index):
    if pynvml is None or not torch.cuda.is_available():
        return None
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(device_index)


def get_gpu_stats(handle, device):
    if handle is None or device.type != "cuda":
        return 0.0, 0.0, 0.0
    util          = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    mem_reserved  = torch.cuda.memory_reserved(device)  / 1024 ** 2
    return util.gpu, mem_allocated, mem_reserved


if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)
gpu_handle = init_gpu_utilization(local_rank if device.type == "cuda" else 0)

############################
# 3. Dataset & task
############################
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
        text_embedder=GloveTextEmbedding(
            device=(f"cuda:{local_rank}" if device.type == "cuda" else "cpu")
        ),
        batch_size=256,
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

data = {
    split: RelGTTokens(
        data=data,
        task=task,
        K=args.num_neighbors,
        split=split,
        undirected=True,
        precompute=args.precompute,
        precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
        num_workers=args.num_workers,
        train_stage=args.train_stage,
    )
    for split in ["train", "val", "test"]
}

############################
# 4. DataLoaders
############################
train_sampler = DistributedSampler(data["train"], shuffle=True, seed=args.seed)
loader_train  = DataLoader(
    data["train"],
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=data["train"].collate,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=(device.type == "cuda"),
)

val_sampler  = DistributedSampler(data["val"],  shuffle=False, seed=args.seed, drop_last=False)
loader_val   = DataLoader(
    data["val"],
    batch_size=args.batch_size,
    sampler=val_sampler,
    collate_fn=data["val"].collate,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=(device.type == "cuda"),
)

test_sampler = DistributedSampler(data["test"], shuffle=False, seed=args.seed, drop_last=False)
loader_test  = DataLoader(
    data["test"],
    batch_size=args.batch_size,
    sampler=test_sampler,
    collate_fn=data["test"].collate,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=(device.type == "cuda"),
)

loader_dict: Dict[str, DataLoader] = {
    "train": loader_train, "val": loader_val, "test": loader_test
}

############################
# 5. Task settings
############################
clamp_min = clamp_max = None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels      = 1
    loss_fn           = BCEWithLogitsLoss()
    tune_metric       = "roc_auc"
    higher_is_better  = True
elif task.task_type == TaskType.REGRESSION:
    out_channels      = 1
    loss_fn           = L1Loss()
    tune_metric       = "mae"
    higher_is_better  = False
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels      = task.num_labels
    loss_fn           = BCEWithLogitsLoss()
    tune_metric       = "multilabel_auprc_macro"
    higher_is_better  = True
else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

############################
# 6. Model + DDP
############################
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

# int16 buffer guard for DDP
for name, param in model.named_parameters():
    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)
for name, buf in model.named_buffers():
    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)

if device.type == "cuda":
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
else:
    model = DDP(model, find_unused_parameters=True)

if rank == 0:
    print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if rank == 0:
    print(f"Total model parameters: {total_params:,}")
args.model_parameters = total_params

if rank == 0:
    wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)

optimizer   = torch.optim.Adam(
    model.parameters(), lr=args.lr * world_size, weight_decay=args.weight_decay
)
global_step = 0

############################
# 7. Training loop
############################
def train_supervised(epoch) -> float:
    global global_step
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    train_sampler.set_epoch(epoch)

    for step, batch in enumerate(
        tqdm(loader_dict["train"], total=total_steps, desc="Train"), start=1
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

        # ── [NOVEL] Auxiliary loss ────────────────────────────────────
        task_loss = loss_fn(pred.float(), labels)
        aux_loss  = model.module.get_aux_loss() * args.aux_loss_weight
        total_loss = task_loss + aux_loss
        # ─────────────────────────────────────────────────────────────

        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = total_loss.detach().item()
        gpu_util, mem_alloc, mem_res = get_gpu_stats(gpu_handle, device)

        if rank == 0:
            wandb.log({
                "train/task_loss":        task_loss.item(),
                "train/aux_loss":         aux_loss.item(),
                "train/total_loss":       loss_value,
                "train/lr":               optimizer.param_groups[0]["lr"],
                "gpu/util_pct":           gpu_util,
                "gpu/mem_allocated_MB":   mem_alloc,
                "gpu/mem_reserved_MB":    mem_res,
                "global_step":            global_step,
            })

        loss_accum  += loss_value * pred.size(0)
        count_accum += pred.size(0)
        global_step += 1

        if step >= args.max_steps_per_epoch:
            break

    return loss_accum / count_accum if count_accum > 0 else float("inf")


############################
# 8. Evaluation
############################
@torch.no_grad()
def test(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    eval_model.eval()
    pred_list, idx_list = [], []

    for batch in tqdm(loader, desc=desc, disable=(rank != 0)):
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
        pred = eval_model(
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

    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs  = np.concatenate(idx_list,  axis=0) if idx_list  else np.array([])

    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)

    if rank == 0:
        all_preds = np.full((len(loader.dataset),), -100.0)
        for g_idx, g_pred in gathered:
            for i, p in zip(g_idx, g_pred):
                all_preds[i] = p
        return all_preds
    return None


############################
# 9. Fine-tuning stage
############################
best_val_metric = -math.inf if higher_is_better else math.inf
state_dict      = None

for epoch in range(1, args.epochs + 1):
    train_loss = train_supervised(epoch)
    dist.barrier()

    eval_model = model.module
    val_pred   = test(loader_dict["val"], eval_model=eval_model, epoch=epoch, desc="Val")

    if rank == 0:
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val={val_metrics}")
        wandb.log({"epoch": epoch, "epoch/train_loss": train_loss,
                   **{f"val/{k}": v for k, v in val_metrics.items()}})

        # [NOVEL] log codebook utilization
        for i, conv in enumerate(eval_model.convs):
            if hasattr(conv, "vq"):
                util = conv.vq.utilization_rate
                temp = conv.vq.temperature.item()
                wandb.log({f"layer{i}/codebook_util": util,
                           f"layer{i}/temperature":   temp})

        improved = (
            (higher_is_better  and val_metrics[tune_metric] >= best_val_metric) or
            (not higher_is_better and val_metrics[tune_metric] <= best_val_metric)
        )
        if improved:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.module.state_dict())
            torch.save(state_dict, os.path.join(output_path, "finetuned.pt"))

    dist.barrier()

# Broadcast best weights to all ranks
if rank == 0 and state_dict is not None:
    model.module.load_state_dict(state_dict)
for p in model.parameters():
    dist.broadcast(p.data, src=0)
for b in model.buffers():
    dist.broadcast(b.data, src=0)
dist.barrier()

# Final evaluation
final_val_preds  = test(loader_dict["val"],  eval_model=model.module, epoch=0, desc="Final Val")
final_test_preds = test(loader_dict["test"], eval_model=model.module, epoch=0, desc="Final Test")

if rank == 0:
    val_metrics  = task.evaluate(final_val_preds,  task.get_table("val"))
    test_metrics = task.evaluate(final_test_preds)
    print(f"Best Val  : {val_metrics}")
    print(f"Best Test : {test_metrics}")

    file_path = os.path.join(output_path, f"{args.seed}.json")
    with open(file_path, "w") as f:
        json.dump({"val_metrics": val_metrics, "test_metrics": test_metrics}, f, indent=4)

    wandb.log({"best_val": val_metrics, "best_test": test_metrics})

############################
# 10. Cleanup
############################
dist.destroy_process_group()
