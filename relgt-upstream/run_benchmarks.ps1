$ErrorActionPreference = "Stop"

$env:WANDB_MODE="offline"
$env:MASTER_ADDR="127.0.0.1"
$env:LOCAL_RANK="0"
$env:RANK="0"
$env:WORLD_SIZE="1"

Write-Host "Running rel-f1 benchmark..."
$env:MASTER_PORT="29631"
conda run --no-capture-output -n relgt_tf python main_node_ddp.py --dataset rel-f1 --task driver-top3 --epochs 10 --batch_size 16 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 0 --max_steps_per_epoch 50 --precompute --out_dir results/benchmark --run_name benchmark_rel_f1

Write-Host "Running rel-event benchmark..."
$env:MASTER_PORT="29632"
conda run --no-capture-output -n relgt_tf python main_node_ddp.py --dataset rel-event --task user-attendance --epochs 10 --batch_size 8 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 0 --max_steps_per_epoch 50 --out_dir results/benchmark --run_name benchmark_rel_event

Write-Host "Running rel-trial benchmark..."
$env:MASTER_PORT="29633"
conda run --no-capture-output -n relgt_tf python main_node_ddp.py --dataset rel-trial --task study-outcome --epochs 10 --batch_size 8 --channels 64 --num_neighbors 64 --num_centroids 256 --num_workers 0 --max_steps_per_epoch 50 --out_dir results/benchmark --run_name benchmark_rel_trial

Write-Host "All benchmarks completed!"
