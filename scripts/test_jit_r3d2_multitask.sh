#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_jit_r3d2_mt

# Test JIT compilation time for R3D2 multitask (2,3,4,5-player Hanabi).
# The multitask variant uses a Python-driven round-robin loop, so it compiles
# one update_fn per player count.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9

source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R3D2 Multitask JIT Compilation Test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Short run: 5e5 timesteps, reduced envs per setting.
python baselines/QLearning/r3d2_multitask_rnn_hanabi.py \
    +alg=r3d2_multitask_rnn_hanabi \
    NUM_SEEDS=1 \
    SEED=0 \
    WANDB_MODE=disabled \
    alg.TOTAL_TIMESTEPS=5e5 \
    alg.NUM_ENVS_PER_SETTING=16 \
    2>&1

echo ""
echo "=== R3D2 Multitask JIT test completed ==="
echo "Date: $(date)"
