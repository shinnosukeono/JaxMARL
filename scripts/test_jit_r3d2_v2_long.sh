#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_r3d2_v2l

# Long-walltime test of refactored r3d2_rnn_hanabi.py.
# We want enough updates to see steady-state throughput. With NUM_ENVS=64,
# NUM_UPDATES = TOTAL_TIMESTEPS / 80 / 64. Set 1e6 → ~195 updates.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R3D2 v2 LONG (Python-loop) test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# 1e6 timesteps / 80 / 64 = 195 updates. With LOG_EVERY=5 we see u=0,5,10,15,...
python -u baselines/QLearning/r3d2_rnn_hanabi.py \
    +alg=r3d2_rnn_hanabi \
    NUM_SEEDS=1 \
    SEED=0 \
    WANDB_MODE=disabled \
    alg.TOTAL_TIMESTEPS=1e6 \
    alg.NUM_ENVS=64 \
    alg.NUM_EPOCHS=1 \
    alg.TEST_DURING_TRAINING=False \
    LOG_EVERY=5 \
    2>&1

echo ""
echo "=== done $(date) ==="
