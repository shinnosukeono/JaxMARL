#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_r2d2t_v2

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R2D2-text v2 (Python-loop) test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

python -u baselines/QLearning/r2d2_text_rnn_hanabi.py \
    +alg=r2d2_text_rnn_hanabi \
    NUM_SEEDS=1 \
    SEED=0 \
    WANDB_MODE=disabled \
    alg.TOTAL_TIMESTEPS=2e5 \
    alg.NUM_ENVS=64 \
    alg.NUM_EPOCHS=1 \
    alg.TEST_DURING_TRAINING=False \
    LOG_EVERY=5 \
    2>&1

echo ""
echo "=== done $(date) ==="
