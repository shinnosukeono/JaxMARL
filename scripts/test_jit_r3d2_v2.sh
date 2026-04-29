#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_r3d2_v2

# Verify refactored r3d2_rnn_hanabi.py: Python loop + JIT'd update_step.
# Expected: each piece compiles in minutes (not 30+ min).

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R3D2 v2 (Python-loop) test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Small config: NUM_ENVS=64, NUM_EPOCHS=1, TEST_DURING_TRAINING=False, very few updates.
# We just want to verify init/update/eval all compile and run.
python -u baselines/QLearning/r3d2_rnn_hanabi.py \
    +alg=r3d2_rnn_hanabi \
    NUM_SEEDS=1 \
    SEED=0 \
    WANDB_MODE=disabled \
    alg.TOTAL_TIMESTEPS=2e5 \
    alg.NUM_ENVS=64 \
    alg.NUM_EPOCHS=1 \
    alg.TEST_DURING_TRAINING=False \
    2>&1

echo ""
echo "=== done $(date) ==="
