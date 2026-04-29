#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_jit_thru

# Test steady-state throughput for R3D2 after the BERT JIT fix.
# Runs r3d2 at full NUM_ENVS=256 for enough steps to measure stable throughput.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9

source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R3D2 Steady-State Throughput Test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ~2e6 timesteps at full NUM_ENVS=256 -> ~100 update steps.
# After JIT compilation, this should show stable per-step throughput.
python baselines/QLearning/r3d2_rnn_hanabi.py \
    +alg=r3d2_rnn_hanabi \
    NUM_SEEDS=1 \
    SEED=0 \
    WANDB_MODE=disabled \
    alg.TOTAL_TIMESTEPS=2e6 \
    alg.NUM_ENVS=256 \
    alg.TEST_DURING_TRAINING=False \
    2>&1

echo ""
echo "=== R3D2 throughput test completed ==="
echo "Date: $(date)"
