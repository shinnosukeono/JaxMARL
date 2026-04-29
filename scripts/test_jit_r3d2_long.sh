#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N test_jit_r3d2_l

# Long-walltime variant of test_jit_r3d2.sh to determine actual JIT compile time.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== R3D2 JIT Compilation Test (LONG WALLTIME) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

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
echo "=== R3D2 JIT test completed ==="
echo "Date: $(date)"
