#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N verify_publ_split

# Verify the publ/priv split refactor in r2d2_publ_rnn_hanabi.py against
# upstream OBLAgentR2D2. See scripts/verify_publ_split.py for what is checked.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

echo "=== verify_publ_split ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

python -u scripts/verify_publ_split.py 2>&1
status=$?

echo ""
echo "=== exit $status @ $(date) ==="
exit "$status"
