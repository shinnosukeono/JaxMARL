#!/bin/bash
# Master launcher: generates and submits PBS jobs for the full R3D2 paper reproduction.
#
# Usage:
#   bash scripts/submit_repro.sh [--dry-run]       # submit all experiments
#   bash scripts/submit_repro.sh --only r3d2_2p    # submit specific experiment
#   bash scripts/submit_repro.sh --list             # list experiment names
#
# Paper experiments (Table 2, Figures 2-4):
#   R3D2-S  @ 2p,3p,4p,5p  (3 seeds each)
#   R3D2-M  @ 2-5p          (3 seeds)
#   R2D2    @ 2p,3p,4p,5p  (3 seeds each)
#   R2D2-text @ 2p          (3 seeds)
#   IPPO    @ 2p            (3 seeds)
#   OBL     @ 2p            (3 seeds, depends on R2D2 + belief checkpoints)

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
JOBS_DIR="${SCRIPTS_DIR}/jobs"
mkdir -p "${JOBS_DIR}"

# Defaults
QUEUE="regular-g"
WALLTIME="24:00:00"
GROUP="gj23"
WANDB_PROJECT="r3d2-jaxmarl-repro"
# WANDB_API_KEY is read from the launching env (do NOT hard-code it here).
# Set it once in your shell before invoking this script:
#   export WANDB_API_KEY=...
# or `wandb login` once and let WANDB_API_KEY come from ~/.netrc.
NUM_SEEDS=5
DRY_RUN=0
ONLY=""

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARN: WANDB_API_KEY is unset. Wandb logging will be skipped at job runtime." >&2
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --list) echo "Experiments:"; echo "  r3d2_2p r3d2_3p r3d2_4p r3d2_5p"; echo "  r3d2_mt"; echo "  r2d2_2p r2d2_3p r2d2_4p r2d2_5p"; echo "  r2d2_text_2p"; echo "  ippo_2p"; echo "  obl_2p (needs R2D2 checkpoint)"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

gen_job() {
    local name="$1"
    local script_cmd="$2"
    local walltime="${3:-$WALLTIME}"
    local queue="${4:-$QUEUE}"
    local jobfile="${JOBS_DIR}/${name}.sh"

    cat > "${jobfile}" <<JOBEOF
#!/bin/bash
#PBS -q ${queue}
#PBS -l select=1
#PBS -l walltime=${walltime}
#PBS -W group_list=${GROUP}
#PBS -j oe
#PBS -N ${name}

cd ${PROJECT_DIR}
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Inherit WANDB_API_KEY from the submitter's env. Empty here is fine; the
# training script will fall back to WANDB_MODE=offline if no key is found.
export WANDB_API_KEY="\${WANDB_API_KEY:-${WANDB_API_KEY:-}}"
# Persistent JAX compile cache — skip ~7-min XLA recompiles on subsequent runs.
export JAX_COMPILATION_CACHE_DIR=/work/gj23/k36132/ideas/r3d2_tom/.jax_cache
mkdir -p \${JAX_COMPILATION_CACHE_DIR}

echo "=== ${name} === \$(date)"
echo "Host: \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

${script_cmd}

echo "=== ${name} done === \$(date)"
JOBEOF
    chmod +x "${jobfile}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] ${jobfile}"
    else
        local jobid
        jobid=$(qsub "${jobfile}" 2>&1)
        echo "[submitted] ${name}: ${jobid}"
    fi
}

should_run() {
    [[ -z "$ONLY" ]] || [[ "$ONLY" == "$1" ]]
}

# ============================================================================
# R3D2-S (single-setting, per player count)
# ============================================================================
for np in 2 3 4 5; do
    name="r3d2_${np}p"
    should_run "${name}" || continue
    gen_job "${name}" "python baselines/QLearning/r3d2_rnn_hanabi.py \\
    +alg=r3d2_rnn_hanabi \\
    NUM_SEEDS=${NUM_SEEDS} SEED=0 \\
    WANDB_MODE=online WANDB_LOG_ALL_SEEDS=True \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name} \\
    alg.ENV_KWARGS.num_agents=${np}"
done

# ============================================================================
# R3D2-M (multitask: one network, all player counts)
# ============================================================================
name="r3d2_mt"
if should_run "${name}"; then
    # R3D2 multitask can't vmap across seeds (Python loop), so run 5 seeds
    # as separate jobs.
    for s in 0 1 2 3 4; do
        gen_job "${name}_s${s}" "python baselines/QLearning/r3d2_multitask_rnn_hanabi.py \\
    +alg=r3d2_multitask_rnn_hanabi \\
    NUM_SEEDS=1 SEED=${s} \\
    WANDB_MODE=online \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name}"
    done
fi

# ============================================================================
# R2D2 (public/private LSTM, per player count)
# ============================================================================
for np in 2 3 4 5; do
    name="r2d2_${np}p"
    should_run "${name}" || continue
    gen_job "${name}" "python baselines/QLearning/r2d2_publ_rnn_hanabi.py \\
    +alg=r2d2_publ_rnn_hanabi \\
    NUM_SEEDS=${NUM_SEEDS} SEED=0 \\
    WANDB_MODE=online WANDB_LOG_ALL_SEEDS=True \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name} \\
    alg.ENV_KWARGS.num_agents=${np}"
done

# ============================================================================
# R2D2-OP (Other-Play: per-episode color permutation, 2-player only)
# ============================================================================
name="r2d2_op_2p"
if should_run "${name}"; then
    gen_job "${name}" "python baselines/QLearning/r2d2_publ_rnn_hanabi.py \\
    +alg=r2d2_op_rnn_hanabi \\
    NUM_SEEDS=${NUM_SEEDS} SEED=0 \\
    WANDB_MODE=online WANDB_LOG_ALL_SEEDS=True \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name}"
fi

# ============================================================================
# R2D2-text (frozen BERT obs encoder, 2-player only)
# ============================================================================
name="r2d2_text_2p"
if should_run "${name}"; then
    gen_job "${name}" "python baselines/QLearning/r2d2_text_rnn_hanabi.py \\
    +alg=r2d2_text_rnn_hanabi \\
    NUM_SEEDS=${NUM_SEEDS} SEED=0 \\
    WANDB_MODE=online WANDB_LOG_ALL_SEEDS=True \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name}"
fi

# ============================================================================
# IPPO (vanilla, no pi-KL — LLM_PRIOR_PATH=null falls back to IPPO)
# IPPO script doesn't support NUM_SEEDS vmap; run 3 separate seed jobs.
# ============================================================================
name="ippo_2p"
if should_run "${name}"; then
    for s in 0 1 2 3 4; do
        gen_job "${name}_s${s}" "python baselines/IPPO/ippo_pikl_rnn_hanabi.py \\
    SEED=${s} \\
    WANDB_MODE=online \\
    PROJECT=${WANDB_PROJECT}"
    done
fi

# ============================================================================
# OBL (needs R2D2 + belief checkpoints — skip if missing)
# ============================================================================
name="obl_2p"
if should_run "${name}"; then
    BP_CKPT="models/r2d2_2p/r2d2_publ_hanabi_2p_seed0_run0.safetensors"
    BELIEF_CKPT="models/obl_belief/belief_seed0.safetensors"
    if [[ -f "$BP_CKPT" && -f "$BELIEF_CKPT" ]]; then
        gen_job "${name}" "python baselines/QLearning/obl_rnn_hanabi.py \\
    +alg=obl_rnn_hanabi \\
    NUM_SEEDS=${NUM_SEEDS} SEED=0 \\
    WANDB_MODE=online \\
    PROJECT=${WANDB_PROJECT} \\
    SAVE_PATH=models/${name} \\
    alg.BP_CHECKPOINT=${BP_CKPT} \\
    alg.BELIEF_CHECKPOINT=${BELIEF_CKPT}"
    else
        echo "[skip] ${name}: waiting for R2D2 (${BP_CKPT}) and belief (${BELIEF_CKPT}) checkpoints"
        echo "       Run r2d2_2p first, then obl_train_belief, then re-run this script with --only obl_2p"
    fi
fi

echo ""
echo "Done. Monitor with: qstat -u \$USER  (if supported) or qstat <jobid>"
