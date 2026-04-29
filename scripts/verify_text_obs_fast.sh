#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N verify_fast

# Verify text_obs_fast.state_to_tokens_for_agent produces the same
# token IDs as text_obs.render_obs_for_agent + HF tokenizer.

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

echo "=== Verify text_obs_fast token equivalence ==="
echo "Date: $(date)"

python -u <<'PYEOF'
import sys
sys.path.insert(0, "baselines/QLearning")

import numpy as np
import jax, jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBertModel

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
from text_obs import render_obs_for_agent
from text_obs_fast import HanabiAtoms, state_to_tokens_for_agent

print("Loading TinyBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "baselines/QLearning/pretrained_text/tinybert_l2_flax"
)

MAX_OBS_TOKENS = 256

# Run for several player counts
for num_agents in [2, 3, 4, 5]:
    print(f"\n=== num_agents={num_agents} ===")
    raw_env = make("hanabi", num_agents=num_agents)
    log_env = LogWrapper(raw_env)
    wrap = CTRolloutManager(log_env, batch_size=4)

    atoms = HanabiAtoms(raw_env, tokenizer)
    print(f"  built {len(atoms.strings)} atoms (max_atom_len={atoms.atoms_arr.shape[1]})")

    rng = jax.random.PRNGKey(42)
    n_check = 0
    n_mismatch = 0
    sample_mismatch = None

    for seed_idx in range(3):
        rng, sub = jax.random.split(rng)
        _, env_state = wrap.batch_reset(sub)
        prev_state = env_state
        last_action = jnp.full((4,), raw_env.num_moves - 1, dtype=jnp.int32)

        for t in range(15):
            # for each batch index and each agent, compare tokenizations
            for bi in range(4):
                inner_new = jax.tree.map(lambda x, bi=bi: np.asarray(x[bi]), env_state.env_state)
                inner_old = jax.tree.map(lambda x, bi=bi: np.asarray(x[bi]), prev_state.env_state)
                for ai in range(raw_env.num_agents):
                    a_int = int(last_action[bi])
                    txt = render_obs_for_agent(
                        raw_env, inner_new, inner_old, a_int, ai, include_belief=False,
                    )
                    hf_ids = tokenizer(
                        txt, padding="max_length", truncation=True,
                        max_length=MAX_OBS_TOKENS, return_tensors="np",
                    )["input_ids"][0]
                    fast_ids, _ = state_to_tokens_for_agent(
                        atoms, inner_new, inner_old, a_int, ai, MAX_OBS_TOKENS,
                    )
                    n_check += 1
                    if not np.array_equal(hf_ids, fast_ids):
                        n_mismatch += 1
                        if sample_mismatch is None:
                            diffs = np.where(hf_ids != fast_ids)[0]
                            sample_mismatch = (txt, hf_ids, fast_ids, diffs)

            # advance with random actions
            rng, sub_a, sub_s = jax.random.split(rng, 3)
            keys_a = jax.random.split(sub_a, raw_env.num_agents)
            actions = {
                a: wrap.batch_sample(keys_a[i], a) for i, a in enumerate(raw_env.agents)
            }
            _, env_state, _, _, _ = wrap.batch_step(sub_s, env_state, actions)
            cur_player_action_per_b = np.zeros(4, dtype=np.int32)
            for bi in range(4):
                cpi = np.asarray(env_state.env_state.cur_player_idx[bi])
                ai_idx = int(np.argmax(cpi))
                cur_player_action_per_b[bi] = int(actions[raw_env.agents[ai_idx]][bi])
            last_action = jnp.asarray(cur_player_action_per_b)
            prev_state = env_state

    print(f"  checks: {n_check}, mismatches: {n_mismatch}")
    if n_mismatch > 0 and sample_mismatch:
        txt, hf_ids, fast_ids, diffs = sample_mismatch
        print(f"  first mismatch at positions: {diffs[:10].tolist()}")
        first = int(diffs[0])
        lo, hi = max(0, first - 8), min(MAX_OBS_TOKENS, first + 15)
        print(f"  hf  [{lo}:{hi}] = {hf_ids[lo:hi].tolist()}")
        print(f"  fast[{lo}:{hi}] = {fast_ids[lo:hi].tolist()}")
        print(f"  hf  decoded around mismatch: {tokenizer.decode(hf_ids[lo:hi])!r}")
        print(f"  fast decoded around mismatch: {tokenizer.decode(fast_ids[lo:hi])!r}")
        # Decode whole sequences (not just first 80) and show diff position.
        hf_full = tokenizer.decode(hf_ids[hf_ids != tokenizer.pad_token_id])
        fast_full = tokenizer.decode(fast_ids[fast_ids != tokenizer.pad_token_id])
        print(f"  hf  full decoded: {hf_full!r}")
        print(f"  fast full decoded: {fast_full!r}")
        # Print full rendered text to see what was being tokenized
        print(f"  rendered text (full {len(txt)} chars):\n{txt}")

print("\n=== DONE ===")
PYEOF

echo "Date: $(date)"
