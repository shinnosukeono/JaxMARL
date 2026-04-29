#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N verify_jax

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

echo "=== Verify text_obs_jax matches text_obs_fast ==="
echo "Date: $(date)"

python -u <<'PYEOF'
import sys
sys.path.insert(0, "baselines/QLearning")

import numpy as np
import jax, jax.numpy as jnp
from transformers import AutoTokenizer

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
from text_obs_fast import HanabiAtoms, state_to_tokens_for_agent
from text_obs_jax import make_tokenize_fn_jax, _state_to_atom_seq, _make_jnp_atoms, _assemble_tokens

tokenizer = AutoTokenizer.from_pretrained(
    "baselines/QLearning/pretrained_text/tinybert_l2_flax"
)

MAX_OBS_TOKENS = 256

for num_agents in [2, 3, 4, 5]:
    print(f"\n=== num_agents={num_agents} ===")
    raw_env = make("hanabi", num_agents=num_agents)
    log_env = LogWrapper(raw_env)
    wrap = CTRolloutManager(log_env, batch_size=1)

    atoms = HanabiAtoms(raw_env, tokenizer)
    A = _make_jnp_atoms(atoms)

    from functools import partial

    @partial(jax.jit, static_argnums=3)  # aidx is static
    def tok_jax(new_state, old_state, action, aidx):
        atom_indices, atom_lens = _state_to_atom_seq(raw_env, A, new_state, old_state, action, aidx)
        return _assemble_tokens(
            atom_indices, atom_lens, A["atoms_arr"], A["atoms_len"],
            MAX_OBS_TOKENS, A["cls"], A["sep"], A["pad"],
        )

    rng = jax.random.PRNGKey(7)
    n_check = 0
    n_mismatch = 0
    sample_mm = None

    for seed_idx in range(2):
        rng, sub = jax.random.split(rng)
        _, env_state = wrap.batch_reset(sub)
        prev_state = env_state
        last_action = jnp.full((1,), raw_env.num_moves - 1, dtype=jnp.int32)
        for t in range(15):
            inner_new_jnp = jax.tree.map(lambda x: x[0], env_state.env_state)
            inner_old_jnp = jax.tree.map(lambda x: x[0], prev_state.env_state)
            inner_new_np = jax.tree.map(lambda x: np.asarray(x), inner_new_jnp)
            inner_old_np = jax.tree.map(lambda x: np.asarray(x), inner_old_jnp)
            for ai in range(raw_env.num_agents):
                a_int = int(last_action[0])
                # Reference: text_obs_fast (verified equivalent to HF)
                ref_ids, ref_mask = state_to_tokens_for_agent(
                    atoms, inner_new_np, inner_old_np, a_int, ai, MAX_OBS_TOKENS,
                )
                # JAX path
                jax_ids, jax_mask = tok_jax(inner_new_jnp, inner_old_jnp, jnp.int32(a_int), ai)
                jax_ids_np = np.asarray(jax_ids)
                jax_mask_np = np.asarray(jax_mask)
                n_check += 1
                if not np.array_equal(ref_ids, jax_ids_np):
                    n_mismatch += 1
                    if sample_mm is None:
                        diffs = np.where(ref_ids != jax_ids_np)[0]
                        sample_mm = (ref_ids, jax_ids_np, diffs, ai, t, a_int)

            rng, sub_a, sub_s = jax.random.split(rng, 3)
            keys_a = jax.random.split(sub_a, raw_env.num_agents)
            actions = {a: wrap.batch_sample(keys_a[i], a) for i, a in enumerate(raw_env.agents)}
            _, env_state, _, _, _ = wrap.batch_step(sub_s, env_state, actions)
            cpi = np.asarray(env_state.env_state.cur_player_idx[0])
            ai_idx = int(np.argmax(cpi))
            cur_action = int(actions[raw_env.agents[ai_idx]][0])
            last_action = jnp.array([cur_action])
            prev_state = env_state

    print(f"  checks: {n_check}, mismatches: {n_mismatch}")
    if n_mismatch and sample_mm:
        ref_ids, jax_ids, diffs, ai, t, a_int = sample_mm
        first = int(diffs[0])
        lo, hi = max(0, first - 8), min(MAX_OBS_TOKENS, first + 12)
        print(f"  first mismatch at pos {first} (ai={ai}, t={t}, action={a_int})")
        print(f"  ref [{lo}:{hi}] = {ref_ids[lo:hi].tolist()}")
        print(f"  jax [{lo}:{hi}] = {jax_ids[lo:hi].tolist()}")
        print(f"  ref decoded: {tokenizer.decode(ref_ids[ref_ids != tokenizer.pad_token_id])!r}")
        print(f"  jax decoded: {tokenizer.decode(jax_ids[jax_ids != tokenizer.pad_token_id])!r}")

print("\n=== DONE ===")
PYEOF

echo "Date: $(date)"
