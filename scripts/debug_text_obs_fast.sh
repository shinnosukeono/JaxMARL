#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N debug_fast

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

echo "=== Debug text_obs_fast token emission ==="
echo "Date: $(date)"

python -u <<'PYEOF'
import sys
sys.path.insert(0, "baselines/QLearning")

import numpy as np
import jax, jax.numpy as jnp
from transformers import AutoTokenizer

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
from text_obs import render_obs_for_agent
from text_obs_fast import HanabiAtoms

tokenizer = AutoTokenizer.from_pretrained(
    "baselines/QLearning/pretrained_text/tinybert_l2_flax"
)
raw_env = make("hanabi", num_agents=2)
log_env = LogWrapper(raw_env)
wrap = CTRolloutManager(log_env, batch_size=1)
atoms = HanabiAtoms(raw_env, tokenizer)

rng = jax.random.PRNGKey(0)
_, env_state = wrap.batch_reset(rng)
prev_state = env_state

inner_new = jax.tree.map(lambda x: np.asarray(x[0]), env_state.env_state)
inner_old = jax.tree.map(lambda x: np.asarray(x[0]), prev_state.env_state)
ai = 0
last_action = raw_env.num_moves - 1

txt = render_obs_for_agent(raw_env, inner_new, inner_old, last_action, ai, include_belief=False)
print("Rendered text:")
print(repr(txt))
print()

# Tokenize via HF (with special tokens)
hf_enc = tokenizer(txt, add_special_tokens=True, padding=False, truncation=False, return_tensors="np")
hf_ids = hf_enc["input_ids"][0]
print(f"HF token count: {len(hf_ids)}")

# Tokenize each "word" separately to show what the per-word tokenization looks like.
words = txt.split()
print(f"\nWord-level tokenization (concatenated):")
all_word_tokens = [tokenizer.cls_token_id]
for w in words:
    wt = tokenizer(w, add_special_tokens=False, return_tensors="np")["input_ids"][0]
    all_word_tokens.extend(wt.tolist())
all_word_tokens.append(tokenizer.sep_token_id)
print(f"Word-concat token count: {len(all_word_tokens)}")
print(f"Match HF: {np.array_equal(np.array(all_word_tokens), hf_ids)}")
if not np.array_equal(np.array(all_word_tokens), hf_ids):
    diff_idx = np.where(np.array(all_word_tokens[:len(hf_ids)]) != hf_ids[:len(all_word_tokens)])[0]
    print(f"First mismatch positions: {diff_idx[:5]}")

# Show first 20 atoms emitted by fast path with their text + tokens.
# We'll trace through state_to_tokens_for_agent manually to show emissions.
print("\nAtom emission trace (first 30 atoms):")
def trace_emit(atoms, label, atom_idx):
    L = int(atoms.atoms_len[atom_idx])
    toks = atoms.atoms_arr[atom_idx, :L].tolist()
    decoded = tokenizer.decode(toks)
    s = atoms.strings[atom_idx]
    print(f"  emit '{label}': atom_idx={atom_idx} string={s!r} tokens={toks} decoded={decoded!r}")

env = atoms.env
new_state, old_state, action, aidx = inner_new, inner_old, last_action, ai
trace_emit(atoms, "Turn:", atoms.idx_turn)
trace_emit(atoms, "<turn-num>", int(atoms.idx_number[int(new_state.turn)]))
trace_emit(atoms, "Score:", atoms.idx_score)
trace_emit(atoms, "<score-num>", int(atoms.idx_number[int(new_state.score)]))
trace_emit(atoms, "Information", atoms.idx_information)
trace_emit(atoms, "available:", atoms.idx_available)
info = int(np.asarray(new_state.info_tokens).sum())
trace_emit(atoms, "<info-num>", int(atoms.idx_number[info]))
trace_emit(atoms, "Lives", atoms.idx_lives)
trace_emit(atoms, "available:", atoms.idx_available)
lives = int(np.asarray(new_state.life_tokens).sum())
trace_emit(atoms, "<lives-num>", int(atoms.idx_number[lives]))
trace_emit(atoms, "Deck", atoms.idx_deck)
trace_emit(atoms, "remaining", atoms.idx_remaining)
trace_emit(atoms, "cards:", atoms.idx_cards_label)
deck = int(np.asarray(new_state.deck).sum())
trace_emit(atoms, "<deck-num>", int(atoms.idx_number[deck]))
trace_emit(atoms, "Discards:", atoms.idx_discards)
trace_emit(atoms, "Fireworks:", atoms.idx_fireworks)
trace_emit(atoms, "Your", atoms.idx_your)
trace_emit(atoms, "Hand:", atoms.idx_hand)
trace_emit(atoms, "<slot=0>", int(atoms.idx_number[0]))
trace_emit(atoms, "Hints:", atoms.idx_hints)
trace_emit(atoms, ",", atoms.idx_only_comma)
trace_emit(atoms, "Possible:", atoms.idx_possible_label)
# All knowledge bits set initially:
trace_emit(atoms, "RYGWB12345", int(atoms.idx_possible[31, 31]))
trace_emit(atoms, "<slot=1>", int(atoms.idx_number[1]))
trace_emit(atoms, "Hints:", atoms.idx_hints)

# Now compare HF tokenization of each word vs my atom for that word.
print("\nPer-word HF vs atom comparison (mismatches only):")
checked = set()
for w in words:
    if w in checked:
        continue
    checked.add(w)
    hf_tok = tokenizer(w, add_special_tokens=False, return_tensors="np")["input_ids"][0].tolist()
    if w in atoms._index:
        atom_idx = atoms._index[w]
        L = int(atoms.atoms_len[atom_idx])
        atom_tok = atoms.atoms_arr[atom_idx, :L].tolist()
        if atom_tok != hf_tok:
            print(f"  MISMATCH word={w!r}: HF={hf_tok} atom={atom_tok}")
    else:
        print(f"  MISSING word={w!r}: not in atom dict (HF tokens={hf_tok})")
PYEOF

echo "=== done $(date) ==="
