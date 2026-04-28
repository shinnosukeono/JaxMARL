"""
LLM prior loader + state-key generator for jaxmarl Hanabi.

The R3D2 paper attaches a Π-KL penalty toward an LLM prior over actions.
The prior files at `pyhanabi/openai/<prompt>/prior.pkl` are dicts of the form

    prior[history_key][move_key]  ->  (prompt_str, action_str, logit)
                                      OR a bare logit float

where `history_key` is a string describing the most recent move (or '[null]'
for the start of the game) and `move_key` is HLE's `move.to_string()`. We
re-implement *both* sides in jaxmarl-compatible form so existing prior pkls
load unchanged.

History key format (from `<(MOVE) by player N [reveal SLOTS]>`):

    [null]                                               # turn 0
    <(Discard 0) by player 1>                            # discard / play
    <(Reveal player +1 color R) by player 1 reveal 0,2>  # hint, with affected slots

Move key format (HLE `move.to_string()`):

    (Play 0)
    (Discard 3)
    (Reveal player +1 color R)
    (Reveal player +1 rank 3)

This file:
  * builds a (jaxmarl_action_idx -> HLE move_key) mapping per env
  * loads `prior.pkl` and produces a `dict[history_key -> np.float32[num_moves]]`
  * exposes `make_prior_key_fn(env)` that maps (action, old_state, new_state)
    -> history_key, suitable for use inside a `jax.pure_callback`.

Usage:
    prior = load_prior(env, "openai/meta_prompt6_color/prior.pkl")
    key_fn = make_prior_key_fn(env)
    h = key_fn(action_int, old_state, new_state)  # returns "[null]" or "<...>"
    log_pi = prior[h] if h in prior else np.zeros(env.num_moves)
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional

import numpy as np

from text_obs import _action_kind, _last_action_feats_np


# ---------------------------------------------------------------------------
# jaxmarl action_idx -> HLE move-key string
# ---------------------------------------------------------------------------


def jaxmarl_action_to_hle_key(env, action: int) -> str:
    """Return HLE-style `(Play 0)` / `(Reveal player +1 color R)` key.

    The relative offset in jaxmarl matches HLE: `+1` is the next player, etc.
    """
    a = int(action)
    kind = _action_kind(env, a)
    if kind == "play":
        slot = a - int(env.play_action_range[0])
        return f"(Play {slot})"
    if kind == "discard":
        slot = a - int(env.discard_action_range[0])
        return f"(Discard {slot})"
    if kind == "hint_color":
        local = a - int(env.color_action_range[0])
        color_idx = local % env.num_colors
        target_rel = local // env.num_colors  # 0..num_agents-2
        return f"(Reveal player +{target_rel + 1} color {env.color_map[color_idx]})"
    if kind == "hint_rank":
        local = a - int(env.rank_action_range[0])
        rank_idx = local % env.num_ranks
        target_rel = local // env.num_ranks
        # HLE prints rank as 1-indexed digit (RankIndexToChar uses "12345")
        return f"(Reveal player +{target_rel + 1} rank {rank_idx + 1})"
    return "(noop)"


def all_hle_move_keys(env) -> List[str]:
    """One HLE move key per jaxmarl action index, in jaxmarl ordering."""
    return [jaxmarl_action_to_hle_key(env, i) for i in range(env.num_moves)]


# ---------------------------------------------------------------------------
# Prior pkl -> dict[history_key -> np.float32[num_moves]]
# ---------------------------------------------------------------------------


def load_prior(env, pkl_path: str) -> Dict[str, np.ndarray]:
    """Load a `prior.pkl` and return per-history logit vectors aligned with
    jaxmarl's action ordering.

    Missing move keys (because the prior was generated for a game variant with
    a different move set, or we hit a noop) get logit 0.
    """
    raw = pickle.load(open(pkl_path, "rb"))
    move_keys = all_hle_move_keys(env)
    prior: Dict[str, np.ndarray] = {}
    for hist_key, move_dict in raw.items():
        logits = np.zeros(env.num_moves, dtype=np.float32)
        for i, mk in enumerate(move_keys):
            v = move_dict.get(mk)
            if v is None:
                continue
            if isinstance(v, tuple):
                logit = v[-1]
            else:
                logit = v
            logits[i] = float(logit)
        prior[hist_key] = logits
    return prior


# ---------------------------------------------------------------------------
# Build history keys at runtime
# ---------------------------------------------------------------------------


def _hint_reveal_slots(env, old_state, new_state, action: int) -> List[int]:
    """Indices of slots affected by a hint action (jaxmarl numpy state)."""
    feats = _last_action_feats_np(env, 0, old_state, new_state, int(action))
    return [int(s) for s in np.where(feats["reveal_outcome"])[0]]


def make_prior_key_fn(env):
    """Return f(action, old_state, new_state) -> hist_key string.

    `old_state`, `new_state` are jaxmarl State pytrees with numpy leaves
    (ie. concrete, off-device). `action` is an int.
    """

    def key_fn(action: int, old_state, new_state) -> str:
        # Turn 0 -> [null]; new_state.turn == 0 means no real action has been played yet.
        if int(np.asarray(new_state.turn)) == 0:
            return "[null]"
        acting_idx = int(np.flatnonzero(np.asarray(old_state.cur_player_idx))[0])
        move_key = jaxmarl_action_to_hle_key(env, int(action))
        kind = _action_kind(env, int(action))
        if kind in ("hint_color", "hint_rank"):
            slots = _hint_reveal_slots(env, old_state, new_state, int(action))
            slot_str = ",".join(str(s) for s in slots)
            return f"<{move_key} by player {acting_idx} reveal {slot_str}>"
        return f"<{move_key} by player {acting_idx}>"

    return key_fn


# ---------------------------------------------------------------------------
# pure_callback wrapper for batched lookups inside jit
# ---------------------------------------------------------------------------


def make_prior_lookup_fn(env, prior: Dict[str, np.ndarray]):
    """Return a host function suitable for `jax.pure_callback` that maps batched
    (action, old_state, new_state) to per-action prior logits.

    Output shape: (B, num_moves), dtype float32.
    Missing histories yield all-zero logits.
    """
    key_fn = make_prior_key_fn(env)
    fallback = np.zeros(env.num_moves, dtype=np.float32)
    num_moves = env.num_moves

    def host_fn(state_leaves, old_leaves, actions):
        from jaxmarl.environments.hanabi.hanabi_game import State
        actions = np.asarray(actions)
        B = actions.shape[0]
        out = np.zeros((B, num_moves), dtype=np.float32)
        new_dict = {k: np.asarray(v) for k, v in state_leaves.items()}
        old_dict = {k: np.asarray(v) for k, v in old_leaves.items()}
        for b in range(B):
            new_s = State(**{k: v[b] for k, v in new_dict.items()})
            old_s = State(**{k: v[b] for k, v in old_dict.items()})
            h = key_fn(int(actions[b]), old_s, new_s)
            out[b] = prior.get(h, fallback)
        return out

    def call(new_state, old_state, last_actions):
        import jax
        import jax.numpy as jnp
        spec = jax.ShapeDtypeStruct((last_actions.shape[0], num_moves), jnp.float32)

        def state_to_dict(s):
            return {k: getattr(s, k) for k in s.__dataclass_fields__}

        return jax.pure_callback(
            host_fn,
            spec,
            state_to_dict(new_state),
            state_to_dict(old_state),
            last_actions,
            vmap_method="sequential",
        )

    return call
