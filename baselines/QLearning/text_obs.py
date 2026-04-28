"""
Text-observation infrastructure for R3D2-style baselines on jaxmarl Hanabi.

The Hanabi env's `get_obs_str()` is pure Python (string formatting + control
flow), so it can't run inside `jit`. We expose two pieces:

  1. `render_obs_for_agent(env, new_state, old_state, action, aidx, ...)`
     A NumPy/Python re-implementation of `HanabiEnv.get_obs_str` that takes
     `aidx` instead of always rendering for the current player.

  2. `make_tokenize_fn(env, tokenizer, max_obs_tokens, ...)`
     Returns a function `f(env_state, old_env_state, last_action) -> tokens`
     that may be called inside a jit'd rollout. Internally it calls
     `jax.pure_callback` once with the full batched state to keep host
     roundtrips at one-per-env-step rather than one-per-(env, agent) pair.

Both pieces also build action-strings (used by R3D2 / DRRN where each legal
action is text-encoded).
"""

from __future__ import annotations

import itertools
from functools import partial
from typing import List

import numpy as np
import jax
import jax.numpy as jnp


# Action-range helpers (numpy-only, mirroring HanabiGame._is_* but without jit).

def _action_kind(env, action: int) -> str:
    a = int(action)
    discard_lo, discard_hi = int(env.discard_action_range[0]), int(env.discard_action_range[-1])
    play_lo, play_hi = int(env.play_action_range[0]), int(env.play_action_range[-1])
    color_lo, color_hi = int(env.color_action_range[0]), int(env.color_action_range[-1])
    rank_lo, rank_hi = int(env.rank_action_range[0]), int(env.rank_action_range[-1])
    if discard_lo <= a <= discard_hi:
        return "discard"
    if play_lo <= a <= play_hi:
        return "play"
    if color_lo <= a <= color_hi:
        return "hint_color"
    if rank_lo <= a <= rank_hi:
        return "hint_rank"
    return "noop"


# ---------------------------------------------------------------------------
# Per-agent text rendering
# ---------------------------------------------------------------------------


def _card_to_string(env, card_arr: np.ndarray) -> str:
    if not card_arr.any():
        return ""
    color = int(np.argmax(card_arr.sum(axis=1), axis=0))
    rank = int(np.argmax(card_arr.sum(axis=0), axis=0))
    return f"{env.color_map[color]}{rank + 1}"


def _last_action_feats_np(env, aidx, old_state, new_state, action):
    """NumPy port of HanabiEnv.get_last_action_feats_ enough for text rendering."""
    kind = _action_kind(env, action)
    feats = {}

    acting_idx = int(np.flatnonzero(np.asarray(old_state.cur_player_idx))[0])

    # Target player + hint index for hint actions.
    target_player = 0
    hint_idx = 0
    if kind == "hint_color":
        action_idx = int(action) - 2 * env.hand_size
        hint_idx = action_idx % env.num_colors
        target_player_rel = action_idx // env.num_colors
        # Convert relative (skipping acting_idx) to absolute.
        rel_to_abs = [(acting_idx + 1 + r) % env.num_agents for r in range(env.num_agents - 1)]
        target_player = rel_to_abs[target_player_rel]
    elif kind == "hint_rank":
        action_idx = int(action) - 2 * env.hand_size - (env.num_agents - 1) * env.num_colors
        hint_idx = action_idx % env.num_ranks
        target_player_rel = action_idx // env.num_ranks
        rel_to_abs = [(acting_idx + 1 + r) % env.num_agents for r in range(env.num_agents - 1)]
        target_player = rel_to_abs[target_player_rel]

    target_hand = np.asarray(new_state.player_hands[target_player])

    color_revealed = np.zeros(env.num_colors)
    rank_revealed = np.zeros(env.num_ranks)
    if kind == "hint_color":
        color_revealed[hint_idx] = 1
    if kind == "hint_rank":
        rank_revealed[hint_idx] = 1

    if kind == "hint_color":
        color_match = (target_hand.sum(axis=2) == color_revealed).all(axis=1)
    else:
        color_match = np.zeros(env.hand_size, dtype=bool)
    if kind == "hint_rank":
        rank_match = (target_hand.sum(axis=1) == rank_revealed).all(axis=1)
    else:
        rank_match = np.zeros(env.hand_size, dtype=bool)
    feats["reveal_outcome"] = (color_match | rank_match).astype(np.int32)

    if kind in ("play", "discard"):
        slot = int(action) % env.hand_size
        actor_hand_before = np.asarray(old_state.player_hands[acting_idx])
        feats["played_discarded_card"] = actor_hand_before[slot].ravel()
    else:
        feats["played_discarded_card"] = np.zeros(env.num_colors * env.num_ranks)

    fw_old = float(np.asarray(old_state.fireworks).sum())
    fw_new = float(np.asarray(new_state.fireworks).sum())
    feats["card_played_score"] = np.array([1 if fw_new != fw_old else 0])

    info_old = float(np.asarray(old_state.info_tokens).sum())
    info_new = float(np.asarray(new_state.info_tokens).sum())
    feats["added_info_tokens"] = np.array([1 if (kind == "play" and info_new > info_old) else 0])

    return feats


def _v0_belief_for_agent(env, aidx, new_state):
    """NumPy port of HanabiEnv.get_v0_belief_feats for a specific viewer aidx."""
    full_deck = np.asarray(env.get_full_deck())  # (deck_size, num_colors, num_ranks)
    discard_pile = np.asarray(new_state.discard_pile)
    fireworks = np.asarray(new_state.fireworks)
    knowledge = np.asarray(new_state.card_knowledge)
    colors_revealed = np.asarray(new_state.colors_revealed)
    ranks_revealed = np.asarray(new_state.ranks_revealed)

    rel = lambda x: np.roll(x, -aidx, axis=0)
    knowledge = rel(knowledge)
    colors_revealed = rel(colors_revealed)
    ranks_revealed = rel(ranks_revealed)

    count = (
        full_deck.sum(axis=0).ravel()
        - discard_pile.sum(axis=0).ravel()
        - fireworks.ravel()
    )

    out = []
    for h in range(env.num_agents):
        hand_knowledge = knowledge[h].reshape(env.hand_size, -1)
        beliefs = []
        for s in range(env.hand_size):
            k = hand_knowledge[s]
            if k.sum() == 0:
                beliefs.append(np.zeros_like(k))
            else:
                num = k * count
                if num.sum() == 0:
                    beliefs.append(np.zeros_like(k))
                else:
                    beliefs.append(num / num.sum())
        out.append(np.stack(beliefs, axis=0))
    return np.stack(out, axis=0)  # (num_agents, hand_size, num_colors*num_ranks)


def render_obs_for_agent(
    env,
    new_state,
    old_state,
    action,
    aidx: int,
    include_belief: bool = False,
    best_belief: int = 5,
) -> str:
    """Per-agent variant of HanabiEnv.get_obs_str.

    `new_state`, `old_state` should be `State` instances with concrete numpy
    leaves (call after pulling them off-device).
    """
    keep_only_last_one = lambda x: np.where(
        np.arange(x.size) < (x.size - 1 - np.argmax(x[::-1])),
        0,
        x,
    )
    fireworks = np.stack(
        [keep_only_last_one(np.asarray(new_state.fireworks[i])) for i in range(env.num_colors)],
        axis=0,
    )
    fireworks_cards = []
    for i in range(env.num_colors):
        c = np.zeros((env.num_colors, env.num_ranks))
        c[i] = fireworks[i]
        fireworks_cards.append(c)

    board_info = {
        "turn": int(new_state.turn),
        "score": int(new_state.score),
        "information available": int(np.asarray(new_state.info_tokens).sum()),
        "lives available": int(np.asarray(new_state.life_tokens).sum()),
        "deck remaining cards": int(np.asarray(new_state.deck).sum()),
        "discards": " ".join(_card_to_string(env, np.asarray(c)) for c in new_state.discard_pile),
        "fireworks": " ".join(_card_to_string(env, c) for c in fireworks_cards),
    }

    output = ""
    for i, (k, v) in enumerate(board_info.items()):
        output += f"{k.capitalize()}: {v}\n"
        if i == 0:
            output += "\n"

    belief = _v0_belief_for_agent(env, aidx, new_state) if include_belief else None
    if belief is not None:
        belief = np.roll(belief, aidx, axis=0)  # match get_obs_str ordering

    colors_revealed_all = np.asarray(new_state.colors_revealed)
    ranks_revealed_all = np.asarray(new_state.ranks_revealed)
    knowledge_all = np.asarray(new_state.card_knowledge)
    hands_all = np.asarray(new_state.player_hands)

    for ai in range(env.num_agents):
        is_self = ai == aidx
        output += ("Your Hand:" if is_self else "Other Hand:") + "\n"
        cr = colors_revealed_all[ai]
        rr = ranks_revealed_all[ai]
        kn = knowledge_all[ai].reshape(env.hand_size, env.num_colors, env.num_ranks)
        hand = hands_all[ai]
        for slot in range(env.hand_size):
            ch = cr[slot]
            rh = rr[slot]
            card_hint = (
                ("" if not ch.any() else env.color_map[int(np.argmax(ch))])
                + ("" if not rh.any() else str(int(np.argmax(rh)) + 1))
            )
            color_known = kn[slot].any(axis=1)
            rank_known = kn[slot].any(axis=0)
            color_str = "".join(c for c, k in zip(env.color_map, color_known) if k)
            rank_str = "".join(str(r + 1) for r, k in enumerate(rank_known) if k)
            ck = f"Hints: {card_hint}, Possible: {color_str + rank_str}"
            if include_belief and belief is not None:
                cb = belief[ai][slot]
                top = np.argsort(-cb)[:best_belief]
                cb_str = " ".join(
                    f"{env.color_map[c]}{r + 1}: {cb[i]:.3f}"
                    for i, (c, r) in enumerate(itertools.product(range(env.num_colors), range(env.num_ranks)))
                    if i in top
                )
                ck += f", Belief: [{cb_str}]"
            card_str = _card_to_string(env, hand[slot])
            output += f"{slot} {('' if is_self else f'Card: {card_str}, ')}{ck}\n"

    # last action
    move_type = env.action_encoding[int(action)]
    output += f"Last action: {move_type}\n"

    if old_state is not None and int(new_state.turn) > 0:
        feats = _last_action_feats_np(env, aidx, old_state, new_state, action)
        if move_type[0] == "H":
            affected = np.where(feats["reveal_outcome"])[0]
            output += f"Cards afected: {affected}\n"
        elif move_type[0] in ("D", "P"):
            cp = feats["played_discarded_card"].reshape(env.num_colors, env.num_ranks)
            output += f"Card Played: {_card_to_string(env, cp)}\n"
        if move_type[0] == "P":
            output += f"Scored: {int(feats['card_played_score'][0])}\n"
            output += f"Added Info: {int(feats['added_info_tokens'][0])}\n"

    # legal actions for this agent
    # env.get_legal_moves uses jnp ops internally — convert state leaves to jnp first.
    from jaxmarl.environments.hanabi.hanabi_game import State
    jnp_state = State(**{k: jnp.asarray(getattr(new_state, k)) for k in new_state.__dataclass_fields__})
    legal_moves = np.asarray(env.get_legal_moves(jnp_state)[env.agents[aidx]])
    legal_actions = [env.action_encoding[int(a)] for a in np.where(legal_moves)[0]]
    output += f"Legal Actions: {legal_actions}\n"

    return output


# ---------------------------------------------------------------------------
# Action-string rendering (used by R3D2's DRRN action encoder)
# ---------------------------------------------------------------------------


def all_action_strings(env) -> List[str]:
    """Return one text string per action index, suitable for tokenization once.

    These strings are independent of state, so we can tokenize them once and
    cache. Note action indices > num_moves never appear (CTRolloutManager pads
    the action space to max_action_space; padded slots correspond to no-ops in
    other player-count settings).
    """
    return [env.action_encoding[i] for i in range(env.num_moves)]


# ---------------------------------------------------------------------------
# pure_callback-based batched tokenizer
# ---------------------------------------------------------------------------


def _state_from_arrays(env, arrays):
    """Reconstruct a HanabiGame State from a flat dict of numpy arrays."""
    from jaxmarl.environments.hanabi.hanabi_game import State
    return State(**arrays)


def _state_field_specs(env, batch_size: int):
    """Return ShapeDtypeStruct specs for one State leaf at batch_size B."""
    # Build a dummy state to extract leaf shapes/dtypes.
    state = env.reset_game(jax.random.PRNGKey(0))
    # state is one example; expand each leaf with leading batch dim.
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct((batch_size,) + x.shape, x.dtype), state
    )


def make_tokenize_fn(
    env,
    tokenizer,
    max_obs_tokens: int = 256,
    include_belief: bool = False,
):
    """Build a function that tokenizes the per-agent text obs of every env in a batch.

    Returned function signature:
        f(new_state, old_state, last_actions)
            -> input_ids:      int32[num_agents, B, max_obs_tokens]
               attention_mask: int32[num_agents, B, max_obs_tokens]

    All inputs are vmap'd batched pytrees (leading axis = B = batch of envs).
    `last_actions` is shape (B,) — the action that took old_state -> new_state.
    """
    num_agents = env.num_agents
    pad_id = int(tokenizer.pad_token_id or 0)

    def host_fn(state_leaves, old_state_leaves, last_actions):
        from jaxmarl.environments.hanabi.hanabi_game import State

        last_actions = np.asarray(last_actions)
        B = last_actions.shape[0]

        # Reconstruct dict of leaves -> per-batch index
        new_field_dict = {k: np.asarray(v) for k, v in state_leaves.items()}
        old_field_dict = {k: np.asarray(v) for k, v in old_state_leaves.items()}

        ids_out = np.full((num_agents, B, max_obs_tokens), pad_id, dtype=np.int32)
        mask_out = np.zeros((num_agents, B, max_obs_tokens), dtype=np.int32)

        for b in range(B):
            new_s = State(**{k: v[b] for k, v in new_field_dict.items()})
            old_s = State(**{k: v[b] for k, v in old_field_dict.items()})
            for ai in range(num_agents):
                txt = render_obs_for_agent(
                    env, new_s, old_s, int(last_actions[b]), ai,
                    include_belief=include_belief,
                )
                enc = tokenizer(
                    txt,
                    padding="max_length",
                    truncation=True,
                    max_length=max_obs_tokens,
                    return_tensors="np",
                )
                ids_out[ai, b] = enc["input_ids"][0]
                mask_out[ai, b] = enc["attention_mask"][0]
        return ids_out, mask_out

    def call(new_state, old_state, last_actions):
        # Build callback result spec from a sample shape.
        # We need to know B at trace time; pull it off last_actions.
        spec = (
            jax.ShapeDtypeStruct((num_agents, last_actions.shape[0], max_obs_tokens), jnp.int32),
            jax.ShapeDtypeStruct((num_agents, last_actions.shape[0], max_obs_tokens), jnp.int32),
        )

        # Convert State pytrees to plain dicts so they cross the callback boundary cleanly.
        def state_to_dict(s):
            from jaxmarl.environments.hanabi.hanabi_game import State
            assert isinstance(s, State), f"expected State, got {type(s)}"
            return {k: getattr(s, k) for k in s.__dataclass_fields__}

        return jax.pure_callback(
            host_fn,
            spec,
            state_to_dict(new_state),
            state_to_dict(old_state),
            last_actions,
            vmap_method="sequential",  # B dimension is consumed inside host_fn
        )

    return call


def tokenize_action_set(env, tokenizer, max_action_tokens: int):
    """Tokenize all action strings ONCE; return jnp arrays for jit.

    Returns:
        action_input_ids: int32[max_action_space, max_action_tokens]
        action_attn_mask: int32[max_action_space, max_action_tokens]

    The first `env.num_moves` rows are real actions; remaining rows
    (if any, due to CTRolloutManager padding to max across player-counts)
    are filled with PAD tokens.
    """
    pad_id = int(tokenizer.pad_token_id or 0)
    strings = all_action_strings(env)
    enc = tokenizer(
        strings,
        padding="max_length",
        truncation=True,
        max_length=max_action_tokens,
        return_tensors="np",
    )
    return jnp.asarray(enc["input_ids"], dtype=jnp.int32), jnp.asarray(
        enc["attention_mask"], dtype=jnp.int32
    )
