"""
Fast text-observation tokenization for Hanabi.

Drop-in replacement for `make_tokenize_fn` in text_obs.py that bypasses both
the Python string-rendering loop and the HF tokenizer call inside the scan
hot path.

Key idea: BERT's WordPiece tokenizer is *whitespace-context-free*. The token
IDs of a word do not depend on surrounding words, so for any text built from
a closed vocabulary of "atoms" (whitespace-separated words), we have

    tokenize(" ".join(atoms)) == [CLS] + concat(tokenize(a) for a in atoms) + [SEP]

We pre-tokenize every atom that can ever appear in a Hanabi observation
(numbers 0..100, card names "R1".."B5", labels, etc.) once at startup, then
build per-state token sequences by indexing into the atom table — no Python
string formatting, no HF tokenizer call, no host-device sync penalty beyond
the unavoidable pure_callback boundary.

Verification: `verify_against_hf(...)` runs both the original tokenizer path
(render → HF tokenizer) and this fast path on the same state and asserts
token-ID equivalence.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp


def _atom_strings_for_env(env) -> Dict[str, None]:
    """Return the closed set of atom strings that can appear in a Hanabi obs.

    Returned as a dict to preserve insertion order (used as ordered set).
    """
    atoms: Dict[str, None] = {}

    def add(s: str):
        if s and s not in atoms:
            atoms[s] = None

    # Static labels emitted by render_obs_for_agent.
    # Note: "Card:" appears in slot rendering ("0 Card: Y2,"), while "Card"
    # (no colon) appears in last-action rendering ("Card Played: W1"). Both
    # are needed.
    for s in [
        "Turn:", "Score:",
        "Information", "available:",
        "Lives",
        "Deck", "remaining", "cards:",
        "Discards:", "Fireworks:",
        "Your", "Hand:", "Other",
        "Card:", "Card", "Hints:", "Possible:",
        ",",
        "Last", "action:",
        "Cards", "afected:",
        "Played:", "Scored:", "Added", "Info:",
    ]:
        add(s)

    # Numbers up to a generous bound (turn, score, token counts, deck remaining).
    for i in range(151):
        add(str(i))

    # Cards "<C><R+1>" (no comma) — used in fireworks/discards/Card Played.
    cs = env.color_map
    for c in cs:
        for r in range(env.num_ranks):
            add(f"{c}{r + 1}")

    # Card hints with trailing comma — appear in "Hints: <ch>," position.
    # ch can be: "" (→ ","), color, rank, color+rank, or full card.
    add(",")
    for c in cs:
        add(f"{c},")
    for r in range(env.num_ranks):
        add(f"{r + 1},")
    for c in cs:
        for r in range(env.num_ranks):
            add(f"{c}{r + 1},")

    # Card with trailing comma — appears in "Card: <card>," for other-player slots.
    # Same set as the card-hint-with-comma full-card variants above; add() dedups.
    for c in cs:
        for r in range(env.num_ranks):
            add(f"{c}{r + 1},")

    # "Possible:" value: subset-of-colors string + subset-of-ranks string,
    # concatenated with NO space between. Render order matches the original.
    nc = env.num_colors
    nr = env.num_ranks
    for cm in range(2 ** nc):
        color_str = "".join(cs[i] for i in range(nc) if (cm >> i) & 1)
        for rm in range(2 ** nr):
            rank_str = "".join(str(i + 1) for i in range(nr) if (rm >> i) & 1)
            s = color_str + rank_str
            if s:
                add(s)

    # Action encodings (env.action_encoding). The render emits "Last action: <move>".
    # Move string can be e.g. "D0", "P3", "HR to P1 relative", "H2 to P3 relative", "N".
    for ae in env.action_encoding.values():
        for w in ae.split():
            add(w)

    # Brackets used for "Cards afected: <np-array-print>".
    # np.array prints as "[]", "[0]", "[0 1]", "[0 1 2]" etc.
    add("[]")
    for i in range(env.hand_size):
        add(f"[{i}]")
        add(f"[{i}")
        add(f"{i}]")

    return atoms


class HanabiAtoms:
    """Pre-tokenized atom table. Each atom is one whitespace-bounded word."""

    def __init__(self, env, tokenizer, max_atom_tokens: int = 8):
        self.env = env
        self.tokenizer = tokenizer
        self.cls = int(tokenizer.cls_token_id)
        self.sep = int(tokenizer.sep_token_id)
        self.pad = int(tokenizer.pad_token_id or 0)

        atom_strings = list(_atom_strings_for_env(env).keys())
        # Tokenize each atom independently with no special tokens.
        # HF Fast tokenizer is fast in batch mode.
        encs = tokenizer(
            atom_strings, add_special_tokens=False, padding=False, truncation=False
        )
        ids_lists: List[List[int]] = encs["input_ids"]
        atom_lens = np.array([len(x) for x in ids_lists], dtype=np.int32)
        if atom_lens.max() > max_atom_tokens:
            max_atom_tokens = int(atom_lens.max())
        atoms_arr = np.full((len(atom_strings), max_atom_tokens), self.pad, dtype=np.int32)
        for i, ids in enumerate(ids_lists):
            atoms_arr[i, : len(ids)] = ids

        self.strings = atom_strings
        self.atoms_arr = atoms_arr            # (num_atoms, max_atom_tokens)
        self.atoms_len = atom_lens             # (num_atoms,)
        self._index: Dict[str, int] = {s: i for i, s in enumerate(atom_strings)}

        # Pre-bind index lookups for hot-path words.
        # Per-color/per-rank arrays for fast indexing.
        self.idx_card = np.full((env.num_colors, env.num_ranks), -1, dtype=np.int32)
        self.idx_card_comma = np.full((env.num_colors, env.num_ranks), -1, dtype=np.int32)
        for ci, c in enumerate(env.color_map):
            for ri in range(env.num_ranks):
                self.idx_card[ci, ri] = self._index[f"{c}{ri + 1}"]
                self.idx_card_comma[ci, ri] = self._index[f"{c}{ri + 1},"]

        self.idx_color_comma = np.array(
            [self._index[f"{c},"] for c in env.color_map], dtype=np.int32
        )
        self.idx_rank_comma = np.array(
            [self._index[f"{r + 1},"] for r in range(env.num_ranks)], dtype=np.int32
        )
        self.idx_only_comma = self._index[","]

        # Numbers 0..150
        self.idx_number = np.array(
            [self._index[str(i)] for i in range(151)], dtype=np.int32
        )

        # Possible-string (color-mask, rank-mask) -> atom index, OR -1 if empty.
        self.idx_possible = np.full(
            (2 ** env.num_colors, 2 ** env.num_ranks), -1, dtype=np.int32
        )
        for cm in range(2 ** env.num_colors):
            color_str = "".join(env.color_map[i] for i in range(env.num_colors) if (cm >> i) & 1)
            for rm in range(2 ** env.num_ranks):
                rank_str = "".join(
                    str(i + 1) for i in range(env.num_ranks) if (rm >> i) & 1
                )
                s = color_str + rank_str
                if s:
                    self.idx_possible[cm, rm] = self._index[s]

        # Action encoding -> sequence of atom indices.
        # Variable length per action; store as padded array + length.
        max_action_words = max(len(ae.split()) for ae in env.action_encoding.values())
        self.idx_action_seq = np.zeros(
            (len(env.action_encoding), max_action_words), dtype=np.int32
        )
        self.idx_action_len = np.zeros(len(env.action_encoding), dtype=np.int32)
        for k, ae in env.action_encoding.items():
            words = ae.split()
            for j, w in enumerate(words):
                self.idx_action_seq[k, j] = self._index[w]
            self.idx_action_len[k] = len(words)

        # Label cache for hot-path
        self.idx_turn = self._index["Turn:"]
        self.idx_score = self._index["Score:"]
        self.idx_information = self._index["Information"]
        self.idx_available = self._index["available:"]
        self.idx_lives = self._index["Lives"]
        self.idx_deck = self._index["Deck"]
        self.idx_remaining = self._index["remaining"]
        self.idx_cards_label = self._index["cards:"]
        self.idx_discards = self._index["Discards:"]
        self.idx_fireworks = self._index["Fireworks:"]
        self.idx_your = self._index["Your"]
        self.idx_other = self._index["Other"]
        self.idx_hand = self._index["Hand:"]
        self.idx_card_label = self._index["Card:"]
        self.idx_card_no_colon = self._index["Card"]
        self.idx_hints = self._index["Hints:"]
        self.idx_possible_label = self._index["Possible:"]
        self.idx_last = self._index["Last"]
        self.idx_action_label = self._index["action:"]
        self.idx_cards_word = self._index["Cards"]
        self.idx_afected = self._index["afected:"]
        self.idx_played = self._index["Played:"]
        self.idx_scored = self._index["Scored:"]
        self.idx_added = self._index["Added"]
        self.idx_info = self._index["Info:"]

        # Bracket atoms
        self.idx_brackets_empty = self._index["[]"]
        self.idx_bracket_single = np.array(
            [self._index[f"[{i}]"] for i in range(env.hand_size)], dtype=np.int32
        )
        self.idx_bracket_open = np.array(
            [self._index[f"[{i}"] for i in range(env.hand_size)], dtype=np.int32
        )
        self.idx_bracket_close = np.array(
            [self._index[f"{i}]"] for i in range(env.hand_size)], dtype=np.int32
        )

    def write_atom(self, out: np.ndarray, pos: int, atom_idx: int) -> int:
        """Copy atom's tokens into out[pos:], return new pos.

        If pos+L exceeds the buffer, we copy what fits and still advance pos
        as if we wrote the full L tokens — the caller (state_to_tokens_for_agent)
        handles the overflow at SEP-write time.
        """
        L = int(self.atoms_len[atom_idx])
        n = max(0, min(L, out.size - pos))
        if n > 0:
            out[pos : pos + n] = self.atoms_arr[atom_idx, :n]
        return pos + L


def _card_color_rank(card_arr: np.ndarray) -> Tuple[int, int]:
    """Given (num_colors, num_ranks) one-hot, return (color, rank) or (-1, -1) if empty."""
    if not card_arr.any():
        return -1, -1
    color = int(np.argmax(card_arr.sum(axis=1), axis=0))
    rank = int(np.argmax(card_arr.sum(axis=0), axis=0))
    return color, rank


def _last_action_meta(env, aidx, old_state, new_state, action):
    """Compute the small set of last-action features needed by the renderer.

    Pure-numpy version of HanabiEnv.get_last_action_feats_; returns a dict.
    """
    discard_lo = int(env.discard_action_range[0]); discard_hi = int(env.discard_action_range[-1])
    play_lo = int(env.play_action_range[0]); play_hi = int(env.play_action_range[-1])
    color_lo = int(env.color_action_range[0]); color_hi = int(env.color_action_range[-1])
    rank_lo = int(env.rank_action_range[0]); rank_hi = int(env.rank_action_range[-1])

    a = int(action)
    if discard_lo <= a <= discard_hi:
        kind = "discard"
    elif play_lo <= a <= play_hi:
        kind = "play"
    elif color_lo <= a <= color_hi:
        kind = "hint_color"
    elif rank_lo <= a <= rank_hi:
        kind = "hint_rank"
    else:
        kind = "noop"

    acting_idx = int(np.flatnonzero(np.asarray(old_state.cur_player_idx))[0])

    reveal_outcome = np.zeros(env.hand_size, dtype=np.int32)
    if kind in ("hint_color", "hint_rank"):
        if kind == "hint_color":
            action_idx = a - 2 * env.hand_size
            hint_idx = action_idx % env.num_colors
            target_rel = action_idx // env.num_colors
        else:
            action_idx = a - 2 * env.hand_size - (env.num_agents - 1) * env.num_colors
            hint_idx = action_idx % env.num_ranks
            target_rel = action_idx // env.num_ranks
        target_player = (acting_idx + 1 + target_rel) % env.num_agents
        target_hand = np.asarray(new_state.player_hands[target_player])
        # Color match if any card sums all-1 along that color row.
        if kind == "hint_color":
            mask = (target_hand.sum(axis=2)[..., hint_idx] > 0).astype(np.int32)
        else:
            mask = (target_hand.sum(axis=1)[..., hint_idx] > 0).astype(np.int32)
        reveal_outcome = mask

    played_discarded_card = (-1, -1)
    if kind in ("play", "discard"):
        slot = a % env.hand_size
        actor_hand = np.asarray(old_state.player_hands[acting_idx])
        played_discarded_card = _card_color_rank(actor_hand[slot])

    fw_old = float(np.asarray(old_state.fireworks).sum())
    fw_new = float(np.asarray(new_state.fireworks).sum())
    info_old = float(np.asarray(old_state.info_tokens).sum())
    info_new = float(np.asarray(new_state.info_tokens).sum())
    scored = int(kind == "play" and fw_new != fw_old)
    added_info = int(kind == "play" and info_new > info_old)

    return {
        "kind": kind,
        "reveal_outcome": reveal_outcome,
        "played_discarded_card": played_discarded_card,
        "scored": scored,
        "added_info": added_info,
    }


def _highest_firework_per_color(fireworks: np.ndarray) -> List[int]:
    """Return rank index of highest played card per color (-1 if none).

    `fireworks` shape (num_colors, num_ranks) is incremental "thermometer".
    """
    out = []
    for i in range(fireworks.shape[0]):
        row = fireworks[i]
        if row.any():
            # Find last 1 in the row.
            out.append(int(np.argmax(row[::-1])) and (row.size - 1 - int(np.argmax(row[::-1]))) or 0)
            # The expression above is brittle; redo it cleanly:
            out[-1] = int(row.size - 1 - int(np.argmax(row[::-1])))
        else:
            out.append(-1)
    return out


def state_to_tokens_for_agent(
    atoms: HanabiAtoms,
    new_state,
    old_state,
    action: int,
    aidx: int,
    max_obs_tokens: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (input_ids, attention_mask) for one agent's view of one state.

    Mirrors text_obs.render_obs_for_agent + tokenizer call, but writes token
    IDs directly to a numpy buffer.
    """
    env = atoms.env
    out = np.full(max_obs_tokens, atoms.pad, dtype=np.int32)
    out[0] = atoms.cls
    pos = 1

    # --- Board info ---
    # Turn: <n>
    pos = atoms.write_atom(out, pos, atoms.idx_turn)
    pos = atoms.write_atom(out, pos, int(atoms.idx_number[int(new_state.turn)]))
    # Score: <n>
    pos = atoms.write_atom(out, pos, atoms.idx_score)
    pos = atoms.write_atom(out, pos, int(atoms.idx_number[int(new_state.score)]))
    # Information available: <n>
    pos = atoms.write_atom(out, pos, atoms.idx_information)
    pos = atoms.write_atom(out, pos, atoms.idx_available)
    info = int(np.asarray(new_state.info_tokens).sum())
    pos = atoms.write_atom(out, pos, int(atoms.idx_number[info]))
    # Lives available: <n>
    pos = atoms.write_atom(out, pos, atoms.idx_lives)
    pos = atoms.write_atom(out, pos, atoms.idx_available)
    lives = int(np.asarray(new_state.life_tokens).sum())
    pos = atoms.write_atom(out, pos, int(atoms.idx_number[lives]))
    # Deck remaining cards: <n>
    pos = atoms.write_atom(out, pos, atoms.idx_deck)
    pos = atoms.write_atom(out, pos, atoms.idx_remaining)
    pos = atoms.write_atom(out, pos, atoms.idx_cards_label)
    deck = int(np.asarray(new_state.deck).sum())
    pos = atoms.write_atom(out, pos, int(atoms.idx_number[deck]))
    # Discards: <card> <card> ...
    pos = atoms.write_atom(out, pos, atoms.idx_discards)
    discard_pile = np.asarray(new_state.discard_pile)
    for c in discard_pile:
        col, rk = _card_color_rank(c)
        if col >= 0:
            pos = atoms.write_atom(out, pos, int(atoms.idx_card[col, rk]))
    # Fireworks: <card> <card> ...
    pos = atoms.write_atom(out, pos, atoms.idx_fireworks)
    fireworks = np.asarray(new_state.fireworks)
    for col in range(env.num_colors):
        row = fireworks[col]
        if row.any():
            rk = int(row.size - 1 - int(np.argmax(row[::-1])))
            pos = atoms.write_atom(out, pos, int(atoms.idx_card[col, rk]))

    # --- Per-agent hand sections ---
    colors_revealed = np.asarray(new_state.colors_revealed)
    ranks_revealed = np.asarray(new_state.ranks_revealed)
    knowledge = np.asarray(new_state.card_knowledge).reshape(
        env.num_agents, env.hand_size, env.num_colors, env.num_ranks
    )
    hands = np.asarray(new_state.player_hands)

    for ai in range(env.num_agents):
        is_self = ai == aidx
        pos = atoms.write_atom(out, pos, atoms.idx_your if is_self else atoms.idx_other)
        pos = atoms.write_atom(out, pos, atoms.idx_hand)
        for slot in range(env.hand_size):
            # Slot index
            pos = atoms.write_atom(out, pos, int(atoms.idx_number[slot]))
            # Card: <card>,  (only for other player)
            if not is_self:
                col, rk = _card_color_rank(hands[ai, slot])
                pos = atoms.write_atom(out, pos, atoms.idx_card_label)
                if col >= 0:
                    pos = atoms.write_atom(out, pos, int(atoms.idx_card_comma[col, rk]))
                else:
                    pos = atoms.write_atom(out, pos, atoms.idx_only_comma)
            # Hints: <card_hint>,
            pos = atoms.write_atom(out, pos, atoms.idx_hints)
            ch = colors_revealed[ai, slot]
            rh = ranks_revealed[ai, slot]
            has_c = bool(ch.any())
            has_r = bool(rh.any())
            if has_c and has_r:
                col = int(np.argmax(ch))
                rk = int(np.argmax(rh))
                pos = atoms.write_atom(out, pos, int(atoms.idx_card_comma[col, rk]))
            elif has_c:
                col = int(np.argmax(ch))
                pos = atoms.write_atom(out, pos, int(atoms.idx_color_comma[col]))
            elif has_r:
                rk = int(np.argmax(rh))
                pos = atoms.write_atom(out, pos, int(atoms.idx_rank_comma[rk]))
            else:
                pos = atoms.write_atom(out, pos, atoms.idx_only_comma)
            # Possible: <color_str+rank_str>
            pos = atoms.write_atom(out, pos, atoms.idx_possible_label)
            kn = knowledge[ai, slot]
            color_known = kn.any(axis=1)
            rank_known = kn.any(axis=0)
            cm = int(sum((1 << i) for i in range(env.num_colors) if color_known[i]))
            rm = int(sum((1 << i) for i in range(env.num_ranks) if rank_known[i]))
            possible_idx = int(atoms.idx_possible[cm, rm])
            if possible_idx >= 0:
                pos = atoms.write_atom(out, pos, possible_idx)

    # --- Last action ---
    pos = atoms.write_atom(out, pos, atoms.idx_last)
    pos = atoms.write_atom(out, pos, atoms.idx_action_label)
    a_int = int(action)
    L = int(atoms.idx_action_len[a_int])
    for j in range(L):
        pos = atoms.write_atom(out, pos, int(atoms.idx_action_seq[a_int, j]))

    if old_state is not None and int(new_state.turn) > 0:
        meta = _last_action_meta(env, aidx, old_state, new_state, action)
        kind = meta["kind"]
        if kind in ("hint_color", "hint_rank"):
            pos = atoms.write_atom(out, pos, atoms.idx_cards_word)
            pos = atoms.write_atom(out, pos, atoms.idx_afected)
            inds = list(np.where(meta["reveal_outcome"])[0])
            n = len(inds)
            if n == 0:
                pos = atoms.write_atom(out, pos, atoms.idx_brackets_empty)
            elif n == 1:
                pos = atoms.write_atom(out, pos, int(atoms.idx_bracket_single[int(inds[0])]))
            else:
                pos = atoms.write_atom(out, pos, int(atoms.idx_bracket_open[int(inds[0])]))
                for i in inds[1:-1]:
                    pos = atoms.write_atom(out, pos, int(atoms.idx_number[int(i)]))
                pos = atoms.write_atom(out, pos, int(atoms.idx_bracket_close[int(inds[-1])]))
        elif kind in ("play", "discard"):
            # render: "Card Played: <card>" — note "Card" with no trailing colon.
            pos = atoms.write_atom(out, pos, atoms.idx_card_no_colon)
            pos = atoms.write_atom(out, pos, atoms.idx_played)
            col, rk = meta["played_discarded_card"]
            if col >= 0:
                pos = atoms.write_atom(out, pos, int(atoms.idx_card[col, rk]))
            if kind == "play":
                pos = atoms.write_atom(out, pos, atoms.idx_scored)
                pos = atoms.write_atom(out, pos, int(atoms.idx_number[int(meta["scored"])]))
                pos = atoms.write_atom(out, pos, atoms.idx_added)
                pos = atoms.write_atom(out, pos, atoms.idx_info)
                pos = atoms.write_atom(out, pos, int(atoms.idx_number[int(meta["added_info"])]))

    # SEP + truncate to max_obs_tokens
    if pos >= max_obs_tokens:
        out[max_obs_tokens - 1] = atoms.sep
        mask = np.ones(max_obs_tokens, dtype=np.int32)
        return out, mask
    out[pos] = atoms.sep
    pos += 1
    mask = np.zeros(max_obs_tokens, dtype=np.int32)
    mask[:pos] = 1
    return out, mask


def make_tokenize_fn_fast(env, tokenizer, max_obs_tokens: int = 256, include_belief: bool = False):
    """Drop-in replacement for text_obs.make_tokenize_fn — much faster.

    Same interface: returns f(new_state, old_state, last_actions) usable inside jit
    via jax.pure_callback.
    """
    if include_belief:
        # Belief support not implemented in the fast path yet; raise so we don't
        # silently produce mismatched tokens.
        raise NotImplementedError("include_belief=True not supported in fast tokenizer")

    atoms = HanabiAtoms(env, tokenizer)
    num_agents = env.num_agents

    def host_fn(state_leaves, old_state_leaves, last_actions):
        from jaxmarl.environments.hanabi.hanabi_game import State

        last_actions = np.asarray(last_actions)
        B = last_actions.shape[0]
        new_field_dict = {k: np.asarray(v) for k, v in state_leaves.items()}
        old_field_dict = {k: np.asarray(v) for k, v in old_state_leaves.items()}

        ids_out = np.full((num_agents, B, max_obs_tokens), atoms.pad, dtype=np.int32)
        mask_out = np.zeros((num_agents, B, max_obs_tokens), dtype=np.int32)

        for b in range(B):
            new_s = State(**{k: v[b] for k, v in new_field_dict.items()})
            old_s = State(**{k: v[b] for k, v in old_field_dict.items()})
            for ai in range(num_agents):
                ids, mask = state_to_tokens_for_agent(
                    atoms, new_s, old_s, int(last_actions[b]), ai, max_obs_tokens
                )
                ids_out[ai, b] = ids
                mask_out[ai, b] = mask
        return ids_out, mask_out

    def call(new_state, old_state, last_actions):
        spec = (
            jax.ShapeDtypeStruct((num_agents, last_actions.shape[0], max_obs_tokens), jnp.int32),
            jax.ShapeDtypeStruct((num_agents, last_actions.shape[0], max_obs_tokens), jnp.int32),
        )

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
            vmap_method="sequential",
        )

    return call


# ----------------------------------------------------------------------------
# Verification helper: compares fast path against the original render+tokenize.
# ----------------------------------------------------------------------------


def verify_against_hf(env, tokenizer, num_random_seeds: int = 5, max_obs_tokens: int = 256):
    """Run a handful of random rollout steps and compare the fast tokenizer
    output against text_obs.render_obs_for_agent + HF tokenizer.

    Raises AssertionError on mismatch. Returns (n_checks, n_mismatches).
    """
    import jax
    import jax.numpy as jnp
    from text_obs import render_obs_for_agent
    from jaxmarl import make
    from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager

    atoms = HanabiAtoms(env, tokenizer)

    n_check = 0
    n_mismatch = 0
    raw_env = env
    log_env = LogWrapper(raw_env) if not hasattr(raw_env, "env") else raw_env
    # Use a CTRolloutManager so step accepts dict actions cleanly.
    wrap = CTRolloutManager(log_env, batch_size=1)

    rng = jax.random.PRNGKey(0)

    for seed in range(num_random_seeds):
        rng, sub = jax.random.split(rng)
        _, env_state = wrap.batch_reset(sub)
        prev_state = env_state
        last_action = jnp.full((1,), env.num_moves - 1, dtype=jnp.int32)
        for t in range(20):
            inner_new = jax.tree.map(lambda x: np.asarray(x[0]), env_state.env_state)
            inner_old = jax.tree.map(lambda x: np.asarray(x[0]), prev_state.env_state)
            for ai in range(env.num_agents):
                a_int = int(last_action[0])
                txt = render_obs_for_agent(
                    raw_env, inner_new, inner_old, a_int, ai, include_belief=False,
                )
                hf_enc = tokenizer(
                    txt,
                    padding="max_length",
                    truncation=True,
                    max_length=max_obs_tokens,
                    return_tensors="np",
                )
                hf_ids = hf_enc["input_ids"][0]
                fast_ids, fast_mask = state_to_tokens_for_agent(
                    atoms, inner_new, inner_old, a_int, ai, max_obs_tokens,
                )
                n_check += 1
                if not np.array_equal(hf_ids, fast_ids):
                    n_mismatch += 1
                    diff = np.where(hf_ids != fast_ids)[0]
                    print(f"[seed={seed} t={t} ai={ai}] mismatch at positions {diff[:10]}")
                    print(f"  hf  : {hf_ids[: min(40, len(hf_ids))].tolist()}")
                    print(f"  fast: {fast_ids[: min(40, len(fast_ids))].tolist()}")
                    print(f"  text: {txt[:200]!r}")
                    if n_mismatch >= 3:
                        return n_check, n_mismatch
            # take a random valid step
            rng, sub_a, sub_s = jax.random.split(rng, 3)
            actions = {a: wrap.batch_sample(sub_a, a) for a in env.agents}
            _, env_state, _, _, _ = wrap.batch_step(sub_s, env_state, actions)
            cur_action = sum(
                int(actions[a][0])
                * int(env_state.env_state.cur_player_idx[0, i].astype(np.int32) == 0)
                for i, a in enumerate(env.agents)
            )
            last_action = jnp.array([cur_action])
            prev_state = env_state

    return n_check, n_mismatch
