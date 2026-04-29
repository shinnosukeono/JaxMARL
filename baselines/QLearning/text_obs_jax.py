"""
Pure-JAX state-to-tokens for Hanabi.

Eliminates the per-scan-step `jax.pure_callback` boundary that dominated
rollout cost (~370 ms each, ~30 sec/update). Builds the same token IDs as
text_obs_fast (verified bit-for-bit) but entirely in jnp ops, so it lives
inside the jit'd rollout graph with no host roundtrips.

The scheme:
  1. At setup, pre-tokenize all "atoms" (whitespace-bounded words) that can
     appear in the obs (numbers, card names, labels, etc.) — same as
     text_obs_fast.HanabiAtoms.
  2. At runtime per state, build a fixed-shape `(max_seq_len,)` array of
     atom indices in the order the obs would emit them. Empty / skipped
     entries get atom_len = 0 (no-op).
  3. `assemble_tokens` runs `lax.scan` over the atom sequence, writing each
     atom's tokens into a fixed `max_obs_tokens` buffer via
     `dynamic_update_slice`, and finally inserts [SEP].

Drop-in replacement for `text_obs.make_tokenize_fn` /
`text_obs_fast.make_tokenize_fn_fast`.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from text_obs_fast import HanabiAtoms


def _make_jnp_atoms(atoms: HanabiAtoms):
    """Bind atoms_arr/atoms_len/lookup tables as jnp arrays for use in jit.

    Action-range bounds are baked in as Python ints (extracted from the env
    at setup time) so they don't become tracers inside jit.
    """
    env = atoms.env
    return dict(
        # static action-range ints
        discard_lo=int(np.asarray(env.discard_action_range[0])),
        discard_hi=int(np.asarray(env.discard_action_range[-1])),
        play_lo=int(np.asarray(env.play_action_range[0])),
        play_hi=int(np.asarray(env.play_action_range[-1])),
        color_lo=int(np.asarray(env.color_action_range[0])),
        color_hi=int(np.asarray(env.color_action_range[-1])),
        rank_lo=int(np.asarray(env.rank_action_range[0])),
        rank_hi=int(np.asarray(env.rank_action_range[-1])),
        # other env constants
        num_agents=int(env.num_agents),
        num_colors=int(env.num_colors),
        num_ranks=int(env.num_ranks),
        hand_size=int(env.hand_size),
        deck_size=int(env.deck_size),
        atoms_arr=jnp.asarray(atoms.atoms_arr),
        atoms_len=jnp.asarray(atoms.atoms_len),
        max_atom_len=int(atoms.atoms_arr.shape[1]),
        idx_card=jnp.asarray(atoms.idx_card),
        idx_card_comma=jnp.asarray(atoms.idx_card_comma),
        idx_color_comma=jnp.asarray(atoms.idx_color_comma),
        idx_rank_comma=jnp.asarray(atoms.idx_rank_comma),
        idx_only_comma=int(atoms.idx_only_comma),
        idx_number=jnp.asarray(atoms.idx_number),
        idx_possible=jnp.asarray(atoms.idx_possible),
        idx_action_seq=jnp.asarray(atoms.idx_action_seq),
        idx_action_len=jnp.asarray(atoms.idx_action_len),
        idx_brackets_empty=int(atoms.idx_brackets_empty),
        idx_bracket_single=jnp.asarray(atoms.idx_bracket_single),
        idx_bracket_open=jnp.asarray(atoms.idx_bracket_open),
        idx_bracket_close=jnp.asarray(atoms.idx_bracket_close),
        idx_turn=int(atoms.idx_turn),
        idx_score=int(atoms.idx_score),
        idx_information=int(atoms.idx_information),
        idx_available=int(atoms.idx_available),
        idx_lives=int(atoms.idx_lives),
        idx_deck=int(atoms.idx_deck),
        idx_remaining=int(atoms.idx_remaining),
        idx_cards_label=int(atoms.idx_cards_label),
        idx_discards=int(atoms.idx_discards),
        idx_fireworks=int(atoms.idx_fireworks),
        idx_your=int(atoms.idx_your),
        idx_other=int(atoms.idx_other),
        idx_hand=int(atoms.idx_hand),
        idx_card_label=int(atoms.idx_card_label),
        idx_card_no_colon=int(atoms.idx_card_no_colon),
        idx_hints=int(atoms.idx_hints),
        idx_possible_label=int(atoms.idx_possible_label),
        idx_last=int(atoms.idx_last),
        idx_action_label=int(atoms.idx_action_label),
        idx_cards_word=int(atoms.idx_cards_word),
        idx_afected=int(atoms.idx_afected),
        idx_played=int(atoms.idx_played),
        idx_scored=int(atoms.idx_scored),
        idx_added=int(atoms.idx_added),
        idx_info=int(atoms.idx_info),
        cls=int(atoms.cls),
        sep=int(atoms.sep),
        pad=int(atoms.pad),
    )


def _assemble_tokens(atom_indices, atom_lens, atoms_arr, atoms_len, max_obs_tokens, cls_id, sep_id, pad_id):
    """JIT-compatible assembly. Pure jnp.

    atom_indices: int32[max_seq_len]  — index into atoms_arr (-1 means skip; we
                  treat any idx < 0 as a no-op via atom_lens=0)
    atom_lens:    int32[max_seq_len]  — length of each atom (0 means skip)
    atoms_arr:    int32[num_atoms, max_atom_len]
    atoms_len:    int32[num_atoms]    — (already encoded in atom_lens)
    """
    max_atom_len = atoms_arr.shape[1]
    pad_arr = jnp.full(max_atom_len, pad_id, dtype=jnp.int32)

    out = jnp.full(max_obs_tokens, pad_id, dtype=jnp.int32)
    out = out.at[0].set(cls_id)

    def body(carry, x):
        out, pos = carry
        atom_idx, atom_len = x
        # Safe atom lookup: atom_idx might be -1 for "skip" — clamp to 0 with len=0.
        safe_idx = jnp.maximum(atom_idx, 0)
        atom_tokens = atoms_arr[safe_idx]  # (max_atom_len,)
        # Use scatter (out.at[idx].set with mode='drop'): write atom_tokens[i] to
        # out[pos+i] only when (i < atom_len) AND (pos+i < max_obs_tokens-1, i.e.
        # leaves room for SEP). For invalid slots, redirect to OOB sentinel so the
        # write is dropped — no overwrite, no overflow.
        rel = jnp.arange(max_atom_len)
        abs_positions = pos + rel
        valid = (rel < atom_len) & (abs_positions < max_obs_tokens - 1)
        sentinel = max_obs_tokens  # out-of-bounds → mode='drop' silently skips
        write_idx = jnp.where(valid, abs_positions, sentinel)
        new_out = out.at[write_idx].set(atom_tokens, mode="drop")
        new_pos = pos + atom_len
        return (new_out, new_pos), None

    (out, pos), _ = jax.lax.scan(body, (out, jnp.int32(1)), (atom_indices, atom_lens))

    # Insert SEP at min(pos, max_obs_tokens - 1).
    sep_pos = jnp.minimum(pos, max_obs_tokens - 1)
    out = out.at[sep_pos].set(sep_id)

    valid_len = jnp.minimum(pos + 1, max_obs_tokens)
    mask = (jnp.arange(max_obs_tokens) < valid_len).astype(jnp.int32)
    return out, mask


def _last_action_atom_seq(env, atoms_jnp, new_state, old_state, action, aidx):
    """Build the (atom_indices, atom_lens) pair for the last-action section.

    Returns two int32 arrays of fixed length L_LAST = 16 (upper bound).
    """
    # action range constants (already Python ints in atoms_jnp dict)
    discard_lo = atoms_jnp["discard_lo"]
    discard_hi = atoms_jnp["discard_hi"]
    play_lo = atoms_jnp["play_lo"]
    play_hi = atoms_jnp["play_hi"]
    color_lo = atoms_jnp["color_lo"]
    color_hi = atoms_jnp["color_hi"]
    rank_lo = atoms_jnp["rank_lo"]
    rank_hi = atoms_jnp["rank_hi"]

    is_discard = (action >= discard_lo) & (action <= discard_hi)
    is_play = (action >= play_lo) & (action <= play_hi)
    is_hint_color = (action >= color_lo) & (action <= color_hi)
    is_hint_rank = (action >= rank_lo) & (action <= rank_hi)
    is_hint = is_hint_color | is_hint_rank
    is_pd = is_discard | is_play

    # First: "Last" "action:" + action sequence (variable length up to 4 words).
    seq_idx = []
    seq_len = []

    seq_idx.append(jnp.int32(atoms_jnp["idx_last"]))
    seq_len.append(atoms_jnp["atoms_len"][atoms_jnp["idx_last"]])
    seq_idx.append(jnp.int32(atoms_jnp["idx_action_label"]))
    seq_len.append(atoms_jnp["atoms_len"][atoms_jnp["idx_action_label"]])

    # action description: up to 4 words (e.g., "HR to P1 relative")
    action_seq = atoms_jnp["idx_action_seq"][action]      # (max_action_words,)
    action_lens = atoms_jnp["atoms_len"][action_seq]       # (max_action_words,)
    action_n = atoms_jnp["idx_action_len"][action]         # scalar
    valid = jnp.arange(action_seq.shape[0]) < action_n
    action_lens = jnp.where(valid, action_lens, 0)
    for i in range(action_seq.shape[0]):
        seq_idx.append(action_seq[i])
        seq_len.append(action_lens[i])

    # Conditional: only emit if turn > 0
    turn_gt0 = new_state.turn > 0

    # If hint: "Cards" + "afected:" + bracket atoms (up to 6: 1 open, up to 4 middle nums, 1 close)
    # If play/discard: "Card" (no colon) + "Played:" + card
    # If play: + "Scored:" + N + "Added" + "Info:" + N

    # We allocate 12 slots for these conditional emissions.
    hand_size = int(env.hand_size)

    # Compute reveal_outcome (for hint actions).
    # acting_player_idx
    acting_idx = jnp.argmax(old_state.cur_player_idx)
    # decode hint action
    hint_color_idx = jnp.where(
        is_hint_color, (action - 2 * hand_size) % env.num_colors, 0
    )
    hint_color_target_rel = jnp.where(
        is_hint_color, (action - 2 * hand_size) // env.num_colors, 0
    )
    hint_rank_idx = jnp.where(
        is_hint_rank,
        (action - 2 * hand_size - (env.num_agents - 1) * env.num_colors) % env.num_ranks,
        0,
    )
    hint_rank_target_rel = jnp.where(
        is_hint_rank,
        (action - 2 * hand_size - (env.num_agents - 1) * env.num_colors) // env.num_ranks,
        0,
    )
    hint_target_rel = jnp.where(is_hint_color, hint_color_target_rel, hint_rank_target_rel)
    hint_target_abs = (acting_idx + 1 + hint_target_rel) % env.num_agents

    target_hand = new_state.player_hands[hint_target_abs]  # (hand_size, num_colors, num_ranks)
    color_match = (target_hand.sum(axis=2)[..., hint_color_idx] > 0).astype(jnp.int32)
    rank_match = (target_hand.sum(axis=1)[..., hint_rank_idx] > 0).astype(jnp.int32)
    reveal_outcome = jnp.where(
        is_hint_color, color_match, jnp.where(is_hint_rank, rank_match, jnp.zeros(hand_size, dtype=jnp.int32))
    )  # (hand_size,)

    # Bracket emission for "Cards afected: <indices>"
    n_revealed = reveal_outcome.sum()
    # Find the indices of reveal_outcome positions. We use argsort by negation.
    # Get sorted-by-mask indices: positions with mask=1 come first.
    # neg_mask = -reveal_outcome (so 1->-1, 0->0). argsort gives stable order: revealed first.
    sorted_idx = jnp.argsort(-reveal_outcome, stable=True)  # (hand_size,)
    # First revealed index, last revealed index, middle positions.

    # We'll emit (in order):
    # - "Cards" idx
    # - "afected:" idx
    # - ONE of: empty bracket "[]", single-bracket "[X]", or open/close pair with middle nums
    # We always emit:
    #   slot 0: "Cards" (always, len 0 if not hint)
    #   slot 1: "afected:"
    #   slot 2: bracket-empty (if 0 revealed)
    #   slot 3: bracket-single[X] (if 1 revealed)
    #   slot 4: bracket-open[X (if 2+ revealed)
    #   slots 5,6,7: middle indices (if 3+, 4+, 5+ revealed) — at most hand_size-2 = 3
    #   slot 8: bracket-close X] (if 2+ revealed)
    #
    # Each slot's len is 0 if condition not met.

    # Helper: select first revealed idx (when 1+ revealed).
    first_rev = sorted_idx[0]
    last_rev = jnp.where(n_revealed > 0, sorted_idx[jnp.maximum(n_revealed - 1, 0)], 0)

    # "Cards" "afected:" — only if hint
    seq_idx.append(jnp.int32(atoms_jnp["idx_cards_word"]))
    seq_len.append(jnp.where(is_hint & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_cards_word"]], 0))
    seq_idx.append(jnp.int32(atoms_jnp["idx_afected"]))
    seq_len.append(jnp.where(is_hint & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_afected"]], 0))

    # Bracket emissions (5 slots: empty, single, open, middle1, middle2, close)
    # empty "[]" if 0 revealed
    seq_idx.append(jnp.int32(atoms_jnp["idx_brackets_empty"]))
    seq_len.append(jnp.where(is_hint & turn_gt0 & (n_revealed == 0), atoms_jnp["atoms_len"][atoms_jnp["idx_brackets_empty"]], 0))
    # "[X]" if exactly 1 revealed
    bracket_single_idx = atoms_jnp["idx_bracket_single"][first_rev]
    seq_idx.append(bracket_single_idx)
    seq_len.append(jnp.where(is_hint & turn_gt0 & (n_revealed == 1), atoms_jnp["atoms_len"][bracket_single_idx], 0))
    # "[X" if 2+ revealed
    bracket_open_idx = atoms_jnp["idx_bracket_open"][first_rev]
    seq_idx.append(bracket_open_idx)
    seq_len.append(jnp.where(is_hint & turn_gt0 & (n_revealed >= 2), atoms_jnp["atoms_len"][bracket_open_idx], 0))
    # Middle indices: positions sorted_idx[1..n_revealed-2]
    # Up to hand_size - 2 = 3 (for hand_size=5) middle positions
    n_middle = jnp.maximum(n_revealed - 2, 0)
    for k in range(int(env.hand_size) - 2):
        # Pick sorted_idx[k+1] as the middle index
        middle_pos = sorted_idx[k + 1]
        middle_atom = atoms_jnp["idx_number"][middle_pos]
        seq_idx.append(middle_atom)
        seq_len.append(jnp.where(is_hint & turn_gt0 & (k < n_middle), atoms_jnp["atoms_len"][middle_atom], 0))
    # "X]" if 2+ revealed
    bracket_close_idx = atoms_jnp["idx_bracket_close"][last_rev]
    seq_idx.append(bracket_close_idx)
    seq_len.append(jnp.where(is_hint & turn_gt0 & (n_revealed >= 2), atoms_jnp["atoms_len"][bracket_close_idx], 0))

    # Card Played: + card  (only if play or discard)
    seq_idx.append(jnp.int32(atoms_jnp["idx_card_no_colon"]))
    seq_len.append(jnp.where(is_pd & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_card_no_colon"]], 0))
    seq_idx.append(jnp.int32(atoms_jnp["idx_played"]))
    seq_len.append(jnp.where(is_pd & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_played"]], 0))
    # Card from acting_player's hand at slot (action % hand_size)
    pd_slot = jnp.where(is_pd, action % hand_size, 0)
    pd_card = old_state.player_hands[acting_idx, pd_slot]  # (num_colors, num_ranks)
    pd_color = jnp.argmax(pd_card.sum(axis=1))
    pd_rank = jnp.argmax(pd_card.sum(axis=0))
    pd_nonempty = pd_card.sum() > 0
    pd_card_idx = atoms_jnp["idx_card"][pd_color, pd_rank]
    seq_idx.append(pd_card_idx)
    seq_len.append(jnp.where(is_pd & turn_gt0 & pd_nonempty, atoms_jnp["atoms_len"][pd_card_idx], 0))

    # If play: Scored: <0|1>, Added Info: <0|1>
    fw_diff = (new_state.fireworks.sum() - old_state.fireworks.sum()) != 0
    info_diff = new_state.info_tokens.sum() > old_state.info_tokens.sum()
    scored = jnp.where(is_play, fw_diff.astype(jnp.int32), 0)
    added = jnp.where(is_play, info_diff.astype(jnp.int32), 0)

    seq_idx.append(jnp.int32(atoms_jnp["idx_scored"]))
    seq_len.append(jnp.where(is_play & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_scored"]], 0))
    scored_atom = atoms_jnp["idx_number"][scored]
    seq_idx.append(scored_atom)
    seq_len.append(jnp.where(is_play & turn_gt0, atoms_jnp["atoms_len"][scored_atom], 0))
    seq_idx.append(jnp.int32(atoms_jnp["idx_added"]))
    seq_len.append(jnp.where(is_play & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_added"]], 0))
    seq_idx.append(jnp.int32(atoms_jnp["idx_info"]))
    seq_len.append(jnp.where(is_play & turn_gt0, atoms_jnp["atoms_len"][atoms_jnp["idx_info"]], 0))
    added_atom = atoms_jnp["idx_number"][added]
    seq_idx.append(added_atom)
    seq_len.append(jnp.where(is_play & turn_gt0, atoms_jnp["atoms_len"][added_atom], 0))

    return seq_idx, seq_len


def _state_to_atom_seq(env, atoms_jnp, new_state, old_state, action, aidx):
    """Build (atom_indices, atom_lens) for one agent's view of one state.

    Returns two int32 jnp arrays of fixed shape (max_seq_len,).
    """
    seq_idx = []   # list of jnp scalars (built at trace time)
    seq_len = []

    A = atoms_jnp  # alias

    def emit(idx_scalar):
        seq_idx.append(idx_scalar)
        seq_len.append(A["atoms_len"][idx_scalar])

    def emit_pair(idx_scalar, len_scalar):
        seq_idx.append(idx_scalar)
        seq_len.append(len_scalar)

    # --- Board info ---
    emit(jnp.int32(A["idx_turn"]))
    emit(A["idx_number"][new_state.turn])
    emit(jnp.int32(A["idx_score"]))
    emit(A["idx_number"][new_state.score])
    emit(jnp.int32(A["idx_information"]))
    emit(jnp.int32(A["idx_available"]))
    info = new_state.info_tokens.sum().astype(jnp.int32)
    emit(A["idx_number"][info])
    emit(jnp.int32(A["idx_lives"]))
    emit(jnp.int32(A["idx_available"]))
    lives = new_state.life_tokens.sum().astype(jnp.int32)
    emit(A["idx_number"][lives])
    emit(jnp.int32(A["idx_deck"]))
    emit(jnp.int32(A["idx_remaining"]))
    emit(jnp.int32(A["idx_cards_label"]))
    deck = new_state.deck.sum().astype(jnp.int32)
    emit(A["idx_number"][deck])

    # --- Discards (deck_size cards, some empty) ---
    emit(jnp.int32(A["idx_discards"]))
    discard_pile = new_state.discard_pile  # (deck_size, num_colors, num_ranks)
    deck_size = int(discard_pile.shape[0])
    discard_color = jnp.argmax(discard_pile.sum(axis=2), axis=1)  # (deck_size,)
    discard_rank = jnp.argmax(discard_pile.sum(axis=1), axis=1)
    discard_nonempty = discard_pile.sum(axis=(1, 2)) > 0
    discard_card_idx = A["idx_card"][discard_color, discard_rank]  # (deck_size,)
    discard_lens_full = A["atoms_len"][discard_card_idx]
    discard_lens = jnp.where(discard_nonempty, discard_lens_full, 0)
    for i in range(deck_size):
        emit_pair(discard_card_idx[i], discard_lens[i])

    # --- Fireworks (5 cards) ---
    emit(jnp.int32(A["idx_fireworks"]))
    fireworks = new_state.fireworks  # (num_colors, num_ranks)
    # Highest played rank per color: argmax of reversed thermometer.
    # row [1,1,1,0,0] → reversed [0,0,1,1,1], argmax=2, rank=size-1-2=2.
    fw_size = int(fireworks.shape[1])
    rev = fireworks[:, ::-1]
    fw_rank = fw_size - 1 - jnp.argmax(rev, axis=1)  # (num_colors,)
    fw_nonempty = fireworks.sum(axis=1) > 0
    for col in range(int(fireworks.shape[0])):
        c_idx = A["idx_card"][col, fw_rank[col]]
        c_len = jnp.where(fw_nonempty[col], A["atoms_len"][c_idx], 0)
        emit_pair(c_idx, c_len)

    # --- Per-agent hand sections ---
    colors_revealed = new_state.colors_revealed  # (num_agents, hand_size, num_colors)
    ranks_revealed = new_state.ranks_revealed    # (num_agents, hand_size, num_ranks)
    knowledge = new_state.card_knowledge.reshape(
        env.num_agents, env.hand_size, env.num_colors, env.num_ranks
    )
    hands = new_state.player_hands  # (num_agents, hand_size, num_colors, num_ranks)
    num_colors = int(env.num_colors)
    num_ranks = int(env.num_ranks)

    for ai in range(env.num_agents):
        is_self_const = (ai == aidx)  # python bool (aidx is concrete static)
        if is_self_const:
            emit(jnp.int32(A["idx_your"]))
        else:
            emit(jnp.int32(A["idx_other"]))
        emit(jnp.int32(A["idx_hand"]))

        for slot in range(int(env.hand_size)):
            # Slot index
            emit(A["idx_number"][slot])

            if not is_self_const:
                # "Card:" + card_with_comma (or "," if empty)
                emit(jnp.int32(A["idx_card_label"]))
                card = hands[ai, slot]
                col_i = jnp.argmax(card.sum(axis=1))
                rk_i = jnp.argmax(card.sum(axis=0))
                nonempty = card.sum() > 0
                card_comma_idx = A["idx_card_comma"][col_i, rk_i]
                emit_pair(
                    jnp.where(nonempty, card_comma_idx, A["idx_only_comma"]),
                    jnp.where(
                        nonempty,
                        A["atoms_len"][card_comma_idx],
                        A["atoms_len"][A["idx_only_comma"]],
                    ),
                )

            # "Hints:" + hint_with_comma
            emit(jnp.int32(A["idx_hints"]))
            ch = colors_revealed[ai, slot]
            rh = ranks_revealed[ai, slot]
            has_c = ch.any()
            has_r = rh.any()
            ch_idx = jnp.argmax(ch)
            rh_idx = jnp.argmax(rh)
            both_idx = A["idx_card_comma"][ch_idx, rh_idx]
            color_only_idx = A["idx_color_comma"][ch_idx]
            rank_only_idx = A["idx_rank_comma"][rh_idx]
            none_idx = A["idx_only_comma"]
            hint_idx_choice = jnp.where(
                has_c & has_r, both_idx,
                jnp.where(has_c, color_only_idx,
                          jnp.where(has_r, rank_only_idx, none_idx))
            )
            emit_pair(hint_idx_choice, A["atoms_len"][hint_idx_choice])

            # "Possible:" + subset
            emit(jnp.int32(A["idx_possible_label"]))
            kn = knowledge[ai, slot]  # (num_colors, num_ranks)
            color_known = kn.any(axis=1)  # (num_colors,)
            rank_known = kn.any(axis=0)   # (num_ranks,)
            cm = jnp.sum(color_known.astype(jnp.int32) * (1 << jnp.arange(num_colors)))
            rm = jnp.sum(rank_known.astype(jnp.int32) * (1 << jnp.arange(num_ranks)))
            possible_idx = A["idx_possible"][cm, rm]
            # idx_possible can be -1 if cm==0 and rm==0 (empty subset). Use len=0 in that case.
            possible_len = jnp.where(possible_idx >= 0, A["atoms_len"][jnp.maximum(possible_idx, 0)], 0)
            emit_pair(jnp.maximum(possible_idx, 0), possible_len)

    # --- Last action (with conditional sub-sections) ---
    last_idx, last_len = _last_action_atom_seq(env, A, new_state, old_state, action, aidx)
    seq_idx.extend(last_idx)
    seq_len.extend(last_len)

    # Stack into fixed-shape arrays
    atom_indices = jnp.stack(seq_idx)
    atom_lens = jnp.stack(seq_len)
    return atom_indices.astype(jnp.int32), atom_lens.astype(jnp.int32)


def make_tokenize_fn_jax(env, tokenizer, max_obs_tokens: int = 256, include_belief: bool = False) -> Callable:
    """Pure-JAX drop-in replacement for make_tokenize_fn.

    Returns f(new_state, old_state, last_actions) -> (input_ids, attention_mask)
    runnable inside jit with no pure_callback.
    """
    if include_belief:
        raise NotImplementedError("include_belief=True not supported in JAX path")

    atoms = HanabiAtoms(env, tokenizer)
    A = _make_jnp_atoms(atoms)
    num_agents = int(env.num_agents)

    def tokenize_one(new_state, old_state, action, aidx):
        atom_indices, atom_lens = _state_to_atom_seq(env, A, new_state, old_state, action, aidx)
        ids, mask = _assemble_tokens(
            atom_indices, atom_lens,
            A["atoms_arr"], A["atoms_len"],
            max_obs_tokens, A["cls"], A["sep"], A["pad"],
        )
        return ids, mask

    def fn(new_state, old_state, last_actions):
        # new_state, old_state: pytree with leading B dim per leaf
        # last_actions: int32[B]
        # Return: (num_agents, B, max_obs_tokens) ids and mask
        per_a_ids = []
        per_a_mask = []
        for ai in range(num_agents):
            ids_b, mask_b = jax.vmap(tokenize_one, in_axes=(0, 0, 0, None))(
                new_state, old_state, last_actions, ai
            )
            per_a_ids.append(ids_b)
            per_a_mask.append(mask_b)
        ids = jnp.stack(per_a_ids, axis=0)   # (num_agents, B, max_obs_tokens)
        mask = jnp.stack(per_a_mask, axis=0)
        return ids, mask

    return fn
