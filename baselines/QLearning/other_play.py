"""
Other-Play (OP) for Hanabi: per-episode random color permutation.

Mirrors the ``shuffle_color`` flag in the original R3D2 C++ actor
(``r2d2_actor.cc``).  At the start of each episode a random permutation
of the ``num_colors`` color indices is drawn.  Observations are
presented with permuted colors, and color-hint actions are inverse-
permuted before being applied to the real game.

The wrapper works on the flat observation vector produced by
``HanabiEnv.get_obs()`` and on integer action indices.  It is designed
for use with ``CTRolloutManager.batch_step / batch_reset``.

Usage (inside the training loop)::

    from other_play import OtherPlayState, op_init, op_permute_obs, op_unpermute_action, op_resample

    # After batch_reset
    op_state = op_init(rng, num_envs, env)

    # Each step
    obs = op_permute_obs(obs_dict, op_state, env)
    # ... compute actions from permuted obs ...
    actions = op_unpermute_action(actions_dict, op_state, env)
    # ... env.batch_step(actions) ...

    # After step, resample permutation for envs that reset
    op_state = op_resample(op_state, dones["__all__"], rng)
"""

from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class OtherPlayState:
    """Per-env color permutation state."""
    color_perm: chex.Array     # (B, num_colors) — forward permutation
    inv_color_perm: chex.Array # (B, num_colors) — inverse permutation


def _inv_perm(perm):
    """Compute inverse permutation."""
    inv = jnp.empty_like(perm)
    return inv.at[perm].set(jnp.arange(perm.shape[0]))


def op_init(rng: chex.PRNGKey, batch_size: int, env) -> OtherPlayState:
    """Sample an independent color permutation for each env in the batch."""
    nc = env.num_colors
    rngs = jax.random.split(rng, batch_size)
    perms = jax.vmap(lambda r: jax.random.permutation(r, nc))(rngs)
    inv_perms = jax.vmap(_inv_perm)(perms)
    return OtherPlayState(color_perm=perms, inv_color_perm=inv_perms)


def op_resample(op_state: OtherPlayState, dones: chex.Array, rng: chex.PRNGKey) -> OtherPlayState:
    """Resample permutation for environments that just reset.

    Args:
        dones: bool array (B,) — True for envs that reset this step.
    """
    B = dones.shape[0]
    nc = op_state.color_perm.shape[1]
    rngs = jax.random.split(rng, B)
    new_perms = jax.vmap(lambda r: jax.random.permutation(r, nc))(rngs)
    new_inv = jax.vmap(_inv_perm)(new_perms)
    perm = jnp.where(dones[:, None], new_perms, op_state.color_perm)
    inv = jnp.where(dones[:, None], new_inv, op_state.inv_color_perm)
    return OtherPlayState(color_perm=perm, inv_color_perm=inv)


# ---- observation permutation ----

def _permute_cards(flat_cards, perm, num_colors, num_ranks):
    """Permute colors in a block of one-hot card features.

    flat_cards: (..., num_cards * num_colors * num_ranks)
    Returns same shape with color axis permuted.
    """
    leading = flat_cards.shape[:-1]
    n = flat_cards.shape[-1]
    num_cards = n // (num_colors * num_ranks)
    cards = flat_cards.reshape(*leading, num_cards, num_colors, num_ranks)
    cards = cards[..., perm, :]  # permute color axis
    return cards.reshape(*leading, n)


def _permute_color_vec(vec, perm):
    """Permute a num_colors-length vector."""
    return vec[..., perm]


def op_permute_obs(obs_dict: dict, op_state: OtherPlayState, env) -> dict:
    """Permute color-dependent features in the observation dict.

    obs_dict: {agent_name: (B, obs_size)} as returned by CTRolloutManager.
    op_state: OtherPlayState with (B, num_colors) permutations.
    """
    nc = env.num_colors
    nr = env.num_ranks
    hs = env.hand_size
    na = env.num_agents

    # Feature index offsets (same structure as HanabiEnv.get_obs)
    hands_end = env.hands_n_feats
    board_start = hands_end
    board_end = board_start + env.board_n_feats
    discard_start = board_end
    discard_end = discard_start + env.discards_n_feats
    last_act_start = discard_end
    last_act_end = last_act_start + env.last_action_n_feats
    belief_start = last_act_end

    # Compute per-env permutation indices
    perm = op_state.color_perm  # (B, nc)

    def permute_single(obs, perm_i):
        # ---- HANDS: other players' cards ----
        n_other_cards = (na - 1) * hs
        card_feats = obs[:n_other_cards * nc * nr]
        card_feats = _permute_cards(card_feats, perm_i, nc, nr)

        # ---- BOARD: fireworks ----
        deck_size = env.deck_size - na * hs
        fw_start = board_start + deck_size
        fw_end = fw_start + nc * nr
        fireworks = obs[fw_start:fw_end]
        fireworks = _permute_cards(fireworks, perm_i, nc, nr)

        # ---- DISCARD PILE ----
        # (nc, num_cards_of_rank.sum()) — permute along color axis
        n_per_color = int(env.num_cards_of_rank.sum())
        discard = obs[discard_start:discard_end].reshape(nc, n_per_color)
        discard = discard[perm_i]
        discard = discard.ravel()

        # ---- LAST ACTION: color_revealed + played/discarded card ----
        # Offsets within last_action_feats:
        # acting_player_relative: na
        # move_type: 4
        # target_player_relative: na
        # color_revealed: nc
        # rank_revealed: nr
        # reveal_outcome: hs
        # pos_played_discarded: hs
        # played_discarded_card: nc*nr
        # card_played_score: 1
        # added_info_tokens: 1
        la_offset = last_act_start
        la_cr_start = la_offset + na + 4 + na  # color_revealed start
        color_revealed = obs[la_cr_start:la_cr_start + nc]
        color_revealed = _permute_color_vec(color_revealed, perm_i)

        la_pdc_start = la_cr_start + nc + nr + hs + hs  # played_discarded_card start
        pdc = obs[la_pdc_start:la_pdc_start + nc * nr]
        pdc = _permute_cards(pdc, perm_i, nc, nr)

        # ---- V0 BELIEF ----
        # (na, hs, 35) where 35 = nc*nr + nc + nr
        belief = obs[belief_start:].reshape(na, hs, nc * nr + nc + nr)
        # permute deductions (first nc*nr)
        deductions = belief[..., :nc * nr].reshape(na, hs, nc, nr)
        deductions = deductions[:, :, perm_i, :]
        deductions = deductions.reshape(na, hs, nc * nr)
        # permute color hints (next nc)
        color_hints = belief[..., nc * nr:nc * nr + nc]
        color_hints = color_hints[:, :, perm_i]
        # rank hints stay (last nr)
        rank_hints = belief[..., nc * nr + nc:]
        belief = jnp.concatenate([deductions, color_hints, rank_hints], axis=-1).ravel()

        # Reassemble observation
        out = jnp.concatenate([
            card_feats,
            obs[n_other_cards * nc * nr:board_start],  # missing_cards
            obs[board_start:fw_start],                 # deck
            fireworks,
            obs[fw_end:discard_start],                 # info+life tokens
            discard,
            obs[last_act_start:la_cr_start],           # acting/move/target
            color_revealed,
            obs[la_cr_start + nc:la_pdc_start],        # rank/reveal/pos
            pdc,
            obs[la_pdc_start + nc * nr:belief_start],  # score/info
            belief,
        ])
        return out

    result = {}
    for agent_name, obs in obs_dict.items():
        result[agent_name] = jax.vmap(permute_single)(obs, perm)
    return result


def op_unpermute_action(actions_dict: dict, op_state: OtherPlayState, env) -> dict:
    """Inverse-permute color hint actions before applying to the game.

    Only color-hint actions are affected; play, discard, rank-hint, noop
    pass through unchanged.
    """
    nc = env.num_colors
    hs = env.hand_size
    color_lo = int(env.color_action_range[0])
    color_hi = int(env.color_action_range[-1])
    inv_perm = op_state.inv_color_perm  # (B, nc)

    def unpermute_single(action, inv_p):
        is_color_hint = (action >= color_lo) & (action <= color_hi)
        # decode: action_idx = action - 2*hs
        action_idx = action - 2 * hs
        color_idx = action_idx % nc
        target_player = action_idx // nc
        new_color = inv_p[color_idx]
        new_action = 2 * hs + target_player * nc + new_color
        return jnp.where(is_color_hint, new_action, action)

    result = {}
    for agent_name, action in actions_dict.items():
        result[agent_name] = jax.vmap(unpermute_single)(action, inv_perm)
    return result
