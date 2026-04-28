"""
Train the auto-regressive belief model (`obl_belief_model.ARBeliefModel`)
on rollouts from a fixed blueprint (BP) policy, mirroring
`pyhanabi/train_belief.py` from the upstream R3D2 repo.

The pipeline:
  1. Load a BP policy checkpoint (R2D2 publ-LSTM or random for cheap dev runs).
  2. Roll out NUM_STEPS x NUM_ENVS trajectories with the BP.
  3. Stash (priv_s, dones, own_hand) tuples in a flashbax trajectory buffer.
  4. Sample minibatches and minimise the AR cross-entropy loss.

Belief targets: at each timestep the agent's `own_hand` is the ground-truth
card matrix (num_agents, hand_size, num_colors, num_ranks). For each viewer
agent we serialise their own hand into one-hot 25-class vectors per slot
(empty slot -> all zeros), then shift right by one to get the AR input.

This file produces a `safetensors` checkpoint for `obl_rnn_hanabi.py`.
"""

from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager

from obl_belief_model import ARBeliefModel, loss_fn, make_ar_input
from r2d2_publ_rnn_hanabi import (
    PublicLSTMQNetwork, ScannedLSTM as PublScannedLSTM,
    hanabi_publ_split, hanabi_feature_widths, reorder_obs_for_split,
)


# ---------------------------------------------------------------------------
# Replay structure
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class BeliefTimestep:
    priv_s: jnp.ndarray   # per-agent reordered obs (num_agents, B, in_dim)
    dones: jnp.ndarray    # per-agent done (num_agents, B)
    own_hand: jnp.ndarray # per-agent one-hot ground-truth hand (num_agents, B, hand_size, 25)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_own_hand(player_hands_per_agent: jnp.ndarray, num_colors: int, num_ranks: int):
    """player_hands shape (num_agents, hand_size, num_colors, num_ranks)
    -> (num_agents, hand_size, num_colors*num_ranks) one-hot.

    Empty slots stay all-zero.
    """
    return player_hands_per_agent.reshape(
        *player_hands_per_agent.shape[:-2], num_colors * num_ranks
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def make_train(config, env, bp_policy=None):
    NUM_ENVS = int(config["NUM_ENVS"])
    NUM_STEPS = int(config["NUM_STEPS"])
    BUFFER_SIZE = int(config["BUFFER_SIZE"])
    BATCH_SIZE = int(config["BUFFER_BATCH_SIZE"])
    TOTAL = int(config["TOTAL_TIMESTEPS"])
    HID = int(config["HIDDEN_SIZE"])
    LR = float(config["LR"])

    widths = hanabi_feature_widths(env)
    in_dim = sum(widths.values())
    num_colors = env.num_colors
    num_ranks = env.num_ranks
    hand_size = env.hand_size
    num_agents = env.num_agents

    belief = ARBeliefModel(
        in_dim=in_dim,
        hid_dim=HID,
        hand_size=hand_size,
        out_dim=num_colors * num_ranks,
    )

    def train(rng):
        wrapped_env = CTRolloutManager(env, batch_size=NUM_ENVS)

        # ---- init params ----
        rng, _rng = jax.random.split(rng)
        init_priv = jnp.zeros((1, 1, in_dim))
        init_dones = jnp.zeros((1, 1))
        init_ar = jnp.zeros((1, 1, hand_size, num_colors * num_ranks))
        params = belief.init(_rng, init_priv, init_dones, init_ar)

        tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 5.0)),
            optax.adam(learning_rate=LR, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=belief.apply, params=params, tx=tx)

        # ---- rollout (with BP policy, or random) ----
        def _rollout_step(carry, _):
            env_state, last_dones, rng = carry
            rng, ka, ks = jax.random.split(rng, 3)
            ka = jax.random.split(ka, num_agents)
            # BP policy: random for now (acts as Identity-blueprint OBL_0).
            # If you have a BP checkpoint, plug its `act` here.
            actions = {a: wrapped_env.batch_sample(ka[i], a)
                       for i, a in enumerate(env.agents)}
            avail = wrapped_env.get_valid_actions(env_state.env_state)  # noqa: F841
            obs, new_env_state, rewards, dones, info = wrapped_env.batch_step(
                ks, env_state, actions
            )

            # priv_s per agent: re-order CTRolloutManager-padded obs.
            priv_s = jnp.stack([
                reorder_obs_for_split(obs[a], widths) for a in env.agents
            ], axis=0)  # (num_agents, NUM_ENVS, in_dim)

            # own_hand per agent: ground truth, from env's State.
            own = encode_own_hand(
                new_env_state.env_state.player_hands, num_colors, num_ranks
            )  # (num_agents, hand_size, 25)
            # Need (num_agents, NUM_ENVS, hand_size, 25): the leading num_agents
            # axis is already there but NUM_ENVS comes from the vmap — reorder:
            own = jnp.swapaxes(own, 0, 1)  # vmap dim is 0 in leaf -> (NUM_ENVS, num_agents, ...)
            own = jnp.transpose(own, (1, 0, 2, 3))  # back to (num_agents, NUM_ENVS, hand_size, 25)

            done_per_agent = jnp.stack(
                [dones[a] for a in env.agents], axis=0
            )  # (num_agents, NUM_ENVS)

            ts = BeliefTimestep(priv_s=priv_s, dones=done_per_agent.astype(jnp.float32),
                                own_hand=own)
            return (new_env_state, dones, rng), ts

        # ---- buffer ----
        rng, _rng = jax.random.split(rng)
        _, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {a: jnp.zeros((NUM_ENVS,), dtype=bool)
                      for a in env.agents + ["__all__"]}
        rng, _rng = jax.random.split(rng)
        _, sample_traj = jax.lax.scan(
            _rollout_step, (env_state, init_dones, _rng), None, NUM_STEPS
        )
        # sample_traj leaves shape: (T, num_agents, NUM_ENVS, ...). Buffer wants
        # leading axis = ADD_BATCH (across NUM_ENVS) and inner time axis. We'll
        # flatten num_agents into NUM_ENVS for storage — same network, just more rows.
        def flatten_agent_axis(x):
            T = x.shape[0]
            return x.reshape(T, num_agents * NUM_ENVS, *x.shape[3:])

        sample_flat = jax.tree.map(flatten_agent_axis, sample_traj)
        sample_unbatched = jax.tree.map(lambda x: x[:, 0], sample_flat)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=BUFFER_SIZE // (NUM_ENVS * num_agents),
            min_length_time_axis=BATCH_SIZE,
            sample_batch_size=BATCH_SIZE,
            add_batch_size=NUM_ENVS * num_agents,
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_unbatched)

        def update_step(carry, _):
            train_state, buffer_state, env_state, last_dones, rng = carry
            rng, _rng = jax.random.split(rng)
            (env_state, last_dones, _rng2), traj = jax.lax.scan(
                _rollout_step, (env_state, last_dones, _rng), None, NUM_STEPS
            )
            # Flatten agent axis into add_batch_size.
            traj_flat = jax.tree.map(flatten_agent_axis, traj)
            traj_buf = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], traj_flat
            )
            buffer_state = buffer.add(buffer_state, traj_buf)

            def _learn(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )

                priv_s = minibatch.priv_s          # (T, B, in_dim)
                dones = minibatch.dones            # (T, B)
                own = minibatch.own_hand           # (T, B, hand_size, 25)
                ar_in = make_ar_input(own)         # (T, B, hand_size, 25)

                def _loss(params):
                    logits = train_state.apply_fn(params, priv_s, dones, ar_in)
                    mask = (1.0 - dones)
                    return loss_fn(logits, own, mask)

                loss, grads = jax.value_and_grad(_loss)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, rng), loss

            rng, _rng = jax.random.split(rng)
            is_learn = buffer.can_sample(buffer_state)
            (train_state, rng), losses = jax.lax.cond(
                is_learn,
                lambda ts, r: jax.lax.scan(_learn, (ts, r), None, config.get("NUM_EPOCHS", 1)),
                lambda ts, r: ((ts, r), jnp.zeros(config.get("NUM_EPOCHS", 1))),
                train_state, _rng,
            )
            return (train_state, buffer_state, env_state, last_dones, rng), losses.mean()

        n_updates = TOTAL // NUM_STEPS // NUM_ENVS
        rng, _rng = jax.random.split(rng)
        carry = (train_state, buffer_state, env_state, init_dones, _rng)
        carry, losses = jax.lax.scan(update_step, carry, None, n_updates)
        return {"train_state": carry[0], "losses": losses}

    return train, belief


def env_from_config(config):
    e = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    return LogWrapper(e), f"{config['ENV_NAME']}_{config['ENV_KWARGS'].get('num_agents', 2)}p"


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))
    env, env_name = env_from_config(copy.deepcopy(config))
    train_fn, belief = make_train(config, env)

    wandb.init(
        entity=config["ENTITY"], project=config["PROJECT"],
        tags=["OBL", "BELIEF", env_name.upper()],
        name=f"belief_{env_name}",
        config=config, mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    out = jax.jit(train_fn)(rng)
    print("final loss:", float(out["losses"][-1]))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(
            out["train_state"].params,
            os.path.join(save_dir, f"belief_seed{config['SEED']}.safetensors"),
        )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    single_run(config)


if __name__ == "__main__":
    main()
