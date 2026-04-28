"""
OBL (Off-Belief Learning) on Hanabi.

Single-step OBL approximation. Per-transition target is computed by:

    1) sampling a fictitious own-hand from the belief model conditioned on the
       acting player's obs at time t,
    2) replacing the acting player's hand in the post-step env state with that
       sample,
    3) re-running env.get_obs to get fictitious obs for every agent at t+1, and
    4) running the frozen BP policy on those fictitious obs to get Q values.

The standard Q-learning loss is then:

    target[t] = r[t] + gamma * (1 - done[t]) * max_a Q_BP(o^fict_{t+1}, a)
    loss      = (Q_online[t, a_t] - stop_grad(target[t]))^2

Compared to "full OBL" (Hu et al. 2021), this version:
  * Replaces only the post-step state's hand (not the entire trajectory).
  * Samples one fictitious hand per timestep (NUM_BELIEF_SAMPLES=1 default).
  * Uses a fixed BP for the whole training run (no iterative OBL_k -> OBL_{k+1}).

Run prerequisites:
  1) Train R2D2 publ-LSTM via r2d2_publ_rnn_hanabi.py and save its
     safetensors checkpoint to BP_CHECKPOINT (or set BP_CHECKPOINT=None to
     use random init as the BP — useful for smoke tests).
  2) Train belief model via obl_train_belief.py and save its safetensors to
     BELIEF_CHECKPOINT (None -> random belief = uniform sampling).
"""

from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, Dict, Optional

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager

from r2d2_publ_rnn_hanabi import (
    PublicLSTMQNetwork, ScannedLSTM,
    hanabi_publ_split, hanabi_feature_widths, reorder_obs_for_split,
    CustomTrainState,
)
from obl_belief_model import ARBeliefModel, make_ar_input  # noqa: F401


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class OBLTimestep:
    obs: dict           # per-agent reordered obs (jnp at time t, shape (B, in_dim))
    target_obs: dict    # per-agent reordered fictitious obs at t+1 (post own-hand sample)
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


# ---------------------------------------------------------------------------
# Fictitious state: replace own hand
# ---------------------------------------------------------------------------


def _ints_to_card_matrices(idx, num_colors, num_ranks, orig_hand):
    """idx: (hand_size,) ints. orig_hand: (hand_size, C, R) used to mask empty slots."""
    color = idx // num_ranks
    rank = idx % num_ranks
    one_hot = (
        jax.nn.one_hot(color, num_colors)[..., None]
        * jax.nn.one_hot(rank, num_ranks)[..., None, :]
    )  # (hand_size, C, R)
    empty = orig_hand.sum(axis=(-2, -1)) == 0
    return jnp.where(empty[:, None, None], jnp.zeros_like(one_hot), one_hot)


def replace_own_hand_per_env(state, agent_idx_int, fict_hand_ints, num_colors, num_ranks):
    """state: single Hanabi State (no batch dim).
    agent_idx_int: scalar int.
    fict_hand_ints: (hand_size,) int32.
    """
    new_card = _ints_to_card_matrices(
        fict_hand_ints, num_colors, num_ranks, state.player_hands[agent_idx_int]
    )
    new_hands = state.player_hands.at[agent_idx_int].set(new_card)
    return state.replace(player_hands=new_hands)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def make_train(config, env, bp_params: Optional[Any], belief_params: Optional[Any], hidden_dim: int):
    NUM_ENVS = int(config["NUM_ENVS"])
    NUM_STEPS = int(config["NUM_STEPS"])
    BUFFER_SIZE = int(config["BUFFER_SIZE"])
    BATCH = int(config["BUFFER_BATCH_SIZE"])
    TOTAL = int(config["TOTAL_TIMESTEPS"])
    HID = int(hidden_dim)
    LR = float(config["LR"])
    gamma = float(config["GAMMA"])
    eps_start = float(config["EPS_START"])
    eps_finish = float(config["EPS_FINISH"])
    eps_decay = float(config["EPS_DECAY"])
    rew_scale = float(config.get("REW_SCALE", 1.0))
    learning_starts = int(config["LEARNING_STARTS"])
    target_update = int(config["TARGET_UPDATE_INTERVAL"])
    tau = float(config["TAU"])
    num_belief_samples = int(config.get("NUM_BELIEF_SAMPLES", 1))
    assert num_belief_samples == 1, "Only NUM_BELIEF_SAMPLES=1 supported in this stub."

    widths = hanabi_feature_widths(env)
    in_dim = sum(widths.values())
    priv_dim, _ = hanabi_publ_split(env)
    num_colors = env.num_colors
    num_ranks = env.num_ranks
    hand_size = env.hand_size
    num_agents = env.num_agents

    # ---- networks ----
    online_net = PublicLSTMQNetwork(
        action_dim=CTRolloutManager(env, batch_size=1).max_action_space,
        hidden_dim=HID, publ_split=priv_dim,
    )
    bp_net = PublicLSTMQNetwork(
        action_dim=CTRolloutManager(env, batch_size=1).max_action_space,
        hidden_dim=HID, publ_split=priv_dim,
    )
    belief = ARBeliefModel(
        in_dim=in_dim, hid_dim=HID, hand_size=hand_size,
        out_dim=num_colors * num_ranks,
    )

    NUM_UPDATES = TOTAL // NUM_STEPS // NUM_ENVS
    eps_scheduler = optax.linear_schedule(
        init_value=eps_start, end_value=eps_finish,
        transition_steps=eps_decay * NUM_UPDATES,
    )

    def get_greedy(q, avail):
        return jnp.argmax(q - (1 - avail) * 1e10, axis=-1)

    def eps_greedy(rng, q, eps, avail):
        ra, re = jax.random.split(rng)
        greedy = get_greedy(q, avail)
        probs = avail / jnp.clip(avail.sum(axis=-1, keepdims=True), a_min=1.0)
        rngs_ = jax.random.split(ra, avail.shape[0])
        rand = jax.vmap(lambda r, p: jax.random.choice(r, jnp.arange(p.shape[-1]), p=p))(
            rngs_, probs
        )
        return jnp.where(jax.random.uniform(re, greedy.shape) < eps, rand, greedy)

    def batchify(d):
        return jnp.stack([d[a] for a in env.agents], axis=0)

    def unbatchify(x):
        return {a: x[i] for i, a in enumerate(env.agents)}

    def reorder_obs_dict(obs):
        out = {a: reorder_obs_for_split(obs[a], widths) for a in env.agents}
        if "__all__" in obs:
            out["__all__"] = obs["__all__"]
        return out

    def sample_fict_hand(rng, priv_s_t):
        """Sample a fictitious own-hand from the belief model.
        priv_s_t: (B, in_dim).
        Returns (B, hand_size) int32 in [0, num_colors*num_ranks).
        """
        if belief_params is None:
            # Uniform-random belief (smoke / no belief model loaded).
            return jax.random.randint(
                rng, (priv_s_t.shape[0], hand_size),
                minval=0, maxval=num_colors * num_ranks,
            )
        priv = priv_s_t[None, :]                # (T=1, B, in_dim)
        dones = jnp.zeros((1, priv_s_t.shape[0]))
        zero_card = jnp.zeros((1, priv_s_t.shape[0], hand_size, num_colors * num_ranks))
        # Note: this samples slots independently from marginal logits — true AR
        # sampling would require an apply with mutable AR state. Acceptable for
        # 1-step OBL approx.
        logits = belief.apply(belief_params, priv, dones, zero_card)
        rng_slots = jax.random.split(rng, hand_size)
        samples = []
        for s in range(hand_size):
            samples.append(jax.random.categorical(rng_slots[s], logits[0, :, s, :]))
        return jnp.stack(samples, axis=-1)

    def make_target_obs(env_state_t1, env_state_t, action_t, rng_b, priv_s_cur):
        """Return per-agent reordered fictitious obs at t+1.

        env_state_t1, env_state_t : LogEnvState (LogWrapper) — vmap'd batches
        action_t                   : (B,) int32
        priv_s_cur                 : (B, in_dim) cur-player priv obs at time t
        """
        cur_player = jnp.argmax(env_state_t.env_state.cur_player_idx, axis=-1)  # (B,)
        fict_hands = sample_fict_hand(rng_b, priv_s_cur)  # (B, hand_size)

        def per_env(state_t1_inner, state_t_inner, cp, fict, act):
            new_state = replace_own_hand_per_env(
                state_t1_inner, cp, fict, num_colors, num_ranks
            )
            return env.get_obs(new_state, state_t_inner, act)

        obs_dict = jax.vmap(per_env)(
            env_state_t1.env_state, env_state_t.env_state, cur_player, fict_hands, action_t
        )

        # Mimic CTRolloutManager._preprocess_obs: append agent-id one-hot, then reorder.
        agents_oh = jnp.eye(num_agents)
        out = {}
        for i, a in enumerate(env.agents):
            o = obs_dict[a]
            o = jnp.concatenate(
                [o, jnp.broadcast_to(agents_oh[i], (o.shape[0], num_agents))], axis=-1,
            )
            out[a] = reorder_obs_for_split(o, widths)
        return out

    def train(rng):
        wrapped_env = CTRolloutManager(env, batch_size=NUM_ENVS)

        # ---- params (init online from BP if compatible, else fresh) ----
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, 1, in_dim)), jnp.zeros((1, 1)))
        init_hs = ScannedLSTM.initialize_carry(HID, 1)
        params = online_net.init(_rng, init_hs, *init_x)
        if bp_params is not None:
            try:
                jax.tree.map(lambda a, b: a + 0 * b, params, bp_params)
                params = bp_params
                print("[obl] online net warm-started from BP checkpoint.")
            except Exception:
                print("[obl] BP shape mismatch — keeping random init for online.")
        bp_params_used = bp_params if bp_params is not None else params

        tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 5.0)),
            optax.adam(learning_rate=LR, eps=config.get("ADAM_EPS", 1.5e-5)),
        )
        train_state = CustomTrainState.create(
            apply_fn=online_net.apply, params=params, target_network_params=params, tx=tx,
        )

        # ---- rollout step ----
        def _step_env(carry, _):
            hs, last_obs, last_dones, env_state, prev_state, last_actions, rng = carry
            rng, ra, rs, rb = jax.random.split(rng, 4)

            _obs = batchify(last_obs)[:, np.newaxis]  # (n_agents, 1, B, in_dim)
            _dones = batchify(last_dones)[:, np.newaxis]
            new_hs, q_vals = jax.vmap(online_net.apply, in_axes=(None, 0, 0, 0))(
                train_state.params, hs, _obs, _dones,
            )
            q_vals = q_vals.squeeze(axis=1)
            avail = wrapped_env.get_valid_actions(env_state.env_state)
            eps = eps_scheduler(train_state.n_updates)
            _rngs = jax.random.split(ra, num_agents)
            actions = jax.vmap(eps_greedy, in_axes=(0, 0, None, 0))(
                _rngs, q_vals, eps, batchify(avail)
            )
            actions = unbatchify(actions)
            cur_player_action = sum(
                actions[a] * env_state.env_state.cur_player_idx[..., i].astype(jnp.int32)
                for i, a in enumerate(env.agents)
            )
            obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                rs, env_state, actions
            )
            obs = reorder_obs_dict(obs)

            # Fictitious target obs.
            cur_idx = jnp.argmax(env_state.env_state.cur_player_idx, axis=-1)  # (B,)
            priv_per_agent = batchify(last_obs)[..., :priv_dim]  # (n_agents, B, priv_dim)
            priv_cur = jax.vmap(lambda p, i: p[:, i, :], in_axes=(0, 0))(  # noqa: E501
                priv_per_agent.transpose(1, 0, 2)[None], cur_idx[None],
            )[0]  # (B, priv_dim)
            # Actually we want priv obs of cur player at t -> use last_obs reordered.
            priv_cur = jax.vmap(lambda obs_, i: obs_[i])(
                jnp.stack([last_obs[a] for a in env.agents], axis=1),  # (B, n_agents, in_dim)
                cur_idx,
            )

            target_obs = make_target_obs(new_env_state, env_state, cur_player_action, rb, priv_cur)

            ts = OBLTimestep(
                obs=last_obs,
                target_obs=target_obs,
                actions=actions,
                rewards=jax.tree.map(lambda x: rew_scale * x, rewards),
                dones=last_dones,
                avail_actions=avail,
            )
            return (
                new_hs, obs, dones, new_env_state, env_state, cur_player_action, rng,
            ), (ts, infos)

        def _initial_state(rng):
            rng, k = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(k)
            init_obs = reorder_obs_dict(init_obs)
            init_dones = {a: jnp.zeros((NUM_ENVS,), dtype=bool)
                          for a in env.agents + ["__all__"]}
            init_actions = jnp.full((NUM_ENVS,), env.num_moves - 1, dtype=jnp.int32)
            init_hs = ScannedLSTM.initialize_carry(HID, num_agents, NUM_ENVS)
            return rng, init_hs, init_obs, init_dones, env_state, init_actions

        # Build a sample trajectory to init flashbax buffer.
        rng, init_hs, init_obs, init_dones, env_state, init_actions = _initial_state(rng)
        rng, _rng = jax.random.split(rng)
        carry0 = (init_hs, init_obs, init_dones, env_state, env_state, init_actions, _rng)
        _, (sample_traj, _) = jax.lax.scan(_step_env, carry0, None, NUM_STEPS)

        sample_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=BUFFER_SIZE // NUM_ENVS,
            min_length_time_axis=BATCH,
            sample_batch_size=BATCH,
            add_batch_size=NUM_ENVS,
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_unbatched)

        # ---- update step ----
        def _update_step(runner_state, unused):
            train_state, buffer_state, rng = runner_state
            # Re-roll the env each update (matches r2d2_publ_rnn_hanabi.py).
            rng, k2 = jax.random.split(rng)
            init_obs_, init_env_state = wrapped_env.batch_reset(k2)
            init_obs_ = reorder_obs_dict(init_obs_)
            init_dones_ = {a: jnp.zeros((NUM_ENVS,), dtype=bool)
                           for a in env.agents + ["__all__"]}
            init_actions_ = jnp.full((NUM_ENVS,), env.num_moves - 1, dtype=jnp.int32)
            init_hs_ = ScannedLSTM.initialize_carry(HID, num_agents, NUM_ENVS)
            rng, _rng = jax.random.split(rng)

            (final_carry, (timesteps, infos)) = jax.lax.scan(
                _step_env,
                (init_hs_, init_obs_, init_dones_, init_env_state, init_env_state, init_actions_, _rng),
                None, NUM_STEPS,
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + NUM_STEPS * NUM_ENVS
            )
            buffer_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj)

            # ---- learn ----
            def _learn(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )

                init_hs_l = ScannedLSTM.initialize_carry(HID, num_agents, BATCH)
                _obs = batchify(minibatch.obs)
                _target_obs = batchify(minibatch.target_obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail = batchify(minibatch.avail_actions)

                # BP forward on fictitious next-obs.
                _, q_bp_target = jax.vmap(bp_net.apply, in_axes=(None, 0, 0, 0))(
                    bp_params_used, init_hs_l, _target_obs, _dones,
                )
                # We use the BP for both target Q lookups; arg-max action under BP.
                bp_argmax = jnp.argmax(q_bp_target - (1 - _avail) * 1e10, axis=-1)
                q_bp_at_argmax = jnp.take_along_axis(
                    q_bp_target, bp_argmax[..., None], axis=-1
                ).squeeze(-1)

                def _loss_fn(params):
                    _, q_online = jax.vmap(online_net.apply, in_axes=(None, 0, 0, 0))(
                        params, init_hs_l, _obs, _dones,
                    )
                    chosen = jnp.take_along_axis(
                        q_online, _actions[..., None], axis=-1
                    ).squeeze(-1)
                    target = (
                        _rewards
                        + (1 - _dones.astype(jnp.float32)) * gamma * q_bp_at_argmax
                    )
                    loss = jnp.mean((chosen - jax.lax.stop_gradient(target)) ** 2)
                    return loss, chosen.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn = (buffer.can_sample(buffer_state)) & (
                train_state.timesteps > learning_starts
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn,
                lambda ts, r: jax.lax.scan(_learn, (ts, r), None, config.get("NUM_EPOCHS", 1)),
                lambda ts, r: ((ts, r), (jnp.zeros(config.get("NUM_EPOCHS", 1)),
                                          jnp.zeros(config.get("NUM_EPOCHS", 1)))),
                train_state, _rng,
            )

            train_state = jax.lax.cond(
                train_state.n_updates % target_update == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, tau
                    )
                ),
                lambda ts: ts,
                operand=train_state,
            )
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            return (train_state, buffer_state, rng), {"loss": loss.mean(), "qvals": qvals.mean()}

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def env_from_config(config):
    e = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    return LogWrapper(e), f"{config['ENV_NAME']}_{config['ENV_KWARGS'].get('num_agents', 2)}p"


def _maybe_load(path):
    if not path or path == "None" or not os.path.exists(path):
        return None
    from jaxmarl.wrappers.baselines import load_params
    return load_params(path)


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))
    env, env_name = env_from_config(copy.deepcopy(config))

    bp_params = _maybe_load(config.get("BP_CHECKPOINT"))
    belief_params = _maybe_load(config.get("BELIEF_CHECKPOINT"))
    print(f"[obl] BP loaded: {bp_params is not None}, "
          f"belief loaded: {belief_params is not None}")
    hidden_dim = int(config["HIDDEN_SIZE"])

    train_fn = make_train(config, env, bp_params, belief_params, hidden_dim)
    rng = jax.random.PRNGKey(config["SEED"])
    out = jax.jit(train_fn)(rng)
    print("OBL training complete.")
    return out


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    single_run(config)


if __name__ == "__main__":
    main()
