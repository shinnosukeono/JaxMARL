"""
R2D2 with Public/Private LSTM for Hanabi.

Faithful port of the R2D2 baseline from
  Hu et al., "Off-Belief Learning"  (PublicLSTMNet, q_net.py:321 in the R3D2 repo)

Optional Other-Play: per-episode random color permutation in observations
and inverse-permuted color hint actions. Toggle via config["OTHER_PLAY"].

JIT structure
-------------
We JIT (init_fn, update_step, eval_step) separately and drive the outer
NUM_UPDATES loop in Python. See r3d2_rnn_hanabi.py for the rationale.
"""

import os
import copy
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager

from other_play import OtherPlayState, op_init, op_resample, op_permute_obs, op_unpermute_action


class ScannedLSTM(nn.Module):
    """Single-layer LSTM scanned over a (time, batch, dim) sequence with
    per-step `resets` flag. Kept for use by the text/R3D2/multitask networks
    that don't need a 2-layer stack."""

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        hidden_size = ins.shape[-1]
        zeros = self.initialize_carry(hidden_size, *ins.shape[:-1])
        carry = jax.tree.map(
            lambda c, z: jnp.where(resets[:, np.newaxis], z, c),
            carry,
            zeros,
        )
        new_carry, y = nn.OptimizedLSTMCell(hidden_size)(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class MultiLayerScannedLSTM(nn.Module):
    """`num_layers`-deep LSTM scanned over time with per-step resets.

    Mirrors PyTorch `nn.LSTM(hid_dim, hid_dim, num_layers=num_layers)` used
    in upstream R3D2 (`pyhanabi/q_net.py:349` PublicLSTMNet,
    `pyhanabi/belief_model.py:47` ARBeliefModel). Carry layout matches the
    JaxMARL OBL Flax loader at
    `jaxmarl/environments/hanabi/pretrained/obl_r2d2_agent.py:MultiLayerLSTM`:
        carry = (c_stack, h_stack), each shape (num_layers, *batch, hidden).
    Per-step reset zeroes all layers' carries on `dones=True`.
    """

    num_layers: int = 2

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        zeros = jax.tree.map(jnp.zeros_like, carry)
        # Broadcast resets to leading layer dim and trailing hidden dim.
        reset_mask = resets[None, ..., None]
        carry = jax.tree.map(
            lambda c, z: jnp.where(reset_mask, z, c),
            carry,
            zeros,
        )
        hidden_size = ins.shape[-1]
        new_cs, new_hs = [], []
        y = ins
        for layer in range(self.num_layers):
            layer_carry = jax.tree.map(lambda x, _l=layer: x[_l], carry)
            new_layer_carry, y = nn.OptimizedLSTMCell(
                hidden_size, name=f"l{layer}"
            )(layer_carry, y)
            new_cs.append(new_layer_carry[0])
            new_hs.append(new_layer_carry[1])
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, y

    @staticmethod
    def initialize_carry(hidden_size, num_layers, *batch_size):
        mem_shape = (num_layers, *batch_size, hidden_size)
        return (jnp.zeros(mem_shape), jnp.zeros(mem_shape))


class PublicLSTMQNetwork(nn.Module):
    """Public/private LSTM dueling Q-network.

    Precise port of `PublicLSTMNet` in upstream R3D2's
    `pyhanabi/q_net.py:321`. Architecture:
        priv_net : 3 × (Linear + ReLU)
        publ_net : 1 × (Linear + ReLU)
        lstm     : nn.LSTM(hid_dim, hid_dim, num_layers=num_lstm_layer)
        head     : fc_v(1) + fc_a(out_dim) — combined as q = v + a
                   (functionally equivalent to upstream's
                    q = v + a*legal_move for any legal action; illegal
                    actions are masked separately by the trainer/eval).
    Default `num_lstm_layer=2` matches `r2d2_main.py:69` upstream.

    Slicing convention matches the upstream OBL loader at
    `jaxmarl/environments/hanabi/pretrained/obl_r2d2_agent.py`:
        priv_s = full obs                 (priv MLP sees everything)
        publ_s = obs[..., hands_dim:]     (public LSTM sees obs minus
                                           the partner-cards block)
    """

    action_dim: int
    hidden_dim: int
    num_lstm_layer: int = 2
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, priv_s, publ_s, dones):
        priv_o = nn.Dense(self.hidden_dim,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(priv_s)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(self.hidden_dim,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(priv_o)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(self.hidden_dim,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(priv_o)
        priv_o = nn.relu(priv_o)

        publ_x = nn.Dense(self.hidden_dim,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(publ_s)
        publ_x = nn.relu(publ_x)

        rnn_in = (publ_x, dones)
        hidden, publ_o = MultiLayerScannedLSTM(num_layers=self.num_lstm_layer)(
            hidden, rnn_in
        )

        o = priv_o * publ_o

        v = nn.Dense(1,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(o)
        a = nn.Dense(self.action_dim,
            kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(o)
        return hidden, v + a

    @staticmethod
    def initialize_carry(hidden_size, num_lstm_layer, *batch_size):
        return MultiLayerScannedLSTM.initialize_carry(
            hidden_size, num_lstm_layer, *batch_size
        )


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def hanabi_feature_widths(env):
    return dict(
        hands=int(env.hands_n_feats),
        board=int(env.board_n_feats),
        discard=int(env.discards_n_feats),
        last_action=int(env.last_action_n_feats),
        belief=int(env.v0_belief_n_feats),
        agent_id=int(env.num_agents),
    )


def hanabi_hands_dim(env):
    """Number of leading obs features that contain the partner-hand block —
    the only genuinely private slice of the JaxMARL Hanabi obs.

    Matches the cut used by the upstream OBL loader at
    `jaxmarl/environments/hanabi/pretrained/obl_r2d2_agent.py:greedy_act`,
    which slices `publ_s = obs[..., 125:]` for 2p (i.e. excludes only the
    `(num_agents-1) * hand_size * num_colors * num_ranks` partner-cards
    block). The remaining JaxMARL `hands_n_feats` dims (per-agent
    missing-card flags) are public and end up in the publ stream.
    """
    return int(
        (env.num_agents - 1) * env.hand_size * env.num_colors * env.num_ranks
    )


def split_obs_publ_priv(obs, hands_dim: int):
    """Slice an obs tensor into (priv_s, publ_s) per the upstream convention.

    The JaxMARL Hanabi obs (after CTRolloutManager appends agent_id) is laid
    out as `[hands | board | discard | last_action | belief | agent_id]`, so
    no reordering is needed — only a slice. priv_s is the *full* obs.
    """
    return obs, obs[..., hands_dim:]


def make_train(config, env):
    """Returns (init_fn, update_step, eval_step) — pure JAX functions."""
    use_other_play = bool(config.get("OTHER_PLAY", False))

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )
    multi_step = int(config.get("MULTI_STEP", 1))
    gamma = float(config["GAMMA"])

    feat_widths = hanabi_feature_widths(env)
    hands_dim = hanabi_hands_dim(env)
    obs_dim = sum(feat_widths.values())
    publ_dim = obs_dim - hands_dim
    num_lstm_layer = int(config.get("NUM_LSTM_LAYER", 2))

    wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
    test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

    network = PublicLSTMQNetwork(
        action_dim=wrapped_env.max_action_space,
        hidden_dim=config["HIDDEN_SIZE"],
        num_lstm_layer=num_lstm_layer,
    )

    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
        min_length_time_axis=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_batch_size=config["NUM_ENVS"],
        sample_sequence_length=1,
        period=1,
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail = 1 - valid_actions
        q_vals = q_vals - (unavail * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy = get_greedy_actions(q_vals, valid_actions)

        def get_random(rng, val):
            return jax.random.choice(
                rng,
                jnp.arange(val.shape[-1]),
                p=val * 1.0 / jnp.sum(val, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random)(_rngs, valid_actions)
        return jnp.where(
            jax.random.uniform(rng_e, greedy.shape) < eps, random_actions, greedy
        )

    def batchify(x: dict):
        return jnp.stack([x[a] for a in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {a: x[i] for i, a in enumerate(env.agents)}

    def reorder_obs_dict(obs_dict):
        # JaxMARL's Hanabi obs is already laid out so that the priv/publ split
        # is a simple slice (`obs[..., hands_dim:]`); no reordering needed.
        return obs_dict

    def _identity_op_state(num_envs):
        return OtherPlayState(
            color_perm=jnp.broadcast_to(
                jnp.arange(env.num_colors), (num_envs, env.num_colors)
            ),
            inv_color_perm=jnp.broadcast_to(
                jnp.arange(env.num_colors), (num_envs, env.num_colors)
            ),
        )

    # ------------------------------------------------------------------
    # init_fn
    # ------------------------------------------------------------------
    def init_fn(rng):
        rng, _rng = jax.random.split(rng)

        def create_agent(rng):
            init_priv = jnp.zeros((1, 1, obs_dim))
            init_publ = jnp.zeros((1, 1, publ_dim))
            init_dones = jnp.zeros((1, 1))
            init_hs = MultiLayerScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], num_lstm_layer, 1
            )
            params = network.init(rng, init_hs, init_priv, init_publ, init_dones)

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=config["NUM_EPOCHS"] * config["NUM_UPDATES"],
            )
            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr, eps=config.get("ADAM_EPS", 1.5e-5)),
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=params,
                target_network_params=params,
                tx=tx,
            )

        train_state = create_agent(_rng)

        def _env_sample_step(carry, unused):
            env_state, rng_s = carry
            rng_s, key_a, key_step = jax.random.split(rng_s, 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                a: wrapped_env.batch_sample(key_a[i], a)
                for i, a in enumerate(env.agents)
            }
            avail = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_step, env_state, actions
            )
            obs = reorder_obs_dict(obs)
            ts = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail,
            )
            return (env_state, rng_s), ts

        rng, _rng = jax.random.split(rng)
        _, _env_state = wrapped_env.batch_reset(_rng)
        rng, _rng = jax.random.split(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, (_env_state, _rng), None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        buffer_state = buffer.init(sample_traj_unbatched)
        return train_state, buffer_state

    # ------------------------------------------------------------------
    # update_step
    # ------------------------------------------------------------------
    def update_step(train_state, buffer_state, rng):
        def _step_env(carry, _):
            hs, last_obs, last_dones, env_state, op_state, rng = carry
            rng, rng_a, rng_s, rng_op = jax.random.split(rng, 4)
            _obs = batchify(last_obs)[:, np.newaxis]
            _dones = batchify(last_dones)[:, np.newaxis]
            _priv, _publ = split_obs_publ_priv(_obs, hands_dim)

            new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                train_state.params, hs, _priv, _publ, _dones
            )
            q_vals = q_vals.squeeze(axis=1)

            avail = wrapped_env.get_valid_actions(env_state.env_state)
            eps = eps_scheduler(train_state.n_updates)
            _rngs = jax.random.split(rng_a, env.num_agents)
            actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                _rngs, q_vals, eps, batchify(avail)
            )
            actions = unbatchify(actions)

            if use_other_play:
                game_actions = op_unpermute_action(actions, op_state, env)
            else:
                game_actions = actions

            new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                rng_s, env_state, game_actions
            )

            if use_other_play:
                op_state = op_resample(op_state, dones["__all__"], rng_op)
                new_obs = op_permute_obs(new_obs, op_state, env)

            new_obs = reorder_obs_dict(new_obs)

            ts = Timestep(
                obs=last_obs,
                actions=actions,
                rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                dones=last_dones,
                avail_actions=avail,
            )
            return (new_hs, new_obs, dones, new_env_state, op_state, rng), (ts, infos)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = wrapped_env.batch_reset(_rng)

        rng, _rng_op = jax.random.split(rng)
        if use_other_play:
            op_state = op_init(_rng_op, config["NUM_ENVS"], env)
            init_obs = op_permute_obs(init_obs, op_state, env)
        else:
            op_state = _identity_op_state(config["NUM_ENVS"])
        init_obs = reorder_obs_dict(init_obs)

        init_dones = {
            a: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            for a in env.agents + ["__all__"]
        }
        init_hs = MultiLayerScannedLSTM.initialize_carry(
            config["HIDDEN_SIZE"], num_lstm_layer, len(env.agents), config["NUM_ENVS"]
        )
        expl_state = (init_hs, init_obs, init_dones, env_state, op_state)
        rng, _rng = jax.random.split(rng)
        _, (timesteps, infos) = jax.lax.scan(
            _step_env, (*expl_state, _rng), None, config["NUM_STEPS"]
        )

        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
        )

        buffer_traj_batch = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
        )
        buffer_state = buffer.add(buffer_state, buffer_traj_batch)

        def _learn_phase(carry, _):
            train_state, rng = carry
            rng, _rng = jax.random.split(rng)
            minibatch = buffer.sample(buffer_state, _rng).experience
            minibatch = jax.tree.map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
            )

            init_hs_l = MultiLayerScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], num_lstm_layer,
                len(env.agents), config["BUFFER_BATCH_SIZE"],
            )
            _obs = batchify(minibatch.obs)
            _dones = batchify(minibatch.dones)
            _actions = batchify(minibatch.actions)
            _rewards = batchify(minibatch.rewards)
            _avail = batchify(minibatch.avail_actions)
            _priv, _publ = split_obs_publ_priv(_obs, hands_dim)

            _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                train_state.target_network_params, init_hs_l, _priv, _publ, _dones
            )

            def _loss_fn(params):
                _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                    params, init_hs_l, _priv, _publ, _dones
                )
                chosen = jnp.take_along_axis(
                    q_vals, _actions[..., None], axis=-1
                ).squeeze(-1)
                unavail = 1 - _avail
                valid_q = q_vals - (unavail * 1e10)
                q_next = jnp.take_along_axis(
                    q_next_target,
                    jnp.argmax(valid_q, axis=-1)[..., None],
                    axis=-1,
                ).squeeze(-1)

                if multi_step <= 1:
                    target = (
                        _rewards[:, :-1]
                        + (1 - _dones[:, :-1]) * gamma * q_next[:, 1:]
                    )
                    chosen_for_target = chosen[:, :-1]
                else:
                    T = _rewards.shape[1]
                    gammas = gamma ** jnp.arange(multi_step)
                    pad = lambda x, val: jnp.concatenate(
                        [x, jnp.full(x.shape[:1] + (multi_step,) + x.shape[2:], val)],
                        axis=1,
                    )
                    r_pad = pad(_rewards, 0.0)
                    d_pad = pad(_dones.astype(jnp.float32), 1.0)
                    ret = jnp.zeros_like(_rewards)
                    not_done = jnp.ones_like(_rewards)
                    for k in range(multi_step):
                        ret = ret + not_done * gammas[k] * r_pad[:, k : k + T]
                        not_done = not_done * (1 - d_pad[:, k : k + T])
                    q_shift = jnp.concatenate(
                        [q_next[:, multi_step:], jnp.zeros_like(q_next[:, :multi_step])],
                        axis=1,
                    )
                    target = ret + not_done * (gamma ** multi_step) * q_shift
                    target = target[:, : T - multi_step]
                    chosen_for_target = chosen[:, : T - multi_step]

                loss = jnp.mean(
                    (chosen_for_target - jax.lax.stop_gradient(target)) ** 2
                )
                return loss, chosen_for_target.mean()

            (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
            return (train_state, rng), (loss, qvals)

        rng, _rng = jax.random.split(rng)
        is_learn_time = (buffer.can_sample(buffer_state)) & (
            train_state.timesteps > config["LEARNING_STARTS"]
        )
        (train_state, rng), (loss, qvals) = jax.lax.cond(
            is_learn_time,
            lambda ts, r: jax.lax.scan(_learn_phase, (ts, r), None, config["NUM_EPOCHS"]),
            lambda ts, r: ((ts, r), (jnp.zeros(config["NUM_EPOCHS"]), jnp.zeros(config["NUM_EPOCHS"]))),
            train_state,
            _rng,
        )

        train_state = jax.lax.cond(
            train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
            lambda ts: ts.replace(
                target_network_params=optax.incremental_update(
                    ts.params, ts.target_network_params, config["TAU"]
                )
            ),
            lambda ts: ts,
            operand=train_state,
        )

        train_state = train_state.replace(n_updates=train_state.n_updates + 1)
        metrics = {
            "env_step": train_state.timesteps,
            "update_steps": train_state.n_updates,
            "grad_steps": train_state.grad_steps,
            "loss": loss.mean(),
            "qvals": qvals.mean(),
        }
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))
        return train_state, buffer_state, metrics

    # ------------------------------------------------------------------
    # eval_step (greedy)
    # ------------------------------------------------------------------
    def eval_step(params, rng):
        def _greedy_step(step_state, unused):
            params, env_state, last_obs, last_dones, hs, rng = step_state
            rng, key_s = jax.random.split(rng)
            _obs = batchify(last_obs)[:, np.newaxis]
            _dones = batchify(last_dones)[:, np.newaxis]
            _priv, _publ = split_obs_publ_priv(_obs, hands_dim)
            hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                params, hs, _priv, _publ, _dones
            )
            q_vals = q_vals.squeeze(axis=1)
            valid = test_env.get_valid_actions(env_state.env_state)
            actions = get_greedy_actions(q_vals, batchify(valid))
            actions = unbatchify(actions)
            obs, env_state, rewards, dones, infos = test_env.batch_step(
                key_s, env_state, actions
            )
            obs = reorder_obs_dict(obs)
            step_state = (params, env_state, obs, dones, hs, rng)
            return step_state, (rewards, dones, infos)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = test_env.batch_reset(_rng)
        init_obs = reorder_obs_dict(init_obs)
        init_dones = {
            a: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            for a in env.agents + ["__all__"]
        }
        rng, _rng = jax.random.split(rng)
        hs = MultiLayerScannedLSTM.initialize_carry(
            config["HIDDEN_SIZE"], num_lstm_layer,
            len(env.agents), config["TEST_NUM_ENVS"]
        )
        step_state = (params, env_state, init_obs, init_dones, hs, _rng)
        step_state, (rewards, dones, infos) = jax.lax.scan(
            _greedy_step, step_state, None, config["TEST_NUM_STEPS"]
        )
        metrics = jax.tree.map(
            lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
            infos,
        )
        return metrics

    return init_fn, update_step, eval_step


def env_from_config(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    return env, f"{config['ENV_NAME']}_{config['ENV_KWARGS'].get('num_agents', 2)}p"


def _np_mean(x):
    return float(np.array(x).mean())


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))
    alg_name = config.get("ALG_NAME", "r2d2_publ_rnn_hanabi")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    init_fn, update_step, eval_step = make_train(config, env)

    num_seeds = int(config["NUM_SEEDS"])
    init_jit = jax.jit(jax.vmap(init_fn))
    update_step_jit = jax.jit(jax.vmap(update_step, in_axes=(0, 0, 0)))
    eval_step_jit = jax.jit(jax.vmap(eval_step, in_axes=(0, 0)))

    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, num_seeds)

    print(f"[init] compiling + running init across {num_seeds} seed(s)...", flush=True)
    t0 = time.time()
    train_state, buffer_state = init_jit(rngs)
    train_state = jax.block_until_ready(train_state)
    print(f"[init] done in {time.time() - t0:.1f}s", flush=True)

    NUM_UPDATES = int(config["NUM_UPDATES"])
    eval_interval = max(1, int(NUM_UPDATES * config.get("TEST_INTERVAL", 0.01)))
    log_every = int(config.get("LOG_EVERY", 50))
    test_during_training = bool(config.get("TEST_DURING_TRAINING", True))

    print(
        f"[train] {NUM_UPDATES} updates, eval_interval={eval_interval}, "
        f"log_every={log_every}, test_during_training={test_during_training}",
        flush=True,
    )

    use_wandb = config["WANDB_MODE"] != "disabled"
    log_all_seeds = bool(config.get("WANDB_LOG_ALL_SEEDS", False))
    seed_root = int(config["SEED"])

    t_loop = time.time()
    for u in range(NUM_UPDATES):
        rng, _rng = jax.random.split(rng)
        rngs_step = jax.random.split(_rng, num_seeds)
        train_state, buffer_state, metrics = update_step_jit(
            train_state, buffer_state, rngs_step
        )

        if u == 0:
            train_state = jax.block_until_ready(train_state)
            print(
                f"[update_step] first compile + run: {time.time() - t_loop:.1f}s",
                flush=True,
            )
            t_loop = time.time()

        if u % log_every == 0 or u == NUM_UPDATES - 1:
            metrics_np = {k: _np_mean(v) for k, v in metrics.items()}
            elapsed = time.time() - t_loop
            ups = (u + 1) / max(1e-9, elapsed)
            print(f"[u={u}/{NUM_UPDATES}] {metrics_np} | {ups:.2f} u/s", flush=True)
            if use_wandb:
                if log_all_seeds:
                    log_dict = {f"u": u, **{f"all/{k}": v for k, v in metrics_np.items()}}
                    for s in range(num_seeds):
                        for k, v in metrics.items():
                            log_dict[f"rng{seed_root + s}/{k}"] = float(np.array(v)[s])
                else:
                    log_dict = {"u": u, **metrics_np}
                wandb.log(log_dict)

        if test_during_training and (u % eval_interval == 0 or u == NUM_UPDATES - 1):
            rng, _rng = jax.random.split(rng)
            rngs_eval = jax.random.split(_rng, num_seeds)
            eval_metrics = eval_step_jit(train_state.params, rngs_eval)
            eval_np = {k: _np_mean(v) for k, v in eval_metrics.items()}
            print(f"[eval u={u}] {eval_np}", flush=True)
            if use_wandb:
                wandb.log({"u": u, **{f"test_{k}": v for k, v in eval_np.items()}})

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        for i in range(num_seeds):
            params = jax.tree.map(lambda x: x[i], train_state.params)
            save_params(
                params,
                os.path.join(
                    save_dir,
                    f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
                ),
            )

    wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
