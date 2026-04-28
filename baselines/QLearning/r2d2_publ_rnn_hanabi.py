"""
R2D2 with Public/Private LSTM for Hanabi.

Faithful port of the R2D2 baseline from
  Hu et al., "Off-Belief Learning"  (PublicLSTMNet, q_net.py:321 in the R3D2 repo)
to JaxMARL, modelled on baselines/QLearning/iql_rnn.py.

Key differences vs. iql_rnn.py:
  - Splits Hanabi observations into (priv, publ) along the natural feature layout
    of jaxmarl.environments.hanabi.HanabiEnv:
      priv = hands_feats + last_action_feats + v0_belief_feats + agent_id one-hot
      publ = board_feats + discard_feats
  - Replaces the single dense+GRU stack with the PublicLSTMNet recipe:
      lstm operates on publ features only (so each agent's recurrent state
      is shareable / public-belief consistent), priv features are gated in
      via element-wise multiplication on the LSTM output.
  - LSTM (OptimizedLSTMCell) instead of GRU, matching the original.
  - Dueling Q head: Q = V + A * legal_move (the "fake dueling" used in
    PublicLSTMNet.forward at q_net.py:421).
  - Multi-step (n-step) returns via config["MULTI_STEP"], default 1.
"""

import os
import copy
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


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------


class ScannedLSTM(nn.Module):
    """LSTM over a (time, batch, dim) sequence with per-step `resets` flag.

    Mirrors iql_rnn.py's ScannedRNN but uses Flax's OptimizedLSTMCell so the
    architecture matches Hu et al.'s PublicLSTMNet.
    """

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
        # On reset, zero the carry.
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


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class PublicLSTMQNetwork(nn.Module):
    """Public/private LSTM dueling Q-network.

    publ -> dense -> LSTM (recurrent) -> publ_o
    priv -> 3x dense (relu)            -> priv_o
    o     = priv_o * publ_o
    Q     = V(o) + A(o) * legal_mask
    """

    action_dim: int
    hidden_dim: int
    publ_split: int  # boundary inside obs vector: obs[..., :publ_split] is priv
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        # obs is (time, batch, obs_dim) in CTRolloutManager order.
        priv_s = obs[..., : self.publ_split]
        publ_s = obs[..., self.publ_split :]

        # Private encoder (3 dense + relu, like PublicLSTMNet.priv_net).
        priv_o = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(priv_s)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(priv_o)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(priv_o)
        priv_o = nn.relu(priv_o)

        # Public encoder (1 dense + relu) -> LSTM.
        publ_x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(publ_s)
        publ_x = nn.relu(publ_x)

        rnn_in = (publ_x, dones)
        hidden, publ_o = ScannedLSTM()(hidden, rnn_in)

        o = priv_o * publ_o

        v = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(o)
        a = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(o)
        # Note: the legal-move masking happens at action-selection / loss time.
        return hidden, v + a


# ---------------------------------------------------------------------------
# Replay timestep
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hanabi obs split
# ---------------------------------------------------------------------------


def hanabi_feature_widths(env):
    """Read the Hanabi feature widths off the (possibly wrapped) env via __getattr__."""
    return dict(
        hands=int(env.hands_n_feats),
        board=int(env.board_n_feats),
        discard=int(env.discards_n_feats),
        last_action=int(env.last_action_n_feats),
        belief=int(env.v0_belief_n_feats),
        agent_id=int(env.num_agents),  # CTRolloutManager appends an agent_id one-hot
    )


def hanabi_publ_split(env):
    """Return (priv_dim, publ_dim) where the reordered obs is [priv | publ]."""
    w = hanabi_feature_widths(env)
    priv_dim = w["hands"] + w["last_action"] + w["belief"] + w["agent_id"]
    publ_dim = w["board"] + w["discard"]
    return priv_dim, publ_dim


def reorder_obs_for_split(obs, widths):
    """Re-order CTRolloutManager-padded obs vectors so [priv | publ] is contiguous.

    Original layout (per agent, after CTRolloutManager._preprocess_obs):
      [hands | board | discard | last_action | belief | agent_id_one_hot]
    Target:
      [hands | last_action | belief | agent_id_one_hot | board | discard]

    Operates on the trailing axis. `widths` is a dict from hanabi_feature_widths().
    """
    h = widths["hands"]
    b = widths["board"]
    d = widths["discard"]
    la = widths["last_action"]
    bel = widths["belief"]
    aid = widths["agent_id"]

    hands = obs[..., :h]
    board = obs[..., h : h + b]
    discard = obs[..., h + b : h + b + d]
    last_action = obs[..., h + b + d : h + b + d + la]
    belief = obs[..., h + b + d + la : h + b + d + la + bel]
    agent_id = obs[..., h + b + d + la + bel : h + b + d + la + bel + aid]

    return jnp.concatenate(
        [hands, last_action, belief, agent_id, board, discard], axis=-1
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_train(config, env):

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

    priv_dim, publ_dim = hanabi_publ_split(env)
    feat_widths = hanabi_feature_widths(env)
    obs_dim = priv_dim + publ_dim  # post-reorder, no padding (Hanabi agents are homogeneous)

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
        # obs_dict has env.agents + ['__all__']; only re-order per-agent slices.
        out = {a: reorder_obs_for_split(obs_dict[a], feat_widths) for a in env.agents}
        if "__all__" in obs_dict:
            out["__all__"] = obs_dict["__all__"]
        return out

    def train(rng):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

        # Network
        network = PublicLSTMQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
            publ_split=priv_dim,
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, obs_dim)),
                jnp.zeros((1, 1)),
            )
            init_hs = ScannedLSTM.initialize_carry(config["HIDDEN_SIZE"], 1)
            params = network.init(rng, init_hs, *init_x)

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

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        # Buffer init: sample a dummy trajectory to learn the structure.
        def _env_sample_step(env_state, unused):
            rng_, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                a: wrapped_env.batch_sample(key_a[i], a)
                for i, a in enumerate(env.agents)
            }
            avail = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            obs = reorder_obs_dict(obs)
            ts = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail,
            )
            return env_state, ts

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------
        def _update_step(runner_state, unused):
            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE -------------------------------------------------
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = batchify(last_obs)[:, np.newaxis]  # (n_agents, 1, n_envs, obs)
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(
                    train_state.params, hs, _obs, _dones
                )
                q_vals = q_vals.squeeze(axis=1)

                avail = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                new_obs = reorder_obs_dict(new_obs)
                ts = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(
                        lambda x: config.get("REW_SCALE", 1) * x, rewards
                    ),
                    dones=last_dones,
                    avail_actions=avail,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (ts, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_obs = reorder_obs_dict(init_obs)
            init_dones = {
                a: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for a in env.agents + ["__all__"]
            }
            init_hs = ScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env, (*expl_state, _rng), None, config["NUM_STEPS"]
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN PHASE --------------------------------------------------
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                # remove dummy seq dim, swap to (T, B, ...)
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )

                init_hs = ScannedLSTM.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params, init_hs, _obs, _dones
                )

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params, init_hs, _obs, _dones
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

                    # n-step return: r_t + gamma * r_{t+1} + ... + gamma^n * Q(s_{t+n})
                    if multi_step <= 1:
                        target = (
                            _rewards[:, :-1]
                            + (1 - _dones[:, :-1]) * gamma * q_next[:, 1:]
                        )
                        chosen_for_target = chosen[:, :-1]
                    else:
                        T = _rewards.shape[1]
                        # accumulate n-step returns (truncated at episode boundary).
                        gammas = gamma ** jnp.arange(multi_step)
                        # pad rewards / dones at the end so we can shift cleanly
                        pad = lambda x, val: jnp.concatenate(
                            [x, jnp.full(x.shape[:1] + (multi_step,) + x.shape[2:], val)],
                            axis=1,
                        )
                        r_pad = pad(_rewards, 0.0)
                        d_pad = pad(_dones.astype(jnp.float32), 1.0)
                        # accumulate
                        ret = jnp.zeros_like(_rewards)
                        not_done = jnp.ones_like(_rewards)
                        for k in range(multi_step):
                            ret = ret + not_done * gammas[k] * r_pad[:, k : k + T]
                            not_done = not_done * (1 - d_pad[:, k : k + T])
                        # target = ret_t + gamma^n * (1 - any_done) * Q(s_{t+n})
                        q_shift = jnp.concatenate(
                            [q_next[:, multi_step:], jnp.zeros_like(q_next[:, :multi_step])],
                            axis=1,
                        )
                        target = ret + not_done * (gamma ** multi_step) * q_shift
                        # truncate the trailing multi_step entries that have no valid bootstrap
                        valid = jnp.arange(T) < (T - multi_step)
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
                lambda ts, r: jax.lax.scan(
                    _learn_phase, (ts, r), None, config["NUM_EPOCHS"]
                ),
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

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            return (train_state, buffer_state, test_state, rng), None

        # ------------------------------------------------------------
        # Eval (greedy)
        # ------------------------------------------------------------
        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params

            def _greedy_step(step_state, unused):
                params, env_state, last_obs, last_dones, hs, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params, hs, _obs, _dones
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
            hs = ScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )
            step_state = (params, env_state, init_obs, init_dones, hs, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(infos["returned_episode"], x, jnp.nan)
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def env_from_config(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    return env, f"{config['ENV_NAME']}_{config['ENV_KWARGS'].get('num_agents', 2)}p"


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

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        for i, r in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_params(
                params,
                os.path.join(
                    save_dir,
                    f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
                ),
            )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
