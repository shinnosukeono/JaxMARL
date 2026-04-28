"""
R2D2-text on Hanabi (numerical -> text observations, fixed action head).

Maps to TextLSTMNet with `out_dim != 1` in the original R3D2 repo
(pyhanabi/q_net.py:465, configs/r2d2_text.yaml). Concretely:

  obs -> get_obs_str() -> tokenize -> FlaxBert -> mean-pool -> LSTM -> Dueling Q

This file:
  * Loads FlaxBert (TinyBERT) ONCE outside the training loop.
  * Tokenises observations on the host via jax.pure_callback per env step.
  * Runs BERT on the resulting input_ids inside jit and stores the resulting
    mean-pooled state embedding in the replay buffer (frozen BERT).
  * Trains only an LSTM + dueling Q head on top of that embedding.

Frozen BERT matches `lm_weights: pretrained` (the default in r2d2_text.yaml).
Trainable BERT (lm_weights = "random" / "lora") needs replay to store raw
input_ids and a different forward; not implemented here.
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

from text_obs import make_tokenize_fn  # local module


# ---------------------------------------------------------------------------
# Recurrence (LSTM)
# ---------------------------------------------------------------------------


class ScannedLSTM(nn.Module):
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


# ---------------------------------------------------------------------------
# Network: takes already-encoded BERT state vector
# ---------------------------------------------------------------------------


class TextRNNQNetwork(nn.Module):
    """LSTM + dueling Q head over a precomputed text-encoded state vector."""

    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, state_emb, dones):
        # state_emb: (T, B, bert_hidden)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(state_emb)
        x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedLSTM()(hidden, rnn_in)

        v = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        a = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return hidden, v + a


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class Timestep:
    state_emb: dict      # per-agent BERT-encoded state, shape (bert_hidden,)
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
# Training
# ---------------------------------------------------------------------------


def make_train(config, env, bert_module, tokenize_fn, bert_hidden: int):
    """Returns `train(rng, bert_params)`.

    `bert_params` is threaded as a runtime argument (not closed over) so JAX
    treats it as an opaque tracer and skips the multi-second constant-folding
    pass it would otherwise do over BERT's embedding tables. Broadcast across
    seeds via `jax.vmap(train, in_axes=(0, None))` and across pmap devices via
    `jax.device_put_replicated`.
    """
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

    def encode_text(bert_params, input_ids, attention_mask):
        # input_ids: (..., max_tokens), attention_mask: (..., max_tokens)
        leading = input_ids.shape[:-1]
        flat_ids = input_ids.reshape(-1, input_ids.shape[-1])
        flat_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        out = bert_module.apply(
            {"params": bert_params},
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        # mean-pool over tokens, weighted by attention mask to match the paper's
        # mean over real tokens.
        h = out.last_hidden_state  # (N, T_tok, hid)
        mask = flat_mask[..., None].astype(h.dtype)
        h = (h * mask).sum(axis=1) / jnp.clip(mask.sum(axis=1), a_min=1.0)
        return h.reshape(*leading, bert_hidden)

    def get_greedy_actions(q_vals, valid_actions):
        unavail = 1 - valid_actions
        q_vals = q_vals - (unavail * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy(rng, q_vals, eps, valid_actions):
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

    def unbatchify(x):
        return {a: x[i] for i, a in enumerate(env.agents)}

    def encode_state_dict(bert_params, env_state, prev_env_state, last_actions):
        """Tokenize per-agent text obs and encode through BERT.

        bert_params is threaded as a runtime arg (not closed over) so XLA does
        not constant-fold BERT's embedding gathers.

        env_state / prev_env_state are LogWrapper-wrapped states (LogEnvState).
        Returns dict[agent -> jnp[B, bert_hidden]].
        """
        new_inner = env_state.env_state
        old_inner = prev_env_state.env_state
        ids, mask = tokenize_fn(new_inner, old_inner, last_actions)
        # ids: (num_agents, B, max_tokens); broadcast through BERT
        emb = encode_text(bert_params, ids, mask)  # (num_agents, B, hidden)
        return {a: emb[i] for i, a in enumerate(env.agents)}

    def train(rng, bert_params):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

        network = TextRNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, bert_hidden)),
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

        # Buffer init
        def _env_sample_step(carry, unused):
            env_state, prev_state, last_actions = carry
            rng_, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                a: wrapped_env.batch_sample(key_a[i], a)
                for i, a in enumerate(env.agents)
            }
            avail = wrapped_env.get_valid_actions(env_state.env_state)
            obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            cur_player_action = sum(
                actions[a] * env_state.env_state.cur_player_idx[..., i].astype(jnp.int32)
                for i, a in enumerate(env.agents)
            )
            state_emb = encode_state_dict(bert_params, new_env_state, env_state, cur_player_action)
            ts = Timestep(
                state_emb=state_emb,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail,
            )
            return (new_env_state, env_state, cur_player_action), ts

        _, _env_state = wrapped_env.batch_reset(rng)
        last_actions_init = jnp.full((config["NUM_ENVS"],), env.num_moves - 1, dtype=jnp.int32)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, (_env_state, _env_state, last_actions_init), None, config["NUM_STEPS"]
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

        # ---- update step ----
        def _update_step(runner_state, unused):
            train_state, buffer_state, test_state, rng = runner_state

            def _step_env(carry, _):
                hs, last_state_emb, last_dones, env_state, prev_state, last_actions, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _emb = batchify(last_state_emb)[:, np.newaxis]  # (n_agents, 1, n_envs, bert_hidden)
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params, hs, _emb, _dones
                )
                q_vals = q_vals.squeeze(axis=1)

                avail = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail)
                )
                actions = unbatchify(actions)

                cur_player_action = sum(
                    actions[a] * env_state.env_state.cur_player_idx[..., i].astype(jnp.int32)
                    for i, a in enumerate(env.agents)
                )
                obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                new_state_emb = encode_state_dict(bert_params, new_env_state, env_state, cur_player_action)
                ts = Timestep(
                    state_emb=last_state_emb,
                    actions=actions,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                    avail_actions=avail,
                )
                return (
                    new_hs,
                    new_state_emb,
                    dones,
                    new_env_state,
                    env_state,
                    cur_player_action,
                    rng,
                ), (ts, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_actions = jnp.full((config["NUM_ENVS"],), env.num_moves - 1, dtype=jnp.int32)
            init_state_emb = encode_state_dict(bert_params, env_state, env_state, init_actions)
            init_dones = {
                a: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for a in env.agents + ["__all__"]
            }
            init_hs = ScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl = (init_hs, init_state_emb, init_dones, env_state, env_state, init_actions)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env, (*expl, _rng), None, config["NUM_STEPS"]
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # ---- learn ----
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )

                init_hs = ScannedLSTM.initialize_carry(
                    config["HIDDEN_SIZE"], len(env.agents), config["BUFFER_BATCH_SIZE"]
                )
                _emb = batchify(minibatch.state_emb)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params, init_hs, _emb, _dones
                )

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params, init_hs, _emb, _dones
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
            is_learn = (buffer.can_sample(buffer_state)) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn,
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

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params

            def _greedy_step(step_state, unused):
                params, env_state, last_state_emb, last_dones, hs, rng = step_state
                rng, key_s = jax.random.split(rng)
                _emb = batchify(last_state_emb)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params, hs, _emb, _dones
                )
                q_vals = q_vals.squeeze(axis=1)
                valid = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid))
                actions = unbatchify(actions)
                cur_player_action = sum(
                    actions[a] * env_state.env_state.cur_player_idx[..., i].astype(jnp.int32)
                    for i, a in enumerate(env.agents)
                )
                obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                new_emb = encode_state_dict(bert_params, new_env_state, env_state, cur_player_action)
                step_state = (params, new_env_state, new_emb, dones, hs, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_actions = jnp.full((config["TEST_NUM_ENVS"],), env.num_moves - 1, dtype=jnp.int32)
            init_state_emb = encode_state_dict(bert_params, env_state, env_state, init_actions)
            init_dones = {
                a: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for a in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hs = ScannedLSTM.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )
            step_state = (params, env_state, init_state_emb, init_dones, hs, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
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


def env_from_config(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    return env, f"{config['ENV_NAME']}_{config['ENV_KWARGS'].get('num_agents', 2)}p"


def load_bert(model_dir: str):
    """Load FlaxBert + tokenizer from a local directory."""
    from transformers import AutoTokenizer, FlaxBertModel
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = FlaxBertModel.from_pretrained(model_dir)
    return tokenizer, model


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))
    alg_name = config.get("ALG_NAME", "r2d2_text_rnn_hanabi")
    env, env_name = env_from_config(copy.deepcopy(config))

    tokenizer, hf_model = load_bert(config["BERT_MODEL_DIR"])
    bert_module = hf_model.module
    bert_params = hf_model.params
    bert_hidden = hf_model.config.hidden_size
    tokenize_fn = make_tokenize_fn(
        env, tokenizer,
        max_obs_tokens=int(config["MAX_OBS_TOKENS"]),
        include_belief=bool(config.get("INCLUDE_BELIEF", False)),
    )

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
    train_fn = make_train(config, env, bert_module, tokenize_fn, bert_hidden)
    # bert_params threaded as a runtime arg (not closed over) and broadcast
    # across the seed-vmap with in_axes=(0, None).
    train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))
    outs = jax.block_until_ready(train_vjit(rngs, bert_params))

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
    single_run(config)


if __name__ == "__main__":
    main()
