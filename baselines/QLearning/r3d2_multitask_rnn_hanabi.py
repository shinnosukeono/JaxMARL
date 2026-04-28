"""
Multi-task R3D2 on Hanabi: train one network across 2,3,4,5-player settings.

Mirrors `pyhanabi/r3d2_main.py` running `--num_player 6`, where one R3D2 agent
is trained over a rotation of player-count games sharing the same parameters.
The original repo ran this with separate game contexts and one shared
replay buffer; we use the same idea (params shared, per-env replay buffers)
but driven from a Python round-robin loop because Hanabi `State` shapes vary
with `num_agents` and so cannot be vmapped or jax.lax.switch'd directly.

Architecture re-uses everything from `r3d2_rnn_hanabi.py`:
    BERT (frozen)  ->  state_emb (T, B, bert_hidden)
    LSTM + Dense   ->  (T, B, hidden_dim)  -> proj -> bert_hidden
    DRRN gate      :   state_proj * action_emb_i           [per-env action_emb]
    dueling head   :   Q = V + A * legal

All params (LSTM, projections, V/A heads) are A-agnostic, so they're shared
across player counts. Only `action_emb` is per-env (different action strings
in different player-count games).

The training loop rotates through envs:

    for u in range(NUM_UPDATES):
        env_idx = u % num_envs
        train_state, buffer_states[env_idx] = update_step_for[env_idx](...)

Each per-env `update_step_for[env_idx]` is JIT-compiled once.
"""

from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, Dict, List

import chex
import flax.linen as nn
import flashbax as fbx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager

from r3d2_rnn_hanabi import (
    R3D2Net, ScannedLSTM, Timestep, CustomTrainState,
    load_bert, precompute_action_embedding,
)
from text_obs import make_tokenize_fn


# ---------------------------------------------------------------------------
# Per-env update step (one JIT compilation per env-shape)
# ---------------------------------------------------------------------------


def _make_update_step(
    config,
    env,
    bert_module,
    tokenize_fn,
    bert_hidden: int,
    action_emb,
    network: R3D2Net,
    buffer,
):
    """Build the (rollout + learn + target-update) function for one env.

    Returns `update_step(train_state, buffer_state, bert_params, rng)` —
    `bert_params` is a runtime arg so XLA does not constant-fold BERT.
    """

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"]
        * (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS_PER_SETTING"]),
    )
    multi_step = int(config.get("MULTI_STEP", 1))
    gamma = float(config["GAMMA"])

    NUM_ENVS = config["NUM_ENVS_PER_SETTING"]
    NUM_STEPS = config["NUM_STEPS"]

    def encode_text(bert_params, input_ids, attention_mask):
        leading = input_ids.shape[:-1]
        flat_ids = input_ids.reshape(-1, input_ids.shape[-1])
        flat_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        out = bert_module.apply(
            {"params": bert_params},
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        h = out.last_hidden_state
        m = flat_mask[..., None].astype(h.dtype)
        h = (h * m).sum(axis=1) / jnp.clip(m.sum(axis=1), a_min=1.0)
        return h.reshape(*leading, bert_hidden)

    def encode_state_dict(bert_params, env_state, prev_env_state, last_actions):
        new_inner = env_state.env_state
        old_inner = prev_env_state.env_state
        ids, mask = tokenize_fn(new_inner, old_inner, last_actions)
        emb = encode_text(bert_params, ids, mask)
        return {a: emb[i] for i, a in enumerate(env.agents)}

    def get_greedy_actions(q_vals, valid_actions):
        unavail = 1 - valid_actions
        return jnp.argmax(q_vals - unavail * 1e10, axis=-1)

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

    wrapped_env = CTRolloutManager(env, batch_size=NUM_ENVS)

    def update_step(train_state, buffer_state, bert_params, rng):
        # ---- rollout ----
        def _step_env(carry, _):
            hs, last_state_emb, last_dones, env_state, prev_state, last_actions, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _emb = batchify(last_state_emb)[:, np.newaxis]
            _dones = batchify(last_dones)[:, np.newaxis]
            new_hs, q_vals = jax.vmap(
                network.apply, in_axes=(None, 0, 0, 0, None)
            )(train_state.params, hs, _emb, _dones, action_emb)
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
                new_hs, new_state_emb, dones, new_env_state, env_state,
                cur_player_action, rng,
            ), (ts, infos)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_actions = jnp.full((NUM_ENVS,), env.num_moves - 1, dtype=jnp.int32)
        init_state_emb = encode_state_dict(bert_params, env_state, env_state, init_actions)
        init_dones = {
            a: jnp.zeros((NUM_ENVS,), dtype=bool) for a in env.agents + ["__all__"]
        }
        init_hs = ScannedLSTM.initialize_carry(
            config["HIDDEN_SIZE"], len(env.agents), NUM_ENVS
        )
        expl = (init_hs, init_state_emb, init_dones, env_state, env_state, init_actions)
        rng, _rng = jax.random.split(rng)
        _, (timesteps, infos) = jax.lax.scan(
            _step_env, (*expl, _rng), None, NUM_STEPS
        )

        train_state = train_state.replace(
            timesteps=train_state.timesteps + NUM_STEPS * NUM_ENVS
        )
        buffer_traj_batch = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
        )
        buffer_state = buffer.add(buffer_state, buffer_traj_batch)

        # ---- learn ----
        def _learn(carry, _):
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

            _, q_next_target = jax.vmap(
                network.apply, in_axes=(None, 0, 0, 0, None)
            )(train_state.target_network_params, init_hs, _emb, _dones, action_emb)

            def _loss_fn(params):
                _, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0, None)
                )(params, init_hs, _emb, _dones, action_emb)
                chosen = jnp.take_along_axis(
                    q_vals, _actions[..., None], axis=-1
                ).squeeze(-1)
                unavail = 1 - _avail
                valid_q = q_vals - unavail * 1e10
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
            lambda ts, r: jax.lax.scan(_learn, (ts, r), None, config["NUM_EPOCHS"]),
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
            "loss": loss.mean(),
            "qvals": qvals.mean(),
            "n_updates": train_state.n_updates,
            "timesteps": train_state.timesteps,
        }
        return train_state, buffer_state, metrics

    return jax.jit(update_step), wrapped_env


# ---------------------------------------------------------------------------
# Multi-task driver
# ---------------------------------------------------------------------------


def make_train_multitask(config, envs, bert_module, bert_params, tokenize_fns,
                         bert_hidden, action_embs):
    """Build a Python-loop driver that rotates through envs and shares params."""

    network = R3D2Net(hidden_dim=config["HIDDEN_SIZE"], bert_hidden=bert_hidden)

    def train(rng):
        # ---- build optimizer + initial train_state from any env (shapes shared) ----
        rng, _rng = jax.random.split(rng)
        # Use the first env's action_emb shape for init; all are bert_hidden anyway.
        init_x = (jnp.zeros((1, 1, bert_hidden)), jnp.zeros((1, 1)))
        init_hs = ScannedLSTM.initialize_carry(config["HIDDEN_SIZE"], 1)
        params = network.init(_rng, init_hs, *init_x, action_embs[0])

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=config.get("ADAM_EPS", 1.5e-5)),
        )
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=params,
            target_network_params=params,
            tx=tx,
        )

        # ---- per-env update fns + buffers ----
        update_fns = []
        wrapped_envs = []
        buffers = []
        buffer_states = []
        for env_i, action_emb_i, tokfn_i in zip(envs, action_embs, tokenize_fns):
            buffer_i = fbx.make_trajectory_buffer(
                max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS_PER_SETTING"],
                min_length_time_axis=config["BUFFER_BATCH_SIZE"],
                sample_batch_size=config["BUFFER_BATCH_SIZE"],
                add_batch_size=config["NUM_ENVS_PER_SETTING"],
                sample_sequence_length=1,
                period=1,
            )
            update_fn_i, wrapped_i = _make_update_step(
                config, env_i, bert_module, tokfn_i, bert_hidden,
                action_emb_i, network, buffer_i,
            )
            update_fns.append(update_fn_i)
            wrapped_envs.append(wrapped_i)
            buffers.append(buffer_i)

            # Initialize each buffer with a sample trajectory.
            sample_traj = _sample_dummy_traj(
                env_i, wrapped_i, tokfn_i, bert_module, bert_params, bert_hidden,
                config["NUM_STEPS"], config["NUM_ENVS_PER_SETTING"], rng,
            )
            buffer_states.append(buffer_i.init(jax.tree.map(lambda x: x[:, 0], sample_traj)))

        n_envs = len(envs)
        total_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS_PER_SETTING"]
        wandb_enabled = config["WANDB_MODE"] != "disabled"

        for u in range(int(total_updates)):
            env_idx = u % n_envs
            rng, _rng = jax.random.split(rng)
            train_state, buffer_states[env_idx], metrics = update_fns[env_idx](
                train_state, buffer_states[env_idx], bert_params, _rng
            )
            if wandb_enabled and u % max(1, int(config.get("LOG_EVERY", 50))) == 0:
                wandb.log(
                    {
                        f"env{env_idx}/loss": float(metrics["loss"]),
                        f"env{env_idx}/qvals": float(metrics["qvals"]),
                        "n_updates": int(metrics["n_updates"]),
                        "timesteps": int(metrics["timesteps"]),
                    }
                )

        return {"runner_state": (train_state, buffer_states)}

    return train


def _sample_dummy_traj(env, wrapped_env, tokenize_fn, bert_module, bert_params,
                       bert_hidden, num_steps, num_envs, rng):
    """Build one full trajectory of dummy timesteps so flashbax knows the shape."""
    def encode_text(input_ids, attention_mask):
        leading = input_ids.shape[:-1]
        flat_ids = input_ids.reshape(-1, input_ids.shape[-1])
        flat_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        out = bert_module.apply(
            {"params": bert_params},
            input_ids=flat_ids, attention_mask=flat_mask,
        )
        h = out.last_hidden_state
        m = flat_mask[..., None].astype(h.dtype)
        h = (h * m).sum(axis=1) / jnp.clip(m.sum(axis=1), a_min=1.0)
        return h.reshape(*leading, bert_hidden)

    rng, k1 = jax.random.split(rng)
    _, env_state = wrapped_env.batch_reset(k1)
    init_actions = jnp.full((num_envs,), env.num_moves - 1, dtype=jnp.int32)

    def step(carry, _):
        env_state, prev_state, last_actions, rng = carry
        rng, ka, ks = jax.random.split(rng, 3)
        ka = jax.random.split(ka, env.num_agents)
        actions = {a: wrapped_env.batch_sample(ka[i], a) for i, a in enumerate(env.agents)}
        avail = wrapped_env.get_valid_actions(env_state.env_state)
        obs, new_env_state, rewards, dones, _ = wrapped_env.batch_step(ks, env_state, actions)
        cur = sum(
            actions[a] * env_state.env_state.cur_player_idx[..., i].astype(jnp.int32)
            for i, a in enumerate(env.agents)
        )
        ids, mask = tokenize_fn(new_env_state.env_state, env_state.env_state, cur)
        emb = encode_text(ids, mask)
        state_emb = {a: emb[i] for i, a in enumerate(env.agents)}
        ts = Timestep(
            state_emb=state_emb,
            actions=actions,
            rewards=rewards,
            dones=dones,
            avail_actions=avail,
        )
        return (new_env_state, env_state, cur, rng), ts

    _, traj = jax.lax.scan(step, (env_state, env_state, init_actions, rng), None, num_steps)
    return traj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def envs_from_config(config):
    """Build one LogWrapper-wrapped env per player count in PLAYER_COUNTS."""
    envs = []
    names = []
    for n in config["PLAYER_COUNTS"]:
        e = make(config["ENV_NAME"], num_agents=int(n))
        envs.append(LogWrapper(e))
        names.append(f"{config['ENV_NAME']}_{n}p")
    return envs, names


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))
    alg_name = config.get("ALG_NAME", "r3d2_multitask_rnn_hanabi")

    envs, env_names = envs_from_config(copy.deepcopy(config))
    tokenizer, hf_model = load_bert(config["BERT_MODEL_DIR"])
    bert_module = hf_model.module
    bert_params = hf_model.params
    bert_hidden = hf_model.config.hidden_size

    tokenize_fns = [
        make_tokenize_fn(
            e, tokenizer,
            max_obs_tokens=int(config["MAX_OBS_TOKENS"]),
            include_belief=bool(config.get("INCLUDE_BELIEF", False)),
        )
        for e in envs
    ]

    action_embs = []
    for e in envs:
        wrapped = CTRolloutManager(e, batch_size=1)
        ae = precompute_action_embedding(
            e, tokenizer, hf_model,
            max_action_tokens=int(config["MAX_ACTION_TOKENS"]),
            max_action_space=wrapped.max_action_space,
        )
        action_embs.append(ae)
    print("action_embs sizes:", [ae.shape for ae in action_embs])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), "MULTITASK", f"jax_{jax.__version__}"],
        name=f"{alg_name}_" + "_".join(env_names),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train_multitask(
        config, envs, bert_module, bert_params, tokenize_fns, bert_hidden, action_embs,
    )
    out = train_fn(rng)

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        train_state = out["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], alg_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))
        save_params(
            train_state.params,
            os.path.join(save_dir, f"seed{config['SEED']}.safetensors"),
        )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    single_run(config)


if __name__ == "__main__":
    main()
