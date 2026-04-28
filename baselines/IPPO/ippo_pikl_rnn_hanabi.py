"""
PPO + π-KL on Hanabi.

A drop-in extension of `ippo_rnn_hanabi.py` that adds a KL-divergence penalty
toward an LLM prior over actions, matching the formulation in the R3D2 repo's
`pyhanabi/ppo.py:170-183`:

    no prior:   loss = value_loss + policy_loss - ent_weight * entropy
    with prior: kl_loss = -sum_a softmax(legal_prior_logits) -> log -> dot pi
                loss   = value_loss + policy_loss + pikl_lambda * (kl_loss - entropy)

The prior is keyed by a "history string" of the most recent move (see
`baselines/QLearning/llm_prior.py` for format), looked up per env-step via
`jax.pure_callback` and threaded through the transition dataclass.
"""

import functools
import os
import sys
from typing import Any, Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

# llm_prior lives under baselines/QLearning; add that to sys.path so this
# file is runnable as a single script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QLEARN_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "QLearning"))
if _QLEARN_DIR not in sys.path:
    sys.path.insert(0, _QLEARN_DIR)
from llm_prior import load_prior, make_prior_lookup_fn  # noqa: E402


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    prior_logits: jnp.ndarray  # (NUM_ACTORS, num_moves) shared across players


def batchify(x, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    # Load LLM prior (optional). When LLM_PRIOR_PATH is None, we skip the KL
    # term entirely and behave like vanilla IPPO.
    prior_path = config.get("LLM_PRIOR_PATH", None)
    if prior_path is not None:
        prior = load_prior(env, prior_path)
        prior_lookup_fn = make_prior_lookup_fn(env, prior)
        print(f"[ippo_pikl] loaded prior from {prior_path}, "
              f"{len(prior)} histories, num_moves={env.num_moves}")
    else:
        prior_lookup_fn = None
        print("[ippo_pikl] no LLM_PRIOR_PATH set -> π-KL disabled")

    pikl_lambda = float(config.get("PIKL_LAMBDA", 0.0))
    pikl_beta = float(config.get("PIKL_BETA", 1.0))  # softmax temperature

    def linear_schedule(count):
        frac = 1.0 - (
            count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def gather_prior(env_state, prev_env_state, last_actions):
        """Per-env prior logits, then broadcast to all NUM_ACTORS."""
        if prior_lookup_fn is None:
            return jnp.zeros((config["NUM_ACTORS"], env.num_moves))
        # env_state.env_state is the inner Hanabi state (LogEnvState wraps it).
        new_inner = env_state.env_state
        old_inner = prev_env_state.env_state
        # shape (NUM_ENVS, num_moves)
        prior_per_env = prior_lookup_fn(new_inner, old_inner, last_actions)
        # All agents share the same history-keyed prior in 2p; broadcast.
        prior_per_actor = jnp.tile(prior_per_env[None, :, :], (env.num_agents, 1, 1))
        return prior_per_actor.reshape(config["NUM_ACTORS"], env.num_moves)

    def train(rng):
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    prev_state,
                    last_actions,  # (NUM_ENVS,) — most recent action *that produced env_state*
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                # Prior logits for the current state (depends on previous transition).
                prior_logits = gather_prior(env_state, prev_state, last_actions)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, new_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Determine which agent's action was actually executed (for prior history).
                cur_idx = env_state.env_state.cur_player_idx  # (NUM_ENVS, num_agents)
                acted_action = sum(
                    env_act[a] * cur_idx[..., i].astype(jnp.int32)
                    for i, a in enumerate(env.agents)
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    prior_logits,
                )
                runner_state = (
                    train_state, new_env_state, obsv, done_batch, hstate,
                    env_state, acted_action, rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-4]  # hstate is at -4 in the new tuple
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_state, env_state, last_obs, last_done, hstate,
                prev_state, last_actions, rng,
            ) = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = transition.global_done
                    value = transition.value
                    reward = transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params, init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        # π-KL term: KL(π || softmax(beta * legal_prior_logits)).
                        # Mirrors pyhanabi/ppo.py:170-183 except we do not subtract
                        # ent_weight*entropy when the prior is active (prior already
                        # provides regularisation via -entropy in pikl term).
                        # `prior_logits` is (T_seq, B_actors, num_moves).
                        prior_logits = traj_batch.prior_logits * pikl_beta
                        # Mask illegal actions.
                        prior_logits = prior_logits - (1 - traj_batch.avail_actions) * 1e10
                        log_prior = jax.nn.log_softmax(prior_logits, axis=-1)
                        # cross-entropy of prior under pi (matches `kl_loss` in original).
                        pi_probs = jax.nn.softmax(pi.logits, axis=-1)
                        kl_loss = -(log_prior * pi_probs).sum(-1).mean()

                        if pikl_lambda > 0 and prior_lookup_fn is not None:
                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                + pikl_lambda * (kl_loss - entropy)
                            )
                        else:
                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            kl_loss,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state, init_hstate.squeeze(), traj_batch, advantages, targets, rng,
                )
                return update_state, total_loss

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info_means = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info_means[0],
                "value_loss": loss_info_means[1][0],
                "actor_loss": loss_info_means[1][1],
                "entropy": loss_info_means[1][2],
                "ratio": loss_info_means[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info_means[1][4],
                "clip_frac": loss_info_means[1][5],
                "kl_loss": loss_info_means[1][6],
            }
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                    },
                    step=metric["update_steps"],
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (
                train_state, env_state, last_obs, last_done, hstate,
                prev_state, last_actions, rng,
            )
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        # Initial prev_state = initial env_state, initial action = noop (last action idx).
        init_actions = jnp.full((config["NUM_ENVS"],), env.num_moves - 1, dtype=jnp.int32)
        runner_state = (
            train_state, env_state, obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate, env_state, init_actions, _rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_pikl_rnn_hanabi")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "PIKL", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    train_jit(rng)


if __name__ == "__main__":
    main()
