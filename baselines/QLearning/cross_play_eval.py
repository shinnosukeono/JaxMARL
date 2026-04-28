"""
Cross-play evaluation harness for the Hanabi baselines.

Pairs two saved policies as agent_0 and agent_1 (or rotates across `num_agents`
seats), runs `NUM_GAMES` episodes, and reports mean score / perfect-game rate.
Mirrors `pyhanabi/scripts/launch_cross_play.sh` from the upstream R3D2 repo.

Supported policy types
----------------------

* `r2d2_publ`  : params from r2d2_publ_rnn_hanabi.py
* `r2d2_text`  : params + frozen BERT, from r2d2_text_rnn_hanabi.py
* `r3d2`       : params + frozen BERT + action_emb, from r3d2_rnn_hanabi.py
* `random`     : uniform over legal actions (sanity baseline)

Each policy is a class implementing `init_carry(batch)` and
`act(carry, obs_packet, avail) -> (carry, action_int)`. `obs_packet` is whatever
the policy needs (numerical obs, encoded state, etc.) and is built by the
harness depending on the policy's declared `obs_kind`.

Usage (programmatic):

    from cross_play_eval import (
        Policy, RandomPolicy, R2D2PublPolicy, R2D2TextPolicy, R3D2Policy,
        cross_play,
    )
    p1 = R2D2PublPolicy.from_safetensors("models/.../seed0.safetensors", env)
    p2 = R3D2Policy.from_safetensors("models/.../seed0.safetensors", env, bert_dir, ...)
    out = cross_play([p1, p2], env, num_games=1000, seed=0)
    print(out)  # {'mean_score': ..., 'perfect_rate': ..., 'pair': ('r2d2_publ', 'r3d2')}
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

# --- local imports (this file lives in baselines/QLearning) ---
from text_obs import make_tokenize_fn  # noqa: E402

# We import the network classes lazily inside policy ctors to avoid circular deps.


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    name: str
    obs_kind: str  # 'numerical' | 'text'
    init_carry: Callable[[int], Any]
    act: Callable[[Any, Dict[str, jnp.ndarray]], jnp.ndarray]
    # obs_kind tells the harness whether to feed reordered numerical obs (for
    # R2D2-publ) or encoded text obs (for r2d2_text / r3d2).


class RandomPolicy:
    """Uniform random over legal actions."""

    def __init__(self, env):
        self.env = env

    def to_policy(self) -> Policy:
        def init_carry(batch):
            return None

        def act(carry, packet):
            avail = packet["avail"]  # (batch, num_actions)
            rng = packet["rng"]
            # sample uniformly from legal moves
            probs = avail / jnp.clip(avail.sum(axis=-1, keepdims=True), a_min=1.0)
            sampled = jax.vmap(
                lambda r, p: jax.random.choice(r, jnp.arange(p.shape[-1]), p=p)
            )(jax.random.split(rng, avail.shape[0]), probs)
            return carry, sampled

        return Policy(
            name="random",
            obs_kind="numerical",
            init_carry=init_carry,
            act=act,
        )


class R2D2PublPolicy:
    """Loads R2D2 publ-LSTM params from safetensors and serves them."""

    def __init__(self, params, env, hidden_dim: int):
        from r2d2_publ_rnn_hanabi import (
            PublicLSTMQNetwork, ScannedLSTM,
            hanabi_publ_split, hanabi_feature_widths, reorder_obs_for_split,
        )
        wrapped_action_dim = self._max_action_space(env)
        priv_dim, _ = hanabi_publ_split(env)
        self._network = PublicLSTMQNetwork(
            action_dim=wrapped_action_dim,
            hidden_dim=hidden_dim,
            publ_split=priv_dim,
        )
        self._params = params
        self._hidden_dim = hidden_dim
        self._widths = hanabi_feature_widths(env)
        self._reorder = reorder_obs_for_split
        self._init_carry_fn = ScannedLSTM.initialize_carry

    @staticmethod
    def _max_action_space(env):
        from jaxmarl.wrappers.baselines import CTRolloutManager
        return CTRolloutManager(env, batch_size=1).max_action_space

    @classmethod
    def from_safetensors(cls, path: str, env, hidden_dim: int):
        from jaxmarl.wrappers.baselines import load_params
        params = load_params(path)
        return cls(params, env, hidden_dim)

    def to_policy(self) -> Policy:
        def init_carry(batch):
            return self._init_carry_fn(self._hidden_dim, batch)

        def act(carry, packet):
            obs = packet["obs_reordered"]   # (batch, obs_dim)
            done = packet["done"]           # (batch,)
            obs_t = obs[None, :]            # add time dim
            done_t = done[None, :].astype(jnp.float32)
            new_carry, q_vals = self._network.apply(
                self._params, carry, obs_t, done_t
            )
            q_vals = q_vals.squeeze(axis=0)
            avail = packet["avail"]
            actions = jnp.argmax(q_vals - (1 - avail) * 1e10, axis=-1)
            return new_carry, actions

        return Policy(
            name="r2d2_publ",
            obs_kind="numerical",
            init_carry=init_carry,
            act=act,
        )


class _TextPolicyBase:
    """Shared helpers for text-obs policies."""

    def __init__(self, params, env, hidden_dim: int, bert_dir: str,
                 max_obs_tokens: int, include_belief: bool = False):
        from transformers import AutoTokenizer, FlaxBertModel
        self._tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self._hf = FlaxBertModel.from_pretrained(bert_dir)
        self._bert_module = self._hf.module
        self._bert_params = self._hf.params
        self._bert_hidden = self._hf.config.hidden_size
        self._tokenize_fn = make_tokenize_fn(
            env, self._tokenizer,
            max_obs_tokens=max_obs_tokens,
            include_belief=include_belief,
        )
        self._params = params
        self._hidden_dim = hidden_dim
        self._env = env

    def _encode_text(self, input_ids, attention_mask):
        leading = input_ids.shape[:-1]
        flat_ids = input_ids.reshape(-1, input_ids.shape[-1])
        flat_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        out = self._bert_module.apply(
            {"params": self._bert_params},
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        h = out.last_hidden_state
        m = flat_mask[..., None].astype(h.dtype)
        h = (h * m).sum(axis=1) / jnp.clip(m.sum(axis=1), a_min=1.0)
        return h.reshape(*leading, self._bert_hidden)

    def encode_state(self, env_state, prev_env_state, last_actions, agent_idx: int):
        """Encode text obs for a specific agent. env_states are LogEnvState."""
        new_inner = env_state.env_state
        old_inner = prev_env_state.env_state
        ids, mask = self._tokenize_fn(new_inner, old_inner, last_actions)
        emb = self._encode_text(ids, mask)  # (num_agents, B, hidden)
        return emb[agent_idx]


class R2D2TextPolicy(_TextPolicyBase):
    def __init__(self, params, env, hidden_dim, bert_dir, max_obs_tokens,
                 include_belief=False):
        super().__init__(params, env, hidden_dim, bert_dir, max_obs_tokens, include_belief)
        from r2d2_text_rnn_hanabi import TextRNNQNetwork, ScannedLSTM
        from jaxmarl.wrappers.baselines import CTRolloutManager
        self._network = TextRNNQNetwork(
            action_dim=CTRolloutManager(env, batch_size=1).max_action_space,
            hidden_dim=hidden_dim,
        )
        self._init_carry_fn = ScannedLSTM.initialize_carry

    @classmethod
    def from_safetensors(cls, path, env, hidden_dim, bert_dir, max_obs_tokens,
                         include_belief=False):
        from jaxmarl.wrappers.baselines import load_params
        params = load_params(path)
        return cls(params, env, hidden_dim, bert_dir, max_obs_tokens, include_belief)

    def to_policy(self) -> Policy:
        def init_carry(batch):
            return self._init_carry_fn(self._hidden_dim, batch)

        def act(carry, packet):
            state_emb = packet["state_emb"][None, :]
            done = packet["done"][None, :].astype(jnp.float32)
            new_carry, q_vals = self._network.apply(self._params, carry, state_emb, done)
            q_vals = q_vals.squeeze(axis=0)
            avail = packet["avail"]
            return new_carry, jnp.argmax(q_vals - (1 - avail) * 1e10, axis=-1)

        return Policy(
            name="r2d2_text",
            obs_kind="text",
            init_carry=init_carry,
            act=act,
        )


class R3D2Policy(_TextPolicyBase):
    def __init__(self, params, env, hidden_dim, bert_dir, max_obs_tokens,
                 max_action_tokens, include_belief=False):
        super().__init__(params, env, hidden_dim, bert_dir, max_obs_tokens, include_belief)
        from r3d2_rnn_hanabi import R3D2Net, ScannedLSTM, precompute_action_embedding
        from jaxmarl.wrappers.baselines import CTRolloutManager
        self._network = R3D2Net(
            hidden_dim=hidden_dim,
            bert_hidden=self._bert_hidden,
        )
        self._action_emb = precompute_action_embedding(
            env, self._tokenizer, self._hf,
            max_action_tokens=max_action_tokens,
            max_action_space=CTRolloutManager(env, batch_size=1).max_action_space,
        )
        self._init_carry_fn = ScannedLSTM.initialize_carry

    @classmethod
    def from_safetensors(cls, path, env, hidden_dim, bert_dir, max_obs_tokens,
                         max_action_tokens, include_belief=False):
        from jaxmarl.wrappers.baselines import load_params
        params = load_params(path)
        return cls(params, env, hidden_dim, bert_dir, max_obs_tokens,
                   max_action_tokens, include_belief)

    def to_policy(self) -> Policy:
        def init_carry(batch):
            return self._init_carry_fn(self._hidden_dim, batch)

        def act(carry, packet):
            state_emb = packet["state_emb"][None, :]
            done = packet["done"][None, :].astype(jnp.float32)
            new_carry, q_vals = self._network.apply(
                self._params, carry, state_emb, done, self._action_emb
            )
            q_vals = q_vals.squeeze(axis=0)
            avail = packet["avail"]
            return new_carry, jnp.argmax(q_vals - (1 - avail) * 1e10, axis=-1)

        return Policy(
            name="r3d2",
            obs_kind="text",
            init_carry=init_carry,
            act=act,
        )


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


def cross_play(policies: List, env, num_games: int = 1000, seed: int = 0,
               max_steps: int = 80) -> Dict[str, float]:
    """Run `num_games` episodes with the given policy lineup (one per agent seat).

    Returns a dict with keys:
      - mean_score, perfect_rate, mean_episode_length, pair
    """
    from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
    if not isinstance(env, LogWrapper):
        env = LogWrapper(env)
    wrapped = CTRolloutManager(env, batch_size=num_games)
    pol_objs = [p.to_policy() if not isinstance(p, Policy) else p for p in policies]
    assert len(pol_objs) == env.num_agents, (
        f"Got {len(pol_objs)} policies for {env.num_agents}-agent env"
    )

    # Pre-build the obs encoders we'll need.
    needs_text = any(p.obs_kind == "text" for p in pol_objs)
    text_policies = {i: p for i, p in enumerate(pol_objs) if p.obs_kind == "text"}

    rng = jax.random.PRNGKey(seed)
    obs, env_state = wrapped.batch_reset(rng)
    prev_state = env_state
    last_actions = jnp.full((num_games,), env.num_moves - 1, dtype=jnp.int32)

    carries = [p.init_carry(num_games) for p in pol_objs]

    # We'll need numerical-reordered obs for R2D2-publ.
    from r2d2_publ_rnn_hanabi import (
        hanabi_feature_widths, reorder_obs_for_split,
    )
    widths = hanabi_feature_widths(env)

    def per_agent_packet(i, env_state, prev_state, last_actions, obs_dict, dones, rng_):
        avail = wrapped.get_valid_actions(env_state.env_state)[env.agents[i]]
        packet = {"avail": avail, "done": dones[env.agents[i]]}
        if pol_objs[i].obs_kind == "numerical":
            o = reorder_obs_for_split(obs_dict[env.agents[i]], widths)
            packet["obs_reordered"] = o
            packet["rng"] = rng_
        elif pol_objs[i].obs_kind == "text":
            tp = text_policies[i]
            packet["state_emb"] = tp.encode_state(
                env_state, prev_state, last_actions, i
            )
            packet["rng"] = rng_
        return packet

    cum_score = jnp.zeros(num_games)
    done_mask = jnp.zeros(num_games, dtype=bool)
    ep_len = jnp.zeros(num_games, dtype=jnp.int32)

    dones = {a: jnp.zeros(num_games, dtype=bool) for a in env.agents + ["__all__"]}
    for step in range(max_steps):
        rng, *agent_rngs = jax.random.split(rng, env.num_agents + 1)
        actions_per_agent = []
        for i, p in enumerate(pol_objs):
            packet = per_agent_packet(
                i, env_state, prev_state, last_actions, obs, dones, agent_rngs[i]
            )
            carries[i], a = p.act(carries[i], packet)
            actions_per_agent.append(a)
        action_dict = {env.agents[i]: actions_per_agent[i] for i in range(env.num_agents)}

        # which agent's action gets executed
        cur_idx = env_state.env_state.cur_player_idx
        executed = sum(
            actions_per_agent[i] * cur_idx[..., i].astype(jnp.int32)
            for i in range(env.num_agents)
        )
        prev_state = env_state
        rng, k = jax.random.split(rng)
        obs, env_state, rewards, dones, info = wrapped.batch_step(
            k, env_state, action_dict
        )
        last_actions = executed

        rew = rewards[env.agents[0]]
        # accumulate score only for envs that haven't terminated yet.
        cum_score = cum_score + jnp.where(done_mask, 0.0, rew)
        ep_len = ep_len + (~done_mask).astype(jnp.int32)
        done_mask = done_mask | dones["__all__"]
        if bool(done_mask.all()):
            break

    mean_score = float(cum_score.mean())
    perfect_rate = float((cum_score >= 25).mean())
    mean_len = float(ep_len.mean())
    return {
        "mean_score": mean_score,
        "perfect_rate": perfect_rate,
        "mean_episode_length": mean_len,
        "pair": tuple(p.name for p in pol_objs),
    }


# ---------------------------------------------------------------------------
# CLI: cross-play between a list of checkpoints
# ---------------------------------------------------------------------------


def main():
    """Tiny CLI: pass a YAML describing the agents to load and run all pairs."""
    import argparse
    import yaml
    from itertools import product
    from jaxmarl import make as make_env

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML describing players and num_games")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    env_kwargs = cfg.get("env_kwargs", {"num_agents": 2})
    env = make_env(cfg.get("env_name", "hanabi"), **env_kwargs)

    # Build players
    players = []
    for spec in cfg["players"]:
        kind = spec["kind"]
        if kind == "random":
            players.append(RandomPolicy(env))
        elif kind == "r2d2_publ":
            players.append(R2D2PublPolicy.from_safetensors(
                spec["params"], env, hidden_dim=spec["hidden_dim"]
            ))
        elif kind == "r2d2_text":
            players.append(R2D2TextPolicy.from_safetensors(
                spec["params"], env,
                hidden_dim=spec["hidden_dim"],
                bert_dir=spec["bert_dir"],
                max_obs_tokens=spec.get("max_obs_tokens", 256),
                include_belief=spec.get("include_belief", False),
            ))
        elif kind == "r3d2":
            players.append(R3D2Policy.from_safetensors(
                spec["params"], env,
                hidden_dim=spec["hidden_dim"],
                bert_dir=spec["bert_dir"],
                max_obs_tokens=spec.get("max_obs_tokens", 256),
                max_action_tokens=spec.get("max_action_tokens", 12),
                include_belief=spec.get("include_belief", False),
            ))
        else:
            raise ValueError(f"unknown player kind {kind}")

    # All ordered pairs (P_i as agent_0, P_j as agent_1).
    print("idx  | mean_score | perfect | mean_len | pair")
    for i, j in product(range(len(players)), repeat=env.num_agents):
        if env.num_agents != 2:
            # For >2 agents, just rotate: lineup = [players[i], players[j], ...]
            lineup = [players[(i + k) % len(players)] for k in range(env.num_agents)]
        else:
            lineup = [players[i], players[j]]
        out = cross_play(lineup, env, num_games=cfg.get("num_games", 200), seed=args.seed)
        print(f"({i},{j}) | {out['mean_score']:.3f} | {out['perfect_rate']:.3f} | "
              f"{out['mean_episode_length']:.1f} | {out['pair']}")


if __name__ == "__main__":
    main()
