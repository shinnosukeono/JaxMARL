"""
Multi-GPU launcher for the Hanabi baselines in this folder.

Strategy: pmap-of-vmap over RNG seeds. NUM_SEEDS must be divisible by the
number of visible GPUs. Each GPU runs `seeds_per_device` independent seeds
in parallel; the training function itself is unchanged.

Usage from another script:

    from multi_gpu_run import run_multigpu
    from r2d2_publ_rnn_hanabi import make_train, env_from_config

    env, _ = env_from_config(config)
    out = run_multigpu(make_train(config, env), config["SEED"], config["NUM_SEEDS"])

CLI:

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    .venv/bin/python baselines/QLearning/multi_gpu_run.py \
        --algo r2d2_publ \
        --config baselines/QLearning/config/alg/r2d2_publ_rnn_hanabi.yaml \
        --num_seeds 8

Each algorithm exposes `make_train(...)` whose first arg is `config` and the
remaining args (env, BERT, etc.) are environment- or pretrained-asset-specific.
The CLI knows the binding for each `--algo` value below.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# pmap-vmap wrapper
# ---------------------------------------------------------------------------


def run_multigpu(train_fn: Callable, seed: int, num_seeds: int):
    """pmap(vmap(train_fn))(rngs) over `num_seeds` parallel seeds.

    Args:
        train_fn : function taking a single (2,) PRNGKey and returning a pytree.
        seed     : root PRNG key seed.
        num_seeds: total seeds; must be a multiple of jax.local_device_count().

    Returns the pytree output of train_fn, with leading axes (num_devices,
    seeds_per_device, ...).
    """
    n_devices = jax.local_device_count()
    if num_seeds % n_devices != 0:
        raise ValueError(
            f"NUM_SEEDS={num_seeds} must be divisible by local_device_count={n_devices}. "
            f"Either reduce CUDA_VISIBLE_DEVICES or pick NUM_SEEDS in "
            f"{[k * n_devices for k in range(1, 5)]}."
        )
    seeds_per_device = num_seeds // n_devices

    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, num_seeds).reshape(n_devices, seeds_per_device, 2)

    print(f"[multi_gpu] {n_devices} devices x {seeds_per_device} seeds/device "
          f"= {num_seeds} total seeds")
    t0 = time.time()
    pmapped = jax.pmap(jax.vmap(train_fn))
    out = jax.block_until_ready(pmapped(rngs))
    print(f"[multi_gpu] done in {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# CLI: dispatcher per algorithm
# ---------------------------------------------------------------------------


def _build_train(algo: str, config: dict):
    """Return (train_fn, label) for the requested algorithm."""
    if algo == "r2d2_publ":
        from r2d2_publ_rnn_hanabi import make_train, env_from_config
        env, name = env_from_config(copy.deepcopy(config))
        return make_train(config, env), f"r2d2_publ_{name}"

    if algo == "r2d2_text":
        from r2d2_text_rnn_hanabi import make_train, env_from_config, load_bert
        from text_obs import make_tokenize_fn
        env, name = env_from_config(copy.deepcopy(config))
        tok, hf = load_bert(config["BERT_MODEL_DIR"])
        tok_fn = make_tokenize_fn(env, tok, max_obs_tokens=int(config["MAX_OBS_TOKENS"]))
        return (
            make_train(config, env, hf.module, hf.params, tok_fn, hf.config.hidden_size),
            f"r2d2_text_{name}",
        )

    if algo == "r3d2":
        from r3d2_rnn_hanabi import (
            make_train, env_from_config, load_bert, precompute_action_embedding,
        )
        from text_obs import make_tokenize_fn
        from jaxmarl.wrappers.baselines import CTRolloutManager
        env, name = env_from_config(copy.deepcopy(config))
        tok, hf = load_bert(config["BERT_MODEL_DIR"])
        tok_fn = make_tokenize_fn(env, tok, max_obs_tokens=int(config["MAX_OBS_TOKENS"]))
        wrapped = CTRolloutManager(env, batch_size=1)
        ae = precompute_action_embedding(
            env, tok, hf,
            max_action_tokens=int(config["MAX_ACTION_TOKENS"]),
            max_action_space=wrapped.max_action_space,
        )
        return (
            make_train(config, env, hf.module, hf.params, tok_fn, hf.config.hidden_size, ae),
            f"r3d2_{name}",
        )

    if algo == "obl":
        from obl_rnn_hanabi import make_train, env_from_config, _maybe_load
        env, name = env_from_config(copy.deepcopy(config))
        bp_params = _maybe_load(config.get("BP_CHECKPOINT"))
        belief_params = _maybe_load(config.get("BELIEF_CHECKPOINT"))
        return (
            make_train(config, env, bp_params, belief_params, int(config["HIDDEN_SIZE"])),
            f"obl_{name}",
        )

    if algo == "obl_belief":
        from obl_train_belief import make_train, env_from_config
        env, name = env_from_config(copy.deepcopy(config))
        train_fn, _ = make_train(config, env)
        return train_fn, f"obl_belief_{name}"

    if algo == "r3d2_multitask":
        # NOTE: multi-task uses a Python-driven rotation across envs and
        # therefore cannot be vmap'd naively. We pmap over seeds only — each
        # device runs the entire round-robin training.
        from r3d2_multitask_rnn_hanabi import (
            make_train_multitask, envs_from_config, load_bert, precompute_action_embedding,
        )
        from text_obs import make_tokenize_fn
        from jaxmarl.wrappers.baselines import CTRolloutManager
        envs, names = envs_from_config(copy.deepcopy(config))
        tok, hf = load_bert(config["BERT_MODEL_DIR"])
        tok_fns = [
            make_tokenize_fn(e, tok, max_obs_tokens=int(config["MAX_OBS_TOKENS"]))
            for e in envs
        ]
        action_embs = [
            precompute_action_embedding(
                e, tok, hf,
                max_action_tokens=int(config["MAX_ACTION_TOKENS"]),
                max_action_space=CTRolloutManager(e, batch_size=1).max_action_space,
            )
            for e in envs
        ]
        return (
            make_train_multitask(
                config, envs, hf.module, hf.params, tok_fns, hf.config.hidden_size, action_embs,
            ),
            f"r3d2_multitask_{'_'.join(names)}",
        )

    raise ValueError(f"unknown --algo {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True,
                    choices=["r2d2_publ", "r2d2_text", "r3d2",
                             "r3d2_multitask", "obl", "obl_belief"])
    ap.add_argument("--config", required=True, help="path to alg yaml")
    ap.add_argument("--num_seeds", type=int, default=jax.local_device_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["SEED"] = args.seed
    cfg["NUM_SEEDS"] = args.num_seeds
    if "WANDB_MODE" not in cfg:
        cfg["WANDB_MODE"] = "disabled"

    train_fn, label = _build_train(args.algo, cfg)
    print(f"[multi_gpu] launching {label}")
    out = run_multigpu(train_fn, args.seed, args.num_seeds)
    print(f"[multi_gpu] {label} done; output keys: {list(out.keys())}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
