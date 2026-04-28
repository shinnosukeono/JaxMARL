"""
Multi-GPU reproduction launcher for the Hanabi baselines.

Differences from `multi_gpu_run.py`:
  * Initialises a single wandb run that aggregates per-seed metrics under
    `rng{seed}/...` prefixes (the `WANDB_LOG_ALL_SEEDS=True` path each
    `make_train` already supports).
  * Saves checkpoints under SAVE_PATH if set, one safetensors per seed.

CLI:
    .venv/bin/python baselines/QLearning/launch_repro.py \\
        --algo r2d2_publ \\
        --config baselines/QLearning/config/alg/r2d2_publ_rnn_hanabi.yaml \\
        --num_seeds 3 \\
        --project r3d2-jaxmarl-repro \\
        --run_name r2d2_publ_2p \\
        [--total_timesteps 5e8] [--save_path models/]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import jax
import yaml


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from multi_gpu_run import _build_train, run_multigpu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True,
                    choices=["r2d2_publ", "r2d2_text", "r3d2",
                             "r3d2_multitask", "obl", "obl_belief"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_seeds", type=int, default=jax.local_device_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="r3d2-jaxmarl-repro")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--total_timesteps", type=float, default=None)
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["SEED"] = args.seed
    cfg["NUM_SEEDS"] = args.num_seeds
    if args.total_timesteps is not None:
        cfg["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.save_path is not None:
        cfg["SAVE_PATH"] = args.save_path
    cfg["WANDB_MODE"] = "online"
    cfg["WANDB_LOG_ALL_SEEDS"] = True
    cfg["ENTITY"] = args.entity or ""
    cfg["PROJECT"] = args.project

    run_name = args.run_name or f"{args.algo}_seed{args.seed}_n{args.num_seeds}"

    # Initialize wandb BEFORE building train fn so jax.debug.callback inside
    # train can log to it.
    import wandb
    wandb.init(
        project=cfg["PROJECT"],
        entity=cfg["ENTITY"] or None,
        name=run_name,
        config=cfg,
        tags=[args.algo, f"seeds={args.num_seeds}"],
    )

    train_fn, label, extras = _build_train(args.algo, cfg)
    print(f"[launch_repro] {label} | seeds={args.num_seeds} | "
          f"total_timesteps={cfg['TOTAL_TIMESTEPS']:.0f} | "
          f"extras={len(extras)}", flush=True)

    t0 = time.time()
    out = run_multigpu(train_fn, args.seed, args.num_seeds, extras=extras)
    elapsed = time.time() - t0
    print(f"[launch_repro] training finished in {elapsed:.1f}s")

    # Save params per seed if requested.
    if cfg.get("SAVE_PATH"):
        from jaxmarl.wrappers.baselines import save_params
        os.makedirs(cfg["SAVE_PATH"], exist_ok=True)
        train_state = out.get("runner_state", out)[0]
        # train_state.params has leading axes (n_devices, seeds_per_device).
        # Flatten to one safetensors per seed.
        params = train_state.params
        n_dev = jax.local_device_count()
        spd = args.num_seeds // n_dev
        for d in range(n_dev):
            for s in range(spd):
                idx = d * spd + s
                seed_params = jax.tree.map(lambda x, d=d, s=s: x[d, s], params)
                fname = os.path.join(
                    cfg["SAVE_PATH"], f"{label}_seed{args.seed}_run{idx}.safetensors"
                )
                save_params(seed_params, fname)
                print(f"  saved {fname}")

    wandb.finish()


if __name__ == "__main__":
    main()
