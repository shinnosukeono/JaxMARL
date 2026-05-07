"""Verify our publ-LSTM port against the upstream R3D2 reference.

Three checks:

  1) Slice equivalence — `split_obs_publ_priv(obs, hanabi_hands_dim(env))`
     must equal `(obs, obs[..., 125:])` for 2p Hanabi (the upstream cut at
     `pyhanabi/r2d2.py:188` and `obl_r2d2_agent.py:95`).

  2) Forward-pass equivalence under OBLAgentR2D2 — feeding both slices
     through the upstream `OBLAgentR2D2` (random params, identical between
     runs) must yield bit-identical advantages and carry. Proves our slice
     composes downstream the same way upstream's hard-coded slice does.

  3) Multi-layer LSTM port — confirm `PublicLSTMQNetwork` now has two
     LSTM-layer parameter blocks (`l0`, `l1`), the carry from
     `MultiLayerScannedLSTM.initialize_carry` is shape
     `(num_layers, *batch, hidden)`, and a forward pass on a real obs
     yields finite Q-values of the expected shape.

No real OBL safetensors needed — only well-typed param trees so we can
exercise forward passes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "baselines", "QLearning"))

from jaxmarl import make  # noqa: E402
from jaxmarl.wrappers.baselines import CTRolloutManager  # noqa: E402
from jaxmarl.environments.hanabi.pretrained.obl_r2d2_agent import (  # noqa: E402
    OBLAgentR2D2,
)
from r2d2_publ_rnn_hanabi import (  # noqa: E402
    PublicLSTMQNetwork,
    MultiLayerScannedLSTM,
    hanabi_hands_dim,
    split_obs_publ_priv,
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    rng = jax.random.PRNGKey(0)
    env = make("hanabi", num_agents=2)
    wrapped = CTRolloutManager(env, batch_size=4)

    rng, k = jax.random.split(rng)
    obs_dict, _ = wrapped.batch_reset(k)
    obs = obs_dict[env.agents[0]]  # (B, obs_dim)
    obs_dim = int(obs.shape[-1])
    B = int(obs.shape[0])

    print("env: hanabi 2p")
    print(f"obs shape: {obs.shape}")
    print(
        f"hands_n_feats={env.hands_n_feats} (encoder = "
        f"(N-1)*hand*colors*ranks + N missing-card flags)"
    )

    # ------- 1. Slice equivalence -------
    hands_dim = hanabi_hands_dim(env)
    print(f"hanabi_hands_dim(env) = {hands_dim}; upstream constant = 125")
    _check(hands_dim == 125, f"hands_dim={hands_dim} != upstream 125 for 2p")

    priv_ours, publ_ours = split_obs_publ_priv(obs, hands_dim)
    priv_ref, publ_ref = obs, obs[..., 125:]

    np.testing.assert_array_equal(np.asarray(priv_ours), np.asarray(priv_ref))
    np.testing.assert_array_equal(np.asarray(publ_ours), np.asarray(publ_ref))
    print(
        f"[OK] slice equivalence: priv {priv_ours.shape}, publ "
        f"{publ_ours.shape}"
    )

    # ------- 2. Forward-pass equivalence through OBLAgentR2D2 -------
    obl = OBLAgentR2D2()
    rng, k_carry = jax.random.split(rng)
    init_carry = obl.initialize_carry(k_carry, batch_dims=(B,))
    rng, k_p = jax.random.split(rng)
    init_priv = jnp.zeros((B, obs_dim), dtype=obs.dtype)
    init_publ = jnp.zeros((B, obs_dim - hands_dim), dtype=obs.dtype)
    obl_params = obl.init(k_p, init_carry, (init_priv, init_publ))

    carry_ref, adv_ref = obl.apply(
        obl_params, init_carry, (priv_ref, publ_ref)
    )
    carry_ours, adv_ours = obl.apply(
        obl_params, init_carry, (priv_ours, publ_ours)
    )

    np.testing.assert_array_equal(np.asarray(adv_ref), np.asarray(adv_ours))
    for cr, co in zip(
        jax.tree_util.tree_leaves(carry_ref),
        jax.tree_util.tree_leaves(carry_ours),
    ):
        np.testing.assert_array_equal(np.asarray(cr), np.asarray(co))
    print(
        f"[OK] OBLAgentR2D2 forward equivalence: adv shape {adv_ref.shape}, "
        f"carry leaves {[l.shape for l in jax.tree_util.tree_leaves(carry_ref)]}"
    )

    # ------- 3. Multi-layer LSTM port -------
    num_lstm_layer = 2
    hidden_dim = 512
    publ_dim = obs_dim - hands_dim

    # 3a. Carry shape from our static helper.
    init_hs = MultiLayerScannedLSTM.initialize_carry(
        hidden_dim, num_lstm_layer, B
    )
    leaves = jax.tree_util.tree_leaves(init_hs)
    print(f"MultiLayerScannedLSTM carry leaves: {[l.shape for l in leaves]}")
    _check(
        len(leaves) == 2
        and all(tuple(l.shape) == (num_lstm_layer, B, hidden_dim) for l in leaves),
        f"carry layout != (c_stack, h_stack) of shape "
        f"({num_lstm_layer}, {B}, {hidden_dim}); got {[l.shape for l in leaves]}",
    )
    print(
        f"[OK] carry shape: ({num_lstm_layer}, {B}, {hidden_dim}) "
        f"matches OBLAgentR2D2.MultiLayerLSTM layout"
    )

    # 3b. Init PublicLSTMQNetwork. Add a (T=1) leading time dim — the
    # network's LSTM is scanned over time.
    qnet = PublicLSTMQNetwork(
        action_dim=21,
        hidden_dim=hidden_dim,
        num_lstm_layer=num_lstm_layer,
    )
    init_priv_t = jnp.zeros((1, B, obs_dim), dtype=obs.dtype)
    init_publ_t = jnp.zeros((1, B, publ_dim), dtype=obs.dtype)
    init_dones_t = jnp.zeros((1, B), dtype=jnp.float32)
    rng, k_qp = jax.random.split(rng)
    qparams = qnet.init(k_qp, init_hs, init_priv_t, init_publ_t, init_dones_t)

    # 3c. Inspect param tree — confirm two LSTM layers + fc_v + fc_a.
    flat_params = flatten_dict(qparams["params"])
    path_strs = ["/".join(str(p) for p in k) for k in flat_params.keys()]
    print("param paths (first 12):")
    for p in path_strs[:12]:
        print(f"  {p}")
    lstm_layer_names = sorted(
        {tok for ps in path_strs for tok in ps.split("/")
         if (tok.startswith("l") and tok[1:].isdigit())}
    )
    print(f"detected LSTM layer names: {lstm_layer_names}")
    _check(
        set(lstm_layer_names) == {"l0", "l1"},
        f"expected LSTM layer names {{'l0','l1'}}, got {lstm_layer_names}",
    )
    # Final two Dense layers in the network are fc_v (out=1) and fc_a (out=21).
    out_dims_of_dense = sorted({
        v.shape[-1] for k, v in flat_params.items()
        if any("Dense" in p for p in (str(x) for x in k)) and str(k[-1]) == "kernel"
    })
    print(f"Dense output dims found: {out_dims_of_dense}")
    _check(1 in out_dims_of_dense, "fc_v (Dense out=1) head missing")
    _check(21 in out_dims_of_dense, "fc_a (Dense out=action_dim=21) head missing")
    print("[OK] param tree has 2 LSTM layers (l0, l1) + fc_v + fc_a")

    # 3d. Forward pass on a real obs.
    priv_t = obs[None, :]  # (T=1, B, obs_dim)
    publ_t = publ_ours[None, :]  # (T=1, B, publ_dim)
    dones_t = jnp.zeros((1, B), dtype=jnp.float32)
    new_carry, q = qnet.apply(qparams, init_hs, priv_t, publ_t, dones_t)
    print(
        f"PublicLSTMQNetwork forward: q shape {q.shape}, "
        f"new_carry leaves {[l.shape for l in jax.tree_util.tree_leaves(new_carry)]}"
    )
    _check(
        tuple(q.shape) == (1, B, 21),
        f"q shape {q.shape} != expected (1, {B}, 21)",
    )
    _check(bool(jnp.all(jnp.isfinite(q))), "q contains non-finite values")
    print(f"[OK] forward pass finite: mean={float(q.mean()):.6f}, "
          f"std={float(q.std()):.6f}")

    # 3e. Per-step reset gate sanity: setting dones[t]=1 should zero the carry.
    dones_reset = jnp.ones((1, B), dtype=jnp.float32)
    nonzero_init = jax.tree_util.tree_map(
        lambda x: jnp.ones_like(x) * 0.5, init_hs
    )
    new_carry_after_reset, _ = qnet.apply(
        qparams, nonzero_init, priv_t, publ_t, dones_reset
    )
    # After processing one step with dones=True, carry would be:
    # zero -> LSTM update; the carry going IN to LSTM is zeros (reset),
    # then LSTM emits a new carry. Just confirm shape and finiteness.
    for leaf in jax.tree_util.tree_leaves(new_carry_after_reset):
        _check(
            tuple(leaf.shape) == (num_lstm_layer, B, hidden_dim),
            f"post-reset carry leaf shape {leaf.shape} != "
            f"({num_lstm_layer},{B},{hidden_dim})",
        )
        _check(bool(jnp.all(jnp.isfinite(leaf))), "post-reset carry not finite")
    print("[OK] per-step reset gate shape/finiteness intact")

    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
