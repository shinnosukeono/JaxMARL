"""Microbenchmark publ-LSTM variants on realistic training shapes.

Measured on the actual H100/GH200 with realistic per-agent shapes from
r2d2_publ_rnn_hanabi.yaml:
    HIDDEN_SIZE=512, NUM_STEPS=80, BUFFER_BATCH_SIZE=64, num_lstm_layer=2

Variants:
  V1  baseline: MultiLayerScannedLSTM with `nn.OptimizedLSTMCell` (current).
  V2  same but `nn.LSTMCell` instead of OptimizedLSTMCell.
  V3  two stacked single-layer ScannedLSTMs chained in series.
  V4  V1 + zeros tensor lifted out of the scan body (hand-hoisted).

Each variant is wrapped in the same publ-LSTM Q-net (3-layer priv MLP +
1-layer publ MLP + LSTM stack + dueling head) so the only thing that
varies is the LSTM portion.

Outputs: median fwd time, median fwd+grad time, both in ms.
"""
from __future__ import annotations

import os
import sys
import time
from functools import partial
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


# ----- LSTM stacks under test -----


class _MultiLayerScan(nn.Module):
    """Multi-layer LSTM scanned over time. `cell_cls` is swappable."""
    num_layers: int = 2
    cell_cls: Any = nn.OptimizedLSTMCell
    lift_zeros: bool = False
    dtype: Any = jnp.float32        # compute dtype
    param_dtype: Any = jnp.float32  # storage dtype

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0, out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        if self.lift_zeros:
            # Hoisted zeros: a precomputed constant of the carry shape,
            # avoiding `zeros_like(carry)` inside the scan body.
            zeros = (
                jnp.zeros(carry[0].shape, dtype=carry[0].dtype),
                jnp.zeros(carry[1].shape, dtype=carry[1].dtype),
            )
        else:
            zeros = jax.tree.map(jnp.zeros_like, carry)
        reset_mask = resets[None, ..., None]
        carry = jax.tree.map(
            lambda c, z: jnp.where(reset_mask, z, c), carry, zeros
        )
        new_cs, new_hs = [], []
        y = ins
        for layer in range(self.num_layers):
            layer_carry = jax.tree.map(lambda x, _l=layer: x[_l], carry)
            new_layer_carry, y = self.cell_cls(
                ins.shape[-1],
                name=f"l{layer}",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(layer_carry, y)
            new_cs.append(new_layer_carry[0])
            new_hs.append(new_layer_carry[1])
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, y

    @staticmethod
    def initialize_carry(hidden, num_layers, *batch):
        shape = (num_layers, *batch, hidden)
        return (jnp.zeros(shape), jnp.zeros(shape))


class _SingleLayerScan(nn.Module):
    """Single-layer LSTM scanned over time."""
    cell_cls: Any = nn.OptimizedLSTMCell

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0, out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        zeros = jax.tree.map(jnp.zeros_like, carry)
        reset_mask = resets[..., None]
        carry = jax.tree.map(
            lambda c, z: jnp.where(reset_mask, z, c), carry, zeros
        )
        new_carry, y = self.cell_cls(ins.shape[-1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(hidden, *batch):
        return (jnp.zeros((*batch, hidden)), jnp.zeros((*batch, hidden)))


class _StackedTwoScans(nn.Module):
    """Two single-layer scans chained sequentially (alternative 2-layer)."""
    cell_cls: Any = nn.OptimizedLSTMCell

    @nn.compact
    def __call__(self, carry, x):
        h0, h1 = carry
        ins, resets = x
        h0_new, y = _SingleLayerScan(cell_cls=self.cell_cls, name="lstm0")(
            h0, (ins, resets)
        )
        h1_new, y = _SingleLayerScan(cell_cls=self.cell_cls, name="lstm1")(
            h1, (y, resets)
        )
        return (h0_new, h1_new), y

    @staticmethod
    def initialize_carry(hidden, *batch):
        zero = (jnp.zeros((*batch, hidden)), jnp.zeros((*batch, hidden)))
        return (zero, zero)


# ----- Q-net wrapping a configurable LSTM stack -----


class _QNet(nn.Module):
    """publ-LSTM Q-net that accepts any LSTM-stack module + initial carry."""
    action_dim: int = 21
    hidden_dim: int = 512
    lstm_module: Any = None  # an nn.Module instance for the LSTM stack
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden, priv_s, publ_s, dones):
        dense_kwargs = dict(
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        priv_o = priv_s
        for _ in range(3):
            priv_o = nn.Dense(self.hidden_dim, **dense_kwargs)(priv_o)
            priv_o = nn.relu(priv_o)
        publ_x = nn.Dense(self.hidden_dim, **dense_kwargs)(publ_s)
        publ_x = nn.relu(publ_x)
        hidden, publ_o = self.lstm_module(hidden, (publ_x, dones))
        o = priv_o * publ_o
        v = nn.Dense(1, **dense_kwargs)(o)
        a = nn.Dense(self.action_dim, **dense_kwargs)(o)
        return hidden, v + a


# ----- Bench harness -----


def _time(fn, n_warmup=5, n_iters=50):
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return float(np.median(times)), float(np.std(times))


def _bench(name, lstm_module, init_carry, T, B, obs_dim, publ_dim, hidden):
    print(f"\n=== {name} ===", flush=True)
    rng = jax.random.PRNGKey(0)
    net = _QNet(action_dim=21, hidden_dim=hidden, lstm_module=lstm_module)
    priv = jnp.zeros((T, B, obs_dim))
    publ = jnp.zeros((T, B, publ_dim))
    dones = jnp.zeros((T, B))

    rng, k = jax.random.split(rng)
    t0 = time.perf_counter()
    params = net.init(k, init_carry, priv, publ, dones)
    jax.block_until_ready(params)
    t_init = (time.perf_counter() - t0) * 1000
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  init: {t_init:.1f} ms,  params: {n_params/1e6:.2f}M")

    @jax.jit
    def fwd(p, h, pr, pu, d):
        return net.apply(p, h, pr, pu, d)[1].sum()

    t0 = time.perf_counter()
    out = fwd(params, init_carry, priv, publ, dones)
    jax.block_until_ready(out)
    print(f"  fwd compile: {(time.perf_counter() - t0)*1000:.1f} ms")
    fwd_med, fwd_std = _time(
        lambda: fwd(params, init_carry, priv, publ, dones)
    )
    print(f"  fwd      : {fwd_med:.3f} ± {fwd_std:.3f} ms")

    @jax.jit
    def grad(p, h, pr, pu, d):
        def loss(pp):
            return net.apply(pp, h, pr, pu, d)[1].sum()
        return jax.grad(loss)(p)

    t0 = time.perf_counter()
    out = grad(params, init_carry, priv, publ, dones)
    jax.block_until_ready(out)
    print(f"  grad compile: {(time.perf_counter() - t0)*1000:.1f} ms")
    g_med, g_std = _time(lambda: grad(params, init_carry, priv, publ, dones))
    print(f"  fwd+grad : {g_med:.3f} ± {g_std:.3f} ms")

    return {
        "name": name, "n_params": n_params,
        "fwd_ms": fwd_med, "fwd_grad_ms": g_med,
    }


def main() -> int:
    T, B, hidden = 80, 64, 512
    obs_dim, publ_dim = 660, 535  # 535 = 660 - 125

    print(f"jax devices: {jax.devices()}")
    print(f"shapes: T={T} B={B} obs={obs_dim} publ={publ_dim} hid={hidden}")

    rows = []

    rows.append(_bench(
        "V1 multi-1scan OptimizedLSTMCell (current)",
        _MultiLayerScan(num_layers=2, cell_cls=nn.OptimizedLSTMCell),
        _MultiLayerScan.initialize_carry(hidden, 2, B),
        T, B, obs_dim, publ_dim, hidden,
    ))

    rows.append(_bench(
        "V2 multi-1scan LSTMCell",
        _MultiLayerScan(num_layers=2, cell_cls=nn.LSTMCell),
        _MultiLayerScan.initialize_carry(hidden, 2, B),
        T, B, obs_dim, publ_dim, hidden,
    ))

    rows.append(_bench(
        "V3 two-stacked-scans OptimizedLSTMCell",
        _StackedTwoScans(cell_cls=nn.OptimizedLSTMCell),
        _StackedTwoScans.initialize_carry(hidden, B),
        T, B, obs_dim, publ_dim, hidden,
    ))

    rows.append(_bench(
        "V4 multi-1scan OptimizedLSTMCell + lifted-zeros",
        _MultiLayerScan(num_layers=2, cell_cls=nn.OptimizedLSTMCell, lift_zeros=True),
        _MultiLayerScan.initialize_carry(hidden, 2, B),
        T, B, obs_dim, publ_dim, hidden,
    ))

    # V5: mixed-precision (params fp32, compute bf16). Standard H100 pattern.
    # We keep `param_dtype=fp32` so orthogonal init's QR works, but cast at
    # use site to bf16 for the matmul.
    print("\n=== V5: V1 with bfloat16 compute (params fp32) ===", flush=True)
    rng = jax.random.PRNGKey(0)
    # Use a custom QNet that takes compute_dtype while keeping param_dtype fp32.
    class _QNetMP(nn.Module):
        action_dim: int = 21
        hidden_dim: int = 512
        compute_dtype: Any = jnp.bfloat16
        lstm_module: Any = None

        @nn.compact
        def __call__(self, hidden, priv_s, publ_s, dones):
            d = dict(
                kernel_init=orthogonal(1.0), bias_init=constant(0.0),
                dtype=self.compute_dtype, param_dtype=jnp.float32,
            )
            priv_o = priv_s.astype(self.compute_dtype)
            for _ in range(3):
                priv_o = nn.Dense(self.hidden_dim, **d)(priv_o)
                priv_o = nn.relu(priv_o)
            publ_x = nn.Dense(self.hidden_dim, **d)(publ_s.astype(self.compute_dtype))
            publ_x = nn.relu(publ_x)
            hidden, publ_o = self.lstm_module(hidden, (publ_x, dones.astype(self.compute_dtype)))
            o = priv_o * publ_o
            v = nn.Dense(1, **d)(o)
            a = nn.Dense(self.action_dim, **d)(o)
            return hidden, v + a

    net16 = _QNetMP(
        action_dim=21, hidden_dim=hidden,
        compute_dtype=jnp.bfloat16,
        lstm_module=_MultiLayerScan(
            num_layers=2, cell_cls=nn.OptimizedLSTMCell,
            dtype=jnp.bfloat16, param_dtype=jnp.float32,
        ),
    )
    priv16 = jnp.zeros((T, B, obs_dim), dtype=jnp.bfloat16)
    publ16 = jnp.zeros((T, B, publ_dim), dtype=jnp.bfloat16)
    dones16 = jnp.zeros((T, B), dtype=jnp.bfloat16)
    init_hs16 = (
        jnp.zeros((2, B, hidden), dtype=jnp.bfloat16),
        jnp.zeros((2, B, hidden), dtype=jnp.bfloat16),
    )
    rng, k = jax.random.split(rng)
    t0 = time.perf_counter()
    params16 = net16.init(k, init_hs16, priv16, publ16, dones16)
    jax.block_until_ready(params16)
    print(f"  init: {(time.perf_counter()-t0)*1000:.1f} ms")

    @jax.jit
    def fwd16(p, h, pr, pu, d):
        return net16.apply(p, h, pr, pu, d)[1].sum()

    t0 = time.perf_counter()
    out = fwd16(params16, init_hs16, priv16, publ16, dones16)
    jax.block_until_ready(out)
    print(f"  fwd compile: {(time.perf_counter()-t0)*1000:.1f} ms")
    f5_med, f5_std = _time(
        lambda: fwd16(params16, init_hs16, priv16, publ16, dones16)
    )
    print(f"  fwd      : {f5_med:.3f} ± {f5_std:.3f} ms")

    @jax.jit
    def grad16(p, h, pr, pu, d):
        def loss(pp):
            return net16.apply(pp, h, pr, pu, d)[1].sum()
        return jax.grad(loss)(p)

    t0 = time.perf_counter()
    out = grad16(params16, init_hs16, priv16, publ16, dones16)
    jax.block_until_ready(out)
    print(f"  grad compile: {(time.perf_counter()-t0)*1000:.1f} ms")
    g5_med, _ = _time(
        lambda: grad16(params16, init_hs16, priv16, publ16, dones16)
    )
    print(f"  fwd+grad : {g5_med:.3f} ms")
    rows.append({
        "name": "V5 V1 + bfloat16",
        "n_params": sum(x.size for x in jax.tree_util.tree_leaves(params16)),
        "fwd_ms": f5_med, "fwd_grad_ms": g5_med,
    })

    print("\n=== summary (median, lower=better) ===")
    print(f"{'variant':50s} {'params(M)':>10s} {'fwd_ms':>10s} {'grad_ms':>10s}")
    for v in rows:
        print(
            f"{v['name']:50s} "
            f"{v['n_params']/1e6:>10.2f} "
            f"{v['fwd_ms']:>10.3f} "
            f"{v['fwd_grad_ms']:>10.3f}"
        )

    base_fwd = rows[0]["fwd_ms"]
    base_grad = rows[0]["fwd_grad_ms"]
    print("\n=== relative to V1 (1.00 = baseline; <1.00 = faster) ===")
    print(f"{'variant':50s} {'fwd':>10s} {'grad':>10s}")
    for v in rows:
        print(
            f"{v['name']:50s} "
            f"{v['fwd_ms']/base_fwd:>10.3f} "
            f"{v['fwd_grad_ms']/base_grad:>10.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
