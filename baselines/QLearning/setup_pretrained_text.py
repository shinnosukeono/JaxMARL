"""
One-time helper: convert TinyBERT-L-2 from PyTorch to Flax and cache locally.

The text-based baselines (r2d2_text, r3d2, r3d2_multitask, obl) reference
`baselines/QLearning/pretrained_text/tinybert_l2_flax/` for the encoder
weights. The conversion needs CPU torch + HF transformers; we don't ship the
binary weights in git.

Usage:
    python baselines/QLearning/setup_pretrained_text.py

You only need this once. After it runs, the configs will pick up the local
folder automatically.
"""

from __future__ import annotations

import os
import sys


def main():
    target = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pretrained_text", "tinybert_l2_flax",
    )
    os.makedirs(target, exist_ok=True)

    try:
        from transformers import AutoTokenizer, FlaxBertModel  # noqa: F401
    except ImportError:
        sys.exit(
            "transformers is not installed in this venv. "
            "Run `uv pip install --python .venv/bin/python "
            "'transformers==4.46.3'` first."
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit(
            "PyTorch (CPU) is needed to convert the HF Bert weights to Flax. "
            "Run `uv pip install --python .venv/bin/python torch "
            "--index-url https://download.pytorch.org/whl/cpu` first."
        )

    print("Loading cross-encoder/ms-marco-TinyBERT-L-2-v2 …")
    tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    model = FlaxBertModel.from_pretrained(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2", from_pt=True
    )

    tok.save_pretrained(target)
    model.save_pretrained(target)
    print(f"Cached to {target}")
    print("Files:", sorted(os.listdir(target)))


if __name__ == "__main__":
    main()
