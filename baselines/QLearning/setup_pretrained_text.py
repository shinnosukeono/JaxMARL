"""
One-time helper: convert TinyBERT-L-2 from PyTorch to Flax and cache locally.

The text-based baselines (r2d2_text, r3d2, r3d2_multitask, obl) reference
`baselines/QLearning/pretrained_text/tinybert_l2_flax/` for the encoder
weights. The conversion needs CPU torch + HF transformers; we don't ship the
binary weights in git.

The original R3D2 repo uses ``deleteEncodingLayers()`` to keep only 1 of
TinyBERT-L-2's 2 encoder layers (controlled by ``num_lm_layer``). We
reproduce that here: after loading the full 2-layer model, we strip it to
1 encoder layer by modifying the config and pruning ``params["encoder"]``.

Usage:
    python baselines/QLearning/setup_pretrained_text.py

You only need this once. After it runs, the configs will pick up the local
folder automatically.
"""

from __future__ import annotations

import copy
import json
import os
import sys


NUM_ENCODER_LAYERS_TO_KEEP = 1


def main():
    target = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pretrained_text", "tinybert_l2_flax",
    )
    os.makedirs(target, exist_ok=True)

    try:
        from transformers import AutoTokenizer, FlaxBertModel, BertConfig  # noqa: F401
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

    print("Loading cross-encoder/ms-marco-TinyBERT-L-2-v2 (full 2-layer) …")
    tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    model = FlaxBertModel.from_pretrained(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2", from_pt=True
    )

    orig_layers = model.config.num_hidden_layers
    keep = NUM_ENCODER_LAYERS_TO_KEEP
    print(f"Stripping encoder layers: {orig_layers} → {keep} "
          f"(matching original R3D2 deleteEncodingLayers)")

    # Prune encoder layer params: keep only layer indices [0, keep).
    new_params = copy.deepcopy(model.params)
    encoder_layers = new_params["encoder"]["layer"]
    pruned_layers = {str(i): encoder_layers[str(i)] for i in range(keep)}
    new_params["encoder"]["layer"] = pruned_layers

    # Save with updated config.
    new_config = copy.deepcopy(model.config)
    new_config.num_hidden_layers = keep
    new_model = FlaxBertModel(new_config)
    new_model.params = new_params

    tok.save_pretrained(target)
    new_model.save_pretrained(target)
    print(f"Cached to {target}")
    print("Files:", sorted(os.listdir(target)))


if __name__ == "__main__":
    main()
