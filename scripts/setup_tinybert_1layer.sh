#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -W group_list=gj23
#PBS -j oe
#PBS -N setup_bert1l

# Convert TinyBERT-L-2 from PyTorch to Flax, then strip to 1 encoder layer
# (matching the original R3D2 repo's deleteEncodingLayers with num_lm_layer=1).

cd /work/gj23/k36132/ideas/r3d2_tom
module load nvidia/25.9 nv-hpcx/25.9
source .venv/bin/activate

echo "=== TinyBERT 1-Layer Setup ==="
echo "Date: $(date)"

# Remove old weights if any
rm -rf baselines/QLearning/pretrained_text/tinybert_l2_flax/

python baselines/QLearning/setup_pretrained_text.py 2>&1

echo ""
echo "=== Verifying ==="
python -c "
from transformers import AutoTokenizer, FlaxBertModel
import jax.numpy as jnp
d = 'baselines/QLearning/pretrained_text/tinybert_l2_flax'
tok = AutoTokenizer.from_pretrained(d)
model = FlaxBertModel.from_pretrained(d)
print(f'Vocab: {tok.vocab_size}, Hidden: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}')
assert model.config.num_hidden_layers == 1, f'Expected 1 layer, got {model.config.num_hidden_layers}'
test = tok('hello world', return_tensors='np')
out = model(input_ids=jnp.array(test['input_ids']), attention_mask=jnp.array(test['attention_mask']))
print(f'Output shape: {out.last_hidden_state.shape}')
print('TinyBERT 1-layer OK')
" 2>&1

echo "Date: $(date)"
