#!/usr/bin/env python3
"""Standalone wrapper: fixes NemotronHModel's MoE-detection bug without
touching the vendored convert_hf_to_gguf.py (matching the established
campaign pattern from pythia410m/deepseek-moe fixes). The original checks
'num_experts_per_tok' in hparams (key presence), and load_hparams() for
nvidia/Nemotron-H-4B-Base-8K reports a full, plausible-looking MoE config
(num_experts_per_tok=2, moe_intermediate_size=7688, n_routed_experts=8) --
but direct inspection of both safetensors shards (311 tensors total) found
zero tensors matching 'expert'/'moe'/'router'. This checkpoint is genuinely
dense; the MoE config fields are stale, inherited from the 8B MoE parent
during pruning/distillation. Fix: hardcode is_moe=False, treating tensor
evidence as authoritative over the misleading config. Replicates the FULL
original __init__ body (not just the is_moe check) so head_dim/d_inner/
_ssm_layers/_mlp_layers still get set correctly.
"""
import sys

sys.path.insert(0, '/home/ryang/work/hackathon/repo/ported_models/llama_cpp_et/src/llama.cpp-et')
import convert_hf_to_gguf as chg
import gguf


def patched_init(self, *args, **kwargs):
    hparams = chg.ModelBase.load_hparams(args[0], self.is_mistral_format)
    if False:  # nvidia/Nemotron-H-4B-Base-8K's config has stale MoE fields (moe_intermediate_size, n_routed_experts, num_experts_per_tok) with zero matching expert tensors in the actual weights -- confirmed via direct safetensors inspection (0/311 tensors match 'expert'/'moe'/'router'). Genuinely dense, not MoE.
        self.model_arch = gguf.MODEL_ARCH.NEMOTRON_H_MOE
        self.is_moe = True

    chg.GraniteHybridModel.__init__(self, *args, **kwargs)

    self.head_dim = self.hparams.get('head_dim', self.hparams.get('attention_head_dim'))
    assert self.head_dim is not None, 'Could not find the attention head dim in config'

    self.d_inner = self.find_hparam(['num_heads']) * self.d_model

    pattern = self.hparams.get('hybrid_override_pattern') or self.hparams.get('layers_block_type')
    if pattern is None:
        self._ssm_layers = []
        self._mlp_layers = []
    elif isinstance(pattern, str):
        self._ssm_layers = [i for i, val in enumerate(pattern) if val == 'M']
        self._mlp_layers = [i for i, val in enumerate(pattern) if val == ('E' if self.is_moe else '-')]
    else:
        self._ssm_layers = [i for i, val in enumerate(pattern) if val == 'mamba']
        self._mlp_layers = [i for i, val in enumerate(pattern) if val == 'moe']


chg.NemotronHModel.__init__ = patched_init

if __name__ == '__main__':
    sys.argv[0] = 'convert_hf_to_gguf.py'
    chg.main()
