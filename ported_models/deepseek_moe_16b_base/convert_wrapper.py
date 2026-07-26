#!/usr/bin/env python3
"""Standalone wrapper: registers deepseek-moe-16b's tokenizer chkhsh at
runtime without touching the vendored convert_hf_to_gguf.py (matching the
established pattern from this campaign's pythia410m rope_parameters fix).
Extracts the real chktxt string via source introspection to guarantee
byte-identical hashing versus the original function -- no manual retyping.
"""
import ast
import inspect
import sys
from hashlib import sha256

sys.path.insert(0, '/home/ryang/work/hackathon/repo/ported_models/llama_cpp_et/src/llama.cpp-et')
import convert_hf_to_gguf as chg

_orig_get_vocab_base_pre = chg.TextModel.get_vocab_base_pre
_KNOWN_HASH = '93105512fde79bc726022fe3cbfb7efef9738465b988d0600beba1296e3a91d8'

# Extract the exact chktxt string literal from the real function's source.
_src = __import__('textwrap').dedent(inspect.getsource(_orig_get_vocab_base_pre))
_tree = ast.parse(_src)
_chktxt = None
for node in ast.walk(_tree):
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'chktxt' for t in node.targets):
        _chktxt = ast.literal_eval(node.value)
        break
assert _chktxt is not None, 'could not extract chktxt from original function source'


def patched_get_vocab_base_pre(self, tokenizer):
    chktok = tokenizer.encode(_chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()
    if chkhsh == _KNOWN_HASH:
        # NOTE: not 'deepseek-moe' -- that name isn't recognized by the
        # separate hardcoded pre-tokenizer registry in llama-vocab.cpp
        # (C++ runtime), which only knows 'deepseek-llm'/'deepseek-coder'/
        # 'deepseek-v3'/'deepseek-r1-qwen'. DeepSeek's early dense/MoE
        # models share the same base BPE tokenizer family, so 'deepseek-llm'
        # is correct and is recognized by both the Python converter and the
        # C++ runtime.
        return 'deepseek-llm'
    return _orig_get_vocab_base_pre(self, tokenizer)


chg.TextModel.get_vocab_base_pre = patched_get_vocab_base_pre

if __name__ == '__main__':
    sys.argv[0] = 'convert_hf_to_gguf.py'
    chg.main()
