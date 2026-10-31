"""Thin-bundle runner shim for Qwen3.5-35B-A3B on one Blackhole p300 board (two dies).

The model code ships in the `tt-metal-models` wheel (models.autoports.qwen_qwen3_coder_next, with the loader patch of this package).
This module re-exports the vLLM adapter class so `model:Qwen3NextForCausalLM` resolves with the bundle directory on PYTHONPATH.
"""
from models.autoports.qwen_qwen3_coder_next.tt.generator_vllm import Qwen3NextForCausalLM  # noqa: F401
