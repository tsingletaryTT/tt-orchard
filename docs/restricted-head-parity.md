# Restricted-head parity: do the chips give the model's own readout the same answers as the CPU?

A restricted-head model is the nearest model's backbone with its `lm_head` cut down to a few rows in a sidecar (the first was
`perplexity-ai/pplx-decider-v1.2`). The harness does not serve it: stage 0 stops the run with the block code `restricted-head` and says what it
found. This is the measurement an operator can ask for after that stop. It reports whether the TT backbone's hidden states, read at each question's
marker token through the readout rows under the attention mask the model declares, give the probabilities a CPU reference gives. It does not
serve the model, publish anything or change tt-metal.

## What it needs
- A run that stopped with `restricted-head` (its `stages/0/delta.json` has `head`, `alignment.raw_rules` and `code_files`).
- The model's `decision_config.json` and `joint_config.json` (protocol version 3, template `joint-question-inline-readout-v1`, attention mode
  `noncausal_full_attention`) and the two code files pinned in `PINNED_CODE` (by sha256; they are hashed, never imported). A model with any other
  protocol is refused with exit 5: it needs its own script.
- An installed v6 bundle of the nearest model (for its `run.sh`, python and TT model code), a board (2 chips) and room for a tensor cache (about 34 GB).
- `torch`, `transformers`, `tokenizers` and `safetensors` in the python that runs the CPU phase (the reference venv; no torchvision needed).

## Running it
```bash
STAGE=<run dir>/stages/3 ; T=<orchard checkout>/orchard/skills
mkdir -p $STAGE && cp $T/restricted-head-templates/*.py $T/sidecar-parity-templates/{prepare_parity,hidden_parity}.py $STAGE/
cat > $STAGE/restricted_head_config.json <<JSON
{"run_dir": "<run dir>", "model_snapshot": "<snapshot of the model>", "bundle_dir": "<tt-model root>/episod/qwen3.8-27b-dflash2-p300",
 "tt_cache": "<run dir>/tt_cache", "mesh_shape": [1, 2]}
JSON
cd $STAGE
<reference python> restricted_head_parity.py --phase cpu                       # about a minute; no chip
python3 prepare_restricted_head.py                                              # parity-model-dir and parity-run.sh
gozer run --chips 2 --who "<you>" --reason "restricted-head parity" -- timeout 3300 bash parity-run.sh --phase device
```
Give the device phase a `timeout`. A cold first run builds the tensor cache (about 3 minutes on this box, 31 GB); a repeat with the cache is under a
minute. The result is `evidence/restricted-head-parity.json`; `evidence/mask-effect.json` comes from the CPU phase alone.

## What it reports
- `result`: the smallest hidden-state correlation (all positions, and at the marker positions the readout reads), the largest probability
  difference, top-1 agreement over the questions, a wiring cosine, and a verdict against the bars (`HIDDEN_PCC_MIN` 0.90, `TOP1_MIN` 0.85,
  `PROB_DIFF_MAX` 0.25 or four times the reference's own noise, at least 10 questions, `WIRING_COSINE_MIN` 0.90). The bars are choices copied
  from the sidecar parity check, which measured a different head.
- `mask_effect_on_cpu`: what a runtime that applied the usual causal mask would change, measured on the CPU reference itself.
- `meta`: the number of causal SDPA calls the shim turned non-causal per record (it fails the run when that is zero), versions, timings.

## The two shims (the run only; undone after)
1. The TT loader reads a bf16 checkpoint through `Qwen3_5ForCausalLM`, which expects `model.*` names. A checkpoint that says `language_model.*`
   and `visual.*` is loaded as random weights with only a warning. The shim passes stage 0's `alignment.raw_rules` as `key_mapping`.
2. The prefill attention calls `scaled_dot_product_attention(..., is_causal=True)` once per full-attention layer. The shim makes that call
   `is_causal=False` with an additive mask hiding the trailing padding keys.

## What it does and does not show
It shows that on a fixed set of text requests the chips and the CPU agree within a measured tolerance, and how much the mask matters. It does not
show agreement on other inputs (images, long states, 255-option questions), that the model's quality is what its card claims, or that the model
can be served: serving needs the multi-position readout and the mask inside the serving stack, which this does not build.
