# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

A single script, `steer_neurons.py`, that turns single-neuron steers (`LAYER:NEURON:DOSE`) on a Hugging Face model into a llama.cpp control-vector GGUF, which is then served with `llama-server --control-vector` or LM Studio. See `README.md` for user-facing usage.

The `/create-steering` skill (`.claude/skills/create-steering/SKILL.md`) wraps the script: it takes a [neuronary.org](https://neuronary.org) recipe (fetched from its read-only API) or a model plus steers, and turns them into `--hf`/`--steer` arguments. neuronary's recipe notation (`L13N7956-1.0`) and dose unit match the script's.

## Commands

```sh
uv sync                                   # create/update .venv from uv.lock
uv run steer_neurons.py --help
uv run steer_neurons.py --hf models/Qwen3.5-4B --steer 14:8949:+1 --dry-run   # print numbers, write nothing
```

There is no test suite, linter config, or build step. The project is deliberately a plain uv project with no `[build-system]` and no package: keep it a single root script run via `uv run steer_neurons.py`.

To exercise the script without downloading a real model, save a tiny random `LlamaForCausalLM` (plus a small `PreTrainedTokenizerFast`) to a temp folder and pass it as `--hf`. Do this outside the repo so no model files land here.

## How the steering math maps to llama.cpp

- Adding `dose` to neuron `n`'s activation in layer `L` adds `dose * W_down[:, n]` to the residual stream at the output of layer `L`, for every token. That is a constant vector, which is exactly what a llama.cpp control vector is. A LoRA can't express it, since it has no bias term. Steers that depend on the input, such as clamping a neuron, can't be done this way.
- The vector is `dose * median(||resid leaving L||, tokens 1:) * W_down[:, n] / ||W_down[:, n]||`. The median is measured on calibration text, and token 0 is skipped because of its outsized norm (attention sink).
- The residual norm is captured with **forward hooks on the decoder layers**, not `output_hidden_states`: HF's last hidden state has the final norm applied, which would mis-scale last-layer steers.
- The output GGUF has arch `controlvector`, keys `controlvector.model_hint` and `controlvector.layer_count`, and one f32 tensor `direction.{L}` per steered layer, with steers on the same layer summed. `L` is the 0-indexed HF layer, and llama.cpp adds `direction.L` after its layer `L`. llama.cpp ignores `direction.0`.
- `--gguf` reads `blk.{L}.ffn_down.weight` from the served GGUF (decoding BF16/F16/F32 directly, otherwise via `gguf.quants.dequantize`) and compares the neuron column to catch checkpoint or indexing mismatches. It also supplies the arch for `model_hint`.
- Model plumbing is heuristic so many architectures work: the decoder stack is the longest `ModuleList` whose children all have `.mlp`, and the down projection is found by name (`DOWN_NAMES`), handling both `nn.Linear` and GPT-2-style `Conv1D` weight layouts. Mixture-of-experts MLPs are rejected on purpose.

## Dependencies

- `requires-python >= 3.12`. Each dependency's minimum is the latest release as of when it was added (`dep>=X.Y.Z`, as `uv add` writes). Raising `requires-python` or the minimums must stay consistent, since newer torch/numpy releases drop older Pythons.
- torch comes from the PyTorch CUDA 13.2 index **on Windows only** (`[tool.uv.sources]` with a `sys_platform == 'win32'` marker), because PyPI's Windows torch is CPU-only. Linux PyPI torch already bundles CUDA 13; macOS (Apple Silicon, macOS 14+) uses PyPI torch with MPS. CUDA 13 wheels cover sm_75+ (Turing and newer).
- The script runs one forward pass, so it works on CPU too; RAM (~2 bytes/param in bf16) is the practical limit.

## Conventions

- Shell examples in `README.md` and the script docstring are single-line (no `\` continuations), so they paste into bash and PowerShell alike, and they use forward-slash paths.
- `models/`, `out/`, `*.gguf`, `*.safetensors` are gitignored; the default output path is `out/<model>_<steers>.gguf`.
