# steer-neurons

Build [llama.cpp](https://github.com/ggml-org/llama.cpp) control vectors that reproduce
single-neuron steering, so you can run steered models in LM Studio or llama.cpp.

Steering neuron `n` in layer `L` by `dose` adds the neuron's output direction to the
residual stream after that layer:

```
dose × median(‖resid leaving L‖, tokens 1:) × W_down[:, n] / ‖W_down[:, n]‖
```

A llama.cpp control vector is exactly a per-layer vector added there, so the result
loads with `--control-vector`.

## Setup

Needs [uv](https://docs.astral.sh/uv/). No install step: `uv run` creates the
environment on first use.

```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

This should print your GPU. On Windows, PyTorch comes from the CUDA 13.2 index
(`+cu132`, configured in `pyproject.toml`), because PyPI's Windows build is CPU-only.
Linux and macOS use the regular PyPI build.

## Usage

1. Download the official Hugging Face weights of the model you serve (the GGUF must be
   converted from the same checkpoint):

   ```sh
   uv run hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
   ```

2. Build the control vector. Each `--steer` is `LAYER:NEURON:DOSE` (0-indexed layers,
   as in Hugging Face); repeat it for several neurons.

   ```sh
   uv run steer_neurons.py --hf models/Qwen3.5-4B --steer 14:8949:+1 --steer 13:7956:-1 --gguf models/Qwen3.5-4B-BF16.gguf
   ```

   The file goes to `out/` with a name like `Qwen3.5-4B_L14N8949p1_L13N7956m1.gguf`.
   `--gguf` checks the served weights match the HF ones and sets the right arch hint.

   Other options: `--out` (file name), `--dry-run` (print numbers only),
   `--calib-file` (own text for measuring residual norms), `--dtype`.

3. Load it alongside the model:

   - `llama-server`: `llama-server -m model.gguf --control-vector out/file.gguf`
   - LM Studio: add one row under **Additional arguments**:
     `--control-vector` → path to the file.

## Checking it works

With one prompt at temperature 0, compare three loads:

| Argument | Expected |
|---|---|
| none | baseline |
| `--control-vector-scaled` → `file.gguf:0` | identical to baseline |
| `--control-vector-scaled` → `file.gguf:5` | clearly different or broken |

If scale 5 changes nothing, llama.cpp isn't applying control vectors for that
architecture.

## Limits

- The vector only matches the checkpoint it was built from. Quants of that checkpoint
  are fine (Q8_0 is near-exact, Q4 is a close approximation); fine-tunes and other
  sizes are not.
- Mixture-of-experts layers aren't supported: an expert neuron fires only when routed,
  which a fixed vector can't reproduce.
- Layer 0 steers have no effect; llama.cpp ignores `direction.0`.
