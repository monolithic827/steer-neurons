r"""
steer-neurons: build a llama.cpp control-vector GGUF from single-neuron steers.

Steering neuron n in layer L by `dose` is the same as adding

    dose * median(||resid leaving L||, tokens 1:) * W_down[:, n] / ||W_down[:, n]||

to the residual stream after layer L. llama.cpp control vectors do exactly that.

Examples
  # the original Qwen3.5-4B recipe
  uv run steer_neurons.py --hf models/Qwen3.5-4B --steer 14:8949:+1 --steer 13:7956:-1

  # any HF model (local folder or hub id), verify against the GGUF you serve
  uv run steer_neurons.py --hf meta-llama/Llama-3.1-8B-Instruct --steer 20:1234:+2 --gguf models/llama-3.1-8b-instruct-Q8_0.gguf --out out/llama_steer.gguf

  # just print the numbers, write nothing
  uv run steer_neurons.py --hf models/Qwen3.5-4B --steer 14:8949:+1 --dry-run

Steer format:  LAYER:NEURON:DOSE   (0-indexed layer as in Hugging Face; dose may be negative)
Several steers on the same layer are summed.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_CALIB = (
    "The history of science is full of surprising discoveries. In 1928, a Scottish "
    "researcher noticed that a mold had killed bacteria in a dish he had left out. "
    "Meanwhile, the city council voted on Tuesday to approve a new budget for road "
    "repairs, and local residents expressed mixed feelings about the plan. "
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n"
    "Recipes for sourdough bread usually call for a long, slow fermentation. "
) * 4

DOWN_NAMES = ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "fc_out", "out_proj")


# ---------------------------------------------------------------- parsing
def parse_steer(s):
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad --steer '{s}', expected LAYER:NEURON:DOSE e.g. 14:8949:+1")
    return int(m[1]), int(m[2]), float(m[3])


# ---------------------------------------------------------------- model plumbing
def load_model(name, dtype):
    import transformers
    errors = []
    for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(transformers, loader)
            m = cls.from_pretrained(name, dtype=dtype, device_map="auto")
            m.eval()
            return m
        except Exception as e:  # noqa: BLE001
            errors.append(f"{loader}: {e}")
    sys.exit("could not load model:\n  " + "\n  ".join(errors))


def find_layers(model):
    """Return the longest ModuleList whose children all have an .mlp (the decoder stack)."""
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0 and all(hasattr(c, "mlp") for c in mod):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        sys.exit("could not find decoder layers with an .mlp; this architecture needs a custom adapter")
    return best


def find_down(layer, L):
    mlp = layer.mlp
    if hasattr(mlp, "experts"):
        sys.exit(f"layer {L} MLP is mixture-of-experts. A neuron inside an expert only fires when that "
                 "expert is routed, so a static control vector cannot reproduce it. Not supported.")
    for nm in DOWN_NAMES:
        if hasattr(mlp, nm):
            return getattr(mlp, nm), nm
    sys.exit(f"layer {L}: no down projection found in mlp (looked for {DOWN_NAMES}); "
             f"children are {[n for n, _ in mlp.named_children()]}")


def neuron_column(down, n, n_embd):
    """Output direction written by neuron n, shape [n_embd], handling Linear and GPT-2 style Conv1D."""
    W = down.weight.detach()
    if W.shape[0] == n_embd and W.shape[1] != n_embd:
        n_ff = W.shape[1]; col = W[:, n] if n < n_ff else None       # nn.Linear: [out, in]
    elif W.shape[1] == n_embd and W.shape[0] != n_embd:
        n_ff = W.shape[0]; col = W[n, :] if n < n_ff else None       # Conv1D: [in, out]
    else:
        sys.exit(f"ambiguous down-proj weight shape {tuple(W.shape)} for hidden size {n_embd}")
    if col is None:
        sys.exit(f"neuron {n} out of range; this layer has {n_ff} neurons (0..{n_ff - 1})")
    return col.float(), n_ff


# ---------------------------------------------------------------- GGUF verification
def gguf_arch(reader):
    f = reader.fields.get("general.architecture")
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode()


def gguf_column(reader, L, n):
    t = next((t for t in reader.tensors if t.name == f"blk.{L}.ffn_down.weight"), None)
    if t is None:
        return None, "tensor not found"
    from gguf import GGMLQuantizationType as Q
    n_ff, n_embd = int(t.shape[0]), int(t.shape[1])  # ne listed innermost-first
    raw = np.asarray(t.data)
    try:
        if t.tensor_type == Q.BF16:
            u16 = raw.reshape(-1).view(np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif t.tensor_type in (Q.F16, Q.F32):
            arr = raw.astype(np.float32)
        else:
            from gguf.quants import dequantize
            arr = dequantize(raw, t.tensor_type).astype(np.float32)
        arr = arr.reshape(n_embd, n_ff)
    except Exception as e:  # noqa: BLE001
        return None, f"could not decode {t.tensor_type.name}: {e}"
    return arr[:, n], t.tensor_type.name


def verify(reader, L, n, col):
    g, kind = gguf_column(reader, L, n)
    if g is None:
        print(f"    verify: skipped ({kind})")
        return
    h = col.cpu().numpy()
    diff = float(np.abs(g - h).max())
    rel = float(np.linalg.norm(g - h) / (np.linalg.norm(h) + 1e-12))
    if diff == 0:
        verdict = "identical"
    elif kind in ("BF16", "F16", "F32") and rel < 1e-2:
        verdict = "OK (float rounding)"
    elif kind not in ("BF16", "F16", "F32") and rel < 0.15:
        verdict = f"OK for a {kind} quant (steer is a close approximation)"
    else:
        verdict = "MISMATCH - wrong model/checkpoint, or layer/neuron indexing differs"
    print(f"    verify vs GGUF ({kind}): max abs diff {diff:.3e}, relative {rel:.2%} -> {verdict}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="HF model id or local folder (the checkpoint the GGUF came from)")
    ap.add_argument("--steer", type=parse_steer, action="append", required=True, metavar="L:N:DOSE",
                    help="repeatable, e.g. --steer 14:8949:+1 --steer 13:7956:-1")
    ap.add_argument("--gguf", help="GGUF you will serve; checks its weights match and sets the arch hint")
    ap.add_argument("--out", help="output file (default: out/<model>_<steers>.gguf)")
    ap.add_argument("--calib-file", help="text file to estimate residual norms on (default: built-in mixed text)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--dry-run", action="store_true", help="print numbers, write nothing")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.hf)
    tcfg = getattr(cfg, "text_config", None) or cfg
    n_layers, n_embd = tcfg.num_hidden_layers, tcfg.hidden_size
    tok = AutoTokenizer.from_pretrained(args.hf)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = load_model(args.hf, dtype)
    layers = find_layers(model)
    if len(layers) != n_layers:
        sys.exit(f"found {len(layers)} decoder layers but config says {n_layers}; layer indexing would be off")
    print(f"model {args.hf}: {n_layers} layers, hidden {n_embd}, type {tcfg.model_type}")

    for L, n, _ in args.steer:
        if not 0 <= L < n_layers:
            sys.exit(f"layer {L} out of range 0..{n_layers - 1}")
        if L == 0:
            print("note: llama.cpp ignores direction.0 in control vectors; a layer-0 steer will have no effect")

    text = Path(args.calib_file).read_text(encoding="utf-8") if args.calib_file else DEFAULT_CALIB
    ids = tok(text, return_tensors="pt").to(model.device)
    # Hook the layers directly: HF's hidden_states[-1] has the final norm applied,
    # which would give the wrong scale for a steer on the last layer.
    resid = {}

    def grab(L):
        def hook(_mod, _inp, out):
            resid[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [layers[L].register_forward_hook(grab(L)) for L in {L for L, _, _ in args.steer}]
    with torch.no_grad():
        model(**ids)
    for h in handles:
        h.remove()
    print(f"calibration: {ids['input_ids'].shape[1]} tokens")

    reader = None
    if args.gguf:
        from gguf import GGUFReader
        reader = GGUFReader(args.gguf)

    dirs = {}
    for L, n, dose in args.steer:
        down, nm = find_down(layers[L], L)
        col, n_ff = neuron_column(down, n, n_embd)
        med = resid[L][0, 1:].float().norm(dim=-1).median().item()
        v = (dose * med * col / col.norm()).cpu()
        dirs[L] = dirs.get(L, torch.zeros(n_embd)) + v
        print(f"  L{L} N{n} dose {dose:+g}: median||resid||={med:.3f} ||{nm}[:,n]||={col.norm().item():.4f} "
              f"-> ||vec||={v.norm().item():.3f}")
        if reader is not None:
            verify(reader, L, n, col)

    if args.dry_run:
        print("dry run, nothing written")
        return

    hint = (gguf_arch(reader) if reader is not None else None) or tcfg.model_type
    if args.out:
        out = args.out
    else:
        stem = Path(args.hf.rstrip("/\\")).name.replace("/", "_")
        tag = "_".join(f"L{L}N{n}{'p' if d >= 0 else 'm'}{abs(d):g}" for L, n, d in args.steer)
        Path("out").mkdir(exist_ok=True)
        out = str(Path("out") / f"{stem}_{tag}.gguf")

    from gguf import GGUFWriter
    w = GGUFWriter(out, "controlvector")
    w.add_string("controlvector.model_hint", hint)
    w.add_uint32("controlvector.layer_count", n_layers)
    for L, v in sorted(dirs.items()):
        w.add_tensor(f"direction.{L}", v.numpy().astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  (layers {sorted(dirs)}, model_hint '{hint}')")


if __name__ == "__main__":
    main()
