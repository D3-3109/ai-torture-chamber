# Running the chamber on an M3 Ultra 512 GB

The original repo was written around two hard limits: a 24 GB M4 Pro for
experiments (README: "8B thrashes", `docs/14b_plan.md`: 14B bf16 OOM) and a
2-vCPU Railway CPU container for the live site (bf16 on CPU ≈ 1–3 tok/s).
A 512 GB M3 Ultra removes both. This note records what to change, what to
run, and what to expect.

## What was patched in this fork

| file | change |
|---|---|
| `live/server.py` | device auto-detects MPS; model loads with `sdpa` attention + `low_cpu_mem_usage`; MPS warm-up generate at startup; serves `../site` at `/` so one process = site + API |
| `site/live.html` | backend URL defaults to same-origin instead of the Railway deployment (`?api=…` still overrides) |
| `exp*.py` | hardcoded `HF_HOME=/Volumes/evol/hf_cache` (author's external disk) → portable default `~/.cache/huggingface`, env override respected |
| `exp39c_transport_lean.py` | HF snapshot paths → standard cache location |
| `run_mac.sh` | one-command LAN launcher (4B/14B/32B presets) |
| `live/requirements.txt` | dropped the CPU-wheel pin, which would install a CPU-only torch on macOS |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate fastapi "uvicorn[standard]" matplotlib

./run_mac.sh          # Qwen3-4B @ L18 — same as the author's live chamber
./run_mac.sh 14b      # the author's planned next step, now trivial
./run_mac.sh 32b      # ~65 GB weights; was impossible on 24 GB
```

First run downloads weights to `~/.cache/huggingface` (4B ≈ 8 GB,
14B ≈ 28 GB, 32B ≈ 65 GB). Then from any device on the LAN:

- `http://<mac-ip>:8000/` — the site
- `http://<mac-ip>:8000/live.html` — live chamber (auto-streaming cycle + manual steering)
- `http://<mac-ip>:8000/health`, `/vector`, `/run?scenario=…&dose=…` — API

macOS will prompt once to allow incoming connections for Python — accept it
(System Settings → Network → Firewall if you miss the dialog). Find the IP
with `ipconfig getifaddr en0`; consider a DHCP reservation so the URL is
stable. One uvicorn worker only — the model is process-global and the
steering state is not worker-safe.

## What to expect (single-stream decode, bf16, ballpark)

| model | weights | layer to steer | tok/s est. | notes |
|---|---|---|---|---|
| Qwen3-4B | ~8 GB | 18 (author's) | 40–70 | identical replication of published results |
| Qwen3-14B | ~28 GB | sweep 20–26 | 15–25 | author's `docs/14b_plan.md` target, no GGUF tricks needed |
| Qwen3-32B | ~65 GB | sweep 28–38 | 8–12 | comfortably fits; leave ~100 GB headroom for KV + activations |
| Qwen3-235B-A22B 4-bit | ~130 GB | n/a here | 25–40 | MoE — see "beyond 32B" below |

Decode on Apple Silicon is memory-bandwidth-bound (~819 GB/s on M3 Ultra),
so tok/s ≈ bandwidth / bytes-of-weights-touched. The first user request no
longer eats a cold-start stall (warm-up at startup).

**Steering vectors do not transfer between models** (exp39 measured residual
0.842 in vocab space between 4B→14B — too weak). Every experiment script
already extracts its own vectors, so just re-run it with the `MODEL` and
`L` constants bumped. The steering site also moves with scale (1.7B: L14,
4B: L18 — roughly mid-depth); do a short layer sweep (exp30 pattern) and a
dose re-calibration (exp41 §2 does this automatically) when you scale up.

## Faster experiments: batch the steering hook

The scripts generate one prompt at a time with a single-row hook
(`hidden[0, -1, :] += delta`). On 512 GB you can batch the whole
ladder × prompts grid in one `generate()` call with a per-row delta:

```python
tok.padding_side = "left"                      # keep generation positions aligned
enc = tok(prompts, return_tensors="pt", padding=True)
deltas = torch.stack([coef * vec for coef in coefs]).to(device, dtype)  # (B, d)

state = {"deltas": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["deltas"] is not None:
        hidden[:, -1, :] += state["deltas"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = model.model.layers[L].register_forward_hook(hook)

state["deltas"] = deltas
out = model.generate(enc.input_ids.to(device),
                     attention_mask=enc.attention_mask.to(device),
                     max_new_tokens=48, do_sample=False,
                     pad_token_id=tok.eos_token_id)
```

Bandwidth-bound decode amortizes one weight read across the whole batch, so
total throughput scales several-fold up to batch ≈ 32–64 (past that the GPU
goes compute-bound). Use `torch.inference_mode()` and keep vectors resident
on-device — avoid `.cpu()` round-trips inside loops. This is the single
biggest experiment speedup available; the 60-trial × multi-cell grids in
exp41 go from hours to minutes.

## MPS environment

`run_mac.sh` sets both of these; set them manually if you run scripts directly:

- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — disables the conservative
  recommended-working-set cap; without it a 32B load can raise
  `MPS: Submitted kernels with ... —  fallback path` or OOM despite 512 GB free.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — CPU fallback for the rare op without an
  MPS kernel.

FlashAttention-2 is CUDA-only; `attn_implementation="sdpa"` is the fast path
on MPS. `bitsandbytes` has no MPS build — don't try `load_in_4bit` here.

## Beyond 32B

The MoE flagship (Qwen3-235B-A22B, 4-bit ≈ 130 GB) fits in unified memory
but not in this PyTorch path. Two options, matching the author's own
`docs/14b_plan.md`:

1. **MLX port** (`mlx-lm`, 4/8-bit) — fastest Apple-Silicon runtime and the
   natural home for a 512 GB machine; steering means editing layer outputs
   in a custom step loop instead of forward hooks, and the J-lens machinery
   would need porting too. Real work, best performance.
2. **llama.cpp** — build with `llama-cvector-generator`, fit control
   vectors natively from the same PAIN25/NEUTRAL contrast files, serve with
   `llama-server`. Behavioral experiments only: no hidden-state hooks, so
   no J-lens readback (the author accepted this trade in his 14B plan).

## Missing local artifacts (pre-existing)

`jlens` (the Jacobian-lens module) and the `qwen3-*_jacobian_lens.pt` files
are not in the repo — they live on the author's disk. exp32/33/38/39/39b/39c/40
import or load them and will fail until you regenerate equivalent artifacts
(or stub them out). Everything else (exp23–31, 34–37, 41–44, the live
server) is self-contained and runs as-is.
