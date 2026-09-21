"""
python inference.py --ckpt checkpoints/iris002/best.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from model import Iris, IrisConfig
def get_tokenizer(name: str = "gpt2"):
    """Must match whatever prepare_data.py used to build the .bin files."""
    if name == "byte":
        return ((lambda s: list(s.encode("utf-8"))),
                (lambda ids: bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")),
                256)

    try:
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return (lambda s: enc.encode_ordinary(s)), (lambda ids: enc.decode(ids)), enc.eot_token
    except Exception as e:
        print(f"[info] tiktoken unavailable for '{name}' ({e}); trying transformers",
              file=sys.stderr)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    eot = tok.eos_token_id if tok.eos_token_id is not None else 0
    return ((lambda s: tok(s, add_special_tokens=False)["input_ids"]),
            (lambda ids: tok.decode(ids, skip_special_tokens=True)), eot)
def load_model(ckpt_path: str, device: torch.device, dtype: torch.dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = IrisConfig(**ck["config"])
    cfg.grad_checkpoint = False
    cfg.dropout = 0.0

    model = Iris(cfg)
    state = ck["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}", file=sys.stderr)

    model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck.get("step", "?")

def filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                  repetition_penalty: float, prev_ids: torch.Tensor) -> torch.Tensor:
    logits = logits.float()

    if repetition_penalty and repetition_penalty != 1.0 and prev_ids.numel():
        for b in range(logits.size(0)):
            uniq = torch.unique(prev_ids[b])
            vals = logits[b, uniq]
            logits[b, uniq] = torch.where(vals > 0, vals / repetition_penalty,
                                          vals * repetition_penalty)

    if temperature <= 0:                   
        return logits

    logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = probs - F.softmax(sorted_logits, dim=-1) > top_p 
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    return logits


@torch.inference_mode()
def generate(model, cfg, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40,
             top_p=0.95, repetition_penalty=1.1, eot_id=None, device="cuda",
             stream_cb=None):
    model_dtype = next(model.parameters()).dtype
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    B, T = idx.shape

    budget = min(cfg.max_seq_len, T + max_new_tokens)
    if T >= budget:
        raise ValueError(f"prompt of {T} tokens leaves no room "
                         f"(max_seq_len={cfg.max_seq_len})")

    model.setup_cache(B, budget, device, model_dtype)
    try:
        logits, _ = model(idx, start_pos=0, use_cache=True)
        pos = T
        out_ids = []

        for _ in range(budget - T):
            logits = filter_logits(logits[:, -1, :], temperature, top_k, top_p,
                                   repetition_penalty, idx)
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            tok = int(nxt[0, 0])
            if eot_id is not None and tok == eot_id:
                break

            out_ids.append(tok)
            idx = torch.cat([idx, nxt], dim=1)
            if stream_cb:
                stream_cb(out_ids)
            logits, _ = model(nxt, start_pos=pos, use_cache=True)
            pos += 1

        return out_ids
    finally:
        model.clear_cache()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--chat", action="store_true", help="interactive prompt loop")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=40, help="0 disables")
    ap.add_argument("--top-p", type=float, default=0.95, help="1.0 disables")
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--bench", action="store_true", help="report tokens/sec")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if (args.dtype == "fp16" and device.type == "cuda") else torch.float32

    encode, decode, eot = get_tokenizer(args.tokenizer)
    model, cfg, step = load_model(args.ckpt, device, dtype)

    print(f"loaded {args.ckpt} | step {step} | {model.num_params()/1e6:.1f}M params "
          f"| {cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h (kv={cfg.n_kv_heads}) "
          f"| {dtype} on {device}")

    def run(prompt: str):
        ids = encode(prompt) or [eot]
        printed = {"n": 0}

        def cb(out_ids):
            if args.no_stream:
                return
            text = decode(out_ids)
            sys.stdout.write(text[printed["n"]:])
            sys.stdout.flush()
            printed["n"] = len(text)

        t0 = time.time()
        out = generate(model, cfg, ids,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.repetition_penalty,
                       eot_id=eot, device=device,
                       stream_cb=None if args.no_stream else cb)
        dt = time.time() - t0

        if args.no_stream:
            print(decode(out), end="")
        print()
        if args.bench:
            print(f"[{len(out)} tokens in {dt:.2f}s = {len(out)/max(dt,1e-9):.1f} tok/s]")

    if args.chat:
        print("interactive mode - blank line or Ctrl-C to quit\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt.strip():
                break
            print(prompt, end="")
            run(prompt)
            print()
        return

    prompt = args.prompt if args.prompt is not None else "Once upon a time"
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- sample {i+1}/{args.num_samples} ---")
        print(prompt, end="")
        run(prompt)


if __name__ == "__main__":
    main()
