"""
To start:
python train.py --model iris002 --data-dir data --seq-len 512 \
    --micro-bs 2 --grad-accum 16 --grad-checkpoint --max-steps 20000
to resume training:
python train.py --model iris002 --resume checkpoints/iris002/last.pt
"""

from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import Iris, IrisConfig, PRESETS


class BinDataset:
    def __init__(self, path: Path, seq_len: int):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run prepare_data.py first")
        self.path = path
        self.seq_len = seq_len
        self.n_tokens = path.stat().st_size // 2          # uint16 = 2 bytes
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{path} has only {self.n_tokens} tokens, need > {seq_len}")

    def batch(self, batch_size: int, device: torch.device, generator: np.random.Generator):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        ix = generator.integers(0, self.n_tokens - self.seq_len - 1, size=batch_size)
        x = np.stack([data[i: i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1: i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if device.type == "cuda":
            xt = xt.pin_memory().to(device, non_blocking=True)
            yt = yt.pin_memory().to(device, non_blocking=True)
        else:
            xt, yt = xt.to(device), yt.to(device)
        return xt, yt

def lr_at(step: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))
def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


@torch.no_grad()
def evaluate(model, ds: BinDataset, batch_size, iters, device, amp_ctx, gen) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = ds.batch(batch_size, device, gen)
        with amp_ctx:
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PRESETS), default="iris002")
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trade ~35% speed for ~50% activation memory")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--micro-bs", type=int, default=2, help="batch that lives on the GPU")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--opt-8bit", action="store_true", help="bitsandbytes AdamW8bit")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile; usually NOT worth it on a GTX 1650")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=str, default="checkpoints")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    device_type = device.type

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True    
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if device_type != "cuda":
        args.precision = "fp32"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 unsupported on this GPU (GTX 1650 is Turing) -> using fp16")
        args.precision = "fp16"

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    amp_ctx = (torch.autocast(device_type=device_type, dtype=amp_dtype)
               if args.precision != "fp32" else nullcontext())
    scaler = torch.amp.GradScaler(device_type, enabled=(args.precision == "fp16"))

    cfg: IrisConfig = PRESETS[args.model]
    cfg.vocab_size = args.vocab_size
    cfg.max_seq_len = max(args.seq_len, 1024)          
    cfg.dropout = args.dropout
    cfg.grad_checkpoint = args.grad_checkpoint

    model = Iris(cfg).to(device)
    raw_model = model

    run_name = args.run_name or args.model
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  PThNEA / {args.model.upper()}")
    print(f"  params            : {human(raw_model.num_params())} "
          f"({raw_model.num_params():,})")
    print(f"  non-embedding     : {human(raw_model.num_params(non_embedding=True))}")
    print(f"  dim/layers/heads  : {cfg.dim} / {cfg.n_layers} / {cfg.n_heads} "
          f"(kv={cfg.n_kv_heads}, GQA {cfg.n_rep}:1)")
    print(f"  seq_len           : {args.seq_len}")
    print(f"  tokens per step   : {args.micro_bs * args.grad_accum * args.seq_len:,}")
    print(f"  precision         : {args.precision} "
          f"(master weights + loss in fp32)")
    print(f"  grad checkpoint   : {args.grad_checkpoint}")
    print(f"  device            : {device}", end="")
    if device_type == "cuda":
        print(f"  [{torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB]")
    else:
        print()
    print("=" * 66)

    optimizer = raw_model.configure_optimizer(
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), device_type=device_type,
        use_8bit=args.opt_8bit,
    )

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_step = ck.get("step", 0) + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    if args.compile:
        print("compiling (first step will be slow)...")
        model = torch.compile(raw_model)
    data_dir = Path(args.data_dir)
    train_ds = BinDataset(data_dir / "train.bin", args.seq_len)
    val_path = data_dir / "val.bin"
    val_ds = BinDataset(val_path, args.seq_len) if val_path.exists() and \
        val_path.stat().st_size > 2 * (args.seq_len + 1) else None
    print(f"train tokens: {train_ds.n_tokens:,}"
          + (f" | val tokens: {val_ds.n_tokens:,}" if val_ds else " | no val set"))

    gen = np.random.default_rng(args.seed + start_step)
    model.train()
    step = start_step
    t0 = time.time()
    tok_per_step = args.micro_bs * args.grad_accum * args.seq_len
    running = None

    try:
        for step in range(start_step, args.max_steps):
            lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for micro in range(args.grad_accum):
                x, y = train_ds.batch(args.micro_bs, device, gen)
                with amp_ctx:
                    _, loss = model(x, targets=y)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                loss_accum += loss.item()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            else:
                gnorm = torch.tensor(0.0)

            scaler.step(optimizer)
            scaler.update()

            running = loss_accum if running is None else 0.9 * running + 0.1 * loss_accum
            if step % args.log_every == 0:
                dt = time.time() - t0
                t0 = time.time()
                tps = tok_per_step * args.log_every / dt if step > start_step else tok_per_step / dt
                mem = (torch.cuda.max_memory_allocated() / 1e9) if device_type == "cuda" else 0.0
                print(f"step {step:>6} | loss {loss_accum:.4f} | ema {running:.4f} "
                      f"| ppl {math.exp(min(20, loss_accum)):>8.1f} | lr {lr:.2e} "
                      f"| gnorm {float(gnorm):.2f} | {tps/1000:.1f}k tok/s "
                      f"| vram {mem:.2f}GB")
            if val_ds and args.eval_every and step > 0 and step % args.eval_every == 0:
                vgen = np.random.default_rng(args.seed + 99991)
                vloss = evaluate(model, val_ds, args.micro_bs, args.eval_iters,
                                 device, amp_ctx, vgen)
                print(f"  >> val loss {vloss:.4f} | val ppl {math.exp(min(20, vloss)):.1f}")
                if vloss < best_val:
                    best_val = vloss
                    save(out_dir / "best.pt", raw_model, optimizer, scaler, cfg,
                         step, best_val, args)
                    print(f"  >> new best, saved {out_dir/'best.pt'}")
                t0 = time.time()
            if args.save_every and step > 0 and step % args.save_every == 0:
                save(out_dir / "last.pt", raw_model, optimizer, scaler, cfg,
                     step, best_val, args)
                t0 = time.time()

    except KeyboardInterrupt:
        print("\ninterrupted - saving before exit")
    except torch.cuda.OutOfMemoryError:
        print("\n[OOM] Out of VRAM. Things that help, in order of effect:")
        print("  1) --grad-checkpoint            (biggest activation saving)")
        print("  2) --micro-bs 1 --grad-accum 32 (same effective batch)")
        print("  3) --seq-len 256                (attention memory is ~quadratic)")
        print("  4) --opt-8bit                   (halves AdamW state)")
        print("  5) --model iris001              (smaller model)")
        raise

    save(out_dir / "last.pt", raw_model, optimizer, scaler, cfg, step, best_val, args)
    print(f"finished. checkpoint -> {out_dir/'last.pt'}")


def save(path: Path, model, optimizer, scaler, cfg, step, best_val, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
        "args": vars(args),
    }, path)
    (path.with_suffix(".json")).write_text(json.dumps(
        {"config": asdict(cfg), "step": step, "best_val": best_val}, indent=2))


if __name__ == "__main__":
    main()
