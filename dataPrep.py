from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Tokenizer
def get_tokenizer(name: str = "gpt2"):
    if name == "byte":
        return (lambda s: list(s.encode("utf-8"))), 256, 257

    try:
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return (lambda s: enc.encode_ordinary(s)), enc.eot_token, enc.n_vocab
    except Exception as e:
        print(f"[info] tiktoken unavailable for '{name}' ({e}); trying transformers")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    eot = tok.eos_token_id if tok.eos_token_id is not None else 0
    return (lambda s: tok(s, add_special_tokens=False)["input_ids"]), eot, tok.vocab_size

class BinWriter:

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.count = 0

    def write(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(self.f)
        self.count += arr.size

    def close(self):
        self.f.close()
        print(f"  wrote {self.path}  ({self.count:,} tokens, "
              f"{self.path.stat().st_size / 1e6:.1f} MB)")


def iter_text_file(path: Path, doc_sep: str):
    """Yields documents. doc_sep='\\n\\n' splits on blank lines; 'line' per line;
    'none' treats the whole file as one document (streamed in chunks)."""
    if doc_sep == "none":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return
                yield chunk
        return

    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if doc_sep == "line":
                if line.strip():
                    yield line
                continue
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def iter_hf(name: str, split: str, key: str, config: str | None, max_docs: int):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_docs and i >= max_docs:
            break
        text = row.get(key)
        if text:
            yield text


# main function, duh
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None,
                    help="path to a .txt file or a directory of .txt files")
    ap.add_argument("--hf-dataset", type=str, default=None)
    ap.add_argument("--hf-config", type=str, default=None)
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--text-key", type=str, default="text")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--doc-sep", choices=["blank", "line", "none"], default="blank")
    ap.add_argument("--val-frac", type=float, default=0.001,
                    help="fraction of documents held out for validation")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.input and not args.hf_dataset:
        ap.error("give either --input or --hf-dataset")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    encode, eot, vocab = get_tokenizer(args.tokenizer)
    print(f"tokenizer: {args.tokenizer}  vocab={vocab}  eot={eot}")
    if vocab > 65535:
        raise SystemExit("vocab > 65535 does not fit in uint16; use a smaller tokenizer")

    if args.hf_dataset:
        docs = iter_hf(args.hf_dataset, args.hf_split, args.text_key,
                       args.hf_config, args.max_docs)
    else:
        p = Path(args.input)
        files = sorted(p.glob("**/*.txt")) if p.is_dir() else [p]
        print(f"reading {len(files)} file(s)")

        def gen():
            n = 0
            for fp in files:
                for d in iter_text_file(fp, args.doc_sep):
                    yield d
                    n += 1
                    if args.max_docs and n >= args.max_docs:
                        return
        docs = gen()

    rng = np.random.default_rng(args.seed)
    train_w = BinWriter(out / "train.bin")
    val_w = BinWriter(out / "val.bin")

    n_docs = 0
    for doc in docs:
        ids = encode(doc)
        if not ids:
            continue
        ids.append(eot)               
        w = val_w if rng.random() < args.val_frac else train_w
        w.write(ids)
        n_docs += 1
        if n_docs % 20000 == 0:
            print(f"  {n_docs:,} docs | train {train_w.count:,} tok | val {val_w.count:,} tok")
    if val_w.count < 4096 and train_w.count > 0:
        print("[warn] validation split is tiny; bump --val-frac next time")

    train_w.close()
    val_w.close()
    print(f"done: {n_docs:,} documents -> {out.resolve()}")
    print(f"set --vocab-size to at least {vocab} when training "
          f"(recommended: {((vocab + 63) // 64) * 64})")


if __name__ == "__main__":
    main()
