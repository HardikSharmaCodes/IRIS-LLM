from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

@dataclass
class IrisConfig:
    vocab_size: int = 50304        
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4     
    ffn_hidden: int = 3072       
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0           
    tie_embeddings: bool = True
    grad_checkpoint: bool = False    

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, "dim must divide evenly into n_heads"
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        return self.n_heads // self.n_kv_heads
PRESETS = {
    "iris001": IrisConfig(
        dim=512, n_layers=8, n_heads=8, n_kv_heads=2,
        ffn_hidden=1536, max_seq_len=1024,
    ),
    "iris002": IrisConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        ffn_hidden=3072, max_seq_len=1024,
    ),
}

class RMSNorm(nn.Module):
    """Layer norm without the mean subtraction. One reduction instead of two."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The reduction is done in fp32 even under AMP: this is the one place where fp16 accumulation actually destabilises training.
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.to(dtype)) * self.weight
def build_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                     device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin once. Positional encoding then costs two multiplies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:        (B, H, T, head_dim)
    cos/sin:  (T, head_dim/2)
    Half-split rotation (LLaMA/HF convention).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)   
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv*n_rep, T, D) via expand, no data copy until reshape."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)
class EfficientSelfAttention(nn.Module):
    """Grouped-query attention + RoPE + SDPA (flash / mem-efficient backend)."""

    def __init__(self, cfg: IrisConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if use_cache and self.cache_k is not None:
            self.cache_k[:B, :, start_pos:start_pos + T] = k
            self.cache_v[:B, :, start_pos:start_pos + T] = v
            k = self.cache_k[:B, :, : start_pos + T]
            v = self.cache_v[:B, :, : start_pos + T]

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        causal = T > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
class SwiGLU(nn.Module):
    def __init__(self, cfg: IrisConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
class Block(nn.Module):
    def __init__(self, cfg: IrisConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = EfficientSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.resid_drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x, cos, sin, start_pos: int = 0, use_cache: bool = False):
        x = x + self.resid_drop(self.attn(self.attn_norm(x), cos, sin, start_pos, use_cache))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x
# model
class Iris(nn.Module):
    def __init__(self, cfg: IrisConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual output projections (GPT-2 trick): keeps the
        # residual stream variance from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                start_pos: int = 0, use_cache: bool = False):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, (
            f"sequence position {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.emb_drop(self.tok_emb(idx))

        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = ckpt_fn(blk, x, cos, sin, start_pos, use_cache, use_reentrant=False)
            else:
                x = blk(x, cos, sin, start_pos, use_cache)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss
        logits = self.lm_head(x[:, -1:, :])
        return logits, None
    @torch.no_grad()
    def setup_cache(self, batch_size: int, max_seq_len: int, device, dtype):
        for blk in self.blocks:
            a = blk.attn
            shape = (batch_size, a.n_kv_heads, max_seq_len, a.head_dim)
            a.cache_k = torch.zeros(shape, device=device, dtype=dtype)
            a.cache_v = torch.zeros(shape, device=device, dtype=dtype)

    def clear_cache(self):
        for blk in self.blocks:
            blk.attn.cache_k = None
            blk.attn.cache_v = None

    # ------------------------- optimizer construction ---------------------- #
    def configure_optimizer(self, lr: float, weight_decay: float, betas, device_type: str,
                            use_8bit: bool = False):
        """Matrices get weight decay; norms/embedding-gains/biases do not."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        if use_8bit:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)
            except ImportError:
                print("[warn] bitsandbytes not installed, falling back to torch AdamW")

        # foreach=False keeps peak memory down on a 4GB card (no giant temporaries).
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, foreach=False)
