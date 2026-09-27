"""
Mini GPT-OSS: the exact model built step by step in 01_mini_gpt_oss_from_scratch.ipynb,
packaged as a module so later notebooks can import it.

The only additions over notebook 01 are a few "record" switches that let us look inside:
  * Attention.record       -> keeps the attention percentages, including the sink column
  * MixtureOfExperts.record-> keeps the router's choices and probabilities per token
  * MiniGPTOSS.forward(..., return_hidden=True) -> also returns the residual stream after every floor
"""
import math
import os
import time
import urllib.request
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- config
@dataclass
class MiniGPTOSSConfig:
    vocab_size: int
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 32
    num_experts: int = 4
    experts_per_token: int = 2
    intermediate_size: int = 256
    swiglu_limit: float = 7.0
    rope_theta: float = 10000.0
    block_size: int = 128


# ----------------------------------------------------------------------------- tokenizer + data
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self):
        return len(self.chars)


# ----------------------------------------------------------------------------- building blocks
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, T: int, device):
        positions = torch.arange(T, device=device).float()
        angles = torch.outer(positions, self.inv_freq.to(device))
        return angles.cos(), angles.sin()

    @staticmethod
    def rotate(x, cos, sin):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, q, k):
        cos, sin = self.cos_sin(q.shape[1], q.device)
        return self.rotate(q, cos, sin), self.rotate(k, cos, sin)


def build_attention_mask(T: int, sliding_window: int, device):
    neg_inf = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(neg_inf, diagonal=1)
    if sliding_window > 0:
        mask = mask + torch.tril(neg_inf, diagonal=-sliding_window)
    return mask


class Attention(nn.Module):
    def __init__(self, cfg: MiniGPTOSSConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.q_mult = self.n_heads // self.n_kv
        self.head_dim = cfg.head_dim
        self.sliding_window = cfg.sliding_window if layer_idx % 2 == 0 else 0
        self.norm = RMSNorm(cfg.hidden_size)
        self.qkv = nn.Linear(cfg.hidden_size, (self.n_heads + 2 * self.n_kv) * self.head_dim, bias=True)
        self.out = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=True)
        self.sinks = nn.Parameter(torch.zeros(self.n_heads))
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.record = False
        self.last_weights = None   # (B, heads, T, T+1); last column = the sink

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm(x)
        q, k, v = torch.split(self.qkv(h), [self.n_heads * self.head_dim,
                                            self.n_kv * self.head_dim,
                                            self.n_kv * self.head_dim], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv, self.head_dim)
        v = v.view(B, T, self.n_kv, self.head_dim)
        q, k = self.rope(q, k)
        q = q.view(B, T, self.n_kv, self.q_mult, self.head_dim).permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.einsum("bgmqd,bgkd->bgmqk", q, k) * self.sm_scale
        scores = scores + build_attention_mask(T, self.sliding_window, x.device)
        sinks = self.sinks.view(1, self.n_kv, self.q_mult, 1, 1).expand(B, -1, -1, T, 1)
        scores = torch.cat([scores, sinks], dim=-1)
        full = torch.softmax(scores, dim=-1)
        if self.record:
            self.last_weights = full.detach().reshape(B, self.n_heads, T, T + 1)
        weights = full[..., :-1]
        mixed = torch.einsum("bgmqk,bgkd->bgmqd", weights, v)
        mixed = mixed.permute(0, 3, 1, 2, 4).reshape(B, T, self.n_heads * self.head_dim)
        return x + self.out(mixed)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class Expert(nn.Module):
    def __init__(self, cfg: MiniGPTOSSConfig):
        super().__init__()
        self.mlp1 = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=True)
        self.mlp2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        self.limit = cfg.swiglu_limit

    def forward(self, x):
        return self.mlp2(swiglu(self.mlp1(x), limit=self.limit))


class MixtureOfExperts(nn.Module):
    def __init__(self, cfg: MiniGPTOSSConfig):
        super().__init__()
        self.n_experts = cfg.num_experts
        self.k = cfg.experts_per_token
        self.norm = RMSNorm(cfg.hidden_size)
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=True)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])
        self.last_counts = None
        self.aux_loss = torch.tensor(0.0)
        self.record = False
        self.last_top_idx = None    # (B*T, k)
        self.last_top_w = None      # (B*T, k)
        self.last_probs = None      # (B*T, experts)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm(x).reshape(B * T, C)
        logits = self.router(h)
        top_vals, top_idx = logits.topk(self.k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)

        out = torch.zeros_like(h)
        counts = []
        for e, expert in enumerate(self.experts):
            token_ids, slot = (top_idx == e).nonzero(as_tuple=True)
            counts.append(token_ids.numel())
            if token_ids.numel() == 0:
                continue
            y = expert(h[token_ids]) * weights[token_ids, slot].unsqueeze(-1)
            out.index_add_(0, token_ids, y)
        self.last_counts = counts

        probs = torch.softmax(logits, dim=-1)
        frac_tokens = F.one_hot(top_idx, self.n_experts).sum(dim=1).float().mean(dim=0) / self.k
        self.aux_loss = self.n_experts * (frac_tokens * probs.mean(dim=0)).sum()
        if self.record:
            self.last_top_idx = top_idx.detach()
            self.last_top_w = weights.detach()
            self.last_probs = probs.detach()
        return x + out.view(B, T, C)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MiniGPTOSSConfig, layer_idx: int):
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.mlp = MixtureOfExperts(cfg)

    def forward(self, x):
        return self.mlp(self.attn(x))


class MiniGPTOSS(nn.Module):
    def __init__(self, cfg: MiniGPTOSSConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.unembedding = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, token_ids, targets=None, aux_weight: float = 0.01, return_hidden: bool = False):
        x = self.embedding(token_ids)
        hidden = [x]
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        logits = self.unembedding(self.norm(x))
        loss = None
        if targets is not None:
            main = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            fairness = torch.stack([b.mlp.aux_loss for b in self.blocks]).mean()
            loss = main + aux_weight * fairness
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def set_record(self, flag: bool):
        for b in self.blocks:
            b.attn.record = flag
            b.mlp.record = flag

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        expert_params = sum(p.numel() for b in self.blocks for p in b.mlp.experts.parameters())
        active = total - expert_params + expert_params * self.cfg.experts_per_token // self.cfg.num_experts
        return total, active


# ----------------------------------------------------------------------------- helpers
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: MiniGPTOSS, tokenizer: CharTokenizer, path: str):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "chars": tokenizer.chars}, path)


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = MiniGPTOSSConfig(**ckpt["config"])
    model = MiniGPTOSS(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), CharTokenizer(ckpt["chars"])


def train(model: MiniGPTOSS, data: torch.Tensor, device, steps: int = 1500, batch_size: int = 32,
          lr: float = 1e-3, log_every: int = 250):
    """The same training loop as notebook 01, for notebooks that need a model but have no checkpoint."""
    cfg = model.cfg
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    model.train()
    t0 = time.time()
    for step in range(steps):
        starts = torch.randint(0, len(data) - cfg.block_size - 1, (batch_size,))
        x = torch.stack([data[s: s + cfg.block_size] for s in starts]).to(device)
        y = torch.stack([data[s + 1: s + cfg.block_size + 1] for s in starts]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:>5} | loss {loss.item():.3f} | {time.time() - t0:5.0f}s")
    model.eval()
    return model


@torch.no_grad()
def generate(model: MiniGPTOSS, tokenizer: CharTokenizer, prompt: str, device, max_new_tokens: int = 300,
             temperature: float = 0.8, top_k=None):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return tokenizer.decode(ids[0].tolist())
