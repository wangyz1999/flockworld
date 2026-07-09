"""FlockDiT: a flow-matching DiT world model for FlockWorld boids.

A faithful PyTorch reduction of the Solaris single-/multi-player world model
(JAX/Flax-nnx, ``solaris/src/models/{singleplayer,multiplayer}/world_model.py``),
adapted for boids:

* operates in **pixel space** (no VAE) but is channel/patch/resolution agnostic
  so swapping to VAE latents later is a config-only change;
* **no CLIP** -- the I2V cross-attention is removed;
* conditioned on **2D steering acceleration** ``(acc_x, acc_y)`` injected via
  adaLN modulation (replacing Solaris's mouse/keyboard action module);
* flow-matching objective with frame-level (Diffusion Forcing) timesteps;
* single-agent and multi-agent share one code path: tokens always carry a
  player axis ``P`` (``P=1`` single-agent) and are interleaved per frame for
  attention (Solaris token-interleaving), with an additive agent embedding when
  ``num_agents > 1`` (added once at the input by default; ``agent_embed_per_layer``
  re-injects it before every block, Solaris-style).

Faithful-to-Solaris pieces: 3D RoPE split (``rope_apply`` / ``apply_rope_mp``),
adaLN-zero DiT block (6 modulation params), sinusoidal timestep embedding,
block-causal attention with per-frame block size ``P*S``, and the
patchify/unpatchify Conv3d scheme.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- #
# Embeddings / norms
# --------------------------------------------------------------------------- #
def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Port of Solaris ``sinusoidal_embedding_1d`` (cos first, then sin)."""
    assert dim % 2 == 0, "Dimension must be even"
    half = dim // 2
    position = position.float()
    inv = torch.pow(
        torch.tensor(10000.0, device=position.device),
        -torch.arange(half, device=position.device, dtype=torch.float32) / half,
    )
    freqs = torch.outer(position, inv)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=1)


class RMSNorm(nn.Module):
    """Port of Solaris ``WanRMSNorm`` -- RMS over the last dim, learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight.to(dtype)


def _layernorm(dim: int, eps: float = 1e-6) -> nn.LayerNorm:
    """Non-affine LayerNorm, matching Solaris ``WanLayerNorm`` defaults."""
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)


# --------------------------------------------------------------------------- #
# 3D RoPE (faithful port of transformer_utils.rope_params / rope_apply)
# --------------------------------------------------------------------------- #
class RoPE3D(nn.Module):
    """Factorised rotary embedding over (T, H, W).

    Head dim ``d`` is split ``[d - 4*(d//6), 2*(d//6), 2*(d//6)]`` for the
    temporal / height / width axes, exactly as Solaris. Adjacent channel pairs
    form complex numbers (interleaved convention -> ``view_as_complex``).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        d = head_dim
        dim_t = d - 4 * (d // 6)
        dim_h = 2 * (d // 6)
        dim_w = 2 * (d // 6)
        assert dim_t + dim_h + dim_w == d, (dim_t, dim_h, dim_w, d)
        self.register_buffer("freqs_t", self._table(max_seq_len, dim_t, theta), persistent=False)
        self.register_buffer("freqs_h", self._table(max_seq_len, dim_h, theta), persistent=False)
        self.register_buffer("freqs_w", self._table(max_seq_len, dim_w, theta), persistent=False)

    @staticmethod
    def _table(max_seq_len: int, dim: int, theta: float) -> torch.Tensor:
        inv = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim)
        angles = torch.outer(torch.arange(max_seq_len).float(), inv)  # (L, dim//2)
        return torch.polar(torch.ones_like(angles), angles)  # complex64

    def grid_freqs(self, f: int, h: int, w: int) -> torch.Tensor:
        """Complex freqs ``(f*h*w, head_dim//2)`` for one frame grid (frame-major)."""
        ft, fh, fw = self.freqs_t[:f], self.freqs_h[:h], self.freqs_w[:w]
        grid = torch.cat(
            [
                ft[:, None, None, :].expand(f, h, w, ft.shape[-1]),
                fh[None, :, None, :].expand(f, h, w, fh.shape[-1]),
                fw[None, None, :, :].expand(f, h, w, fw.shape[-1]),
            ],
            dim=-1,
        )
        return grid.reshape(f * h * w, -1)

    @staticmethod
    def apply(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """x: (B, S, N, D) -> rotated. freqs_cis: (S, D//2) complex."""
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        xc = xc * freqs_cis[None, :, None, :]
        return torch.view_as_real(xc).reshape(*x.shape).type_as(x)


# --------------------------------------------------------------------------- #
# Self-attention with qk-RMSNorm + 3D RoPE
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs_cis, attn_mask, f, p, s):
        """x: (B, F*P*S, dim). RoPE applied per-agent on the (F,S) grid."""
        n = self.num_heads
        b = x.shape[0]
        q = rearrange(self.norm_q(self.q(x)), "b l (n d) -> b l n d", n=n)
        k = rearrange(self.norm_k(self.k(x)), "b l (n d) -> b l n d", n=n)
        v = rearrange(self.v(x), "b l (n d) -> b l n d", n=n)

        def rope(t):  # per-agent positions: (b,(f p s),n,d) -> (b p,(f s),n,d) -> back
            t = rearrange(t, "b (f p s) n d -> (b p) (f s) n d", f=f, p=p, s=s)
            t = RoPE3D.apply(t, freqs_cis)
            return rearrange(t, "(b p) (f s) n d -> b (f p s) n d", b=b, p=p, f=f, s=s)

        q, k = rope(q), rope(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, N, L, D)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out.transpose(1, 2), "b l n d -> b l (n d)")
        return self.o(out)


# --------------------------------------------------------------------------- #
# DiT block (adaLN-zero self-attn + FFN), no cross-attention
# --------------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.norm1 = _layernorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = _layernorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        # 6 adaLN-zero modulation params, init / sqrt(dim) (Solaris convention)
        self.modulation = nn.Parameter(torch.randn(1, 1, 1, 6, dim) / math.sqrt(dim))

    def forward(self, x, e, freqs_cis, attn_mask):
        """x: (B, F, P, S, dim); e: (B, F, P, 6, dim) timestep+action modulation."""
        f, p, s = x.shape[1], x.shape[2], x.shape[3]
        m = (self.modulation + e).unsqueeze(4).chunk(6, dim=3)  # 6 x (B,F,P,1,1,dim)
        m = [mi.squeeze(3) for mi in m]  # 6 x (B,F,P,1,dim)

        pack = lambda t: rearrange(t, "b f p s c -> b (f p s) c")
        unpack = lambda t: rearrange(t, "b (f p s) c -> b f p s c", f=f, p=p, s=s)

        h = self.norm1(x) * (1 + m[1]) + m[0]
        y = unpack(self.self_attn(pack(h), freqs_cis, attn_mask, f, p, s))
        x = x + y * m[2]

        h = self.norm2(x) * (1 + m[4]) + m[3]
        x = x + self.ffn(h) * m[5]
        return x


class OutputHead(nn.Module):
    """Final adaLN (2 params) + linear projection back to patch space."""

    def __init__(self, dim, out_channels, patch, eps=1e-6):
        super().__init__()
        self.norm = _layernorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch) * out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 1, 1, 2, dim) / math.sqrt(dim))

    def forward(self, x, e_bd):
        # x: (B, F, P, S, dim); e_bd: (B, F, P, dim)
        # chunk leaves a singleton param-axis at dim 3 that broadcasts over S.
        shift, scale = (self.modulation + e_bd.unsqueeze(3)).chunk(2, dim=3)
        x = self.norm(x) * (1 + scale) + shift
        return self.head(x)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class FlockDiT(nn.Module):
    """Action-conditioned flow-matching DiT.

    Single-agent ``forward(x, t, actions)``:
        x:       (B, F, C, H, W)   noised frames (context frames clean)
        t:       (B, F)            per-frame timestep in [0, 1000]
        actions: (B, F, A)         per-frame action (A=2: acc_x, acc_y)
    Multi-agent (``num_agents > 1``): x is (B, P, F, C, H, W); t is (B, P, F);
    actions is (B, P, F, A). Returns predicted velocity, same shape as ``x``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        ffn_dim: int = 2048,
        patch: int = 8,
        patch_t: int = 1,
        action_dim: int = 2,
        freq_dim: int = 256,
        num_agents: int = 1,
        local_attn_size: int = -1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        max_seq_len: int = 1024,
        grad_checkpointing: bool = False,
        agent_embed_per_layer: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.patch_size = (patch_t, patch, patch)
        self.freq_dim = freq_dim
        self.num_agents = num_agents
        self.local_attn_size = local_attn_size
        self.grad_checkpointing = grad_checkpointing
        self.agent_embed_per_layer = agent_embed_per_layer

        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_embedding = nn.Sequential(nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(dim, ffn_dim, heads, qk_norm, eps) for _ in range(depth)])
        self.head = OutputHead(dim, out_channels, self.patch_size, eps)
        self.rope = RoPE3D(self.head_dim, max_seq_len=max_seq_len)
        if num_agents > 1:
            self.agent_embed = nn.Embedding(num_agents, dim)

    # -- attention mask: block-causal across frames, full within a frame --- #
    def _block_causal_mask(self, n_frames, block, device):
        fi = torch.arange(n_frames, device=device).repeat_interleave(block)
        qf, kf = fi[:, None], fi[None, :]
        mask = kf <= qf
        if self.local_attn_size != -1:
            mask = mask & (kf >= qf - self.local_attn_size + 1)
        return mask

    # -- modulation from timestep + action --------------------------------- #
    def _modulation(self, t, actions, b, p, f):
        """t: (B,P,F); actions: (B,P,F,A) -> e0 (B,F,P,6,dim), e_bd (B,F,P,dim)."""
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.reshape(-1)))
        e = e + self.action_embedding(actions).reshape(b * p * f, self.dim)
        e0 = self.time_projection(e).reshape(b, p, f, 6, self.dim)
        e0 = rearrange(e0, "b p f r c -> b f p r c")
        e_bd = rearrange(e.reshape(b, p, f, self.dim), "b p f c -> b f p c")
        return e0, e_bd

    def forward(self, x, t, actions):
        single = x.dim() == 5
        if single:  # add a singleton player axis
            x = x.unsqueeze(1)
            t = t.unsqueeze(1)
            actions = actions.unsqueeze(1)
        b, p, f, _, h, w = x.shape
        pt, ph, pw = self.patch_size
        hp, wp = h // ph, w // pw
        fp, s = f // pt, hp * wp

        # patchify per agent -> tokens (B, F, P, S, dim)
        tok = rearrange(x, "b p f c h w -> (b p) c f h w")
        tok = self.patch_embed(tok)  # (B*P, dim, F', H', W')
        tok = rearrange(tok, "(b p) c f hh ww -> b f p (hh ww) c", b=b, p=p)
        # Agent identity embedding. Default: added once here at the input (standard;
        # the residual stream carries it). Opt-in agent_embed_per_layer re-injects it
        # before every block (Solaris-style), in case a one-shot add washes out over
        # depth -- kept as a flag so model size and embedding scheme stay separable.
        agent_bias = (self.agent_embed(torch.arange(p, device=x.device))[None, None, :, None, :]
                      if self.num_agents > 1 else None)
        if agent_bias is not None and not self.agent_embed_per_layer:
            tok = tok + agent_bias      # once at input (default)

        e0, e_bd = self._modulation(t, actions, b, p, fp)
        freqs = self.rope.grid_freqs(fp, hp, wp).to(tok.device)  # (F*S, hd//2)
        mask = self._block_causal_mask(fp, p * s, x.device)

        for block in self.blocks:
            if agent_bias is not None and self.agent_embed_per_layer:
                tok = tok + agent_bias      # per-layer identity re-injection (opt-in)
            if self.grad_checkpointing and self.training:
                # recompute block activations in backward -> less memory, ~1.3x compute
                tok = checkpoint(block, tok, e0, freqs, mask, use_reentrant=False)
            else:
                tok = block(tok, e0, freqs, mask)
        out = self.head(tok, e_bd)  # (B, F, P, S, patch_prod*out_c)

        out = rearrange(
            out,
            "b f p (hh ww) (p1 p2 p3 c) -> b p (f p1) c (hh p2) (ww p3)",
            hh=hp, ww=wp, p1=pt, p2=ph, p3=pw, c=self.out_channels,
        )
        return out.squeeze(1) if single else out
