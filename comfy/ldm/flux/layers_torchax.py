"""
ComfyUI Flux Layers (Torchax TPU Version)

This module provides TPU-optimized layer implementations for Flux models,
including DoubleStreamBlock, SingleStreamBlock, and supporting components.

Key TPU optimizations:
1. Sharding constraints on input/output tensors
2. Optimized attention using Splash Attention
3. Avoiding inplace operations for XLA compatibility
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# 使用 torchax 版本的 math 模块
from .math_torchax import attention, rope
import comfy.ops
import comfy.ldm.common_dit

# ============================================================================
# Torchax/JAX imports
# ============================================================================
try:
    import jax
    from torchax import interop
    from jax.sharding import PartitionSpec as P
    
    mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)
    TORCHAX_AVAILABLE = True
except ImportError:
    TORCHAX_AVAILABLE = False
    mark_sharding = lambda x, p: x


# ============================================================================
# Position Embedding
# ============================================================================
class EmbedND(nn.Module):
    """N-dimensional positional embedding using RoPE."""
    
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


# ============================================================================
# Timestep Embedding
# ============================================================================
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    
    :param t: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


# ============================================================================
# MLP Embedder
# ============================================================================
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_layer = operations.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.silu = nn.SiLU()
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


# ============================================================================
# YakMLP (SwiGLU variant)
# ============================================================================
class YakMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=True, dtype=dtype, device=device)
        self.up_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=True, dtype=dtype, device=device)
        self.down_proj = operations.Linear(self.intermediate_size, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=False, yak_mlp=False, dtype=None, device=None, operations=None):
    """Build MLP layer based on configuration."""
    if yak_mlp:
        return YakMLP(hidden_size, mlp_hidden_dim, dtype=dtype, device=device, operations=operations)
    if mlp_silu_act:
        return nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim * 2, bias=False, dtype=dtype, device=device),
            SiLUActivation(),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=False, dtype=dtype, device=device),
        )
    else:
        return nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )


# ============================================================================
# Normalization
# ============================================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))

    def forward(self, x: Tensor):
        return comfy.ldm.common_dit.rms_norm(x, self.scale, 1e-6)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


# ============================================================================
# Self Attention
# ============================================================================
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_bias: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, bias=proj_bias, dtype=dtype, device=device)


# ============================================================================
# Modulation
# ============================================================================
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=bias, dtype=dtype, device=device)

    def forward(self, vec: Tensor) -> tuple:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    """Apply modulation to tensor (TPU-compatible, avoiding inplace ops)."""
    if modulation_dims is None:
        if m_add is not None:
            # 使用 torch.addcmul 替代 inplace 操作
            return torch.addcmul(m_add, tensor, m_mult)
        else:
            return tensor * m_mult
    else:
        # 创建新 tensor 避免 inplace 操作
        result = tensor.clone()
        for d in modulation_dims:
            result[:, d[0]:d[1]] = result[:, d[0]:d[1]] * m_mult[:, d[2]]
            if m_add is not None:
                result[:, d[0]:d[1]] = result[:, d[0]:d[1]] + m_add[:, d[2]]
        return result


# ============================================================================
# SiLU Activation (SwiGLU style)
# ============================================================================
class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


# ============================================================================
# Double Stream Block (Torchax TPU Version)
# ============================================================================
class DoubleStreamBlock(nn.Module):
    """
    Double Stream Block for Flux DiT (Torchax TPU Version).
    
    Processes image and text streams in parallel with cross-attention.
    TPU optimizations include sharding constraints and splash attention.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, 
                 flipped_img_txt=False, modulation=True, mlp_silu_act=False, proj_bias=True, 
                 yak_mlp=False, dtype=None, device=None, operations=None):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.modulation = modulation

        if self.modulation:
            self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)

        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, dtype=dtype, device=device, operations=operations)

        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.img_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp, dtype=dtype, device=device, operations=operations)

        if self.modulation:
            self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)

        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, dtype=dtype, device=device, operations=operations)

        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.txt_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp, dtype=dtype, device=device, operations=operations)

        self.flipped_img_txt = flipped_img_txt

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, 
                modulation_dims_img=None, modulation_dims_txt=None, transformer_options={}):
        """
        Forward pass (Torchax TPU Version).
        
        TPU optimization: Apply sharding constraints at block boundaries.
        """
        # TPU: Apply sharding constraint at block entry
        if TORCHAX_AVAILABLE:
            img = mark_sharding(img, P())
            txt = mark_sharding(txt, P())
        
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
            q = torch.cat((img_q, txt_q), dim=2)
            k = torch.cat((img_k, txt_k), dim=2)
            v = torch.cat((img_v, txt_v), dim=2)
            # run actual attention (uses TPU-optimized attention from math_torchax)
            attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
            # run actual attention (uses TPU-optimized attention from math_torchax)
            attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # calculate the img blocks (避免 inplace 操作)
        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

        # calculate the txt blocks (避免 inplace 操作)
        txt = txt + apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        txt = txt + apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


# ============================================================================
# Single Stream Block (Torchax TPU Version)
# ============================================================================
class SingleStreamBlock(nn.Module):
    """
    Single Stream Block for Flux DiT (Torchax TPU Version).
    
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        modulation=True,
        mlp_silu_act=False,
        bias=True,
        yak_mlp=False,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp_hidden_dim_first = self.mlp_hidden_dim
        self.yak_mlp = yak_mlp
        if mlp_silu_act:
            self.mlp_hidden_dim_first = int(hidden_size * mlp_ratio * 2)
            self.mlp_act = SiLUActivation()
        else:
            self.mlp_act = nn.GELU(approximate="tanh")

        if self.yak_mlp:
            self.mlp_hidden_dim_first *= 2
            self.mlp_act = nn.SiLU()

        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim_first, bias=bias, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=bias, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        if modulation:
            self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)
        else:
            self.modulation = None

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None, 
                transformer_options={}) -> Tensor:
        """
        Forward pass (Torchax TPU Version).
        """
        # TPU: Apply sharding constraint at block entry
        if TORCHAX_AVAILABLE:
            x = mark_sharding(x, P())
        
        if self.modulation:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention (uses TPU-optimized attention from math_torchax)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        
        # compute activation in mlp stream, cat again and run second linear layer
        if self.yak_mlp:
            mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        
        # 避免 inplace 操作
        x = x + apply_mod(output, mod.gate, None, modulation_dims)
        
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


# ============================================================================
# Last Layer
# ============================================================================
class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=bias, dtype=dtype, device=device))

    def forward(self, x: Tensor, vec: Tensor, modulation_dims=None) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        x = self.linear(x)
        return x
