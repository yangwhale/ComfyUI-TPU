"""
ComfyUI Flux Math Operations (Torchax TPU Version)

This module provides TPU-optimized math operations for Flux models,
including attention and RoPE (Rotary Position Embedding).

Key TPU optimizations:
1. Splash Attention for long sequences (>20k tokens)
2. K-smoothing for numerical stability
3. JAX-compatible implementations
"""

import torch
from einops import rearrange
from torch import Tensor

# ============================================================================
# Torchax/JAX imports
# ============================================================================
try:
    import jax
    import jax.numpy as jnp
    from torchax import interop
    from jax.sharding import PartitionSpec as P
    
    # 创建 mark_sharding 函数
    mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)
    TORCHAX_AVAILABLE = True
except ImportError:
    TORCHAX_AVAILABLE = False
    mark_sharding = lambda x, p: x

# ============================================================================
# 配置常量
# ============================================================================
USE_K_SMOOTH = True  # K-smoothing for numerical stability
SPLASH_ATTENTION_THRESHOLD = 20000  # 序列长度阈值，超过使用 Splash Attention


# ============================================================================
# TPU Splash Attention (长序列优化)
# ============================================================================
def tpu_splash_attention(query, key, value, mesh=None, scale=None):
    """
    TPU Splash Attention 实现。
    
    对于超长序列 (>20k tokens)，使用 TPU 专用的 Splash Attention 算法，
    具有更好的内存效率和性能。
    
    Args:
        query: Query tensor [B, H, S, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        mesh: JAX mesh for sharding
        scale: Attention scale factor
    """
    try:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
        from jax.experimental.pallas.ops.tpu.splash_attention import BlockSizes
        
        # 配置块大小
        block_sizes = BlockSizes(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_kv_dkv=128,
            block_kv_dq=128,
            block_q_dq=128,
        )
        
        # 计算 scale
        if scale is None:
            scale = query.shape[-1] ** -0.5
        
        # 运行 Splash Attention
        result = splash_attention_kernel(
            query, key, value,
            scale=scale,
            block_sizes=block_sizes,
        )
        
        return result
        
    except ImportError:
        # 回退到标准 attention
        return sdpa_reference(query, key, value, scale=scale)


def sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0, 
                   is_causal=False, scale=None, enable_gqa=False):
    """
    参考实现的 Scaled Dot-Product Attention。
    
    适用于短序列或作为 Splash Attention 的回退。
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    
    # 计算注意力分数
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # 应用 mask
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    
    if is_causal:
        L, S = query.shape[-2], key.shape[-2]
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), 
            diagonal=1
        )
        attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))
    
    # Softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # Dropout
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    # 计算输出
    output = torch.matmul(attn_weight, value)
    
    return output


# ============================================================================
# Torchax-optimized Attention
# ============================================================================
def attention_torchax(q: Tensor, k: Tensor, v: Tensor, pe: Tensor = None,
                      mask=None, transformer_options={},
                      env=None, mesh=None) -> Tensor:
    """
    TPU 优化的 Attention 实现。
    
    根据序列长度自动选择最优算法:
    - 长序列 (>20k): Splash Attention
    - 短序列: 标准 SDPA
    
    Args:
        q: Query tensor [B, H, S, D]
        k: Key tensor [B, H, S, D]
        v: Value tensor [B, H, S, D]
        pe: Positional embedding for RoPE
        mask: Attention mask
        transformer_options: Additional options
        env: Torchax environment
        mesh: JAX mesh for sharding
    """
    # 应用 RoPE
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    
    heads = q.shape[1]
    seq_len = k.shape[2]
    
    # 计算 scale
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5
    
    # 根据序列长度选择算法
    if TORCHAX_AVAILABLE and seq_len > SPLASH_ATTENTION_THRESHOLD:
        # 长序列: 使用 Splash Attention
        if env is not None and mesh is not None:
            # 转换为 JAX tensors
            jquery, jkey, jvalue = env.t2j_iso((q, k, v))
            
            # K-smoothing for numerical stability
            if USE_K_SMOOTH:
                jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
            
            # 运行 Splash Attention
            result = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
            
            # 转换回 PyTorch tensor
            x = env.j2t_iso(result)
        else:
            # 无 JAX 环境，使用参考实现
            if USE_K_SMOOTH:
                k = k - k.mean(dim=2, keepdim=True)
            x = sdpa_reference(q, k, v, attn_mask=mask, scale=scale)
    else:
        # 短序列: 使用标准 SDPA
        x = sdpa_reference(q, k, v, attn_mask=mask, scale=scale)
    
    # Reshape output: [B, H, S, D] -> [B, S, H*D]
    x = rearrange(x, 'b h s d -> b s (h d)')
    
    return x


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, 
              mask=None, transformer_options={}) -> Tensor:
    """
    主 Attention 入口函数 (Torchax TPU Version)。
    
    这是对原始 attention 函数的替代，自动检测 TPU 环境并选择最优实现。
    """
    # 尝试从 transformer_options 获取 torchax 环境
    env = transformer_options.get('torchax_env', None)
    mesh = transformer_options.get('torchax_mesh', None)
    
    # 尝试从全局导入获取
    if env is None and TORCHAX_AVAILABLE:
        try:
            import torchax
            env = torchax.default_env()
        except:
            pass
    
    return attention_torchax(q, k, v, pe, mask, transformer_options, env, mesh)


# ============================================================================
# RoPE (Rotary Position Embedding)
# ============================================================================
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    计算 Rotary Position Embedding。
    
    在 TPU 上运行时，使用 JAX 进行计算。
    """
    assert dim % 2 == 0
    
    # TPU/JAX 兼容的设备处理
    device = pos.device
    
    # 使用 float64 计算以保持精度
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=device)


def apply_rope1(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    应用 RoPE 到单个 tensor。
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]  # 避免 inplace 操作

    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple:
    """
    应用 RoPE 到 Query 和 Key tensors。
    """
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


# ============================================================================
# Optimized Attention (兼容原始接口)
# ============================================================================
def optimized_attention_torchax(q, k, v, heads, skip_reshape=False, mask=None, 
                                 transformer_options={}):
    """
    优化的 Attention 实现，兼容 ComfyUI 原始接口。
    
    这是对 comfy.ldm.modules.attention.optimized_attention 的 TPU 替代。
    """
    if not skip_reshape:
        # Reshape: [B, S, H*D] -> [B, H, S, D]
        b, s, _ = q.shape
        d = q.shape[-1] // heads
        q = q.view(b, s, heads, d).transpose(1, 2)
        k = k.view(b, s, heads, d).transpose(1, 2)
        v = v.view(b, s, heads, d).transpose(1, 2)
    
    # 计算 attention
    x = attention_torchax(q, k, v, pe=None, mask=mask, 
                          transformer_options=transformer_options)
    
    return x
