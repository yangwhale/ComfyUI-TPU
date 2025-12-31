"""
ComfyUI Flux.2 TPU - 工具模块
============================

包含:
  - 分片策略配置
  - 权重分片函数
  - PyTree 注册
  - JAX 配置
"""

import re

import jax
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node


# ============================================================================
# Transformer 分片策略
# ============================================================================
# 规则：输出投影 ('tp', None)，输入投影 (None, 'tp')

TRANSFORMER_SHARDINGS = {
    # Double-stream Blocks - Attention
    r'transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_out.0.weight': (None, 'tp'),
    r'transformer_blocks.*.attn.add_q_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_k_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_v_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_add_out.weight': (None, 'tp'),
    # Double-stream Blocks - FeedForward
    r'transformer_blocks.*.ff.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff.linear_out.weight': (None, 'tp'),
    r'transformer_blocks.*.ff_context.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff_context.linear_out.weight': (None, 'tp'),
    # Single-stream Blocks
    r'single_transformer_blocks.*.attn.to_qkv_mlp_proj.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_out.weight': (None, 'tp'),
    # Embedders & Projections
    r'x_embedder.weight': ('tp', None),
    r'context_embedder.weight': ('tp', None),
    r'proj_out.weight': (None, 'tp'),
    # Modulation
    r'double_stream_modulation_img.linear.weight': ('tp', None),
    r'double_stream_modulation_txt.linear.weight': ('tp', None),
    r'single_stream_modulation.linear.weight': ('tp', None),
    # Time + Guidance Embedding
    r'time_guidance_embed.timestep_embedder.linear_1.weight': ('tp', None),
    r'time_guidance_embed.timestep_embedder.linear_2.weight': (None, 'tp'),
    r'time_guidance_embed.guidance_embedder.linear_1.weight': ('tp', None),
    r'time_guidance_embed.guidance_embedder.linear_2.weight': (None, 'tp'),
}

# VAE 不分片（使用 replicate）
VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}


# ============================================================================
# 权重分片函数
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh, debug=False):
    """
    按模式匹配应用权重分片。
    
    Args:
        weight_dict: 权重字典 {name: tensor}
        sharding_dict: 分片规则 {pattern: (axis0, axis1)}
        mesh: JAX Mesh
        debug: 是否打印详细信息
    
    Returns:
        分片后的权重字典
    """
    result = {}
    sharded_count = replicated_count = 0
    sharded_bytes = replicated_bytes = 0
    
    for k, v in weight_dict.items():
        tensor_bytes = np.prod(v.shape) * 2 if hasattr(v, 'shape') else 0
            
        if isinstance(v, torch.Tensor):
            with jax.default_device("cpu"):
                v = v.to("jax")
        
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.fullmatch(pattern, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                sharded_count += 1
                sharded_bytes += tensor_bytes
                if debug:
                    print(f"  ✓ SHARDED: {k} -> {sharding}")
                break
        
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes
        
        result[k] = v
    
    print(f"  分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    return result


def move_module_to_xla(env, module):
    """
    将 PyTorch 模块权重转换为 torchax tensor。
    
    Args:
        env: torchax 环境
        module: PyTorch 模块
    """
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# PyTree 注册
# ============================================================================

def setup_pytree_registrations():
    """
    注册必要的 PyTree 节点以支持 JAX 转换。
    
    注册的类型:
      - BaseModelOutputWithPastAndCrossAttentions (transformers)
      - DecoderOutput (diffusers VAE)
      - AutoencoderKLOutput (diffusers VAE)
    """
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
    from diffusers.models.autoencoders import vae as diffusers_vae
    from transformers import modeling_outputs
    
    print("注册 PyTree 节点...")
    
    def flatten(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten(aux, children):
        return aux(*children)
    
    classes = [
        (modeling_outputs.BaseModelOutputWithPastAndCrossAttentions, "BaseModelOutputWithPastAndCrossAttentions"),
        (diffusers_vae.DecoderOutput, "DecoderOutput"),
        (diffusers_modeling_outputs.AutoencoderKLOutput, "AutoencoderKLOutput"),
    ]
    
    for cls, name in classes:
        try:
            register_pytree_node(cls, flatten, unflatten)
            print(f"  - {name} 已注册")
        except ValueError:
            print(f"  - {name} 已存在")


# ============================================================================
# JAX 配置
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存以加速后续编译。"""
    import os
    cache_dir = os.path.expanduser("~/.cache/jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"✓ JAX 编译缓存: {cache_dir}")
