"""
ComfyUI Wan 2.1 TPU - 工具模块
==============================

包含:
  - 分片策略配置
  - 权重分片函数
  - PyTree 注册
  - 视频处理工具
  - JAX 配置
"""

import os
import re

import jax
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node


# ============================================================================
# 视频生成默认参数
# ============================================================================

# 720P 默认参数
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_FLOW_SHIFT = 5.0  # 5.0 for 720P, 3.0 for 480P


# ============================================================================
# Text Encoder 分片策略 (T5-XXL)
# ============================================================================

TEXT_ENCODER_SHARDINGS = {
    r'shared\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight': (None, ('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight': (None, ('dp', 'tp'),),
}


# ============================================================================
# Transformer 分片策略 (WanTransformer3DModel)
# ============================================================================

TRANSFORMER_SHARDINGS = {
    # Condition Embedder
    r'condition_embedder\.time_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.time_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.time_embedder\.linear_2\.weight': (None, 'tp',),
    r'condition_embedder\.text_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.text_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.text_embedder\.linear_2\.weight': (None, 'tp',),
    # Self Attention
    r'blocks\.\d+\.attn1\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_out\.\d+\.weight': (None, 'tp',),
    # Cross Attention
    r'blocks\.\d+\.attn2\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_out\.\d+\.weight': (None, 'tp',),
    # FFN
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.weight': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.bias': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.weight': (None, 'tp',),
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
        sharding_dict: 分片规则 {pattern: (axis0, axis1, ...)}
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
      - DiagonalGaussianDistribution (diffusers VAE)
    """
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
    from diffusers.models.autoencoders import vae as diffusers_vae
    from transformers import modeling_outputs
    
    print("注册 PyTree 节点...")
    
    def flatten(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten(aux, children):
        return aux(*children)
    
    # 标准模型输出
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
    
    # DiagonalGaussianDistribution 需要特殊处理
    def flatten_gaussian(obj):
        return (obj.parameters, obj.mean, obj.logvar, obj.deterministic,
                obj.std, obj.var), None
    
    def unflatten_gaussian(aux, children):
        obj = object.__new__(diffusers_vae.DiagonalGaussianDistribution)
        obj.parameters = children[0]
        obj.mean = children[1]
        obj.logvar = children[2]
        obj.deterministic = children[3]
        obj.std = children[4]
        obj.var = children[5]
        return obj
    
    try:
        register_pytree_node(
            diffusers_vae.DiagonalGaussianDistribution,
            flatten_gaussian,
            unflatten_gaussian
        )
        print("  - DiagonalGaussianDistribution 已注册")
    except ValueError:
        print("  - DiagonalGaussianDistribution 已存在")


# ============================================================================
# 视频处理工具
# ============================================================================

def prepare_video_for_export(video, target_frames):
    """
    准备视频 tensor 用于导出。
    
    输入: JAX VAE 输出格式 [B, T, H, W, C]
    输出: numpy array [T, H, W, C] (float32, 范围 [0, 1])
    
    Args:
        video: 视频 tensor 或 numpy array
        target_frames: 目标帧数（用于验证）
        
    Returns:
        numpy array: [T, H, W, C] 格式的 float32 视频
    """
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        if video.dim() == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = video.permute(0, 4, 1, 2, 3)  # -> [B, C, T, H, W]
            
            batch_vid = video[0]  # [C, T, H, W]
            batch_vid = batch_vid.permute(1, 0, 2, 3)  # -> [T, C, H, W]
            batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)  # denormalize
            video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()  # -> [T, H, W, C]
            
        elif video.dim() == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video.permute(1, 0, 2, 3)
                batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
                video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()
            elif video.shape[-1] == 3:  # [T, H, W, C]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.cpu().float().numpy()
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = np.transpose(video, (0, 4, 1, 2, 3))
            
            batch_vid = video[0]  # [C, T, H, W]
            batch_vid = np.transpose(batch_vid, (1, 0, 2, 3))  # -> [T, C, H, W]
            
            if batch_vid.min() < 0:
                batch_vid = np.clip(batch_vid * 0.5 + 0.5, 0, 1)
            
            video = np.transpose(batch_vid, (0, 2, 3, 1))  # -> [T, H, W, C]
            
        elif video.ndim == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                video = np.transpose(video, (1, 0, 2, 3))
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
                video = np.transpose(video, (0, 2, 3, 1))
            elif video.shape[-1] == 3:  # [T, H, W, C]
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
        
        video = video.astype(np.float32)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    return video


# ============================================================================
# JAX 配置
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存以加速后续编译。"""
    cache_dir = os.path.expanduser("~/.cache/jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"✓ JAX 编译缓存: {cache_dir}")
