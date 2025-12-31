"""
ComfyUI-Flux-TPU
================

使用 diffusers torchax 优化模型在 TPU 上运行 Flux.2。

注意: torchax.enable_globally() 只在需要运行 TPU 代码时调用，
不在模块导入时调用，以避免与 ComfyUI 的其他组件冲突。

Nodes:
  - Flux2TextEncoder: CPU 上运行 Mistral3 编码 prompt
  - Flux2TPUSampler: TPU 上运行 Transformer 生成 latents  
  - Flux2TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为图像
  - Flux2TPUPipeline: 端到端 Pipeline
"""

import logging
import os
import warnings

# ============================================================================
# 环境配置
# ============================================================================

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


# ============================================================================
# JAX/TPU 初始化
# ============================================================================

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from .utils import setup_jax_cache, setup_pytree_registrations

print("[ComfyUI-Flux-TPU] Initializing JAX/TPU environment...")
setup_jax_cache()

try:
    devices = jax.devices('tpu')
    print(f"[ComfyUI-Flux-TPU] Detected {len(devices)} TPU cores")
    tp_dim = len(devices)
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"[ComfyUI-Flux-TPU] Created Mesh: tp={tp_dim}")
except RuntimeError:
    print("[ComfyUI-Flux-TPU] WARNING: No TPU detected, falling back to CPU")
    import jax.numpy as jnp
    devices = jax.devices('cpu')
    tp_dim = 1
    mesh = Mesh(jnp.array(devices).reshape(1,), ("tp",))


# ============================================================================
# Splash Attention (可选)
# ============================================================================

HAS_SPLASH_ATTENTION = False
try:
    from .splash_attention import sdpa_reference, tpu_splash_attention
    HAS_SPLASH_ATTENTION = True
    print("[ComfyUI-Flux-TPU] Splash Attention loaded")
except ImportError as e:
    print(f"[ComfyUI-Flux-TPU] WARNING: Splash Attention not available: {e}")


# ============================================================================
# PyTree 注册
# ============================================================================

setup_pytree_registrations()


# ============================================================================
# 导出 Nodes
# ============================================================================

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'mesh',
    'HAS_SPLASH_ATTENTION',
]
