"""
ComfyUI-Wan-TPU
===============

使用 diffusers torchax 优化模型在 TPU 上运行 Wan 2.1 视频生成。

注意: torchax.enable_globally() 只在需要运行 TPU 代码时调用，
不在模块导入时调用，以避免与 ComfyUI 的其他组件冲突。

Nodes:
  - Wan21TextEncoder: TPU 上运行 T5-XXL 编码 prompt
  - Wan21TPUSampler: TPU 上运行 Transformer 生成 latents  
  - Wan21TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为视频
  - Wan21TPUPipeline: 端到端 Pipeline

技术特点:
  - 2D Mesh: (dp=2, tp=4) 配置 8 个 TPU chips
  - Splash Attention: exp2 优化
  - K-Smooth: 减少 attention 数值溢出
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
# JAX/TPU 初始化 - 2D Mesh
# ============================================================================

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from .utils import setup_jax_cache, setup_pytree_registrations

print("[ComfyUI-Wan-TPU] Initializing JAX/TPU environment...")
setup_jax_cache()

# 默认使用 2D Mesh: dp=2, tp=4 for 8 chips
DEFAULT_DP = 2

try:
    devices = jax.devices('tpu')
    num_devices = len(devices)
    print(f"[ComfyUI-Wan-TPU] Detected {num_devices} TPU cores")
    
    # 创建 2D Mesh
    dp_dim = min(DEFAULT_DP, num_devices)
    tp_dim = num_devices // dp_dim
    mesh_devices = mesh_utils.create_device_mesh(
        (dp_dim, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"[ComfyUI-Wan-TPU] Created Mesh: dp={dp_dim}, tp={tp_dim}")
except RuntimeError:
    print("[ComfyUI-Wan-TPU] WARNING: No TPU detected, falling back to CPU")
    import jax.numpy as jnp
    devices = jax.devices('cpu')
    mesh = Mesh(jnp.array(devices).reshape(1, 1), ("dp", "tp"))


# ============================================================================
# Splash Attention (可选)
# ============================================================================

HAS_SPLASH_ATTENTION = False
try:
    from .splash_attention import sdpa_reference, tpu_splash_attention
    HAS_SPLASH_ATTENTION = True
    print("[ComfyUI-Wan-TPU] Splash Attention loaded")
except ImportError as e:
    print(f"[ComfyUI-Wan-TPU] WARNING: Splash Attention not available: {e}")


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
