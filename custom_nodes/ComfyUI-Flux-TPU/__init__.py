"""
ComfyUI-Flux-TPU: 使用 diffusers torchax 优化模型在 TPU 上运行 Flux.2

注意：torchax.enable_globally() 只在需要运行 TPU 代码时调用，
     不在模块导入时调用，以避免与 ComfyUI 的其他组件冲突。
"""

import os
import warnings
import logging

# 环境配置
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from .utils import setup_jax_cache, setup_pytree_registrations

# ============================================================================
# JAX/TPU 初始化（不启用 torchax）
# ============================================================================

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
    devices = jax.devices('cpu')
    tp_dim = 1
    mesh = Mesh(jax.numpy.array(devices).reshape(1,), ("tp",))

# ============================================================================
# Splash Attention（延迟导入）
# ============================================================================

HAS_SPLASH_ATTENTION = False
try:
    from .splash_attention import tpu_splash_attention, sdpa_reference
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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'mesh', 'HAS_SPLASH_ATTENTION']
