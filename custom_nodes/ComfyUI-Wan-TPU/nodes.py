"""
ComfyUI Wan 2.1 TPU Nodes
=========================

使用 diffusers 的 torchax 优化模型在 TPU 上运行 Wan 2.1 视频生成。
基于 gpu-tpu-pedia/tpu/Wan2.1/generate_diffusers_torchax_staged 实现。

Nodes:
  - Wan21TextEncoder: TPU 上运行 T5-XXL 编码 prompt
  - Wan21TPUSampler: TPU 上运行 Transformer 生成 latents
  - Wan21TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为视频
  - Wan21TPUPipeline: 端到端视频生成 Pipeline

核心设计（Hybrid 方案）：
  - 使用 `enable_globally()` 保持 Mode 栈激活（解决 XLA tensor 逃逸问题）
  - 模型缓存后权重保持 XLA 状态
  - 节点返回值必须转为 CPU tensor（确保与 ComfyUI 兼容）
  - 参考：docs/torchax_comfyui_integration.md
"""

import functools
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Hybrid 方案：使用 enable_globally() 保持 Mode 栈激活
# 这样缓存的 XLA 模型可以在后续调用中正常工作

# 延迟导入 utils（支持独立测试）
try:
    from .utils import (
        TEXT_ENCODER_SHARDINGS,
        TRANSFORMER_SHARDINGS,
        VAE_DECODER_SHARDINGS,
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        DEFAULT_FRAMES,
        DEFAULT_FPS,
        DEFAULT_FLOW_SHIFT,
        move_module_to_xla,
        prepare_video_for_export,
        shard_weight_dict,
    )
except ImportError:
    # 独立运行时使用本地定义 - 使用完整的 regex 模式
    DEFAULT_WIDTH = 832
    DEFAULT_HEIGHT = 480
    DEFAULT_FRAMES = 81
    DEFAULT_FPS = 16
    DEFAULT_FLOW_SHIFT = 3.0
    
    # Text Encoder sharding (T5-XXL) - 完整 regex 模式
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
    
    # Transformer sharding (WanTransformer3DModel) - 完整 regex 模式
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
    VAE_DECODER_SHARDINGS = {}
    
    # ====== Fallback function definitions ======
    
    def move_module_to_xla(env, module):
        """将 PyTorch 模块权重转换为 torchax tensor"""
        with jax.default_device("cpu"):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    def shard_weight_dict(weight_dict, sharding_dict, mesh, debug=False):
        """按模式匹配应用权重分片"""
        import re
        
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
    
    def prepare_video_for_export(video_tensor, num_frames):
        """将视频 tensor 转换为可导出的格式 [T, H, W, C]"""
        # 处理 XLA tensor - 需要先转换 dtype
        if hasattr(video_tensor, '_elem'):
            # XLA tensor: 先转换为 float32 再转为 numpy
            jax_arr = video_tensor._elem
            if jax_arr.dtype == jnp.bfloat16:
                jax_arr = jax_arr.astype(jnp.float32)
            video_np = np.array(jax_arr)
        elif hasattr(video_tensor, 'numpy'):
            # 普通 PyTorch tensor
            if video_tensor.dtype == torch.bfloat16:
                video_tensor = video_tensor.float()
            video_np = video_tensor.numpy()
        else:
            video_np = np.array(video_tensor)
        
        # [B, C, T, H, W] -> [T, H, W, C]
        if len(video_np.shape) == 5:
            video_np = video_np[0]  # Remove batch: [C, T, H, W]
            video_np = video_np.transpose(1, 2, 3, 0)  # [T, H, W, C]
        
        # 取前 num_frames 帧
        video_np = video_np[:num_frames]
        
        # 归一化到 [0, 1]
        video_np = (video_np + 1) / 2
        video_np = np.clip(video_np, 0, 1)
        
        return video_np.astype(np.float32)

# 全局 mesh（延迟创建）
_mesh = None


def get_mesh():
    """获取全局 mesh，如果不存在则创建"""
    global _mesh
    if _mesh is None:
        print("[Wan21] Creating 2D Mesh for TPU...")
        devices = jax.devices('tpu')
        dp_dim = min(2, len(devices))
        tp_dim = len(devices) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        _mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"[Wan21] Created Mesh: dp={dp_dim}, tp={tp_dim}")
    return _mesh


# ============================================================================
# PyTree 注册
# ============================================================================

_pytree_registered = False

def _setup_pytree():
    """注册必要的 PyTree 节点以支持 JAX 转换"""
    global _pytree_registered
    if _pytree_registered:
        return
    
    from jax.tree_util import register_pytree_node
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
    
    _pytree_registered = True


# ============================================================================
# 自定义算子注册
# ============================================================================

def _register_text_encoder_ops(env):
    """
    注册 T5/UMT5 Text Encoder 需要的算子。
    
    UMT5 模型使用以下特殊操作：
    - dropout: 推理时直接返回输入
    - minimum/maximum: 元素级 min/max（用于 relative position bucket）
    - min.other: torch.min(a, b) 元素级比较
    - clamp: 张量裁剪
    - abs: 绝对值
    - log: 对数
    - floor: 向下取整
    """
    from torchax.ops import ops_registry
    
    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
    
    # ---- dropout ----
    # 推理时直接返回输入（不做 dropout）
    def dropout_impl(input, p=0.5, training=False, inplace=False, env=env):
        # 推理模式：直接返回输入
        return input
    
    def native_dropout_impl(input, p, train, env=env):
        # 推理模式：返回 (input, 全1 mask)
        if hasattr(input, '_elem'):
            # XLA tensor
            mask = torch.ones(input.shape, dtype=torch.bool)
            return input, mask.to('jax')
        return input, torch.ones_like(input, dtype=torch.bool)
    
    try:
        override_op(torch.ops.aten.dropout.default, functools.partial(dropout_impl, env=env))
        override_op(torch.ops.aten.native_dropout.default, functools.partial(native_dropout_impl, env=env))
        print("  - Registered dropout operators")
    except Exception as e:
        print(f"  - Warning: Failed to register dropout: {e}")
    
    # ---- minimum/maximum (元素级) ----
    # UMT5 的 _relative_position_bucket 使用 torch.min(a, b) 进行元素级比较
    def minimum_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.minimum(jinput, jother))
    
    def maximum_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.maximum(jinput, jother))
    
    try:
        override_op(torch.ops.aten.minimum.default, functools.partial(minimum_impl, env=env))
        override_op(torch.ops.aten.maximum.default, functools.partial(maximum_impl, env=env))
        print("  - Registered minimum/maximum operators")
    except Exception as e:
        print(f"  - Warning: Failed to register minimum/maximum: {e}")
    
    # ---- min.other / max.other (两个 tensor 的元素级比较) ----
    # 当调用 torch.min(a, b) 且 b 是 tensor 时，PyTorch 使用 aten.min.other
    # torchax 默认的 aten.min 把第二个参数当作 dim，导致错误
    def min_other_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.minimum(jinput, jother))
    
    def max_other_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.maximum(jinput, jother))
    
    try:
        override_op(torch.ops.aten.min.other, functools.partial(min_other_impl, env=env))
        override_op(torch.ops.aten.max.other, functools.partial(max_other_impl, env=env))
        print("  - Registered min.other/max.other operators")
    except Exception as e:
        print(f"  - Warning: Failed to register min.other/max.other: {e}")
    
    # ---- clamp ----
    def clamp_impl(input, min_val=None, max_val=None, env=env):
        jinput = env.t2j_iso(input)
        if min_val is not None:
            jinput = jnp.maximum(jinput, min_val)
        if max_val is not None:
            jinput = jnp.minimum(jinput, max_val)
        return env.j2t_iso(jinput)
    
    def clamp_tensor_impl(input, min_tensor=None, max_tensor=None, env=env):
        jinput = env.t2j_iso(input)
        if min_tensor is not None:
            jmin = env.t2j_iso(min_tensor)
            jinput = jnp.maximum(jinput, jmin)
        if max_tensor is not None:
            jmax = env.t2j_iso(max_tensor)
            jinput = jnp.minimum(jinput, jmax)
        return env.j2t_iso(jinput)
    
    try:
        override_op(torch.ops.aten.clamp.default, functools.partial(clamp_impl, env=env))
        override_op(torch.ops.aten.clamp.Tensor, functools.partial(clamp_tensor_impl, env=env))
        print("  - Registered clamp operators")
    except Exception as e:
        print(f"  - Warning: Failed to register clamp: {e}")
    
    # ---- abs ----
    def abs_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.abs(jinput))
    
    try:
        override_op(torch.ops.aten.abs.default, functools.partial(abs_impl, env=env))
        print("  - Registered abs operator")
    except Exception as e:
        print(f"  - Warning: Failed to register abs: {e}")
    
    # ---- log ----
    def log_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.log(jinput))
    
    try:
        override_op(torch.ops.aten.log.default, functools.partial(log_impl, env=env))
        print("  - Registered log operator")
    except Exception as e:
        print(f"  - Warning: Failed to register log: {e}")
    
    # ---- floor ----
    def floor_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.floor(jinput))
    
    try:
        override_op(torch.ops.aten.floor.default, functools.partial(floor_impl, env=env))
        print("  - Registered floor operator")
    except Exception as e:
        print(f"  - Warning: Failed to register floor: {e}")
    
    # ---- item (tensor -> scalar) ----
    # scheduler 使用 .item() 将单元素 tensor 转换为 Python 标量
    def item_impl(input, env=env):
        jinput = env.t2j_iso(input)
        # 获取标量值，保持正确的类型（int 或 float）
        scalar_val = np.array(jinput).item()
        # 保持原始类型：如果是整数类型，返回 int
        if jinput.dtype in (jnp.int32, jnp.int64, jnp.int16, jnp.int8, jnp.uint32, jnp.uint64):
            return int(scalar_val)
        return scalar_val  # 对于 float 类型，numpy.item() 返回正确类型
    
    try:
        override_op(torch.ops.aten.item.default, functools.partial(item_impl, env=env))
        override_op(torch.ops.aten._local_scalar_dense.default, functools.partial(item_impl, env=env))
        print("  - Registered item operator")
    except Exception as e:
        print(f"  - Warning: Failed to register item: {e}")


def _register_operators_on_env(env, mesh_obj):
    """
    在 torchax 环境上注册 TPU 所需的自定义算子。
    
    注册的算子:
      - conv2d: 2D 卷积
      - cartesian_prod: 笛卡尔积
      - chunk: 张量分块
      - layer_norm / native_layer_norm: 层归一化
      - unflatten: 维度展开
      - rms_norm: RMS 归一化
      - dropout / native_dropout: Dropout（推理时直接返回）
      - group_norm / native_group_norm: 组归一化
      - expand_as: 张量扩展（用于 F.normalize）
      - scaled_dot_product_attention: Splash Attention（可选）
    """
    # 延迟导入 torchax 组件
    from torchax.ops import jaten, ops_registry
    
    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
    
    # ---- conv2d ----
    def conv2d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)
    
    override_op(torch.nn.functional.conv2d, functools.partial(conv2d_impl, env=env))
    
    # ---- conv3d ----
    # 使用 torchax 的通用 _aten_convolution 实现，支持 3D 卷积
    def conv3d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        """
        3D 卷积实现，用于 WanTransformer3DModel 的 patch_embedding。
        
        Args:
            input: [N, C, D, H, W] - 输入张量
            weight: [out_channels, in_channels/groups, kD, kH, kW] - 卷积核
            bias: [out_channels] - 偏置（可选）
            stride: 步长
            padding: 填充
            dilation: 膨胀
            groups: 组数
        
        Returns:
            [N, out_channels, D', H', W'] - 输出张量
        """
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        # 使用 torchax 的通用卷积实现
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed=False,
            output_padding=1,  # 非转置卷积忽略此参数
            groups=groups
        )
        return env.j2t_iso(res)
    
    # 注册所有可能的 conv3d 变体（重要：必须全部注册！）
    print("  - Registering conv3d operator variants...")
    conv3d_fn = functools.partial(conv3d_impl, env=env)
    
    # 1. torch.nn.functional.conv3d
    override_op(torch.nn.functional.conv3d, conv3d_fn)
    print("    ✓ torch.nn.functional.conv3d")
    
    # 2. torch.ops.aten.conv3d (OpOverloadPacket) - 关键！
    try:
        override_op(torch.ops.aten.conv3d, conv3d_fn)
        print("    ✓ torch.ops.aten.conv3d")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.conv3d: {e}")
    
    # 3. torch.ops.aten.conv3d.default (OpOverload)
    try:
        override_op(torch.ops.aten.conv3d.default, conv3d_fn)
        print("    ✓ torch.ops.aten.conv3d.default")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.conv3d.default: {e}")
    
    # 4. torch.ops.aten.convolution (通用卷积接口)
    def convolution_impl(input, weight, bias=None, stride=1, padding=0, dilation=1,
                         transposed=False, output_padding=0, groups=1, *, env=env):
        """通用卷积实现，支持 2D 和 3D"""
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed, output_padding, groups
        )
        return env.j2t_iso(res)
    
    try:
        override_op(torch.ops.aten.convolution, functools.partial(convolution_impl, env=env))
        override_op(torch.ops.aten.convolution.default, functools.partial(convolution_impl, env=env))
        print("    ✓ torch.ops.aten.convolution")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.convolution: {e}")
    
    # ---- cartesian_prod ----
    def cartesian_prod_impl(tensors, env=env):
        if len(tensors) == 0:
            return env.j2t_iso(jnp.empty((0, 0)))
        if len(tensors) == 1:
            jt = env.t2j_iso(tensors[0])
            return env.j2t_iso(jnp.expand_dims(jt, axis=1))
        jarrays = [env.t2j_iso(t) for t in tensors]
        grids = jnp.meshgrid(*jarrays, indexing='ij')
        result = jnp.stack([g.ravel() for g in grids], axis=-1)
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.cartesian_prod.default, functools.partial(cartesian_prod_impl, env=env))
    except Exception:
        pass
    
    # ---- chunk ----
    def chunk_impl(input, chunks, dim=0, env=env):
        jinput = env.t2j_iso(input)
        if dim < 0:
            dim = len(jinput.shape) + dim
        size = jinput.shape[dim]
        chunk_size = (size + chunks - 1) // chunks
        splits = []
        for i in range(chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, size)
            if start >= size:
                break
            slices = [slice(None)] * len(jinput.shape)
            slices[dim] = slice(start, end)
            splits.append(env.j2t_iso(jinput[tuple(slices)]))
        return splits
    
    try:
        override_op(torch.ops.aten.chunk.default, functools.partial(chunk_impl, env=env))
    except Exception:
        pass
    
    # ---- layer_norm ----
    def layer_norm_impl(input, normalized_shape, weight=None, bias=None, eps=1e-5, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        mean = jnp.mean(jinput, axis=axis, keepdims=True)
        var = jnp.var(jinput, axis=axis, keepdims=True)
        result = (jinput - mean) / jnp.sqrt(var + eps)
        
        if jweight is not None:
            result = result * jweight
        if jbias is not None:
            result = result + jbias
        return env.j2t_iso(result)
    
    def native_layer_norm_impl(input, normalized_shape, weight, bias, eps, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        mean = jnp.mean(jinput, axis=axis, keepdims=True)
        var = jnp.var(jinput, axis=axis, keepdims=True)
        rstd = 1.0 / jnp.sqrt(var + eps)
        result = (jinput - mean) * rstd
        
        if jweight is not None:
            result = result * jweight
        if jbias is not None:
            result = result + jbias
        return env.j2t_iso(result), env.j2t_iso(mean.squeeze(axis)), env.j2t_iso(rstd.squeeze(axis))
    
    try:
        override_op(torch.ops.aten.layer_norm.default, functools.partial(layer_norm_impl, env=env))
        override_op(torch.ops.aten.native_layer_norm.default, functools.partial(native_layer_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- unflatten ----
    def unflatten_impl(input, dim, sizes, env=env):
        jinput = env.t2j_iso(input)
        shape = list(jinput.shape)
        if dim < 0:
            dim = len(shape) + dim
        
        sizes = list(sizes)
        if -1 in sizes:
            neg_idx = sizes.index(-1)
            known_prod = 1
            for i, s in enumerate(sizes):
                if i != neg_idx:
                    known_prod *= s
            sizes[neg_idx] = shape[dim] // known_prod
        
        new_shape = shape[:dim] + sizes + shape[dim+1:]
        return env.j2t_iso(jnp.reshape(jinput, new_shape))
    
    try:
        override_op(torch.ops.aten.unflatten.int, functools.partial(unflatten_impl, env=env))
    except Exception:
        pass
    
    # ---- rms_norm ----
    def rms_norm_impl(input, normalized_shape, weight=None, eps=1e-6, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        rms = jnp.sqrt(jnp.mean(jinput ** 2, axis=axis, keepdims=True) + eps)
        result = jinput / rms
        
        if jweight is not None:
            result = result * jweight
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.rms_norm.default, functools.partial(rms_norm_impl, env=env))
        override_op(torch.rms_norm, functools.partial(rms_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- dropout ----
    def dropout_impl(input, p=0.5, training=False, inplace=False, env=env):
        if not training or p == 0:
            return input
        jinput = env.t2j_iso(input)
        key = jax.random.PRNGKey(42)
        mask = jax.random.bernoulli(key, 1 - p, shape=jinput.shape)
        return env.j2t_iso(jinput * mask / (1 - p))
    
    def native_dropout_impl(input, p, train, env=env):
        if not train or p == 0:
            return input, torch.ones_like(input, dtype=torch.bool)
        jinput = env.t2j_iso(input)
        key = jax.random.PRNGKey(42)
        mask = jax.random.bernoulli(key, 1 - p, shape=jinput.shape)
        return env.j2t_iso(jinput * mask / (1 - p)), env.j2t_iso(mask.astype(jnp.bool_))
    
    try:
        override_op(torch.ops.aten.dropout.default, functools.partial(dropout_impl, env=env))
        override_op(torch.ops.aten.native_dropout.default, functools.partial(native_dropout_impl, env=env))
    except Exception:
        pass
    
    # ---- group_norm ----
    def group_norm_impl(input, num_groups, weight=None, bias=None, eps=1e-5, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        shape = jinput.shape
        N, C = shape[0], shape[1]
        spatial_dims = shape[2:]
        group_size = C // num_groups
        
        x = jnp.reshape(jinput, (N, num_groups, group_size) + spatial_dims)
        reduce_axes = tuple(range(2, len(x.shape)))
        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.var(x, axis=reduce_axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        result = jnp.reshape(x, shape)
        
        if jweight is not None:
            weight_shape = (1, C) + (1,) * len(spatial_dims)
            result = result * jnp.reshape(jweight, weight_shape)
        if jbias is not None:
            bias_shape = (1, C) + (1,) * len(spatial_dims)
            result = result + jnp.reshape(jbias, bias_shape)
        return env.j2t_iso(result)
    
    def native_group_norm_impl(input, weight, bias, N, C, HxW, group, eps, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        shape = jinput.shape
        spatial_dims = shape[2:]
        group_size = C // group
        
        x = jnp.reshape(jinput, (N, group, group_size) + spatial_dims)
        reduce_axes = tuple(range(2, len(x.shape)))
        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.var(x, axis=reduce_axes, keepdims=True)
        rstd = 1.0 / jnp.sqrt(var + eps)
        x = (x - mean) * rstd
        result = jnp.reshape(x, shape)
        
        if jweight is not None:
            weight_shape = (1, C) + (1,) * len(spatial_dims)
            result = result * jnp.reshape(jweight, weight_shape)
        if jbias is not None:
            bias_shape = (1, C) + (1,) * len(spatial_dims)
            result = result + jnp.reshape(jbias, bias_shape)
        
        mean_out = jnp.mean(x, axis=reduce_axes).reshape(N, group)
        rstd_out = jnp.mean(rstd, axis=reduce_axes).reshape(N, group)
        return env.j2t_iso(result), env.j2t_iso(mean_out), env.j2t_iso(rstd_out)
    
    try:
        override_op(torch.ops.aten.group_norm.default, functools.partial(group_norm_impl, env=env))
        override_op(torch.ops.aten.native_group_norm.default, functools.partial(native_group_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- expand_as (用于 F.normalize) ----
    def expand_as_impl(input, other, env=env):
        """
        将 input 扩展到与 other 相同的形状。
        用于 torch.nn.functional.normalize 中的 denom.expand_as(input)
        """
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        target_shape = jother.shape
        # 使用 JAX broadcast 扩展
        result = jnp.broadcast_to(jinput, target_shape)
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.expand_as.default, functools.partial(expand_as_impl, env=env))
        print("  - Registered expand_as operator")
    except Exception as e:
        print(f"  - Warning: Failed to register expand_as: {e}")
    
    # ---- Splash Attention ----
    try:
        try:
            from .splash_attention import sdpa_reference, tpu_splash_attention
        except ImportError:
            from splash_attention import sdpa_reference, tpu_splash_attention
        USE_K_SMOOTH = True
        
        def sdpa_tpu(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False, env=env, mesh=mesh_obj):
            # 仅对长序列使用 Splash Attention
            if key.shape[2] > 20000:
                jquery, jkey, jvalue = env.t2j_iso((query, key, value))
                if USE_K_SMOOTH:
                    jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
                res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
                return env.j2t_iso(res)
            return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        
        override_op(torch.nn.functional.scaled_dot_product_attention,
                    functools.partial(sdpa_tpu, env=env, mesh=mesh_obj))
    except ImportError:
        pass


# ============================================================================
# Torchax 环境管理（Hybrid 方案：enable_globally）
# ============================================================================

# 全局状态
_torchax_env = None
_ops_registered = False
_globally_enabled = False


def ensure_torchax_enabled(mesh_obj=None):
    """
    确保 torchax 全局启用，返回 env。
    
    Hybrid 方案：
    - 首次调用时 enable_globally()，之后保持启用
    - 这样缓存的 XLA 模型权重可以在后续调用中正常工作
    - 节点返回值必须转为 CPU tensor（由各节点负责）
    """
    global _torchax_env, _ops_registered, _globally_enabled
    import torchax
    
    # 首次调用时全局启用
    if not _globally_enabled:
        print("[Torchax] Enabling globally (Hybrid mode)...")
        torchax.enable_globally()
        _globally_enabled = True
    
    # 获取 env
    if _torchax_env is None:
        _torchax_env = torchax.default_env()
    
    # 注册算子（只需一次）
    if not _ops_registered and mesh_obj is not None:
        print("[Torchax] Registering operators...")
        _register_operators_on_env(_torchax_env, mesh_obj)
        _ops_registered = True
    
    return _torchax_env


# 保留旧函数名以兼容
def get_torchax_env(mesh_obj=None):
    """兼容旧代码，内部调用 ensure_torchax_enabled"""
    return ensure_torchax_enabled(mesh_obj)


# ============================================================================
# Wan 2.1 Text Encoder (TPU) - Hybrid 方案
# ============================================================================

class Wan21TextEncoder:
    """
    Wan 2.1 Text Encoder - 在 TPU 上运行 T5-XXL 编码 prompt。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: prompt, negative_prompt 文本
    输出: prompt_embeds, negative_prompt_embeds tensor (CPU)
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _is_compiled = False
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "A cat and a dog baking a cake together in a kitchen."}),
                "negative_prompt": ("STRING", {"multiline": True,
                    "default": "Bright tones, overexposed, static, blurred details, low quality"}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("prompt_embeds", "negative_prompt_embeds")
    FUNCTION = "encode"
    CATEGORY = "TPU/Wan2.1"
    
    def encode(self, prompt, negative_prompt, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        print(f"\n[Wan21TextEncoder] Encoding prompt on TPU...")
        print(f"  Prompt: {prompt[:50]}...")
        
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        print("  Encoding prompts...")
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                device='jax',
            )
            
            # 转换回 CPU（返回给 ComfyUI）
            prompt_embeds_cpu = self._to_cpu_tensor(prompt_embeds)
            negative_prompt_embeds_cpu = self._to_cpu_tensor(negative_prompt_embeds)
        
        print(f"  prompt_embeds shape: {prompt_embeds_cpu.shape}")
        return (prompt_embeds_cpu, negative_prompt_embeds_cpu)
    
    @staticmethod
    def _to_cpu_tensor(tensor):
        """将 XLA tensor 安全转换为 CPU tensor"""
        if hasattr(tensor, '_elem'):
            # XLA tensor: 转换为 numpy 再转为 torch
            jax_arr = tensor._elem
            if jax_arr.dtype == jnp.bfloat16:
                np_arr = np.array(jax_arr.astype(jnp.float32))
                return torch.from_numpy(np_arr).to(torch.bfloat16)
            else:
                return torch.from_numpy(np.array(jax_arr))
        elif hasattr(tensor, 'cpu'):
            return tensor.cpu()
        else:
            return tensor
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id, mesh):
        """
        加载和配置 Pipeline（Hybrid 方案）
        
        流程：
        1. 注册 PyTree
        2. 临时禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）
        3. 启用 torchax 并注册算子
        4. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        
        if (cls._cached_pipe is not None and
            cls._cached_model_id == model_id and
            cls._is_compiled):
            print("  Using cached pipeline")
            return cls._cached_pipe, cls._env
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # ===== 步骤 1：注册 PyTree =====
        _setup_pytree()
        
        # ===== 步骤 2：加载模型（禁用 torchax 避免拦截 transformers 加载逻辑）=====
        # 参考：stage2_transformer.py:514-530
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers import WanPipeline
        torch.set_default_dtype(torch.bfloat16)
        
        pipe = WanPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # 注册 Text Encoder 专用算子
        print("  注册 Text Encoder 算子...")
        _register_text_encoder_ops(env)
        
        # ===== 步骤 4：在 with mesh: 块内设置 Text Encoder =====
        print(f"  Mesh: {mesh}")
        with mesh:
            print("  - 移动 Text Encoder 到 TPU...")
            move_module_to_xla(env, pipe.text_encoder)
            pipe.text_encoder = torchax.compile(pipe.text_encoder)
            
            print(f"  - Sharding Text Encoder weights...")
            pipe.text_encoder.params = shard_weight_dict(
                pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            pipe.text_encoder.buffers = shard_weight_dict(
                pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )
            
            # 等待分片完成
            torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
        
        print("  ✓ Text Encoder 设置完成")
        
        # 删除不需要的组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            del pipe.transformer
            pipe.transformer = None
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        cls._cached_pipe = pipe
        cls._cached_model_id = model_id
        cls._is_compiled = True
        cls._env = env
        print("  Text Encoder ready!")
        return pipe, env


# ============================================================================
# Wan 2.1 TPU Sampler - Hybrid 方案
# ============================================================================

def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, enable_gqa=False,
                                   env=None, mesh=None):
    """封装 SDPA，长序列使用 TPU Splash Attention。"""
    try:
        from .splash_attention import sdpa_reference, tpu_splash_attention
    except ImportError:
        from splash_attention import sdpa_reference, tpu_splash_attention
    
    USE_K_SMOOTH = True
    
    # 仅对长序列（self-attention）使用 TPU Splash Attention
    if key.shape[2] > 20000:
        assert attn_mask is None
        assert dropout_p == 0.0
        assert is_causal is False
        assert enable_gqa is False
        assert scale is None
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return sdpa_reference(query, key, value, attn_mask, dropout_p,
                          is_causal, scale, enable_gqa)


def _setup_pipeline_for_transformer_only(pipe, mesh, env):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    
    注意：此函数应该在 `with mesh, env:` 块内调用！
    """
    from torchax.ops import ops_registry, jaten
    import torchax
    
    print("\n=== 配置 Transformer (TPU) ===")

    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )

    # Register conv3d for WanTransformer3DModel.patch_embedding
    print("- 注册 conv3d 算子...")
    def conv3d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        """
        3D 卷积实现，用于 WanTransformer3DModel 的 patch_embedding。
        """
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed=False,
            output_padding=1,
            groups=groups
        )
        return env.j2t_iso(res)
    
    # 注册所有可能的 conv3d 变体
    override_op(torch.nn.functional.conv3d, functools.partial(conv3d_impl, env=env))
    try:
        override_op(torch.ops.aten.conv3d, functools.partial(conv3d_impl, env=env))
        override_op(torch.ops.aten.conv3d.default, functools.partial(conv3d_impl, env=env))
    except Exception as e:
        print(f"  Warning: Failed to register aten.conv3d variants: {e}")

    # Register custom attention
    print("- 注册自定义 JAX 算子...")
    custom_attention = functools.partial(
        _scaled_dot_product_attention,
        env=env,
        mesh=mesh,
    )
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    override_op(op_to_override, custom_attention)

    # Move Transformer to XLA
    print("- 将 Transformer 移到 TPU...")
    move_module_to_xla(env, pipe.transformer)
    
    # Move rope embeddings to JAX
    if hasattr(pipe.transformer.rope, 'freqs'):
        pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    else:
        pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
        pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')

    # Compile Transformer
    print("- 编译 Transformer...")
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)

    # Apply sharding
    print("- 对 Transformer 进行权重分片...")
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    # Wait for sharding to complete
    torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)

    # Delete VAE to save memory (not needed in stage 2)
    print("- 删除 VAE 以节省内存...")
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        del pipe.vae
        pipe.vae = None

    # Delete Text Encoder (already used in stage 1)
    print("- 删除 Text Encoder...")
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None

    print("✓ Transformer 配置完成")
    return pipe


class Wan21TPUSampler:
    """
    Wan 2.1 TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: prompt_embeds, negative_prompt_embeds
    输出: latents (用于 VAE Decoder)
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    _mesh = None  # 缓存 mesh 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_embeds": ("TENSOR",),
                "negative_prompt_embeds": ("TENSOR",),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": 256, "max": 1280, "step": 16}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": 256, "max": 1280, "step": 16}),
                "num_frames": ("INT", {"default": DEFAULT_FRAMES, "min": 17, "max": 121, "step": 4}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "flow_shift": ("FLOAT", {"default": DEFAULT_FLOW_SHIFT, "min": 1.0, "max": 10.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latents", "num_frames")
    FUNCTION = "sample"
    CATEGORY = "TPU/Wan2.1"
    
    def sample(self, prompt_embeds, negative_prompt_embeds, height, width, num_frames,
               num_inference_steps, guidance_scale, seed,
               model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", flow_shift=DEFAULT_FLOW_SHIFT):
        """
        运行 Transformer 推理生成 latents（Hybrid 方案）
        """
        print(f"\n[Wan21TPUSampler] Starting TPU inference...")
        print(f"  Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # 注册 PyTree
        _setup_pytree()
        
        # 加载 Pipeline（如果需要）
        pipe, mesh, env = self._get_or_create_pipeline(model_id, flow_shift)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # 运行推理
        print(f"\n=== 阶段2：Transformer 推理 ===")
        print(f"推理步数: {num_inference_steps}")
        print(f"帧数: {num_frames}")
        print(f"引导尺度: {guidance_scale}")
        
        # 创建 ComfyUI 进度条
        import comfy.utils
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        
        # 进度回调函数
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs
        
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with mesh:
            # 将 embeddings 转换为 XLA tensor
            prompt_embeds_xla = prompt_embeds.to('jax')
            negative_prompt_embeds_xla = negative_prompt_embeds.to('jax')
            
            result = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds_xla,
                negative_prompt_embeds=negative_prompt_embeds_xla,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type='latent',
                use_dp=True,
                callback_on_step_end=progress_callback,
            )
            jax.effects_barrier()
            
            # 转换 latents 为 CPU tensor（返回给 ComfyUI）
            torch_latents = self._to_cpu_tensor(result.frames)
        
        elapsed = time.perf_counter() - start_time
        print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
        print(f"  平均每步时间: {elapsed/num_inference_steps:.2f}s")
        print(f"  Latents shape: {torch_latents.shape}")
        print(f"  Latents dtype: {torch_latents.dtype}")
        
        return ({"samples": torch_latents, "num_frames": num_frames}, num_frames)
    
    @staticmethod
    def _to_cpu_tensor(tensor):
        """将 XLA tensor 安全转换为 CPU tensor"""
        if hasattr(tensor, '_elem'):
            jax_arr = tensor._elem
            if jax_arr.dtype == jnp.bfloat16:
                np_arr = np.array(jax_arr.astype(jnp.float32))
                return torch.from_numpy(np_arr).to(torch.bfloat16)
            else:
                return torch.from_numpy(np.array(jax_arr))
        elif hasattr(tensor, 'cpu'):
            return tensor.cpu()
        else:
            return tensor
    
    def _get_or_create_pipeline(self, model_id, flow_shift):
        """
        加载和配置 Pipeline（Hybrid 方案）
        
        流程：
        1. 创建 mesh
        2. 临时禁用 torchax 加载模型
        3. 启用 torchax 并注册算子
        4. 在 with mesh: 块内配置 Pipeline
        """
        import torchax
        
        if (Wan21TPUSampler._cached_pipe is not None and
            Wan21TPUSampler._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Wan21TPUSampler._cached_pipe, Wan21TPUSampler._mesh, Wan21TPUSampler._env
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # ===== 步骤 1：创建 mesh =====
        dp_dim = 2
        tp_dim = len(jax.devices()) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"Mesh: {mesh}")
        print(f"  dp_dim={dp_dim}, tp_dim={tp_dim}")
        print(f"  总设备数: {len(jax.devices())}")
        
        # ===== 步骤 2：禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）=====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        # 设置 default dtype
        torch.set_default_dtype(torch.bfloat16)
        
        from diffusers.pipelines.wan.pipeline_wan_torchax import WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift
        )
        
        pipe = WanPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        pipe.scheduler = scheduler
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 4：在 with mesh: 块内配置 Pipeline =====
        with mesh:
            pipe = _setup_pipeline_for_transformer_only(pipe, mesh, env)
        
        gc.collect()
        Wan21TPUSampler._cached_pipe = pipe
        Wan21TPUSampler._cached_model_id = model_id
        Wan21TPUSampler._env = env
        Wan21TPUSampler._mesh = mesh
        print("  Pipeline ready!")
        return pipe, mesh, env


# ============================================================================
# Wan 2.1 TPU VAE Decoder - Hybrid 方案
# ============================================================================

class Wan21TPUVAEDecoder:
    """
    Wan 2.1 VAE Decoder - 在 TPU 上解码 latents 为视频。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: latents (来自 Sampler)
    输出: video frames (ComfyUI IMAGE 格式)
    """
    
    _cached_vae = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "fps": ("INT", {"default": DEFAULT_FPS, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "decode"
    CATEGORY = "TPU/Wan2.1"
    
    def decode(self, latents, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", fps=DEFAULT_FPS):
        print(f"\n[Wan21TPUVAEDecoder] Starting VAE decode...")
        
        if isinstance(latents, dict):
            latent_tensor = latents["samples"]
            num_frames = latents.get("num_frames", DEFAULT_FRAMES)
        else:
            latent_tensor = latents
            num_frames = DEFAULT_FRAMES
        
        real_mesh = get_mesh()
        vae, env = self._get_or_create_vae(model_id, real_mesh)
        
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 处理 latents
            processed_latents = latent_tensor.to(vae.dtype)  # 转为 bfloat16
            processed_latents = env.to_xla(processed_latents)  # 转为 XLA
            processed_latents = self._denormalize_latents(processed_latents, vae, env)
            
            print("  Decoding...")
            with torch.no_grad():
                video = vae.decode(processed_latents).sample
            jax.effects_barrier()
            
            # 转换回 CPU（返回给 ComfyUI）
            video_cpu = self._to_cpu_tensor(video)
        
        print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
        
        # 后处理（在 CPU 上）
        frames = prepare_video_for_export(video_cpu, num_frames)
        frames_tensor = torch.from_numpy(frames)
        
        print(f"  Output: {frames_tensor.shape}")
        return (frames_tensor, fps)
    
    @staticmethod
    def _to_cpu_tensor(tensor):
        """将 XLA tensor 安全转换为 CPU tensor"""
        if hasattr(tensor, '_elem'):
            jax_arr = tensor._elem
            if jax_arr.dtype == jnp.bfloat16:
                np_arr = np.array(jax_arr.astype(jnp.float32))
                return torch.from_numpy(np_arr).to(torch.bfloat16)
            else:
                return torch.from_numpy(np.array(jax_arr))
        elif hasattr(tensor, 'cpu'):
            return tensor.cpu()
        else:
            return tensor
    
    def _denormalize_latents(self, latents, vae, env):
        """
        Denormalize latents: x * std + mean
        
        注意：此时 latents 已经是 XLA tensor (bfloat16)
        必须在 with env: 块内调用！
        """
        latents_mean = getattr(vae.config, 'latents_mean', None)
        latents_std = getattr(vae.config, 'latents_std', None)
        
        if latents_mean is None or latents_std is None:
            return latents
        
        # 创建 mean 和 std tensor，转换为 XLA
        mean = torch.tensor(latents_mean, dtype=torch.bfloat16).view(1, 16, 1, 1, 1).to('jax')
        std = torch.tensor(latents_std, dtype=torch.bfloat16).view(1, 16, 1, 1, 1).to('jax')
        return latents * std + mean
    
    def _get_or_create_vae(self, model_id, mesh):
        """
        加载和配置 VAE（Hybrid 方案）
        
        流程：
        1. 临时禁用 torchax 加载模型
        2. 启用 torchax 并注册算子
        3. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        
        if (Wan21TPUVAEDecoder._cached_vae is not None and
            Wan21TPUVAEDecoder._cached_model_id == model_id):
            print("  Using cached VAE")
            return Wan21TPUVAEDecoder._cached_vae, Wan21TPUVAEDecoder._env
        
        print(f"  Loading VAE from {model_id}...")
        
        # ===== 步骤 1：禁用 torchax 加载模型 =====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )
        print("  ✓ VAE 加载完成")
        
        # ===== 步骤 2：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # 注册 conv2d 算子
        print("  - 注册 conv2d 算子...")
        from torchax.ops import jaten, ops_registry
        
        def conv2d_impl(input, weight, bias=None, stride=1, padding=0,
                        dilation=1, groups=1):
            jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
            res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
            return env.j2t_iso(res)
        
        env._ops[torch.nn.functional.conv2d] = ops_registry.Operator(
            torch.nn.functional.conv2d, conv2d_impl,
            is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
        
        # 注册 expand_as 算子（用于 F.normalize）
        print("  - 注册 expand_as 算子...")
        def expand_as_impl(input, other):
            jinput = env.t2j_iso(input)
            jother = env.t2j_iso(other)
            target_shape = jother.shape
            result = jnp.broadcast_to(jinput, target_shape)
            return env.j2t_iso(result)
        
        env._ops[torch.ops.aten.expand_as.default] = ops_registry.Operator(
            torch.ops.aten.expand_as.default, expand_as_impl,
            is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
        
        # ===== 步骤 3：在 with mesh: 块内设置 VAE Decoder =====
        with mesh:
            print("  - 将 VAE 移到 TPU...")
            move_module_to_xla(env, vae)
            
            print("  - 编译 VAE Decoder...")
            vae.decoder = torchax.compile(vae.decoder)
            
            num_devices = mesh.devices.size
            print(f"  - 复制权重到 {num_devices} TPU cores...")
            vae.decoder.params = shard_weight_dict(
                vae.decoder.params, VAE_DECODER_SHARDINGS, mesh
            )
            vae.decoder.buffers = shard_weight_dict(
                vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh
            )
        
        print("  ✓ VAE Decoder JIT 编译完成")
        
        gc.collect()
        Wan21TPUVAEDecoder._cached_vae = vae
        Wan21TPUVAEDecoder._cached_model_id = model_id
        Wan21TPUVAEDecoder._env = env
        print("  VAE ready!")
        return vae, env


# ============================================================================
# Wan 2.1 Full Pipeline
# ============================================================================

class Wan21TPUPipeline:
    """
    Wan 2.1 TPU Full Pipeline - 端到端视频生成。
    
    组合 TextEncoder -> Sampler -> VAEDecoder 三个阶段。
    
    重要：按照参考实现的三阶段设计：
    - 每个阶段只加载需要的组件
    - 阶段间清理内存
    - 这样可以在有限的 HBM 上运行 14B 模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "A cat and a dog baking a cake together in a kitchen."}),
                "negative_prompt": ("STRING", {"multiline": True,
                    "default": "Bright tones, overexposed, static, blurred details, low quality"}),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": 256, "max": 1280, "step": 16}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": 256, "max": 1280, "step": 16}),
                "num_frames": ("INT", {"default": DEFAULT_FRAMES, "min": 17, "max": 121, "step": 4}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "fps": ("INT", {"default": DEFAULT_FPS, "min": 1, "max": 60}),
                "flow_shift": ("FLOAT", {"default": DEFAULT_FLOW_SHIFT, "min": 1.0, "max": 10.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "generate"
    CATEGORY = "TPU/Wan2.1"
    
    def generate(self, prompt, negative_prompt, height, width, num_frames,
                 num_inference_steps, guidance_scale, seed,
                 model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                 fps=DEFAULT_FPS, flow_shift=DEFAULT_FLOW_SHIFT):
        
        print(f"\n{'='*60}")
        print("Wan 2.1 TPU Full Pipeline")
        print(f"{'='*60}")
        
        # Stage 1: Text Encoding (TPU)
        prompt_embeds, negative_prompt_embeds = Wan21TextEncoder().encode(
            prompt, negative_prompt, model_id
        )
        
        # 清理 Stage 1 的缓存，释放 HBM
        print("\n[Pipeline] 清理 Text Encoder 以释放 HBM...")
        Wan21TextEncoder._cached_pipe = None
        Wan21TextEncoder._cached_model_id = None
        Wan21TextEncoder._is_compiled = False
        gc.collect()
        
        # Stage 2: Denoising (TPU)
        latents, _ = Wan21TPUSampler().sample(
            prompt_embeds, negative_prompt_embeds,
            height, width, num_frames,
            num_inference_steps, guidance_scale, seed,
            model_id, flow_shift
        )
        
        # 清理 Stage 2 的缓存，释放 HBM
        print("\n[Pipeline] 清理 Transformer 以释放 HBM...")
        Wan21TPUSampler._cached_pipe = None
        Wan21TPUSampler._cached_model_id = None
        gc.collect()
        
        # Stage 3: VAE Decoding (TPU)
        frames, fps_out = Wan21TPUVAEDecoder().decode(latents, model_id, fps)
        
        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"{'='*60}")
        
        return (frames, fps_out)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Wan21TextEncoder": Wan21TextEncoder,
    "Wan21TPUSampler": Wan21TPUSampler,
    "Wan21TPUVAEDecoder": Wan21TPUVAEDecoder,
    "Wan21TPUPipeline": Wan21TPUPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan21TextEncoder": "Wan 2.1 Text Encoder (TPU)",
    "Wan21TPUSampler": "Wan 2.1 TPU Sampler",
    "Wan21TPUVAEDecoder": "Wan 2.1 TPU VAE Decoder",
    "Wan21TPUPipeline": "Wan 2.1 TPU Full Pipeline",
}
