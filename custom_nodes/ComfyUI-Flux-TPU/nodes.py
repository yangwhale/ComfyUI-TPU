"""
ComfyUI Flux.2 TPU Nodes
========================

使用 diffusers 的 torchax 优化模型在 TPU 上运行 Flux.2 推理。
基于 gpu-tpu-pedia/tpu/Flux.2/generate_diffusers_torchax_staged 实现。

核心设计（Hybrid 方案）：
  - 使用 `enable_globally()` 保持 Mode 栈激活（解决 XLA tensor 逃逸问题）
  - 模型缓存后权重保持 XLA 状态
  - 节点返回值必须转为 CPU tensor（确保与 ComfyUI 兼容）

Nodes:
  - Flux2TextEncoder: CPU 上运行 Mistral3 编码 prompt
  - Flux2TPUSampler: TPU 上运行 Transformer 生成 latents
  - Flux2TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为图像
  - Flux2TPUPipeline: 端到端 Pipeline（组合以上三个）
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

# 延迟导入 utils（支持独立测试）
try:
    from .utils import (
        TRANSFORMER_SHARDINGS,
        VAE_DECODER_SHARDINGS,
        move_module_to_xla,
        shard_weight_dict,
        setup_jax_cache,
        setup_pytree_registrations,
    )
except ImportError:
    # 独立运行时使用本地定义
    TRANSFORMER_SHARDINGS = {}
    VAE_DECODER_SHARDINGS = {}
    
    def move_module_to_xla(env, module):
        with jax.default_device("cpu"):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    def shard_weight_dict(weight_dict, sharding_dict, mesh, debug=False):
        import re
        result = {}
        for k, v in weight_dict.items():
            if isinstance(v, torch.Tensor):
                with jax.default_device("cpu"):
                    v = v.to("jax")
            matched = False
            for pattern, sharding in sharding_dict.items():
                if re.fullmatch(pattern, k) is not None:
                    v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                    matched = True
                    break
            if not matched:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            result[k] = v
        return result
    
    def setup_jax_cache():
        pass
    
    def setup_pytree_registrations():
        pass


# ============================================================================
# 全局 mesh（延迟创建）
# ============================================================================

_mesh = None


def get_mesh():
    """获取全局 mesh，如果不存在则创建"""
    global _mesh
    if _mesh is None:
        print("[Flux2] Creating 1D Mesh for TPU...")
        devices = jax.devices('tpu')
        tp_dim = len(devices)
        mesh_devices = mesh_utils.create_device_mesh(
            (tp_dim,), allow_split_physical_axes=True
        )
        _mesh = Mesh(mesh_devices, ("tp",))
        print(f"[Flux2] Created Mesh: tp={tp_dim}")
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
    
    _pytree_registered = True


# ============================================================================
# 自定义算子注册
# ============================================================================

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
    
    # ---- Splash Attention (可选) ----
    try:
        from .splash_attention import sdpa_reference, tpu_splash_attention
        
        def sdpa_tpu(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False, env=env, mesh=mesh_obj):
            if key.shape[2] > 20000:
                jquery, jkey, jvalue = env.t2j_iso((query, key, value))
                jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)  # K-smooth
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
# Flux.2 Text Encoder (CPU)
# ============================================================================

class Flux2TextEncoder:
    """
    Flux.2 Text Encoder - 在 CPU 上运行 Mistral3 编码 prompt。
    
    输入: prompt 文本
    输出: prompt_embeds tensor (用于 Sampler)
    """
    
    _cached_encoder = None
    _cached_tokenizer = None
    _cached_model_id = None
    
    SYSTEM_MESSAGE = (
        "You are an AI that reasons about image descriptions. "
        "You give structured responses focusing on object relationships, "
        "object attribution and actions without speculation."
    )
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over the ocean"}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "black-forest-labs/FLUX.2-dev"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("prompt_embeds",)
    FUNCTION = "encode"
    CATEGORY = "TPU/Flux.2"
    
    def encode(self, prompt, model_id="black-forest-labs/FLUX.2-dev"):
        print(f"\n[Flux2TextEncoder] Encoding prompt on CPU...")
        print(f"  Prompt: {prompt[:50]}...")
        
        text_encoder, tokenizer = self._get_or_create_encoder(model_id)
        
        messages = [[
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "text", "text": prompt.replace("[IMG]", "")}]},
        ]]
        
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt",
            padding="max_length", truncation=True, max_length=512,
        )
        
        with torch.no_grad():
            output = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        
        # 提取指定层的 hidden states
        hidden_states_layers = (10, 20, 30)
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=torch.bfloat16)
        
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        
        print(f"  Prompt embeddings shape: {prompt_embeds.shape}")
        return (prompt_embeds,)
    
    def _get_or_create_encoder(self, model_id):
        if (Flux2TextEncoder._cached_encoder is not None and 
            Flux2TextEncoder._cached_model_id == model_id):
            return Flux2TextEncoder._cached_encoder, Flux2TextEncoder._cached_tokenizer
        
        print(f"  Loading Mistral3 Text Encoder from {model_id}...")
        from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
        
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        text_encoder.eval()
        tokenizer = PixtralProcessor.from_pretrained(model_id, subfolder="tokenizer")
        
        Flux2TextEncoder._cached_encoder = text_encoder
        Flux2TextEncoder._cached_tokenizer = tokenizer
        Flux2TextEncoder._cached_model_id = model_id
        print("  Text Encoder loaded!")
        return text_encoder, tokenizer


# ============================================================================
# Flux.2 TPU Sampler - Hybrid 方案
# ============================================================================

class Flux2TPUSampler:
    """
    Flux.2 TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: prompt_embeds (来自 TextEncoder)
    输出: latents (用于 VAE Decoder)
    """
    
    _cached_pipeline = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_embeds": ("TENSOR",),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "black-forest-labs/FLUX.2-dev"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TPU/Flux.2"
    
    def sample(self, prompt_embeds, height, width, num_inference_steps,
               guidance_scale, seed, model_id="black-forest-labs/FLUX.2-dev"):
        """
        运行 Transformer 推理生成 latents（Hybrid 方案）
        """
        print(f"\n[Flux2TPUSampler] Starting TPU inference...")
        print(f"  Height: {height}, Width: {width}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # 注册 PyTree
        _setup_pytree()
        
        # 加载 Pipeline（如果需要）
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            prompt_embeds_xla = prompt_embeds.to('jax')
            
            print(f"  Running denoising loop...")
            start_time = time.perf_counter()
            
            result = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds_xla,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type='latent',
            )
            jax.effects_barrier()
            
            elapsed = time.perf_counter() - start_time
            print(f"  Done: {elapsed:.2f}s ({elapsed/num_inference_steps:.2f}s/step)")
            
            # 转换 latents 为 CPU tensor（返回给 ComfyUI）
            torch_latents = self._to_cpu_tensor(result.images)
        
        return ({"samples": torch_latents},)
    
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
    
    def _get_or_create_pipeline(self, model_id, mesh):
        """
        加载和配置 Pipeline（Hybrid 方案）
        
        流程：
        1. 临时禁用 torchax 加载模型
        2. 启用 torchax 并注册算子
        3. 在 with mesh: 块内配置 Pipeline
        """
        import torchax
        
        if (Flux2TPUSampler._cached_pipeline is not None and
            Flux2TPUSampler._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Flux2TPUSampler._cached_pipeline, Flux2TPUSampler._env
        
        print(f"  Loading Flux.2 Pipeline from {model_id}...")
        
        # ===== 步骤 1：禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）=====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        torch.set_default_dtype(torch.bfloat16)
        
        from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
        from diffusers.models.transformers.transformer_flux2_torchax import Flux2Transformer2DModel
        from diffusers.pipelines.flux2.pipeline_flux2_torchax import Flux2Pipeline
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        
        vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = Flux2Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler()
        
        pipe = Flux2Pipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, text_encoder=None,
            vae=vae, transformer=transformer, scheduler=scheduler,
        )
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 2：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 3：在 with mesh: 块内配置 Pipeline =====
        with mesh:
            print("  - Converting Transformer to XLA...")
            move_module_to_xla(env, pipe.transformer)
            
            print("  - Compiling Transformer...")
            pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict',)}))
            
            print(f"  - Sharding weights to {len(mesh.devices)} TPU cores...")
            pipe.transformer.params = shard_weight_dict(pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
            pipe.transformer.buffers = shard_weight_dict(pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
            torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
        
        print("  ✓ Transformer 配置完成")
        
        # 释放不需要的 VAE（Sampler 只需要 Transformer）
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        Flux2TPUSampler._cached_pipeline = pipe
        Flux2TPUSampler._cached_model_id = model_id
        Flux2TPUSampler._env = env
        print("  Pipeline ready!")
        return pipe, env


# ============================================================================
# Flux.2 TPU VAE Decoder - Hybrid 方案
# ============================================================================

class Flux2TPUVAEDecoder:
    """
    Flux.2 VAE Decoder - 在 TPU 上解码 latents 为图像。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: latents (来自 Sampler)
    输出: image tensor (ComfyUI IMAGE 格式)
    """
    
    _cached_vae = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "black-forest-labs/FLUX.2-dev"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "TPU/Flux.2"
    
    def decode(self, latents, height, width, model_id="black-forest-labs/FLUX.2-dev"):
        print(f"\n[Flux2TPUVAEDecoder] Starting VAE decode...")
        
        latent_tensor = latents["samples"] if isinstance(latents, dict) else latents
        
        real_mesh = get_mesh()
        vae, env = self._get_or_create_vae(model_id, real_mesh)
        
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 处理 latents
            processed_latents = self._process_latents(latent_tensor, height, width, vae)
            processed_latents = env.to_xla(processed_latents.to(torch.bfloat16))
            
            print("  Decoding...")
            with torch.no_grad():
                image = vae.decode(processed_latents, return_dict=False)[0]
            jax.effects_barrier()
            
            # 转换回 CPU（返回给 ComfyUI）
            image_cpu = self._to_cpu_tensor(image)
        
        print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
        
        # 后处理（在 CPU 上）
        image_output = self._postprocess_image(image_cpu)
        
        return (image_output,)
    
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
    
    def _get_or_create_vae(self, model_id, mesh):
        """
        加载和配置 VAE（Hybrid 方案）
        
        流程：
        1. 临时禁用 torchax 加载模型
        2. 启用 torchax 并注册算子
        3. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        
        if (Flux2TPUVAEDecoder._cached_vae is not None and
            Flux2TPUVAEDecoder._cached_model_id == model_id):
            print("  Using cached VAE")
            return Flux2TPUVAEDecoder._cached_vae, Flux2TPUVAEDecoder._env
        
        print(f"  Loading VAE from {model_id}...")
        
        # ===== 步骤 1：禁用 torchax 加载模型 =====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        print("  ✓ VAE 加载完成")
        
        # ===== 步骤 2：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 3：在 with mesh: 块内设置 VAE Decoder =====
        with mesh:
            print("  - Converting VAE to XLA...")
            move_module_to_xla(env, vae)
            
            print("  - Compiling VAE Decoder...")
            vae.decoder = torchax.compile(vae.decoder)
            
            num_devices = mesh.devices.size
            print(f"  - Replicating weights to {num_devices} TPU cores...")
            vae.decoder.params = shard_weight_dict(vae.decoder.params, VAE_DECODER_SHARDINGS, mesh)
            vae.decoder.buffers = shard_weight_dict(vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh)
        
        print("  ✓ VAE Decoder 配置完成")
        
        gc.collect()
        Flux2TPUVAEDecoder._cached_vae = vae
        Flux2TPUVAEDecoder._cached_model_id = model_id
        Flux2TPUVAEDecoder._env = env
        print("  VAE ready!")
        return vae, env
    
    def _prepare_latent_ids(self, height, width, device=None):
        """生成 latent 位置 ID"""
        t = torch.arange(1, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.arange(1, device=device)
        return torch.cartesian_prod(t, h, w, l).unsqueeze(0)
    
    def _unpack_latents(self, x, x_ids):
        """将打包的 latents 展开为空间格式"""
        x_list = []
        for data, pos in zip(x, x_ids):
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)
            h, w = torch.max(h_ids) + 1, torch.max(w_ids) + 1
            flat_ids = h_ids * w + w_ids
            out = torch.zeros((h * w, data.shape[1]), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, data.shape[1]), data)
            out = out.view(h, w, data.shape[1]).permute(2, 0, 1)
            x_list.append(out)
        return torch.stack(x_list)
    
    def _unpatchify_latents(self, latents):
        """将 patchified latents 还原为原始形状"""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)
    
    def _process_latents(self, latents, height, width, vae):
        """处理 latents：unpack -> denormalize -> unpatchify"""
        print(f"  Processing latents: {latents.shape}")
        vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        latent_h = 2 * (height // (vae_scale * 2))
        latent_w = 2 * (width // (vae_scale * 2))
        
        latent_ids = self._prepare_latent_ids(latent_h // 2, latent_w // 2, device=latents.device)
        latents = self._unpack_latents(latents, latent_ids)
        print(f"  Unpacked: {latents.shape}")
        
        # 反归一化
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * torch.sqrt(bn_var + vae.config.batch_norm_eps) + bn_mean
        
        latents = self._unpatchify_latents(latents)
        print(f"  Unpatchified: {latents.shape}")
        return latents
    
    def _postprocess_image(self, image):
        """后处理图像：CPU tensor -> ComfyUI 格式"""
        # 转换为 ComfyUI 格式: (B, H, W, C), 范围 [0, 1]
        if image.dtype == torch.bfloat16:
            image = image.float()
        image = image.permute(0, 2, 3, 1)
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


# ============================================================================
# Flux.2 Full Pipeline
# ============================================================================

class Flux2TPUPipeline:
    """
    Flux.2 TPU Full Pipeline - 端到端图像生成。
    
    组合 TextEncoder -> Sampler -> VAEDecoder 三个阶段。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over the ocean"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "black-forest-labs/FLUX.2-dev"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "TPU/Flux.2"
    
    def generate(self, prompt, height, width, num_inference_steps, 
                 guidance_scale, seed, model_id="black-forest-labs/FLUX.2-dev"):
        
        print(f"\n{'='*60}")
        print("Flux.2 TPU Full Pipeline")
        print(f"{'='*60}")
        
        # Stage 1: Text Encoding (CPU)
        prompt_embeds, = Flux2TextEncoder().encode(prompt, model_id)
        
        # Stage 2: Denoising (TPU)
        latents, = Flux2TPUSampler().sample(
            prompt_embeds, height, width, 
            num_inference_steps, guidance_scale, seed, model_id
        )
        
        # Stage 3: VAE Decoding (TPU)
        image, = Flux2TPUVAEDecoder().decode(latents, height, width, model_id)
        
        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"{'='*60}")
        
        return (image,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Flux2TextEncoder": Flux2TextEncoder,
    "Flux2TPUSampler": Flux2TPUSampler,
    "Flux2TPUVAEDecoder": Flux2TPUVAEDecoder,
    "Flux2TPUPipeline": Flux2TPUPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2TextEncoder": "Flux.2 Text Encoder (CPU)",
    "Flux2TPUSampler": "Flux.2 TPU Sampler",
    "Flux2TPUVAEDecoder": "Flux.2 TPU VAE Decoder",
    "Flux2TPUPipeline": "Flux.2 TPU Full Pipeline",
}
