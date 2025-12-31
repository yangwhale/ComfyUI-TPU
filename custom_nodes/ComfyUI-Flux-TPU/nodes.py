"""
ComfyUI Flux.2 TPU Nodes
========================

使用 diffusers 的 torchax 优化模型在 TPU 上运行 Flux.2 推理。
基于 gpu-tpu-pedia/tpu/Flux.2/generate_diffusers_torchax_staged 实现。

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
import torchax
from jax.sharding import NamedSharding
from torchax.ops import jaten, ops_registry

from . import mesh
from .utils import (
    TRANSFORMER_SHARDINGS,
    VAE_DECODER_SHARDINGS,
    move_module_to_xla,
    shard_weight_dict,
)


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
# Torchax 环境管理（单例模式）
# ============================================================================

class TorchaxEnvManager:
    """
    Torchax 环境的单例管理器。
    
    - 首次调用 get_env() 时执行 enable_globally() 并注册算子
    - 后续调用直接返回已有的 env
    - pause()/resume() 用于模型加载时临时禁用环境
    """
    _env = None
    _initialized = False
    
    @classmethod
    def get_env(cls, mesh_obj=None):
        """获取或创建全局 torchax 环境"""
        if not cls._initialized:
            print("[TorchaxEnvManager] Initializing global torchax environment...")
            torchax.enable_globally()
            cls._env = torchax.default_env()
            _register_operators_on_env(cls._env, mesh_obj or mesh)
            cls._initialized = True
            print("[TorchaxEnvManager] Environment initialized.")
        return cls._env
    
    @classmethod
    def pause(cls):
        """临时暂停 torchax 环境（用于模型加载）"""
        if cls._initialized:
            try:
                torchax.disable_globally()
            except Exception:
                pass
    
    @classmethod
    def resume(cls):
        """恢复 torchax 环境"""
        if cls._initialized:
            try:
                torchax.enable_globally()
            except Exception:
                pass
    
    @classmethod
    def reset(cls):
        """完全重置环境"""
        if cls._initialized:
            try:
                torchax.disable_globally()
            except Exception:
                pass
            cls._env = None
            cls._initialized = False


class TorchaxContext:
    """
    Torchax 环境上下文管理器。
    
    用法:
        with TorchaxContext() as ctx:
            # ctx.env 是 torchax 环境
            xla_tensor = ctx.env.to_xla(tensor)
    """
    def __init__(self, mesh_obj=None, disable_on_exit=False):
        self.mesh_obj = mesh_obj or mesh
        self.disable_on_exit = disable_on_exit
        self.env = None
    
    def __enter__(self):
        self.env = TorchaxEnvManager.get_env(self.mesh_obj)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_on_exit:
            TorchaxEnvManager.reset()


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
# Flux.2 TPU Sampler
# ============================================================================

class Flux2TPUSampler:
    """
    Flux.2 TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    输入: prompt_embeds (来自 TextEncoder)
    输出: latents (用于 VAE Decoder)
    """
    
    _cached_pipeline = None
    _cached_model_id = None
    
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
        
        print(f"\n[Flux2TPUSampler] Starting TPU inference...")
        print(f"  Height: {height}, Width: {width}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        pipe = self._get_or_create_pipeline(model_id)
        
        with TorchaxContext() as ctx:
            prompt_embeds_xla = prompt_embeds.to('jax')
            
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            with mesh:
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
            
            latents = result.images
            torch_latents = self._convert_latents_to_cpu(latents)
        
        return ({"samples": torch_latents},)
    
    def _convert_latents_to_cpu(self, latents):
        """将 XLA latents 转换回 CPU tensor"""
        if hasattr(latents, '_elem'):
            jax_latents = latents._elem
            if jax_latents.dtype == jnp.bfloat16:
                return torch.from_numpy(np.array(jax_latents.astype(jnp.float32))).to(torch.bfloat16)
            return torch.from_numpy(np.array(jax_latents))
        return latents.cpu()
    
    def _get_or_create_pipeline(self, model_id):
        if (Flux2TPUSampler._cached_pipeline is not None and 
            Flux2TPUSampler._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Flux2TPUSampler._cached_pipeline
        
        print(f"  Loading Flux.2 Pipeline from {model_id}...")
        
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
        
        with TorchaxContext() as ctx:
            print("  - Converting Transformer to XLA...")
            move_module_to_xla(ctx.env, pipe.transformer)
            
            print("  - Compiling Transformer...")
            pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict',)}))
            
            print(f"  - Sharding weights to {len(mesh.devices)} TPU cores...")
            pipe.transformer.params = shard_weight_dict(pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
            pipe.transformer.buffers = shard_weight_dict(pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
            torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
        
        # 释放不需要的 VAE（Sampler 只需要 Transformer）
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        Flux2TPUSampler._cached_pipeline = pipe
        Flux2TPUSampler._cached_model_id = model_id
        print("  Pipeline ready!")
        return pipe


# ============================================================================
# Flux.2 TPU VAE Decoder
# ============================================================================

class Flux2TPUVAEDecoder:
    """
    Flux.2 VAE Decoder - 在 TPU 上解码 latents 为图像。
    
    输入: latents (来自 Sampler)
    输出: image tensor (ComfyUI IMAGE 格式)
    """
    
    _cached_vae = None
    _cached_model_id = None
    
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
        vae = self._get_or_create_vae(model_id)
        
        with TorchaxContext() as ctx:
            processed_latents = self._process_latents(latent_tensor, height, width, vae)
            processed_latents = ctx.env.to_xla(processed_latents.to(torch.bfloat16))
            
            with mesh:
                print("  Decoding...")
                start_time = time.perf_counter()
                with torch.no_grad():
                    image = vae.decode(processed_latents, return_dict=False)[0]
                jax.effects_barrier()
                print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
            
            image = self._postprocess_image(image)
        
        return (image,)
    
    def _get_or_create_vae(self, model_id):
        if (Flux2TPUVAEDecoder._cached_vae is not None and 
            Flux2TPUVAEDecoder._cached_model_id == model_id):
            print("  Using cached VAE")
            return Flux2TPUVAEDecoder._cached_vae
        
        print(f"  Loading VAE from {model_id}...")
        from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
        
        # 重要：模型加载必须在 torchax 环境外进行
        was_initialized = TorchaxEnvManager._initialized
        if was_initialized:
            TorchaxEnvManager.pause()
        
        try:
            vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        finally:
            if was_initialized:
                TorchaxEnvManager.resume()
        
        with TorchaxContext() as ctx:
            print("  - Converting VAE to XLA...")
            move_module_to_xla(ctx.env, vae)
            print("  - Compiling VAE Decoder...")
            vae.decoder = torchax.compile(vae.decoder)
            print(f"  - Replicating weights to {len(mesh.devices)} TPU cores...")
            vae.decoder.params = shard_weight_dict(vae.decoder.params, VAE_DECODER_SHARDINGS, mesh)
            vae.decoder.buffers = shard_weight_dict(vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh)
        
        gc.collect()
        Flux2TPUVAEDecoder._cached_vae = vae
        Flux2TPUVAEDecoder._cached_model_id = model_id
        print("  VAE ready!")
        return vae
    
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
        """后处理图像：XLA -> CPU -> ComfyUI 格式"""
        if hasattr(image, '_elem'):
            jax_image = image._elem
            if jax_image.dtype == jnp.bfloat16:
                np_image = np.array(jax_image.astype(jnp.float32))
            else:
                np_image = np.array(jax_image)
            image = torch.from_numpy(np_image)
        else:
            image = image.cpu()
        
        # 转换为 ComfyUI 格式: (B, H, W, C), 范围 [0, 1]
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
