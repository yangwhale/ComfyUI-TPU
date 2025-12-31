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
"""

import functools
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec as P
from torchax.ops import jaten, ops_registry

from . import mesh
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


# ============================================================================
# 自定义算子注册
# ============================================================================

def _register_operators_on_env(env, mesh_obj):
    """
    在 torchax 环境上注册 TPU 所需的自定义算子。
    
    注册的算子:
      - conv2d: 2D 卷积
      - layer_norm / native_layer_norm: 层归一化
      - group_norm / native_group_norm: 组归一化
      - scaled_dot_product_attention: Splash Attention
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
    
    # ---- Splash Attention ----
    try:
        from .splash_attention import sdpa_reference, tpu_splash_attention
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
# Wan 2.1 Text Encoder (TPU)
# ============================================================================

class Wan21TextEncoder:
    """
    Wan 2.1 Text Encoder - 在 TPU 上运行 T5-XXL 编码 prompt。
    
    输入: prompt, negative_prompt 文本
    输出: prompt_embeds, negative_prompt_embeds tensor
    """
    
    _cached_pipe = None
    _cached_model_id = None
    
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
        
        pipe = self._get_or_create_pipeline(model_id)
        
        with TorchaxContext() as ctx:
            with mesh:
                print("  Encoding prompts...")
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    device='jax',
                )
            
            # 转换回 CPU
            prompt_embeds_cpu = prompt_embeds.to('cpu')
            negative_prompt_embeds_cpu = negative_prompt_embeds.to('cpu')
        
        print(f"  prompt_embeds shape: {prompt_embeds_cpu.shape}")
        return (prompt_embeds_cpu, negative_prompt_embeds_cpu)
    
    def _get_or_create_pipeline(self, model_id):
        if (Wan21TextEncoder._cached_pipe is not None and 
            Wan21TextEncoder._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Wan21TextEncoder._cached_pipe
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # 加载时需要暂停 torchax
        was_initialized = TorchaxEnvManager._initialized
        if was_initialized:
            TorchaxEnvManager.pause()
        
        try:
            from diffusers import WanPipeline
            pipe = WanPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, use_safetensors=True
            )
        finally:
            if was_initialized:
                TorchaxEnvManager.resume()
        
        with TorchaxContext() as ctx:
            print("  - Converting Text Encoder to XLA...")
            move_module_to_xla(ctx.env, pipe.text_encoder)
            pipe.text_encoder = torchax.compile(pipe.text_encoder)
            
            print(f"  - Sharding Text Encoder weights...")
            pipe.text_encoder.params = shard_weight_dict(
                pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            pipe.text_encoder.buffers = shard_weight_dict(
                pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )
            torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
        
        # 删除不需要的组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            del pipe.transformer
            pipe.transformer = None
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        Wan21TextEncoder._cached_pipe = pipe
        Wan21TextEncoder._cached_model_id = model_id
        print("  Text Encoder ready!")
        return pipe


# ============================================================================
# Wan 2.1 TPU Sampler
# ============================================================================

class Wan21TPUSampler:
    """
    Wan 2.1 TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    输入: prompt_embeds, negative_prompt_embeds
    输出: latents (用于 VAE Decoder)
    """
    
    _cached_pipe = None
    _cached_model_id = None
    
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
        
        print(f"\n[Wan21TPUSampler] Starting TPU inference...")
        print(f"  Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        pipe = self._get_or_create_pipeline(model_id, flow_shift)
        
        with TorchaxContext() as ctx:
            prompt_embeds_xla = prompt_embeds.to('jax')
            negative_prompt_embeds_xla = negative_prompt_embeds.to('jax')
            
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            with mesh:
                print(f"  Running denoising loop...")
                start_time = time.perf_counter()
                
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
                )
                jax.effects_barrier()
                
                elapsed = time.perf_counter() - start_time
                print(f"  Done: {elapsed:.2f}s ({elapsed/num_inference_steps:.2f}s/step)")
            
            latents = result.frames
            torch_latents = self._convert_latents_to_cpu(latents)
        
        return ({"samples": torch_latents, "num_frames": num_frames}, num_frames)
    
    def _convert_latents_to_cpu(self, latents):
        """将 XLA latents 转换回 CPU tensor"""
        if hasattr(latents, '_elem'):
            jax_latents = latents._elem
            if jax_latents.dtype == jnp.bfloat16:
                return torch.from_numpy(np.array(jax_latents.astype(jnp.float32))).to(torch.bfloat16)
            return torch.from_numpy(np.array(jax_latents))
        return latents.cpu()
    
    def _get_or_create_pipeline(self, model_id, flow_shift):
        if (Wan21TPUSampler._cached_pipe is not None and 
            Wan21TPUSampler._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Wan21TPUSampler._cached_pipe
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # 加载时需要暂停 torchax
        was_initialized = TorchaxEnvManager._initialized
        if was_initialized:
            TorchaxEnvManager.pause()
        
        try:
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
        finally:
            if was_initialized:
                TorchaxEnvManager.resume()
        
        with TorchaxContext() as ctx:
            print("  - Converting Transformer to XLA...")
            move_module_to_xla(ctx.env, pipe.transformer)
            
            # Move rope embeddings
            if hasattr(pipe.transformer.rope, 'freqs'):
                pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
            else:
                pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
                pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')
            
            print("  - Compiling Transformer...")
            pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict',)}))
            
            print(f"  - Sharding weights to {len(mesh.devices)} TPU cores...")
            pipe.transformer.params = shard_weight_dict(pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
            pipe.transformer.buffers = shard_weight_dict(pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
            torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
        
        # 删除不需要的组件
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        
        gc.collect()
        Wan21TPUSampler._cached_pipe = pipe
        Wan21TPUSampler._cached_model_id = model_id
        print("  Pipeline ready!")
        return pipe


# ============================================================================
# Wan 2.1 TPU VAE Decoder
# ============================================================================

class Wan21TPUVAEDecoder:
    """
    Wan 2.1 VAE Decoder - 在 TPU 上解码 latents 为视频。
    
    输入: latents (来自 Sampler)
    输出: video frames (ComfyUI IMAGE 格式)
    """
    
    _cached_vae = None
    _cached_model_id = None
    
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
        
        vae = self._get_or_create_vae(model_id)
        
        with TorchaxContext() as ctx:
            # 处理 latents
            processed_latents = self._process_latents(latent_tensor, vae)
            processed_latents = ctx.env.to_xla(processed_latents.to(torch.bfloat16))
            
            with mesh:
                print("  Decoding...")
                start_time = time.perf_counter()
                with torch.no_grad():
                    video = vae.decode(processed_latents).sample
                jax.effects_barrier()
                print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
            
            # 后处理
            video = video.to('cpu')
            frames = prepare_video_for_export(video, num_frames)
        
        # 转换为 ComfyUI IMAGE 格式 [T, H, W, C]
        frames_tensor = torch.from_numpy(frames)
        
        print(f"  Output: {frames_tensor.shape}")
        return (frames_tensor, fps)
    
    def _process_latents(self, latents, vae):
        """处理 latents: denormalize"""
        print(f"  Processing latents: {latents.shape}")
        
        # 检查 nan
        nan_count = torch.isnan(latents).sum().item()
        if nan_count > 0:
            print(f"  Warning: {nan_count} nan values, replacing with 0")
            latents = torch.nan_to_num(latents, nan=0.0)
        
        # Denormalize: x * std + mean
        latents_mean = getattr(vae.config, 'latents_mean', None)
        latents_std = getattr(vae.config, 'latents_std', None)
        
        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
            std = torch.tensor(latents_std).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents * std + mean
        
        return latents
    
    def _get_or_create_vae(self, model_id):
        if (Wan21TPUVAEDecoder._cached_vae is not None and 
            Wan21TPUVAEDecoder._cached_model_id == model_id):
            print("  Using cached VAE")
            return Wan21TPUVAEDecoder._cached_vae
        
        print(f"  Loading VAE from {model_id}...")
        
        # 加载时需要暂停 torchax
        was_initialized = TorchaxEnvManager._initialized
        if was_initialized:
            TorchaxEnvManager.pause()
        
        try:
            from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.bfloat16
            )
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
        Wan21TPUVAEDecoder._cached_vae = vae
        Wan21TPUVAEDecoder._cached_model_id = model_id
        print("  VAE ready!")
        return vae


# ============================================================================
# Wan 2.1 Full Pipeline
# ============================================================================

class Wan21TPUPipeline:
    """
    Wan 2.1 TPU Full Pipeline - 端到端视频生成。
    
    组合 TextEncoder -> Sampler -> VAEDecoder 三个阶段。
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
        
        # Stage 2: Denoising (TPU)
        latents, _ = Wan21TPUSampler().sample(
            prompt_embeds, negative_prompt_embeds,
            height, width, num_frames,
            num_inference_steps, guidance_scale, seed,
            model_id, flow_shift
        )
        
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
