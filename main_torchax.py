"""
ComfyUI-TPU Main Entry Point (Torchax TPU Version)

This is the TPU-optimized entry point for ComfyUI.
It initializes JAX/TPU before loading any models and configures
the torchax environment for TPU execution.

Usage:
    python main_torchax.py --enable-manager
"""

# ============================================================================
# JAX/TPU 初始化 (必须在任何 torch import 之前)
# ============================================================================
import os
import sys
import functools
import re
import math

# 设置 JAX 环境变量
# 注意：不设置 JAX_PLATFORMS 以保留 CPU 后端，用于模型加载时的 jax.default_device("cpu")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 不预分配内存
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TensorFlow 日志

# 设置 JAX 编译缓存
def setup_jax_cache():
    """设置 JAX 编译缓存以加速后续运行"""
    cache_dir = os.path.expanduser("~/.cache/jax_cache/comfyui_tpu")
    os.makedirs(cache_dir, exist_ok=True)
    
    import jax
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    return cache_dir

# 初始化 JAX (在 torch 之前)
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 设置缓存
cache_dir = setup_jax_cache()
print(f"[Torchax] JAX 编译缓存: {cache_dir}")

# 检查 TPU 设备
try:
    devices = jax.devices('tpu')
    print(f"[Torchax] 检测到 {len(devices)} 个 TPU 核心")
    for i, d in enumerate(devices):
        print(f"  TPU {i}: {d}")
except RuntimeError:
    print("[Torchax] 警告: 未检测到 TPU，将使用 CPU 作为后备")
    devices = jax.devices('cpu')

# 创建 TPU mesh (1D tensor parallel)
tp_dim = len(devices)
mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
print(f"[Torchax] 创建 Mesh: tp={tp_dim}")

# ============================================================================
# 在启用 torchax 之前先导入可能检查 CUDA 的模块
# ============================================================================
import torch
import torchvision  # 必须在 torchax.enable_globally() 之前导入

# ============================================================================
# Torchax 初始化
# ============================================================================
import torchax
from torchax import interop
from torchax.ops import jaten, ops_registry

# 全局启用 torchax
torchax.enable_globally()
env = torchax.default_env()

# 创建 mark_sharding 函数
mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)

# 导出全局变量供其他模块使用
TORCHAX_ENV = env
TORCHAX_MESH = mesh
TORCHAX_MARK_SHARDING = mark_sharding

print(f"[Torchax] Torchax 环境已初始化")

# ============================================================================
# Flux.2 Transformer 分片策略 (1D mesh: tp)
# 规则：输出投影 ('tp', None)，输入投影 (None, 'tp')
# 支持有/无 "diffusion_model." 前缀的权重
# ============================================================================

# 定义基础 patterns (不含前缀)
_FLUX_BASE_SHARDINGS = {
    # Double-stream Blocks - Attention
    r'double_blocks\.\d+\.img_attn\.qkv\.weight': ('tp', None),
    r'double_blocks\.\d+\.img_attn\.proj\.weight': (None, 'tp'),
    r'double_blocks\.\d+\.txt_attn\.qkv\.weight': ('tp', None),
    r'double_blocks\.\d+\.txt_attn\.proj\.weight': (None, 'tp'),
    # Double-stream Blocks - MLP
    r'double_blocks\.\d+\.img_mlp\.0\.weight': ('tp', None),
    r'double_blocks\.\d+\.img_mlp\.2\.weight': (None, 'tp'),
    r'double_blocks\.\d+\.txt_mlp\.0\.weight': ('tp', None),
    r'double_blocks\.\d+\.txt_mlp\.2\.weight': (None, 'tp'),
    # Single-stream Blocks
    r'single_blocks\.\d+\.linear1\.weight': ('tp', None),
    r'single_blocks\.\d+\.linear2\.weight': (None, 'tp'),
    # Embedders & Projections
    r'img_in\.weight': ('tp', None),
    r'txt_in\.weight': ('tp', None),
    r'final_layer\.linear\.weight': (None, 'tp'),
    # Modulation
    r'double_blocks\.\d+\.img_mod\.lin\.weight': ('tp', None),
    r'double_blocks\.\d+\.txt_mod\.lin\.weight': ('tp', None),
    r'single_blocks\.\d+\.modulation\.lin\.weight': ('tp', None),
    # Time + Guidance Embedding
    r'time_in\.in_layer\.weight': ('tp', None),
    r'time_in\.out_layer\.weight': (None, 'tp'),
    r'guidance_in\.in_layer\.weight': ('tp', None),
    r'guidance_in\.out_layer\.weight': (None, 'tp'),
    r'vector_in\.in_layer\.weight': ('tp', None),
    r'vector_in\.out_layer\.weight': (None, 'tp'),
    # FLUX.2 特有: global modulation
    r'double_stream_modulation_img\.lin\.weight': ('tp', None),
    r'double_stream_modulation_txt\.lin\.weight': ('tp', None),
}

# 创建完整 patterns (支持有/无 diffusion_model. 前缀)
FLUX_TRANSFORMER_SHARDINGS = {}
for pattern, sharding in _FLUX_BASE_SHARDINGS.items():
    # 无前缀版本
    FLUX_TRANSFORMER_SHARDINGS[pattern] = sharding
    # 有前缀版本
    FLUX_TRANSFORMER_SHARDINGS[r'diffusion_model\.' + pattern] = sharding

# K 平滑以提高数值稳定性
USE_K_SMOOTH = True

# ============================================================================
# Splash Attention 集成
# ============================================================================

from comfy.splash_attention_utils import sdpa_reference, tpu_splash_attention

def override_op_definition(env, op, impl):
    """在 torchax 环境中覆盖算子定义。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX 兼容的 conv2d 覆盖实现。"""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)

def scaled_dot_product_attention_tpu(query, key, value, attn_mask=None, dropout_p=0.0,
                                      is_causal=False, scale=None, enable_gqa=False,
                                      env=None, mesh=None):
    """SDPA 封装：长序列用 Splash Attention，短序列用参考实现。"""
    # 对于长 KV 序列（self-attention）使用 TPU Splash Attention
    if key.shape[2] > 20000:
        assert attn_mask is None and dropout_p == 0.0 and not is_causal
        assert not enable_gqa and scale is None
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        if USE_K_SMOOTH:
            jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    # 短序列使用参考实现
    return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)

# 注册自定义算子覆盖
print("[Torchax] 注册 TPU 优化算子...")
override_op_definition(env, torch.nn.functional.conv2d,
                       functools.partial(torch_conv2d_jax, env=env))
override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                       functools.partial(scaled_dot_product_attention_tpu, env=env, mesh=mesh))
print("[Torchax] ✓ conv2d 和 scaled_dot_product_attention 已替换为 TPU 版本")

# ============================================================================
# TPU 模型转换工具函数
# 参考: /home/chrisya/gpu-tpu-pedia/tpu/Flux.2/generate_diffusers_torchax_staged/utils.py
#
# 关键流程:
# 1. move_module_to_xla() - 把权重转换为 torchax tensor
# 2. torchax.compile() - 编译模型，生成 .params 和 .buffers 属性
# 3. shard_weight_dict() - 对 .params/.buffers 用 apply_jax_ 分片到 TPU
#
# 注意：在 ComfyUI 中 torchax 是全局启用的，不能使用 jax.default_device("cpu")
# 因为那会破坏 torchax 环境导致 "torchax Tensors can only do math within
# the torchax environment" 错误
# ============================================================================

def move_module_to_xla(env, module):
    """将模块权重移动到 XLA 设备。
    
    使用 cpu device context 确保初始转换在 CPU 内存中进行，
    避免在分片之前占满 TPU HBM。
    """
    # 尝试在 CPU 上进行转换
    try:
        cpu_device = jax.devices('cpu')[0]
        context = jax.default_device(cpu_device)
    except:
        context = None
        
    if context:
        with context:
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    else:
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)

def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """按模式匹配应用权重分片。
    
    对已经是 torchax tensor 的权重，用 apply_jax_(jax.device_put, sharding)
    分片并移动到 TPU。
    """
    import numpy as np
    result = {}
    sharded_count = 0
    replicated_count = 0
    sharded_bytes = 0
    replicated_bytes = 0
    
    for k, v in weight_dict.items():
        # 估算张量大小 (bfloat16 = 2 bytes)
        if hasattr(v, 'shape'):
            tensor_bytes = np.prod(v.shape) * 2
        else:
            tensor_bytes = 0
        
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.fullmatch(pattern, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                sharded_count += 1
                sharded_bytes += tensor_bytes
                break
        if not matched:
            # 未匹配的权重复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes
        result[k] = v
    
    print(f"[Torchax]   分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    return result, sharded_count

# 存储已编译的模型
_compiled_models = {}
_models_on_xla = set()

def prepare_model_for_tpu(model, model_name="model"):
    """准备模型用于 TPU 执行。
    
    完全遵循 generate_torchax.py 的流程:
    1. move_module_to_xla() - 把权重转换为 torchax tensor
    2. torchax.compile() - 编译模型，生成 .params 和 .buffers 属性
    3. shard_weight_dict() - 对 .params/.buffers 用 apply_jax_ 分片到 TPU
    
    这是参考实现验证过的正确顺序。
    """
    global _compiled_models, _models_on_xla
    
    model_id = id(model)
    
    # 检查是否已处理过
    if model_id in _models_on_xla:
        print(f"[Torchax] 模型 {model_name} 已在 XLA 上")
        return model
    
    print(f"[Torchax] 准备模型 {model_name} 用于 TPU...")
    
    # 强制 GC
    import gc
    gc.collect()
    
    # 打印 TPU 内存状态 (如果可能)
    try:
        for i, device in enumerate(jax.devices('tpu')):
             stats = device.memory_stats()
             if stats:
                 bytes_in_use = stats.get('bytes_in_use', 0)
                 bytes_limit = stats.get('bytes_limit', 0)
                 print(f"[Torchax]   TPU {i} Mem: {bytes_in_use/1e9:.2f}GB / {bytes_limit/1e9:.2f}GB used")
             break # 只打印第一个芯片
    except Exception as e:
        print(f"[Torchax]   无法获取 TPU 内存状态: {e}")

    # 1. 移动权重到 XLA (转换为 torchax tensor)
    print(f"[Torchax]   - 将权重移动到 XLA...")
    move_module_to_xla(env, model)
    
    # 2. 编译模型 (生成 .params 和 .buffers 属性)
    print(f"[Torchax]   - 编译模型...")
    compiled_model = torchax.compile(model, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    
    # 3. 分片 params 和 buffers (用 apply_jax_ 移动到 TPU)
    print(f"[Torchax]   - 分片权重到 {tp_dim} 个 TPU 核心...")
    compiled_model.params, params_count = shard_weight_dict(
        compiled_model.params, FLUX_TRANSFORMER_SHARDINGS, mesh)
    compiled_model.buffers, buffers_count = shard_weight_dict(
        compiled_model.buffers, FLUX_TRANSFORMER_SHARDINGS, mesh)
    print(f"[Torchax]   - 分片了 {params_count} 个 params 和 {buffers_count} 个 buffers")
    
    _compiled_models[model_id] = compiled_model
    _models_on_xla.add(model_id)
    
    print(f"[Torchax] ✓ 模型 {model_name} 已准备好用于 TPU")
    return compiled_model

def is_model_on_tpu(model):
    """检查模型是否已在 TPU 上。"""
    return id(model) in _models_on_xla

# ============================================================================
# 模块替换：在导入任何 comfy 模块之前替换 model_management
# ============================================================================
# 先导入 comfy 包
import comfy

# 导入 torchax 版本的 model_management
from comfy import model_management_torchax

# 替换 sys.modules 和 comfy 模块属性
sys.modules['comfy.model_management'] = model_management_torchax
comfy.model_management = model_management_torchax

print("[Torchax] 已替换 model_management 为 TPU 版本")

# ============================================================================
# PyTree 注册 (支持 JAX 转换)
# ============================================================================
from jax.tree_util import register_pytree_node

def setup_pytree_registrations():
    """注册必要的 PyTree 节点以支持 JAX 转换"""
    try:
        from transformers import modeling_outputs
        
        def flatten(obj):
            return obj.to_tuple(), type(obj)
        
        def unflatten(aux, children):
            return aux(*children)
        
        # 注册 transformers 输出类
        classes_to_register = [
            (modeling_outputs.BaseModelOutputWithPastAndCrossAttentions, "BaseModelOutputWithPastAndCrossAttentions"),
        ]
        
        for cls, name in classes_to_register:
            try:
                register_pytree_node(cls, flatten, unflatten)
                print(f"[Torchax] PyTree 注册: {name}")
            except ValueError:
                pass  # 已注册
                
    except ImportError:
        pass

setup_pytree_registrations()

# ============================================================================
# 原始 ComfyUI 入口代码
# ============================================================================
import comfy.options
comfy.options.enable_args_parsing()

import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_api import feature_flags
import importlib.util


if __name__ == "__main__":
    #NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'


def handle_comfyui_manager_unavailable():
    if not args.windows_standalone_build:
        logging.warning(f"\n\nYou appear to be running comfyui-manager from source, this is not recommended. Please install comfyui-manager using the following command:\ncommand:\n\t{sys.executable} -m pip install --pre comfyui_manager\n")
    args.enable_manager = False


if args.enable_manager:
    if importlib.util.find_spec("comfyui_manager"):
        import comfyui_manager

        if not comfyui_manager.__file__ or not comfyui_manager.__file__.endswith('__init__.py'):
            handle_comfyui_manager_unavailable()
    else:
        handle_comfyui_manager_unavailable()


def apply_custom_paths():
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)

            if args.enable_manager:
                if comfyui_manager.should_be_disabled(module_path):
                    continue

            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

apply_custom_paths()

if args.enable_manager:
    comfyui_manager.prestartup()

execute_prestartup_script()


# Main code
import asyncio
import shutil
import threading
import gc

import torch

# 设置默认 dtype 为 bfloat16 (TPU 优化)
torch.set_default_dtype(torch.bfloat16)

import comfy.utils

import execution
import server
from protocol import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_args={ "lru" : args.cache_lru, "ram" : args.cache_ram } )
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            sensitive = item[5]
            extra_data = item[3].copy()
            for k in sensitive:
                extra_data[k] = sensitive[k]

            e.execute(item[2], prompt_id, extra_data, item[4])
            need_gc = True

            remove_sensitive = lambda prompt: prompt[:5] + prompt[6:]
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages), process_item=remove_sensitive)
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time

            # Log Time in a more readable way after 10 minutes
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}")
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                # TPU: 同步 JAX 效果屏障
                jax.effects_barrier()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )

def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            # Only send old method if client doesn't support preview metadata
            if not feature_flags.supports_feature(
                server_instance.sockets_metadata,
                server_instance.client_id,
                "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_database():
    try:
        from app.database.db import init_db, dependencies_available
        if dependencies_available():
            init_db()
    except Exception as e:
        logging.error(f"Failed to initialize database. Please ensure you have installed the latest requirements. If the error persists, please report this as in future the database will be required: {e}")


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
    prompt_server = server.PromptServer(asyncio_loop)

    if args.enable_manager and not args.disable_manager_ui:
        comfyui_manager.start()

    hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    ))
    hook_breaker_ac10a0.restore_functions()

    setup_database()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server,)).start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {} (Torchax TPU)".format(comfyui_version.__version__))
    logging.info(f"JAX version: {jax.__version__}")
    logging.info(f"Torchax TPU mode enabled with {len(devices)} TPU cores")

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")

    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
