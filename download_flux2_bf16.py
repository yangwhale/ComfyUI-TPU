#!/usr/bin/env python3
"""
下载 FLUX.2-dev bf16 权重并转换为 ComfyUI 格式

使用方法：
    python download_flux2_bf16.py

需要事先登录 HuggingFace:
    huggingface-cli login
"""

import os
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file, load_file

# 输出目录
OUTPUT_DIR = Path("models/diffusion_models")
OUTPUT_FILE = OUTPUT_DIR / "flux2-dev-bf16.safetensors"

MODEL_ID = "black-forest-labs/FLUX.2-dev"


def download_transformer():
    """从 HuggingFace 下载 FLUX.2 transformer 权重。"""
    print(f"正在从 {MODEL_ID} 下载 transformer...")
    
    # 下载 transformer 子文件夹中的所有 safetensors 文件
    local_dir = snapshot_download(
        MODEL_ID,
        allow_patterns=["transformer/*.safetensors", "transformer/config.json"],
        local_dir="/dev/shm/flux2_download",
    )
    
    print(f"下载完成: {local_dir}")
    return Path(local_dir) / "transformer"


def convert_diffusers_to_comfyui(diffusers_dir: Path) -> dict:
    """将 Diffusers 格式转换为 ComfyUI 格式。
    
    Diffusers 命名: transformer_blocks.0.attn.to_q.weight
    ComfyUI 命名:   diffusion_model.double_blocks.0.img_attn.qkv.weight (合并 QKV)
    """
    print("正在加载 Diffusers 权重...")
    
    # 加载所有分片
    state_dict = {}
    for f in sorted(diffusers_dir.glob("*.safetensors")):
        print(f"  加载 {f.name}...")
        shard = load_file(str(f))
        state_dict.update(shard)
    
    print(f"原始权重数量: {len(state_dict)}")
    
    # Diffusers -> ComfyUI 映射
    # Double Blocks (transformer_blocks)
    # Single Blocks (single_transformer_blocks)
    
    comfyui_state = {}
    
    # 辅助函数：添加前缀
    def add_prefix(name):
        return f"diffusion_model.{name}"
    
    for key, tensor in state_dict.items():
        new_key = None
        
        # === Double Blocks ===
        # Attention Q/K/V -> 合并为 qkv
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.(to_q|to_k|to_v)\.weight', key)
        if m:
            block_id = m.group(1)
            qkv_key = f"double_blocks.{block_id}.img_attn.qkv.weight"
            # 跳过，稍后合并处理
            continue
        
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.to_out\.0\.weight', key)
        if m:
            block_id = m.group(1)
            new_key = add_prefix(f"double_blocks.{block_id}.img_attn.proj.weight")
        
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.(add_q_proj|add_k_proj|add_v_proj)\.weight', key)
        if m:
            # txt_attn 的 Q/K/V，需要合并
            continue
        
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.to_add_out\.weight', key)
        if m:
            block_id = m.group(1)
            new_key = add_prefix(f"double_blocks.{block_id}.txt_attn.proj.weight")
        
        # MLP (ff/ff_context)
        m = re.match(r'transformer_blocks\.(\d+)\.ff\.(linear_in|linear_out)\.weight', key)
        if m:
            block_id, layer = m.groups()
            idx = "0" if layer == "linear_in" else "2"
            new_key = add_prefix(f"double_blocks.{block_id}.img_mlp.{idx}.weight")
        
        m = re.match(r'transformer_blocks\.(\d+)\.ff_context\.(linear_in|linear_out)\.weight', key)
        if m:
            block_id, layer = m.groups()
            idx = "0" if layer == "linear_in" else "2"
            new_key = add_prefix(f"double_blocks.{block_id}.txt_mlp.{idx}.weight")
        
        # Modulation
        m = re.match(r'transformer_blocks\.(\d+)\.(double_stream_modulation_img|double_stream_modulation_txt)\.linear\.weight', key)
        if m:
            block_id, mod_type = m.groups()
            mod_name = "img_mod" if "img" in mod_type else "txt_mod"
            new_key = add_prefix(f"double_blocks.{block_id}.{mod_name}.lin.weight")
        
        # Norm
        m = re.match(r'transformer_blocks\.(\d+)\.norm1(_context)?\.linear\.weight', key)
        if m:
            block_id = m.group(1)
            is_context = m.group(2) is not None
            norm_name = "txt_norm" if is_context else "img_norm"
            # ComfyUI 可能没有单独的 norm weight
            continue
        
        # === Single Blocks ===
        m = re.match(r'single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj\.weight', key)
        if m:
            block_id = m.group(1)
            new_key = add_prefix(f"single_blocks.{block_id}.linear1.weight")
        
        m = re.match(r'single_transformer_blocks\.(\d+)\.attn\.to_out\.weight', key)
        if m:
            block_id = m.group(1)
            new_key = add_prefix(f"single_blocks.{block_id}.linear2.weight")
        
        m = re.match(r'single_transformer_blocks\.(\d+)\.single_stream_modulation\.linear\.weight', key)
        if m:
            block_id = m.group(1)
            new_key = add_prefix(f"single_blocks.{block_id}.modulation.lin.weight")
        
        # === Embedders ===
        if key == 'x_embedder.weight':
            new_key = add_prefix("img_in.weight")
        elif key == 'context_embedder.weight':
            new_key = add_prefix("txt_in.weight")
        elif key == 'proj_out.weight':
            new_key = add_prefix("final_layer.linear.weight")
        
        # Time + Guidance
        m = re.match(r'time_guidance_embed\.(timestep_embedder|guidance_embedder)\.(linear_1|linear_2)\.weight', key)
        if m:
            embed_type, linear = m.groups()
            embed_name = "time_in" if embed_type == "timestep_embedder" else "guidance_in"
            layer_name = "in_layer" if linear == "linear_1" else "out_layer"
            new_key = add_prefix(f"{embed_name}.{layer_name}.weight")
        
        if new_key:
            comfyui_state[new_key] = tensor.to(torch.bfloat16)
        elif not any([
            "to_q" in key, "to_k" in key, "to_v" in key,
            "add_q" in key, "add_k" in key, "add_v" in key,
            "norm" in key
        ]):
            # 直接复制未匹配的（添加前缀）
            comfyui_state[add_prefix(key)] = tensor.to(torch.bfloat16)
    
    # 合并 Q/K/V 权重为 qkv
    print("正在合并 QKV 权重...")
    for block_id in range(100):  # 假设最多 100 个 block
        # img_attn qkv
        q_key = f"transformer_blocks.{block_id}.attn.to_q.weight"
        k_key = f"transformer_blocks.{block_id}.attn.to_k.weight"
        v_key = f"transformer_blocks.{block_id}.attn.to_v.weight"
        
        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            q = state_dict[q_key]
            k = state_dict[k_key]
            v = state_dict[v_key]
            qkv = torch.cat([q, k, v], dim=0).to(torch.bfloat16)
            comfyui_state[add_prefix(f"double_blocks.{block_id}.img_attn.qkv.weight")] = qkv
        
        # txt_attn qkv
        q_key = f"transformer_blocks.{block_id}.attn.add_q_proj.weight"
        k_key = f"transformer_blocks.{block_id}.attn.add_k_proj.weight"
        v_key = f"transformer_blocks.{block_id}.attn.add_v_proj.weight"
        
        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            q = state_dict[q_key]
            k = state_dict[k_key]
            v = state_dict[v_key]
            qkv = torch.cat([q, k, v], dim=0).to(torch.bfloat16)
            comfyui_state[add_prefix(f"double_blocks.{block_id}.txt_attn.qkv.weight")] = qkv
    
    print(f"转换后权重数量: {len(comfyui_state)}")
    return comfyui_state


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FLUX.2-dev BF16 模型下载与转换")
    print("=" * 60)
    
    # 下载
    transformer_dir = download_transformer()
    
    # 转换
    comfyui_state = convert_diffusers_to_comfyui(transformer_dir)
    
    # 保存
    print(f"正在保存到 {OUTPUT_FILE}...")
    save_file(comfyui_state, str(OUTPUT_FILE))
    
    # 计算文件大小
    size_gb = OUTPUT_FILE.stat().st_size / (1024 ** 3)
    print(f"✓ 保存完成: {OUTPUT_FILE} ({size_gb:.2f} GB)")
    
    # 清理
    print("清理临时文件...")
    import shutil
    shutil.rmtree("/dev/shm/flux2_download", ignore_errors=True)
    
    print("=" * 60)
    print("✓ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
