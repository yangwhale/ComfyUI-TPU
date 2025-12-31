#!/usr/bin/env python3
"""
下载 ComfyUI 格式的 FLUX 模型（bf16）

可选模型:
1. FLUX.2-dev (最新版本，约 23GB)
2. FLUX.1-dev / FLUX.1-schnell

使用方法：
    python download_flux_comfyui.py flux2-dev-bf16

需要事先登录 HuggingFace:
    huggingface-cli login
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# 输出目录
OUTPUT_DIR = Path("models/diffusion_models")

# 可选的模型源
MODELS = {
    # ===== FLUX.2 (最新) =====
    "flux2-dev-bf16": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "output": "flux2-dev-bf16.safetensors"
    },
    
    # ===== FLUX.1 =====
    # Comfy-Org 官方提供的 ComfyUI 格式 (fp8)
    "comfy-org-dev": {
        "repo_id": "Comfy-Org/flux1-dev",
        "filename": "flux1-dev-fp8.safetensors",
        "output": "flux1-dev-fp8.safetensors"
    },
    "comfy-org-schnell": {
        "repo_id": "Comfy-Org/flux1-schnell",
        "filename": "flux1-schnell-fp8.safetensors",
        "output": "flux1-schnell-fp8.safetensors"
    },
    # Black Forest Labs 原始 bf16 版本
    "bfl-dev-bf16": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "filename": "flux1-dev.safetensors",
        "output": "flux1-dev-bf16.safetensors"
    },
    "bfl-schnell-bf16": {
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "filename": "flux1-schnell.safetensors",
        "output": "flux1-schnell-bf16.safetensors"
    },
}


def download_model(model_key: str = "bfl-dev-bf16"):
    """下载指定的模型。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_key not in MODELS:
        print(f"未知模型: {model_key}")
        print(f"可用模型: {list(MODELS.keys())}")
        return
    
    model_info = MODELS[model_key]
    output_path = OUTPUT_DIR / model_info["output"]
    
    if output_path.exists():
        print(f"模型已存在: {output_path}")
        return output_path
    
    print(f"正在下载 {model_info['repo_id']} / {model_info['filename']}...")
    print(f"目标路径: {output_path}")
    
    downloaded = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
    )
    
    # 重命名（如果需要）
    downloaded_path = Path(downloaded)
    if downloaded_path.name != model_info["output"]:
        downloaded_path.rename(output_path)
    
    size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"✓ 下载完成: {output_path} ({size_gb:.2f} GB)")
    
    return output_path


def main():
    import sys
    
    print("=" * 60)
    print("FLUX 模型下载器 (ComfyUI 格式)")
    print("=" * 60)
    print()
    print("可用模型:")
    for key, info in MODELS.items():
        print(f"  {key}: {info['repo_id']} / {info['filename']}")
    print()
    
    # 默认下载 FLUX.2 bf16 版本（用于 TPU）
    model_key = sys.argv[1] if len(sys.argv) > 1 else "flux2-dev-bf16"
    
    print(f"选择: {model_key}")
    print()
    
    download_model(model_key)
    
    print()
    print("=" * 60)
    print("✓ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
