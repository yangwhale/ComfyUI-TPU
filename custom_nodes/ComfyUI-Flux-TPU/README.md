# ComfyUI-Flux-TPU

在 Google Cloud TPU 上运行 Flux.2 图像生成的 ComfyUI 自定义节点。

## 功能特性

- 🚀 **TPU 加速**: 使用 torchax 在 TPU 上运行 Flux.2 Transformer
- 🔧 **模块化设计**: 分离的 Text Encoder、Sampler 和 VAE Decoder 节点
- ⚡ **Splash Attention**: 针对长序列的 TPU 优化 attention 实现
- 🔄 **自动分片**: 自动将模型权重分布到 8 个 TPU 核心

## 节点说明

| 节点 | 运行位置 | 功能 |
|------|----------|------|
| **Flux.2 Text Encoder (CPU)** | CPU | 使用 Mistral3 编码文本 prompt |
| **Flux.2 TPU Sampler** | TPU | 运行 Transformer 去噪，生成 latents |
| **Flux.2 TPU VAE Decoder** | TPU | 解码 latents 为最终图像 |
| **Flux.2 TPU Full Pipeline** | TPU | 端到端图像生成（组合以上三个） |

## 安装

1. 确保你在 TPU v5litepod-8 或兼容的 TPU 环境中

2. 克隆仓库到 ComfyUI custom_nodes 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yangwhale/ComfyUI-TPU.git
cd ComfyUI-TPU/custom_nodes/ComfyUI-Flux-TPU
```

3. 安装依赖：
```bash
pip install torchax diffusers transformers
```

## 使用方法

### 方法 1: 加载示例 Workflow

1. 启动 ComfyUI: `python main.py --cpu`
2. 在 ComfyUI 界面中，点击 **Load** 按钮
3. 选择 `custom_nodes/ComfyUI-Flux-TPU/examples/flux2_tpu_basic.json`

### 方法 2: 手动创建 Workflow

1. 添加 **Flux.2 Text Encoder (CPU)** 节点
   - 输入你的 prompt
   - 输出连接到 Sampler

2. 添加 **Flux.2 TPU Sampler** 节点
   - 设置 height/width (如 1024x1024)
   - 设置 steps (推荐 50)
   - 设置 guidance_scale (推荐 4.0)
   - 输出连接到 VAE Decoder

3. 添加 **Flux.2 TPU VAE Decoder** 节点
   - 确保 height/width 与 Sampler 一致
   - 输出连接到 Preview Image

4. 添加 **Preview Image** 节点查看结果

## Workflow 示意图

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│ Flux.2 Text Encoder │────▶│ Flux.2 TPU      │────▶│ Flux.2 TPU VAE      │────▶│ Preview Image │
│ (CPU)               │     │ Sampler         │     │ Decoder             │     │               │
│                     │     │                 │     │                     │     │               │
│ prompt: "..."       │     │ height: 1024    │     │ height: 1024        │     │               │
│ model_id: ...       │     │ width: 1024     │     │ width: 1024         │     │               │
│                     │     │ steps: 50       │     │ model_id: ...       │     │               │
│                     │     │ guidance: 4.0   │     │                     │     │               │
│                     │     │ seed: ...       │     │                     │     │               │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘     └───────────────┘
       prompt_embeds ────────────▶ LATENT ─────────────────▶ IMAGE ─────────────────▶
```

## 参数说明

### Text Encoder
- **prompt**: 图像描述文本
- **model_id**: 模型 ID (默认: `black-forest-labs/FLUX.2-dev`)

### TPU Sampler
- **height/width**: 输出图像尺寸 (256-2048, 步长 64)
- **num_inference_steps**: 去噪步数 (推荐 50)
- **guidance_scale**: 引导强度 (推荐 4.0)
- **seed**: 随机种子

### VAE Decoder
- **height/width**: 必须与 Sampler 设置一致
- **model_id**: 模型 ID

## 示例 Workflow 文件

示例 workflow 文件位于 `examples/` 目录：

- [`flux2_tpu_basic.json`](examples/flux2_tpu_basic.json) - 基础 workflow

## 保存自定义 Workflow

在 ComfyUI 中创建好 workflow 后：

1. 点击界面顶部的 **Save** 按钮（或 Ctrl+S）
2. 输入文件名，选择保存位置
3. workflow 将保存为 `.json` 文件

你可以将保存的 workflow 分享给其他用户，他们可以通过 **Load** 按钮加载使用。

## 性能

在 TPU v5litepod-8 上的典型性能：

| 分辨率 | Steps | 时间 |
|--------|-------|------|
| 512x512 | 50 | ~30s |
| 1024x1024 | 50 | ~60s |

## 问题排查

### "torchax Tensors can only do math within the torchax environment"

这个错误已经在最新版本中修复。确保使用最新代码。

### 模型加载失败

确保你有访问 `black-forest-labs/FLUX.2-dev` 模型的权限，并且已经登录 HuggingFace：

```bash
huggingface-cli login
```

## 许可证

MIT License
