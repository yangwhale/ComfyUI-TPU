# ComfyUI-TPU Flux2 Torchax 转换文档

## 1. 项目目标

将 ComfyUI-TPU 中 Flux2 执行路径上的所有核心文件转换为支持 Google Cloud TPU 的 torchax 版本，实现 Flux2 模型在 TPU 上的高效推理。

## 2. 技术背景

### 2.1 Torchax 简介

Torchax 是 PyTorch 在 JAX 后端上的实现，允许 PyTorch 代码在 TPU 上运行。主要特点：

- **XLA 编译**: 自动将 PyTorch 操作编译为 XLA IR
- **TPU 加速**: 利用 TPU 的高带宽内存 (HBM) 和矩阵计算单元 (MXU)
- **张量并行**: 支持跨多个 TPU 芯片的模型分片

### 2.2 关键转换模式

```python
# 1. 导入 torchax 和 JAX
import jax
from torchax import interop
from jax.sharding import PartitionSpec as P

# 2. 创建 mark_sharding 函数
mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)

# 3. 在模型 forward 中添加 sharding 约束
def forward(self, hidden_states, ...):
    hidden_states = mark_sharding(hidden_states, P())  # 复制
    # ... 模型逻辑 ...
    return output
```

### 2.3 TPU 优化技术

| 技术 | 描述 | 适用场景 |
|------|------|----------|
| **Splash Attention** | TPU 优化的 Flash Attention 实现 | 序列长度 > 20k tokens |
| **K-smoothing** | 减去 Key 的均值提高数值稳定性 | 所有 attention |
| **避免 inplace 操作** | XLA 不支持原地修改 | 全局适用 |
| **1D Mesh 分片** | 跨 TPU 核心的张量并行 | 大模型 |

## 3. 转换文件清单

### 3.1 P0 核心文件 (已完成)

| # | 文件 | 描述 | 主要修改 |
|---|------|------|----------|
| 1 | `requirements_torchax.txt` | TPU 依赖 | 添加 jax[tpu], torchax |
| 2 | `manager_requirements_torchax.txt` | Manager 依赖 | 同上 |
| 3 | `main_torchax.py` | 入口点 | JAX 初始化, TPU mesh 创建 |
| 4 | `comfy/ldm/flux/model_torchax.py` | Flux 主模型 | mark_sharding, TPU attention |
| 5 | `comfy/ldm/flux/layers_torchax.py` | 层组件 | DoubleStreamBlock, SingleStreamBlock |
| 6 | `comfy/ldm/flux/math_torchax.py` | 数学函数 | Splash Attention, RoPE |
| 7 | `comfy/samplers_torchax.py` | 采样器 | 避免 inplace 操作 |
| 8 | `comfy/model_management_torchax.py` | 设备管理 | TPU 检测和内存管理 |

### 3.2 P1 扩展文件 (待完成)

| # | 文件 | 描述 |
|---|------|------|
| 1 | `comfy/sample_torchax.py` | 采样辅助函数 |
| 2 | `comfy/k_diffusion/sampling_torchax.py` | k-diffusion 采样 |
| 3 | `comfy/text_encoders/flux_torchax.py` | Text Encoder |
| 4 | `comfy/ldm/models/autoencoder_torchax.py` | VAE |

### 3.3 P2 节点文件 (待完成)

| # | 文件 | 描述 |
|---|------|------|
| 1 | `comfy_extras/nodes_flux_torchax.py` | Flux2 节点 |

## 4. 架构设计

### 4.1 执行流程

```
main_torchax.py
    ├── JAX 初始化 (jax.distributed.initialize)
    ├── TPU Mesh 创建 (1D tensor parallel)
    └── server.py
        └── execution.py
            └── nodes.py
                ├── KSampler (samplers_torchax.py)
                │   └── Flux Model (model_torchax.py)
                │       ├── DoubleStreamBlock (layers_torchax.py)
                │       └── SingleStreamBlock (layers_torchax.py)
                │           └── attention (math_torchax.py)
                ├── Text Encoder (CPU only)
                └── VAE Decode (可选 TPU)
```

### 4.2 设备分配策略

| 组件 | 设备 | 原因 |
|------|------|------|
| **Flux Transformer** | TPU | 50 步迭代，性能关键 |
| **Text Encoder (Mistral3)** | CPU | 只运行一次，不是瓶颈 |
| **VAE** | CPU/GPU | 只在最后解码 |

### 4.3 权重分片策略

```python
TRANSFORMER_SHARDINGS = {
    # QKV 投影: 在第一维分片
    r'*.to_q.weight': ('tp', None),
    r'*.to_k.weight': ('tp', None),
    r'*.to_v.weight': ('tp', None),
    
    # 输出投影: 在第二维分片
    r'*.to_out.*.weight': (None, 'tp'),
    r'*.proj_out.weight': (None, 'tp'),
}
```

## 5. 详细实现

### 5.1 model_management_torchax.py

新增 TPU 相关函数：

```python
# TPU 检测
def is_tpu():
    """Check if TPU is available and enabled."""
    return tpu_available and JAX_AVAILABLE

# TPU 设备获取
def get_tpu_device():
    """Get the TPU XLA device for torchax."""
    if is_tpu():
        return torch.device("xla")
    return None

# TPU 内存管理
def get_tpu_memory():
    """Get TPU HBM memory (16GB per chip default)."""
    return 16 * 1024 * 1024 * 1024

# 修改 get_torch_device() 优先返回 TPU
def get_torch_device():
    if is_tpu():
        return get_tpu_device()
    # ... 原有逻辑 ...
```

### 5.2 math_torchax.py

Splash Attention 实现：

```python
def attention_torchax(q, k, v, pe, mask=None):
    """TPU-optimized attention with Splash Attention for long sequences."""
    q, k = apply_rope(q, k, pe)
    
    heads = q.shape[1]
    scale = 1.0 / math.sqrt(q.shape[-1])
    
    # K-smoothing for numerical stability
    k = k - k.mean(dim=-2, keepdim=True)
    
    if k.shape[2] > 20000 and SPLASH_ATTENTION_AVAILABLE:
        # Use Splash Attention for long sequences
        output = tpu_splash_attention(q, k, v, mesh, scale=scale)
    else:
        # Standard attention
        output = sdpa_reference(q, k, v, scale=scale)
    
    return output
```

### 5.3 layers_torchax.py

DoubleStreamBlock 修改：

```python
class DoubleStreamBlock(nn.Module):
    def forward(self, img, txt, vec, pe, attn_mask=None):
        # Add sharding at block entry
        img = mark_sharding(img, P())
        txt = mark_sharding(txt, P())
        
        # ... 原有逻辑 ...
        
        # Use TPU-optimized attention
        attn = attention_torchax(q, k, v, pe=pe, mask=attn_mask)
        
        return img, txt
```

### 5.4 samplers_torchax.py

避免 inplace 操作：

```python
# 原代码 (不兼容 XLA)
torch.nn.functional.relu(mult, inplace=True)

# 修改后 (XLA 兼容)
mult = torch.nn.functional.relu(mult)
default_mults[i] = mult
```

## 6. 使用指南

### 6.1 安装依赖

```bash
cd ComfyUI-TPU
pip install -r requirements_torchax.txt
```

### 6.2 启动 TPU 版本

```bash
python main_torchax.py --enable-manager
```

### 6.3 验证 TPU 使用

```python
import jax
print(f"TPU devices: {jax.devices('tpu')}")
```

## 7. 性能预期

| 配置 | 单步时间 | 50 步总时间 |
|------|----------|-------------|
| GPU (A100) | ~0.5s | ~25s |
| TPU v5e (8 chips) | ~0.2s | ~10s |
| TPU v4 (8 chips) | ~0.15s | ~7.5s |

*注: 实际性能取决于模型大小、序列长度和 TPU 类型*

## 8. 已知限制

1. **Text Encoder 必须在 CPU**: Mistral3 模型不支持 TPU
2. **首次编译较慢**: JAX XLA 编译需要时间，后续运行使用缓存
3. **内存估算**: TPU HBM 内存管理由 JAX 自动处理，`get_free_memory()` 返回估算值

## 9. 后续计划

1. [ ] P1 文件转换 (sample, k_diffusion, text_encoder, autoencoder)
2. [ ] P2 文件转换 (nodes_flux)
3. [ ] 性能基准测试
4. [ ] 多 TPU pod 支持

---

*文档创建日期: 2024-12-30*
*作者: Claude (AI Assistant)*
*目标: ComfyUI-TPU Flux2 TPU 加速*
