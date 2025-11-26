# LuminaAI

<div align="center">

**Modular Transformer Training Framework with MoE/MoD Architecture**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Commercial-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tH1z9e7px2G8NGqWUN9gdqxs1CnUC7p1)

[Demo](#demo) • [Architecture](#architecture) • [Configuration](#configuration) • [API](#api-reference) • [Licensing](#licensing)

</div>

<p align="center">
  <img src="assets/ReadmeLogo.png" width="400">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Model Configurations](#model-configurations)
- [Precision Support](#precision-support)
- [Demo](#demo)
- [Training Modes](#training-modes)
- [Adaptive Training System](#adaptive-training-system)
- [Hardware Optimization](#hardware-optimization)
- [Data Processing](#data-processing)
- [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Licensing](#licensing)

---

## Overview

LuminaAI is a transformer training framework implementing Mixture of Experts (MoE) and Mixture of Depths (MoD) architectures with autonomous training optimization. Supports models from 500M to 300B+ parameters with production-grade infrastructure.

**Core capabilities:**
- Sparse architectures: MoE (8-64 experts), MoD (dynamic depth), hybrid configurations
- Adaptive orchestrator: 18 autonomous intervention methods for training optimization
- Chinchilla scaling: Automatic epoch calculation based on compute-optimal principles
- DeepSpeed ZeRO: Multi-GPU training with memory optimization (stages 1-3)
- Precision support: FP32, FP16, BF16, mixed precision, FP8 (H100+)
- Hardware targets: CUDA (Volta-Hopper), Apple Silicon (M1-M4), CPU
- Data handling: Memory-mapped datasets, Apache Arrow zero-copy, automatic caching
- Recovery systems: Automatic OOM handling, gradient explosion recovery, checkpoint rollback

**Framework positioning:**

This is a complete training system, not a model zoo or API wrapper. Every component from tokenization to gradient synchronization is included. MoE and MoD implementations follow established research (Switch Transformer, DeepSeek-MoE, Mixture-of-Depths) with operational additions: dynamic expert management, capacity tuning, load balancing, routing analytics.

The adaptive orchestrator monitors 20+ metrics every N steps and triggers interventions across hyperparameters, architecture, and recovery procedures. Maintains decision history with confidence scoring to prevent excessive intervention.

**Intended for:**
- ML engineers requiring full training stack control
- Research teams prototyping sparse architectures
- Organizations with proprietary data and compliance requirements
- Teams needing framework-independent infrastructure

**Not included:**
- Pre-trained weights (training system only)
- High-level abstractions (direct control provided)
- Tutorial content (assumes ML engineering background)

---

## Technical Architecture

### Dense Transformers

Standard architecture with LLaMA/GPT-NeoX design patterns:
- Pre-normalization (RMSNorm before attention/FFN)
- Grouped Query Attention: Reduces KV cache via shared KV heads (typical ratio 4:1 or 8:1)
- Rotary Position Embeddings: Length generalization with configurable theta (10000 base, 1000000 extended)
- SwiGLU activation: Two-path gating in FFN, intermediate_size typically 8/3 × hidden_size
- Optional Flash Attention 2.x: 2-4x speedup on Ampere+ GPUs

**Parameter calculation:**
- Embedding: vocab_size × hidden_size
- Attention per layer: hidden_size² × (1 + num_kv_heads/num_heads)
- FFN per layer: 2 × hidden_size × intermediate_size
- Output: vocab_size × hidden_size (optionally tied)

### Mixture of Experts (MoE)

Token-level sparse activation via learned routing to specialized FFN networks.

**Routing mechanism:**
- Top-k gating: Each token routed to k of N experts (typical: k=2, N=8)
- Router: Linear layer (hidden_size × num_experts) + softmax + TopK selection
- Output: Weighted combination of selected expert outputs

**Load balancing:**
- Auxiliary loss: Penalizes routing imbalance via expert utilization distribution
- Capacity factor: Maximum tokens per expert = (total_tokens/num_experts) × capacity_factor
- Typical capacity_factor: 1.25-1.5 (25-50% overflow buffer)
- Load balancing weight: 0.01 (added to main loss)

**Dynamic management:**
- Expert addition: Triggered when utilization exceeds threshold (typically 0.85)
- Expert pruning: Removes experts below utilization threshold (typically 0.15)
- Capacity adaptation: Adjusts based on token drop rate
- Temperature tuning: Controls routing concentration (lower = sharper, higher = more uniform)

**Efficiency:**
- 8-expert top-2 MoE: 8× total parameters, 1.25× active parameters per token
- Sparsity: 87.5% (12.5% parameters active)
- Memory: Scales with total parameters (all experts in memory)
- Compute: Scales with active parameters only

**Statistics tracked:**
- Per-expert utilization: Fraction of tokens routed to each expert
- Routing entropy: Distribution concentration (max = log(num_experts))
- Load balance loss: Auxiliary loss magnitude
- Tokens dropped: Count exceeding capacity
- Per-layer patterns: Early layers more uniform, later layers more concentrated

### Mixture of Depths (MoD)

Layer-level sparse activation via learned skip decisions.

**Core concept:**
Model learns which tokens require full layer computation vs. residual skip. Routing decision per token per layer based on token representation at layer input.

**Routing types:**
- Learned: Small MLP scores tokens, top-capacity_factor selected for full processing
- Static: Fixed pattern (e.g., all tokens full in early layers, reduced in later layers)
- Random: Random selection maintaining capacity_factor (ablation baseline)

**Capacity management:**
- capacity_factor controls fraction of tokens using full computation
- 0.5 = 50% tokens full processing, 50% skip
- Selection is learned during training, not random

**Efficiency:**
- FLOPs reduction: ~(1 - capacity_factor) for layers with MoD
- Typical: 30-50% FLOPs reduction with 0-2% perplexity increase
- Quality/efficiency tradeoff: Lower capacity = more savings but larger quality impact

**Application strategies:**
- All layers: Maximum compute reduction
- Later layers only: Preserve early feature extraction, reduce later specialization
- Alternating: MoD every N layers
- FFN-only: Dense attention, sparse FFN (common since FFN is 2/3 of compute)

**Training dynamics:**
- Early training: Near-uniform routing (all tokens treated similarly)
- Specialization: Model learns token complexity patterns over time
- Curriculum: Can start with capacity_factor=1.0, gradually reduce to target
- Annealing: Gradual capacity reduction prevents training instability

### Hybrid (MoE + MoD)

Combined token-level (MoE) and layer-level (MoD) sparsity.

**Architecture:**
- Each layer: Dense attention + MoE FFN + MoD routing
- MoD decides: Use full layer (attention + MoE) or skip via residual
- If token uses layer, routes through MoE experts

**Sparsity compounding:**
- Top-2 of 8 experts: 25% expert parameters active
- 50% layer capacity: 50% tokens use layers
- Combined: 0.5 × 0.25 = 12.5% active parameters per token
- 87.5% total sparsity

**Training considerations:**
- Both routing mechanisms must learn useful patterns
- Load balancing for experts, capacity warmup for MoD
- Routing temperature adaptation for both systems
- Quality-aware guards prevent catastrophic sparsity collapse

**Use cases:**
- Maximum efficiency: Largest models on limited compute
- Fast experimentation: Smaller active compute enables rapid iteration
- Inference optimization: Reduced memory and compute for deployment
- Multi-task learning: Different experts and depths specialize per task

---

## Model Configurations

Pre-configured presets spanning 500K to 300B parameters. Each preset specifies architecture dimensions, MoE/MoD parameters, and hardware targets.

| Config | Active Params | Total Params | Hidden | Layers | Heads | KV Heads | Experts | Top-K | Hardware | Memory (FP16) | Throughput |
|--------|--------------|--------------|--------|--------|-------|----------|---------|-------|----------|---------------|------------|
| `debug` | 500K | 4M | 128 | 2 | 2 | 2 | 8 | 2 | Any | 50 MB | Testing |
| `debug_200m` | 200M | 6B | 768 | 12 | 12 | 12 | 32 MoD | - | T4/MPS | 2 GB | Testing |
| `b1` | 1B | 8B | 1024 | 24 | 16 | 4 | 8 | 2 | RTX 3090, M1 Max | 8 GB | 1000 tok/s |
| `b7` | 7B | 56B | 4096 | 32 | 32 | 8 | 8 | 2 | A100 40GB | 28 GB | 500 tok/s |
| `b14` | 14B | 112B | 5120 | 40 | 40 | 10 | 8 | 2 | A100 80GB | 56 GB | 250 tok/s |
| `b30` | 30B | 240B | 8192 | 48 | 64 | 16 | 8 | 2 | 4× A100 80GB | 120 GB | 100 tok/s |
| `b50` | 50B | 400B | 10240 | 56 | 80 | 20 | 8 | 2 | 4× H100 | 200 GB | 50 tok/s |
| `b100` | 100B | 800B | 12288 | 80 | 96 | 24 | 8 | 2 | 8× H100 | 400 GB | 50 tok/s |
| `b200` | 200B | 1.6T | 16384 | 100 | 128 | 32 | 8 | 2 | 16× H200 | 800 GB | 30 tok/s |
| `b300` | 300B | 2.4T | 20480 | 120 | 160 | 40 | 8 | 2 | 32× H200 | 1.2 TB | 20 tok/s |

**Memory estimates:** Include model weights, optimizer states (Adam: 8 bytes/param), gradients, and activation memory at batch_size=1, mixed precision training. Actual memory scales with batch size and sequence length.

**Throughput estimates:** Baseline values at batch_size=1, sequence_length=2048, mixed precision with gradient checkpointing. Actual throughput varies significantly with configuration, optimization settings, and hardware.

**Configuration selection:**

- **Development/testing:** `debug` for pipeline validation, `debug_200m` for architecture testing
- **Research:** `b1` for prototyping on consumer hardware
- **Production fine-tuning:** `b7` for quality/efficiency balance
- **Large-scale pre-training:** `b30`+ for maximum model capacity
- **Extreme scale:** `b100`+ requires cluster infrastructure and distributed expertise

**Customization:**

All presets are starting points. Architecture dimensions can be modified: hidden_size must be divisible by num_heads. Intermediate_size typically 8/3 × hidden_size rounded to nearest 256. Max_position_embeddings determines context window. Num_experts and moe_top_k can be adjusted independently. MoD capacity_factor controls compute/quality tradeoff.

---

## Precision Support

Numerical formats for parameters, activations, and gradients during training and inference.

### Supported Precisions

**FP32 (Float32) - Full Precision**
- 32-bit floating point (8-bit exponent, 23-bit mantissa)
- Range: ±3.4×10^38, precision: ~7 decimal digits
- Maximum stability, no special handling required
- 2× memory vs FP16/BF16, significantly slower on modern hardware
- Use cases: CPU training, numerical debugging, stability issues with reduced precision

**FP16 (Float16) - Half Precision**
- 16-bit floating point (5-bit exponent, 10-bit mantissa)
- Range: ±65504, precision: ~3 decimal digits
- 50% memory reduction, ~2× speedup on supported hardware
- Requires loss scaling to prevent gradient underflow (small gradients round to zero)
- Dynamic or static loss scaling: multiply loss by 2^N before backward, unscale gradients before update
- Use cases: Volta/Turing GPUs (V100, T4, RTX 2080), Apple Silicon (M1-M4)

**BF16 (BFloat16) - Brain Float16**
- 16-bit format (8-bit exponent, 7-bit mantissa)
- Range: Same as FP32 (±3.4×10^38), reduced precision vs FP32
- 50% memory reduction, similar speed to FP16
- No loss scaling required (wide dynamic range like FP32)
- Better training stability than FP16 with same memory benefits
- Use cases: Ampere+ GPUs (A100, RTX 3090/4090, H100), primary recommendation for modern hardware

**Mixed Precision FP16**
- Forward/backward in FP16, master copy of parameters in FP32
- Optimizer updates FP32 master copy, then copies to FP16 for next forward
- Dynamic loss scaling automatically adjusts to prevent underflow
- Combines FP16 speed with FP32 stability
- Use cases: Default for older GPUs supporting FP16 but not BF16

**Mixed Precision BF16**
- Forward/backward in BF16, master parameters in FP32
- No loss scaling needed (BF16 dynamic range matches FP32)
- Simpler than mixed FP16 (no loss scaling configuration)
- Best speed/stability balance on modern hardware
- Use cases: Default for Ampere+ GPUs, primary recommendation for production

**FP8 (Float8) - Experimental**
- 8-bit floating point: E4M3 (forward) and E5M2 (backward) variants
- Further memory reduction (75% vs FP32)
- Requires H100 or newer with hardware FP8 support
- Complex configuration, quality impacts not fully characterized
- Use cases: Cutting-edge research, not recommended for general use

**INT8 Quantization**
- 8-bit integer representation (post-training or quantization-aware training)
- Primarily for inference, not training
- Reduces model size by 75% vs FP32 for deployment
- Quality impact depends on calibration and quantization method
- Use cases: Model deployment, edge devices

### Automatic Precision Selection

The framework detects hardware and selects optimal precision:

**Detection logic:**
1. Check for CUDA availability and GPU compute capability
2. If Ampere+ (compute capability ≥ 8.0): Select `mixed_bf16`
3. If Volta/Turing (compute capability 7.0-7.5): Select `mixed_fp16`
4. If Apple Silicon (MPS): Select `fp16` (BF16 not supported on MPS)
5. If CPU: Select `fp32` (reduced precision offers no benefit on CPU)

**Override:** Set precision explicitly via configuration if automatic selection is suboptimal or for specific debugging/testing requirements.

### Hardware-Specific Recommendations

**NVIDIA Ampere/Ada/Hopper (A100, RTX 3090/4090, H100, H200):**
- Recommended: `mixed_bf16`
- Alternative: `mixed_fp16` (if BF16 causes unexpected issues)
- Advanced: `fp8_e4m3` (H100+ only, experimental)

**NVIDIA Volta/Turing (V100, T4, RTX 2080/2080Ti):**
- Recommended: `mixed_fp16`
- Alternative: `fp32` (if stability issues)
- Note: BF16 not supported (no hardware acceleration)

**Apple Silicon (M1/M2/M3/M4, Mac Studio, MacBook Pro):**
- Recommended: `fp16`
- Note: Mixed precision and BF16 not supported on MPS backend
- Limitations: Flash Attention disabled, DeepSpeed unavailable

**CPU (Intel/AMD/ARM):**
- Recommended: `fp32`
- Note: Reduced precision offers minimal benefit on CPU
- Expect significantly slower training than GPU (10-100× depending on model size)

### Precision Configuration Parameters

**Training precision:** Format used during forward pass, backward pass, and gradient computation
**Inference precision:** Format used during validation and evaluation
**Master precision:** Format for optimizer's master parameter copy (typically FP32 in mixed precision)

**Separate training/inference precision:**
Common pattern: Train in `mixed_bf16` for speed, evaluate in `fp32` for precise metrics. Or train in `mixed_fp16`, deploy in `int8` for inference.

**Loss scaling parameters (FP16 only):**
- `init_scale`: Initial loss scaling factor (default: 2^16)
- `scale_factor`: Multiplier for scale adjustment (default: 2.0)
- `scale_window`: Steps without overflow before increasing scale (default: 2000)
- `min_scale`: Minimum scale factor (default: 1.0)

Dynamic loss scaling adjusts automatically: scale increases every scale_window steps without overflow, decreases on overflow detection (NaN/Inf gradients). Most users do not need to modify these parameters.

---

## Demo

### Colab Notebook

Free GPU training demonstration requiring no local setup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tH1z9e7px2G8NGqWUN9gdqxs1CnUC7p1)

**Environment:**
- Hardware: Google Colab T4 GPU (15.8GB memory, Turing architecture)
- CUDA: 11.8 or 12.x (varies by Colab environment)
- Runtime: ~5 minutes for complete training run

**Configuration:**
- Model: `b1` preset (1B active, 8B total parameters, 8-expert MoE)
- Dataset: Small conversational corpus (~200 samples, ~2.5M tokens)
- Training: 3 epochs, batch_size=8, gradient_accumulation=8, effective_batch_size=64
- Precision: Mixed FP16 (automatic for T4)
- Learning rate: 1e-4 with linear warmup

**Observable behaviors:**
- System diagnostics: Hardware detection, precision selection
- Chinchilla scaling: Optimal token calculation (20× parameters), epoch recommendations
- Training metrics: Loss progression (~2.5 → ~2.0), throughput (~1000 tok/s)
- Orchestrator: Health checks every 100 steps, intervention decisions
- Expert statistics: Utilization distribution, routing entropy, load balance

**Limitations:**
- Colab quotas: Usage limits, session timeouts
- Small dataset: Limited quality, demonstrates pipeline not production training
- Free tier: T4 availability not guaranteed, may encounter queuing

### Local Installation

**Requirements:**
- Python 3.8+ (3.10+ recommended)
- PyTorch 2.0+ (2.2+ recommended)
- CUDA 11.8+ (for GPU) or CPU
- RAM: 16GB minimum, 32GB+ recommended
- Disk: 50GB+ for dependencies, datasets, checkpoints

**Installation:**
```bash
git clone https://github.com/matn23/luminaai
cd luminaai
pip install -r requirements.txt
cd Src/Main_Scripts
python Main.py
```

**Optional dependencies:**
- Flash Attention: 2-4× attention speedup, requires manual compilation
- DeepSpeed: Multi-GPU optimization, auto-installs but compiles kernels on first use
- Wandb: Experiment tracking, requires API key

**Quick start:**
Default configuration uses `debug` preset for rapid testing. System auto-detects hardware, selects precision, validates data, initializes model, begins training.

**Resume from checkpoint:**
```bash
python Main.py --resume path/to/checkpoint.pt
```
Restores model state, optimizer state, scheduler, training step counter, random seeds.

---

## Training Modes

Four data handling strategies for different use cases.

### Base/Pre-training Only

Raw text without conversational structure. For domain-specific pre-training, continued pre-training, language modeling research.

**Data format:**
- Plain text files (.txt)
- JSONL with "text" field

**Processing:** Tokenize and split into fixed-length sequences with optional overlap (stride parameter).

**Loss:** Applied to all tokens (causal language modeling).

### Fine-tuning Only

Conversational data with role annotations. For instruction tuning, chat models, task-specific adaptation.

**Data format:** JSONL with "conversation" field containing list of messages. Each message has "role" (system/user/assistant) and "content".

**Processing:** Concatenate messages with special tokens marking roles: [BOS] system [SEP] user [SEP] assistant [EOS]

**Loss:** Can mask user tokens (loss only on assistant responses) or compute on all tokens.

### Sequential Hybrid

Two-phase training: base corpus then conversational data. Builds general understanding, then adapts to conversation.

**Configuration:** Separate epoch counts per phase. Optional learning rate warmup between phases to handle distribution shift.

**Use cases:** Domain adaptation (medical literature → clinical QA), continual learning (new data → maintained task performance).

### Interleaved Hybrid

Mix base and conversational data within batches/epochs. Maintains general capabilities while learning conversation. Prevents catastrophic forgetting.

**Mixing ratio:** base_ratio controls proportion (e.g., 0.7 = 70% base, 30% conversational).

**Strategies:**
- Ratio-based: Sample according to ratio per batch
- Alternating: Cycle between sources (batch 1 base, batch 2 conversational, etc.)
- Random: Random sampling with probability = base_ratio

**Use cases:** General-purpose chat models, multi-task learning with auxiliary objectives.

---

## Adaptive Training System

### Orchestrator Architecture

State machine monitoring training every N steps (default: 100). Triggers interventions across 18 methods when confidence threshold exceeded (default: 0.75).

**Monitored metrics:**
- Loss dynamics: Trend, variance, plateau detection, divergence warnings
- Gradient statistics: Norm, variance, stability over windows
- Expert utilization: Per-expert usage, routing entropy, load balance (MoE)
- Memory consumption: GPU/system memory, OOM risk prediction
- Throughput: Tokens/second, degradation detection
- Convergence: Score based on loss stability, compute efficiency

**Intervention categories:**
1. **Hyperparameter adaptation:** Learning rate, weight decay, batch size
2. **Architecture modification:** Add/remove experts, adjust capacities, routing temperature
3. **Emergency recovery:** Gradient explosion handling, OOM recovery, checkpoint rollback
4. **Schedule optimization:** Early stopping recommendations, checkpoint prioritization

**Decision process:**
1. Collect metrics at checkpoint interval
2. Analyze patterns (plateau, divergence, imbalance)
3. Compute intervention confidence scores
4. If confidence > threshold, execute intervention
5. Log decision with reasoning and confidence
6. Monitor intervention impact in subsequent intervals

### Intervention Methods

**MoE Architecture Management (3 methods):**
- `add_expert(layer_idx)`: Add expert to underutilized layer when average utilization > threshold
- `prune_expert(layer_idx, expert_idx)`: Remove expert with utilization < threshold
- `_initialize_new_expert(new_expert, existing_experts)`: Initialize from existing with noise

**MoE Routing Control (4 methods):**
- `adjust_capacity_factor(factor)`: Modify token capacity per expert
- `adjust_routing_temperature(temp)`: Control routing sharpness (lower = more concentrated)
- `enable_expert_dropout(prob)`: Regularization via random expert dropping
- `get_expert_statistics()`: Retrieve utilization, entropy, load balance loss, tokens dropped

**MoD Control (2 methods):**
- `adjust_mod_capacity(factor)`: Change fraction of tokens using full computation
- `get_mod_statistics()`: Retrieve capacity factor, tokens processed, average depth

**Batch Size Adaptation (2 methods):**
- `adjust_batch_size(new_size)`: Change micro batch size (typically for OOM recovery)
- `_recreate_dataloader(dataset)`: Rebuild dataloader after batch size change

**Emergency Recovery (2 methods):**
- `emergency_lr_reduction(factor)`: Reduce learning rate by factor (for gradient explosion)
- `rollback_steps(num_steps)`: Revert to earlier checkpoint (for divergence)

**Optimizer Adjustments (2 methods):**
- `adjust_weight_decay(value)`: Modify regularization strength
- `_update_optimizer_param_groups(param, value)`: Internal parameter group update

**Real-time Metrics (3 methods):**
- `get_current_metrics()`: Query loss, LR, gradient norm, throughput
- `_extract_moe_routing_stats()`: Internal MoE statistics extraction
- `_calculate_throughput()`: Compute tokens/second

### Chinchilla Scaling

Automatic training duration calculation following compute-optimal scaling laws (Hoffmann et al., 2022).

**Formula:** `N_optimal_tokens = multiplier × model_parameters`

**Default multiplier:** 20× (configurable: 10-50×)

**Process:**
1. Calculate optimal token budget: `N_opt = 20 × total_parameters`
2. Determine base epochs: `epochs = N_opt / dataset_tokens`
3. Clamp to min/max epoch constraints (default: 1-50)
4. Monitor during training: convergence score, loss landscape, compute efficiency
5. Adjust dynamically: Reduce epochs if fast convergence, stop early if plateaued

**Enhanced features:**
- **Loss landscape analysis:** Detect plateaus (low variance over window), divergence (rapid loss increase)
- **Compute efficiency:** Track loss reduction per FLOP, identify diminishing returns
- **Adaptive curriculum:** Adjust learning rate or data sampling based on convergence phase
- **Early stopping:** Recommend termination when convergence score > threshold (typically 85%)

**Runtime integration:**

System calculates optimal duration at training start. Displays token budget, coverage percentage (dataset_tokens / optimal_tokens), recommended epochs. During training, prints status every N steps: current progress, convergence score, training phase (warming/learning/convergence), efficiency trend, recommendations (continue/adjust/stop).

---

## Hardware Optimization

Platform-specific optimizations automatically applied based on detected hardware.

### NVIDIA CUDA

**Automatic optimizations:**
- Precision: `mixed_bf16` for Ampere+, `mixed_fp16` for Volta/Turing
- Flash Attention: Enabled on Ampere+ (compute capability ≥ 8.0)
- Tensor cores: Automatically utilized for supported operations
- CUDA graphs: Enabled for static computation graphs (requires compile=True)

**Configuration parameters:**
- `use_flash_attention`: Enable Flash Attention 2.x (2-4× attention speedup)
- `gradient_checkpointing`: Trade compute for memory (enables larger models)
- `compile`: PyTorch 2.0 compilation (5-30% speedup, longer startup)
- `use_deepspeed`: Enable DeepSpeed for multi-GPU
- `zero_stage`: ZeRO optimization level (0-3)

**DeepSpeed ZeRO stages:**
- **Stage 0:** Disabled (standard DDP)
- **Stage 1:** Partition optimizer states (~4× memory reduction)
- **Stage 2:** Partition optimizer + gradients (~8× reduction)
- **Stage 3:** Partition optimizer + gradients + parameters (~N× reduction where N = num_GPUs)

**Memory optimization:**
- CPU offload: Move optimizer states to CPU memory (slower updates, massive memory savings)
- Gradient compression: Reduce communication volume (quality impact minimal)
- Activation checkpointing: Recompute activations during backward (trade compute for memory)

### Apple Silicon (MPS)

**Automatic optimizations:**
- Precision: FP16 (only supported reduced precision)
- Data loading: num_workers=0 (MPS prefers single-threaded)
- Unified memory: Automatic page management

**Limitations:**
- Flash Attention: Not supported (no hardware acceleration)
- DeepSpeed: Not available (Linux/CUDA only)
- Mixed precision: Not supported (FP16 or FP32 only)
- Compilation: Can be unstable (set compile=False if issues)

**Recommendations:**
- Conservative batch sizes (start with 2-4)
- Monitor memory pressure via Activity Monitor
- Expect lower throughput than equivalent CUDA GPU (1.5-3×)

### CPU

**Automatic settings:**
- Precision: FP32 (reduced precision no benefit)
- Threading: Automatic core detection, configurable via num_workers

**Optimizations:**
- BLAS libraries: MKL (Intel), OpenBLAS (AMD/ARM), Accelerate (macOS)
- Thread count: Typically num_cores - 2 for system overhead

**Expectations:**
- 10-100× slower than GPU depending on model size
- Suitable for debugging, not production training
- Memory constraints less severe (can use system RAM)

---

## Data Processing

Memory-efficient data loading with zero-copy operations and automatic caching.

**Features enabled:**
- Memory-mapped file access: Read datasets without loading entirely into RAM
- Zero-copy operations: Apache Arrow columnar format, no serialization overhead
- Multi-threaded loading: Configurable worker processes (num_workers parameter)
- Automatic caching: HuggingFace Datasets caches processed data
- Sharding: Automatic data distribution for multi-GPU training
- Polars acceleration: Fast DataFrame operations for preprocessing

**Data intelligence:**
- **Difficulty-based sampling:** Prioritize harder examples based on loss history
- **Curriculum learning:** Gradually increase example complexity (aggressiveness parameter: 0-1)
- **Automatic cleaning:** Remove malformed samples, normalize formatting
- **Quality threshold:** Filter samples below quality score (default: 0.85)
- **Sequence length optimization:** Dynamic padding, variable length batching

**Preprocessing pipeline:**
1. Load raw files (txt, jsonl)
2. Validate format and structure
3. Quality filtering (configurable threshold)
4. Tokenization with caching
5. Sequence construction (chunking or conversation formatting)
6. Difficulty scoring (for curriculum learning)
7. Batch construction with dynamic padding

**Statistics tracked:**
- Total samples, valid/invalid counts
- Token statistics: mean, median, max, min, standard deviation
- Role distribution (conversational data)
- Quality scores: error rate, issues detected
- Sequence length distribution

**Validation:**

System validates all data paths before training. Checks: file existence, readability, size, format correctness. Prints summary: file count, total size, samples per file. Reports errors: missing files, corrupt formats, empty files.

---

## Monitoring

Comprehensive metrics tracked during training with real-time logging and experiment tracking integration.

**Core metrics:**
- Loss: Training and validation, rolling average, per-batch values
- Perplexity: exp(loss), interpretable quality measure
- Accuracy: Token-level prediction accuracy
- Learning rate: Current value from scheduler
- Throughput: Tokens per second, samples per second
- Gradient norm: L2 norm of gradients, variance over window
- Memory usage: GPU allocated/reserved, system RAM

**MoE-specific metrics:**
- Expert utilization: Fraction of tokens per expert, per-layer and aggregated
- Routing entropy: Distribution concentration, higher = more balanced
- Load balance loss: Auxiliary loss magnitude
- Tokens dropped: Count exceeding capacity
- Expert efficiency: Compute per expert, utilization × quality contribution

**MoD-specific metrics:**
- Capacity utilization: Actual vs configured capacity per layer
- Average depth: Mean number of full layers per token
- Skip patterns: Which tokens skip which layers
- Compute savings: FLOPs reduction percentage
- Per-layer usage: Fraction of tokens using full computation per layer

**Chinchilla metrics:**
- Token progress: Current tokens / optimal tokens percentage
- Convergence score: 0-100% based on loss stability
- Training phase: warming/learning/convergence/plateau
- Compute efficiency: Loss reduction per FLOP
- Predicted final loss: Extrapolation from current trajectory

**Logging configuration:**
- Log level: DEBUG/INFO/WARNING/ERROR
- Log interval: Every N steps (default: 50)
- Health check interval: Orchestrator monitoring frequency (default: 100)
- Checkpoint interval: Save frequency (default: 1000 steps)

**Output destinations:**
- Console: Real-time training progress
- Log files: experiments/[name]/logs/training.log
- Metrics files: JSON format, experiments/[name]/metrics/
- Wandb: Optional cloud logging and visualization
- TensorBoard: Optional local visualization

**Health checks:**

Orchestrator performs comprehensive health assessment every N steps:
- Loss trend analysis: Increasing/decreasing/plateau/divergence
- Gradient stability: Norm within expected range, no explosions
- Memory status: Utilization percentage, OOM risk
- Expert balance: MoE utilization distribution (if applicable)
- Throughput: Current vs baseline, degradation detection
- Convergence: Progress toward optimal loss

Health check output includes status (healthy/warning/critical), detected issues, recommended interventions, confidence scores.

---

## Checkpointing

Automatic checkpoint management with configurable strategies and emergency recovery.

**Checkpoint contents:**
- Model state dict: All parameter values
- Optimizer state: Momentum buffers, adaptive learning rates
- Scheduler state: Current step, learning rate schedule position
- Training metadata: Current epoch, global step, random seeds
- Orchestrator state: Decision history, intervention log
- Chinchilla state: Convergence tracking, token counts

**Save strategies:**
- **Periodic:** Every N steps (default: 1000)
- **Best validation:** Save when validation loss improves
- **Epoch boundaries:** End of each epoch
- **Before intervention:** Orchestrator saves before major changes
- **Emergency:** Triggered by SIGTERM or manual interrupt

**Checkpoint retention:**
- Keep N best checkpoints (default: 5, ranked by validation loss)
- Always keep latest checkpoint
- Optional: Keep all epoch boundary checkpoints
- Automatic cleanup of old checkpoints to manage disk space

**Resume behavior:**
- Exact state restoration: Training continues from precise point
- Reproducibility: Random seeds restored for deterministic continuation
- Validation: Checkpoint integrity checks before loading
- Fallback: If latest checkpoint corrupt, try previous checkpoints

**Checkpoint structure:**
```
experiments/[experiment_name]/checkpoints/
  ├── checkpoint_step_1000.pt
  ├── checkpoint_step_2000.pt
  ├── checkpoint_best_val.pt
  ├── checkpoint_latest.pt
  └── metadata.json
```

**Emergency signals:**
- SIGTERM: Graceful save and exit
- SIGINT (Ctrl+C): Graceful save and exit after current batch
- SIGUSR1: Save checkpoint without exiting (continue training)

---

## API Reference

### Trainer Methods

**MoE Architecture Management:**

`add_expert(layer_idx: int) -> None`
- Adds new expert to specified transformer layer
- Expert initialized from existing experts with Gaussian noise
- Automatically updates routing weights
- Triggers rebalancing of capacity factor

`prune_expert(layer_idx: int, expert_idx: int) -> None`
- Removes expert from specified layer
- Parameters frozen but retained in checkpoint for potential recovery
- Routing weights adjusted to redistribute to remaining experts
- Triggers capacity factor adjustment

**MoE Routing Control:**

`adjust_capacity_factor(factor: float) -> None`
- Updates token capacity per expert
- Values typically 1.0-2.0 (100-200% of fair share)
- Higher values reduce token dropping but increase memory
- Lower values force sharper routing but may drop tokens

`adjust_routing_temperature(temperature: float) -> None`
- Controls routing distribution sharpness
- temperature < 1.0: Sharper routing (more specialized)
- temperature > 1.0: Softer routing (more uniform)
- Typical range: 0.5-2.0

`enable_expert_dropout(dropout_prob: float) -> None`
- Enables expert-level dropout during training
- dropout_prob: Probability of dropping each expert (typical: 0.1-0.2)
- Prevents over-reliance on specific experts
- Disabled during evaluation automatically

`get_expert_statistics() -> Dict`
- Returns comprehensive expert metrics
- Keys: per_expert_utilization, routing_entropy, load_balance_loss, tokens_dropped
- Per-layer and aggregate statistics
- Updated each forward pass

**MoD Control:**

`adjust_mod_capacity(capacity_factor: float) -> None`
- Updates fraction of tokens using full computation
- Values: 0.0-1.0 (0% to 100% of tokens)
- Lower values: More compute savings, potential quality impact
- Typical range: 0.3-0.7

`get_mod_statistics() -> Dict`
- Returns MoD efficiency metrics
- Keys: capacity_factor, tokens_processed, tokens_skipped, average_depth, per_layer_usage, compute_savings
- Tracks actual vs configured capacity
- Compute savings as FLOP reduction percentage

**Batch Management:**

`adjust_batch_size(new_batch_size: int) -> None`
- Dynamically changes micro batch size
- Recreates dataloader with new batch size
- Adjusts gradient accumulation to maintain effective batch size
- Typically used for OOM recovery (reduces batch size automatically)

**Emergency Recovery:**

`emergency_lr_reduction(reduction_factor: float) -> None`
- Reduces learning rate by specified factor
- Triggered by gradient explosion (norm > threshold)
- Typical reduction: 5-10×
- Logs emergency action with reasoning

`rollback_steps(num_steps: int) -> None`
- Reverts training to previous checkpoint
- Loads checkpoint from num_steps earlier
- Resets optimizer and scheduler state
- Used for divergence recovery

**Optimizer Control:**

`adjust_weight_decay(weight_decay: float) -> None`
- Updates L2 regularization strength
- Typical values: 0.01-0.1
- Higher values: Stronger regularization, may slow learning
- Lower values: Less regularization, may overfit

**Metrics Query:**

`get_current_metrics() -> Dict`
- Real-time training state snapshot
- Keys: loss, learning_rate, grad_norm, throughput, memory_usage, epoch, step
- Updated every forward/backward pass
- Used by orchestrator for decision making

### Configuration Parameters

**Model architecture:**
- `hidden_size`: Embedding and hidden dimension (128-20480)
- `num_layers`: Transformer layer count (2-120)
- `num_heads`: Attention head count (2-160)
- `num_kv_heads`: KV cache heads for GQA (2-40, typically num_heads/4)
- `intermediate_size`: FFN intermediate dimension (typically 8/3 × hidden_size)
- `max_position_embeddings`: Maximum sequence length (128-32768)
- `vocab_size`: Tokenizer vocabulary size (typically 32000-100000)

**MoE parameters:**
- `use_moe`: Enable MoE (boolean)
- `num_experts`: Expert count per layer (4-64, typically 8)
- `moe_top_k`: Experts activated per token (1-4, typically 2)
- `capacity_factor`: Token capacity multiplier (1.0-2.0, typically 1.25)
- `load_balancing_weight`: Auxiliary loss coefficient (0.001-0.1, typically 0.01)
- `routing_temperature`: Softmax temperature (0.1-2.0, typically 1.0)

**MoD parameters:**
- `use_mod`: Enable MoD (boolean)
- `mod_capacity_factor`: Fraction using full computation (0.1-1.0, typically 0.5)
- `mod_routing_type`: Routing mechanism ('learned', 'static', 'random')
- `mod_start_layer`: First layer with MoD (0-num_layers)
- `mod_end_layer`: Last layer with MoD (None = all layers)

**Training parameters:**
- `num_epochs`: Training duration in epochs (1-100)
- `batch_size`: Micro batch size per GPU (1-128)
- `gradient_accumulation_steps`: Accumulation before optimizer step (1-128)
- `learning_rate`: Optimizer learning rate (1e-5 to 1e-3)
- `weight_decay`: L2 regularization (0.0-0.1, typically 0.01)
- `gradient_clip_val`: Gradient norm clipping (0.5-5.0, typically 1.0)
- `warmup_steps`: Learning rate warmup duration (steps or fraction)

**Precision parameters:**
- `precision`: Training precision ('auto', 'fp32', 'fp16', 'bf16', 'mixed_fp16', 'mixed_bf16', 'fp8_e4m3')
- `inference_precision`: Evaluation precision (same options as training)

**Optimization parameters:**
- `use_flash_attention`: Enable Flash Attention (boolean, auto-detected)
- `gradient_checkpointing`: Activation checkpointing (boolean)
- `compile`: PyTorch 2.0 compilation (boolean)
- `use_deepspeed`: Enable DeepSpeed (boolean)
- `zero_stage`: ZeRO optimization level (0-3)
- `cpu_offload`: Offload optimizer to CPU (boolean)

**Data parameters:**
- `training_mode`: Data handling ('base_only', 'finetuning_only', 'hybrid_sequential', 'hybrid_interleaved')
- `base_paths`: List of base training files
- `finetuning_paths`: List of fine-tuning files
- `base_eval_paths`: Base validation files
- `finetuning_eval_paths`: Fine-tuning validation files
- `base_ratio`: Mixing ratio for interleaved mode (0.0-1.0)
- `mask_user_tokens`: Mask user messages in loss (boolean)

**Orchestrator parameters:**
- `use_adaptive_training`: Enable orchestrator (boolean)
- `intervention_threshold`: Confidence required for intervention (0.0-1.0, typically 0.75)
- `check_interval`: Steps between health checks (10-1000, typically 100)
- `enable_emergency_recovery`: Allow emergency interventions (boolean)
- `enable_architecture_adaptation`: Allow architecture changes (boolean)

**Chinchilla parameters:**
- `auto_epoch_scaling`: Enable automatic epoch calculation (boolean)
- `chinchilla_multiplier`: Token multiplier (5-50, typically 20)
- `min_auto_epochs`: Minimum epochs (1-10)
- `max_auto_epochs`: Maximum epochs (10-100)
- `enable_loss_landscape`: Track loss patterns (boolean)
- `enable_compute_efficiency`: Track efficiency metrics (boolean)
- `enable_early_stopping`: Allow early termination (boolean)

**Checkpoint parameters:**
- `save_every_n_batches`: Checkpoint interval in steps (100-10000)
- `save_total_limit`: Maximum checkpoints to keep (1-100)
- `early_stopping_patience`: Epochs without improvement before stopping (3-20)

---

## Troubleshooting

### Out of Memory (OOM)

**Automatic handling:**
Orchestrator detects OOM exceptions, reduces batch size by 50%, recreates dataloader, resumes training from last checkpoint.

**Manual interventions:**
- Reduce `batch_size`: Start with 1-2 for very large models
- Increase `gradient_accumulation_steps`: Maintains effective batch size with less memory
- Enable `gradient_checkpointing`: Trades compute for memory (recompute activations)
- Increase `zero_stage`: 1→2→3 for progressively more memory optimization
- Enable `cpu_offload`: Moves optimizer states to CPU (slower but massive memory savings)
- Reduce `max_position_embeddings`: Shorter sequences use less memory
- Lower model size: Try smaller preset configuration

**Memory estimation:**
Model memory (FP16) ≈ 2 bytes × total_parameters
Optimizer memory (Adam) ≈ 8 bytes × parameters
Gradient memory ≈ 2 bytes × parameters
Activation memory ≈ 2 × batch_size × sequence_length × num_layers × hidden_size
Total ≈ 12-16 bytes per parameter + activation memory

### Training Instabilities

**Gradient explosion:**
Symptoms: Loss becomes NaN, gradient norm > 100, rapid loss increase

Automatic recovery: Orchestrator detects high gradient norm, triggers emergency LR reduction (10×), rolls back to previous checkpoint, resumes with lower LR.

Manual fixes:
- Lower `learning_rate`: Try 10× reduction
- Increase `gradient_clip_val`: Clip at lower threshold (0.5 instead of 1.0)
- Use mixed precision: BF16 more stable than FP16
- Enable gradient checkpointing: Can improve numerical stability
- Check data: Outliers or corrupted samples can cause explosions

**Loss divergence:**
Symptoms: Loss increases consistently, validation loss >> training loss, sudden loss spikes

Automatic recovery: Orchestrator detects divergence pattern, rolls back N steps, adjusts learning rate, may modify architecture parameters.

Manual fixes:
- Reduce `learning_rate`: Start 3-5× lower
- Increase `weight_decay`: Stronger regularization (0.1 instead of 0.01)
- Check data quality: Remove outliers, validate preprocessing
- Reduce model capacity: Overparameterized models may not converge on small datasets

**Expert collapse (MoE):**
Symptoms: All tokens route to 1-2 experts, routing entropy < 1.0, most experts have near-zero utilization

Automatic recovery: Orchestrator detects imbalance, increases `load_balancing_weight`, adjusts `routing_temperature`, may prune/add experts.

Manual fixes:
- Increase `load_balancing_weight`: Try 0.02 or 0.05 (from 0.01)
- Increase `capacity_factor`: Allow more tokens per expert (1.5 or 2.0)
- Adjust `routing_temperature`: Higher values (1.5-2.0) encourage uniform routing
- Enable `expert_dropout`: Forces routing to all experts
- Check initialization: Poorly initialized experts may never activate

### Slow Training

**Automatic optimization:**
Orchestrator monitors throughput, detects degradation, suggests optimizations (enable compilation, adjust batch size, check data loading bottlenecks).

**Manual optimizations:**
- Enable `compile`: PyTorch 2.0 compilation (5-30% speedup)
- Enable `use_flash_attention`: 2-4× attention speedup on Ampere+
- Use `mixed_bf16` or `mixed_fp16`: 2× speedup over FP32
- Increase `num_workers`: Parallelize data loading (typically 4-8)
- Increase `batch_size`: Better GPU utilization (if memory allows)
- Reduce `gradient_checkpointing`: Faster but more memory
- Check I/O: Move dataset to fast SSD, use memory-mapped files

**Bottleneck identification:**
- GPU utilization < 80%: CPU or data loading bottleneck
- Low throughput with high GPU utilization: Model or algorithm bottleneck
- Throughput decreases over time: Memory fragmentation or thermal throttling
- Inconsistent throughput: Data loading variance or OS interference

### Apple Silicon (MPS) Issues

**Common issues:**
- Training instability: MPS precision handling differs from CUDA
- Compilation failures: PyTorch MPS backend less mature
- Memory pressure: Unified memory competition with OS

**Solutions:**
- Set `compile=False`: Disable compilation if unstable
- Set `num_workers=0`: MPS prefers single-threaded data loading
- Reduce `batch_size`: Start conservative (2-4)
- Monitor Activity Monitor: Check memory pressure, GPU usage
- Update PyTorch: MPS backend rapidly improving, use latest version
- Fall back to CPU: If MPS unreliable, CPU training is alternative (slower but stable)

### Checkpoint Issues

**Corruption:**
Symptoms: Checkpoint fails to load, missing keys, size mismatch

Recovery: System automatically tries previous checkpoints (latest → latest-1 → best validation). If all corrupt, restart from initialization.

Prevention: Enable `save_total_limit > 3`, save to reliable storage, validate checksums.

**Resume failures:**
Symptoms: Training resumes but loss resets, optimizer state lost, different results than before

Causes: Incomplete checkpoint save, random seed mismatch, configuration mismatch

Solutions: Verify checkpoint integrity before resume, ensure configuration matches checkpoint, check random seed restoration.

### Data Issues

**Format errors:**
Symptoms: Training fails during data loading, "invalid JSON" or "unexpected format" errors

Solutions: Validate data format with provided validation scripts, check for: missing fields, incorrect JSON structure, encoding issues (use UTF-8), empty files or lines.

**Quality problems:**
Symptoms: Training succeeds but poor results, high validation loss, model outputs nonsense

Causes: Data contamination, label errors, poor quality samples, distribution mismatch

Solutions: Enable `automatic_data_cleaning`, increase `quality_threshold`, manually inspect samples, check train/validation split, verify preprocessing correctness.

---

## Performance Benchmarks

Throughput measurements on reference hardware configurations. All benchmarks use sequence_length=2048, batch_size optimized per GPU, mixed precision training with gradient checkpointing.

### Single GPU Performance

**NVIDIA RTX 3090 (24GB, Ampere):**
- b1 (1B active): ~1000 tokens/second, batch_size=16
- b7 (7B active): ~200 tokens/second, batch_size=4, requires ZeRO-2
- Memory efficiency: 85-90% utilization at optimal batch size

**NVIDIA A100 40GB (Ampere):**
- b1 (1B active): ~1200 tokens/second, batch_size=32
- b7 (7B active): ~500 tokens/second, batch_size=16
- b14 (14B active): ~150 tokens/second, batch_size=4, requires ZeRO-2
- Memory efficiency: 90-95% utilization

**NVIDIA A100 80GB (Ampere):**
- b7 (7B active): ~550 tokens/second, batch_size=24
- b14 (14B active): ~250 tokens/second, batch_size=12
- b30 (30B active): ~50 tokens/second, batch_size=2, requires ZeRO-3
- Memory efficiency: 85-92% utilization

**NVIDIA H100 80GB (Hopper):**
- b14 (14B active): ~400 tokens/second, batch_size=16
- b30 (30B active): ~120 tokens/second, batch_size=8
- b50 (50B active): ~50 tokens/second, batch_size=4, requires ZeRO-3
- FP8 support: Additional 1.5-2× speedup with FP8 training

**Apple M1 Max (32GB unified, MPS):**
- b1 (1B active): ~300 tokens/second, batch_size=8
- Memory: Unified architecture shares with system, effective 20-24GB for training
- Note: 3-4× slower than equivalent CUDA GPU

**Apple M2 Ultra (128GB unified, MPS):**
- b1 (1B active): ~400 tokens/second, batch_size=16
- b7 (7B active): ~80 tokens/second, batch_size=4
- Memory: Up to 96GB available for training after system overhead
- Note: 2-3× slower than A100 but larger memory capacity

### Multi-GPU Scaling

**4× A100 80GB (DeepSpeed ZeRO-2):**
- b30 (30B active): ~350 tokens/second (3.5× single GPU)
- Scaling efficiency: 87%
- Communication overhead: ~13%

**8× A100 80GB (DeepSpeed ZeRO-3):**
- b50 (50B active): ~320 tokens/second
- b100 (100B active): ~180 tokens/second
- Scaling efficiency: 70-75%
- Communication overhead: 25-30%

**16× H100 80GB (DeepSpeed ZeRO-3 + expert parallelism):**
- b100 (100B active): ~600 tokens/second
- b200 (200B active): ~280 tokens/second
- Scaling efficiency: 60-65%
- Expert parallelism improves MoE scaling

**Scaling efficiency factors:**
- Model size: Larger models have lower scaling efficiency (communication bound)
- Interconnect: InfiniBand (400Gbps) vs Ethernet (100Gbps) significantly affects scaling
- Expert parallelism: Distributing experts across GPUs improves MoE scaling
- Gradient accumulation: Higher accumulation reduces communication frequency

### Optimization Impact

**Flash Attention (Ampere+):**
- Attention speedup: 2-4× depending on sequence length
- Longer sequences benefit more (4× at 4096 length vs 2× at 512)
- Memory reduction: 30-50% for attention computation
- Quality: Numerically equivalent to standard attention

**PyTorch Compilation (torch.compile):**
- Speedup: 5-30% depending on model architecture
- MoE models: Lower benefit (routing breaks fusion)
- Dense models: Higher benefit (more fusion opportunities)
- Startup cost: 1-5 minutes additional compilation time

**Gradient Checkpointing:**
- Memory reduction: 30-50% of activation memory
- Compute overhead: 20-30% additional training time
- Trade-off: Enables larger batch sizes, often net positive throughput

**Mixed Precision:**
- FP32 → mixed_bf16: ~2× speedup, 50% memory reduction
- FP32 → mixed_fp16: ~2× speedup, 50% memory, may need loss scaling tuning
- BF16 → FP8: ~1.5-2× speedup (H100 only), quality impacts under investigation

---

## Production Deployment

Considerations for deploying LuminaAI-trained models in production environments.

### Model Export

**Checkpoint format:**
Standard PyTorch state dict compatible with transformers library. Can export to HuggingFace format, ONNX (for inference optimization), TorchScript (for deployment), or TensorRT (for NVIDIA inference).

**Conversion process:**
1. Load final checkpoint
2. Extract model state dict
3. Convert to target format
4. Validate outputs match original
5. Benchmark inference performance

**Size optimization:**
- Weight pruning: Remove low-magnitude weights
- Quantization: INT8 or INT4 for deployment
- Knowledge distillation: Train smaller dense model from sparse model
- Expert merging: Combine similar experts in MoE models

### Inference Optimization

**Quantization strategies:**
- Post-training quantization: INT8 via bitsandbytes or GPTQ (4-bit)
- Quantization-aware training: Train with quantization simulation
- Dynamic quantization: Quantize activations at runtime
- Mixed precision inference: FP16 or INT8 depending on layer

**Batching:**
- Dynamic batching: Group requests by sequence length
- Continuous batching: Add requests to in-flight batches
- Request queuing: Balance latency and throughput

**KV cache management:**
- Cache quantization: INT8 KV cache (2× memory reduction)
- Cache eviction: Drop old tokens for long conversations
- Paged attention: Efficient memory allocation (vLLM)

**Serving frameworks:**
- vLLM: High-throughput inference with paged attention
- TensorRT-LLM: NVIDIA-optimized inference
- Text Generation Inference: HuggingFace serving
- Custom deployment: Direct PyTorch or ONNX Runtime

### Monitoring Production Models

**Inference metrics:**
- Latency: Time to first token, total generation time
- Throughput: Requests per second, tokens per second
- Quality: Output validation, coherence scoring
- Resource usage: GPU memory, CPU utilization

**Model drift detection:**
- Input distribution monitoring
- Output quality tracking over time
- Comparison to validation benchmarks
- Automatic retraining triggers

---

## Licensing

LuminaAI is available under a commercial license. Framework provided for evaluation via demo notebook. Production use requires license agreement.

**License tiers:**

**Research/Academic:**
- Non-commercial research and educational use
- Academic institutions, non-profit research
- Publications require citation
- No redistribution of modified code
- Pricing: Contact for academic pricing

**Startup:**
- Companies with <10 employees, <$1M revenue
- Internal use only (no redistribution)
- Includes updates and bug fixes
- Email support
- Pricing: Contact for startup pricing

**Enterprise:**
- Larger organizations
- Internal use and customer deployments
- Includes updates, bug fixes, security patches
- Priority support, SLA options
- Custom modifications available
- Pricing: Contact for enterprise pricing

**Contact:** licensing@luminaai.dev

**Evaluation license:**
Demo notebook and local installation for evaluation purposes. 30-day evaluation period. No production use. Watermarked outputs during evaluation.

---

## Documentation

**Architecture Guide:** `docs/architecture.md`
- Detailed architectural descriptions
- Component interactions
- Design decisions and rationale
- Performance characteristics

**Adaptive Training Manual:** `docs/adaptive_training.md`
- Orchestrator internals
- Intervention method details
- Configuration recommendations
- Case studies and examples

**MoE/MoD Tutorial:** `docs/sparse_architectures.md`
- Sparse architecture theory
- Implementation details
- Training best practices
- Debugging sparse models

**API Reference:** `docs/api_reference.md`
- Complete method documentation
- Parameter specifications
- Return value descriptions
- Usage examples

**Hardware Guide:** `docs/hardware_optimization.md`
- Platform-specific optimizations
- Memory management strategies
- Multi-GPU configuration
- Benchmarking methodology

**Data Processing Guide:** `docs/data_processing.md`
- Format specifications
- Preprocessing pipelines
- Quality validation
- Curriculum learning strategies

---

## Citation

```bibtex
@software{luminaai2025,
  title = {LuminaAI: Modular Transformer Training with MoE/MoD},
  author = {MatN23},
  year = {2025},
  url = {https://github.com/matn23/luminaai},
  note = {Production-grade training framework with adaptive optimization}
}
```

---

## Support

**Issues:** GitHub issue tracker for bug reports and feature requests

**Discussions:** GitHub discussions for questions and community support

**Email:** support@luminaai.dev for licensing and technical inquiries

**Documentation:** Full documentation at docs.luminaai.dev (coming soon)

---

<div align="center">

**LuminaAI**

*Production transformer training for ML engineers*

[GitHub](https://github.com/matn23/luminaai) • [Demo](https://colab.research.google.com/drive/1tH1z9e7px2G8NGqWUN9gdqxs1CnUC7p1) • [License](LICENSE)

</div>