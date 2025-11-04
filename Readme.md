# LuminaAI

<div align="center">

**Next-Generation Transformer Training with AI-Driven Adaptive Intelligence**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Custom-green.svg)](LICENSE)

*A production-ready system featuring autonomous training optimization, real-time intelligence, and state-of-the-art architecture support*

[Features](#key-features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Advanced Features](#advanced-features)

</div>

---

## Overview

LuminaAI is a comprehensive deep learning training system that combines cutting-edge transformer architectures with **autonomous AI-driven training optimization**. At its core is the **Adaptive Training Orchestrator** - an intelligent system that continuously monitors, analyzes, and optimizes training in real-time, making hundreds of micro-decisions to ensure optimal convergence and resource utilization.

The system supports both traditional dense architectures and advanced sparse architectures including **Mixture-of-Experts (MoE)** and **Mixture-of-Depths (MoD)**, with full support for **hybrid architectures** that combine both techniques.

### Why LuminaAI?

- **Autonomous Intelligence**: Self-optimizing training with AI-driven decision making
- **Production-Ready**: Comprehensive error handling, checkpointing, and monitoring
- **Efficient**: Advanced optimizations with minimal memory overhead
- **Flexible**: Support for multiple training paradigms (base, fine-tuning, hybrid)
- **Observable**: Extensive metrics, profiling, and real-time health monitoring
- **Scalable**: From single GPU to multi-node distributed training
- **Advanced**: MoE, MoD, Flash Attention, GQA, and hybrid architectures
- **Chinchilla Scaling**: Automatic epoch optimization based on compute-optimal principles

---

## Key Features

### Adaptive Training Orchestrator

**The brain of LuminaAI** - an autonomous training intelligence system that revolutionizes the training process:

#### Real-Time Adaptive Decision Making
The orchestrator makes intelligent decisions during training:

- **Automatic Hyperparameter Tuning**: Adjusts learning rate, batch size, and regularization on-the-fly
- **Architecture Evolution**: Dynamically adds/removes MoE experts based on utilization
- **Emergency Recovery**: Detects and recovers from training instabilities automatically
- **Resource Optimization**: Balances compute, memory, and throughput in real-time
- **Checkpoint Rollback**: Time-travel to previous checkpoints when issues detected

#### Enhanced Chinchilla Scaler

**Automatic Epoch Optimization** - No more guessing optimal training duration:

- **Compute-Optimal Training**: Automatically calculates optimal epochs based on model size and dataset
- **Multi-Signal Convergence Detection**: 
  - Loss landscape analysis (plateau detection, divergence monitoring)
  - Compute efficiency tracking (loss reduction per FLOP)
  - Gradient variance monitoring
  - Convergence score calculation
- **Adaptive Early Stopping**: Stops training when:
  - Model converges (>85% convergence score)
  - Compute efficiency collapses (>30% decline)
  - Training diverges or plateaus for extended periods
- **Real-Time Recommendations**: Dynamic epoch adjustment based on training progress
- **Chinchilla Formula**: Follows N_opt ≈ 20 × P (20 tokens per parameter)

**Example Output:**
```
ENHANCED CHINCHILLA SCALER INITIALIZED
Model Parameters: 1.2B (1.200B)
Dataset Tokens: 24.5B (24.500B)
Chinchilla Optimal: 24.0B tokens
Base optimal epochs: 5
Token Budget Coverage: 102.1%

CHINCHILLA STATUS - Step 5000
Current epochs: 4 (adjusted from 5)
Token progress: 83.4%
Convergence: 87% (High)
Training phase: convergence
Compute efficiency: Stable
```

#### Advanced Analytics & Monitoring
- **Anomaly Detection**: Identifies loss spikes, gradient explosions, and expert imbalance
- **Health Scoring**: Continuous assessment of training health with predictive alerts
- **Performance Profiling**: Layer-by-layer analysis of bottlenecks
- **Convergence Analysis**: Real-time loss dynamics and plateau detection

#### 18 New Adaptive Methods

**MoE Architecture (3 methods):**
- `add_expert()` - Dynamically add experts mid-training
- `prune_expert()` - Remove underutilized experts
- `_initialize_new_expert()` - Smart knowledge distillation

**MoE Routing (4 methods):**
- `adjust_capacity_factor()` - Change token routing capacity
- `adjust_routing_temperature()` - Adjust routing concentration
- `enable_expert_dropout()` - Prevent expert collapse
- `get_expert_statistics()` - Comprehensive usage metrics

**MoD Routing (2 methods):**
- `adjust_mod_capacity()` - Change compute ratio
- `get_mod_statistics()` - Efficiency metrics

**Batch Size Adaptation (2 methods):**
- `adjust_batch_size()` - Dynamic batch size changes
- `_recreate_dataloader()` - Rebuild with new settings

**Orchestrator Communication (3 methods):**
- `get_current_metrics()` - Real-time training state
- `_extract_moe_routing_stats()` - Expert utilization
- `_calculate_throughput()` - Tokens/second measurement

**Emergency Recovery (2 methods):**
- `emergency_lr_reduction()` - 10x LR cut for gradient explosion
- `rollback_steps()` - Checkpoint-based time travel

**Advanced Optimizer (2 methods):**
- `adjust_weight_decay()` - Dynamic regularization
- `_update_optimizer_param_groups()` - Live optimizer updates

---

### Architecture Capabilities

#### Mixture of Experts (MoE)
- Sparse activation for massive parameter efficiency
- 8-64 expert configurations with top-k routing
- Advanced load balancing to prevent expert collapse
- 40-60% less active parameters for same capacity
- Flexible MoE patterns: all layers, interleaved, sandwich
- **Adaptive expert management**: Add/remove experts during training

#### Mixture of Depths (MoD)
- Dynamic token-level compute allocation
- Learn which tokens need full computation vs skip connections
- 30-50% FLOPs reduction with minimal quality loss
- Adaptive capacity factors for different use cases
- **Real-time capacity adjustment** based on training phase

#### Hybrid MoE + MoD
- **Combine MoE and MoD in the same model**
- MoE layers for expert routing on complex layers
- MoD routing on dense layers for token efficiency
- Maximum parameter AND compute efficiency

#### Dense Models with Advanced Features
- Grouped Query Attention (GQA) for efficient KV caching
- Rotary Position Embeddings (RoPE) for better length generalization
- SwiGLU activation for improved performance
- RMS Normalization for training stability

---

### Training & Optimization

#### Multi-Device Support
- **NVIDIA CUDA**: Full feature support with DeepSpeed integration
- **Apple Silicon (MPS)**: Native M1/M2/M3/M4 support with automatic optimizations
- **CPU**: Fallback for development and debugging

#### Comprehensive Precision Support

**Floating Point Precisions:**
- `fp32` - Standard training, maximum stability
- `fp16` - Mixed precision for memory efficiency (CUDA)
- `bf16` - Best for modern GPUs (Ampere+), FP32 range with FP16 memory
- `tf32` - Automatic speedup on Ampere+ GPUs
- `fp8_e4m3` / `fp8_e5m2` - Cutting-edge H100+ GPUs (experimental)

**Integer Precisions:**
- `int8` - Quantized inference with BitsAndBytes
- `int4` - Maximum memory savings with GPTQ/AWQ
- `int2` - Experimental ultra-low memory

**Mixed Precision Modes:**
- `mixed_fp16` - FP16 compute with FP32 accumulation
- `mixed_bf16` - BF16 compute with FP32 accumulation
- `mixed_fp8` - FP8 compute for H100+ (experimental)

#### Quantization Support

**Methods Available:**
- **BitsAndBytes (bnb)**: 4-bit and 8-bit quantization
- **AutoGPTQ**: Advanced 4-bit quantization
- **Optimum Quanto**: Flexible quantization framework

**Configuration:**
```python
# 8-bit quantization
config.quantization_method = 'bnb'
config.quantization_bits = 8

# 4-bit quantization
config.quantization_method = 'bnb'
config.quantization_bits = 4
```

#### DeepSpeed Integration
- ZeRO optimization stages 1-3 for memory efficiency
- CPU/NVMe offloading for training massive models
- Gradient compression for multi-GPU communication
- Automatic mixed precision training (FP16/BF16)
- **Adaptive ZeRO stage selection** based on available memory
- **Optimized MoE configuration** with expert parallelism

#### Advanced Training Features
- **Adaptive gradient checkpointing** for optimal memory/speed tradeoff
- **Dynamic learning rate scheduling** with anomaly-aware adjustments
- **Automatic batch size tuning** with OOM recovery
- **Smart early stopping** with patience configuration
- **Continuous checkpointing** with best model tracking and rollback support

---

### Data Processing

#### Multi-Dataset Support
- **Base/Pre-training**: Raw text (The Pile, C4, etc.) - supports `.txt` and `.jsonl`
- **Fine-tuning**: Conversational data (OASST, custom) - `.jsonl` only
- **Multi-file support**: Combine multiple datasets automatically
- **Streaming support**: For datasets >10GB

#### Training Modes
- **Base Only**: Pre-training on raw text
- **Fine-tuning Only**: Instruction tuning on conversations
- **Hybrid**: Sequential (base → fine-tuning)
- **Interleaved**: Mixed base and fine-tuning with adaptive ratios

#### Smart Data Loading
- Automatic format detection (JSONL, TXT)
- Chunking for efficient base training
- Conversation validation and preprocessing
- Configurable data caching
- Dynamic batching and prefetching
- **Adaptive data sampling** based on training phase

---

## Quick Start

### Prerequisites

```bash
# System Requirements
Python 3.8+
PyTorch 2.0+
CUDA 11.7+ (for GPU) or MPS (Apple Silicon)

# Storage
- 10GB+ free disk space for checkpoints
- 50GB+ recommended for large models
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/luminaai.git
cd luminaai

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Optional: Install DeepSpeed (not supported on MPS)
pip install deepspeed

# Optional: Install Flash Attention
pip install flash-attn --no-build-isolation
```

### Basic Training Example

```python
# Main.py - Basic configuration

# 1. Choose model size
config_choice = 'b1'  # 1B active parameters

# 2. Configure adaptive training
use_adaptive_training = True  # Enable AI-driven optimization

# 3. Training parameters
training_params = {
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'precision': 'auto',  # Auto-detects best precision for hardware
}

# 4. Specify data
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': [
        'data/train.jsonl',
    ],
    'finetuning_eval_paths': [
        'data/eval.jsonl',
    ],
}

# Run training - The Orchestrator handles the rest
# python Main.py
```

### Hardware-Specific Setup

#### NVIDIA GPU

```python
# Optimal configuration for GPU
training_params = {
    'precision': 'mixed_bf16',  # BF16 for Ampere+, FP16 for older
    'use_flash_attention': True,
    'use_deepspeed': True,
    'compile': True,
}
```

#### Apple Silicon (M1/M2/M3/M4)

```python
# MPS-optimized configuration (auto-applied)
training_params = {
    'precision': 'fp16',
    'batch_size': 4,
    'use_flash_attention': False,  # Not supported on MPS
    'use_deepspeed': False,  # Not supported on MPS
}
# System automatically applies these optimizations
```

---

## Model Configurations

### Available Presets

LuminaAI includes pre-configured model sizes optimized for different hardware and use cases. Each configuration is designed with the Adaptive Orchestrator in mind:

| Preset | Active Params | Total Params | Architecture | Use Case | Hardware | Adaptive Recommendations |
|--------|--------------|--------------|--------------|----------|----------|--------------------------|
| `debug` | ~500K | ~4M | 8x MoE | Development & Testing | Any | Minimal adaptive features, fast iteration |
| `debug_200m` | ~200M | ~6B | 32x MoD | Architecture Testing | T4/MPS/CPU | Test adaptive routing, MoD optimization |
| `b1` | ~1B | ~8B | 8x MoE | Small-Scale Experiments | RTX 3090, M1 Max, A10 | Full adaptive features, rapid convergence |
| `b7` | ~7B | ~56B | 8x MoE | General Purpose | A100 40GB, M2 Ultra, RTX 4090 | Meta-learning enabled, expert management |
| `b14` | ~14B | ~112B | 8x MoE | High Performance | A100 80GB, Multi-A10 | Advanced recovery, memory optimization |
| `b30` | ~30B | ~240B | 8x MoE | Advanced Research | Multi-A100, H100 | Full orchestrator suite, multi-node adaptive |
| `b50` | ~50B | ~400B | 8x MoE | Enterprise Scale | Multi-H100 | Distributed adaptive intelligence |
| `b100` | ~100B | ~800B | 8x MoE | Cutting Edge | Large H100 Cluster | Cluster-wide optimization |
| `b200` | ~200B | ~1.6T | 8x MoE | Frontier Research | Massive H100 Cluster | Advanced meta-learning across nodes |
| `b300` | ~300B | ~2.4T | 8x MoE | Ultra-Scale | Ultra-Large Cluster | Global adaptive optimization |

### Configuration Examples

#### Small Efficient Model (1B) with Adaptive Training

```python
config_choice = 'b1'

# Enable adaptive orchestrator for autonomous optimization
use_adaptive_training = True

training_params = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'use_moe': True,
    'use_mod': False,  # Pure MoE
}

# The orchestrator will automatically:
# - Adjust learning rate based on loss curves
# - Monitor and balance expert utilization
# - Detect and recover from anomalies
# - Optimize batch size for your hardware
```

#### Medium Model with Hybrid Architecture (7B)

```python
config_choice = 'b7'

use_adaptive_training = True  # Strongly recommended for complex architectures

training_params = {
    'batch_size': 16,
    'gradient_accumulation_steps': 8,
    'use_moe': True,
    'use_mod': True,  # HYBRID MODE
    'num_experts': 8,
    'moe_top_k': 2,
    'mod_capacity_factor': 0.5,
}

# Adaptive features for hybrid models:
# - Dynamic expert addition/removal
# - MoD capacity adjustment
# - Routing temperature optimization
# - Cross-architecture balancing
```

#### Large Model with Chinchilla Scaling (14B)

```python
config_choice = 'b14'

use_adaptive_training = True  # Essential for stable large-scale training

training_params = {
    'use_deepspeed': True,
    'zero_stage': 3,
    'cpu_offload': True,
    'gradient_checkpointing': True,
    'precision': 'mixed_bf16',
}

# Chinchilla scaler parameters
chinchilla_params = {
    'auto_epoch_scaling': True,       # Enable automatic epoch optimization
    'chinchilla_multiplier': 20,      # Tokens per parameter
    'min_auto_epochs': 1,             # Safety minimum
    'max_auto_epochs': 50,            # Safety maximum
    'enable_loss_landscape': True,    # Loss plateau/divergence detection
    'enable_compute_efficiency': True,# Track loss reduction per FLOP
    'enable_early_stopping': True,    # Stop when converged
}

# The orchestrator provides:
# - Automatic OOM recovery with batch size reduction
# - Gradient explosion detection and emergency LR cuts
# - Memory pressure monitoring with preemptive actions
# - Convergence prediction and adaptive early stopping
# - Optimal training duration based on model size
```

### Custom Configuration

```python
from config.config_manager import Config

config = Config(
    # Model architecture
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,  # GQA for efficient KV cache
    seq_length=2048,
    
    # MoE settings
    use_moe=True,
    num_experts=8,
    moe_top_k=2,
    capacity_factor=1.25,
    
    # MoD settings (optional)
    use_mod=True,
    mod_capacity_factor=0.5,
    
    # Training
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_epochs=3,
)

# Enable adaptive orchestrator
use_adaptive_training = True
```

---

## Advanced Features

### Chinchilla Scaler Configuration

The Enhanced Chinchilla Scaler automatically determines optimal training duration:

```python
chinchilla_params = {
    # Core settings
    'auto_epoch_scaling': True,           # Enable auto-scaling
    'chinchilla_multiplier': 20,          # Tokens per parameter (standard: 20)
    'min_auto_epochs': 1,                 # Minimum epochs
    'max_auto_epochs': 50,                # Maximum epochs
    
    # Advanced features
    'enable_loss_landscape': True,        # Plateau/divergence detection
    'enable_compute_efficiency': True,    # Track efficiency metrics
    'enable_adaptive_curriculum': True,   # Dynamic difficulty adjustment
    'enable_early_stopping': True,        # Stop when converged
    
    # Thresholds
    'plateau_patience': 5,                # Epochs before plateau detection
    'efficiency_decline_threshold': 0.3,  # Flag efficiency drops
    'convergence_threshold': 0.85,        # Consider converged at 85%
}
```

**How it works:**
1. Calculates optimal tokens: `N_opt = 20 × model_parameters`
2. Determines base epochs: `epochs = N_opt / dataset_tokens`
3. Monitors training dynamics in real-time
4. Adjusts epochs based on:
   - Convergence score
   - Compute efficiency trends
   - Loss landscape analysis
5. Stops early when appropriate

### Adaptive Training Orchestrator

The orchestrator provides AI-driven training optimization:

```python
# Enable in Main.py
use_adaptive_training = True  # Recommended for all production training

# What the orchestrator does automatically:
# - Meta-learning from previous runs
# - Real-time performance monitoring
# - Automatic hyperparameter adjustments
# - Anomaly detection and recovery
# - Expert utilization optimization
# - Memory pressure management
# - Convergence prediction
# - Emergency intervention
```

#### Key Adaptive Features

**1. Meta-Learning**
```python
# The orchestrator learns from your training history
# and applies successful strategies automatically:
#
# - Optimal learning rate schedules
# - Best batch size configurations
# - Successful regularization patterns
# - Convergence acceleration techniques
```

**2. Real-Time Decision Making**
```python
# During training, the orchestrator makes intelligent decisions:
#
# IF loss plateaus -> Increase LR or add expert
# IF gradients explode -> Emergency LR reduction
# IF expert imbalance -> Adjust routing parameters
# IF memory pressure -> Reduce batch size
# IF convergence predicted -> Optimize for final phase
```

**3. Advanced Recovery**
```python
# Automatic recovery from common training issues:
#
# - OOM errors: Automatic batch size reduction
# - Gradient explosion: Emergency LR cuts
# - Loss spikes: Checkpoint rollback
# - Expert collapse: Dynamic expert management
# - Convergence stalls: Adaptive interventions
```

### 18 New Adaptive Methods

The enhanced trainer provides fine-grained control:

```python
# MoE Architecture (3 methods)
trainer.add_expert(layer_idx=5)           # Add expert mid-training
trainer.prune_expert(layer_idx=0, expert_idx=3)  # Remove underutilized expert
trainer._initialize_new_expert(new_expert, existing_experts)  # Smart initialization

# MoE Routing (4 methods)
trainer.adjust_capacity_factor(2.0)        # Change token capacity
trainer.adjust_routing_temperature(1.5)    # Adjust routing concentration
trainer.enable_expert_dropout(0.1)         # Prevent expert collapse
stats = trainer.get_expert_statistics()    # Comprehensive metrics

# MoD Routing (2 methods)
trainer.adjust_mod_capacity(0.7)           # Change compute ratio
stats = trainer.get_mod_statistics()       # Efficiency metrics

# Batch Size Adaptation (2 methods)
trainer.adjust_batch_size(4)               # Dynamic batch size
trainer._recreate_dataloader(dataset)      # Rebuild dataloader

# Orchestrator Communication (3 methods) - CRITICAL
metrics = trainer.get_current_metrics()    # Real-time training state
routing = trainer._extract_moe_routing_stats()  # Expert utilization
throughput = trainer._calculate_throughput()    # Tokens/second

# Emergency Recovery (2 methods)
trainer.emergency_lr_reduction(10.0)       # 10x LR cut
trainer.rollback_steps(100)                # Time-travel to previous state

# Advanced Optimizer (2 methods)
trainer.adjust_weight_decay(0.05)          # Dynamic regularization
trainer._update_optimizer_param_groups('lr', 1e-5)  # Live updates
```

### Quantization Support

Train with reduced precision for memory efficiency:

```python
quantization_params = {
    'quantization_method': 'bnb',  # BitsAndBytes
    'quantization_bits': 8,  # 4 or 8 bit
}

# INT8 inference precision for faster evaluation
training_params = {
    'precision': 'mixed_bf16',  # Training
    'inference_precision': 'int8',  # Inference
}
```

### Gradient Checkpointing

Trade computation for memory:

```python
training_params = {
    'gradient_checkpointing': True,
    # Reduces memory by ~40% at ~20% speed cost
}
```

### Multi-Node Training

Scale across multiple machines:

```bash
# Node 0 (master)
deepspeed --num_nodes=4 --num_gpus=8 Main.py

# Nodes 1-3 (workers)
deepspeed --num_nodes=4 --num_gpus=8 --node_rank=N Main.py
```

---

## Monitoring & Debugging

### Training Metrics

The system tracks comprehensive metrics with adaptive insights:

- **Loss & Perplexity**: Training and validation curves with trend analysis
- **Accuracy**: Token-level prediction accuracy
- **Throughput**: Tokens/second processing speed
- **Memory**: GPU/MPS/CPU utilization
- **Learning Rate**: Current LR with scheduler info and adaptive adjustments
- **Gradient Norms**: Detect training instability
- **Expert Utilization**: Real-time MoE routing statistics
- **Adaptive Decisions**: Log of all orchestrator interventions
- **Chinchilla Metrics**: Token progress, convergence score, compute efficiency

### Health Monitoring

Automatic health checks detect:

- Loss spikes or divergence
- Gradient explosion/vanishing
- Expert imbalance (MoE)
- Token routing issues (MoD)
- Memory pressure
- Training stagnation
- **Convergence predictions**
- **Performance anomalies**
- **Compute efficiency decline**

### Adaptive Insights

The orchestrator provides detailed reports:

```python
# Get adaptive status
status = orchestrator.get_adaptive_status()

print(f"Adaptive decisions made: {status['adaptive_decisions_made']}")
print(f"Meta-learning runs: {status['meta_learning_runs']}")
print(f"Monitoring active: {status['monitoring_active']}")

# Recent adaptive decisions with reasoning
for decision in status['recent_decisions']:
    print(f"{decision['type']}: {decision['reasoning']}")
    print(f"  Confidence: {decision['confidence']:.2%}")
```

### Chinchilla Status

```python
# Get Chinchilla scaler status
if hasattr(trainer, 'chinchilla_scaler'):
    status = trainer.chinchilla_scaler.get_status_report()
    
    print(f"Token progress: {status['chinchilla']['progress']:.1f}%")
    print(f"Convergence score: {status['training']['convergence_score']:.2%}")
    print(f"Training phase: {status['training']['training_phase']}")
    print(f"Current optimal epochs: {status['chinchilla']['current_optimal_epochs']}")
```

### Logging Levels

```python
monitoring_params = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_every_n_steps': 50,
    'health_check_interval': 100,
}
```

### Checkpointing

Automatic checkpoint management with rollback support:

```python
training_params = {
    'save_every_n_batches': 1000,
    'save_total_limit': 5,  # Keep only N best checkpoints
    'early_stopping_patience': 10,
}

# Resume from checkpoint
checkpoint_params = {
    'resume_from_checkpoint': 'experiments/my_exp/checkpoints/checkpoint_epoch_5.pt',
    'resume_training': True,
}
```

---

## Troubleshooting

### Out of Memory (OOM)

The **Adaptive Orchestrator automatically handles OOM**:

1. **Automatic Detection**: Catches OOM errors during training
2. **Smart Recovery**: Reduces batch size while maintaining effective batch size via gradient accumulation
3. **Adaptive Retry**: Recreates training components and continues
4. **Configuration Saving**: Saves optimal settings for future runs

Manual intervention (if needed):
```python
training_params = {
    'batch_size': 2,  # Reduce
    'gradient_accumulation_steps': 16,  # Increase
    'gradient_checkpointing': True,
    'cpu_offload': True,  # If using DeepSpeed
}
```

### Slow Training

The orchestrator monitors and optimizes throughput automatically, but you can also:

- Enable `compile=True` (PyTorch 2.0+)
- Use `precision='mixed_bf16'` or `'mixed_fp16'`
- Enable Flash Attention (`use_flash_attention=True`)
- Increase `num_workers` for data loading
- Check `batch_size` isn't too small
- Enable DeepSpeed for multi-GPU

The orchestrator will suggest optimizations if it detects performance issues.

### Training Instabilities

The **Adaptive Orchestrator handles most instabilities automatically**:

**Gradient Explosion:**
```python
# Automatic emergency LR reduction when grad_norm > 100
# The orchestrator detects and responds immediately
```

**Loss Spikes:**
```python
# Automatic rollback to previous checkpoint
# The orchestrator maintains checkpoint history for recovery
```

**Expert Imbalance (MoE):**
```python
# Automatic routing adjustments
# The orchestrator monitors and balances expert utilization
```

Manual tuning (if needed):
```python
training_params = {
    'load_balancing_weight': 0.02,  # Increase (from 0.01)
    'routing_temperature': 1.2,  # Increase exploration
    'capacity_factor': 1.5,  # Allow more tokens per expert
}
```

### MPS (Apple Silicon) Issues

The system auto-optimizes for MPS, but manual tips:

```python
# If encountering issues
training_params = {
    'compile': False,  # Can be unstable on MPS
    'num_workers': 0,  # MPS prefers main thread loading
    'batch_size': 2,  # Start small
}
```

---

## Example: Adaptive Training in Action

Here's what happens during a typical adaptive training run:

```
[STEP 100] Normal training...
  Loss: 2.456 | PPL: 11.65 | Acc: 45.2%

[ORCHESTRATOR] Detected loss plateau
  Decision: Increase learning rate by 1.5x
  Confidence: 75%
  Reasoning: Loss variance < 0.001 for 50 steps

[STEP 200] Improved convergence...
  Loss: 2.234 | PPL: 9.34 | Acc: 48.7%

[ORCHESTRATOR] Expert imbalance detected
  Decision: Adjust capacity factor 1.25 -> 1.75
  Confidence: 82%
  Reasoning: Expert 3 utilization at 92%, Expert 5 at 8%

[STEP 300] Gradient spike detected...
  Loss: 2.189 | GradNorm: 156.2

[ORCHESTRATOR] Emergency intervention
  Decision: Emergency LR reduction by 10x
  Confidence: 95%
  Reasoning: Gradient norm > 100 (explosion detected)

[STEP 350] Training stabilized...
  Loss: 2.156 | PPL: 8.63 | Acc: 52.1%

[CHINCHILLA] Convergence predicted in ~200 steps
  Convergence score: 88%
  Expected final loss: 2.05 ± 0.15
  Recommend continuing training
```

---

## Citation

If you use LuminaAI in your research, please cite:

```bibtex
@software{luminaai2025,
  title = {LuminaAI: Advanced Transformer Training with Adaptive Intelligence},
  author = {MatN23},
  year = {2025},
  url = {https://github.com/matn23/luminaai},
  note = {AI-driven training optimization with MoE, MoD, and Chinchilla scaling}
}
```

---

## License

This project is licensed under a Custom License. See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- DeepSpeed team for distributed training framework
- Flash Attention authors for efficient attention implementation
- tiktoken for fast tokenization
- PyTorch team for the foundation
- OpenAssistant for conversational datasets
- Chinchilla paper authors for compute-optimal scaling insights
- The open-source AI community for continuous inspiration

---

## Additional Resources

- **Documentation**: [docs/](docs/)
- **Adaptive Training Guide**: [docs/adaptive_training.md](docs/adaptive_training.md)
- **MoE/MoD Tutorial**: [docs/sparse_architectures.md](docs/sparse_architectures.md)

---

<div align="center">

**Built with care for the AI research community**

*Featuring autonomous training intelligence that learns, adapts, and optimizes*

</div>