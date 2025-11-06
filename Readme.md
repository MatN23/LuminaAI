# LuminaAI

<div align="center">

**Production-Ready Transformer Training with Autonomous Optimization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Custom-green.svg)](LICENSE)

*Autonomous training system with MoE/MoD architectures and intelligent optimization*

[Quick Start](#quick-start) • [Architecture](#architecture) • [Configuration](#configuration) • [Adaptive Training](#adaptive-training)

</div>

<p align="center">
  <img src="assets/ReadmeLogo.png" width="400">
</p>


---

## What is LuminaAI?

LuminaAI is a production-ready deep learning training framework that combines modern transformer architectures with autonomous optimization. Think of it as your training co-pilot that makes intelligent decisions in real-time.

**Core Philosophy:**
- **Autonomous**: The system optimizes itself during training (learning rates, batch sizes, architecture)
- **Production-Ready**: Comprehensive error handling, checkpointing, and recovery mechanisms
- **Flexible**: Supports dense models, MoE, MoD, and hybrid architectures
- **Observable**: Extensive metrics and real-time monitoring

**Key Capabilities:**
- **Adaptive Training Orchestrator**: AI-driven training optimization that adjusts hyperparameters on-the-fly
- **Chinchilla Scaling**: Automatic epoch calculation based on compute-optimal principles
- **Sparse Architectures**: MoE (Mixture of Experts) and MoD (Mixture of Depths) with dynamic management
- **Emergency Recovery**: Automatic detection and recovery from training instabilities
- **Multi-Device**: Full support for CUDA, Apple Silicon (MPS), and CPU

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/matn23/luminaai.git
cd luminaai

# Install PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Optional: DeepSpeed for distributed training
pip install deepspeed

# Optional: Flash Attention for efficiency
pip install flash-attn --no-build-isolation
```

### Minimal Training Example

```python
# Main.py - Basic fine-tuning setup

# 1. Select model size
config_choice = 'b1'  # 1B active parameters (8B total with MoE)

# 2. Enable adaptive training (recommended)
use_adaptive_training = True

# 3. Configure training
training_params = {
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'precision': 'auto',  # Automatically selects best precision
}

# 4. Specify dataset
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': ['data/train.jsonl'],
    'finetuning_eval_paths': ['data/eval.jsonl'],
}

# Run: python Main.py
# The orchestrator handles optimization automatically
```

**What happens automatically:**
- Detects your hardware (GPU/MPS/CPU) and optimizes accordingly
- Monitors training health (loss spikes, gradient explosions)
- Adjusts learning rate based on convergence patterns
- Manages expert utilization for MoE models
- Recovers from OOM errors by reducing batch size
- Calculates optimal training duration via Chinchilla scaling

---

## Architecture

### Supported Model Types

#### 1. Dense Transformers (Standard)
Traditional transformer architecture with modern optimizations:
- **Grouped Query Attention (GQA)**: Reduces KV cache memory
- **Rotary Position Embeddings (RoPE)**: Better length generalization
- **SwiGLU Activation**: Improved performance over standard FFN
- **RMSNorm**: Training stability

```python
# Dense model configuration
config = Config(
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,  # GQA: 4 KV heads for 16 Q heads
    use_moe=False,
    use_mod=False,
)
```

#### 2. Mixture of Experts (MoE)
Sparse activation for parameter efficiency:
- **8-64 experts** per layer with top-k routing
- **40-60% fewer active parameters** while maintaining capacity
- **Load balancing** to prevent expert collapse
- **Dynamic expert management**: Add/remove experts during training

```python
# MoE configuration
config = Config(
    use_moe=True,
    num_experts=8,
    moe_top_k=2,  # Activate top-2 experts per token
    capacity_factor=1.25,  # Token routing capacity
    load_balancing_weight=0.01,  # Balance expert usage
)
```

**How MoE Works:**
1. Each token is routed to top-k experts (typically k=2)
2. Only activated experts process the token (sparse computation)
3. Load balancing ensures all experts are utilized
4. Result: 8B total parameters, ~1B active per token

#### 3. Mixture of Depths (MoD)
Dynamic compute allocation per token:
- **30-50% FLOPs reduction** with minimal quality loss
- **Learn which tokens need full computation** vs skip connections
- **Adaptive capacity**: Adjust compute ratio during training

```python
# MoD configuration
config = Config(
    use_mod=True,
    mod_capacity_factor=0.5,  # 50% of tokens get full computation
)
```

**How MoD Works:**
1. Each layer learns to route tokens through full computation or skip
2. Important tokens (rare words, complex syntax) get full processing
3. Simple tokens (common words, filler) use residual skip
4. Result: Faster training and inference with maintained quality

#### 4. Hybrid (MoE + MoD)
Combine both for maximum efficiency:
- **MoE on complex layers** for expert routing
- **MoD on dense layers** for token efficiency
- **Parameter AND compute efficiency** simultaneously

```python
# Hybrid configuration
config = Config(
    use_moe=True,
    num_experts=8,
    moe_top_k=2,
    use_mod=True,
    mod_capacity_factor=0.5,
)
```

### Pre-configured Model Sizes

| Preset | Active Params | Total Params | Architecture | Recommended Hardware | Training Speed |
|--------|--------------|--------------|--------------|---------------------|----------------|
| `debug` | ~500K | ~4M | 8x MoE | Any | Fast (testing) |
| `debug_200m` | ~200M | ~6B | 32x MoD | T4/MPS/CPU | Fast (arch testing) |
| `b1` | ~1B | ~8B | 8x MoE | RTX 3090, M1 Max | ~1000 tok/s |
| `b7` | ~7B | ~56B | 8x MoE | A100 40GB, M2 Ultra | ~500 tok/s |
| `b14` | ~14B | ~112B | 8x MoE | A100 80GB | ~250 tok/s |
| `b30` | ~30B | ~240B | 8x MoE | Multi-A100 | ~100 tok/s |
| `b50` | ~50B | ~400B | 8x MoE | Multi-H100 | ~50 tok/s |
| `b100` | ~100B | ~800B | 8x MoE | Large-H100-Server | ~50 tok/s |
| `b200` | ~200B | ~1600B | 8x MoE | OPENAI-GRADE-h200-Server | ~30 tok/s |
| `b300` | ~300B | ~2400B | 8x MoE | OPENAI-GRADE-h200-Server | ~20 tok/s |

**Note:** Throughput is approximate and depends on sequence length, batch size, and hardware.

---

## Configuration

### Training Modes

```python
# 1. Base/Pre-training only (raw text)
data_params = {
    'training_mode': 'base_only',
    'base_paths': ['data/raw_text.txt'],
    'base_eval_paths': ['data/eval_text.txt'],
}

# 2. Fine-tuning only (conversations)
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': ['data/conversations.jsonl'],
}

# 3. Hybrid (base then fine-tuning)
data_params = {
    'training_mode': 'hybrid_sequential',
    'base_paths': ['data/raw_text.txt'],
    'finetuning_paths': ['data/conversations.jsonl'],
}

# 4. Interleaved (mixed batches)
data_params = {
    'training_mode': 'hybrid_interleaved',
    'base_paths': ['data/raw_text.txt'],
    'finetuning_paths': ['data/conversations.jsonl'],
    'base_ratio': 0.7,  # 70% base, 30% fine-tuning
}
```

### Precision Options

```python
training_params = {
    # Automatic (recommended)
    'precision': 'auto',  # Selects best for hardware
    
    # Manual options
    'precision': 'fp32',       # Full precision (slow, stable)
    'precision': 'fp16',       # Half precision (2x faster, less stable)
    'precision': 'bf16',       # Brain float (best for Ampere+)
    'precision': 'mixed_fp16', # Mixed precision FP16
    'precision': 'mixed_bf16', # Mixed precision BF16 (recommended)
    
    # Advanced (experimental)
    'precision': 'fp8_e4m3',   # H100+ only
    'precision': 'int8',       # Quantized (inference)
}
```

**Precision Guide:**
- **CUDA (Ampere+)**: Use `mixed_bf16` (A100, RTX 3090+)
- **CUDA (Older)**: Use `mixed_fp16` (V100, RTX 2080)
- **Apple Silicon**: Use `fp16` (auto-detected)
- **CPU**: Use `fp32`

### Hardware-Specific Optimization

#### NVIDIA GPU (CUDA)

```python
training_params = {
    'precision': 'mixed_bf16',
    'use_flash_attention': True,    # 2-4x faster attention
    'use_deepspeed': True,          # Multi-GPU optimization
    'zero_stage': 2,                # Memory optimization
    'compile': True,                # PyTorch 2.0 compilation
}
```

#### Apple Silicon (M1/M2/M3/M4)

```python
# Auto-applied optimizations (no configuration needed)
# System detects MPS and applies:
# - precision='fp16'
# - use_flash_attention=False
# - use_deepspeed=False
# - num_workers=0

# Override if needed
training_params = {
    'batch_size': 4,  # Start conservative
    'compile': False,  # Can be unstable
}
```

#### Multi-GPU Training

```bash
# DeepSpeed (recommended)
deepspeed --num_gpus=4 Main.py

# PyTorch DDP
torchrun --nproc_per_node=4 Main.py
```

### Quantization

```python
quantization_params = {
    'quantization_method': 'bnb',  # BitsAndBytes
    'quantization_bits': 8,        # 4 or 8
}

# Separate training/inference precision
training_params = {
    'precision': 'mixed_bf16',      # Training
    'inference_precision': 'int8',  # Evaluation
}
```

---

## Adaptive Training

### What is the Adaptive Orchestrator?

The orchestrator is an autonomous system that monitors and optimizes training in real-time. It makes hundreds of micro-decisions to ensure optimal convergence.

**Key Capabilities:**

1. **Automatic Hyperparameter Tuning**
   - Learning rate adjustments based on loss curves
   - Batch size optimization with OOM recovery
   - Weight decay adaptation for regularization

2. **Architecture Management**
   - Add/remove MoE experts based on utilization
   - Adjust MoD capacity factors
   - Balance routing parameters

3. **Emergency Recovery**
   - Detect gradient explosions → emergency LR reduction
   - Detect loss spikes → rollback to previous checkpoint
   - Detect OOM → reduce batch size and retry

4. **Performance Monitoring**
   - Track loss dynamics (plateau, divergence)
   - Monitor expert utilization (MoE)
   - Analyze compute efficiency (MoD)
   - Predict convergence

### Enabling Adaptive Training

```python
# In Main.py
use_adaptive_training = True  # Strongly recommended

# The orchestrator automatically:
# - Monitors training health every 100 steps
# - Makes adaptive decisions when needed
# - Logs all decisions with reasoning
# - Saves optimal configurations for future runs
```

### Adaptive Methods (18 New APIs)

The trainer exposes 18 methods for fine-grained control:

#### MoE Architecture (3 methods)

```python
# Add expert to underperforming layer
trainer.add_expert(layer_idx=5)

# Remove underutilized expert
trainer.prune_expert(layer_idx=0, expert_idx=3)

# Initialize new expert (internal use)
trainer._initialize_new_expert(new_expert, existing_experts)
```

#### MoE Routing (4 methods)

```python
# Adjust token routing capacity
trainer.adjust_capacity_factor(2.0)  # Allow more tokens per expert

# Adjust routing concentration
trainer.adjust_routing_temperature(1.5)  # Higher = more exploration

# Enable expert dropout
trainer.enable_expert_dropout(0.1)  # Prevent expert collapse

# Get comprehensive statistics
stats = trainer.get_expert_statistics()
# Returns: {
#   'per_expert_utilization': [...],
#   'routing_entropy': 2.45,
#   'load_balance_loss': 0.012,
# }
```

#### MoD Routing (2 methods)

```python
# Adjust compute allocation
trainer.adjust_mod_capacity(0.7)  # 70% of tokens get full compute

# Get efficiency metrics
stats = trainer.get_mod_statistics()
# Returns: {
#   'capacity_factor': 0.5,
#   'tokens_processed': 1000000,
#   'average_depth': 0.48,
# }
```

#### Batch Size Adaptation (2 methods)

```python
# Dynamically change batch size
trainer.adjust_batch_size(4)  # Reduce for OOM

# Recreate dataloader (internal use)
trainer._recreate_dataloader(dataset)
```

#### Orchestrator Communication (3 methods)

```python
# Get real-time training state
metrics = trainer.get_current_metrics()
# Returns: {
#   'loss': 2.456,
#   'learning_rate': 0.0001,
#   'grad_norm': 1.23,
#   'throughput': 1000.5,
# }

# Extract MoE routing statistics
routing = trainer._extract_moe_routing_stats()

# Calculate throughput
throughput = trainer._calculate_throughput()  # tokens/second
```

#### Emergency Recovery (2 methods)

```python
# Emergency LR reduction (gradient explosion)
trainer.emergency_lr_reduction(10.0)  # Reduce by 10x

# Rollback to previous state
trainer.rollback_steps(100)  # Go back 100 steps
```

#### Advanced Optimizer (2 methods)

```python
# Adjust weight decay
trainer.adjust_weight_decay(0.05)

# Update optimizer parameter groups (internal use)
trainer._update_optimizer_param_groups('lr', 1e-5)
```

### Chinchilla Scaling

Automatic epoch optimization based on compute-optimal principles.

**Core Principle:** For optimal training, use approximately 20 tokens per parameter.

```python
# Enable in Main.py
chinchilla_params = {
    'auto_epoch_scaling': True,       # Enable automatic calculation
    'chinchilla_multiplier': 20,      # Tokens per parameter
    'min_auto_epochs': 1,
    'max_auto_epochs': 50,
    
    # Advanced features
    'enable_loss_landscape': True,    # Plateau/divergence detection
    'enable_compute_efficiency': True,# Track loss per FLOP
    'enable_early_stopping': True,    # Stop when converged
}
```

**How it works:**

1. **Calculate optimal tokens:** `N_opt = 20 × model_parameters`
   - Example: 1B params → 20B optimal tokens

2. **Determine base epochs:** `epochs = N_opt / dataset_tokens`
   - Example: 20B tokens / 10B dataset → 2 epochs

3. **Monitor during training:**
   - Convergence score (loss stability)
   - Compute efficiency (loss reduction per FLOP)
   - Loss landscape (plateaus, divergence)

4. **Adjust dynamically:**
   - If converging fast → reduce epochs
   - If plateau detected → adjust or stop
   - If diverging → emergency intervention

**Example output:**

```
ENHANCED CHINCHILLA SCALER INITIALIZED
Model Parameters: 1.2B
Dataset Tokens: 24.5B
Chinchilla Optimal: 24.0B tokens
Base optimal epochs: 5
Token Budget Coverage: 102.1%

[Step 5000] CHINCHILLA STATUS
Current epochs: 4 (adjusted from 5)
Token progress: 83.4%
Convergence: 87% (High)
Training phase: convergence
Compute efficiency: Stable
Recommendation: Continue training
```

### Example: Adaptive Training in Action

```
[Step 100] Training normally...
Loss: 2.456 | Perplexity: 11.65 | Accuracy: 45.2%

[Orchestrator] Detected loss plateau
Decision: Increase learning rate by 1.5x
Reasoning: Loss variance < 0.001 for 50 steps
Confidence: 75%

[Step 200] Improved convergence...
Loss: 2.234 | Perplexity: 9.34 | Accuracy: 48.7%

[Orchestrator] Expert imbalance detected
Decision: Adjust capacity factor 1.25 → 1.75
Reasoning: Expert 3 at 92%, Expert 5 at 8%
Confidence: 82%

[Step 300] Gradient spike detected...
Loss: 2.189 | Grad Norm: 156.2

[Orchestrator] Emergency intervention
Decision: Emergency LR reduction by 10x
Reasoning: Gradient norm > 100 (explosion)
Confidence: 95%

[Step 350] Training stabilized...
Loss: 2.156 | Perplexity: 8.63 | Accuracy: 52.1%

[Chinchilla] Convergence predicted in ~200 steps
Convergence score: 88%
Expected final loss: 2.05 ± 0.15
```

---

## Monitoring & Debugging

### Training Metrics

Tracked automatically:

- **Loss & Perplexity**: Training/validation curves
- **Accuracy**: Token-level prediction accuracy
- **Throughput**: Tokens per second
- **Memory**: GPU/MPS/CPU utilization
- **Learning Rate**: Current value with scheduler info
- **Gradient Norms**: Detect instability
- **Expert Stats**: MoE routing statistics (if applicable)
- **Chinchilla Metrics**: Token progress, convergence score

### Logging Configuration

```python
monitoring_params = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_every_n_steps': 50,
    'health_check_interval': 100,
}
```

### Checkpointing

```python
training_params = {
    'save_every_n_batches': 1000,
    'save_total_limit': 5,  # Keep only N best checkpoints
    'early_stopping_patience': 10,
}

# Resume from checkpoint
checkpoint_params = {
    'resume_from_checkpoint': 'path/to/checkpoint.pt',
    'resume_training': True,
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Automatic Handling:** The orchestrator catches OOM and reduces batch size automatically.

**Manual intervention:**

```python
# Reduce memory usage
training_params = {
    'batch_size': 2,                    # Smaller batches
    'gradient_accumulation_steps': 16,  # Maintain effective batch size
    'gradient_checkpointing': True,     # Trade compute for memory
    'use_deepspeed': True,
    'zero_stage': 3,                    # Maximum memory optimization
    'cpu_offload': True,                # Offload to CPU
}
```

### Slow Training

**Automatic:** Orchestrator monitors throughput and suggests optimizations.

**Manual optimizations:**

```python
training_params = {
    'compile': True,              # PyTorch 2.0 compilation
    'precision': 'mixed_bf16',    # Faster than FP32
    'use_flash_attention': True,  # 2-4x faster attention
    'num_workers': 4,             # Data loading parallelism
}
```

### Training Instabilities

**Automatic Recovery:**
- Gradient explosion → Emergency LR reduction
- Loss spikes → Checkpoint rollback
- Expert collapse → Routing adjustments

**Manual tuning:**

```python
# Gradient explosion
training_params = {
    'gradient_clip_val': 1.0,  # Clip gradients
    'learning_rate': 5e-5,     # Lower LR
}

# Expert imbalance (MoE)
training_params = {
    'load_balancing_weight': 0.02,  # Stronger balancing
    'capacity_factor': 1.5,         # More token capacity
    'routing_temperature': 1.2,     # More exploration
}
```

### MPS (Apple Silicon) Issues

```python
# If encountering stability issues
training_params = {
    'compile': False,       # Can be unstable on MPS
    'num_workers': 0,       # MPS prefers single-threaded
    'batch_size': 2,        # Start small
    'precision': 'fp16',    # Explicitly set
}
```

---

## Complete Configuration Example

```python
# Main.py - Production configuration for 7B model

config_choice = 'b7'  # 7B active, 56B total
use_adaptive_training = True  # Enable orchestrator

# Training configuration
training_params = {
    # Core settings
    'num_epochs': 3,
    'batch_size': 16,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-4,
    
    # Precision
    'precision': 'mixed_bf16',
    
    # Architecture
    'use_moe': True,
    'use_mod': True,  # Hybrid mode
    'num_experts': 8,
    'moe_top_k': 2,
    'mod_capacity_factor': 0.5,
    
    # Optimization
    'use_flash_attention': True,
    'gradient_checkpointing': True,
    'compile': True,
    
    # DeepSpeed
    'use_deepspeed': True,
    'zero_stage': 2,
    'gradient_compression': True,
    
    # Monitoring
    'save_every_n_batches': 1000,
    'log_every_n_steps': 50,
    'early_stopping_patience': 10,
}

# Data configuration
data_params = {
    'training_mode': 'finetuning_only',
    'finetuning_paths': [
        'data/train_1.jsonl',
        'data/train_2.jsonl',
    ],
    'finetuning_eval_paths': ['data/eval.jsonl'],
}

# Chinchilla scaling
chinchilla_params = {
    'auto_epoch_scaling': True,
    'chinchilla_multiplier': 20,
    'enable_loss_landscape': True,
    'enable_compute_efficiency': True,
    'enable_early_stopping': True,
}

# Quantization (optional)
quantization_params = {
    'quantization_method': 'bnb',
    'quantization_bits': 8,
}
```

---

## Best Practices

### For AI Engineers

1. **Start with adaptive training enabled** - Let the system optimize itself
2. **Use pre-configured model sizes** - They're tested and optimized
3. **Monitor the orchestrator logs** - Learn what decisions are being made
4. **Trust automatic recovery** - The system handles most issues
5. **Use Chinchilla scaling** - Stop guessing optimal epochs

### For Production

1. **Enable checkpointing** - Save every 1000 steps
2. **Use mixed precision** - `mixed_bf16` for modern GPUs
3. **Enable DeepSpeed** - For multi-GPU setups
4. **Monitor memory** - Set `save_total_limit` to avoid disk issues
5. **Test recovery** - Ensure checkpoint resumption works

### For Experimentation

1. **Use `debug` config first** - Fast iteration
2. **Enable verbose logging** - `log_level='DEBUG'`
3. **Disable compilation initially** - `compile=False` for easier debugging
4. **Monitor expert utilization** - For MoE models
5. **Check Chinchilla recommendations** - Optimize training duration

---

## Citation

```bibtex
@software{luminaai2025,
  title = {LuminaAI: Transformer Training with Adaptive Intelligence},
  author = {MatN23},
  year = {2025},
  url = {https://github.com/matn23/luminaai},
  note = {Autonomous training optimization with MoE, MoD, and Chinchilla scaling}
}
```

---

## License

Custom License - See [LICENSE](LICENSE) file for details.

---

## Resources

- **Documentation**: [docs/](docs/)
- **Adaptive Training Guide**: [docs/adaptive_training.md](docs/adaptive_training.md)
- **MoE/MoD Tutorial**: [docs/sparse_architectures.md](docs/sparse_architectures.md)

---

<div align="center">

**Built for the AI engineering community**

*Training systems that learn, adapt, and optimize themselves*

</div>