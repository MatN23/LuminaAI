# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import math
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from typing import Dict, Optional, Any, Union, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import numpy as np
import json
import os

# DeepSpeed imports with fallback
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.runtime.utils import see_memory_usage
    DEEPSPEED_AVAILABLE = True
    logging.info("DeepSpeed available")
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available - falling back to standard training")
    
    # Mock classes for fallback
    class DeepSpeedEngine:
        pass

try:
    from core.dataset import create_dataloader
except ImportError:
    # Fallback dataloader creation
    from torch.utils.data import DataLoader
    def create_dataloader(dataset, config, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=getattr(config, 'batch_size', 1),
            shuffle=shuffle,
            num_workers=getattr(config, 'num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

try:
    from monitoring.logger import TrainingHealthMonitor
except ImportError:
    # Fallback logger
    class TrainingHealthMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

try:
    from training.checkpoint import CheckpointManager
except ImportError:
    # Fallback checkpoint manager
    class CheckpointManager:
        def __init__(self, *args, **kwargs):
            pass
        def save_checkpoint(self, *args, **kwargs):
            pass


class MoEOptimizationManager:
    """Manages MoE-specific optimizations for routing balance and communication efficiency."""
    
    def __init__(self, config):
        self.config = config
        self.routing_stats = {
            'expert_usage': {},
            'load_balance_losses': [],
            'routing_decisions': [],
            'communication_overhead': []
        }
        self.optimization_history = []
        
    def create_deepspeed_moe_config(self, base_config: dict) -> dict:
        """Create optimized DeepSpeed MoE configuration addressing common issues."""
        
        # Calculate optimal expert parallel size based on available GPUs
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        num_experts = getattr(self.config, 'num_experts', 8)
        
        # Expert parallelism: Use smaller groups to reduce all-to-all overhead
        # Rule: expert_parallel_size should divide evenly into world_size and num_experts
        optimal_ep_size = self._calculate_optimal_expert_parallel_size(world_size, num_experts)
        
        moe_config = {
            # Core MoE settings
            "moe": {
                "enabled": True,
                "num_experts": num_experts,
                "expert_parallel_size": optimal_ep_size,
                "top_k": getattr(self.config, 'moe_top_k', 2),
                
                # ROUTING BALANCE OPTIMIZATIONS
                "capacity_factor": getattr(self.config, 'capacity_factor', 1),  # Increased from default 1.25
                "eval_capacity_factor": 3.2,  # Higher for evaluation
                "min_capacity": 16,  # Ensure minimum tokens per expert
                "use_residual": True,  # Handle dropped tokens
                
                # LOAD BALANCING
                "load_balance_loss_coef": getattr(self.config, 'load_balancing_weight', 0.08),  # Increased
                "load_balance_type": "aux_loss",  # Use auxiliary loss for better balance
                "router_jitter_noise": 0.01,  # Add noise to prevent hot experts
                
                # COMMUNICATION OPTIMIZATIONS
                "enable_expert_tensor_parallelism": True,
                "all_to_all_dispatch": True,
                "overlap_alltoall": True,  # Critical for multi-node
                "comm_dtype": "fp16" if self.config.precision in ["fp16", "mixed_fp16"] else "bf16",
                
                # MEMORY OPTIMIZATIONS  
                "pad_expert_input_to_capacity": True,  # Better GPU utilization
                "enable_expert_weight_parallelism": True,
                "moe_param_group": True,  # Group MoE params for ZeRO
                
                # EXPERT PLACEMENT STRATEGY
                "expert_placement_policy": "balanced",  # Distribute experts evenly
                "use_tutel": False,  # Disable for stability unless specifically needed
            }
        }
        
        # Update base config with MoE optimizations
        base_config.update(moe_config)
        
        # Add ZeRO-3 configuration for MoE parameter sharding
        if "zero_optimization" not in base_config:
            base_config["zero_optimization"] = {}
            
        base_config["zero_optimization"].update({
            "stage": 3,  # ZeRO-3 for parameter sharding
            "offload_param": {
                "device": "cpu" if getattr(self.config, 'cpu_offload', False) else "none",
                "nvme_path": getattr(self.config, 'nvme_path', None),
                "buffer_count": 5,
                "buffer_size": 100000000.0,
                "max_in_cpu": 1000000000.0
            },
            "offload_optimizer": {
                "device": "cpu" if getattr(self.config, 'cpu_offload_optimizer', False) else "none",
                "nvme_path": getattr(self.config, 'nvme_path', None),
                "buffer_count": 4,
                "pin_memory": True,
                "pipeline_read": True,
                "pipeline_write": True,
                "fast_init": False
            },
            "stage3_param_persistence_threshold": 10000.0,  # Aggressive parameter offloading
            "stage3_max_live_parameters": 1000000000.0,
            "stage3_prefetch_bucket_size": 50000000.0,
            "memory_efficient_linear": True,  # Critical for MoE
            "stage3_max_reuse_distance": 1000
        })
        
        logging.info(f"MoE Config: {num_experts} experts, EP size: {optimal_ep_size}, "
                    f"capacity factor: {moe_config['moe']['capacity_factor']}")
        
        return base_config
    
    def _calculate_optimal_expert_parallel_size(self, world_size: int, num_experts: int) -> int:
        """Calculate optimal expert parallel size to minimize all-to-all overhead."""
        
        # Find divisors of world_size that also work well with num_experts
        possible_ep_sizes = []
        for i in range(1, world_size + 1):
            if world_size % i == 0:
                # Check if this EP size works well with number of experts
                experts_per_group = num_experts // i
                if experts_per_group >= 1:  # At least 1 expert per parallel group
                    possible_ep_sizes.append((i, experts_per_group))
        
        if not possible_ep_sizes:
            return 1
        
        # Scoring function: prefer smaller EP sizes (less all-to-all) but not too small
        # that we get too few experts per group
        best_ep_size = 1
        best_score = 0
        
        for ep_size, experts_per_group in possible_ep_sizes:
            # Score based on:
            # 1. Communication efficiency (smaller EP groups better)
            # 2. Expert utilization (more experts per group better to a point)
            # 3. Load balancing potential
            
            comm_score = 1.0 / ep_size  # Smaller EP size = less communication overhead
            expert_score = min(experts_per_group / 4.0, 1.0)  # Diminishing returns after 4 experts/group
            balance_score = 1.0 if experts_per_group > 1 else 0.5  # Prefer multiple experts per group
            
            total_score = comm_score * expert_score * balance_score
            
            if total_score > best_score:
                best_score = total_score
                best_ep_size = ep_size
        
        return best_ep_size
    
    def monitor_routing_balance(self, aux_losses: Dict[str, torch.Tensor], 
                              routing_probs: Optional[torch.Tensor] = None):
        """Monitor and log routing balance metrics."""
        
        # Track load balancing losses
        if 'load_balance_loss' in aux_losses:
            self.routing_stats['load_balance_losses'].append(aux_losses['load_balance_loss'].item())
        
        # Analyze routing probabilities if available
        if routing_probs is not None:
            # Calculate expert usage statistics
            expert_usage = routing_probs.sum(dim=0).cpu().numpy()
            total_tokens = routing_probs.sum().item()
            
            if total_tokens > 0:
                usage_percentages = expert_usage / total_tokens * 100
                
                # Update running statistics
                for expert_id, usage_pct in enumerate(usage_percentages):
                    if expert_id not in self.routing_stats['expert_usage']:
                        self.routing_stats['expert_usage'][expert_id] = []
                    self.routing_stats['expert_usage'][expert_id].append(usage_pct)
                
                # Log warnings for severe imbalance
                max_usage = usage_percentages.max()
                min_usage = usage_percentages.min()
                imbalance_ratio = max_usage / max(min_usage, 0.1)  # Avoid division by zero
                
                if imbalance_ratio > 10:  # More than 10x difference
                    logging.warning(f"Severe routing imbalance detected: "
                                  f"max usage {max_usage:.1f}%, min usage {min_usage:.1f}%")
    
    def get_routing_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive routing diagnostics."""
        diagnostics = {
            'timestamp': time.time(),
            'load_balance_trend': [],
            'expert_balance_score': 0.0,
            'routing_efficiency': 0.0,
            'recommendations': []
        }
        
        # Analyze load balance losses
        if self.routing_stats['load_balance_losses']:
            recent_losses = self.routing_stats['load_balance_losses'][-100:]  # Last 100 steps
            diagnostics['load_balance_trend'] = {
                'recent_avg': np.mean(recent_losses),
                'trend': 'improving' if len(recent_losses) > 10 and np.mean(recent_losses[-5:]) < np.mean(recent_losses[-10:-5]) else 'stable'
            }
        
        # Analyze expert usage balance
        if self.routing_stats['expert_usage']:
            expert_usages = []
            for expert_id, usage_history in self.routing_stats['expert_usage'].items():
                if usage_history:
                    expert_usages.append(np.mean(usage_history[-50:]))  # Recent average
            
            if expert_usages:
                usage_std = np.std(expert_usages)
                usage_mean = np.mean(expert_usages)
                balance_score = max(0, 1.0 - (usage_std / max(usage_mean, 1.0)))
                diagnostics['expert_balance_score'] = balance_score
                
                # Generate recommendations
                if balance_score < 0.7:
                    diagnostics['recommendations'].append("Consider increasing load_balance_loss_coef")
                    diagnostics['recommendations'].append("Try adding router jitter noise")
                    
                if usage_std > 5.0:
                    diagnostics['recommendations'].append("Severe imbalance: consider adjusting capacity_factor")
        
        return diagnostics


class EnhancedConversationTrainer:
    """Production trainer with DeepSpeed, MoE optimizations, and CPU offloading."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced managers
        self.moe_optimizer = MoEOptimizationManager(config) if hasattr(config, 'use_moe') and config.use_moe else None
        
        # DeepSpeed integration - CRITICAL FIX
        self.use_deepspeed = DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False)
        self.deepspeed_engine = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Initialize training metrics
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': []
        }
        
        # Setup training components
        self._setup_training()
        
    def _setup_training(self):
        """Setup training components based on DeepSpeed availability."""
        if self.use_deepspeed:
            self._setup_deepspeed_training()
        else:
            self._setup_standard_training()

    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training with MoE and CPU offloading optimizations."""
        print("="*60)
        print("INITIALIZING DEEPSPEED TRAINING")
        print("="*60)
        
        # Debug information
        print(f"DeepSpeed available: {DEEPSPEED_AVAILABLE}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Config use_deepspeed: {getattr(self.config, 'use_deepspeed', False)}")
        print(f"World size: {int(os.environ.get('WORLD_SIZE', 1))}")
        print(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
        
        # Create DeepSpeed configuration
        ds_config = self._create_deepspeed_config()
        
        # Log the configuration for debugging
        print("DeepSpeed Configuration:")
        config_str = json.dumps(ds_config, indent=2, default=str)
        print(config_str[:2000])  # Print first 2000 chars to avoid spam
        
        # Initialize DeepSpeed engine
        try:
            print("Attempting DeepSpeed initialization...")
            
            self.deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=self.model.parameters()
            )
            
            self.optimizer = optimizer
            self.scheduler = lr_scheduler
            self.model = self.deepspeed_engine
            
            # CRITICAL: Set this flag to indicate successful DeepSpeed init
            self.use_deepspeed = True
            
            # Log DeepSpeed setup success
            print("✅ DEEPSPEED INITIALIZATION SUCCESSFUL!")
            print(f"  World size: {self.deepspeed_engine.world_size}")
            print(f"  Local rank: {self.deepspeed_engine.local_rank}")
            print(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'disabled')}")
            
            if ds_config.get('moe', {}).get('enabled', False):
                print(f"  MoE enabled: {ds_config['moe']['num_experts']} experts")
                print(f"  Expert parallel size: {ds_config['moe']['expert_parallel_size']}")
            
        except Exception as e:
            print("❌ DEEPSPEED INITIALIZATION FAILED!")
            print(f"Error: {e}")
            
            # Import traceback for detailed error info
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
            print("🔄 Falling back to standard PyTorch training...")
            self.use_deepspeed = False
            self._setup_standard_training()
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create comprehensive DeepSpeed configuration with FIXED batch size calculation."""
        
        # CRITICAL FIX: Calculate effective batch size correctly
        micro_batch_size = getattr(self.config, 'batch_size', 1)
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Calculate train_batch_size correctly: micro_batch * grad_accum * world_size
        train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
        
        print(f"Batch size calculation:")
        print(f"  Micro batch size: {micro_batch_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  World size: {world_size}")
        print(f"  Train batch size: {train_batch_size}")
        
        # Base configuration with FIXED parameters
        ds_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            # Precision settings - SIMPLIFIED
            "fp16": {
                "enabled": getattr(self.config, 'precision', 'fp32') in ["fp16", "mixed_fp16"],
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": getattr(self.config, 'precision', 'fp32') in ["bf16", "mixed_bf16"]
            },
            
            # Gradient clipping
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            
            # FIXED: Simplified scheduler configuration
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 1e-6,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": 10  # Fixed number instead of calculation
                }
            },
            
            # Communication settings - SIMPLIFIED
            "allgather_partitions": True,
            "allgather_bucket_size": int(5e8),
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": int(5e8),
            "contiguous_gradients": True,
            
            # Logging
            "steps_per_print": 1,  # Log every step for debugging
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        # Add FIXED optimizer configuration
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": self.config.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": getattr(self.config, 'weight_decay', 0.01)
            }
        }
        
        # Add MoE configuration if model uses MoE
        if hasattr(self.config, 'use_moe') and self.config.use_moe and self.moe_optimizer:
            print("Adding MoE configuration to DeepSpeed config...")
            ds_config = self.moe_optimizer.create_deepspeed_moe_config(ds_config)
        else:
            # Standard ZeRO configuration - SIMPLIFIED
            zero_stage = getattr(self.config, 'zero_stage', 2)
            ds_config["zero_optimization"] = {
                "stage": zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": int(5e8),
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": int(5e8),
                "contiguous_gradients": True
            }
            
            # Add CPU offloading ONLY if explicitly enabled
            if getattr(self.config, 'cpu_offload', False):
                print("Adding CPU offloading configuration...")
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "nvme_path": getattr(self.config, 'nvme_path', None),
                    "buffer_count": 4,
                    "pin_memory": True,
                    "pipeline_read": True,
                    "pipeline_write": True,
                    "fast_init": False
                }
                
                if getattr(self.config, 'cpu_offload_parameters', False):
                    ds_config["zero_optimization"]["offload_param"] = {
                        "device": "cpu",
                        "nvme_path": getattr(self.config, 'nvme_path', None),
                        "buffer_count": 5,
                        "buffer_size": 100000000.0,
                        "max_in_cpu": 1000000000.0,
                        "pin_memory": True
                    }
        
        return ds_config
    
    def _setup_standard_training(self):
        """Setup standard PyTorch training as fallback."""
        print("="*60)
        print("SETTING UP STANDARD PYTORCH TRAINING")
        print("="*60)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_standard_optimizer()
        self.scheduler = None
        
        # Mixed precision setup
        self.training_precision = getattr(self.config, 'precision', 'fp32')
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Model compilation
        if getattr(self.config, 'compile', True) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        print(f"✅ Standard training setup complete - Device: {self.device}")
    
    def _create_standard_optimizer(self) -> torch.optim.Optimizer:
        """Create standard PyTorch optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': getattr(self.config, 'weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        try:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
            )
        except Exception:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _get_autocast_context(self, precision: Optional[str] = None, for_inference: bool = False):
        """Get autocast context with comprehensive precision support."""
        if self.use_deepspeed:
            return nullcontext()  # DeepSpeed handles precision internally
        
        # Use existing precision logic for standard training
        target_precision = precision or (getattr(self.config, 'inference_precision', self.training_precision) if for_inference else self.training_precision)
        
        if target_precision == "fp32" or not torch.cuda.is_available():
            return nullcontext()
        elif target_precision in ["fp16", "mixed_fp16"]:
            try:
                return autocast('cuda', dtype=torch.float16)
            except TypeError:
                return autocast('cuda')
        elif target_precision in ["bf16", "mixed_bf16"]:
            try:
                return autocast('cuda', dtype=torch.bfloat16)
            except TypeError:
                return autocast('cuda')
        else:
            return nullcontext() 
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with MoE auxiliary losses."""
        
        # For next-token prediction, shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten tensors
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Create attention mask (ignore padding tokens)
        mask = (flat_labels != getattr(self.tokenizer, 'pad_token_id', 0)).float()
        
        # Compute base loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # Apply weights if provided
        if loss_weights is not None:
            # Also shift and flatten loss weights
            shift_weights = loss_weights[..., 1:].contiguous()
            flat_weights = shift_weights.view(-1)
            weighted_loss = loss * flat_weights * mask
        else:
            weighted_loss = loss * mask
        
        # Check for numerical issues
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            print("NaN or Inf detected in loss computation")
            return {
                'loss': torch.tensor(0.0, device=loss.device, requires_grad=True),
                'raw_loss': torch.tensor(0.0, device=loss.device),
                'perplexity': torch.tensor(float('inf'), device=loss.device),
                'valid_tokens': torch.tensor(0.0, device=loss.device)
            }
        
        # Compute final loss
        total_loss = weighted_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        # Compute additional metrics
        raw_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        perplexity = torch.exp(raw_loss.clamp(max=10))
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with DeepSpeed and MoE support."""
        if self.use_deepspeed:
            return self._deepspeed_train_step(batch)
        else:
            return self._standard_train_step(batch)
    
    def _deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """DeepSpeed training step with guaranteed metric return."""
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)  # Use input_ids as labels if not provided
        loss_weights = batch.get('loss_weights')
        
        if input_ids is None or input_ids.numel() == 0:
            # Return safe defaults
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0
            }
        
        try:
            # Forward pass
            output = self.deepspeed_engine(input_ids, attention_mask)
            
            # Handle MoE outputs
            aux_losses = {}
            if isinstance(output, tuple):
                if len(output) == 3:  # (logits, total_aux_loss, aux_losses_dict)
                    logits, total_aux_loss, aux_losses = output
                else:  # (logits, total_aux_loss)
                    logits, total_aux_loss = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
                loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
                
                # Monitor MoE routing if available
                if aux_losses and self.moe_optimizer:
                    self.moe_optimizer.monitor_routing_balance(aux_losses)
            else:
                logits = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
            
            loss = loss_dict['loss']
            
            # Backward pass (DeepSpeed handles everything)
            self.deepspeed_engine.backward(loss)
            
            # Extract values safely
            loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
            raw_loss_value = loss_dict['raw_loss'].item() if hasattr(loss_dict['raw_loss'], 'item') else float(loss_dict['raw_loss'])
            perplexity_value = loss_dict['perplexity'].item() if hasattr(loss_dict['perplexity'], 'item') else float(loss_dict['perplexity'])
            valid_tokens_value = loss_dict['valid_tokens'].item() if hasattr(loss_dict['valid_tokens'], 'item') else float(loss_dict['valid_tokens'])
            
            return {
                'loss': loss_value,
                'raw_loss': raw_loss_value,
                'perplexity': perplexity_value,
                'valid_tokens': valid_tokens_value
            }
            
        except Exception as e:
            print(f"DeepSpeed training step error: {e}")
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0
            }
    
    def _standard_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard PyTorch training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)  # Use input_ids as labels if not provided
        loss_weights = batch.get('loss_weights')
        
        if input_ids is None or input_ids.numel() == 0:
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0
            }
        
        # Forward pass with precision
        with self._get_autocast_context(for_inference=False):
            output = self.model(input_ids, attention_mask)
            
            if isinstance(output, tuple):
                logits, total_aux_loss, aux_losses = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
                loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
                
                # Monitor MoE routing if available
                if aux_losses and self.moe_optimizer:
                    self.moe_optimizer.monitor_routing_balance(aux_losses)
            else:
                logits = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
        
        loss = loss_dict['loss']
        
        # Check for valid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Invalid loss detected, skipping batch")
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0
            }
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item(),
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with DeepSpeed support."""
        if self.use_deepspeed:
            return self._deepspeed_optimizer_step()
        else:
            return self._standard_optimizer_step()
    
    def _deepspeed_optimizer_step(self) -> Dict[str, float]:
        """DeepSpeed optimizer step with proper gradient norm handling."""
        # DeepSpeed handles gradient clipping, optimization, and LR scheduling internally
        self.deepspeed_engine.step()
        
        # Get metrics with proper error handling
        current_lr = self.config.learning_rate  # Default fallback
        try:
            if hasattr(self.deepspeed_engine, 'get_lr') and callable(self.deepspeed_engine.get_lr):
                lr_list = self.deepspeed_engine.get_lr()
                if lr_list and len(lr_list) > 0:
                    current_lr = lr_list[0]
        except Exception as e:
            print(f"Could not get learning rate from DeepSpeed: {e}")
        
        # Get gradient norm with proper error handling
        grad_norm = 0.0
        try:
            if hasattr(self.deepspeed_engine, 'get_global_grad_norm'):
                norm = self.deepspeed_engine.get_global_grad_norm()
                if norm is not None and not (math.isnan(norm) or math.isinf(norm)):
                    grad_norm = float(norm)
        except Exception as e:
            print(f"Could not get gradient norm from DeepSpeed: {e}")
            grad_norm = 0.0
        
        return {
            'grad_norm': grad_norm,
            'lr': current_lr
        }
    
    def _standard_optimizer_step(self) -> Dict[str, float]:
        """Standard optimizer step."""
        # Unscale gradients for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )
        
        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN/Inf gradients detected, skipping step")
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                self.scaler.update()
            return {'grad_norm': 0.0, 'lr': 0.0}
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 100) -> Dict[str, float]:
        """Enhanced evaluation with DeepSpeed support."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
        else:
            self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Monitor memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            labels = batch.get('labels', input_ids)
            loss_weights = batch.get('loss_weights')
            
            if input_ids is None or input_ids.numel() == 0:
                continue
            
            # Forward pass
            if self.use_deepspeed:
                output = self.deepspeed_engine(input_ids, attention_mask)
            else:
                with self._get_autocast_context(for_inference=True):
                    output = self.model(input_ids, attention_mask)
            
            # Handle outputs
            if isinstance(output, tuple):
                logits = output[0]  # Just take the logits for evaluation
            else:
                logits = output
            
            loss_dict = self.compute_loss(logits, labels, loss_weights)
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                num_batches += 1
        
        eval_time = time.time() - eval_start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_time': eval_time,
                'eval_throughput': 0.0,
                'eval_peak_memory_mb': peak_memory
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        perplexity = math.exp(min(avg_raw_loss, 10))
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_time': eval_time,
            'eval_throughput': throughput,
            'eval_peak_memory_mb': peak_memory
        }
    
    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with FIXED logging."""
        if self.use_deepspeed:
            self.deepspeed_engine.train()
        else:
            self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_raw_loss': 0.0,
            'total_tokens': 0,
            'num_batches': 0,
            'grad_norm_sum': 0.0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'raw_loss': 0.0,
            'tokens': 0
        }
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        epoch_start_time = time.time()
        last_log_time = time.time()
        
        print(f"Starting epoch {epoch + 1} with {len(train_dataloader)} batches")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # ALWAYS log every step for debugging
            if batch_idx < 10 or batch_idx % 5 == 0:
                print(f"DEBUG: Batch {batch_idx}, Step metrics: {step_metrics}")
            
            # Skip invalid batches
            if step_metrics['loss'] == 0.0 or math.isnan(step_metrics['loss']) or math.isinf(step_metrics['loss']):
                print(f"Skipping batch {batch_idx} due to invalid loss: {step_metrics['loss']}")
                continue
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss'] / gradient_accumulation_steps
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                # Update epoch metrics
                if accumulation_metrics['loss'] > 0:
                    epoch_metrics['total_loss'] += accumulation_metrics['loss']
                    epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                    epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                    epoch_metrics['num_batches'] += 1
                    if 'grad_norm' in opt_metrics and opt_metrics['grad_norm'] is not None:
                        epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                # Calculate throughput
                step_time = time.time() - step_start_time
                tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                
                # FORCE logging - log every step for the first 20 steps, then every 5 steps
                should_log = (
                    self.global_step <= 20 or 
                    self.global_step % 5 == 0 or 
                    time.time() - last_log_time > 10
                )
                
                if should_log:
                    self._log_training_step(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = time.time()
                
                # System monitoring
                if self.global_step % 20 == 0:
                    self._log_memory_usage(f"Step {self.global_step}")
                
                # Reset accumulation metrics
                accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
        else:
            avg_loss = avg_raw_loss = avg_grad_norm = avg_tokens_per_sec = 0.0
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
              f"Avg Loss: {avg_loss:.6f} | "
              f"Avg Grad Norm: {avg_grad_norm:.4f} | "
              f"Throughput: {avg_tokens_per_sec:.0f} tokens/s")
        
        return {
            'avg_loss': avg_loss,
            'avg_raw_loss': avg_raw_loss,
            'avg_grad_norm': avg_grad_norm,
            'epoch_time': epoch_time,
            'throughput': avg_tokens_per_sec
        }
    
    def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                          metrics, opt_metrics, tokens_per_sec: float):
        """FIXED logging with guaranteed output."""
        
        try:
            # Memory info with fallback
            memory_info = ""
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
                except:
                    memory_info = " | GPU: N/A"
            
            # Training mode info
            mode_info = " | DeepSpeed" if self.use_deepspeed else " | Standard"
            
            # Safe metric extraction with defaults
            loss = metrics.get('loss', 0.0)
            raw_loss = metrics.get('raw_loss', loss)
            lr = opt_metrics.get('lr', 0.0)
            grad_norm = opt_metrics.get('grad_norm', 0.0)
            
            # Safe perplexity calculation
            try:
                ppl_value = min(raw_loss, 50)  # Cap to prevent overflow
                perplexity = math.exp(ppl_value)
                ppl_str = f"{perplexity:.2e}" if perplexity > 10000 else f"{perplexity:.2f}"
            except:
                ppl_str = "N/A"
            
            # FORCE the log message
            log_message = (
                f"Epoch {epoch+1} | Step {self.global_step:6d} | "
                f"Batch {batch_idx+1:4d}/{total_batches} | "
                f"Loss: {loss:.6f} | "
                f"PPL: {ppl_str} | "
                f"LR: {lr:.2e} | "
                f"GradNorm: {grad_norm:.4f} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
                f"{mode_info}{memory_info}"
            )
            
            # Multiple logging attempts to ensure visibility
            logging.info(log_message)
            print(f"[TRAINING] {log_message}")
            
        except Exception as e:
            # Emergency fallback logging
            fallback_msg = f"Step {self.global_step} | Loss: {metrics.get('loss', 'N/A')} | Logging Error: {e}"
            logging.error(fallback_msg)
            print(f"[TRAINING ERROR] {fallback_msg}")
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with enhanced logging."""
        print("="*80)
        if self.use_deepspeed:
            print("STARTING DEEPSPEED TRAINING WITH ENHANCED LOGGING")
        else:
            print("STARTING STANDARD TRAINING WITH ENHANCED LOGGING")
        print("="*80)
        
        # Store eval dataset
        self.eval_dataset = eval_dataset
        
        # Setup data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        if len(train_dataloader) == 0:
            print("ERROR: Train dataloader is empty!")
            return
        
        # Calculate total steps (DeepSpeed handles this internally)
        if not self.use_deepspeed:
            gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
            total_steps = len(train_dataloader) * self.config.num_epochs // gradient_accumulation_steps
            self._setup_scheduler(total_steps)
        
        # Log training configuration
        self._log_training_config(len(train_dataloader))
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*60}")
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Evaluation
                if eval_dataset is not None:
                    print("Running evaluation...")
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    print(f"Epoch {epoch + 1} Summary:")
                    print(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    print(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    print(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Early stopping check
                    if getattr(self.config, 'early_stopping_patience', None):
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Checkpointing
                if self.use_deepspeed:
                    self._save_deepspeed_checkpoint(epoch + 1)
                else:
                    self._save_standard_checkpoint(epoch + 1)
                
                self.current_epoch = epoch + 1
                
                # MoE diagnostics
                if self.moe_optimizer:
                    moe_diagnostics = self.moe_optimizer.get_routing_diagnostics()
                    if moe_diagnostics.get('recommendations', []):
                        print("MoE Routing Recommendations:")
                        for rec in moe_diagnostics['recommendations']:
                            print(f"  - {rec}")
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            total_training_time = time.time() - training_start_time
            print(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            # Final checkpoint
            if self.use_deepspeed:
                self._save_deepspeed_checkpoint(self.current_epoch, final=True)
            else:
                self._save_standard_checkpoint(self.current_epoch, final=True)
    
    def _save_deepspeed_checkpoint(self, epoch: int, final: bool = False):
        """Save DeepSpeed checkpoint."""
        try:
            checkpoint_dir = Path(f"checkpoints/deepspeed_epoch_{epoch}")
            if final:
                checkpoint_dir = Path("checkpoints/deepspeed_final")
            
            self.deepspeed_engine.save_checkpoint(str(checkpoint_dir))
            print(f"DeepSpeed checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            print(f"Failed to save DeepSpeed checkpoint: {e}")
    
    def _save_standard_checkpoint(self, epoch: int, final: bool = False):
        """Save standard PyTorch checkpoint."""
        try:
            suffix = "final" if final else f"epoch_{epoch:03d}"
            checkpoint_path = Path(f"checkpoints/checkpoint_{suffix}_{self.global_step}.pt")
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': self.global_step,
                'epoch': epoch,
                'config': self.config
            }, checkpoint_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler for standard training."""
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        
        lr_scheduler = getattr(self.config, 'lr_scheduler', 'linear')
        
        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, 
                eta_min=getattr(self.config, 'min_lr', 1e-6)
            )
        elif lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=warmup_ratio
            )
    
    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            self.should_stop = True
    
    def _log_training_config(self, batches_per_epoch: int):
        """Log comprehensive training configuration."""
        try:
            model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except:
            model_params = "Unknown"
        
        config_info = [
            f"Training Mode: {'DeepSpeed' if self.use_deepspeed else 'Standard PyTorch'}",
            f"Model Parameters: {model_params:,}" if isinstance(model_params, int) else f"Model Parameters: {model_params}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Micro batch size: {getattr(self.config, 'batch_size', 1)}",
            f"Gradient accumulation: {getattr(self.config, 'gradient_accumulation_steps', 1)}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {getattr(self.config, 'weight_decay', 0.01)}",
            f"Precision: {getattr(self.config, 'precision', 'fp32')}",
            f"Device: {self.device}"
        ]
        
        if self.use_deepspeed:
            config_info.extend([
                f"World size: {int(os.environ.get('WORLD_SIZE', 1))}",
                f"CPU offloading: {'Enabled' if getattr(self.config, 'cpu_offload', False) else 'Disabled'}",
            ])
            
            if hasattr(self.config, 'use_moe') and self.config.use_moe:
                config_info.extend([
                    f"MoE experts: {getattr(self.config, 'num_experts', 8)}",
                    f"MoE top-k: {getattr(self.config, 'moe_top_k', 2)}"
                ])
        
        print("Training Configuration:")
        for info in config_info:
            print(f"  {info}")
    
    def _log_memory_usage(self, context: str):
        """Log memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            print(f"{context} - GPU Memory: {allocated:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"{context} - System Memory: {memory.percent:.1f}% used, "
                  f"{memory.available / 1e9:.1f}GB available")
        except ImportError:
            pass
    
    def get_moe_diagnostics(self) -> Dict[str, Any]:
        """Get MoE routing and performance diagnostics."""
        if not self.moe_optimizer:
            return {"error": "MoE not enabled or not available"}
        
        return self.moe_optimizer.get_routing_diagnostics()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        
        # CPU memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            stats['cpu'] = {
                'usage_percent': memory.percent,
                'available_gb': memory.available / 1e9,
                'total_gb': memory.total / 1e9
            }
        except ImportError:
            stats['cpu'] = {'error': 'psutil not available'}
        
        return stats
    
    def optimize_for_sequence_length(self, sequence_length: int):
        """Optimize training configuration for specific sequence length."""
        if not self.use_deepspeed:
            print("Sequence length optimization requires DeepSpeed")
            return
        
        # Calculate optimal batch size for long sequences
        if sequence_length > 50000:  # Very long sequences
            # Reduce batch size, increase gradient accumulation
            optimal_micro_batch = max(1, self.config.batch_size // 4)
            optimal_grad_accum = self.config.batch_size * 4
            
            print(f"Optimizing for very long sequences ({sequence_length})")
            print(f"Reducing micro batch size to {optimal_micro_batch}")
            print(f"Increasing gradient accumulation to {optimal_grad_accum}")
            
            # Update DeepSpeed configuration if possible
            try:
                self.deepspeed_engine.train_micro_batch_size_per_gpu = optimal_micro_batch
                self.deepspeed_engine.gradient_accumulation_steps = optimal_grad_accum
            except AttributeError:
                print("Could not update DeepSpeed batch sizes dynamically")
    
    def get_current_metrics(self):
        """Get current training metrics for adaptive processing."""
        try:
            from training.orchestrator import TrainingMetrics
        except ImportError:
            # Fallback if orchestrator not available
            from dataclasses import dataclass
            from datetime import datetime
            
            @dataclass
            class TrainingMetrics:
                epoch: int
                step: int
                loss: float
                grad_norm: float
                learning_rate: float
                expert_utilization: dict
                memory_usage: dict
                throughput: float
                semantic_coherence: float
                factual_accuracy: float
                reasoning_score: float
                timestamp: datetime
        
        return TrainingMetrics(
            epoch=self.current_epoch,
            step=self.global_step,
            loss=self.metrics['train_losses'][-1] if self.metrics['train_losses'] else 0.0,
            grad_norm=self.metrics['gradient_norms'][-1] if self.metrics['gradient_norms'] else 0.0,
            learning_rate=self.metrics['learning_rates'][-1] if self.metrics['learning_rates'] else self.config.learning_rate,
            expert_utilization={f'expert_{i}': 0.125 for i in range(8)},  # Mock data
            memory_usage=self.get_memory_stats(),
            throughput=self.metrics['throughput'][-1] if self.metrics['throughput'] else 0.0,
            semantic_coherence=0.8,  # Mock
            factual_accuracy=0.7,    # Mock
            reasoning_score=0.6,     # Mock
            timestamp=datetime.now()
        )
    
    def adjust_learning_rate(self, new_lr: float):
        """Adjust learning rate during training."""
        old_lr = self.config.learning_rate
        self.config.learning_rate = new_lr
        
        if self.use_deepspeed:
            # Update DeepSpeed learning rate
            try:
                if hasattr(self.deepspeed_engine, 'optimizer'):
                    for param_group in self.deepspeed_engine.optimizer.param_groups:
                        param_group['lr'] = new_lr
                print(f"DeepSpeed learning rate adjusted: {old_lr:.2e} -> {new_lr:.2e}")
            except Exception as e:
                print(f"Failed to adjust DeepSpeed learning rate: {e}")
        else:
            # Update standard optimizer learning rate
            if self.optimizer:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate adjusted: {old_lr:.2e} -> {new_lr:.2e}")
    
    def emergency_lr_reduction(self, factor: float = 0.1):
        """Emergency learning rate reduction for stability."""
        current_lr = self.config.learning_rate
        new_lr = current_lr * factor
        self.adjust_learning_rate(new_lr)
        print(f"EMERGENCY: Learning rate reduced by {factor}x: {current_lr:.2e} -> {new_lr:.2e}")


class DeepSpeedConfigGenerator:
    """Helper class to generate optimized DeepSpeed configurations."""
    
    @staticmethod
    def create_moe_config_with_expert_parallelism(
        num_gpus: int, 
        num_experts: int, 
        model_size_gb: float,
        sequence_length: int
    ) -> Dict[str, Any]:
        """Create MoE configuration optimized for expert parallelism."""
        
        # Calculate optimal expert parallel size
        # Rule: minimize all-to-all communication while balancing expert load
        optimal_ep_size = DeepSpeedConfigGenerator._calculate_expert_parallel_size(
            num_gpus, num_experts
        )
        
        # Calculate capacity factor based on sequence length
        # Longer sequences need higher capacity to avoid dropping tokens
        if sequence_length > 100000:
            capacity_factor = 3.5
        elif sequence_length > 50000:
            capacity_factor = 3.0  
        elif sequence_length > 20000:
            capacity_factor = 2.8
        else:
            capacity_factor = 1
        
        config = {
            "moe": {
                "enabled": True,
                "num_experts": num_experts,
                "expert_parallel_size": optimal_ep_size,
                "top_k": 2,
                
                # Routing optimizations
                "capacity_factor": capacity_factor,
                "eval_capacity_factor": capacity_factor + 0.4,
                "min_capacity": max(16, sequence_length // 10000),
                "use_residual": True,
                
                # Load balancing
                "load_balance_loss_coef": 0.02,
                "load_balance_type": "aux_loss",
                "router_jitter_noise": 0.01,
                
                # Communication optimizations
                "enable_expert_tensor_parallelism": True,
                "all_to_all_dispatch": True,
                "overlap_alltoall": True,
                "comm_dtype": "bf16",
                
                # Memory optimizations
                "pad_expert_input_to_capacity": True,
                "enable_expert_weight_parallelism": True,
                "moe_param_group": True,
                "expert_placement_policy": "balanced"
            }
        }
        
        print(f"Generated MoE config:")
        print(f"  Expert parallel size: {optimal_ep_size} (from {num_gpus} GPUs)")
        print(f"  Capacity factor: {capacity_factor} (for seq_len {sequence_length})")
        print(f"  Experts per parallel group: {num_experts // optimal_ep_size}")
        
        return config
    
    @staticmethod
    def _calculate_expert_parallel_size(num_gpus: int, num_experts: int) -> int:
        """Calculate optimal expert parallel size."""
        # Find all divisors of num_gpus
        divisors = []
        for i in range(1, num_gpus + 1):
            if num_gpus % i == 0:
                divisors.append(i)
        
        # Score each divisor based on:
        # 1. Communication efficiency (smaller groups = less all-to-all overhead)
        # 2. Expert utilization (more experts per group = better load balancing)
        # 3. Memory efficiency
        
        best_ep_size = 1
        best_score = 0
        
        for ep_size in divisors:
            experts_per_group = num_experts / ep_size
            
            # Skip if we can't distribute experts evenly
            if experts_per_group < 1:
                continue
            
            # Communication score: prefer smaller groups (less all-to-all)
            comm_score = 1.0 / ep_size
            
            # Expert utilization score: prefer 2-8 experts per group
            if 2 <= experts_per_group <= 8:
                util_score = 1.0
            elif experts_per_group > 8:
                util_score = 8.0 / experts_per_group
            else:
                util_score = experts_per_group / 2.0
            
            # Memory score: prefer configurations that allow for good memory distribution
            memory_score = min(1.0, experts_per_group / 4.0)
            
            total_score = comm_score * util_score * memory_score
            
            if total_score > best_score:
                best_score = total_score
                best_ep_size = ep_size
        
        return best_ep_size


# Utility functions for backward compatibility and helper methods
def _count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Additional helper methods for debugging
def debug_dataloader(dataloader, tokenizer, max_batches=3):
    """Debug dataloader to check if data is flowing correctly."""
    print(f"\n=== DEBUGGING DATALOADER ===")
    print(f"Dataloader length: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        print(f"\nBatch {batch_idx}:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key == 'input_ids' and hasattr(tokenizer, 'decode'):
                    try:
                        sample_text = tokenizer.decode(value[0][:50].tolist())
                        print(f"    Sample text: {repr(sample_text[:100])}")
                    except:
                        print(f"    Sample tokens: {value[0][:10].tolist()}")
            else:
                print(f"  {key}: {type(value)}")
    
    print(f"=== DATALOADER DEBUG COMPLETE ===\n")


# Add debugging method to trainer
def debug_training_setup(self):
    """Debug training setup to identify issues."""
    print(f"\n=== DEBUGGING TRAINING SETUP ===")
    print(f"Model: {type(self.model)}")
    print(f"Use DeepSpeed: {self.use_deepspeed}")
    print(f"Device: {self.device}")
    print(f"Global step: {self.global_step}")
    print(f"Current epoch: {self.current_epoch}")
    
    if hasattr(self, 'optimizer'):
        print(f"Optimizer: {type(self.optimizer)}")
    
    if self.use_deepspeed and hasattr(self, 'deepspeed_engine'):
        print(f"DeepSpeed engine: {type(self.deepspeed_engine)}")
        print(f"DeepSpeed world size: {getattr(self.deepspeed_engine, 'world_size', 'N/A')}")
    
    print(f"Config batch size: {getattr(self.config, 'batch_size', 'N/A')}")
    print(f"Config grad accumulation: {getattr(self.config, 'gradient_accumulation_steps', 'N/A')}")
    print(f"=== TRAINING SETUP DEBUG COMPLETE ===\n")


# Patch the debug method to the trainer
EnhancedConversationTrainer.debug_training_setup = debug_training_setup