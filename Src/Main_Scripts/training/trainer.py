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

from core.dataset import create_dataloader
from monitoring.logger import TrainingHealthMonitor
from training.checkpoint import CheckpointManager


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
                "capacity_factor": getattr(self.config, 'capacity_factor', 2.8),  # Increased from default 1.25
                "eval_capacity_factor": 3.2,  # Higher for evaluation
                "min_capacity": 16,  # Ensure minimum tokens per expert
                "use_residual": True,  # Handle dropped tokens
                
                # LOAD BALANCING
                "load_balance_loss_coef": getattr(self.config, 'load_balancing_weight', 0.02),  # Increased
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
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
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
            "stage3_param_persistence_threshold": 1e4,  # Aggressive parameter offloading
            "stage3_max_live_parameters": 1e9,
            "stage3_prefetch_bucket_size": 5e7,
            "memory_efficient_linear": True,  # Critical for MoE
            "stage3_max_reuse_distance": 1000,
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


class CPUOffloadManager:
    """Manages CPU offloading strategies for memory optimization."""
    
    def __init__(self, config):
        self.config = config
        self.offload_stats = {
            'cpu_memory_usage': [],
            'gpu_memory_saved': [],
            'transfer_times': []
        }
        
    def setup_cpu_offload(self, model, optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup CPU offloading configuration."""
        
        # Determine what to offload based on available memory
        cpu_offload_config = {
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"}
        }
        
        if getattr(self.config, 'cpu_offload', False):
            # Parameter offloading
            cpu_offload_config["offload_param"] = {
                "device": "cpu",
                "nvme_path": getattr(self.config, 'nvme_path', None),
                "buffer_count": 5,
                "buffer_size": int(1e8),
                "max_in_cpu": int(1e9),
                "pin_memory": True
            }
            
            # Optimizer state offloading
            if getattr(self.config, 'cpu_offload_optimizer', True):
                cpu_offload_config["offload_optimizer"] = {
                    "device": "cpu",
                    "nvme_path": getattr(self.config, 'nvme_path', None),
                    "buffer_count": 4,
                    "pin_memory": True,
                    "pipeline_read": True,
                    "pipeline_write": True,
                    "fast_init": False
                }
            
            logging.info("CPU offloading enabled for parameters and optimizer states")
            
        # Advanced memory management for large models
        if hasattr(self.config, 'aggressive_cpu_offload') and self.config.aggressive_cpu_offload:
            # More aggressive settings for very large models
            cpu_offload_config.update({
                "stage3_param_persistence_threshold": 1e3,  # More aggressive
                "stage3_max_live_parameters": int(5e8),     # Keep fewer params on GPU
                "stage3_prefetch_bucket_size": int(1e7),    # Smaller buckets
                "memory_efficient_linear": True,
                "stage3_max_reuse_distance": 500,           # More aggressive reuse
            })
            
            logging.info("Aggressive CPU offloading enabled")
        
        return cpu_offload_config
    
    def monitor_memory_usage(self):
        """Monitor memory usage during training."""
        try:
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                gpu_memory = gpu_reserved = 0
            
            # CPU memory
            try:
                import psutil
                cpu_memory = psutil.virtual_memory()
                cpu_usage_gb = (cpu_memory.total - cpu_memory.available) / 1e9
                self.offload_stats['cpu_memory_usage'].append(cpu_usage_gb)
            except ImportError:
                cpu_usage_gb = 0
            
            # Log periodically
            if len(self.offload_stats['cpu_memory_usage']) % 100 == 0:
                logging.info(f"Memory usage - GPU: {gpu_memory:.2f}GB/{gpu_reserved:.2f}GB, "
                           f"CPU: {cpu_usage_gb:.2f}GB")
            
        except Exception as e:
            logging.debug(f"Memory monitoring failed: {e}")


class PrecisionManager:
    """Manages different precision types and their configurations."""
    
    # Comprehensive precision definitions
    PRECISION_CONFIGS = {
        "fp32": {
            "dtype": torch.float32,
            "name": "Float32",
            "description": "Full precision (32-bit)",
            "memory_efficiency": 1.0,
            "speed_multiplier": 1.0,
            "numerical_stability": "excellent",
            "supported_devices": ["cpu", "cuda"]
        },
        "fp16": {
            "dtype": torch.float16,
            "name": "Float16",
            "description": "Half precision (16-bit)",
            "memory_efficiency": 2.0,
            "speed_multiplier": 1.5,
            "numerical_stability": "good",
            "supported_devices": ["cuda"]
        },
        "bf16": {
            "dtype": torch.bfloat16,
            "name": "BFloat16",
            "description": "Brain floating point (16-bit with extended range)",
            "memory_efficiency": 2.0,
            "speed_multiplier": 1.4,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"]
        },
        "mixed_fp16": {
            "dtype": torch.float16,
            "name": "Mixed Float16",
            "description": "Mixed precision with fp16 forward, fp32 gradients",
            "memory_efficiency": 1.8,
            "speed_multiplier": 1.6,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"]
        },
        "mixed_bf16": {
            "dtype": torch.bfloat16,
            "name": "Mixed BFloat16",
            "description": "Mixed precision with bf16 forward, fp32 gradients",
            "memory_efficiency": 1.8,
            "speed_multiplier": 1.5,
            "numerical_stability": "excellent",
            "supported_devices": ["cuda"]
        },
        "tf32": {
            "dtype": None,
            "name": "TensorFloat-32",
            "description": "NVIDIA Tensor Float (19-bit precision)",
            "memory_efficiency": 1.0,
            "speed_multiplier": 1.2,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"]
        },
        "dynamic": {
            "dtype": None,
            "name": "Dynamic",
            "description": "Automatically select best precision",
            "memory_efficiency": "variable",
            "speed_multiplier": "variable",
            "numerical_stability": "variable",
            "supported_devices": ["cpu", "cuda"]
        }
    }
    
    @classmethod
    def get_supported_precisions(cls, device: torch.device) -> List[str]:
        """Get list of supported precisions for the given device."""
        device_type = device.type
        supported = []
        
        for precision, config in cls.PRECISION_CONFIGS.items():
            if device_type in config["supported_devices"]:
                if precision in ["bf16", "mixed_bf16"]:
                    if device_type == "cuda" and torch.cuda.is_available():
                        try:
                            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
                            supported.append(precision)
                        except:
                            continue
                elif precision == "tf32":
                    if device_type == "cuda" and torch.cuda.is_available():
                        if hasattr(torch.cuda, 'get_device_capability'):
                            capability = torch.cuda.get_device_capability(device.index or 0)
                            if capability[0] >= 8:
                                supported.append(precision)
                else:
                    supported.append(precision)
        
        return supported
    
    @classmethod
    def auto_select_precision(cls, device: torch.device, priority: str = "balanced") -> str:
        """Automatically select the best precision for the device."""
        supported = cls.get_supported_precisions(device)
        
        if not supported:
            return "fp32"
        
        if priority == "speed":
            for precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "fp32"]:
                if precision in supported:
                    return precision
        elif priority == "memory":
            for precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16", "fp32"]:
                if precision in supported:
                    return precision
        elif priority == "stability":
            for precision in ["bf16", "mixed_bf16", "fp32", "mixed_fp16", "fp16"]:
                if precision in supported:
                    return precision
        else:  # balanced
            for precision in ["mixed_bf16", "bf16", "mixed_fp16", "tf32", "fp16", "fp32"]:
                if precision in supported:
                    return precision
        
        return "fp32"


class EnhancedConversationTrainer:
    """Production trainer with DeepSpeed, MoE optimizations, and CPU offloading."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced managers
        self.precision_manager = PrecisionManager()
        self.moe_optimizer = MoEOptimizationManager(config)
        self.cpu_offload_manager = CPUOffloadManager(config)
        
        # DeepSpeed integration - CRITICAL FIX
        self.use_deepspeed = DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False)
        self.deepspeed_engine = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Setup training components
        self._setup_training()
        
    def _setup_training(self):
        """Setup training components based on DeepSpeed availability."""
        if self.use_deepspeed:
            self._setup_deepspeed_training()
        else:
            self._setup_standard_training()

    def _create_dataloader(self, dataset):
        """Create dataloader for training."""
        return create_dataloader(dataset, self.config, shuffle=True)

    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training with MoE and CPU offloading optimizations."""
        logging.info("="*60)
        logging.info("INITIALIZING DEEPSPEED TRAINING")
        logging.info("="*60)
        
        # Debug information
        logging.info(f"DeepSpeed available: {DEEPSPEED_AVAILABLE}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"Config use_deepspeed: {getattr(self.config, 'use_deepspeed', False)}")
        logging.info(f"World size: {int(os.environ.get('WORLD_SIZE', 1))}")
        logging.info(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
        
        # Create DeepSpeed configuration
        ds_config = self._create_deepspeed_config()
        
        # Log the configuration for debugging
        logging.info("DeepSpeed Configuration:")
        config_str = json.dumps(ds_config, indent=2, default=str)
        logging.info(config_str)
        
        # Initialize DeepSpeed engine
        try:
            logging.info("Attempting DeepSpeed initialization...")
            
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
            logging.info("✅ DEEPSPEED INITIALIZATION SUCCESSFUL!")
            logging.info(f"  World size: {self.deepspeed_engine.world_size}")
            logging.info(f"  Local rank: {self.deepspeed_engine.local_rank}")
            logging.info(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'disabled')}")
            
            if ds_config.get('moe', {}).get('enabled', False):
                logging.info(f"  MoE enabled: {ds_config['moe']['num_experts']} experts")
                logging.info(f"  Expert parallel size: {ds_config['moe']['expert_parallel_size']}")
            
            # Memory estimation
            if hasattr(self.deepspeed_engine, 'estimate_model_mem_needs'):
                try:
                    mem_estimate = self.deepspeed_engine.estimate_model_mem_needs()
                    logging.info(f"  Estimated memory needs: {mem_estimate}")
                except Exception as e:
                    logging.debug(f"Memory estimation failed: {e}")
            
        except Exception as e:
            logging.error("❌ DEEPSPEED INITIALIZATION FAILED!")
            logging.error(f"Error: {e}")
            logging.error(f"DeepSpeed config that failed: {ds_config}")
            
            # Import traceback for detailed error info
            import traceback
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            
            logging.info("🔄 Falling back to standard PyTorch training...")
            self.use_deepspeed = False
            self._setup_standard_training()
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create comprehensive DeepSpeed configuration."""
        
        # Base configuration with FIXED parameters
        ds_config = {
            "train_batch_size": getattr(self.config, 'effective_batch_size', 
                                      self.config.batch_size * getattr(self.config, 'gradient_accumulation_steps', 1)),
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 1),
            
            # Precision settings - SIMPLIFIED
            "fp16": {
                "enabled": self.config.precision in ["fp16", "mixed_fp16"],
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.config.precision in ["bf16", "mixed_bf16"]
            },
            
            # Gradient clipping
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            
            # FIXED: Simplified scheduler configuration
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": 100  # Fixed number instead of calculation
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
            "steps_per_print": 100,
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
        if hasattr(self.config, 'use_moe') and self.config.use_moe:
            logging.info("Adding MoE configuration to DeepSpeed config...")
            ds_config = self.moe_optimizer.create_deepspeed_moe_config(ds_config)
        else:
            # Standard ZeRO configuration - SIMPLIFIED
            ds_config["zero_optimization"] = {
                "stage": getattr(self.config, 'zero_stage', 2),  # Use stage 2 by default for stability
                "allgather_partitions": True,
                "allgather_bucket_size": int(5e8),
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": int(5e8),
                "contiguous_gradients": True
            }
        
        # Add CPU offloading ONLY if explicitly enabled
        if getattr(self.config, 'cpu_offload', False):
            logging.info("Adding CPU offloading configuration...")
            cpu_offload_config = self.cpu_offload_manager.setup_cpu_offload(self.model, ds_config["optimizer"])
            if "zero_optimization" in ds_config:
                ds_config["zero_optimization"].update(cpu_offload_config)
        
        return ds_config
    
    def _setup_standard_training(self):
        """Setup standard PyTorch training as fallback."""
        logging.info("="*60)
        logging.info("SETTING UP STANDARD PYTORCH TRAINING")
        logging.info("="*60)
        
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
        if getattr(self.config, 'compile', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        logging.info(f"✅ Standard training setup complete - Device: {self.device}")
    
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
        target_precision = precision or (getattr(self, 'inference_precision', self.training_precision) if for_inference else self.training_precision)
        
        if target_precision == "dynamic":
            target_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed" if for_inference else "balanced"
            )
        
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
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with MoE auxiliary losses."""
        # Flatten tensors
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Compute base loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # Apply weights and mask padding
        mask = (flat_labels != 0).float()
        weighted_loss = loss * flat_weights * mask
        
        # Check for numerical issues
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            logging.warning("NaN or Inf detected in loss computation")
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
        """DeepSpeed training step with MoE optimization monitoring."""
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass
        output = self.deepspeed_engine(batch['input_ids'], batch['attention_mask'])
        
        # Handle MoE outputs
        aux_losses = {}
        if isinstance(output, tuple):
            if len(output) == 3:  # (logits, total_aux_loss, aux_losses_dict)
                logits, total_aux_loss, aux_losses = output
            else:  # (logits, total_aux_loss)
                logits, total_aux_loss = output
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
            
            # Monitor MoE routing if available
            if aux_losses:
                self.moe_optimizer.monitor_routing_balance(aux_losses)
        else:
            logits = output
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
        
        loss = loss_dict['loss']
        
        # Backward pass (DeepSpeed handles everything)
        self.deepspeed_engine.backward(loss)
        
        return {
            'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def _standard_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard PyTorch training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with precision
        with self._get_autocast_context(for_inference=False):
            output = self.model(batch['input_ids'], batch['attention_mask'])
            
            if isinstance(output, tuple):
                logits, total_aux_loss, aux_losses = output
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
            else:
                logits = output
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
        
        loss = loss_dict['loss']
        
        # Check for valid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning("Invalid loss detected, skipping batch")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
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
        """DeepSpeed optimizer step."""
        # DeepSpeed handles gradient clipping, optimization, and LR scheduling internally
        self.deepspeed_engine.step()
        
        # Get metrics
        current_lr = self.deepspeed_engine.get_lr()[0] if hasattr(self.deepspeed_engine, 'get_lr') else self.config.learning_rate
        
        # Get gradient norm if available
        grad_norm = 0.0
        try:
            if hasattr(self.deepspeed_engine, 'get_global_grad_norm'):
                grad_norm = self.deepspeed_engine.get_global_grad_norm()
        except:
            pass
        
        return {'grad_norm': grad_norm, 'lr': current_lr}
    
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
            logging.warning("NaN/Inf gradients detected, skipping step")
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
            
            if batch['input_ids'].numel() == 0:
                continue
            
            # Forward pass
            if self.use_deepspeed:
                output = self.deepspeed_engine(batch['input_ids'], batch['attention_mask'])
            else:
                with self._get_autocast_context(for_inference=True):
                    output = self.model(batch['input_ids'], batch['attention_mask'])
            
            # Handle outputs
            if isinstance(output, tuple):
                logits = output[0]  # Just take the logits for evaluation
            else:
                logits = output
            
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
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
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with DeepSpeed and MoE optimizations."""
        logging.info("="*80)
        if self.use_deepspeed:
            logging.info("STARTING DEEPSPEED TRAINING WITH MOE AND CPU OFFLOADING")
        else:
            logging.info("STARTING STANDARD TRAINING")
        logging.info("="*80)
        
        # Store eval dataset
        self.eval_dataset = eval_dataset
        
        # Setup data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        if len(train_dataloader) == 0:
            logging.error("ERROR: Train dataloader is empty!")
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
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Evaluation
                if eval_dataset is not None:
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    logging.info(f"Epoch {epoch + 1} Summary:")
                    logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Early stopping check
                    if getattr(self.config, 'early_stopping_patience', None):
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Checkpointing
                if self.use_deepspeed:
                    self._save_deepspeed_checkpoint(epoch + 1)
                else:
                    self._save_standard_checkpoint(epoch + 1)
                
                self.current_epoch = epoch + 1
                
                # Memory monitoring
                self.cpu_offload_manager.monitor_memory_usage()
                
                # MoE diagnostics
                if hasattr(self.config, 'use_moe') and self.config.use_moe:
                    moe_diagnostics = self.moe_optimizer.get_routing_diagnostics()
                    if moe_diagnostics['recommendations']:
                        logging.info("MoE Routing Recommendations:")
                        for rec in moe_diagnostics['recommendations']:
                            logging.info(f"  - {rec}")
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            logging.info(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            # Final checkpoint
            if self.use_deepspeed:
                self._save_deepspeed_checkpoint(self.current_epoch, final=True)
            else:
                self._save_standard_checkpoint(self.current_epoch, final=True)

    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with enhanced monitoring."""
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
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss']
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
                
                # Periodic logging
                current_time = time.time()
                if self.global_step % 100 == 0 or current_time - last_log_time > 300:
                    self._log_training_step(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = current_time
                
                # System monitoring
                if self.global_step % 500 == 0:
                    self.cpu_offload_manager.monitor_memory_usage()
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
        
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
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
    
    def _save_deepspeed_checkpoint(self, epoch: int, final: bool = False):
        """Save DeepSpeed checkpoint."""
        try:
            checkpoint_dir = Path(f"checkpoints/deepspeed_epoch_{epoch}")
            if final:
                checkpoint_dir = Path("checkpoints/deepspeed_final")
            
            self.deepspeed_engine.save_checkpoint(str(checkpoint_dir))
            logging.info(f"DeepSpeed checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            logging.error(f"Failed to save DeepSpeed checkpoint: {e}")
    
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
            
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
    
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
            logging.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
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
            f"Effective batch size: {getattr(self.config, 'effective_batch_size', self.config.batch_size)}",
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
        
        logging.info("Training Configuration:")
        for info in config_info:
            logging.info(f"  {info}")
    
    def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                          metrics, opt_metrics, tokens_per_sec: float):
        """Log training step with comprehensive information."""
        # Memory info
        memory_info = ""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
        
        # Training mode info
        mode_info = " | DeepSpeed" if self.use_deepspeed else " | Standard"
        
        # Perplexity calculation
        try:
            raw_loss_clamped = min(metrics['raw_loss'], 50)
            perplexity = math.exp(raw_loss_clamped)
            ppl_str = f"{perplexity:.2e}" if perplexity > 10000 else f"{perplexity:.2f}"
        except (OverflowError, ValueError):
            ppl_str = "INF"
        
        logging.info(
            f"Epoch {epoch+1} | Step {self.global_step:6d} | "
            f"Batch {batch_idx+1:4d}/{total_batches} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"PPL: {ppl_str} | "
            f"LR: {opt_metrics['lr']:.2e} | "
            f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
            f"{mode_info}{memory_info}"
        )
    
    def _log_memory_usage(self, context: str):
        """Log memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            logging.info(f"{context} - GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            logging.info(f"{context} - System Memory: {memory.percent:.1f}% used, "
                        f"{memory.available / 1e9:.1f}GB available")
        except ImportError:
            pass
    
    def get_moe_diagnostics(self) -> Dict[str, Any]:
        """Get MoE routing and performance diagnostics."""
        if not (hasattr(self.config, 'use_moe') and self.config.use_moe):
            return {"error": "MoE not enabled"}
        
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
        
        # CPU offload stats
        stats['offload'] = self.cpu_offload_manager.offload_stats
        
        return stats
    
    def optimize_for_sequence_length(self, sequence_length: int):
        """Optimize training configuration for specific sequence length."""
        if not self.use_deepspeed:
            logging.warning("Sequence length optimization requires DeepSpeed")
            return
        
        # Calculate optimal batch size for long sequences
        if sequence_length > 50000:  # Very long sequences
            # Reduce batch size, increase gradient accumulation
            optimal_micro_batch = max(1, self.config.batch_size // 4)
            optimal_grad_accum = self.config.batch_size * 4
            
            logging.info(f"Optimizing for very long sequences ({sequence_length})")
            logging.info(f"Reducing micro batch size to {optimal_micro_batch}")
            logging.info(f"Increasing gradient accumulation to {optimal_grad_accum}")
            
            # Update DeepSpeed configuration if possible
            try:
                self.deepspeed_engine.train_micro_batch_size_per_gpu = optimal_micro_batch
                self.deepspeed_engine.gradient_accumulation_steps = optimal_grad_accum
            except AttributeError:
                logging.warning("Could not update DeepSpeed batch sizes dynamically")
    
    def benchmark_moe_routing(self, num_batches: int = 100) -> Dict[str, Any]:
        """Benchmark MoE routing performance and balance."""
        if not (hasattr(self.config, 'use_moe') and self.config.use_moe):
            return {"error": "MoE not enabled"}
        
        if not self.use_deepspeed:
            return {"error": "MoE benchmarking requires DeepSpeed"}
        
        logging.info(f"Benchmarking MoE routing over {num_batches} batches...")
        
        routing_stats = {
            'expert_utilization': {},
            'communication_times': [],
            'load_balance_losses': [],
            'throughput_comparison': {}
        }
        
        # This would require integration with the actual training loop
        # and access to MoE internal metrics
        logging.warning("MoE benchmarking requires integration with training data")
        
        return routing_stats
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 **kwargs) -> str:
        """Generate response with DeepSpeed support."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        try:
            # Create conversation format
            conversation = {
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            # Encode input
            input_tokens = self.tokenizer.encode_conversation(conversation)
            input_tokens.extend([
                self.tokenizer.special_tokens["<|im_start|>"],
                self.tokenizer.special_tokens["<|assistant|>"]
            ])
            
            # Ensure reasonable context length
            if len(input_tokens) >= self.config.seq_length:
                input_tokens = input_tokens[-(self.config.seq_length//2):]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Generation parameters
            temperature = kwargs.get('temperature', 0.7)
            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 0.9)
            
            # Generation loop
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Check sequence length
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass
                if self.use_deepspeed:
                    logits = model(input_ids)
                else:
                    with self._get_autocast_context(for_inference=True):
                        logits = model(input_ids)
                
                # Handle MoE outputs
                if isinstance(logits, tuple):
                    logits = logits[0]  # Take only the logits
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Check for stop tokens
                if next_token.item() == self.tokenizer.special_tokens["<|im_end|>"]:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response."
        finally:
            if self.use_deepspeed:
                self.deepspeed_engine.train()
            else:
                self.model.train()


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
            capacity_factor = 2.5
        
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
        
        logging.info(f"Generated MoE config:")
        logging.info(f"  Expert parallel size: {optimal_ep_size} (from {num_gpus} GPUs)")
        logging.info(f"  Capacity factor: {capacity_factor} (for seq_len {sequence_length})")
        logging.info(f"  Experts per parallel group: {num_experts // optimal_ep_size}")
        
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
    
    @staticmethod
    def create_cpu_offload_config(
        model_size_gb: float,
        available_gpu_memory_gb: float,
        available_cpu_memory_gb: float,
        nvme_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create CPU offload configuration based on available resources."""
        
        config = {
            "zero_optimization": {
                "stage": 3,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
            }
        }
        
        # Determine what to offload based on memory constraints
        gpu_memory_needed = model_size_gb * 1.5  # Model + optimizer states + activations
        
        if gpu_memory_needed > available_gpu_memory_gb * 0.8:
            # Need aggressive offloading
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "nvme_path": nvme_path,
                "buffer_count": 5,
                "buffer_size": int(1e8),
                "max_in_cpu": min(int(available_cpu_memory_gb * 0.5 * 1e9), int(5e9)),
                "pin_memory": True
            }
            
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "nvme_path": nvme_path,
                "buffer_count": 4,
                "pin_memory": True,
                "pipeline_read": True,
                "pipeline_write": True,
                "fast_init": False
            }
            
            # More aggressive ZeRO-3 settings
            config["zero_optimization"].update({
                "stage3_param_persistence_threshold": 1e3,
                "stage3_max_live_parameters": int(available_gpu_memory_gb * 0.3 * 1e9),
                "stage3_prefetch_bucket_size": int(2e7),
                "memory_efficient_linear": True,
                "stage3_max_reuse_distance": 500,
            })
            
            logging.info("Aggressive CPU offloading enabled")
            
        elif gpu_memory_needed > available_gpu_memory_gb * 0.6:
            # Moderate offloading
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "nvme_path": nvme_path,
                "buffer_count": 4,
                "pin_memory": True,
                "pipeline_read": True,
                "pipeline_write": True,
                "fast_init": False
            }
            
            logging.info("Moderate CPU offloading enabled (optimizer only)")
            
        else:
            # No offloading needed
            logging.info("No CPU offloading needed")
        
        return config