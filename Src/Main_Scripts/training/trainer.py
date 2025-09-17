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
from torch.cuda.amp import autocast, GradScaler
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

class DeepSpeedConfigGenerator:
    """Generate DeepSpeed configurations for different scenarios."""
    
    @staticmethod
    def create_base_config(
        batch_size: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        zero_stage: int = 3,
        precision: str = "bf16",
        cpu_offload: bool = False,
        nvme_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a base DeepSpeed configuration."""
        
        config = {
            "train_batch_size": batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            # Precision settings
            "fp16": {
                "enabled": precision == "fp16",
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": precision == "bf16"
            },
            
            # Gradient clipping
            "gradient_clipping": 1.0,
            
            # Optimizer
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            
            # ZeRO optimization
            "zero_optimization": {
                "stage": zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            
            # Communication settings
            "communication_data_type": precision,
            "steps_per_print": 50,
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        # Add CPU offloading if requested
        if cpu_offload and zero_stage >= 2:
            config["zero_optimization"]["cpu_offload"] = True
            
            if zero_stage == 3:
                config["zero_optimization"]["cpu_offload_params"] = True
                config["zero_optimization"]["cpu_offload_optimizer"] = True
        
        # Add NVMe offloading if path provided
        if nvme_path and zero_stage == 3:
            config["zero_optimization"]["nvme_path"] = nvme_path
            
        return config
    
    @staticmethod
    def create_moe_config_with_expert_parallelism(
        world_size: int,
        num_experts: int,
        model_size_gb: float,
        sequence_length: int
    ) -> Dict[str, Any]:
        """Create MoE-specific DeepSpeed configuration."""
        
        # Calculate expert parallel size
        expert_parallel_size = min(world_size, num_experts)
        if world_size % expert_parallel_size != 0:
            # Find largest divisor of world_size that's <= num_experts
            for ep_size in range(min(world_size, num_experts), 0, -1):
                if world_size % ep_size == 0:
                    expert_parallel_size = ep_size
                    break
        
        moe_config = {
            "moe": {
                "enabled": True,
                "num_experts": num_experts,
                "expert_parallel_size": expert_parallel_size,
                "moe_param_groups": True,
                "use_residual": True,
                
                # Load balancing
                "load_balancing": {
                    "type": "tokens",
                    "loss_weight": 0.01
                },
                
                # Capacity and routing
                "capacity_factor": 1.25,
                "eval_capacity_factor": 1.0,
                "min_capacity": 1,
                "use_tutel": False,  # Set to True if Tutel is available
                
                # Memory optimizations
                "enable_expert_tensor_parallelism": False,
                "all_to_all_group_size": expert_parallel_size,
            }
        }
        
        # Adjust capacity for long sequences or large models
        if sequence_length > 8192 or model_size_gb > 50:
            moe_config["moe"]["capacity_factor"] = 1.0
            moe_config["moe"]["eval_capacity_factor"] = 0.8
        
        return moe_config
    
    @staticmethod
    def optimize_for_t4(config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize DeepSpeed config for T4 GPUs."""
        
        # T4-specific optimizations
        config["fp16"]["enabled"] = True
        config["bf16"]["enabled"] = False  # T4 doesn't support bf16
        
        # Smaller bucket sizes for T4's memory constraints
        if "zero_optimization" in config:
            config["zero_optimization"]["allgather_bucket_size"] = 1e8
            config["zero_optimization"]["reduce_bucket_size"] = 1e8
        
        # Enable CPU offload more aggressively on T4
        if config.get("zero_optimization", {}).get("stage", 0) >= 2:
            config["zero_optimization"]["cpu_offload"] = True
            
        return config
    
    @staticmethod
    def create_config_for_sequence_length(
        sequence_length: int,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize config for specific sequence lengths."""
        
        config = base_config.copy()
        
        # Very long sequences (>16K tokens)
        if sequence_length > 16384:
            # Force smaller micro batches
            config["train_micro_batch_size_per_gpu"] = 1
            config["gradient_accumulation_steps"] = max(8, config.get("gradient_accumulation_steps", 4))
            
            # Enable more aggressive memory optimizations
            config["zero_optimization"]["stage"] = 3
            config["zero_optimization"]["cpu_offload"] = True
            
            # Reduce bucket sizes further
            config["zero_optimization"]["allgather_bucket_size"] = 5e7
            config["zero_optimization"]["reduce_bucket_size"] = 5e7
        
        # Long sequences (8K-16K tokens)
        elif sequence_length > 8192:
            config["train_micro_batch_size_per_gpu"] = min(2, config.get("train_micro_batch_size_per_gpu", 4))
            config["gradient_accumulation_steps"] = max(4, config.get("gradient_accumulation_steps", 2))
        
        return config

class T4OptimizationManager:
    """T4-specific optimizations and compatibility fixes."""
    
    def __init__(self, config):
        self.config = config
        self.device_info = self._get_device_info()
        self.optimized_settings = self._calculate_optimal_settings()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed T4 device information."""
        if not torch.cuda.is_available():
            return {"type": "cpu", "memory_gb": 0, "compute_capability": None}
        
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        try:
            capability = torch.cuda.get_device_capability(0)
        except:
            capability = None
        
        return {
            "type": "gpu",
            "name": device_name,
            "memory_gb": memory_gb,
            "compute_capability": capability,
            "is_t4": "T4" in device_name
        }
    
    def _calculate_optimal_settings(self) -> Dict[str, Any]:
        """Calculate optimal settings for T4."""
        settings = {
            "precision": "fp16",  # T4 supports fp16, not bf16
            "max_batch_size": 8,   # Conservative for T4's 16GB
            "gradient_checkpointing": True,
            "use_amp": True,
            "compile_model": False  # Often causes issues on older hardware
        }
        
        if self.device_info["is_t4"]:
            # T4-specific optimizations
            settings.update({
                "precision": "fp16",  # Force fp16 on T4
                "max_batch_size": min(6, self.config.batch_size),  # Very conservative
                "gradient_accumulation_steps": max(4, getattr(self.config, 'gradient_accumulation_steps', 1)),
                "use_fused_optimizer": False,  # Can cause issues on T4
                "memory_efficient_attention": True
            })
            
            logging.info("Applied T4-specific optimizations")
            logging.info(f"  Precision: {settings['precision']}")
            logging.info(f"  Max batch size: {settings['max_batch_size']}")
            logging.info(f"  Gradient accumulation: {settings['gradient_accumulation_steps']}")
        
        return settings
    
    def get_autocast_context(self, enabled: bool = True):
        """Get proper autocast context for T4."""
        if not enabled or not torch.cuda.is_available():
            return nullcontext()
        
        # Use the new API syntax to avoid deprecation warnings
        try:
            if self.device_info["is_t4"]:
                # Force fp16 on T4
                return torch.amp.autocast('cuda', dtype=torch.float16, enabled=True)
            else:
                # Other GPUs can use bf16 if supported
                return torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True)
        except (AttributeError, TypeError):
            # Fallback to old API
            return autocast(enabled=True)
    
    def optimize_memory_usage(self):
        """Apply T4-specific memory optimizations."""
        if torch.cuda.is_available():
            # The correct way to set memory fraction in modern PyTorch
            if self.device_info["is_t4"]:
                try:
                    # Use the correct API - set_per_process_memory_fraction
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    logging.info("Set CUDA memory fraction to 95% for T4")
                except AttributeError:
                    # Fallback - just log that we tried
                    logging.info("Memory fraction setting not available in this PyTorch version")
        
            # Clear cache
            torch.cuda.empty_cache()
        
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logging.info("Enabled Flash Attention for memory efficiency")
            except (AttributeError, RuntimeError):
                logging.debug("Flash Attention not available or failed to enable")
                pass

class EnhancedConversationTrainer:
    """Production trainer optimized for T4 and other GPUs."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # T4 optimization manager
        self.t4_optimizer = T4OptimizationManager(config)
    
        # Apply T4 optimizations to config
        self._apply_hardware_optimizations()
    
        # DeepSpeed integration
        self.use_deepspeed = DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False)
        self.deepspeed_engine = None
    
        # ADD ALL THESE MISSING ATTRIBUTES:
        self.use_amp = False  # Will be set properly in _setup_training
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
    
        # Training state - THESE ARE THE ONES MISSING NOW:
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
        # Setup training components
        self._setup_training()
    
        logging.info(f"Trainer initialized for {self.t4_optimizer.device_info['name']}")
        logging.info(f"Using precision: {self.training_precision}")
        logging.info(f"Effective batch size: {self.effective_batch_size}")
    
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations to config."""
        optimal = self.t4_optimizer.optimized_settings
        
        # Update config with optimal settings
        self.config.batch_size = min(self.config.batch_size, optimal["max_batch_size"])
        
        if not hasattr(self.config, 'gradient_accumulation_steps'):
            self.config.gradient_accumulation_steps = optimal.get("gradient_accumulation_steps", 1)
        
        # Set precision
        self.training_precision = optimal["precision"]
        if hasattr(self.config, 'precision'):
            self.config.precision = self.training_precision
        
        # Calculate effective batch size
        self.effective_batch_size = self.config.batch_size * getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Enable gradient checkpointing for memory efficiency
        if optimal["gradient_checkpointing"] and hasattr(self.config, 'gradient_checkpointing'):
            self.config.gradient_checkpointing = True
    
    def _setup_training(self):
        """Setup training components based on DeepSpeed availability."""
        if self.use_deepspeed:
            self._setup_deepspeed_training()
        else:
            self._setup_standard_training()
        
        # Apply memory optimizations
        self.t4_optimizer.optimize_memory_usage()
    
    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training."""
        logging.info("Initializing DeepSpeed training...")
        
        # Create DeepSpeed configuration
        ds_config = self._create_deepspeed_config()
        
        # Initialize DeepSpeed engine
        try:
            self.deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=self.model.parameters()
            )
            
            self.optimizer = optimizer
            self.scheduler = lr_scheduler
            self.model = self.deepspeed_engine
            
            logging.info("DeepSpeed initialization successful")
            
        except Exception as e:
            logging.error(f"DeepSpeed initialization failed: {e}")
            logging.info("Falling back to standard training")
            self.use_deepspeed = False
            self._setup_standard_training()
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration optimized for T4."""
        ds_config = {
            "train_batch_size": self.effective_batch_size,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 1),
            
            # Precision settings for T4
            "fp16": {
                "enabled": self.training_precision == "fp16",
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.training_precision == "bf16" and not self.t4_optimizer.device_info["is_t4"]
            },
            
            # Gradient clipping
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            
            # Optimizer
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                }
            },
            
            # ZeRO for memory efficiency
            "zero_optimization": {
                "stage": 2,  # ZeRO-2 for T4 compatibility
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,  # Smaller buckets for T4
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            
            # Communication settings
            "communication_data_type": "fp16",
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            
            # Logging
            "steps_per_print": 50,
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        return ds_config
    
    def _setup_standard_training(self):
        """Setup standard PyTorch training optimized for T4."""
        logging.info("Setting up standard PyTorch training...")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if getattr(self.config, 'gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                logging.info("Gradient checkpointing enabled")
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Mixed precision setup
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"]
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Model compilation (disabled for T4 by default)
        if getattr(self.config, 'compile', False) and not self.t4_optimizer.device_info["is_t4"]:
            try:
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        logging.info(f"Standard training setup complete")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Precision: {self.training_precision}")
        logging.info(f"  AMP enabled: {self.use_amp}")
        logging.info(f"  Gradient checkpointing: {getattr(self.config, 'gradient_checkpointing', False)}")
        
    def optimize_for_sequence_length(self, sequence_length: int):
        """Optimize training configuration for specific sequence lengths."""
        logging.info(f"Optimizing for sequence length: {sequence_length:,}")
    
        # Very long sequences (>16K tokens)
        if sequence_length > 16384:
            logging.info("Very long sequence detected - applying aggressive optimizations")
        
            # Force smaller micro batches
            if hasattr(self.config, 'batch_size'):
                original_batch_size = self.config.batch_size
                self.config.batch_size = 1
                logging.info(f"Reduced batch size from {original_batch_size} to 1")
        
            # Update effective batch size
            self.effective_batch_size = self.config.batch_size * getattr(self.config, 'gradient_accumulation_steps', 1)
    
        # Clear CUDA cache after configuration changes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with T4 optimizations."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in ['bias', 'norm', 'embed', 'layernorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': getattr(self.config, 'weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Use fused optimizer only if not T4
        use_fused = torch.cuda.is_available() and not self.t4_optimizer.device_info["is_t4"]
        
        try:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused
            )
        except Exception:
            # Fallback without fused
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with numerical stability."""
        # Flatten tensors
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Compute base loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # Apply weights and mask padding
        mask = (flat_labels != 0).float()
        weighted_loss = loss * flat_weights * mask
        
        # Check for numerical issues (common on T4)
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
        raw_loss = (loss * mask).sum() / total_weight
        perplexity = torch.exp(raw_loss.clamp(max=10))
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with T4 optimizations."""
        if self.use_deepspeed:
            return self._deepspeed_train_step(batch)
        else:
            return self._standard_train_step(batch)
    
    def _deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """DeepSpeed training step."""
        # Move batch to device with error handling
        try:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        except Exception as e:
            logging.warning(f"Error moving batch to device: {e}")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Check for empty batch
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass
        try:
            output = self.deepspeed_engine(batch['input_ids'], batch['attention_mask'])
            
            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss']
            
            # Backward pass (DeepSpeed handles everything)
            self.deepspeed_engine.backward(loss)
            
            return {
                'loss': loss.item(),
                'raw_loss': loss_dict['raw_loss'].item(),
                'perplexity': loss_dict['perplexity'].item(),
                'valid_tokens': loss_dict['valid_tokens'].item()
            }
            
        except Exception as e:
            logging.warning(f"Error in DeepSpeed training step: {e}")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
    
    def _standard_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard PyTorch training step with T4 optimizations."""
        self.model.train()
        
        # Move batch to device with error handling
        try:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        except Exception as e:
            logging.warning(f"Error moving batch to device: {e}")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Check for empty batch
        if batch['input_ids'].numel() == 0:
            logging.warning("Empty batch detected, skipping")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with proper autocast
        try:
            with self.t4_optimizer.get_autocast_context(enabled=self.use_amp):
                output = self.model(batch['input_ids'], batch['attention_mask'])
                
                # Handle different output formats
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                loss = loss_dict['loss']
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("Invalid loss detected, skipping batch")
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            
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
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("CUDA out of memory! Trying to recover...")
                torch.cuda.empty_cache()
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            else:
                logging.warning(f"Runtime error in training step: {e}")
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        except Exception as e:
            logging.warning(f"Error in standard training step: {e}")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with error handling."""
        if self.use_deepspeed:
            return self._deepspeed_optimizer_step()
        else:
            return self._standard_optimizer_step()
    
    def _deepspeed_optimizer_step(self) -> Dict[str, float]:
        """DeepSpeed optimizer step."""
        try:
            self.deepspeed_engine.step()
            
            # Get metrics
            current_lr = (self.deepspeed_engine.get_lr()[0] 
                         if hasattr(self.deepspeed_engine, 'get_lr') 
                         else self.config.learning_rate)
            
            # Get gradient norm if available
            grad_norm = 0.0
            try:
                if hasattr(self.deepspeed_engine, 'get_global_grad_norm'):
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
            except:
                pass
            
            return {'grad_norm': grad_norm, 'lr': current_lr}
            
        except Exception as e:
            logging.warning(f"Error in DeepSpeed optimizer step: {e}")
            return {'grad_norm': 0.0, 'lr': 0.0}
    
    def _standard_optimizer_step(self) -> Dict[str, float]:
        """Standard optimizer step with T4 optimizations."""
        try:
            # Unscale gradients for AMP
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Compute gradient norm before clipping
            max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
            
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
            except Exception as e:
                logging.warning(f"Gradient clipping failed: {e}")
                grad_norm = 0.0
            
            # Check for NaN gradients (common on T4)
            if grad_norm is None or torch.isnan(grad_norm) or torch.isinf(grad_norm):
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
            current_lr = (self.scheduler.get_last_lr()[0] 
                         if self.scheduler 
                         else self.config.learning_rate)
            
            return {'grad_norm': float(grad_norm) if grad_norm is not None else 0.0, 'lr': current_lr}
            
        except Exception as e:
            logging.warning(f"Error in standard optimizer step: {e}")
            return {'grad_norm': 0.0, 'lr': 0.0}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 50) -> Dict[str, float]:
        """Enhanced evaluation with T4 optimizations."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
        else:
            self.model.eval()
        
        # Reduce max_batches for T4 to avoid memory issues
        if self.t4_optimizer.device_info["is_t4"]:
            max_batches = min(max_batches, 25)
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Clear memory before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            try:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                if batch['input_ids'].numel() == 0:
                    continue
                
                # Forward pass
                if self.use_deepspeed:
                    output = self.deepspeed_engine(batch['input_ids'], batch['attention_mask'])
                else:
                    with self.t4_optimizer.get_autocast_context(enabled=self.use_amp):
                        output = self.model(batch['input_ids'], batch['attention_mask'])
                
                # Handle outputs
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                
                if not (torch.isnan(loss_dict['loss']) or torch.isinf(loss_dict['loss'])):
                    total_loss += loss_dict['loss'].item()
                    total_raw_loss += loss_dict['raw_loss'].item()
                    total_tokens += loss_dict['valid_tokens'].item()
                    num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning("OOM during evaluation, clearing cache")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue
        
        eval_time = time.time() - eval_start_time
        peak_memory = (torch.cuda.max_memory_allocated() / 1e6 
                      if torch.cuda.is_available() else 0)
        
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
        """Main training loop with T4 optimizations."""
        logging.info("="*80)
        logging.info(f"STARTING TRAINING ON {self.t4_optimizer.device_info['name']}")
        logging.info("="*80)
        
        # Store eval dataset
        self.eval_dataset = eval_dataset
        
        # Setup data loaders with error handling
        try:
            train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        except Exception as e:
            logging.error(f"Failed to create train dataloader: {e}")
            return
        
        if len(train_dataloader) == 0:
            logging.error("ERROR: Train dataloader is empty!")
            return
        
        # Calculate total steps
        if not self.use_deepspeed:
            gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
            total_steps = len(train_dataloader) * self.config.num_epochs // gradient_accumulation_steps
            self._setup_scheduler(total_steps)
        
        # Log training configuration
        self._log_training_config(len(train_dataloader))
        
        training_start_time = time.time()
        last_memory_check = 0
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Clear memory before each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Periodic memory check
                current_time = time.time()
                if current_time - last_memory_check > 300:  # Every 5 minutes
                    self._log_memory_usage(f"Epoch {epoch + 1} completed")
                    last_memory_check = current_time
                
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
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("CUDA out of memory during training!")
                logging.error("Try reducing batch size or enabling gradient checkpointing")
            else:
                logging.error(f"Runtime error: {e}")
            raise
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
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with T4 optimizations."""
        if self.use_deepspeed:
            self.deepspeed_engine.train()
        else:
            self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_raw_loss': 0.0,
            'total_tokens': 0,
            'num_batches': 0,
            'grad_norm_sum': 0.0,
            'skipped_batches': 0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'raw_loss': 0.0,
            'tokens': 0
        }
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        epoch_start_time = time.time()
        last_log_time = time.time()
        
        # Progress tracking
        total_batches = len(train_dataloader)
        log_interval = max(1, min(50, total_batches // 10))  # Log at most 10 times per epoch
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.should_stop:
                break
            
            step_start_time = time.time()
            
            # Training step with error handling
            try:
                step_metrics = self.train_step(batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"OOM at batch {batch_idx}, clearing cache and skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    epoch_metrics['skipped_batches'] += 1
                    continue
                else:
                    raise e
            except Exception as e:
                logging.warning(f"Error in batch {batch_idx}: {e}")
                epoch_metrics['skipped_batches'] += 1
                continue
            
            # Skip if step returned invalid metrics
            if step_metrics['loss'] == 0.0 or step_metrics['valid_tokens'] == 0:
                epoch_metrics['skipped_batches'] += 1
                continue
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss']
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                try:
                    opt_metrics = self.optimizer_step()
                    self.global_step += 1
                    
                    # Update epoch metrics
                    if accumulation_metrics['loss'] > 0:
                        epoch_metrics['total_loss'] += accumulation_metrics['loss']
                        epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                        epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                        epoch_metrics['num_batches'] += 1
                        if opt_metrics['grad_norm'] is not None and opt_metrics['grad_norm'] > 0:
                            epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                    
                    # Calculate throughput
                    step_time = time.time() - step_start_time
                    tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                    
                    # Periodic logging
                    current_time = time.time()
                    if (self.global_step % log_interval == 0 or 
                        current_time - last_log_time > 120 or  # Every 2 minutes
                        batch_idx == 0):  # First batch
                        self._log_training_step(
                            epoch, batch_idx, total_batches,
                            accumulation_metrics, opt_metrics, tokens_per_sec
                        )
                        last_log_time = current_time
                    
                    # Reset accumulation metrics
                    accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
                    
                except Exception as e:
                    logging.warning(f"Error in optimizer step at batch {batch_idx}: {e}")
                    continue
            
            # Memory cleanup for T4
            if (self.t4_optimizer.device_info["is_t4"] and 
                batch_idx % 100 == 0 and 
                torch.cuda.is_available()):
                torch.cuda.empty_cache()
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time if epoch_time > 0 else 0
        else:
            avg_loss = avg_raw_loss = avg_grad_norm = avg_tokens_per_sec = 0.0
        
        # Log epoch completion
        success_rate = ((epoch_metrics['num_batches']) / 
                       max(1, epoch_metrics['num_batches'] + epoch_metrics['skipped_batches'])) * 100
        
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logging.info(f"  Success rate: {success_rate:.1f}% ({epoch_metrics['skipped_batches']} batches skipped)")
        logging.info(f"  Avg Loss: {avg_loss:.6f}")
        logging.info(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
        logging.info(f"  Throughput: {avg_tokens_per_sec:.0f} tokens/s")
        
        return {
            'avg_loss': avg_loss,
            'avg_raw_loss': avg_raw_loss,
            'avg_grad_norm': avg_grad_norm,
            'epoch_time': epoch_time,
            'throughput': avg_tokens_per_sec,
            'success_rate': success_rate
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
            
            # Create checkpoint dict
            checkpoint_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'epoch': epoch,
                'config': self.config,
                'training_precision': self.training_precision,
                'best_eval_loss': self.best_eval_loss
            }
            
            if self.scheduler:
                checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        
        lr_scheduler = getattr(self.config, 'lr_scheduler', 'linear')
        
        try:
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
            else:
                # Simple step decay as fallback
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.optimizer, step_size=total_steps//4, gamma=0.5)
                
            logging.info(f"Scheduler: {lr_scheduler}, warmup steps: {warmup_steps}")
        except Exception as e:
            logging.warning(f"Failed to create scheduler: {e}")
            self.scheduler = None
    
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
        
        device_info = self.t4_optimizer.device_info
        
        config_info = [
            f"Device: {device_info['name']} ({device_info['memory_gb']:.1f}GB)",
            f"Training Mode: {'DeepSpeed' if self.use_deepspeed else 'Standard PyTorch'}",
            f"Model Parameters: {model_params:,}" if isinstance(model_params, int) else f"Model Parameters: {model_params}",
            f"Precision: {self.training_precision}",
            f"Batch size: {self.config.batch_size}",
            f"Gradient accumulation: {getattr(self.config, 'gradient_accumulation_steps', 1)}",
            f"Effective batch size: {self.effective_batch_size}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {getattr(self.config, 'weight_decay', 0.01)}",
            f"Gradient checkpointing: {getattr(self.config, 'gradient_checkpointing', False)}",
            f"AMP enabled: {self.use_amp}"
        ]
        
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
        
        # Progress percentage
        progress = (batch_idx + 1) / total_batches * 100
        
        # Perplexity calculation
        try:
            raw_loss_clamped = min(metrics['raw_loss'], 50)
            perplexity = math.exp(raw_loss_clamped)
            ppl_str = f"{perplexity:.2e}" if perplexity > 10000 else f"{perplexity:.2f}"
        except (OverflowError, ValueError):
            ppl_str = "INF"
        
        logging.info(
            f"Epoch {epoch+1} | Step {self.global_step:6d} | "
            f"Progress: {progress:5.1f}% | "
            f"Loss: {metrics['loss']:.6f} | "
            f"PPL: {ppl_str} | "
            f"LR: {opt_metrics['lr']:.2e} | "
            f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
            f"{memory_info}"
        )
    
    def _log_memory_usage(self, context: str):
        """Log memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            logging.info(f"{context}")
            logging.info(f"  GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
            
            # Memory efficiency for T4
            if self.t4_optimizer.device_info["is_t4"]:
                memory_efficiency = allocated / self.t4_optimizer.device_info["memory_gb"] * 100
                logging.info(f"  T4 Memory Usage: {memory_efficiency:.1f}% of {self.t4_optimizer.device_info['memory_gb']:.1f}GB")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            logging.info(f"  System Memory: {memory.percent:.1f}% used, "
                        f"{memory.available / 1e9:.1f}GB available")
        except ImportError:
            pass
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 **kwargs) -> str:
        """Generate response with T4 optimizations."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
            model = self.deepspeed_engine
        else:
            self.model.eval()
            model = self.model
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        # Reduce max tokens for T4 to avoid OOM
        if self.t4_optimizer.device_info["is_t4"]:
            max_new_tokens = min(max_new_tokens, 256)
        
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
            max_context = min(self.config.seq_length // 2, 1024)  # Conservative for T4
            if len(input_tokens) >= max_context:
                input_tokens = input_tokens[-max_context:]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Generation parameters
            temperature = kwargs.get('temperature', 0.7)
            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 0.9)
            
            # Generation loop
            generated_tokens = []
            
            with self.t4_optimizer.get_autocast_context(enabled=True):
                for step in range(max_new_tokens):
                    # Check sequence length
                    if input_ids.size(1) >= max_context:
                        input_ids = input_ids[:, -max_context//2:]
                    
                    # Forward pass
                    try:
                        if self.use_deepspeed:
                            logits = model(input_ids)
                        else:
                            logits = model(input_ids)
                        
                        # Handle different output formats
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
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
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logging.warning("OOM during generation, stopping early")
                            torch.cuda.empty_cache()
                            break
                        else:
                            raise e
            
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
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        return {
            **self.t4_optimizer.device_info,
            'optimized_settings': self.t4_optimizer.optimized_settings,
            'training_precision': self.training_precision,
            'effective_batch_size': self.effective_batch_size
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        stats = {'device_info': self.t4_optimizer.device_info}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                'device_total_gb': self.t4_optimizer.device_info["memory_gb"]
            }
            
            # Memory efficiency
            if stats['gpu']['device_total_gb'] > 0:
                stats['gpu']['efficiency_percent'] = (
                    stats['gpu']['allocated_gb'] / stats['gpu']['device_total_gb'] * 100
                )
        
        # System memory
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
    
    def save_training_summary(self, save_path: str = "training_summary.json"):
        """Save comprehensive training summary."""
        summary = {
            'training_completed': time.time(),
            'device_info': self.get_device_info(),
            'final_stats': {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_loss': self.best_eval_loss
            },
            'memory_stats': self.get_memory_stats(),
            'config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'precision': self.training_precision,
                'use_deepspeed': self.use_deepspeed
            }
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logging.info(f"Training summary saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save training summary: {e}")