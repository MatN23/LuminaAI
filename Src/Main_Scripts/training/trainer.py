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
import gc
from threading import Thread
import queue
from collections import deque

from core.dataset import create_dataloader
from monitoring.logger import TrainingHealthMonitor
from training.checkpoint import CheckpointManager


class CPUOffloadingManager:
    """Manages CPU offloading for model parameters and optimizer states."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.cpu_device = torch.device('cpu')
        
        # Offloading settings
        self.enable_param_offload = getattr(config, 'cpu_offload_params', False)
        self.enable_optimizer_offload = getattr(config, 'cpu_offload_optimizer', True)
        self.enable_gradient_offload = getattr(config, 'cpu_offload_gradients', False)
        self.offload_threshold_mb = getattr(config, 'offload_threshold_mb', 8000)  # 8GB threshold
        
        # Async offloading
        self.enable_async_offload = getattr(config, 'async_offload', True)
        self.offload_queue = queue.Queue() if self.enable_async_offload else None
        self.offload_thread = None
        
        # Memory tracking
        self.peak_memory_mb = 0
        self.offloaded_params = set()
        self.offloaded_optimizer_states = {}
        self.offloaded_gradients = {}
        
        # Performance metrics
        self.offload_stats = {
            'params_offloaded': 0,
            'params_loaded': 0,
            'optimizer_states_offloaded': 0,
            'gradients_offloaded': 0,
            'offload_time': 0.0,
            'load_time': 0.0,
            'memory_saved_mb': 0.0
        }
        
        if self.enable_async_offload:
            self._start_offload_thread()
        
        logging.info(f"CPU Offloading Manager initialized:")
        logging.info(f"  Parameters: {'enabled' if self.enable_param_offload else 'disabled'}")
        logging.info(f"  Optimizer: {'enabled' if self.enable_optimizer_offload else 'disabled'}")
        logging.info(f"  Gradients: {'enabled' if self.enable_gradient_offload else 'disabled'}")
        logging.info(f"  Async offloading: {'enabled' if self.enable_async_offload else 'disabled'}")
        logging.info(f"  Memory threshold: {self.offload_threshold_mb}MB")
    
    def _start_offload_thread(self):
        """Start background thread for async offloading."""
        def offload_worker():
            while True:
                try:
                    task = self.offload_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break
                    
                    task_type, data = task
                    if task_type == 'param':
                        self._offload_parameter_sync(data['param'], data['name'])
                    elif task_type == 'optimizer':
                        self._offload_optimizer_state_sync(data['state'], data['name'])
                    elif task_type == 'gradient':
                        self._offload_gradient_sync(data['param'], data['name'])
                    
                    self.offload_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Offload worker error: {e}")
        
        self.offload_thread = Thread(target=offload_worker, daemon=True)
        self.offload_thread.start()
    
    def should_offload(self) -> bool:
        """Check if offloading should be triggered based on memory usage."""
        if not torch.cuda.is_available():
            return False
        
        current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return current_memory_mb > self.offload_threshold_mb
    
    def offload_model_parameters(self, model: nn.Module, exclude_patterns: List[str] = None):
        """Offload model parameters to CPU."""
        if not self.enable_param_offload or not self.should_offload():
            return
        
        exclude_patterns = exclude_patterns or ['embed', 'norm', 'head']  # Keep critical params on GPU
        
        offloaded_count = 0
        memory_saved = 0
        
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in exclude_patterns):
                continue
            
            if param.device == self.device and name not in self.offloaded_params:
                if self.enable_async_offload:
                    self.offload_queue.put(('param', {'param': param, 'name': name}))
                else:
                    self._offload_parameter_sync(param, name)
                
                memory_saved += param.numel() * param.element_size()
                offloaded_count += 1
        
        self.offload_stats['params_offloaded'] += offloaded_count
        self.offload_stats['memory_saved_mb'] += memory_saved / 1024 / 1024
        
        if offloaded_count > 0:
            logging.debug(f"Offloaded {offloaded_count} parameters, saved {memory_saved/1024/1024:.1f}MB")
    
    def _offload_parameter_sync(self, param: torch.Tensor, name: str):
        """Synchronously offload a parameter to CPU."""
        if param.device != self.cpu_device:
            start_time = time.time()
            param.data = param.data.to(self.cpu_device, non_blocking=True)
            if param.grad is not None:
                param.grad = param.grad.to(self.cpu_device, non_blocking=True)
            self.offloaded_params.add(name)
            self.offload_stats['offload_time'] += time.time() - start_time
    
    def load_model_parameters(self, model: nn.Module, param_names: List[str] = None):
        """Load offloaded parameters back to GPU."""
        if not param_names:
            param_names = list(self.offloaded_params)
        
        loaded_count = 0
        
        for name, param in model.named_parameters():
            if name in param_names and name in self.offloaded_params:
                start_time = time.time()
                param.data = param.data.to(self.device, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(self.device, non_blocking=True)
                self.offloaded_params.remove(name)
                self.offload_stats['load_time'] += time.time() - start_time
                loaded_count += 1
        
        self.offload_stats['params_loaded'] += loaded_count
        
        if loaded_count > 0:
            logging.debug(f"Loaded {loaded_count} parameters back to GPU")
    
    def offload_optimizer_states(self, optimizer: torch.optim.Optimizer):
        """Offload optimizer states to CPU."""
        if not self.enable_optimizer_offload or not self.should_offload():
            return
        
        offloaded_count = 0
        
        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_id = id(param)
                
                if param_id in optimizer.state and param_id not in self.offloaded_optimizer_states:
                    state = optimizer.state[param_id]
                    cpu_state = {}
                    
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.device == self.device:
                            cpu_state[key] = value.to(self.cpu_device, non_blocking=True)
                        else:
                            cpu_state[key] = value
                    
                    self.offloaded_optimizer_states[param_id] = cpu_state
                    del optimizer.state[param_id]  # Remove from GPU
                    offloaded_count += 1
        
        self.offload_stats['optimizer_states_offloaded'] += offloaded_count
        
        if offloaded_count > 0:
            logging.debug(f"Offloaded {offloaded_count} optimizer states")
    
    def load_optimizer_states(self, optimizer: torch.optim.Optimizer, param_ids: List[int] = None):
        """Load optimizer states back to GPU."""
        if not param_ids:
            param_ids = list(self.offloaded_optimizer_states.keys())
        
        loaded_count = 0
        
        for param_id in param_ids:
            if param_id in self.offloaded_optimizer_states:
                cpu_state = self.offloaded_optimizer_states[param_id]
                gpu_state = {}
                
                for key, value in cpu_state.items():
                    if isinstance(value, torch.Tensor):
                        gpu_state[key] = value.to(self.device, non_blocking=True)
                    else:
                        gpu_state[key] = value
                
                optimizer.state[param_id] = gpu_state
                del self.offloaded_optimizer_states[param_id]
                loaded_count += 1
        
        if loaded_count > 0:
            logging.debug(f"Loaded {loaded_count} optimizer states back to GPU")
    
    def offload_gradients(self, model: nn.Module):
        """Offload gradients to CPU after backward pass."""
        if not self.enable_gradient_offload:
            return
        
        offloaded_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.device == self.device:
                if name not in self.offloaded_gradients:
                    self.offloaded_gradients[name] = param.grad.to(self.cpu_device, non_blocking=True)
                    param.grad = None  # Free GPU memory
                    offloaded_count += 1
        
        self.offload_stats['gradients_offloaded'] += offloaded_count
        
        if offloaded_count > 0:
            logging.debug(f"Offloaded {offloaded_count} gradients")
    
    def load_gradients(self, model: nn.Module, param_names: List[str] = None):
        """Load gradients back to GPU before optimizer step."""
        if not param_names:
            param_names = list(self.offloaded_gradients.keys())
        
        loaded_count = 0
        
        for name, param in model.named_parameters():
            if name in param_names and name in self.offloaded_gradients:
                param.grad = self.offloaded_gradients[name].to(self.device, non_blocking=True)
                del self.offloaded_gradients[name]
                loaded_count += 1
        
        if loaded_count > 0:
            logging.debug(f"Loaded {loaded_count} gradients back to GPU")
    
    def memory_cleanup(self):
        """Perform memory cleanup and garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        # Update peak memory tracking
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'peak_memory_mb': self.peak_memory_mb,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            'offloaded_params': len(self.offloaded_params),
            'offloaded_optimizer_states': len(self.offloaded_optimizer_states),
            'offloaded_gradients': len(self.offloaded_gradients),
            'offload_stats': self.offload_stats.copy()
        }
        return stats
    
    def shutdown(self):
        """Shutdown offloading manager and cleanup."""
        if self.enable_async_offload and self.offload_thread:
            self.offload_queue.put(None)  # Shutdown signal
            self.offload_thread.join(timeout=5.0)
        
        # Clear all offloaded data
        self.offloaded_params.clear()
        self.offloaded_optimizer_states.clear()
        self.offloaded_gradients.clear()
        
        self.memory_cleanup()


class PrecisionManager:
    """Manages different precision types and their configurations."""
    
    # Comprehensive precision definitions
    PRECISION_CONFIGS = {
        # Standard precisions
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
        
        # Mixed precision variants
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
        
        # TensorFloat-32
        "tf32": {
            "dtype": None,
            "name": "TensorFloat-32",
            "description": "NVIDIA Tensor Float (19-bit precision)",
            "memory_efficiency": 1.0,
            "speed_multiplier": 1.2,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"]
        },
        
        # Dynamic precision
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
    """Production trainer with CPU offloading, comprehensive monitoring, and precision support."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize managers
        self.precision_manager = PrecisionManager()
        self.cpu_offload_manager = CPUOffloadingManager(config, self.device)
        
        # GPU setup and memory management
        self._setup_gpu()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training precision setup
        self.training_precision = getattr(config, 'precision', 'fp32')
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Inference precision setup
        self.inference_precision = getattr(config, 'inference_precision', 'auto')
        self._setup_inference_precision()
        
        # TF32 setup
        self._setup_tf32()
        
        # Model compilation
        self._compile_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        self.last_backup_time = time.time()
        
        # Memory management settings
        self.memory_cleanup_interval = getattr(config, 'memory_cleanup_interval', 100)
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
        self.offload_frequency = getattr(config, 'offload_frequency', 10)
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing and hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing enabled")
        
        # Metrics and monitoring
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': [],
            'memory_usage': [],
            'offload_stats': []
        }
        
        # Health monitoring
        try:
            self.health_monitor = TrainingHealthMonitor()
        except:
            class SimpleHealthMonitor:
                def update(self, loss, grad_norm): pass
                def get_status(self): return "OK"
                def get_summary(self): return {}
            self.health_monitor = SimpleHealthMonitor()
        
        # Checkpoint management
        try:
            self.checkpoint_manager = CheckpointManager(config)
        except:
            class SimpleCheckpointManager:
                def __init__(self, config):
                    self.config = config
                    self.checkpoint_dir = Path("checkpoints")
                    self.checkpoint_dir.mkdir(exist_ok=True)
                
                def save_checkpoint(self, model, optimizer, scheduler, global_step, epoch, metrics, suffix=""):
                    checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}_{global_step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'global_step': global_step,
                        'epoch': epoch,
                        'metrics': metrics
                    }, checkpoint_path)
                    logging.info(f"Checkpoint saved: {checkpoint_path}")
                    return str(checkpoint_path)
                
                def load_checkpoint(self, path, model, optimizer=None, scheduler=None):
                    checkpoint = torch.load(path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if optimizer and 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    return checkpoint.get('epoch', 0)
                
                def get_resume_path(self):
                    checkpoints = list(self.checkpoint_dir.glob("*.pt"))
                    return str(max(checkpoints, key=lambda x: x.stat().st_mtime)) if checkpoints else None
            
            self.checkpoint_manager = SimpleCheckpointManager(config)
        
        # Log initialization
        logging.info(f"Enhanced Trainer with CPU Offloading initialized:")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Model parameters: {self._count_parameters():,}")
        logging.info(f"  Training precision: {self.training_precision}")
        logging.info(f"  Inference precision: {self.inference_precision}")
        logging.info(f"  Gradient checkpointing: {'enabled' if self.gradient_checkpointing else 'disabled'}")
        self._log_memory_usage("Initial")
    
    def _setup_gpu(self):
        """Setup GPU with optimal configuration."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set memory fraction for offloading
            memory_fraction = getattr(self.config, 'gpu_memory_fraction', 0.85)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            gpu_props = torch.cuda.get_device_properties(0)
            logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
            logging.info(f"GPU Memory Fraction: {memory_fraction}")
            
            if hasattr(torch.cuda, 'get_device_capability'):
                capability = torch.cuda.get_device_capability(0)
                logging.info(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
        else:
            logging.warning("CUDA not available, using CPU")
    
    def _setup_tf32(self):
        """Setup TensorFloat-32 for modern NVIDIA GPUs."""
        if torch.cuda.is_available() and hasattr(torch, 'backends'):
            try:
                if hasattr(torch.backends.cuda, 'matmul'):
                    if self.training_precision == "tf32" or self.inference_precision == "tf32":
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        logging.info("TF32 enabled for CUDA operations")
                    else:
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False
            except Exception as e:
                logging.debug(f"TF32 setup failed: {e}")
    
    def _setup_inference_precision(self):
        """Setup inference precision with comprehensive auto-detection."""
        if self.inference_precision == "auto":
            self.inference_precision = self.precision_manager.auto_select_precision(
                self.device, priority="balanced"
            )
            logging.info(f"Auto-selected inference precision: {self.inference_precision}")
    
    def _count_parameters(self):
        """Count model parameters."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except:
            return 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter grouping."""
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
        except:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _compile_model(self):
        """Compile model with error handling."""
        if getattr(self.config, 'compile', False) and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
    
    def _get_autocast_context(self, precision: Optional[str] = None, for_inference: bool = False):
        """Get autocast context with comprehensive precision support."""
        target_precision = precision or (self.inference_precision if for_inference else self.training_precision)
        
        if target_precision == "dynamic":
            target_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed" if for_inference else "balanced"
            )
        
        if target_precision == "fp32" or not torch.cuda.is_available():
            return nullcontext()
        elif target_precision == "tf32":
            return nullcontext()
        elif target_precision in ["fp16", "mixed_fp16"]:
            try:
                return autocast(device_type='cuda', dtype=torch.float16)
            except TypeError:
                return autocast(enabled=True)
        elif target_precision in ["bf16", "mixed_bf16"]:
            try:
                return autocast(device_type='cuda', dtype=torch.bfloat16)
            except TypeError:
                return autocast(enabled=True)
        else:
            return nullcontext()
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with detailed metrics."""
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_weights = loss_weights.view(-1)
        
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        mask = (flat_labels != 0).float()
        weighted_loss = loss * flat_weights * mask
        
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            logging.warning("NaN or Inf detected in loss computation")
            return {
                'loss': torch.tensor(0.0, device=loss.device, requires_grad=True),
                'raw_loss': torch.tensor(0.0, device=loss.device),
                'perplexity': torch.tensor(float('inf'), device=loss.device),
                'valid_tokens': torch.tensor(0.0, device=loss.device)
            }
        
        total_loss = weighted_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        raw_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        perplexity = torch.exp(raw_loss.clamp(max=10))
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with CPU offloading and memory management."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Memory management - offload if needed
        if self.global_step % self.offload_frequency == 0:
            self.cpu_offload_manager.offload_optimizer_states(self.optimizer)
        
        # Forward pass with training precision
        with self._get_autocast_context(for_inference=False):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss'] / getattr(self.config, 'gradient_accumulation_steps', 1)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning("Invalid loss detected, skipping batch")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Offload gradients if enabled
        if self.cpu_offload_manager.enable_gradient_offload:
            self.cpu_offload_manager.offload_gradients(self.model)
        
        return {
            'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with CPU offloading support."""
        # Load gradients back if they were offloaded
        if self.cpu_offload_manager.enable_gradient_offload:
            self.cpu_offload_manager.load_gradients(self.model)
        
        # Load optimizer states if they were offloaded
        active_param_ids = [id(p) for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
        self.cpu_offload_manager.load_optimizer_states(self.optimizer, active_param_ids)
        
        # Unscale gradients for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )
        
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
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Memory cleanup
        if self.global_step % self.memory_cleanup_interval == 0:
            self.cpu_offload_manager.memory_cleanup()
        
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 100, 
                 precision_override: Optional[str] = None) -> Dict[str, float]:
        """Comprehensive evaluation with precision control and memory management."""
        self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        eval_precision = precision_override or self.inference_precision
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Memory management for evaluation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Load all parameters back for evaluation
        self.cpu_offload_manager.load_model_parameters(self.model)
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if batch['input_ids'].numel() == 0:
                continue
            
            with self._get_autocast_context(precision=eval_precision, for_inference=True):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                num_batches += 1
            
            # Memory cleanup during evaluation
            if batch_idx % 20 == 0:
                self.cpu_offload_manager.memory_cleanup()
        
        eval_time = time.time() - eval_start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_time': eval_time,
                'eval_throughput': 0.0,
                'eval_precision': eval_precision,
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
            'eval_precision': eval_precision,
            'eval_peak_memory_mb': peak_memory
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with CPU offloading and comprehensive monitoring."""
        logging.info("="*80)
        logging.info("STARTING PRODUCTION TRAINING WITH CPU OFFLOADING")
        logging.info("="*80)
        
        self.eval_dataset = eval_dataset
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        total_steps = len(train_dataloader) * self.config.num_epochs // gradient_accumulation_steps
        self._setup_scheduler(total_steps)
        
        self._log_training_config(len(train_dataloader), total_steps)
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                logging.info(f"{'='*60}")
                
                # Train epoch with memory management
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Full evaluation at epoch end
                if eval_dataset is not None:
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    logging.info(f"Epoch {epoch + 1} Summary:")
                    logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    logging.info(f"  Memory Peak: {eval_metrics['eval_peak_memory_mb']:.1f}MB")
                    
                    # Early stopping check
                    if getattr(self.config, 'early_stopping_patience', None):
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Log memory stats
                memory_stats = self.cpu_offload_manager.get_memory_stats()
                logging.info(f"Memory Stats: Current {memory_stats['current_memory_mb']:.1f}MB, "
                           f"Peak {memory_stats['peak_memory_mb']:.1f}MB, "
                           f"Offloaded: {memory_stats['offloaded_params']} params, "
                           f"{memory_stats['offloaded_optimizer_states']} opt states")
                
                self.metrics['memory_usage'].append(memory_stats)
                self.metrics['offload_stats'].append(memory_stats['offload_stats'].copy())
                
                # Log epoch metrics
                try:
                    self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                except Exception as e:
                    logging.warning(f"Failed to log epoch metrics: {e}")
                
                # Save epoch checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, epoch + 1, self.metrics,
                    f"epoch_{epoch + 1:03d}"
                )
                
                self.current_epoch = epoch + 1
                
                # Aggressive memory management at epoch end
                self.cpu_offload_manager.offload_model_parameters(self.model)
                self.cpu_offload_manager.offload_optimizer_states(self.optimizer)
                self.cpu_offload_manager.memory_cleanup()
                
                # Backup checkpoint periodically
                current_time = time.time()
                backup_interval = getattr(self.config, 'backup_every_n_hours', 6) * 3600
                if (current_time - self.last_backup_time) > backup_interval:
                    self._create_backup()
                    self.last_backup_time = current_time
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            logging.info(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            # Final memory stats
            final_memory_stats = self.cpu_offload_manager.get_memory_stats()
            logging.info(f"Final Memory Stats:")
            logging.info(f"  Peak GPU Memory: {final_memory_stats['peak_memory_mb']:.1f}MB")
            logging.info(f"  Total Params Offloaded: {final_memory_stats['offload_stats']['params_offloaded']}")
            logging.info(f"  Total Memory Saved: {final_memory_stats['offload_stats']['memory_saved_mb']:.1f}MB")
            logging.info(f"  Offload Time: {final_memory_stats['offload_stats']['offload_time']:.2f}s")
            logging.info(f"  Load Time: {final_memory_stats['offload_stats']['load_time']:.2f}s")
            
            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "final"
            )
            
            # Save training summary with memory stats
            self._save_training_summary(total_training_time)
            
            # Cleanup
            self.cpu_offload_manager.shutdown()
    
    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with CPU offloading and memory management."""
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
            
            # Training step with CPU offloading
            step_metrics = self.train_step(batch)
            
            accumulation_metrics['loss'] += step_metrics['loss']
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                if accumulation_metrics['loss'] > 0:
                    epoch_metrics['total_loss'] += accumulation_metrics['loss']
                    epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                    epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                    epoch_metrics['num_batches'] += 1
                    epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                step_time = time.time() - step_start_time
                tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                
                self.metrics['train_losses'].append(accumulation_metrics['loss'])
                self.metrics['learning_rates'].append(opt_metrics['lr'])
                self.metrics['gradient_norms'].append(opt_metrics['grad_norm'])
                self.metrics['throughput'].append(tokens_per_sec)
                
                self.health_monitor.update(accumulation_metrics['loss'], opt_metrics['grad_norm'])
                
                # Enhanced logging with memory info
                current_time = time.time()
                if self.global_step % 50 == 0 or current_time - last_log_time > 30:
                    self._log_training_step_with_memory(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = current_time
                
                # Log to monitoring backends
                if self.global_step % 10 == 0:
                    try:
                        memory_stats = self.cpu_offload_manager.get_memory_stats()
                        metrics_to_log = {
                            'train_loss': accumulation_metrics['loss'],
                            'learning_rate': opt_metrics['lr'],
                            'gradient_norm': opt_metrics['grad_norm'],
                            'throughput_tokens_per_sec': tokens_per_sec,
                            'perplexity': math.exp(min(accumulation_metrics['raw_loss'], 10)),
                            'gpu_memory_mb': memory_stats['current_memory_mb'],
                            'offloaded_params': memory_stats['offloaded_params'],
                            'offloaded_optimizer_states': memory_stats['offloaded_optimizer_states']
                        }
                        self.logger.log_metrics(metrics_to_log, self.global_step, "train")
                    except Exception as e:
                        logging.debug(f"Failed to log training metrics: {e}")
                
                # System monitoring with memory management
                health_check_interval = getattr(self.config, 'health_check_interval', 100)
                if self.global_step % health_check_interval == 0:
                    try:
                        if hasattr(self.logger, 'log_system_stats'):
                            self.logger.log_system_stats(self.global_step)
                    except Exception as e:
                        logging.debug(f"Failed to log system stats: {e}")
                    self._log_memory_usage_detailed(f"Step {self.global_step}")
                
                # Periodic evaluation
                eval_every_n_batches = getattr(self.config, 'eval_every_n_batches', 0)
                if (eval_every_n_batches > 0 and 
                    self.global_step % eval_every_n_batches == 0):
                    self._periodic_evaluation()
                
                # Periodic checkpointing
                save_every_n_batches = getattr(self.config, 'save_every_n_batches', 0)
                if (save_every_n_batches > 0 and 
                    self.global_step % save_every_n_batches == 0):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.current_epoch, self.metrics,
                        f"step_{self.global_step:06d}"
                    )
                
                accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
        
        # Handle remaining gradients
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            opt_metrics = self.optimizer_step()
            self.global_step += 1
        
        # Compute epoch statistics
        epoch_time = time.time() - epoch_start_time
        self.metrics['epoch_times'].append(epoch_time)
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
        else:
            avg_loss = float('inf')
            avg_raw_loss = float('inf')
            avg_grad_norm = 0.0
            avg_tokens_per_sec = 0.0
        
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
    
    def _log_training_step_with_memory(self, epoch: int, batch_idx: int, total_batches: int,
                                     metrics, opt_metrics, tokens_per_sec: float):
        """Enhanced training step logging with CPU offloading info."""
        # Memory info including offloading stats
        memory_info = ""
        offload_info = ""
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
        
        # Offloading stats
        memory_stats = self.cpu_offload_manager.get_memory_stats()
        if memory_stats['offloaded_params'] > 0 or memory_stats['offloaded_optimizer_states'] > 0:
            offload_info = f" | Offloaded: {memory_stats['offloaded_params']}P/{memory_stats['offloaded_optimizer_states']}O"
        
        health_status = self.health_monitor.get_status()
        health_info = f" | Health: {health_status}"
        precision_info = f" | Train: {self.training_precision} | Infer: {self.inference_precision}"
        
        logging.info(
            f"Epoch {epoch+1} | Step {self.global_step:6d} | "
            f"Batch {batch_idx+1:4d}/{total_batches} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"PPL: {math.exp(min(metrics['raw_loss'], 10)):.2f} | "
            f"LR: {opt_metrics['lr']:.2e} | "
            f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
            f"{precision_info}{memory_info}{offload_info}{health_info}"
        )
    
    def _log_memory_usage_detailed(self, context: str):
        """Enhanced memory usage logging with offloading details."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            logging.info(f"{context} - GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # Offloading stats
        memory_stats = self.cpu_offload_manager.get_memory_stats()
        if memory_stats['offloaded_params'] > 0 or memory_stats['offloaded_optimizer_states'] > 0:
            logging.info(f"{context} - CPU Offloading: "
                        f"{memory_stats['offloaded_params']} params, "
                        f"{memory_stats['offloaded_optimizer_states']} opt states, "
                        f"{memory_stats['memory_saved_mb']:.1f}MB saved")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            logging.info(f"{context} - System Memory: {memory.percent:.1f}% used, "
                        f"{memory.available / 1e9:.1f}GB available")
        except ImportError:
            pass
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        
        lr_scheduler = getattr(self.config, 'lr_scheduler', 'cosine')
        
        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=getattr(self.config, 'min_lr', 1e-6)
            )
        elif lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate,
                total_steps=total_steps, pct_start=warmup_ratio
            )
        else:  # linear
            try:
                from torch.optim.lr_scheduler import LinearLR
                self.scheduler = LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_steps
                )
            except ImportError:
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.optimizer, step_size=warmup_steps, gamma=0.1)
    
    def _log_training_config(self, batches_per_epoch: int, total_steps: int):
        """Log comprehensive training configuration including offloading settings."""
        try:
            model_params = self._count_parameters()
        except:
            model_params = "Unknown"
        
        config_info = [
            f"Model Parameters: {model_params:,}" if isinstance(model_params, int) else f"Model Parameters: {model_params}",
            f"Epochs: {self.config.num_epochs}",
            f"Batches per epoch: {batches_per_epoch:,}",
            f"Total steps: {total_steps:,}",
            f"Effective batch size: {getattr(self.config, 'effective_batch_size', self.config.batch_size)}",
            f"Learning rate: {self.config.learning_rate:.2e}",
            f"Weight decay: {getattr(self.config, 'weight_decay', 0.01)}",
            f"Warmup ratio: {getattr(self.config, 'warmup_ratio', 0.1)}",
            f"Max grad norm: {getattr(self.config, 'max_grad_norm', 1.0)}",
            f"Training precision: {self.training_precision}",
            f"Inference precision: {self.inference_precision}",
            f"Device: {self.device}",
            f"CPU Offloading: Parameters={self.cpu_offload_manager.enable_param_offload}, "
            f"Optimizer={self.cpu_offload_manager.enable_optimizer_offload}, "
            f"Gradients={self.cpu_offload_manager.enable_gradient_offload}",
            f"Gradient Checkpointing: {'enabled' if self.gradient_checkpointing else 'disabled'}",
            f"Memory Threshold: {self.cpu_offload_manager.offload_threshold_mb}MB"
        ]
        
        logging.info("Training Configuration:")
        for info in config_info:
            logging.info(f"  {info}")
    
    def _log_memory_usage(self, context: str):
        """Basic memory usage logging."""
        self._log_memory_usage_detailed(context)
    
    def _periodic_evaluation(self):
        """Perform periodic evaluation during training with memory management."""
        if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
            eval_metrics = self.evaluate(self.eval_dataset, max_batches=50)
            
            try:
                self.logger.log_metrics(eval_metrics, self.global_step, "eval")
            except Exception as e:
                logging.debug(f"Failed to log eval metrics: {e}")
            
            logging.info(f"Eval | Step {self.global_step} | "
                        f"Loss: {eval_metrics['eval_loss']:.6f} | "
                        f"PPL: {eval_metrics['eval_perplexity']:.2f} | "
                        f"Memory: {eval_metrics['eval_peak_memory_mb']:.1f}MB | "
                        f"Precision: {eval_metrics['eval_precision']}")
            
            if getattr(self.config, 'early_stopping_patience', None):
                self._check_early_stopping(eval_metrics['eval_loss'])
    
    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "best_model"
            )
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            self.should_stop = True
    
    def _create_backup(self):
        """Create backup of current training state."""
        backup_dir = Path("backups") / getattr(self.config, 'experiment_name', 'default')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                f"backup_{timestamp}"
            )
            logging.info(f"Backup created at step {self.global_step}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
    
    def _save_training_summary(self, total_time: float):
        """Save comprehensive training summary with memory and offloading stats."""
        try:
            try:
                model_config = asdict(self.config)
            except:
                model_config = {
                    attr: getattr(self.config, attr) 
                    for attr in dir(self.config) 
                    if not attr.startswith('_') and not callable(getattr(self.config, attr))
                }
            
            final_memory_stats = self.cpu_offload_manager.get_memory_stats()
            
            summary = {
                'experiment_name': getattr(self.config, 'experiment_name', 'unknown'),
                'total_training_time_hours': total_time / 3600,
                'total_epochs': self.current_epoch,
                'total_steps': self.global_step,
                'final_metrics': {
                    'best_eval_loss': self.best_eval_loss,
                    'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                    'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
                },
                'precision_settings': {
                    'training_precision': self.training_precision,
                    'inference_precision': self.inference_precision
                },
                'cpu_offloading_stats': {
                    'peak_memory_mb': final_memory_stats['peak_memory_mb'],
                    'memory_saved_mb': final_memory_stats['offload_stats']['memory_saved_mb'],
                    'params_offloaded': final_memory_stats['offload_stats']['params_offloaded'],
                    'params_loaded': final_memory_stats['offload_stats']['params_loaded'],
                    'optimizer_states_offloaded': final_memory_stats['offload_stats']['optimizer_states_offloaded'],
                    'gradients_offloaded': final_memory_stats['offload_stats']['gradients_offloaded'],
                    'total_offload_time': final_memory_stats['offload_stats']['offload_time'],
                    'total_load_time': final_memory_stats['offload_stats']['load_time']
                },
                'memory_management': {
                    'gradient_checkpointing': self.gradient_checkpointing,
                    'memory_cleanup_interval': self.memory_cleanup_interval,
                    'offload_frequency': self.offload_frequency,
                    'enable_param_offload': self.cpu_offload_manager.enable_param_offload,
                    'enable_optimizer_offload': self.cpu_offload_manager.enable_optimizer_offload,
                    'enable_gradient_offload': self.cpu_offload_manager.enable_gradient_offload
                },
                'model_config': model_config,
                'health_summary': self.health_monitor.get_summary()
            }
            
            summary_path = Path(f"experiments/{getattr(self.config, 'experiment_name', 'default')}/training_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            logging.info(f"Training summary saved: {summary_path}")
        
        except Exception as e:
            logging.error(f"Failed to save training summary: {e}")
    
    def save_checkpoint(self, epoch_or_step: int, emergency: bool = False, final: bool = False):
        """Save checkpoint (delegated to checkpoint manager)."""
        suffix = "emergency" if emergency else ("final" if final else f"manual_{epoch_or_step}")
        return self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.current_epoch, self.metrics,
            suffix
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return current epoch."""
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 precision_override: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs) -> str:
        """Generate response with CPU offloading support and precision control."""
        self.model.eval()
        
        # Load model parameters for inference if they were offloaded
        self.cpu_offload_manager.load_model_parameters(self.model)
        
        gen_precision = precision_override or self.inference_precision
        if gen_precision == "dynamic":
            gen_precision = self.precision_manager.auto_select_precision(self.device, priority="speed")
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        temperature = temperature if temperature is not None else getattr(self.config, 'temperature', 0.7)
        top_k = top_k if top_k is not None else getattr(self.config, 'top_k', 50)
        top_p = top_p if top_p is not None else getattr(self.config, 'top_p', 0.9)
        
        generation_start_time = time.time()
        
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
            
            if len(input_tokens) >= self.config.seq_length:
                input_tokens = input_tokens[-(self.config.seq_length//2):]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            generated_tokens = []
            
            logging.debug(f"Starting generation with precision: {gen_precision}")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            for step in range(max_new_tokens):
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass with specified precision
                with self._get_autocast_context(precision=gen_precision, for_inference=True):
                    logits = self.model(input_ids)
                
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
                
                # Memory cleanup during long generations
                if step > 0 and step % 50 == 0:
                    self.cpu_offload_manager.memory_cleanup()
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - generation_start_time
            tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
            peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            
            logging.debug(f"Generation completed: {len(generated_tokens)} tokens in {generation_time:.2f}s "
                         f"({tokens_per_second:.1f} tok/s) using {gen_precision} precision, "
                         f"peak memory: {peak_memory:.1f}MB")
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed with {gen_precision} precision: {e}")
            return "I apologize, but I encountered an error while generating a response."
        
        finally:
            # Optionally offload parameters back after generation
            if getattr(self.config, 'offload_after_generation', False):
                self.cpu_offload_manager.offload_model_parameters(self.model)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics including CPU offloading."""
        base_stats = self.cpu_offload_manager.get_memory_stats()
        
        # Add training-specific stats
        base_stats.update({
            'gradient_checkpointing_enabled': self.gradient_checkpointing,
            'memory_cleanup_interval': self.memory_cleanup_interval,
            'offload_frequency': self.offload_frequency,
            'training_precision': self.training_precision,
            'inference_precision': self.inference_precision
        })
        
        return base_stats
    
    def optimize_memory_usage(self):
        """Optimize memory usage by offloading and cleaning up."""
        logging.info("Optimizing memory usage...")
        
        # Offload everything possible
        self.cpu_offload_manager.offload_model_parameters(self.model)
        self.cpu_offload_manager.offload_optimizer_states(self.optimizer)
        
        # Memory cleanup
        self.cpu_offload_manager.memory_cleanup()
        
        # Log results
        memory_stats = self.get_memory_stats()
        logging.info(f"Memory optimization complete:")
        logging.info(f"  Current GPU memory: {memory_stats['current_memory_mb']:.1f}MB")
        logging.info(f"  Offloaded params: {memory_stats['offloaded_params']}")
        logging.info(f"  Offloaded optimizer states: {memory_stats['offloaded_optimizer_states']}")
        logging.info(f"  Memory saved: {memory_stats['memory_saved_mb']:.1f}MB")
    
    def get_offloading_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for optimal offloading settings."""
        current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        gpu_total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 if torch.cuda.is_available() else 0
        
        memory_utilization = current_memory_mb / gpu_total_memory_mb if gpu_total_memory_mb > 0 else 0
        
        recommendations = {
            'current_memory_mb': current_memory_mb,
            'gpu_total_memory_mb': gpu_total_memory_mb,
            'memory_utilization': memory_utilization,
            'recommendations': {}
        }
        
        if memory_utilization > 0.9:
            recommendations['recommendations'] = {
                'priority': 'HIGH',
                'actions': [
                    'Enable parameter offloading',
                    'Enable gradient offloading', 
                    'Enable optimizer offloading',
                    'Enable gradient checkpointing',
                    'Reduce batch size',
                    'Lower offload threshold'
                ],
                'suggested_settings': {
                    'cpu_offload_params': True,
                    'cpu_offload_optimizer': True,
                    'cpu_offload_gradients': True,
                    'gradient_checkpointing': True,
                    'offload_threshold_mb': max(4000, current_memory_mb * 0.7),
                    'memory_cleanup_interval': 50
                }
            }
        elif memory_utilization > 0.7:
            recommendations['recommendations'] = {
                'priority': 'MEDIUM',
                'actions': [
                    'Enable optimizer offloading',
                    'Consider parameter offloading',
                    'Enable gradient checkpointing if not enabled'
                ],
                'suggested_settings': {
                    'cpu_offload_optimizer': True,
                    'cpu_offload_params': False,
                    'gradient_checkpointing': True,
                    'offload_threshold_mb': current_memory_mb * 0.8,
                    'memory_cleanup_interval': 100
                }
            }
        else:
            recommendations['recommendations'] = {
                'priority': 'LOW',
                'actions': [
                    'Current settings are likely sufficient',
                    'Monitor memory usage during training'
                ],
                'suggested_settings': {
                    'cpu_offload_optimizer': True,  # Still recommended for safety
                    'cpu_offload_params': False,
                    'gradient_checkpointing': False,
                    'offload_threshold_mb': gpu_total_memory_mb * 0.8,
                    'memory_cleanup_interval': 200
                }
            }
        
        return recommendations
    
    def apply_offloading_recommendations(self, recommendations: Optional[Dict] = None):
        """Apply offloading recommendations to optimize memory usage."""
        if recommendations is None:
            recommendations = self.get_offloading_recommendations()
        
        suggested_settings = recommendations.get('recommendations', {}).get('suggested_settings', {})
        
        if not suggested_settings:
            logging.info("No offloading changes recommended")
            return
        
        logging.info(f"Applying offloading recommendations (Priority: {recommendations['recommendations']['priority']}):")
        
        # Apply settings
        for setting, value in suggested_settings.items():
            if hasattr(self.cpu_offload_manager, setting):
                setattr(self.cpu_offload_manager, setting, value)
                logging.info(f"  {setting}: {value}")
            elif hasattr(self, setting):
                setattr(self, setting, value)
                logging.info(f"  {setting}: {value}")
            elif hasattr(self.config, setting):
                setattr(self.config, setting, value)
                logging.info(f"  {setting}: {value}")
        
        # Enable gradient checkpointing if recommended
        if suggested_settings.get('gradient_checkpointing') and hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            self.gradient_checkpointing = True
            logging.info("  Gradient checkpointing enabled")
        
        logging.info("Offloading recommendations applied successfully")
    
    def cleanup(self):
        """Clean up trainer resources and CPU offloading manager."""
        logging.info("Cleaning up trainer resources...")
        
        # Shutdown CPU offloading manager
        if hasattr(self, 'cpu_offload_manager'):
            self.cpu_offload_manager.shutdown()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Close logger if possible
        if hasattr(self.logger, 'close'):
            self.logger.close()
        
        logging.info("Trainer cleanup completed")