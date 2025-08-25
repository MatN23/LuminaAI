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
import weakref
from collections import deque

from core.dataset import create_dataloader
from monitoring.logger import TrainingHealthMonitor
from training.checkpoint import CheckpointManager


class MemoryOptimizedCache:
    """Memory-efficient cache with automatic cleanup."""
    
    def __init__(self, max_size: int = 10, auto_cleanup: bool = True):
        self.max_size = max_size
        self.auto_cleanup = auto_cleanup
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self._memory_threshold = 0.9  # Clear cache at 90% GPU memory
    
    def get(self, key: str):
        if key in self.cache:
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value):
        if self.auto_cleanup and torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if memory_usage > self._memory_threshold:
                self.clear()
        
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            if self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Int8QuantizationManager:
    """Manages INT8 quantization for inference with memory optimization."""
    
    def __init__(self, max_cached_models: int = 3):
        self.quantized_models = MemoryOptimizedCache(max_cached_models)
        self.calibration_data = None
        self.is_available = self._check_quantization_availability()
        self._temp_storage = {}  # Temporary storage during calibration
        
    def _check_quantization_availability(self) -> bool:
        """Check if INT8 quantization is available."""
        try:
            import torch.ao.quantization as quantization
            return True
        except ImportError:
            try:
                import torch.quantization as quantization
                return True
            except ImportError:
                logging.warning("PyTorch quantization not available")
                return False
    
    def prepare_for_quantization(self, model: nn.Module, example_inputs: torch.Tensor) -> nn.Module:
        """Prepare model for quantization with memory optimization."""
        if not self.is_available:
            return model
            
        try:
            # Clear cache before preparation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use newer quantization API if available
            try:
                import torch.ao.quantization as quantization
                from torch.ao.quantization import get_default_qconfig_mapping
                from torch.ao.quantization.quantize_fx import prepare_fx
                
                model.eval()
                qconfig_mapping = get_default_qconfig_mapping("x86")
                
                if example_inputs.is_cuda:
                    try:
                        qconfig_mapping = get_default_qconfig_mapping("cuda")
                    except:
                        qconfig_mapping = get_default_qconfig_mapping("x86")
                
                # Prepare with memory-conscious settings
                prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
                
                # Clear example inputs immediately
                del example_inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return prepared_model
                
            except ImportError:
                # Fallback to older API
                import torch.quantization as quantization
                
                model.eval()
                model.qconfig = quantization.get_default_qconfig('x86')
                prepared_model = quantization.prepare(model, inplace=False)
                
                # Clear temporary data
                del example_inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return prepared_model
                
        except Exception as e:
            logging.warning(f"Failed to prepare model for quantization: {e}")
            return model
    
    def calibrate_model(self, model: nn.Module, calibration_data: List[torch.Tensor], 
                       max_samples: int = 50) -> nn.Module:  # Reduced from 100
        """Calibrate model with memory-efficient batching."""
        if not self.is_available:
            return model
            
        try:
            model.eval()
            
            # Process in smaller batches to save memory
            batch_size = min(10, max_samples // 5)  # Process 10 samples at a time
            
            with torch.no_grad():
                for i in range(0, min(len(calibration_data), max_samples), batch_size):
                    batch_end = min(i + batch_size, len(calibration_data), max_samples)
                    
                    # Process batch
                    for j in range(i, batch_end):
                        # Move data to device only when needed
                        data = calibration_data[j]
                        if not data.is_cuda and torch.cuda.is_available():
                            data = data.cuda(non_blocking=True)
                        
                        _ = model(data)
                        
                        # Clear data immediately after use
                        del data
                    
                    # Clear cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # Check memory and break if getting full
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                        if memory_usage > 0.85:  # Stop if memory usage too high
                            logging.warning(f"Stopping calibration early due to memory constraints at sample {batch_end}")
                            break
            
            logging.info(f"Model calibrated with {min(len(calibration_data), max_samples)} samples")
            return model
            
        except Exception as e:
            logging.warning(f"Model calibration failed: {e}")
            return model
    
    def quantize_model(self, model: nn.Module, model_id: str = "default") -> nn.Module:
        """Convert calibrated model to quantized version with caching."""
        if not self.is_available:
            return model
            
        # Check cache first
        cached_model = self.quantized_models.get(model_id)
        if cached_model is not None:
            return cached_model
            
        try:
            # Clear memory before quantization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use newer quantization API if available
            try:
                import torch.ao.quantization as quantization
                from torch.ao.quantization.quantize_fx import convert_fx
                
                quantized_model = convert_fx(model)
                
            except ImportError:
                # Fallback to older API
                import torch.quantization as quantization
                
                quantized_model = quantization.convert(model, inplace=False)
            
            # Cache the quantized model
            self.quantized_models.put(model_id, quantized_model)
            
            logging.info(f"Model quantized to INT8 (ID: {model_id})")
            return quantized_model
            
        except Exception as e:
            logging.error(f"Model quantization failed: {e}")
            return model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size information."""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            total_size = param_size + buffer_size
            
            return {
                'param_size_mb': param_size / (1024 * 1024),
                'buffer_size_mb': buffer_size / (1024 * 1024),
                'total_size_mb': total_size / (1024 * 1024)
            }
        except Exception as e:
            logging.warning(f"Failed to calculate model size: {e}")
            return {'param_size_mb': 0, 'buffer_size_mb': 0, 'total_size_mb': 0}
    
    def cleanup(self):
        """Manual cleanup of cached models."""
        self.quantized_models.clear()
        self._temp_storage.clear()
        gc.collect()


class PrecisionManager:
    """Manages different precision types with memory considerations."""
    
    PRECISION_CONFIGS = {
        "fp32": {
            "dtype": torch.float32,
            "name": "Float32",
            "description": "Full precision (32-bit)",
            "memory_efficiency": 1.0,
            "speed_multiplier": 1.0,
            "numerical_stability": "excellent",
            "supported_devices": ["cpu", "cuda"],
            "quantized": False
        },
        "fp16": {
            "dtype": torch.float16,
            "name": "Float16",
            "description": "Half precision (16-bit)",
            "memory_efficiency": 2.0,
            "speed_multiplier": 1.5,
            "numerical_stability": "good",
            "supported_devices": ["cuda"],
            "quantized": False
        },
        "bf16": {
            "dtype": torch.bfloat16,
            "name": "BFloat16",
            "description": "Brain floating point (16-bit with extended range)",
            "memory_efficiency": 2.0,
            "speed_multiplier": 1.4,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"],
            "quantized": False
        },
        "mixed_fp16": {
            "dtype": torch.float16,
            "name": "Mixed Float16",
            "description": "Mixed precision with fp16 forward, fp32 gradients",
            "memory_efficiency": 1.8,
            "speed_multiplier": 1.6,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"],
            "quantized": False
        },
        "mixed_bf16": {
            "dtype": torch.bfloat16,
            "name": "Mixed BFloat16",
            "description": "Mixed precision with bf16 forward, fp32 gradients",
            "memory_efficiency": 1.8,
            "speed_multiplier": 1.5,
            "numerical_stability": "excellent",
            "supported_devices": ["cuda"],
            "quantized": False
        },
        "tf32": {
            "dtype": None,
            "name": "TensorFloat-32",
            "description": "NVIDIA Tensor Float (19-bit precision)",
            "memory_efficiency": 1.0,
            "speed_multiplier": 1.2,
            "numerical_stability": "very good",
            "supported_devices": ["cuda"],
            "quantized": False
        },
        "int8": {
            "dtype": torch.int8,
            "name": "INT8 Quantized",
            "description": "8-bit integer quantization for inference",
            "memory_efficiency": 4.0,
            "speed_multiplier": 2.0,
            "numerical_stability": "good",
            "supported_devices": ["cpu", "cuda"],
            "quantized": True,
            "inference_only": True
        },
        "dynamic": {
            "dtype": None,
            "name": "Dynamic",
            "description": "Automatically select best precision",
            "memory_efficiency": "variable",
            "speed_multiplier": "variable",
            "numerical_stability": "variable",
            "supported_devices": ["cpu", "cuda"],
            "quantized": False
        }
    }
    
    @classmethod
    def get_supported_precisions(cls, device: torch.device, 
                                include_quantization: bool = True) -> List[str]:
        """Get list of supported precisions for the given device."""
        device_type = device.type
        supported = []
        
        for precision, config in cls.PRECISION_CONFIGS.items():
            if device_type in config["supported_devices"]:
                if config.get("quantized", False) and not include_quantization:
                    continue
                    
                if precision in ["bf16", "mixed_bf16"]:
                    if device_type == "cuda" and torch.cuda.is_available():
                        try:
                            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
                            del test_tensor  # Clean up immediately
                            supported.append(precision)
                        except:
                            continue
                            
                elif precision == "tf32":
                    if device_type == "cuda" and torch.cuda.is_available():
                        if hasattr(torch.cuda, 'get_device_capability'):
                            capability = torch.cuda.get_device_capability(device.index or 0)
                            if capability[0] >= 8:
                                supported.append(precision)
                                
                elif precision == "int8":
                    try:
                        import torch.ao.quantization
                        supported.append(precision)
                    except ImportError:
                        try:
                            import torch.quantization
                            supported.append(precision)
                        except ImportError:
                            continue
                            
                else:
                    supported.append(precision)
        
        return supported
    
    @classmethod
    def get_precision_info(cls, precision: str) -> Dict[str, Any]:
        """Get detailed information about a precision type."""
        return cls.PRECISION_CONFIGS.get(precision, {})
    
    @classmethod
    def auto_select_precision(cls, device: torch.device, 
                            priority: str = "balanced",
                            include_quantization: bool = True,
                            memory_pressure: float = 0.0) -> str:
        """Auto-select precision considering memory pressure."""
        supported = cls.get_supported_precisions(device, include_quantization)
        
        if not supported:
            return "fp32"
        
        # Adjust priority based on memory pressure
        if memory_pressure > 0.8:
            priority = "memory"  # Force memory priority if under pressure
        
        if priority == "speed":
            priority_order = ["int8", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "fp32"]
        elif priority == "memory":
            priority_order = ["int8", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "fp32"]
        elif priority == "stability":
            priority_order = ["bf16", "mixed_bf16", "fp32", "mixed_fp16", "fp16", "int8"]
        else:  # balanced
            priority_order = ["mixed_bf16", "bf16", "int8", "mixed_fp16", "tf32", "fp16", "fp32"]
        
        for precision in priority_order:
            if precision in supported:
                return precision
        
        return "fp32"


class MemoryManager:
    """Centralized memory management for the trainer."""
    
    def __init__(self, device: torch.device, aggressive_cleanup: bool = True):
        self.device = device
        self.aggressive_cleanup = aggressive_cleanup
        self.memory_threshold = 0.85  # Trigger cleanup at 85%
        self.emergency_threshold = 0.95  # Emergency cleanup at 95%
        
        # Memory tracking
        self.peak_memory = 0
        self.allocation_history = deque(maxlen=100)
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'cached_mb': 0, 'usage_percent': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        cached = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
        usage_percent = allocated / total if total > 0 else 0
        
        return {
            'allocated_mb': allocated,
            'cached_mb': cached,
            'total_mb': total,
            'usage_percent': usage_percent
        }
    
    def check_memory_pressure(self) -> float:
        """Check current memory pressure (0-1)."""
        if not torch.cuda.is_available():
            return 0.0
        
        info = self.get_memory_info()
        return info['usage_percent']
    
    def cleanup_memory(self, aggressive: bool = None):
        """Perform memory cleanup."""
        if aggressive is None:
            aggressive = self.aggressive_cleanup
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if aggressive:
            gc.collect()
            
        # Log cleanup
        info = self.get_memory_info()
        logging.debug(f"Memory cleanup: {info['allocated_mb']:.1f}MB allocated, "
                     f"{info['usage_percent']:.1%} usage")
    
    def emergency_cleanup(self):
        """Emergency memory cleanup."""
        logging.warning("Performing emergency memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache aggressively
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Additional cleanup
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
    
    def auto_cleanup_if_needed(self):
        """Automatically cleanup if memory pressure is high."""
        pressure = self.check_memory_pressure()
        
        if pressure > self.emergency_threshold:
            self.emergency_cleanup()
        elif pressure > self.memory_threshold:
            self.cleanup_memory(aggressive=True)
    
    def context_cleanup(self):
        """Context manager for automatic cleanup."""
        return MemoryCleanupContext(self)


class MemoryCleanupContext:
    """Context manager for automatic memory cleanup."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_manager.cleanup_memory()


class EnhancedConversationTrainer:
    """Memory-optimized production trainer with comprehensive monitoring."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory manager first
        self.memory_manager = MemoryManager(self.device, aggressive_cleanup=True)
        
        # Initialize precision manager
        self.precision_manager = PrecisionManager()
        
        # Initialize INT8 quantization manager with memory optimization
        self.int8_manager = Int8QuantizationManager(max_cached_models=2)
        
        # GPU setup and memory management
        self._setup_gpu()
        
        # Move model to device with memory management
        with self.memory_manager.context_cleanup():
            self.model = self.model.to(self.device)
        
        # Training precision setup (INT8 not supported for training)
        self.training_precision = getattr(config, 'precision', 'mixed_bf16' if torch.cuda.is_available() else 'fp32')
        if self.training_precision == 'int8':
            logging.warning("INT8 precision not supported for training, falling back to mixed_bf16")
            self.training_precision = 'mixed_bf16' if torch.cuda.is_available() else 'fp32'
            
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Check PyTorch version for autocast compatibility
        self.torch_version = torch.__version__
        logging.info(f"PyTorch version: {self.torch_version}")
        
        # INFERENCE PRECISION SETUP INCLUDING INT8
        memory_pressure = self.memory_manager.check_memory_pressure()
        self.inference_precision = getattr(config, 'inference_precision', 'auto')
        self._setup_inference_precision(memory_pressure)
        
        # INT8 specific setup
        self.quantized_model = None
        self.int8_calibrated = False
        self.prepared_model = None
        
        # TF32 setup for modern GPUs
        self._setup_tf32()
        
        # Model compilation with memory check
        if memory_pressure < 0.7:  # Only compile if we have enough memory
            self._compile_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        self.last_backup_time = time.time()
        
        # Optimized metrics storage (use deque for memory efficiency)
        self.metrics = {
            'train_losses': deque(maxlen=1000),  # Keep only last 1000
            'eval_losses': deque(maxlen=100),    # Keep only last 100
            'learning_rates': deque(maxlen=1000),
            'gradient_norms': deque(maxlen=1000),
            'throughput': deque(maxlen=500),
            'epoch_times': deque(maxlen=20),
            'precision_performance': {},
            'quantization_metrics': {}
        }
        
        # Precision tracking (optimized)
        self.precision_stats = {
            'training_precision': self.training_precision,
            'inference_precision': self.inference_precision,
            'supported_precisions': self.precision_manager.get_supported_precisions(self.device),
            'precision_switches': deque(maxlen=50),  # Keep only recent switches
            'performance_metrics': {},
            'int8_available': self.int8_manager.is_available,
            'int8_calibrated': self.int8_calibrated
        }
        
        # Health monitoring
        try:
            self.health_monitor = TrainingHealthMonitor()
        except Exception as e:
            logging.warning(f"Failed to initialize health monitor: {e}")
            class SimpleHealthMonitor:
                def update(self, loss, grad_norm): pass
                def get_status(self): return "OK"
                def get_summary(self): return {}
            self.health_monitor = SimpleHealthMonitor()
        
        # Checkpoint management
        try:
            self.checkpoint_manager = CheckpointManager(config)
        except Exception as e:
            logging.error(f"Failed to initialize checkpoint manager: {e}")
            self.checkpoint_manager = self._create_simple_checkpoint_manager(config)
        
        # Optimizer and scheduler - after precision setup
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Initial memory cleanup
        self.memory_manager.cleanup_memory()
        
        # Log initialization
        logging.info(f"Trainer initialized on {self.device}")
        logging.info(f"Model parameters: {self._count_parameters():,}")
        logging.info(f"Training precision: {self.training_precision}")
        logging.info(f"Inference precision: {self.inference_precision}")
        logging.info(f"INT8 quantization available: {self.int8_manager.is_available}")
        logging.info(f"Supported precisions: {', '.join(self.precision_stats['supported_precisions'])}")
        self._log_memory_usage("Initial")
    
    def _create_simple_checkpoint_manager(self, config):
        """Create a simple checkpoint manager fallback."""
        class SimpleCheckpointManager:
            def __init__(self, config):
                self.config = config
                self.checkpoint_dir = Path("checkpoints")
                self.checkpoint_dir.mkdir(exist_ok=True)
            
            def save_checkpoint(self, model, optimizer, scheduler, global_step, epoch, metrics, suffix=""):
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}_{global_step}.pt"
                
                # Create checkpoint dict
                checkpoint_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'global_step': global_step,
                    'epoch': epoch,
                    'metrics': metrics
                }
                
                # Save with memory management
                torch.save(checkpoint_dict, checkpoint_path)
                
                # Clear the dict immediately
                del checkpoint_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logging.info(f"Checkpoint saved: {checkpoint_path}")
                return str(checkpoint_path)
            
            def load_checkpoint(self, path, model, optimizer=None, scheduler=None):
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                epoch = checkpoint.get('epoch', 0)
                
                # Clear checkpoint from memory
                del checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return epoch
            
            def get_resume_path(self):
                checkpoints = list(self.checkpoint_dir.glob("*.pt"))
                return str(max(checkpoints, key=lambda x: x.stat().st_mtime)) if checkpoints else None
        
        return SimpleCheckpointManager(config)

    def calibrate_int8_model(self, calibration_dataset, max_samples: int = 50):  # Reduced default
        """Calibrate model for INT8 quantization with memory optimization."""
        if not self.int8_manager.is_available:
            logging.warning("INT8 quantization not available")
            return False
            
        if self.int8_calibrated:
            logging.info("Model already calibrated for INT8")
            return True
            
        logging.info(f"Calibrating model for INT8 quantization with {max_samples} samples...")
        
        # Check memory before starting
        memory_pressure = self.memory_manager.check_memory_pressure()
        if memory_pressure > 0.8:
            logging.warning(f"High memory pressure ({memory_pressure:.1%}), reducing calibration samples")
            max_samples = min(max_samples, 25)
        
        try:
            # Memory cleanup before calibration
            with self.memory_manager.context_cleanup():
                # Create calibration dataloader with smaller batch size
                original_batch_size = getattr(self.config, 'batch_size', 8)
                self.config.batch_size = min(2, original_batch_size)  # Use tiny batches
                
                calibration_dataloader = create_dataloader(
                    calibration_dataset, self.config, shuffle=False
                )
                
                # Restore original batch size
                self.config.batch_size = original_batch_size
                
                # Prepare example input
                first_batch = next(iter(calibration_dataloader))
                example_input = first_batch['input_ids'][:1].to(self.device)
                
                # Prepare model for quantization
                self.model.eval()
                self.prepared_model = self.int8_manager.prepare_for_quantization(
                    self.model, example_input
                )
                
                # Clear example input
                del example_input, first_batch
                
                # Collect calibration data with memory management
                calibration_data = []
                sample_count = 0
                
                for batch_idx, batch in enumerate(calibration_dataloader):
                    if sample_count >= max_samples:
                        break
                    
                    # Memory check
                    if batch_idx % 5 == 0:  # Check every 5 batches
                        self.memory_manager.auto_cleanup_if_needed()
                    
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) 
                            for k, v in batch.items()}
                    
                    if batch['input_ids'].numel() > 0:
                        # Process samples individually for better memory control
                        for i in range(min(batch['input_ids'].size(0), max_samples - sample_count)):
                            input_sample = batch['input_ids'][i:i+1]
                            calibration_data.append(input_sample)
                            
                            # Run calibration step
                            try:
                                with torch.no_grad():
                                    if 'attention_mask' in batch:
                                        mask_sample = batch['attention_mask'][i:i+1]
                                        _ = self.prepared_model(input_sample, mask_sample)
                                        del mask_sample
                                    else:
                                        _ = self.prepared_model(input_sample)
                                    
                                sample_count += 1
                                
                                # Clear sample immediately
                                del input_sample
                                
                            except Exception as e:
                                logging.debug(f"Calibration step {sample_count} failed: {e}")
                                continue
                    
                    # Clear batch
                    del batch
                    
                    # Emergency memory check
                    if self.memory_manager.check_memory_pressure() > 0.9:
                        logging.warning("Memory pressure too high, stopping calibration early")
                        break
                
                if calibration_data:
                    # Perform final calibration
                    self.prepared_model = self.int8_manager.calibrate_model(
                        self.prepared_model, calibration_data, len(calibration_data)
                    )
                    
                    self.int8_calibrated = True
                    self.precision_stats['int8_calibrated'] = True
                    
                    # Clear calibration data immediately
                    del calibration_data
                    
                    logging.info(f"INT8 calibration completed with {sample_count} samples")
                    return True
                else:
                    logging.error("No valid calibration data found")
                    return False
                    
        except Exception as e:
            logging.error(f"INT8 calibration failed: {e}")
            return False
        finally:
            # Always cleanup after calibration
            self.memory_manager.cleanup_memory(aggressive=True)
    
    def _get_int8_model(self, model_id: str = "default") -> nn.Module:
        """Get or create INT8 quantized model with memory management."""
        if not self.int8_manager.is_available:
            return self.model
            
        if not self.int8_calibrated:
            logging.warning("Model not calibrated for INT8, using original model")
            return self.model
            
        try:
            # Check cache first
            cached_model = self.int8_manager.quantized_models.get(model_id)
            if cached_model is not None:
                return cached_model
            
            # Create quantized model with memory management
            with self.memory_manager.context_cleanup():
                quantized_model = self.int8_manager.quantize_model(
                    self.prepared_model, model_id
                )
                
                # Log size comparison
                original_size = self.int8_manager.get_model_size(self.model)
                quantized_size = self.int8_manager.get_model_size(quantized_model)
                
                size_reduction = (1 - quantized_size['total_size_mb'] / max(original_size['total_size_mb'], 0.001)) * 100
                
                logging.info(f"INT8 model created: {original_size['total_size_mb']:.1f}MB → "
                            f"{quantized_size['total_size_mb']:.1f}MB ({size_reduction:.1f}% reduction)")
                
                # Store quantization metrics efficiently
                if 'int8' not in self.metrics['quantization_metrics']:
                    self.metrics['quantization_metrics']['int8'] = deque(maxlen=10)
                    
                self.metrics['quantization_metrics']['int8'].append({
                    'original_size_mb': original_size['total_size_mb'],
                    'quantized_size_mb': quantized_size['total_size_mb'],
                    'size_reduction_percent': size_reduction,
                    'timestamp': time.time()
                })
                
                return quantized_model
            
        except Exception as e:
            logging.error(f"Failed to create INT8 model: {e}")
            return self.model
    
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
    
    def _setup_inference_precision(self, memory_pressure: float = 0.0):
        """Setup inference precision with memory considerations."""
        if self.inference_precision == "auto":
            self.inference_precision = self.precision_manager.auto_select_precision(
                self.device, priority="balanced", include_quantization=True,
                memory_pressure=memory_pressure
            )
            logging.info(f"Auto-selected inference precision: {self.inference_precision}")
        elif self.inference_precision == "dynamic":
            logging.info("Dynamic precision enabled - will select best precision per inference")
        
        # Validate precision
        supported = self.precision_manager.get_supported_precisions(self.device, include_quantization=True)
        if self.inference_precision not in supported and self.inference_precision != "dynamic":
            logging.warning(f"Inference precision {self.inference_precision} not supported, falling back to auto-selection")
            self.inference_precision = self.precision_manager.auto_select_precision(self.device, memory_pressure=memory_pressure)
        
        # Log precision info
        if self.inference_precision != "dynamic":
            precision_info = self.precision_manager.get_precision_info(self.inference_precision)
            if precision_info:
                logging.info(f"Inference precision info: {precision_info['name']} - {precision_info['description']}")
                if precision_info.get('quantized', False):
                    logging.info("INT8 quantization will be used for inference")
    
    def set_inference_precision(self, precision: str):
        """Dynamically change inference precision with memory validation."""
        old_precision = self.inference_precision
        
        # Check memory pressure before switching
        memory_pressure = self.memory_manager.check_memory_pressure()
        
        # Validate precision with memory considerations
        if precision == "auto":
            precision = self.precision_manager.auto_select_precision(
                self.device, include_quantization=True, memory_pressure=memory_pressure
            )
        elif precision not in ["dynamic"] + self.precision_manager.get_supported_precisions(self.device, include_quantization=True):
            logging.error(f"Precision {precision} not supported on {self.device}")
            return False
        
        self.inference_precision = precision
        
        # Special handling for INT8
        if precision == "int8" and not self.int8_calibrated:
            logging.warning("INT8 precision selected but model not calibrated. Run calibrate_int8_model() first for optimal performance.")
        
        # Update TF32 settings if needed
        if precision == "tf32" or old_precision == "tf32":
            self._setup_tf32()
        
        # Track precision change efficiently
        self.precision_stats['precision_switches'].append({
            'timestamp': time.time(),
            'old_precision': old_precision,
            'new_precision': precision,
            'step': self.global_step,
            'memory_pressure': memory_pressure
        })
        
        logging.info(f"Inference precision changed: {old_precision} → {precision}")
        return True
    
    def _count_parameters(self):
        """Count model parameters."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except:
            return 0
    
    def _setup_gpu(self):
        """Setup GPU with memory-optimized configuration."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Conservative memory fraction for stability
            memory_fraction = getattr(self.config, 'memory_fraction', 0.8)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Log GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
            logging.info(f"Memory fraction set to: {memory_fraction:.1%}")
            
            # Log compute capability for optimization features
            if hasattr(torch.cuda, 'get_device_capability'):
                capability = torch.cuda.get_device_capability(0)
                logging.info(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
                if capability[0] >= 8:
                    logging.info("TF32 and advanced optimizations available (Ampere+ GPU)")
        else:
            logging.warning("CUDA not available, using CPU")
    
    def _log_memory_usage(self, phase: str):
        """Log current memory usage with details."""
        info = self.memory_manager.get_memory_info()
        logging.info(f"{phase} - GPU Memory: {info['allocated_mb']:.1f}MB allocated "
                    f"({info['usage_percent']:.1%} of {info['total_mb']:.0f}MB total)")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create memory-efficient optimizer with parameter grouping."""
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
            # Use fused optimizer if available for better memory efficiency
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available() and hasattr(torch.optim.AdamW, 'fused')
            )
        except Exception:
            # Fallback to standard AdamW
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _compile_model(self):
        """Compile model with memory-aware error handling."""
        if not getattr(self.config, 'compile', False) or not hasattr(torch, 'compile'):
            return
            
        try:
            logging.info("Compiling model...")
            
            # Check memory before compilation
            memory_info = self.memory_manager.get_memory_info()
            if memory_info['usage_percent'] > 0.7:
                logging.warning("High memory usage, skipping model compilation")
                return
            
            # Compile with memory-friendly settings
            self.model = torch.compile(
                self.model, 
                mode='default',
                dynamic=True  # Allow dynamic shapes for better memory management
            )
            logging.info("Model compiled successfully")
            
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")
    
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
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with memory-efficient implementation."""
        # Use view instead of reshape for better memory efficiency
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
        total_weight = mask.sum().clamp(min=1)
        final_loss = weighted_loss.sum() / total_weight
        
        # Compute additional metrics
        raw_loss = (loss * mask).sum() / total_weight
        perplexity = torch.exp(raw_loss.clamp(max=10))  # Clamp to prevent overflow
        
        # Clear intermediate tensors
        del flat_logits, flat_labels, flat_weights, mask, weighted_loss
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': total_weight
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Memory-optimized training step with automatic cleanup."""
        self.model.train()
        
        # Move batch to device with non-blocking transfer
        with self.memory_manager.context_cleanup():
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Skip empty batches
            if batch['input_ids'].numel() == 0:
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            
            # Check memory pressure before forward pass
            if self.memory_manager.check_memory_pressure() > 0.9:
                self.memory_manager.emergency_cleanup()
            
            # Forward pass with training precision (never INT8 for training)
            with self._get_autocast_context(for_inference=False):
                try:
                    logits = self.model(batch['input_ids'], batch['attention_mask'])
                    loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                    loss = loss_dict['loss'] / getattr(self.config, 'gradient_accumulation_steps', 1)
                    
                    # Clear logits immediately to save memory
                    del logits
                    
                except torch.cuda.OutOfMemoryError:
                    logging.warning("CUDA OOM in forward pass, performing emergency cleanup")
                    self.memory_manager.emergency_cleanup()
                    return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            
            # Check for valid loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning("Invalid loss detected, skipping batch")
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            
            # Backward pass with memory management
            try:
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except torch.cuda.OutOfMemoryError:
                logging.warning("CUDA OOM in backward pass, performing emergency cleanup")
                self.memory_manager.emergency_cleanup()
                return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
            
            # Return metrics
            return {
                'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
                'raw_loss': loss_dict['raw_loss'].item(),
                'perplexity': loss_dict['perplexity'].item(),
                'valid_tokens': loss_dict['valid_tokens'].item()
            }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Memory-optimized optimizer step with gradient monitoring."""
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
        
        # Optimizer step with memory management
        try:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        except torch.cuda.OutOfMemoryError:
            logging.warning("CUDA OOM in optimizer step")
            self.memory_manager.emergency_cleanup()
        
        # Clear gradients with memory-efficient option
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        # Periodic memory cleanup
        if self.global_step % 50 == 0:  # Every 50 steps
            self.memory_manager.auto_cleanup_if_needed()
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 50,  # Reduced from 100
                 precision_override: Optional[str] = None) -> Dict[str, float]:
        """Memory-optimized evaluation with precision control."""
        # Determine evaluation precision considering memory pressure
        eval_precision = precision_override or self.inference_precision
        memory_pressure = self.memory_manager.check_memory_pressure()
        
        # Adjust evaluation based on memory pressure
        if memory_pressure > 0.8:
            max_batches = min(max_batches, 25)  # Reduce batches under memory pressure
            if eval_precision == "fp32":
                eval_precision = "fp16" if "fp16" in self.precision_manager.get_supported_precisions(self.device) else eval_precision
        
        # Get appropriate model for evaluation
        eval_model = self._get_model_for_inference(eval_precision)
        eval_model.eval()
        
        # Create smaller dataloader for evaluation
        original_batch_size = getattr(self.config, 'batch_size', 8)
        self.config.batch_size = max(1, original_batch_size // 2)  # Use smaller batches
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        # Restore original batch size
        self.config.batch_size = original_batch_size
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Track memory usage for this precision
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with self.memory_manager.context_cleanup():
            for batch_idx, batch in enumerate(eval_dataloader):
                if batch_idx >= max_batches:
                    break
                
                # Memory check every 10 batches
                if batch_idx % 10 == 0:
                    self.memory_manager.auto_cleanup_if_needed()
                
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                if batch['input_ids'].numel() == 0:
                    continue
                
                try:
                    # For INT8, we don't use autocast
                    if eval_precision == "int8":
                        logits = eval_model(batch['input_ids'], batch['attention_mask'])
                        loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                    else:
                        with self._get_autocast_context(precision=eval_precision, for_inference=True):
                            logits = eval_model(batch['input_ids'], batch['attention_mask'])
                            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                    
                    # Clear logits immediately
                    del logits
                    
                    if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                        total_loss += loss_dict['loss'].item()
                        total_raw_loss += loss_dict['raw_loss'].item()
                        total_tokens += loss_dict['valid_tokens'].item()
                        num_batches += 1
                
                except torch.cuda.OutOfMemoryError:
                    logging.warning(f"CUDA OOM in evaluation batch {batch_idx}, skipping")
                    self.memory_manager.emergency_cleanup()
                    continue
                
                # Clear batch immediately
                del batch
        
        eval_time = time.time() - eval_start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_time': eval_time,
                'eval_throughput': 0.0,
                'eval_precision': eval_precision,
                'eval_peak_memory_mb': peak_memory,
                'eval_int8_used': eval_precision == "int8"
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        perplexity = math.exp(min(avg_raw_loss, 10))
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        # Store precision performance metrics efficiently
        if eval_precision not in self.metrics['precision_performance']:
            self.metrics['precision_performance'][eval_precision] = deque(maxlen=20)
        
        performance_entry = {
            'throughput': throughput,
            'memory': peak_memory,
            'timestamp': time.time()
        }
        
        # Add INT8 specific metrics
        if eval_precision == "int8" and self.metrics['quantization_metrics'].get('int8'):
            if len(self.metrics['quantization_metrics']['int8']) > 0:
                latest_quant = self.metrics['quantization_metrics']['int8'][-1]
                performance_entry['size_reduction_percent'] = latest_quant['size_reduction_percent']
        
        self.metrics['precision_performance'][eval_precision].append(performance_entry)
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_time': eval_time,
            'eval_throughput': throughput,
            'eval_precision': eval_precision,
            'eval_peak_memory_mb': peak_memory,
            'eval_int8_used': eval_precision == "int8"
        }
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 precision_override: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs) -> str:
        """Memory-optimized generation with comprehensive precision control."""
        # Determine generation precision considering memory pressure
        memory_pressure = self.memory_manager.check_memory_pressure()
        gen_precision = precision_override or self.inference_precision
        
        if gen_precision == "dynamic":
            gen_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed", include_quantization=True, 
                memory_pressure=memory_pressure
            )
        
        # Validate precision
        supported_precisions = self.precision_manager.get_supported_precisions(
            self.device, include_quantization=True
        )
        if gen_precision not in supported_precisions:
            logging.warning(f"Precision {gen_precision} not supported, falling back to fp32")
            gen_precision = "fp32"
        
        # Get appropriate model for generation
        gen_model = self._get_model_for_inference(gen_precision)
        gen_model.eval()
        
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'max_new_tokens', 256)  # Reduced from 512
            
        # Adjust max_new_tokens based on memory pressure
        if memory_pressure > 0.8:
            max_new_tokens = min(max_new_tokens, 128)
        
        # Use provided or default generation parameters
        temperature = temperature if temperature is not None else getattr(self.config, 'temperature', 0.7)
        top_k = top_k if top_k is not None else getattr(self.config, 'top_k', 50)
        top_p = top_p if top_p is not None else getattr(self.config, 'top_p', 0.9)
        
        generation_start_time = time.time()
        
        try:
            with self.memory_manager.context_cleanup():
                # Create conversation format
                conversation = {
                    'messages': [{'role': 'user', 'content': prompt}]
                }
                
                # Encode input
                input_tokens = self.tokenizer.encode_conversation(conversation)
                
                # Add assistant start tokens
                input_tokens.extend([
                    self.tokenizer.special_tokens["<|im_start|>"],
                    self.tokenizer.special_tokens["<|assistant|>"]
                ])
                
                # Ensure reasonable context length with memory considerations
                max_context = min(self.config.seq_length, 2048)  # Limit context for memory
                if len(input_tokens) >= max_context:
                    input_tokens = input_tokens[-max_context//2:]
                
                input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
                
                # Clear input_tokens list
                del input_tokens
                
                # Generation loop with memory management
                generated_tokens = []
                
                logging.debug(f"Starting generation with precision: {gen_precision}")
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                for step in range(max_new_tokens):
                    # Memory check every 20 steps
                    if step % 20 == 0 and step > 0:
                        if self.memory_manager.check_memory_pressure() > 0.95:
                            logging.warning("Memory pressure too high, stopping generation early")
                            break
                    
                    # Check sequence length
                    if input_ids.size(1) >= max_context:
                        # Truncate from the beginning, keeping recent context
                        input_ids = input_ids[:, -max_context//2:]
                    
                    try:
                        # Forward pass with specified precision
                        if gen_precision == "int8":
                            logits = gen_model(input_ids)
                        else:
                            with self._get_autocast_context(precision=gen_precision, for_inference=True):
                                logits = gen_model(input_ids)
                        
                        # Get next token logits
                        next_token_logits = logits[0, -1, :] / temperature
                        
                        # Clear full logits to save memory
                        del logits
                        
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
                        
                        # Clear intermediate tensors
                        del next_token_logits, probs
                        
                        # Check for stop tokens
                        if next_token.item() == self.tokenizer.special_tokens["<|im_end|>"]:
                            break
                        
                        generated_tokens.append(next_token.item())
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                        
                        # Clear next_token
                        del next_token
                        
                    except torch.cuda.OutOfMemoryError:
                        logging.warning(f"CUDA OOM at generation step {step}, stopping early")
                        self.memory_manager.emergency_cleanup()
                        break
                    except Exception as e:
                        logging.error(f"Generation error at step {step}: {e}")
                        break
                
                # Clear input_ids
                del input_ids
                
                # Decode response
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                generation_time = time.time() - generation_start_time
                tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
                peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                
                logging.debug(f"Generation completed: {len(generated_tokens)} tokens in {generation_time:.2f}s "
                             f"({tokens_per_second:.1f} tok/s) using {gen_precision} precision, "
                             f"peak memory: {peak_memory:.1f}MB")
                
                # Record performance metrics efficiently
                if gen_precision not in self.precision_stats['performance_metrics']:
                    self.precision_stats['performance_metrics'][gen_precision] = deque(maxlen=50)
                
                perf_entry = {
                    'tokens_per_second': tokens_per_second,
                    'peak_memory_mb': peak_memory,
                    'generation_time': generation_time,
                    'tokens_generated': len(generated_tokens),
                    'timestamp': time.time()
                }
                
                # Add INT8 specific information
                if gen_precision == "int8" and self.metrics['quantization_metrics'].get('int8'):
                    if len(self.metrics['quantization_metrics']['int8']) > 0:
                        latest_quant = self.metrics['quantization_metrics']['int8'][-1]
                        perf_entry['model_size_mb'] = latest_quant['quantized_size_mb']
                        perf_entry['size_reduction_percent'] = latest_quant['size_reduction_percent']
                
                self.precision_stats['performance_metrics'][gen_precision].append(perf_entry)
                
                return response.strip()
                
        except Exception as e:
            logging.error(f"Generation failed with {gen_precision} precision: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def cleanup_resources(self):
        """Comprehensive resource cleanup."""
        logging.info("Cleaning up trainer resources...")
        
        # Clear quantization manager cache
        self.int8_manager.cleanup()
        
        # Clear metrics history (keep only recent data)
        for key in ['train_losses', 'gradient_norms', 'learning_rates', 'throughput']:
            if hasattr(self.metrics[key], 'clear'):
                # Keep only last 100 items
                if len(self.metrics[key]) > 100:
                    recent_items = list(self.metrics[key])[-100:]
                    self.metrics[key].clear()
                    self.metrics[key].extend(recent_items)
        
        # Clear precision performance history
        for precision in self.metrics['precision_performance']:
            if len(self.metrics['precision_performance'][precision]) > 20:
                recent_items = list(self.metrics['precision_performance'][precision])[-20:]
                self.metrics['precision_performance'][precision].clear()
                self.metrics['precision_performance'][precision].extend(recent_items)
        
        # Force memory cleanup
        self.memory_manager.emergency_cleanup()
        
        logging.info("Resource cleanup completed")
    
    def train(self, train_dataset, eval_dataset=None, num_epochs: int = None, 
              resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """Memory-optimized training loop with comprehensive monitoring."""
        if num_epochs is None:
            num_epochs = getattr(self.config, 'num_epochs', 3)
        
        logging.info(f"Starting memory-optimized training for {num_epochs} epochs")
        logging.info(f"Training precision: {self.training_precision}")
        logging.info(f"Inference precision: {self.inference_precision}")
        
        # Initial memory check
        initial_memory = self.memory_manager.get_memory_info()
        logging.info(f"Initial memory usage: {initial_memory['usage_percent']:.1%}")
        
        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_path = self.checkpoint_manager.get_resume_path()
            if checkpoint_path:
                logging.info(f"Resuming from checkpoint: {checkpoint_path}")
                start_epoch = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path, self.model, self.optimizer, self.scheduler
                )
        
        # Create dataloader with memory-conscious batch size
        memory_pressure = self.memory_manager.check_memory_pressure()
        if memory_pressure > 0.7:
            # Reduce batch size under memory pressure
            original_batch_size = getattr(self.config, 'batch_size', 8)
            self.config.batch_size = max(1, original_batch_size // 2)
            logging.info(f"Reduced batch size from {original_batch_size} to {self.config.batch_size} due to memory pressure")
        
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        # Calculate total steps and setup scheduler
        steps_per_epoch = len(train_dataloader) // getattr(self.config, 'gradient_accumulation_steps', 1)
        total_steps = steps_per_epoch * num_epochs
        self._setup_scheduler(total_steps)
        
        # Training state
        best_eval_loss = float('inf')
        patience_counter = 0
        training_start_time = time.time()
        
        logging.info(f"Training setup complete:")
        logging.info(f"  Steps per epoch: {steps_per_epoch}")
        logging.info(f"  Total steps: {total_steps}")
        logging.info(f"  Gradient accumulation: {getattr(self.config, 'gradient_accumulation_steps', 1)}")
        
        try:
            # Training loop with memory monitoring
            for epoch in range(start_epoch, num_epochs):
                if self.should_stop:
                    logging.info("Early stopping triggered")
                    break
                
                epoch_start_time = time.time()
                self.current_epoch = epoch
                
                # Pre-epoch memory cleanup
                if epoch > 0:  # Skip for first epoch
                    self.memory_manager.cleanup_memory(aggressive=True)
                
                # Training epoch with memory monitoring
                try:
                    train_metrics = self._train_epoch(train_dataloader, epoch)
                except torch.cuda.OutOfMemoryError:
                    logging.error("CUDA OOM during training epoch, performing emergency cleanup")
                    self.memory_manager.emergency_cleanup()
                    # Try to continue with reduced functionality
                    continue
                
                # Evaluation with memory management
                eval_metrics = {}
                if eval_dataset is not None:
                    try:
                        # Reduce evaluation scope under memory pressure
                        eval_batches = 50 if self.memory_manager.check_memory_pressure() < 0.8 else 25
                        eval_metrics = self.evaluate(eval_dataset, max_batches=eval_batches)
                        
                        # Early stopping check
                        eval_loss = eval_metrics.get('eval_loss', float('inf'))
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            patience_counter = 0
                            
                            # Save best model with memory management
                            try:
                                self.checkpoint_manager.save_checkpoint(
                                    self.model, self.optimizer, self.scheduler,
                                    self.global_step, epoch, {**train_metrics, **eval_metrics},
                                    suffix="best"
                                )
                            except Exception as e:
                                logging.warning(f"Failed to save best checkpoint: {e}")
                        else:
                            patience_counter += 1
                            
                        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 5)
                        if patience_counter >= early_stopping_patience:
                            logging.info(f"Early stopping after {patience_counter} epochs without improvement")
                            self.should_stop = True
                    
                    except torch.cuda.OutOfMemoryError:
                        logging.warning("CUDA OOM during evaluation, skipping this epoch's evaluation")
                        self.memory_manager.emergency_cleanup()
                        eval_metrics = {'eval_loss': float('inf'), 'eval_perplexity': float('inf')}
                
                # Log epoch results
                epoch_time = time.time() - epoch_start_time
                self.metrics['epoch_times'].append(epoch_time)
                
                # Memory usage logging
                memory_info = self.memory_manager.get_memory_info()
                
                logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
                logging.info(f"  Train loss: {train_metrics.get('avg_loss', 0):.4f}")
                logging.info(f"  Train perplexity: {train_metrics.get('avg_perplexity', 0):.2f}")
                if eval_metrics:
                    logging.info(f"  Eval loss: {eval_metrics.get('eval_loss', 0):.4f}")
                    logging.info(f"  Eval perplexity: {eval_metrics.get('eval_perplexity', 0):.2f}")
                logging.info(f"  Memory usage: {memory_info['usage_percent']:.1%}")
                
                # Regular checkpoint with memory management
                if (epoch + 1) % getattr(self.config, 'save_every', 1) == 0:
                    try:
                        self.checkpoint_manager.save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, {**train_metrics, **eval_metrics},
                            suffix=f"epoch_{epoch+1}"
                        )
                    except Exception as e:
                        logging.warning(f"Failed to save epoch checkpoint: {e}")
                
                # Periodic resource cleanup
                if (epoch + 1) % 2 == 0:  # Every 2 epochs
                    self.cleanup_resources()
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            # Attempt to save emergency checkpoint
            try:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, self.current_epoch, {},
                    suffix="emergency"
                )
            except:
                pass
            raise
        
        finally:
            # Final cleanup
            self.cleanup_resources()
        
        total_training_time = time.time() - training_start_time
        
        # Final results
        results = {
            'training_completed': True,
            'total_epochs': num_epochs,
            'total_training_time': total_training_time,
            'total_steps': self.global_step,
            'best_eval_loss': best_eval_loss,
            'final_train_metrics': train_metrics,
            'final_eval_metrics': eval_metrics,
            'precision_stats': self.precision_stats,
            'health_summary': self.health_monitor.get_summary(),
            'training_precision': self.training_precision,
            'inference_precision': self.inference_precision,
            'memory_efficiency': {
                'initial_memory_mb': initial_memory['allocated_mb'],
                'final_memory_mb': self.memory_manager.get_memory_info()['allocated_mb'],
                'peak_memory_usage': max(initial_memory['usage_percent'], 
                                       self.memory_manager.get_memory_info()['usage_percent'])
            }
        }
        
        logging.info(f"Training completed in {total_training_time:.2f}s")
        logging.info(f"Best evaluation loss: {best_eval_loss:.4f}")
        logging.info(f"Peak memory usage: {results['memory_efficiency']['peak_memory_usage']:.1%}")
        
        return results
    
    def _train_epoch(self, train_dataloader, epoch: int) -> Dict[str, float]:
        """Memory-optimized training epoch with comprehensive monitoring."""
        self.model.train()
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_perplexity = 0.0
        total_tokens = 0
        batch_count = 0
        
        epoch_start_time = time.time()
        
        # Memory monitoring during epoch
        memory_warnings = 0
        max_memory_warnings = 5  # Stop epoch if too many memory warnings
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            # Memory pressure check
            if batch_idx % 10 == 0:  # Check every 10 batches
                memory_pressure = self.memory_manager.check_memory_pressure()
                if memory_pressure > 0.9:
                    memory_warnings += 1
                    logging.warning(f"High memory pressure: {memory_pressure:.1%}")
                    self.memory_manager.emergency_cleanup()
                    
                    if memory_warnings >= max_memory_warnings:
                        logging.error("Too many memory warnings, stopping epoch early")
                        break
            
            # Training step with memory management
            try:
                step_metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"CUDA OOM at batch {batch_idx}, skipping")
                self.memory_manager.emergency_cleanup()
                continue
            
            # Accumulate gradients
            if (batch_idx + 1) % getattr(self.config, 'gradient_accumulation_steps', 1) == 0:
                # Optimizer step
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                # Update health monitor
                self.health_monitor.update(
                    step_metrics.get('loss', 0),
                    opt_metrics.get('grad_norm', 0)
                )
                
                # Record metrics efficiently
                self.metrics['train_losses'].append(step_metrics.get('loss', 0))
                self.metrics['gradient_norms'].append(opt_metrics.get('grad_norm', 0))
                self.metrics['learning_rates'].append(opt_metrics.get('lr', 0))
                
                # Calculate throughput
                batch_time = time.time() - batch_start_time
                throughput = step_metrics.get('valid_tokens', 0) / batch_time if batch_time > 0 else 0
                self.metrics['throughput'].append(throughput)
            
            # Accumulate epoch metrics
            if step_metrics.get('valid_tokens', 0) > 0:
                total_loss += step_metrics.get('loss', 0)
                total_raw_loss += step_metrics.get('raw_loss', 0)
                total_perplexity += step_metrics.get('perplexity', 0)
                total_tokens += step_metrics.get('valid_tokens', 0)
                batch_count += 1
            
            # Periodic logging with memory info
            if self.global_step % getattr(self.config, 'log_every', 100) == 0:
                avg_loss = total_loss / max(batch_count, 1)
                avg_throughput = np.mean(list(self.metrics['throughput'])[-10:]) if self.metrics['throughput'] else 0
                memory_info = self.memory_manager.get_memory_info()
                
                logging.info(f"Step {self.global_step}: loss={avg_loss:.4f}, "
                           f"throughput={avg_throughput:.0f} tok/s, "
                           f"memory={memory_info['usage_percent']:.1%}, "
                           f"health={self.health_monitor.get_status()}")
        
        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        
        return {
            'avg_loss': total_loss / max(batch_count, 1),
            'avg_raw_loss': total_raw_loss / max(batch_count, 1),
            'avg_perplexity': total_perplexity / max(batch_count, 1),
            'total_tokens': total_tokens,
            'batch_count': batch_count,
            'epoch_time': epoch_time,
            'tokens_per_second': total_tokens / epoch_time if epoch_time > 0 else 0,
            'memory_warnings': memory_warnings
        }
    
    def _get_autocast_context(self, precision: Optional[str] = None, for_inference: bool = False):
        """Get autocast context with memory-efficient precision support."""
        # Determine target precision
        if precision is None:
            target_precision = self.inference_precision if for_inference else self.training_precision
        else:
            target_precision = precision
        
        # Handle dynamic precision with memory consideration
        if target_precision == "dynamic":
            memory_pressure = self.memory_manager.check_memory_pressure()
            target_precision = self.precision_manager.auto_select_precision(
                self.device, 
                priority="memory" if memory_pressure > 0.8 else ("speed" if for_inference else "balanced"),
                include_quantization=for_inference,
                memory_pressure=memory_pressure
            )
        
        # INT8 doesn't use autocast (handled separately)
        if target_precision == "int8":
            return nullcontext()
        
        # Handle different precision types
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
    
    def _get_model_for_inference(self, precision: Optional[str] = None) -> nn.Module:
        """Get the appropriate model for inference based on precision."""
        target_precision = precision or self.inference_precision
        
        if target_precision == "dynamic":
            memory_pressure = self.memory_manager.check_memory_pressure()
            target_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed", include_quantization=True,
                memory_pressure=memory_pressure
            )
        
        # Return INT8 quantized model if requested
        if target_precision == "int8":
            return self._get_int8_model()
        
        # Return original model for all other precisions
        return self.model
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        try:
            self.cleanup_resources()
        except:
            pass