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

from core.dataset import create_dataloader
from monitoring.logger import TrainingHealthMonitor
from training.checkpoint import CheckpointManager


class Int8QuantizationManager:
    """Manages INT8 quantization for inference."""
    
    def __init__(self):
        self.quantized_models = {}  # Cache for quantized models
        self.calibration_data = None
        self.is_available = self._check_quantization_availability()
        
    def _check_quantization_availability(self) -> bool:
        """Check if INT8 quantization is available."""
        try:
            # Check if quantization modules are available
            import torch.ao.quantization as quantization
            return True
        except ImportError:
            try:
                # Fallback to older quantization API
                import torch.quantization as quantization
                return True
            except ImportError:
                logging.warning("PyTorch quantization not available")
                return False
    
    def prepare_for_quantization(self, model: nn.Module, example_inputs: torch.Tensor) -> nn.Module:
        """Prepare model for quantization."""
        if not self.is_available:
            return model
            
        try:
            # Use newer quantization API if available
            try:
                import torch.ao.quantization as quantization
                from torch.ao.quantization import get_default_qconfig_mapping
                from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
                
                # Prepare model for quantization
                model.eval()
                qconfig_mapping = get_default_qconfig_mapping("x86")
                
                # For CUDA, try to use appropriate backend
                if example_inputs.is_cuda:
                    try:
                        qconfig_mapping = get_default_qconfig_mapping("cuda")
                    except:
                        qconfig_mapping = get_default_qconfig_mapping("x86")
                
                prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
                return prepared_model
                
            except ImportError:
                # Fallback to older API
                import torch.quantization as quantization
                
                model.eval()
                model.qconfig = quantization.get_default_qconfig('x86')
                prepared_model = quantization.prepare(model, inplace=False)
                return prepared_model
                
        except Exception as e:
            logging.warning(f"Failed to prepare model for quantization: {e}")
            return model
    
    def calibrate_model(self, model: nn.Module, calibration_data: List[torch.Tensor], 
                       max_samples: int = 100) -> nn.Module:
        """Calibrate model with representative data for quantization."""
        if not self.is_available:
            return model
            
        try:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(calibration_data):
                    if i >= max_samples:
                        break
                    _ = model(data)
            
            logging.info(f"Model calibrated with {min(len(calibration_data), max_samples)} samples")
            return model
            
        except Exception as e:
            logging.warning(f"Model calibration failed: {e}")
            return model
    
    def quantize_model(self, model: nn.Module, model_id: str = "default") -> nn.Module:
        """Convert calibrated model to quantized version."""
        if not self.is_available:
            return model
            
        if model_id in self.quantized_models:
            return self.quantized_models[model_id]
            
        try:
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
            self.quantized_models[model_id] = quantized_model
            
            logging.info(f"Model quantized to INT8 (ID: {model_id})")
            return quantized_model
            
        except Exception as e:
            logging.error(f"Model quantization failed: {e}")
            return model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size information."""
        try:
            # Calculate parameter size
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
        
        # Mixed precision variants
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
        
        # Experimental precisions
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
        
        # INT8 quantization - NEW
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
        
        # Dynamic precision
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
                # Skip quantized precisions if not requested
                if config.get("quantized", False) and not include_quantization:
                    continue
                    
                # Additional checks for specific precisions
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
                                
                elif precision == "int8":
                    # Check if quantization is available
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
                            include_quantization: bool = True) -> str:
        """
        Automatically select the best precision for the device.
        
        Args:
            device: Target device
            priority: Selection priority - "speed", "memory", "stability", "balanced"
            include_quantization: Whether to consider quantized precisions
        """
        supported = cls.get_supported_precisions(device, include_quantization)
        
        if not supported:
            return "fp32"
        
        if priority == "speed":
            # Prioritize speed: int8 > fp16 > bf16 > mixed > fp32
            priority_order = ["int8", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "fp32"]
        elif priority == "memory":
            # Prioritize memory efficiency: int8 > fp16/bf16 > mixed > fp32
            priority_order = ["int8", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "fp32"]
        elif priority == "stability":
            # Prioritize numerical stability: bf16 > mixed_bf16 > fp32 > mixed_fp16 > fp16 > int8
            priority_order = ["bf16", "mixed_bf16", "fp32", "mixed_fp16", "fp16", "int8"]
        else:  # balanced
            # Balance all factors: int8 > mixed_bf16 > bf16 > mixed_fp16 > tf32 > fp16 > fp32
            priority_order = ["int8", "mixed_bf16", "bf16", "mixed_fp16", "tf32", "fp16", "fp32"]
        
        for precision in priority_order:
            if precision in supported:
                return precision
        
        return "fp32"  # Fallback


class EnhancedConversationTrainer:
    """Production trainer with comprehensive monitoring, fault tolerance, and multiple precision support including INT8."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize precision manager
        self.precision_manager = PrecisionManager()
        
        # Initialize INT8 quantization manager
        self.int8_manager = Int8QuantizationManager()
        
        # GPU setup and memory management
        self._setup_gpu()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training precision setup (INT8 not supported for training)
        self.training_precision = getattr(config, 'precision', 'fp32')
        if self.training_precision == 'int8':
            logging.warning("INT8 precision not supported for training, falling back to fp32")
            self.training_precision = 'fp32'
            
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Check PyTorch version for autocast compatibility
        self.torch_version = torch.__version__
        logging.info(f"PyTorch version: {self.torch_version}")
        
        # COMPREHENSIVE INFERENCE PRECISION SETUP INCLUDING INT8
        self.inference_precision = getattr(config, 'inference_precision', 'auto')
        self._setup_inference_precision()
        
        # INT8 specific setup
        self.quantized_model = None
        self.int8_calibrated = False
        self.prepared_model = None  # Fixed: Add prepared model storage
        
        # TF32 setup for modern GPUs
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
        
        # Precision tracking
        self.precision_stats = {
            'training_precision': self.training_precision,
            'inference_precision': self.inference_precision,
            'supported_precisions': self.precision_manager.get_supported_precisions(self.device),
            'precision_switches': [],
            'performance_metrics': {},
            'int8_available': self.int8_manager.is_available,
            'int8_calibrated': self.int8_calibrated
        }
        
        # Metrics and monitoring
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': [],
            'precision_performance': {},
            'quantization_metrics': {}
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
        
        # Optimizer and scheduler - Fixed: Move after precision setup
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
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
        
        return SimpleCheckpointManager(config)

    def calibrate_int8_model(self, calibration_dataset, max_samples: int = 100):
        """
        Calibrate model for INT8 quantization using representative data.
        
        Args:
            calibration_dataset: Dataset to use for calibration
            max_samples: Maximum number of samples to use for calibration
        """
        if not self.int8_manager.is_available:
            logging.warning("INT8 quantization not available")
            return False
            
        if self.int8_calibrated:
            logging.info("Model already calibrated for INT8")
            return True
            
        logging.info(f"Calibrating model for INT8 quantization with {max_samples} samples...")
        
        try:
            # Create calibration dataloader
            calibration_dataloader = create_dataloader(
                calibration_dataset, self.config, shuffle=False
            )
            
            # Prepare example input for quantization setup
            first_batch = next(iter(calibration_dataloader))
            example_input = first_batch['input_ids'][:1].to(self.device)
            
            # Prepare model for quantization
            self.model.eval()
            self.prepared_model = self.int8_manager.prepare_for_quantization(
                self.model, example_input
            )
            
            # Collect calibration data
            calibration_data = []
            with torch.no_grad():
                for i, batch in enumerate(calibration_dataloader):
                    if i >= max_samples:
                        break
                        
                    batch = {k: v.to(self.device, non_blocking=True) 
                            for k, v in batch.items()}
                    
                    if batch['input_ids'].numel() > 0:
                        calibration_data.append(batch['input_ids'])
                        
                        # Run through prepared model for calibration
                        try:
                            # Fixed: Pass only input_ids if attention_mask not needed
                            if 'attention_mask' in batch:
                                _ = self.prepared_model(batch['input_ids'], batch['attention_mask'])
                            else:
                                _ = self.prepared_model(batch['input_ids'])
                        except Exception as e:
                            logging.debug(f"Calibration step {i} failed: {e}")
                            continue
            
            if calibration_data:
                # Perform calibration
                self.prepared_model = self.int8_manager.calibrate_model(
                    self.prepared_model, calibration_data, max_samples
                )
                
                self.int8_calibrated = True
                self.precision_stats['int8_calibrated'] = True
                
                logging.info(f"INT8 calibration completed with {len(calibration_data)} samples")
                return True
            else:
                logging.error("No valid calibration data found")
                return False
                
        except Exception as e:
            logging.error(f"INT8 calibration failed: {e}")
            return False
    
    def _get_int8_model(self, model_id: str = "default") -> nn.Module:
        """Get or create INT8 quantized model."""
        if not self.int8_manager.is_available:
            return self.model
            
        if not self.int8_calibrated:
            logging.warning("Model not calibrated for INT8, using original model")
            return self.model
            
        try:
            # Check if we already have a quantized model cached
            if model_id in self.int8_manager.quantized_models:
                return self.int8_manager.quantized_models[model_id]
            
            # Create quantized model
            quantized_model = self.int8_manager.quantize_model(
                self.prepared_model, model_id
            )
            
            # Log size comparison
            original_size = self.int8_manager.get_model_size(self.model)
            quantized_size = self.int8_manager.get_model_size(quantized_model)
            
            size_reduction = (1 - quantized_size['total_size_mb'] / max(original_size['total_size_mb'], 0.001)) * 100
            
            logging.info(f"INT8 model created: {original_size['total_size_mb']:.1f}MB → "
                        f"{quantized_size['total_size_mb']:.1f}MB ({size_reduction:.1f}% reduction)")
            
            # Store quantization metrics
            if 'int8' not in self.metrics['quantization_metrics']:
                self.metrics['quantization_metrics']['int8'] = []
                
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
    
    def _setup_inference_precision(self):
        """Setup inference precision with comprehensive auto-detection including INT8."""
        if self.inference_precision == "auto":
            self.inference_precision = self.precision_manager.auto_select_precision(
                self.device, priority="balanced", include_quantization=True
            )
            logging.info(f"Auto-selected inference precision: {self.inference_precision}")
        elif self.inference_precision == "dynamic":
            logging.info("Dynamic precision enabled - will select best precision per inference")
        
        # Validate precision
        supported = self.precision_manager.get_supported_precisions(self.device, include_quantization=True)
        if self.inference_precision not in supported and self.inference_precision != "dynamic":
            logging.warning(f"Inference precision {self.inference_precision} not supported, falling back to auto-selection")
            self.inference_precision = self.precision_manager.auto_select_precision(self.device)
        
        # Log precision info
        if self.inference_precision != "dynamic":
            precision_info = self.precision_manager.get_precision_info(self.inference_precision)
            if precision_info:
                logging.info(f"Inference precision info: {precision_info['name']} - {precision_info['description']}")
                if precision_info.get('quantized', False):
                    logging.info("INT8 quantization will be used for inference")
    
    def set_inference_precision(self, precision: str):
        """
        Dynamically change inference precision with validation and INT8 support.
        
        Args:
            precision: Target precision type
        """
        old_precision = self.inference_precision
        
        # Validate precision
        if precision == "auto":
            precision = self.precision_manager.auto_select_precision(self.device, include_quantization=True)
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
        
        # Track precision change
        self.precision_stats['precision_switches'].append({
            'timestamp': time.time(),
            'old_precision': old_precision,
            'new_precision': precision,
            'step': self.global_step
        })
        
        logging.info(f"Inference precision changed: {old_precision} → {precision}")
        return True
    
    def get_all_precision_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all precision types including INT8."""
        info = {}
        supported = self.precision_manager.get_supported_precisions(self.device, include_quantization=True)
        
        for precision in self.precision_manager.PRECISION_CONFIGS.keys():
            precision_info = self.precision_manager.get_precision_info(precision).copy()
            precision_info['supported'] = precision in supported
            precision_info['current_training'] = precision == self.training_precision
            precision_info['current_inference'] = precision == self.inference_precision
            
            # Add INT8 specific information
            if precision == 'int8':
                precision_info['calibrated'] = self.int8_calibrated
                precision_info['available'] = self.int8_manager.is_available
                if self.metrics['quantization_metrics'].get('int8'):
                    latest_metrics = self.metrics['quantization_metrics']['int8'][-1]
                    precision_info['size_reduction_percent'] = latest_metrics['size_reduction_percent']
            
            info[precision] = precision_info
        
        return info
    
    def _get_autocast_context(self, precision: Optional[str] = None, for_inference: bool = False):
        """
        Get autocast context with comprehensive precision support including INT8.
        
        Args:
            precision: Override precision (if None, uses training/inference precision)
            for_inference: Whether this is for inference (affects precision selection)
        """
        # Determine target precision
        if precision is None:
            target_precision = self.inference_precision if for_inference else self.training_precision
        else:
            target_precision = precision
        
        # Handle dynamic precision
        if target_precision == "dynamic":
            target_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed" if for_inference else "balanced",
                include_quantization=for_inference  # Only consider quantization for inference
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
            target_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed", include_quantization=True
            )
        
        # Return INT8 quantized model if requested
        if target_precision == "int8":
            return self._get_int8_model()
        
        # Return original model for all other precisions
        return self.model
    
    def _count_parameters(self):
        """Count model parameters."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except:
            return 0
    
    def _setup_gpu(self):
        """Setup GPU with optimal configuration."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Log GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            logging.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
            
            # Log compute capability for TF32 support
            if hasattr(torch.cuda, 'get_device_capability'):
                capability = torch.cuda.get_device_capability(0)
                logging.info(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
                if capability[0] >= 8:
                    logging.info("TF32 precision available (Ampere+ GPU)")
        else:
            logging.warning("CUDA not available, using CPU")
    
    def _log_memory_usage(self, phase: str):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"{phase} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter grouping."""
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
            # Use fused optimizer if available
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
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
        """Compile model with error handling."""
        if getattr(self.config, 'compile', False) and hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model...")
                self.model = torch.compile(self.model, mode='default')
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
                # Fallback for older PyTorch versions
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.optimizer, step_size=warmup_steps, gamma=0.1)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with detailed metrics."""
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
        perplexity = torch.exp(raw_loss.clamp(max=10))  # Clamp to prevent overflow
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss,
            'perplexity': perplexity,
            'valid_tokens': mask.sum()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with comprehensive monitoring."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Skip empty batches
        if batch['input_ids'].numel() == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Forward pass with training precision (never INT8 for training)
        with self._get_autocast_context(for_inference=False):
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            loss = loss_dict['loss'] / getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Check for valid loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.warning("Invalid loss detected, skipping batch")
            return {'loss': 0.0, 'perplexity': float('inf'), 'valid_tokens': 0}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return metrics
        return {
            'loss': loss.item() * getattr(self.config, 'gradient_accumulation_steps', 1),
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with gradient monitoring."""
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
    def evaluate(self, eval_dataset, max_batches: int = 100, 
                 precision_override: Optional[str] = None) -> Dict[str, float]:
        """Comprehensive evaluation with precision control including INT8."""
        # Determine evaluation precision
        eval_precision = precision_override or self.inference_precision
        
        # Get appropriate model for evaluation
        eval_model = self._get_model_for_inference(eval_precision)
        eval_model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        # Track memory usage for this precision
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if batch['input_ids'].numel() == 0:
                continue
            
            # For INT8, we don't use autocast
            if eval_precision == "int8":
                try:
                    logits = eval_model(batch['input_ids'], batch['attention_mask'])
                    loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
                except Exception as e:
                    logging.warning(f"INT8 evaluation failed on batch {batch_idx}: {e}")
                    continue
            else:
                with self._get_autocast_context(precision=eval_precision, for_inference=True):
                    logits = eval_model(batch['input_ids'], batch['attention_mask'])
                    loss_dict = self.compute_loss(logits, batch['labels'], batch['loss_weights'])
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                num_batches += 1
        
        eval_time = time.time() - eval_start_time
        
        # Memory usage
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
        perplexity = math.exp(min(avg_raw_loss, 10))  # Clamp to prevent overflow
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        # Store precision performance metrics
        if eval_precision not in self.metrics['precision_performance']:
            self.metrics['precision_performance'][eval_precision] = []
        
        performance_entry = {
            'throughput': throughput,
            'memory': peak_memory,
            'timestamp': time.time()
        }
        
        # Add INT8 specific metrics
        if eval_precision == "int8":
            if self.metrics['quantization_metrics'].get('int8'):
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
        """
        Generate response with enhanced error handling and comprehensive precision control including INT8.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            precision_override: Override inference precision for this generation
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            **kwargs: Additional generation parameters
        """
        # Determine generation precision
        gen_precision = precision_override or self.inference_precision
        if gen_precision == "dynamic":
            gen_precision = self.precision_manager.auto_select_precision(
                self.device, priority="speed", include_quantization=True
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
            max_new_tokens = getattr(self.config, 'max_new_tokens', 512)
        
        # Use provided or default generation parameters
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
            
            # Add assistant start tokens
            input_tokens.extend([
                self.tokenizer.special_tokens["<|im_start|>"],
                self.tokenizer.special_tokens["<|assistant|>"]
            ])
            
            # Ensure reasonable context length
            if len(input_tokens) >= self.config.seq_length:
                input_tokens = input_tokens[-(self.config.seq_length//2):]
            
            input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # Generation loop with safety checks and specified precision
            generated_tokens = []
            
            logging.debug(f"Starting generation with precision: {gen_precision}")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            for step in range(max_new_tokens):
                # Check sequence length
                if input_ids.size(1) >= self.config.seq_length:
                    input_ids = input_ids[:, -self.config.seq_length//2:]
                
                # Forward pass with specified precision
                if gen_precision == "int8":
                    # Direct forward pass for INT8 quantized model
                    logits = gen_model(input_ids)
                else:
                    # Use autocast for other precisions
                    with self._get_autocast_context(precision=gen_precision, for_inference=True):
                        logits = gen_model(input_ids)
                
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
                
                # Safety check for infinite loops
                if step > 0 and step % 100 == 0:
                    logging.debug(f"Generation step {step}/{max_new_tokens}")
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - generation_start_time
            tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
            peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            
            logging.debug(f"Generation completed: {len(generated_tokens)} tokens in {generation_time:.2f}s "
                         f"({tokens_per_second:.1f} tok/s) using {gen_precision} precision, "
                         f"peak memory: {peak_memory:.1f}MB")
            
            # Record performance metrics
            if gen_precision not in self.precision_stats['performance_metrics']:
                self.precision_stats['performance_metrics'][gen_precision] = []
            
            perf_entry = {
                'tokens_per_second': tokens_per_second,
                'peak_memory_mb': peak_memory,
                'generation_time': generation_time,
                'tokens_generated': len(generated_tokens),
                'timestamp': time.time()
            }
            
            # Add INT8 specific information
            if gen_precision == "int8" and self.metrics['quantization_metrics'].get('int8'):
                latest_quant = self.metrics['quantization_metrics']['int8'][-1]
                perf_entry['model_size_mb'] = latest_quant['quantized_size_mb']
                perf_entry['size_reduction_percent'] = latest_quant['size_reduction_percent']
            
            self.precision_stats['performance_metrics'][gen_precision].append(perf_entry)
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation failed with {gen_precision} precision: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def benchmark_inference_precision(self, test_prompts: Optional[List[str]] = None, 
                                    max_new_tokens: int = 100,
                                    include_int8: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different inference precisions with comprehensive metrics including INT8.
        
        Args:
            test_prompts: List of test prompts (uses default if None)
            max_new_tokens: Maximum tokens to generate per prompt
            include_int8: Whether to include INT8 in benchmarking
            
        Returns:
            Dictionary with benchmark results for each precision
        """
        if test_prompts is None:
            test_prompts = [
                "What is machine learning?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
                "How does neural attention work?",
                "Compare different sorting algorithms."
            ]
        
        # Get all supported precisions including INT8
        supported_precisions = self.precision_manager.get_supported_precisions(
            self.device, include_quantization=include_int8
        )
        
        # Add some additional precisions to test if they're supported
        test_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32"]
        if include_int8 and "int8" in supported_precisions:
            test_precisions.append("int8")
            
        precisions_to_test = [p for p in test_precisions if p in supported_precisions]
        
        results = {}
        original_precision = self.inference_precision
        
        logging.info(f"Benchmarking precisions: {', '.join(precisions_to_test)}")
        
        # Special preparation for INT8 if needed
        if "int8" in precisions_to_test and not self.int8_calibrated:
            logging.warning("INT8 precision will be benchmarked but model is not calibrated. "
                          "Results may not be optimal.")
        
        for precision in precisions_to_test:
            logging.info(f"Benchmarking {precision} precision...")
            
            # Get precision info
            precision_info = self.precision_manager.get_precision_info(precision)
            
            # Warm up
            try:
                _ = self.generate("Hello", max_new_tokens=5, precision_override=precision, temperature=0.1)
            except Exception as e:
                logging.warning(f"Warmup failed for {precision}: {e}, skipping...")
                continue
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Benchmark
            total_time = 0
            total_tokens = 0
            memory_measurements = []
            generations = []
            generation_times = []
            
            for i, prompt in enumerate(test_prompts):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                try:
                    response = self.generate(
                        prompt, 
                        max_new_tokens=max_new_tokens,
                        precision_override=precision,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                    generations.append(response)
                    
                    # Count tokens (approximate)
                    token_count = len(self.tokenizer.tokenizer.encode(response))
                    total_tokens += token_count
                    
                except Exception as e:
                    logging.error(f"Generation failed for {precision} on prompt {i+1}: {e}")
                    generations.append(f"ERROR: {str(e)}")
                    token_count = 0
                
                end_time = time.time()
                generation_time = end_time - start_time
                total_time += generation_time
                generation_times.append(generation_time)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    memory_measurements.append(peak_memory)
            
            # Calculate comprehensive metrics
            successful_generations = len([g for g in generations if not g.startswith("ERROR:")])
            success_rate = successful_generations / len(test_prompts) if test_prompts else 0
            avg_time_per_prompt = total_time / len(test_prompts) if test_prompts else 0
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            avg_memory = np.mean(memory_measurements) if memory_measurements else 0
            max_memory = max(memory_measurements) if memory_measurements else 0
            
            # Calculate generation time statistics
            if generation_times:
                min_time = min(generation_times)
                max_time = max(generation_times)
                std_time = np.std(generation_times)
            else:
                min_time = max_time = std_time = 0
            
            results[precision] = {
                'precision_info': precision_info,
                'success_rate': success_rate,
                'successful_generations': successful_generations,
                'total_time_seconds': total_time,
                'avg_time_per_prompt': avg_time_per_prompt,
                'min_time_per_prompt': min_time,
                'max_time_per_prompt': max_time,
                'std_time_per_prompt': std_time,
                'total_tokens_generated': total_tokens,
                'tokens_per_second': tokens_per_second,
                'avg_memory_mb': avg_memory,
                'peak_memory_mb': max_memory,
                'memory_efficiency_score': precision_info.get('memory_efficiency', 1.0) if precision_info else 1.0,
                'speed_multiplier': precision_info.get('speed_multiplier', 1.0) if precision_info else 1.0,
                'numerical_stability': precision_info.get('numerical_stability', 'unknown') if precision_info else 'unknown',
                'is_quantized': precision_info.get('quantized', False) if precision_info else False,
                'generations': generations,
                'generation_times': generation_times
            }
            
            # Add INT8 specific metrics
            if precision == "int8" and self.metrics['quantization_metrics'].get('int8'):
                latest_quant = self.metrics['quantization_metrics']['int8'][-1]
                results[precision]['model_size_reduction_percent'] = latest_quant['size_reduction_percent']
                results[precision]['quantized_model_size_mb'] = latest_quant['quantized_size_mb']
            
            logging.info(f"{precision} results: {tokens_per_second:.1f} tokens/s, "
                        f"{avg_time_per_prompt:.2f}s/prompt (±{std_time:.2f}s), "
                        f"{avg_memory:.1f}MB avg memory, {success_rate:.1%} success rate")
        
        # Restore original precision
        self.set_inference_precision(original_precision)
        
        return results
    
    def get_precision_recommendations(self, use_case: str = "balanced",
                                    include_quantization: bool = True) -> Dict[str, Any]:
        """
        Get precision recommendations based on use case, including INT8 quantization.
        
        Args:
            use_case: "speed", "memory", "quality", "balanced", or "production"
            include_quantization: Whether to consider quantized precisions
            
        Returns:
            Precision recommendations with rationale
        """
        supported = self.precision_manager.get_supported_precisions(
            self.device, include_quantization=include_quantization
        )
        
        recommendations = {
            'use_case': use_case,
            'device': str(self.device),
            'supported_precisions': supported,
            'quantization_available': self.int8_manager.is_available,
            'int8_calibrated': self.int8_calibrated
        }
        
        # Use case specific recommendations
        if use_case == "speed":
            recommendations.update({
                'primary_recommendation': self.precision_manager.auto_select_precision(
                    self.device, priority="speed", include_quantization=include_quantization
                ),
                'alternatives': ["int8", "fp16", "mixed_fp16", "bf16"],
                'rationale': "Prioritizing maximum inference speed. INT8 offers best speed but requires calibration. FP16 is fastest uncalibrated option.",
                'considerations': [
                    "INT8 requires model calibration but offers maximum speed and memory efficiency",
                    "FP16 provides good speed without calibration but may have stability issues",
                    "Mixed precision balances speed and stability"
                ]
            })
        
        elif use_case == "memory":
            recommendations.update({
                'primary_recommendation': self.precision_manager.auto_select_precision(
                    self.device, priority="memory", include_quantization=include_quantization
                ),
                'alternatives': ["int8", "fp16", "bf16", "mixed_fp16"],
                'rationale': "Prioritizing memory efficiency. INT8 offers 4x memory reduction, FP16/BF16 offer 2x reduction.",
                'considerations': [
                    "INT8 provides maximum memory savings (up to 75% reduction)",
                    "FP16/BF16 offer good memory efficiency with easier setup",
                    "Mixed precision provides some memory benefits while maintaining stability"
                ]
            })
        
        elif use_case == "quality":
            recommendations.update({
                'primary_recommendation': self.precision_manager.auto_select_precision(
                    self.device, priority="stability", include_quantization=False
                ),
                'alternatives': ["bf16", "fp32", "mixed_bf16"],
                'rationale': "Prioritizing output quality and numerical stability. BF16 offers best balance of efficiency and stability.",
                'considerations': [
                    "BF16 provides excellent numerical stability with 2x memory efficiency",
                    "FP32 offers maximum precision but uses more memory",
                    "Avoid INT8 for quality-critical applications unless extensively tested"
                ]
            })
        
        elif use_case == "production":
            recommendations.update({
                'primary_recommendation': "mixed_bf16" if "mixed_bf16" in supported else "bf16" if "bf16" in supported else "fp32",
                'alternatives': ["int8", "mixed_bf16", "bf16", "fp32"],
                'rationale': "Production environments need reliability. Mixed BF16 offers optimal balance of performance, stability, and efficiency.",
                'considerations': [
                    "Mixed BF16 provides excellent stability with good performance",
                    "INT8 can be used for high-throughput scenarios after proper validation",
                    "Always benchmark thoroughly before deployment",
                    "Consider fallback mechanisms for precision-related issues"
                ]
            })
        
        else:  # balanced
            recommendations.update({
                'primary_recommendation': self.precision_manager.auto_select_precision(
                    self.device, priority="balanced", include_quantization=include_quantization
                ),
                'alternatives': ["mixed_bf16", "bf16", "int8", "mixed_fp16"],
                'rationale': "Balanced approach considering speed, memory, and stability. Auto-selected precision provides best overall performance.",
                'considerations': [
                    "Mixed BF16 typically offers the best balance for most use cases",
                    "INT8 can provide significant benefits if calibration is feasible",
                    "Consider your specific hardware and use case requirements"
                ]
            })
        
        # Add device-specific considerations
        if self.device.type == "cuda":
            recommendations['device_considerations'] = [
                f"GPU compute capability: {torch.cuda.get_device_capability(0) if hasattr(torch.cuda, 'get_device_capability') else 'unknown'}",
                "CUDA device supports most precision types",
                "Consider TF32 for Ampere+ GPUs" if "tf32" in supported else "TF32 not available on this GPU"
            ]
        else:
            recommendations['device_considerations'] = [
                "CPU execution - limited precision support",
                "FP32 and INT8 are primary options for CPU",
                "Consider INT8 quantization for CPU deployment"
            ]
        
        # Add INT8 specific guidance
        if include_quantization and self.int8_manager.is_available:
            recommendations['int8_guidance'] = {
                'available': True,
                'calibrated': self.int8_calibrated,
                'setup_required': not self.int8_calibrated,
                'benefits': [
                    "4x memory reduction",
                    "2x speed improvement (typical)",
                    "Enables larger model deployment"
                ],
                'drawbacks': [
                    "Requires calibration dataset",
                    "Potential quality degradation",
                    "Inference-only (no training support)"
                ],
                'recommendation': "Consider INT8 for production deployment after validation" if self.int8_calibrated else "Calibrate model first, then evaluate INT8 performance"
            }
        else:
            recommendations['int8_guidance'] = {
                'available': False,
                'reason': "PyTorch quantization not available or not requested"
            }
        
        return recommendations
    
    def train(self, train_dataset, eval_dataset=None, num_epochs: int = None, 
              resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Main training loop with comprehensive monitoring and fault tolerance.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Training results and metrics
        """
        if num_epochs is None:
            num_epochs = getattr(self.config, 'num_epochs', 3)
        
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Training precision: {self.training_precision}")
        logging.info(f"Inference precision: {self.inference_precision}")
        
        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_path = self.checkpoint_manager.get_resume_path()
            if checkpoint_path:
                logging.info(f"Resuming from checkpoint: {checkpoint_path}")
                start_epoch = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path, self.model, self.optimizer, self.scheduler
                )
        
        # Create dataloader
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
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            if self.should_stop:
                logging.info("Early stopping triggered")
                break
            
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Evaluation
            eval_metrics = {}
            if eval_dataset is not None:
                eval_metrics = self.evaluate(eval_dataset)
                
                # Early stopping check
                eval_loss = eval_metrics.get('eval_loss', float('inf'))
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, epoch, {**train_metrics, **eval_metrics},
                        suffix="best"
                    )
                else:
                    patience_counter += 1
                    
                early_stopping_patience = getattr(self.config, 'early_stopping_patience', 5)
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping after {patience_counter} epochs without improvement")
                    self.should_stop = True
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.metrics['epoch_times'].append(epoch_time)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            logging.info(f"  Train loss: {train_metrics.get('avg_loss', 0):.4f}")
            logging.info(f"  Train perplexity: {train_metrics.get('avg_perplexity', 0):.2f}")
            if eval_metrics:
                logging.info(f"  Eval loss: {eval_metrics.get('eval_loss', 0):.4f}")
                logging.info(f"  Eval perplexity: {eval_metrics.get('eval_perplexity', 0):.2f}")
            
            # Regular checkpoint
            if (epoch + 1) % getattr(self.config, 'save_every', 1) == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, epoch, {**train_metrics, **eval_metrics},
                    suffix=f"epoch_{epoch+1}"
                )
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
            'inference_precision': self.inference_precision
        }
        
        logging.info(f"Training completed in {total_training_time:.2f}s")
        logging.info(f"Best evaluation loss: {best_eval_loss:.4f}")
        
        return results
    
    def _train_epoch(self, train_dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with comprehensive monitoring."""
        self.model.train()
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_perplexity = 0.0
        total_tokens = 0
        batch_count = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
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
                
                # Record metrics
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
            
            # Periodic logging
            if self.global_step % getattr(self.config, 'log_every', 100) == 0:
                avg_loss = total_loss / max(batch_count, 1)
                avg_throughput = np.mean(self.metrics['throughput'][-10:]) if self.metrics['throughput'] else 0
                
                logging.info(f"Step {self.global_step}: loss={avg_loss:.4f}, "
                           f"throughput={avg_throughput:.0f} tok/s, "
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
            'tokens_per_second': total_tokens / epoch_time if epoch_time > 0 else 0
        }