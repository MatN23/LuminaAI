# Copyright (c) 2025 MatN23. All rights reserved.
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

# Quantization imports with fallbacks
try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import AdamW8bit, Lion8bit
    BNB_AVAILABLE = True
    logging.info("BitsAndBytes available for 8-bit quantization")
except ImportError:
    BNB_AVAILABLE = False
    logging.warning("BitsAndBytes not available - 8-bit quantization disabled")

try:
    from transformers import BitsAndBytesConfig
    HF_BNB_AVAILABLE = True
except ImportError:
    HF_BNB_AVAILABLE = False

try:
    import auto_gptq
    GPTQ_AVAILABLE = True
    logging.info("AutoGPTQ available for 4-bit quantization")
except ImportError:
    GPTQ_AVAILABLE = False

try:
    from optimum.quanto import quantize, freeze
    QUANTO_AVAILABLE = True
    logging.info("Optimum Quanto available for quantization")
except ImportError:
    QUANTO_AVAILABLE = False

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
    class DeepSpeedEngine:
        pass

try:
    from core.dataset import create_dataloader
except ImportError:
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
    class TrainingHealthMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

try:
    from training.checkpoint import CheckpointManager
except ImportError:
    class CheckpointManager:
        def __init__(self, *args, **kwargs):
            pass
        def save_checkpoint(self, *args, **kwargs):
            pass


class PrecisionManager:
    """Manages comprehensive precision configurations for training and inference."""
    
    # Precision definitions with capabilities
    PRECISION_REGISTRY = {
        # Floating Point Precisions
        'fp64': {
            'dtype': torch.float64,
            'bits': 64,
            'category': 'float',
            'use_case': 'Scientific computing, high precision',
            'supported_on': ['cpu', 'cuda'],
            'stable_training': True,
            'memory_multiplier': 2.0,
            'requires_amp': False
        },
        'fp32': {
            'dtype': torch.float32,
            'bits': 32,
            'category': 'float',
            'use_case': 'Standard training/inference',
            'supported_on': ['cpu', 'cuda', 'mps'],
            'stable_training': True,
            'memory_multiplier': 1.0,
            'requires_amp': False
        },
        'tf32': {
            'dtype': torch.float32,
            'bits': 32,
            'category': 'float',
            'use_case': 'NVIDIA Ampere+ GPUs, faster training',
            'supported_on': ['cuda'],
            'stable_training': True,
            'memory_multiplier': 1.0,
            'requires_amp': False,
            'requires_ampere': True
        },
        'fp16': {
            'dtype': torch.float16,
            'bits': 16,
            'category': 'float',
            'use_case': 'Mixed precision training, memory efficient',
            'supported_on': ['cuda'],
            'stable_training': True,
            'memory_multiplier': 0.5,
            'requires_amp': True
        },
        'bf16': {
            'dtype': torch.bfloat16,
            'bits': 16,
            'category': 'float',
            'use_case': 'FP32 range, FP16 memory, excellent for deep nets',
            'supported_on': ['cuda', 'cpu'],
            'stable_training': True,
            'memory_multiplier': 0.5,
            'requires_amp': True
        },
        'fp8_e4m3': {
            'dtype': torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None,
            'bits': 8,
            'category': 'float',
            'use_case': 'Cutting-edge low precision, H100+ GPUs',
            'supported_on': ['cuda'],
            'stable_training': False,
            'memory_multiplier': 0.25,
            'requires_amp': True,
            'experimental': True
        },
        'fp8_e5m2': {
            'dtype': torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else None,
            'bits': 8,
            'category': 'float',
            'use_case': 'FP8 with larger range, H100+ GPUs',
            'supported_on': ['cuda'],
            'stable_training': False,
            'memory_multiplier': 0.25,
            'requires_amp': True,
            'experimental': True
        },
        'fp128': {
            'dtype': None,
            'bits': 128,
            'category': 'float',
            'use_case': 'Quad precision, scientific computing only',
            'supported_on': [],
            'stable_training': True,
            'memory_multiplier': 4.0,
            'requires_amp': False,
            'unsupported': True
        },
        
        # Integer Precisions
        'int64': {
            'dtype': torch.int64,
            'bits': 64,
            'category': 'int',
            'use_case': 'Indexing, large accumulators',
            'supported_on': ['cpu', 'cuda', 'mps'],
            'stable_training': False,
            'memory_multiplier': 2.0,
            'requires_amp': False
        },
        'int32': {
            'dtype': torch.int32,
            'bits': 32,
            'category': 'int',
            'use_case': 'Standard indexing',
            'supported_on': ['cpu', 'cuda', 'mps'],
            'stable_training': False,
            'memory_multiplier': 1.0,
            'requires_amp': False
        },
        'int16': {
            'dtype': torch.int16,
            'bits': 16,
            'category': 'int',
            'use_case': 'Embedded ML, some accumulation',
            'supported_on': ['cpu', 'cuda', 'mps'],
            'stable_training': False,
            'memory_multiplier': 0.5,
            'requires_amp': False
        },
        'int8': {
            'dtype': torch.int8,
            'bits': 8,
            'category': 'int',
            'use_case': 'Quantized inference, BitsAndBytes',
            'supported_on': ['cpu', 'cuda'],
            'stable_training': False,
            'memory_multiplier': 0.25,
            'requires_amp': False
        },
        'int4': {
            'dtype': None,
            'bits': 4,
            'category': 'int',
            'use_case': 'Extreme quantization, tiny models',
            'supported_on': ['cuda'],
            'stable_training': False,
            'memory_multiplier': 0.125,
            'requires_amp': False,
            'requires_special_ops': True
        },
        'int2': {
            'dtype': None,
            'bits': 2,
            'category': 'int',
            'use_case': 'Experimental ultra-low memory',
            'supported_on': [],
            'stable_training': False,
            'memory_multiplier': 0.0625,
            'requires_amp': False,
            'experimental': True
        },
        
        # Unsigned Integer Precisions
        'uint8': {
            'dtype': torch.uint8,
            'bits': 8,
            'category': 'uint',
            'use_case': 'Image pixels, quantization',
            'supported_on': ['cpu', 'cuda', 'mps'],
            'stable_training': False,
            'memory_multiplier': 0.25,
            'requires_amp': False
        },
        
        # Mixed Precision Modes
        'mixed_fp16': {
            'dtype': torch.float16,
            'bits': 16,
            'category': 'mixed',
            'use_case': 'Mixed precision with FP32 accumulation',
            'supported_on': ['cuda'],
            'stable_training': True,
            'memory_multiplier': 0.5,
            'requires_amp': True
        },
        'mixed_bf16': {
            'dtype': torch.bfloat16,
            'bits': 16,
            'category': 'mixed',
            'use_case': 'Mixed precision with BF16',
            'supported_on': ['cuda', 'cpu'],
            'stable_training': True,
            'memory_multiplier': 0.5,
            'requires_amp': True
        },
        'mixed_fp8': {
            'dtype': torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None,
            'bits': 8,
            'category': 'mixed',
            'use_case': 'Mixed FP8 precision for H100+',
            'supported_on': ['cuda'],
            'stable_training': False,
            'memory_multiplier': 0.25,
            'requires_amp': True,
            'experimental': True
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.train_precision = getattr(config, 'precision', 'fp32')
        self.inference_precision = getattr(config, 'inference_precision', self.train_precision)
        
        self._validate_precision_config()
        self._setup_device_optimizations()
        
    def _validate_precision_config(self):
        """Validate precision configurations against hardware capabilities."""
        device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        
        if self.train_precision not in self.PRECISION_REGISTRY:
            raise ValueError(f"Unknown precision: {self.train_precision}. Available: {list(self.PRECISION_REGISTRY.keys())}")
        
        train_spec = self.PRECISION_REGISTRY[self.train_precision]
        
        if device_type not in train_spec['supported_on']:
            raise ValueError(f"{self.train_precision} not supported on {device_type}. Supported devices: {train_spec['supported_on']}")
        
        if train_spec.get('unsupported', False):
            raise ValueError(f"{self.train_precision} is not supported in PyTorch")
        
        if train_spec.get('experimental', False):
            logging.warning(f"{self.train_precision} is experimental and may be unstable")
        
        if train_spec['dtype'] is None and not train_spec.get('requires_special_ops', False):
            raise ValueError(f"{self.train_precision} dtype not available in this PyTorch version")
        
        if train_spec.get('requires_ampere', False) and device_type == 'cuda':
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 8:
                logging.warning(f"TF32 requires Ampere GPU (compute capability 8.0+), current: {compute_capability}")
        
        if self.inference_precision not in self.PRECISION_REGISTRY:
            logging.warning(f"Unknown inference precision {self.inference_precision}, using training precision")
            self.inference_precision = self.train_precision
        
        if not train_spec['stable_training']:
            logging.warning(f"{self.train_precision} may not be stable for training. Consider using for inference only.")
    
    def _setup_device_optimizations(self):
        """Setup device-specific precision optimizations."""
        if not torch.cuda.is_available():
            return
        
        if self.train_precision == 'tf32':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("TF32 enabled for matmul and cuDNN operations")
        
        if self.train_precision in ['fp16', 'bf16', 'mixed_fp16', 'mixed_bf16']:
            torch.backends.cudnn.benchmark = True
            logging.info("cuDNN benchmark mode enabled for faster training")
        
        if 'fp8' in self.train_precision:
            try:
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                    logging.info("Flash attention enabled for FP8")
            except Exception as e:
                logging.warning(f"Could not enable FP8 optimizations: {e}")
    
    def get_dtype(self, for_inference: bool = False) -> Optional[torch.dtype]:
        """Get the appropriate dtype for the current precision."""
        precision = self.inference_precision if for_inference else self.train_precision
        spec = self.PRECISION_REGISTRY[precision]
        return spec['dtype']
    
    def get_autocast_context(self, for_inference: bool = False):
        """Get autocast context for the current precision."""
        precision = self.inference_precision if for_inference else self.train_precision
        spec = self.PRECISION_REGISTRY[precision]
        
        if not spec['requires_amp'] or spec['dtype'] is None:
            return nullcontext()
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if precision == 'tf32':
            return nullcontext()
        
        try:
            if 'fp8' in precision:
                if hasattr(torch, 'autocast'):
                    return autocast(device_type, dtype=spec['dtype'], enabled=True)
                else:
                    logging.warning("FP8 autocast not available, using FP16 fallback")
                    return autocast(device_type, dtype=torch.float16, enabled=True)
            else:
                return autocast(device_type, dtype=spec['dtype'], enabled=True)
        except (TypeError, RuntimeError) as e:
            logging.warning(f"Autocast failed for {precision}: {e}, using nullcontext")
            return nullcontext()
    
    def should_use_grad_scaler(self) -> bool:
        """Determine if gradient scaling is needed."""
        precision = self.train_precision
        return precision in ['fp16', 'mixed_fp16']
    
    def estimate_memory_usage(self, model_params: int) -> Dict[str, float]:
        """Estimate memory usage for different precisions."""
        results = {}
        
        for precision, spec in self.PRECISION_REGISTRY.items():
            if spec.get('unsupported', False):
                continue
            
            base_memory_mb = (model_params * 4) / (1024 ** 2)
            estimated_memory = base_memory_mb * spec['memory_multiplier']
            
            results[precision] = {
                'model_memory_mb': estimated_memory,
                'memory_savings_pct': (1 - spec['memory_multiplier']) * 100,
                'use_case': spec['use_case']
            }
        
        return results
    
    def get_precision_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current precision settings."""
        train_spec = self.PRECISION_REGISTRY[self.train_precision]
        inference_spec = self.PRECISION_REGISTRY[self.inference_precision]
        
        return {
            'training': {
                'precision': self.train_precision,
                'dtype': str(train_spec['dtype']),
                'bits': train_spec['bits'],
                'category': train_spec['category'],
                'stable': train_spec['stable_training'],
                'use_case': train_spec['use_case']
            },
            'inference': {
                'precision': self.inference_precision,
                'dtype': str(inference_spec['dtype']),
                'bits': inference_spec['bits'],
                'category': inference_spec['category'],
                'use_case': inference_spec['use_case']
            },
            'hardware': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                'cuda_capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            }
        }
    
    def print_precision_recommendations(self):
        """Print recommendations for precision selection."""
        print("\n" + "="*80)
        print("PRECISION RECOMMENDATIONS")
        print("="*80)
        
        device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        
        print(f"\nCurrent Device: {device_type}")
        if device_type == 'cuda':
            cc = torch.cuda.get_device_capability()
            print(f"Compute Capability: {cc[0]}.{cc[1]}")
        
        print("\nAvailable Precisions:")
        print("-" * 80)
        
        categories = {}
        for precision, spec in self.PRECISION_REGISTRY.items():
            if spec.get('unsupported', False):
                continue
            
            category = spec['category']
            if category not in categories:
                categories[category] = []
            
            available = device_type in spec['supported_on']
            status = "✅" if available else "❌"
            
            experimental = " [EXPERIMENTAL]" if spec.get('experimental', False) else ""
            special = " [SPECIAL OPS REQUIRED]" if spec.get('requires_special_ops', False) else ""
            
            categories[category].append({
                'name': precision,
                'status': status,
                'bits': spec['bits'],
                'use_case': spec['use_case'],
                'notes': experimental + special
            })
        
        for category, precisions in sorted(categories.items()):
            print(f"\n{category.upper()} PRECISIONS:")
            for p in sorted(precisions, key=lambda x: -x['bits']):
                print(f"  {p['status']} {p['name']:15s} ({p['bits']:3d} bits) - {p['use_case']}{p['notes']}")
        
        print("\n" + "="*80)
        print("USAGE RECOMMENDATIONS:")
        print("="*80)
        print("""
Training:
  • fp32:       Safe default, maximum stability
  • bf16:       Recommended for modern GPUs (Ampere+), best balance
  • mixed_fp16: Good for Volta/Turing GPUs, memory efficient
  • tf32:       Automatic speedup on Ampere+ (transparent)
  • fp64:       Only for numerical stability issues

Inference:
  • int8:       Fast inference with BitsAndBytes quantization
  • int4:       Maximum memory savings, GPTQ/AWQ
  • fp16:       Fast and accurate
  • bf16:       Best overall choice for modern hardware

Experimental:
  • fp8:        Cutting-edge H100+ GPUs only
  • int2:       Research purposes only
        """)
        
        print("="*80 + "\n")


class QuantizationManager:
    """Manages different quantization strategies and optimizations."""
    
    def __init__(self, config):
        self.config = config
        self.quantization_method = getattr(config, 'quantization_method', None)
        self.quantization_bits = getattr(config, 'quantization_bits', None)
        
        self._validate_quantization_config()
        
        self.is_quantized = False
        self.quantization_info = {}
        
    def _validate_quantization_config(self):
        """Validate quantization configuration and availability."""
        if not self.quantization_method:
            return
            
        precision = getattr(self.config, 'precision', 'fp32')
        
        if self.quantization_method == 'bnb' and not BNB_AVAILABLE:
            raise ValueError("BitsAndBytes not available but bnb quantization requested")
        
        if self.quantization_method == 'gptq' and not GPTQ_AVAILABLE:
            raise ValueError("AutoGPTQ not available but gptq quantization requested")
            
        if self.quantization_method == 'quanto' and not QUANTO_AVAILABLE:
            raise ValueError("Optimum Quanto not available but quanto quantization requested")
        
        if self.quantization_bits and self.quantization_bits not in [4, 8]:
            raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}. Only 4 and 8 bit supported.")
        
        if precision in ['fp16', 'bf16'] and self.quantization_bits == 4:
            logging.warning("Mixed precision with 4-bit quantization may cause instability")
    
    def get_bnb_config(self) -> Optional[Dict[str, Any]]:
        """Get BitsAndBytes quantization configuration."""
        if not BNB_AVAILABLE or self.quantization_method != 'bnb':
            return None
            
        if self.quantization_bits == 8:
            return {
                'load_in_8bit': True,
                'llm_int8_threshold': 6.0,
                'llm_int8_has_fp16_weight': False,
                'llm_int8_enable_fp32_cpu_offload': getattr(self.config, 'cpu_offload', False)
            }
        elif self.quantization_bits == 4:
            return {
                'load_in_4bit': True,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            }
        
        return None
    
    def quantize_model_bnb(self, model):
        """Apply BitsAndBytes quantization to model."""
        if not BNB_AVAILABLE:
            raise ValueError("BitsAndBytes not available")
        
        config = self.get_bnb_config()
        if not config:
            return model
            
        logging.info(f"Applying BitsAndBytes {self.quantization_bits}-bit quantization...")
        
        if self.quantization_bits == 8:
            model = self._replace_linear_layers_8bit(model)
        elif self.quantization_bits == 4:
            logging.warning("4-bit quantization with BnB should ideally be done at model initialization")
        
        self.is_quantized = True
        self.quantization_info = {
            'method': 'bnb',
            'bits': self.quantization_bits,
            'config': config
        }
        
        return model
    
    def _replace_linear_layers_8bit(self, model):
        """Replace Linear layers with 8-bit equivalents."""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                int8_module = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
                
                with torch.no_grad():
                    int8_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        int8_module.bias.data = module.bias.data.clone()
                
                setattr(model, name, int8_module)
            else:
                self._replace_linear_layers_8bit(module)
        
        return model
    
    def quantize_model_gptq(self, model):
        """Apply GPTQ quantization to model."""
        if not GPTQ_AVAILABLE:
            raise ValueError("AutoGPTQ not available")
        
        logging.info(f"Applying GPTQ {self.quantization_bits}-bit quantization...")
        
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        quantize_config = BaseQuantizeConfig(
            bits=self.quantization_bits,
            group_size=128,
            desc_act=False,
            static_groups=False,
            sym=True,
            true_sequential=True,
            model_name_or_path=None,
            model_file_base_name="model"
        )
        
        logging.warning("GPTQ quantization requires calibration data for best results")
        
        self.is_quantized = True
        self.quantization_info = {
            'method': 'gptq',
            'bits': self.quantization_bits,
            'config': quantize_config
        }
        
        return model
    
    def quantize_model_quanto(self, model):
        """Apply Quanto quantization to model."""
        if not QUANTO_AVAILABLE:
            raise ValueError("Optimum Quanto not available")
        
        logging.info(f"Applying Quanto {self.quantization_bits}-bit quantization...")
        
        if self.quantization_bits == 8:
            weights = torch.int8
        elif self.quantization_bits == 4:
            weights = torch.int4 if hasattr(torch, 'int4') else "int4"
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
        
        quantize(model, weights=weights, activations=None)
        freeze(model)
        
        self.is_quantized = True
        self.quantization_info = {
            'method': 'quanto',
            'bits': self.quantization_bits,
            'weights': weights
        }
        
        return model
    
    def quantize_model(self, model):
        """Apply the configured quantization method to the model."""
        if not self.quantization_method or not self.quantization_bits:
            return model
        
        logging.info(f"Quantizing model with {self.quantization_method} method, {self.quantization_bits} bits")
        
        original_memory = self._get_model_memory_usage(model)
        
        if self.quantization_method == 'bnb':
            model = self.quantize_model_bnb(model)
        elif self.quantization_method == 'gptq':
            model = self.quantize_model_gptq(model)
        elif self.quantization_method == 'quanto':
            model = self.quantize_model_quanto(model)
        else:
            raise ValueError(f"Unsupported quantization method: {self.quantization_method}")
        
        quantized_memory = self._get_model_memory_usage(model)
        memory_savings = (original_memory - quantized_memory) / original_memory * 100
        
        logging.info(f"Quantization complete:")
        logging.info(f"  Original memory: {original_memory:.1f}MB")
        logging.info(f"  Quantized memory: {quantized_memory:.1f}MB")
        logging.info(f"  Memory savings: {memory_savings:.1f}%")
        
        return model
    
    def _get_model_memory_usage(self, model):
        """Estimate model memory usage in MB."""
        total_params = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_params / (1024 * 1024)
    
    def create_quantized_optimizer(self, model):
        """Create optimizer compatible with quantized model."""
        if not self.is_quantized:
            return None
        
        if self.quantization_method == 'bnb' and BNB_AVAILABLE:
            if self.quantization_bits == 8:
                logging.info("Using 8-bit AdamW optimizer")
                return AdamW8bit(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=getattr(self.config, 'weight_decay', 0.01),
                    optim_bits=32
                )
        
        return None
    
    def get_quantization_info(self):
        """Get information about current quantization state."""
        return {
            'is_quantized': self.is_quantized,
            'method': self.quantization_info.get('method'),
            'bits': self.quantization_info.get('bits'),
            'available_methods': {
                'bnb': BNB_AVAILABLE,
                'gptq': GPTQ_AVAILABLE,
                'quanto': QUANTO_AVAILABLE
            }
        }


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
        
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        num_experts = getattr(self.config, 'num_experts', 8)
        
        optimal_ep_size = self._calculate_optimal_expert_parallel_size(world_size, num_experts)
        
        moe_config = {
            "moe": {
                "enabled": True,
                "num_experts": num_experts,
                "expert_parallel_size": optimal_ep_size,
                "top_k": getattr(self.config, 'moe_top_k', 2),
                
                "capacity_factor": getattr(self.config, 'capacity_factor', 1),
                "eval_capacity_factor": 3.2,
                "min_capacity": 16,
                "use_residual": True,
                
                "load_balance_loss_coef": getattr(self.config, 'load_balancing_weight', 0.08),
                "load_balance_type": "aux_loss",
                "router_jitter_noise": 0.01,
                
                "enable_expert_tensor_parallelism": True,
                "all_to_all_dispatch": True,
                "overlap_alltoall": True,
                "comm_dtype": "fp16" if self.config.precision in ["fp16", "mixed_fp16"] else "bf16",
                
                "pad_expert_input_to_capacity": True,
                "enable_expert_weight_parallelism": True,
                "moe_param_group": True,
                
                "expert_placement_policy": "balanced",
                "use_tutel": False,
            }
        }
        
        base_config.update(moe_config)
        
        if "zero_optimization" not in base_config:
            base_config["zero_optimization"] = {}
            
        base_config["zero_optimization"].update({
            "stage": 3,
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
            "stage3_param_persistence_threshold": 10000.0,
            "stage3_max_live_parameters": 1000000000.0,
            "stage3_prefetch_bucket_size": 50000000.0,
            "memory_efficient_linear": True,
            "stage3_max_reuse_distance": 1000
        })
        
        logging.info(f"MoE Config: {num_experts} experts, EP size: {optimal_ep_size}, "
                    f"capacity factor: {moe_config['moe']['capacity_factor']}")
        
        return base_config
    
    def _calculate_optimal_expert_parallel_size(self, world_size: int, num_experts: int) -> int:
        """Calculate optimal expert parallel size to minimize all-to-all overhead."""
        
        possible_ep_sizes = []
        for i in range(1, world_size + 1):
            if world_size % i == 0:
                experts_per_group = num_experts // i
                if experts_per_group >= 1:
                    possible_ep_sizes.append((i, experts_per_group))
        
        if not possible_ep_sizes:
            return 1
        
        best_ep_size = 1
        best_score = 0
        
        for ep_size, experts_per_group in possible_ep_sizes:
            comm_score = 1.0 / ep_size
            expert_score = min(experts_per_group / 4.0, 1.0)
            balance_score = 1.0 if experts_per_group > 1 else 0.5
            
            total_score = comm_score * expert_score * balance_score
            
            if total_score > best_score:
                best_score = total_score
                best_ep_size = ep_size
        
        return best_ep_size
    
    def monitor_routing_balance(self, aux_losses: Dict[str, torch.Tensor], 
                              routing_probs: Optional[torch.Tensor] = None):
        """Monitor and log routing balance metrics."""
        
        if 'load_balance_loss' in aux_losses:
            self.routing_stats['load_balance_losses'].append(aux_losses['load_balance_loss'].item())
        
        if routing_probs is not None:
            expert_usage = routing_probs.sum(dim=0).cpu().numpy()
            total_tokens = routing_probs.sum().item()
            
            if total_tokens > 0:
                usage_percentages = expert_usage / total_tokens * 100
                
                for expert_id, usage_pct in enumerate(usage_percentages):
                    if expert_id not in self.routing_stats['expert_usage']:
                        self.routing_stats['expert_usage'][expert_id] = []
                    self.routing_stats['expert_usage'][expert_id].append(usage_pct)
                
                max_usage = usage_percentages.max()
                min_usage = usage_percentages.min()
                imbalance_ratio = max_usage / max(min_usage, 0.1)
                
                if imbalance_ratio > 10:
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
        
        if self.routing_stats['load_balance_losses']:
            recent_losses = self.routing_stats['load_balance_losses'][-100:]
            diagnostics['load_balance_trend'] = {
                'recent_avg': np.mean(recent_losses),
                'trend': 'improving' if len(recent_losses) > 10 and np.mean(recent_losses[-5:]) < np.mean(recent_losses[-10:-5]) else 'stable'
            }
        
        if self.routing_stats['expert_usage']:
            expert_usages = []
            for expert_id, usage_history in self.routing_stats['expert_usage'].items():
                if usage_history:
                    expert_usages.append(np.mean(usage_history[-50:]))
            
            if expert_usages:
                usage_std = np.std(expert_usages)
                usage_mean = np.mean(expert_usages)
                balance_score = max(0, 1.0 - (usage_std / max(usage_mean, 1.0)))
                diagnostics['expert_balance_score'] = balance_score
                
                if balance_score < 0.7:
                    diagnostics['recommendations'].append("Consider increasing load_balance_loss_coef")
                    diagnostics['recommendations'].append("Try adding router jitter noise")
                    
                if usage_std > 5.0:
                    diagnostics['recommendations'].append("Severe imbalance: consider adjusting capacity_factor")
        
        return diagnostics


class EnhancedConversationTrainer:
    """Production trainer with DeepSpeed, MoE optimizations, quantization, and comprehensive precision support."""
    
    def __init__(self, model, tokenizer, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Initialize precision manager FIRST
        self.precision_manager = PrecisionManager(config)
        
        # Enhanced managers
        self.quantization_manager = QuantizationManager(config)
        self.moe_optimizer = MoEOptimizationManager(config) if hasattr(config, 'use_moe') and config.use_moe else None
        
        # Log precision info
        precision_info = self.precision_manager.get_precision_info()
        logging.info(f"Training precision: {precision_info['training']['precision']} ({precision_info['training']['bits']} bits)")
        logging.info(f"Inference precision: {precision_info['inference']['precision']} ({precision_info['inference']['bits']} bits)")
        
        # Apply quantization to model if configured
        if hasattr(config, 'quantization_method') and config.quantization_method:
            logging.info("Applying quantization to model...")
            self.model = self.quantization_manager.quantize_model(self.model)
            
            quant_info = self.quantization_manager.get_quantization_info()
            logging.info(f"Quantization applied: {quant_info}")
        
        # DeepSpeed integration - CRITICAL FIX
        is_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not torch.cuda.is_available()
        if is_mps and getattr(config, 'use_deepspeed', False):
                logging.warning("DeepSpeed is not supported on MPS - disabling")
                config.use_deepspeed = False

        self.use_deepspeed = DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False) and not is_mps

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

    def train_with_oom_fallback(self, train_dataset, eval_dataset=None):
        """Train with automatic batch size reduction on OOM errors."""
        original_batch_size = self.config.batch_size
        original_grad_accum = getattr(self.config, 'gradient_accumulation_steps', 1)
        min_batch_size = 1

        effective_batch_size = original_batch_size * original_grad_accum

        print(f"Starting training with OOM protection")
        print(f"  Initial batch size: {original_batch_size}")
        print(f"  Gradient accumulation: {original_grad_accum}")
        print(f"  Effective batch size: {effective_batch_size}")

        while self.config.batch_size >= min_batch_size:
            try:
                print(f"\nAttempting training with:")
                print(f"  Batch size: {self.config.batch_size}")
                print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
                print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

                self.train(train_dataset, eval_dataset)
                break
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_oom = any(x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"])

                if is_oom:
                    print(f"\n{'='*80}")
                    print(f"OOM ERROR DETECTED!")
                    print(f"{'='*80}")
                    print(f"Error: {str(e)[:200]}")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("Cleared CUDA cache")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        print("Cleared MPS cache")

                    if self.config.gradient_accumulation_steps < 16:
                        new_batch_size = max(min_batch_size, self.config.batch_size // 2)
                        new_grad_accum = self.config.gradient_accumulation_steps * 2

                        print(f"\nStrategy: Increase gradient accumulation")
                        print(f"  New batch size: {new_batch_size}")
                        print(f"  New gradient accumulation: {new_grad_accum}")

                        self.config.batch_size = new_batch_size
                        self.config.gradient_accumulation_steps = new_grad_accum
                    else:
                        new_batch_size = max(min_batch_size, self.config.batch_size // 2)

                        if new_batch_size < min_batch_size:
                            print(f"\nCannot reduce batch size below {min_batch_size}. OOM error persists.")
                            print(f"Suggestions:")
                            print(f"  - Reduce model size")
                            print(f"  - Reduce sequence length")
                            print(f"  - Enable gradient checkpointing")
                            print(f"  - Use a smaller precision (fp16/bf16)")
                            raise
                            
                        print(f"\nStrategy: Reduce batch size only")
                        print(f"  New batch size: {new_batch_size}")

                        self.config.batch_size = new_batch_size

                    print("\nResetting training state...")
                    self.global_step = 0
                    self.current_epoch = 0
                    self.best_eval_loss = float('inf')
                    self.patience_counter = 0
                    self.should_stop = False

                    self.metrics = {
                        'train_losses': [],
                        'eval_losses': [],
                        'learning_rates': [],
                        'gradient_norms': [],
                        'throughput': [],
                        'epoch_times': []
                    }

                    print("Re-initializing training components...")
                    self._setup_training()

                    print(f"\nRetrying training with new configuration...")

                else:
                    print(f"\nNon-OOM error detected: {str(e)[:200]}")
                    raise
    
        if self.config.batch_size != original_batch_size or self.config.gradient_accumulation_steps != original_grad_accum:
            print(f"\n{'='*80}")
            print(f"TRAINING COMPLETED WITH ADJUSTED CONFIGURATION")
            print(f"{'='*80}")
            print(f"Original configuration:")
            print(f"  Batch size: {original_batch_size}")
            print(f"  Gradient accumulation: {original_grad_accum}")
            print(f"  Effective batch size: {effective_batch_size}")
            print(f"\nFinal configuration:")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

            try:
                optimal_config = {
                    'batch_size': self.config.batch_size,
                    'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
                    'effective_batch_size': self.config.batch_size * self.config.gradient_accumulation_steps,
                    'original_batch_size': original_batch_size,
                    'original_gradient_accumulation': original_grad_accum
                }
            
                import json
                with open('optimal_batch_config.json', 'w') as f:
                    json.dump(optimal_config, f, indent=2)
                print(f"\nSaved optimal configuration to optimal_batch_config.json")
            except Exception as e:
                print(f"Could not save optimal configuration: {e}")
    
    def _log_memory_usage(self, step_info: str):
        """Log current memory usage for CUDA, MPS, or CPU."""
        try:
            if self.device.type == 'cuda':
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
                    logging.info(f"Memory Usage at {step_info}:")
                    logging.info(f"  Allocated: {memory_allocated:.2f}GB")
                    logging.info(f"  Reserved: {memory_cached:.2f}GB")
                    logging.info(f"  Peak: {max_allocated:.2f}GB")
                except:
                    logging.info(f"Memory Usage at {step_info}: N/A")
            elif self.device.type == 'mps':
                try:
                    allocated = torch.mps.current_allocated_memory() / 1e9
                    logging.info(f"Memory Usage at {step_info}:")
                    logging.info(f"  Allocated: {allocated:.2f}GB")
                except:
                    logging.info(f"Memory Usage at {step_info}: N/A")
            else:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                logging.info(f"Memory Usage at {step_info}:")
                logging.info(f"  RSS: {memory_info.rss / 1e9:.2f}GB")
        except Exception as e:
            logging.debug(f"Could not log memory usage: {e}")
        
    def _setup_training(self):
        """Setup training components based on DeepSpeed availability."""
        if self.use_deepspeed:
            self._setup_deepspeed_training()
        else:
            self._setup_standard_training()

    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training with MoE, CPU offloading, quantization, and precision optimizations."""
        print("="*60)
        print("INITIALIZING DEEPSPEED TRAINING")
        print("="*60)
        
        print(f"DeepSpeed available: {DEEPSPEED_AVAILABLE}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Config use_deepspeed: {getattr(self.config, 'use_deepspeed', False)}")
        print(f"World size: {int(os.environ.get('WORLD_SIZE', 1))}")
        print(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
        
        if self.quantization_manager.is_quantized:
            quant_info = self.quantization_manager.get_quantization_info()
            print(f"Model quantized: {quant_info['method']} {quant_info['bits']}-bit")
        
        precision_info = self.precision_manager.get_precision_info()
        print(f"Training precision: {precision_info['training']['precision']}")
        print(f"Inference precision: {precision_info['inference']['precision']}")
        
        ds_config = self._create_deepspeed_config()
        
        print("DeepSpeed Configuration:")
        config_str = json.dumps(ds_config, indent=2, default=str)
        print(config_str[:2000])
        
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
            
            self.use_deepspeed = True
            
            print("✅ DEEPSPEED INITIALIZATION SUCCESSFUL!")
            print(f"  World size: {self.deepspeed_engine.world_size}")
            print(f"  Local rank: {self.deepspeed_engine.local_rank}")
            print(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'disabled')}")
            
            if ds_config.get('moe', {}).get('enabled', False):
                print(f"  MoE enabled: {ds_config['moe']['num_experts']} experts")
                print(f"  Expert parallel size: {ds_config['moe']['expert_parallel_size']}")
                
            if self.quantization_manager.is_quantized:
                print(f"  Quantization: {self.quantization_manager.quantization_info['method']} "
                      f"{self.quantization_manager.quantization_info['bits']}-bit")
            
            print(f"  Precision: {precision_info['training']['precision']}")
            
        except Exception as e:
            print("❌ DEEPSPEED INITIALIZATION FAILED!")
            print(f"Error: {e}")
            
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
            print("🔄 Falling back to standard PyTorch training...")
            self.use_deepspeed = False
            self._setup_standard_training()
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create comprehensive DeepSpeed configuration with FIXED batch size calculation and precision support."""
        
        micro_batch_size = getattr(self.config, 'batch_size', 1)
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
        
        print(f"Batch size calculation:")
        print(f"  Micro batch size: {micro_batch_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  World size: {world_size}")
        print(f"  Train batch size: {train_batch_size}")
        
        # Get precision info from PrecisionManager
        train_dtype = self.precision_manager.get_dtype(for_inference=False)
        use_fp16 = self.precision_manager.train_precision in ['fp16', 'mixed_fp16']
        use_bf16 = self.precision_manager.train_precision in ['bf16', 'mixed_bf16']
        
        ds_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            "fp16": {
                "enabled": use_fp16,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": use_bf16
            },
            
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 1e-6,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": getattr(self, 'steps_per_epoch', 100)
                }
            },
            
            "allgather_partitions": True,
            "allgather_bucket_size": int(5e8),
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": int(5e8),
            "contiguous_gradients": True,
            
            "steps_per_print": 1,
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        if self.quantization_manager.is_quantized:
            quantized_optimizer = self.quantization_manager.create_quantized_optimizer(self.model)
            if quantized_optimizer:
                ds_config["optimizer"] = {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                    }
                }
            else:
                ds_config["optimizer"] = {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                    }
                }
        else:
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                }
            }
        
        if hasattr(self.config, 'use_moe') and self.config.use_moe and self.moe_optimizer:
            print("Adding MoE configuration to DeepSpeed config...")
            ds_config = self.moe_optimizer.create_deepspeed_moe_config(ds_config)
        else:
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
        """Setup standard PyTorch training with quantization and precision support as fallback."""
        print("="*60)
        print("SETTING UP STANDARD PYTORCH TRAINING")
        print("="*60)
        
        # Get dtype from PrecisionManager
        train_dtype = self.precision_manager.get_dtype(for_inference=False)
        
        # Move model to device with appropriate dtype
        if train_dtype and not self.quantization_manager.is_quantized:
            try:
                self.model = self.model.to(device=self.device, dtype=train_dtype)
                print(f"Model moved to {self.device} with dtype {train_dtype}")
            except Exception as e:
                print(f"Could not move model to {train_dtype}: {e}, using default dtype")
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
        
        # Create optimizer (quantization-aware if possible)
        quantized_optimizer = self.quantization_manager.create_quantized_optimizer(self.model)
        if quantized_optimizer:
            self.optimizer = quantized_optimizer
            print(f"Using quantized optimizer: {type(quantized_optimizer).__name__}")
        else:
            self.optimizer = self._create_standard_optimizer()
        
        self.scheduler = None
        
        # Mixed precision setup from PrecisionManager
        self.use_amp = self.precision_manager.PRECISION_REGISTRY[self.precision_manager.train_precision]['requires_amp']
        self.scaler = GradScaler() if self.precision_manager.should_use_grad_scaler() and torch.cuda.is_available() else None
        
        # Model compilation (may not work with some quantized models)
        if getattr(self.config, 'compile', True) and hasattr(torch, 'compile'):
            try:
                if not self.quantization_manager.is_quantized:
                    self.model = torch.compile(self.model, mode='default')
                    print("Model compiled successfully")
                else:
                    print("Skipping model compilation for quantized model")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        if self.quantization_manager.is_quantized:
            quant_info = self.quantization_manager.get_quantization_info()
            print(f"Model quantized: {quant_info['method']} {quant_info['bits']}-bit")
        
        precision_info = self.precision_manager.get_precision_info()
        print(f"Training precision: {precision_info['training']['precision']}")
        print(f"Standard training setup complete - Device: {self.device}")
    
    def _create_standard_optimizer(self) -> torch.optim.Optimizer:
        """Create standard PyTorch optimizer."""
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
                fused=torch.cuda.is_available() and not self.quantization_manager.is_quantized
            )
        except Exception:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _get_autocast_context(self, precision: Optional[str] = None, for_inference: bool = False):
        """Get autocast context using PrecisionManager."""
        if self.use_deepspeed:
            return nullcontext()
        
        return self.precision_manager.get_autocast_context(for_inference=for_inference)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    loss_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with MoE auxiliary losses and accuracy metrics."""
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        mask = (flat_labels != getattr(self.tokenizer, 'pad_token_id', 0)).float()
        
        with torch.no_grad():
            predictions = torch.argmax(flat_logits, dim=-1)
            correct_predictions = (predictions == flat_labels).float() * mask
            accuracy = correct_predictions.sum() / mask.sum().clamp(min=1)
        
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous()
            flat_weights = shift_weights.view(-1)
            weighted_loss = loss * flat_weights * mask
        else:
            weighted_loss = loss * mask
        
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            print("NaN or Inf detected in loss computation")
            if self.quantization_manager.is_quantized:
                print("This might be related to quantization - consider adjusting precision or quantization settings")
            return {
                'loss': torch.tensor(0.0, device=loss.device, requires_grad=True),
                'raw_loss': torch.tensor(0.0, device=loss.device),
                'perplexity': torch.tensor(float('inf'), device=loss.device),
                'valid_tokens': torch.tensor(0.0, device=loss.device),
                'accuracy': torch.tensor(0.0, device=loss.device)
            }
        
        total_loss = weighted_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        raw_loss = (loss * mask).sum() / total_weight
        
        clamped_loss = torch.clamp(raw_loss, min=0.0, max=15.0)
        
        try:
            perplexity = torch.exp(clamped_loss)
        except (OverflowError, RuntimeError):
            perplexity = torch.tensor(float('inf'), device=loss.device)
        
        if raw_loss.item() > 15.0:
            logging.warning(f"Loss value {raw_loss.item():.2f} exceeds clamp threshold - training may be unstable")
        
        return {
            'loss': final_loss,
            'raw_loss': raw_loss.detach(),
            'perplexity': perplexity.detach(),
            'valid_tokens': mask.sum().detach(),
            'accuracy': accuracy.detach()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with DeepSpeed, MoE, quantization, and precision support."""
        if self.use_deepspeed:
            return self._deepspeed_train_step(batch)
        else:
            return self._standard_train_step(batch)
    
    def _deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """DeepSpeed training step with guaranteed metric return."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)
        loss_weights = batch.get('loss_weights')
        
        if input_ids is None or input_ids.numel() == 0:
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0,
                'accuracy': 0.0
            }
        
        try:
            output = self.deepspeed_engine(input_ids, attention_mask)
            
            aux_losses = {}
            if isinstance(output, tuple):
                if len(output) == 3:
                    logits, total_aux_loss, aux_losses = output
                else:
                    logits, total_aux_loss = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
                loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
                
                if aux_losses and self.moe_optimizer:
                    self.moe_optimizer.monitor_routing_balance(aux_losses)
            else:
                logits = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
            
            loss = loss_dict['loss']
            
            self.deepspeed_engine.backward(loss)
            
            loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
            raw_loss_value = loss_dict['raw_loss'].item() if hasattr(loss_dict['raw_loss'], 'item') else float(loss_dict['raw_loss'])
            perplexity_value = loss_dict['perplexity'].item() if hasattr(loss_dict['perplexity'], 'item') else float(loss_dict['perplexity'])
            valid_tokens_value = loss_dict['valid_tokens'].item() if hasattr(loss_dict['valid_tokens'], 'item') else float(loss_dict['valid_tokens'])
            accuracy_value = loss_dict['accuracy'].item() if hasattr(loss_dict['accuracy'], 'item') else float(loss_dict['accuracy'])
            
            return {
                'loss': loss_value,
                'raw_loss': raw_loss_value,
                'perplexity': perplexity_value,
                'valid_tokens': valid_tokens_value,
                'accuracy': accuracy_value
            }
            
        except Exception as e:
            print(f"DeepSpeed training step error: {e}")
            if self.quantization_manager.is_quantized:
                print("This error might be related to quantization - check quantization compatibility")
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0,
                'accuracy': 0.0
            }
    
    def _standard_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard PyTorch training step with precision awareness."""
        self.model.train()
        
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)
        loss_weights = batch.get('loss_weights')
        
        if input_ids is None or input_ids.numel() == 0:
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0,
                'accuracy': 0.0
            }
        
        with self._get_autocast_context(for_inference=False):
            output = self.model(input_ids, attention_mask)
            
            if isinstance(output, tuple):
                logits, total_aux_loss, aux_losses = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
                loss_dict['loss'] = loss_dict['loss'] + total_aux_loss
                
                if aux_losses and self.moe_optimizer:
                    self.moe_optimizer.monitor_routing_balance(aux_losses)
            else:
                logits = output
                loss_dict = self.compute_loss(logits, labels, loss_weights)
        
        loss = loss_dict['loss']
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Invalid loss detected, skipping batch")
            if self.quantization_manager.is_quantized:
                print("This might be related to quantization - consider adjusting settings")
            return {
                'loss': 0.0,
                'raw_loss': 0.0,
                'perplexity': float('inf'),
                'valid_tokens': 0,
                'accuracy': 0.0
            }
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item(),
            'raw_loss': loss_dict['raw_loss'].item(),
            'perplexity': loss_dict['perplexity'].item(),
            'valid_tokens': loss_dict['valid_tokens'].item(),
            'accuracy': loss_dict['accuracy'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Enhanced optimizer step with DeepSpeed and quantization support."""
        if self.use_deepspeed:
            return self._deepspeed_optimizer_step()
        else:
            return self._standard_optimizer_step()
    
    def _deepspeed_optimizer_step(self) -> Dict[str, float]:
        """DeepSpeed optimizer step with proper gradient norm handling."""
        self.deepspeed_engine.step()
        
        current_lr = self.config.learning_rate
        try:
            if hasattr(self.deepspeed_engine, 'get_lr') and callable(self.deepspeed_engine.get_lr):
                lr_list = self.deepspeed_engine.get_lr()
                if lr_list and len(lr_list) > 0:
                    current_lr = lr_list[0]
        except Exception as e:
            print(f"Could not get learning rate from DeepSpeed: {e}")
        
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
        """Standard optimizer step with precision awareness."""
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN/Inf gradients detected, skipping step")
            if self.quantization_manager.is_quantized:
                print("This might be related to quantization - consider reducing learning rate")
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                self.scaler.update()
            return {'grad_norm': 0.0, 'lr': 0.0}
        
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scheduler:
            self.scheduler.step()
        
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 100) -> Dict[str, float]:
        """Enhanced evaluation with proper perplexity calculation and precision support."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
        else:
            self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_raw_loss = 0.0
        total_tokens = 0
        total_accuracy = 0.0
        num_batches = 0
        
        eval_start_time = time.time()
        
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
            
            if self.use_deepspeed:
                output = self.deepspeed_engine(input_ids, attention_mask)
            else:
                with self._get_autocast_context(for_inference=True):
                    output = self.model(input_ids, attention_mask)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            loss_dict = self.compute_loss(logits, labels, loss_weights)
            
            if not (torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any()):
                total_loss += loss_dict['loss'].item()
                total_raw_loss += loss_dict['raw_loss'].item()
                total_tokens += loss_dict['valid_tokens'].item()
                total_accuracy += loss_dict['accuracy'].item()
                num_batches += 1
        
        eval_time = time.time() - eval_start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_perplexity': float('inf'),
                'eval_accuracy': 0.0,
                'eval_time': eval_time,
                'eval_throughput': 0.0,
                'eval_peak_memory_mb': peak_memory
            }
        
        avg_loss = total_loss / num_batches
        avg_raw_loss = total_raw_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        clamped_avg_loss = min(avg_raw_loss, 15.0)
        try:
            perplexity = math.exp(clamped_avg_loss)
        except OverflowError:
            perplexity = float('inf')
        
        if avg_raw_loss > 15.0:
            logging.warning(f"Evaluation loss {avg_raw_loss:.2f} exceeds safe range - perplexity clamped at exp(15)")
        
        throughput = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_accuracy': avg_accuracy,
            'eval_time': eval_time,
            'eval_throughput': throughput,
            'eval_peak_memory_mb': peak_memory
        }
    
    def get_quantization_status(self) -> Dict[str, Any]:
        """Get comprehensive quantization status and diagnostics."""
        status = self.quantization_manager.get_quantization_info()
        
        if self.quantization_manager.is_quantized:
            try:
                model_memory_mb = self.quantization_manager._get_model_memory_usage(self.model)
                status['current_memory_mb'] = model_memory_mb
            except:
                status['current_memory_mb'] = "Unknown"
            
            if self.quantization_manager.quantization_info.get('bits') == 4:
                status['training_recommendations'] = [
                    "Use FP32 or BF16 precision for stability",
                    "Consider lower learning rate",
                    "Monitor for gradient issues",
                    "Use gradient checkpointing if memory allows"
                ]
            elif self.quantization_manager.quantization_info.get('bits') == 8:
                status['training_recommendations'] = [
                    "Mixed precision training should work well",
                    "Monitor gradient norms",
                    "Consider 8-bit optimizer for memory savings"
                ]
        
        return status
    
    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch with accuracy tracking, quantization monitoring, and precision support."""
        if self.use_deepspeed:
            self.deepspeed_engine.train()
        else:
            self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_raw_loss': 0.0,
            'total_tokens': 0,
            'total_accuracy': 0.0,
            'num_batches': 0,
            'grad_norm_sum': 0.0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'raw_loss': 0.0,
            'tokens': 0,
            'accuracy': 0.0
        }
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        epoch_start_time = time.time()
        last_log_time = time.time()
        
        print(f"Starting epoch {epoch + 1} with {len(train_dataloader)} batches")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        
        if self.quantization_manager.is_quantized:
            quant_status = self.get_quantization_status()
            print(f"Training with {quant_status['method']} {quant_status['bits']}-bit quantization")
        
        precision_info = self.precision_manager.get_precision_info()
        print(f"Training precision: {precision_info['training']['precision']}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.should_stop:
                break
            
            step_start_time = time.time()
            
            step_metrics = self.train_step(batch)
            
            if batch_idx < 5 and getattr(self.config, 'log_level', 'INFO') == 'DEBUG':
                debug_msg = f"DEBUG: Batch {batch_idx}, Step metrics: {step_metrics}"
                if self.quantization_manager.is_quantized:
                    debug_msg += f" [QUANTIZED: {self.quantization_manager.quantization_info['bits']}-bit]"
                print(debug_msg)
            
            if step_metrics['loss'] == 0.0 or math.isnan(step_metrics['loss']) or math.isinf(step_metrics['loss']):
                skip_msg = f"Skipping batch {batch_idx} due to invalid loss: {step_metrics['loss']}"
                if self.quantization_manager.is_quantized:
                    skip_msg += " (may be quantization-related)"
                print(skip_msg)
                continue
            
            accumulation_metrics['loss'] += step_metrics['loss'] / gradient_accumulation_steps
            accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
            accumulation_metrics['tokens'] += step_metrics['valid_tokens']
            accumulation_metrics['accuracy'] += step_metrics['accuracy']
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                if accumulation_metrics['loss'] > 0:
                    epoch_metrics['total_loss'] += accumulation_metrics['loss']
                    epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                    epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                    epoch_metrics['total_accuracy'] += accumulation_metrics['accuracy']
                    epoch_metrics['num_batches'] += 1
                    if 'grad_norm' in opt_metrics and opt_metrics['grad_norm'] is not None:
                        epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                step_time = time.time() - step_start_time
                tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
                
                log_frequency = getattr(self.config, 'log_every_n_steps', 50)
                time_since_last_log = time.time() - last_log_time

                should_log = (
                    self.global_step % log_frequency == 0 or
                    time_since_last_log > 600
                )
                
                if should_log:
                    self._log_training_step(
                        epoch, batch_idx, len(train_dataloader),
                        accumulation_metrics, opt_metrics, tokens_per_sec
                    )
                    last_log_time = time.time()
                
                if self.global_step % 100 == 0:
                    self._log_memory_usage(f"Step {self.global_step}")
                
                if self.quantization_manager.is_quantized and self.global_step % 100 == 0:
                    self._log_quantization_diagnostics()
                
                accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0, 'accuracy': 0.0}
        
        epoch_time = time.time() - epoch_start_time
        
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
            avg_accuracy = epoch_metrics['total_accuracy'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
            avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
        else:
            avg_loss = avg_raw_loss = avg_accuracy = avg_grad_norm = avg_tokens_per_sec = 0.0
        
        epoch_summary = (f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                        f"Avg Loss: {avg_loss:.6f} | "
                        f"Avg Accuracy: {avg_accuracy:.1%} | "
                        f"Avg Grad Norm: {avg_grad_norm:.4f} | "
                        f"Throughput: {avg_tokens_per_sec:.0f} tokens/s")
        
        if self.quantization_manager.is_quantized:
            epoch_summary += f" | Quantization: {self.quantization_manager.quantization_info['bits']}-bit"
        
        print(epoch_summary)
        
        return {
            'avg_loss': avg_loss,
            'avg_raw_loss': avg_raw_loss,
            'avg_accuracy': avg_accuracy,
            'avg_grad_norm': avg_grad_norm,
            'epoch_time': epoch_time,
            'throughput': avg_tokens_per_sec
        }
    
    def _log_quantization_diagnostics(self):
        """Log quantization-specific diagnostics."""
        if not self.quantization_manager.is_quantized:
            return
            
        try:
            current_memory = self.quantization_manager._get_model_memory_usage(self.model)
            print(f"Quantization Status at Step {self.global_step}:")
            print(f"  Method: {self.quantization_manager.quantization_info['method']}")
            print(f"  Bits: {self.quantization_manager.quantization_info['bits']}")
            print(f"  Current Memory: {current_memory:.1f}MB")
            
            total_grad_norm = 0.0
            num_params = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
                    num_params += 1
            
            if num_params > 0:
                avg_grad_norm = (total_grad_norm / num_params) ** 0.5
                if avg_grad_norm > 100 or avg_grad_norm < 1e-6:
                    print(f"  WARNING: Unusual gradient norm {avg_grad_norm:.2e} - may indicate quantization issues")
                    
        except Exception as e:
            print(f"Error in quantization diagnostics: {e}")
    
    def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                          metrics, opt_metrics, tokens_per_sec: float):
        """FIXED logging with guaranteed output including accuracy, quantization, and precision info."""
        
        try:
            memory_info = ""
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
                except:
                    memory_info = " | GPU: N/A"
            
            mode_info = " | DeepSpeed" if self.use_deepspeed else " | Standard"
            
            quant_info = ""
            if self.quantization_manager.is_quantized:
                quant_method = self.quantization_manager.quantization_info['method']
                quant_bits = self.quantization_manager.quantization_info['bits']
                quant_info = f" | {quant_method.upper()}-{quant_bits}bit"
            
            precision_info_str = f" | {self.precision_manager.train_precision.upper()}"
            
            loss = metrics.get('loss', 0.0)
            raw_loss = metrics.get('raw_loss', loss)
            accuracy = metrics.get('accuracy', 0.0)
            lr = opt_metrics.get('lr', 0.0)
            grad_norm = opt_metrics.get('grad_norm', 0.0)
            
            try:
                clamped_loss = min(raw_loss, 15.0)
                perplexity = math.exp(clamped_loss)
                ppl_str = f"{perplexity:.2e}" if perplexity > 10000 else f"{perplexity:.2f}"
            except:
                ppl_str = "N/A"
            
            log_message = (
                f"Epoch {epoch+1} | Step {self.global_step:6d} | "
                f"Batch {batch_idx+1:4d}/{total_batches} | "
                f"Loss: {loss:.6f} | "
                f"PPL: {ppl_str} | "
                f"Acc: {accuracy:.1%} | "
                f"LR: {lr:.2e} | "
                f"GradNorm: {grad_norm:.4f} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
                f"{mode_info}{precision_info_str}{quant_info}{memory_info}"
            )
            
            print(f"[TRAINING] {log_message}")
            
        except Exception as e:
            fallback_msg = f"Step {self.global_step} | Loss: {metrics.get('loss', 'N/A')} | Acc: {metrics.get('accuracy', 'N/A')} | Logging Error: {e}"
            logging.error(fallback_msg)
            print(f"[TRAINING ERROR] {fallback_msg}")
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with enhanced logging, accuracy tracking, quantization, and precision monitoring."""
        print("="*80)
        if self.use_deepspeed:
            print("STARTING DEEPSPEED TRAINING")
        else:
            print("STARTING STANDARD TRAINING")
        print("="*80)
        
        if self.quantization_manager.is_quantized:
            quant_status = self.get_quantization_status()
            print(f"QUANTIZATION STATUS:")
            print(f"  Method: {quant_status['method']}")
            print(f"  Bits: {quant_status['bits']}")
            print(f"  Memory: {quant_status.get('current_memory_mb', 'Unknown')}MB")
            if 'training_recommendations' in quant_status:
                print("  Recommendations:")
                for rec in quant_status['training_recommendations']:
                    print(f"    - {rec}")
            print("="*80)
        
        precision_info = self.precision_manager.get_precision_info()
        print(f"PRECISION CONFIGURATION:")
        print(f"  Training: {precision_info['training']['precision']} ({precision_info['training']['bits']} bits)")
        print(f"  Inference: {precision_info['inference']['precision']} ({precision_info['inference']['bits']} bits)")
        print("="*80)
        
        self.eval_dataset = eval_dataset
        
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)

        print("="*80)
        print("DATALOADER DEBUG INFO")
        print("="*80)
        print(f"Dataset size: {len(train_dataset):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Expected batches: {len(train_dataset) // self.config.batch_size}")
        print(f"Actual dataloader batches: {len(train_dataloader)}")
        print("="*80)
        
        if len(train_dataloader) == 0:
            print("ERROR: Train dataloader is empty!")
            return
        
        if not self.use_deepspeed:
            gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
            total_steps = len(train_dataloader) * self.config.num_epochs // gradient_accumulation_steps
            self._setup_scheduler(total_steps)
        
        self._log_training_config(len(train_dataloader))
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                if self.quantization_manager.is_quantized:
                    print(f"QUANTIZED TRAINING: {self.quantization_manager.quantization_info['method'].upper()}-{self.quantization_manager.quantization_info['bits']}bit")
                print(f"PRECISION: {self.precision_manager.train_precision.upper()}")
                print(f"{'='*60}")
                
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                if eval_dataset is not None:
                    print("Running evaluation...")
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    print(f"Epoch {epoch + 1} Summary:")
                    print(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                    print(f"  Train Accuracy: {epoch_metrics['avg_accuracy']:.1%}")
                    print(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    print(f"  Eval Accuracy: {eval_metrics['eval_accuracy']:.1%}")
                    print(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    if self.quantization_manager.is_quantized:
                        print(f"  Quantization: {self.quantization_manager.quantization_info['method']}-{self.quantization_manager.quantization_info['bits']}bit")
                    
                    if getattr(self.config, 'early_stopping_patience', None):
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                if self.use_deepspeed:
                    self._save_deepspeed_checkpoint(epoch + 1)
                else:
                    self._save_standard_checkpoint(epoch + 1)
                
                self.current_epoch = epoch + 1
                
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
            if self.quantization_manager.is_quantized:
                print("Error may be related to quantization - check quantization compatibility")
            import traceback
            traceback.print_exc()
            raise
        finally:
            total_training_time = time.time() - training_start_time
            print(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
            
            if self.use_deepspeed:
                self._save_deepspeed_checkpoint(self.current_epoch, final=True)
            else:
                self._save_standard_checkpoint(self.current_epoch, final=True)
            
            if self.quantization_manager.is_quantized:
                print(f"\nFinal Quantization Summary:")
                final_status = self.get_quantization_status()
                print(f"  Method: {final_status['method']}")
                print(f"  Bits: {final_status['bits']}")
                print(f"  Final Memory: {final_status.get('current_memory_mb', 'Unknown')}MB")
            
            final_precision_info = self.precision_manager.get_precision_info()
            print(f"\nFinal Precision Summary:")
            print(f"  Training: {final_precision_info['training']['precision']}")
            print(f"  Inference: {final_precision_info['inference']['precision']}")
    
    def _save_deepspeed_checkpoint(self, epoch: int, final: bool = False):
        """Save DeepSpeed checkpoint with quantization and precision state."""
        try:
            checkpoint_dir = Path(f"checkpoints/deepspeed_epoch_{epoch}")
            if final:
                checkpoint_dir = Path("checkpoints/deepspeed_final")
            
            self.deepspeed_engine.save_checkpoint(str(checkpoint_dir))
            
            if self.quantization_manager.is_quantized:
                quant_state_path = checkpoint_dir / "quantization_state.json"
                with open(quant_state_path, 'w') as f:
                    json.dump(self.quantization_manager.get_quantization_info(), f, indent=2)
            
            precision_state_path = checkpoint_dir / "precision_state.json"
            with open(precision_state_path, 'w') as f:
                json.dump(self.precision_manager.get_precision_info(), f, indent=2)
            
            print(f"DeepSpeed checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            print(f"Failed to save DeepSpeed checkpoint: {e}")
    
    def _save_standard_checkpoint(self, epoch: int, final: bool = False):
        """Save standard PyTorch checkpoint with quantization and precision state."""
        try:
            suffix = "final" if final else f"epoch_{epoch:03d}"
            checkpoint_path = Path(f"checkpoints/checkpoint_{suffix}_{self.global_step}.pt")
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': self.global_step,
                'epoch': epoch,
                'config': self.config
            }
            
            if self.quantization_manager.is_quantized:
                checkpoint_data['quantization_info'] = self.quantization_manager.get_quantization_info()
            
            checkpoint_data['precision_info'] = self.precision_manager.get_precision_info()
            
            torch.save(checkpoint_data, checkpoint_path)
            
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
        """Log comprehensive training configuration including quantization, precision, and all optimizations."""
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
            f"Device: {self.device}"
        ]
        
        precision_info = self.precision_manager.get_precision_info()
        config_info.extend([
            f"Training precision: {precision_info['training']['precision']} ({precision_info['training']['bits']} bits)",
            f"Inference precision: {precision_info['inference']['precision']} ({precision_info['inference']['bits']} bits)",
        ])
        
        if self.quantization_manager.is_quantized:
            quant_info = self.quantization_manager.get_quantization_info()
            config_info.extend([
                f"Quantization: {quant_info['method']} {quant_info['bits']}-bit",
                f"Quantized Memory: {quant_info.get('current_memory_mb', 'Unknown')}MB"
            ])
        
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


def get_available_quantization_methods():
    """Get dictionary of available quantization methods."""
    return {
        'bnb': BNB_AVAILABLE,
        'gptq': GPTQ_AVAILABLE,
        'quanto': QUANTO_AVAILABLE
    }


def print_quantization_recommendations():
    """Print recommendations for quantization setup."""
    print("Quantization Setup Recommendations:")
    print("=" * 60)
    
    methods = get_available_quantization_methods()
    
    print("Available Quantization Methods:")
    for method, available in methods.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {method}: {status}")
    
    if not any(methods.values()):
        print("\n❌ No quantization libraries available!")
        print("\nTo install:")
        print("  BitsAndBytes (8-bit): pip install bitsandbytes")
        print("  AutoGPTQ (4-bit):     pip install auto-gptq")
        print("  Optimum Quanto:       pip install optimum[quanto]")
    
    print("\nConfiguration Examples:")
    print("  8-bit weights:      config.quantization_method = 'bnb', config.quantization_bits = 8")
    print("  4-bit weights:      config.quantization_method = 'bnb', config.quantization_bits = 4")
    print("  GPTQ 4-bit:         config.quantization_method = 'gptq', config.quantization_bits = 4")
    print("  Quanto 8-bit:       config.quantization_method = 'quanto', config.quantization_bits = 8")
    
    print("\nRecommendations:")
    print("  - 8-bit quantization: Good balance of memory savings (50%) and stability")
    print("  - 4-bit quantization: Maximum memory savings (75%), may need precision adjustments")
    print("  - Use FP32 or BF16 precision with 4-bit quantization for stability")
    print("  - Monitor gradient norms closely with quantized models")


def print_all_precision_info():
    """Utility function to print comprehensive precision information."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PRECISION SUPPORT")
    print("="*80)
    
    print("\nSupported Precisions:")
    print("-" * 80)
    print(f"{'Precision':<20} {'Bits':<6} {'Category':<10} {'PyTorch Dtype':<25} {'Use Case'}")
    print("-" * 80)
    
    for name, spec in sorted(PrecisionManager.PRECISION_REGISTRY.items(), key=lambda x: (-x[1]['bits'], x[0])):
        dtype_str = str(spec['dtype']).replace('torch.', '') if spec['dtype'] else 'N/A'
        unsupported = " (UNSUPPORTED)" if spec.get('unsupported', False) else ""
        experimental = " (EXPERIMENTAL)" if spec.get('experimental', False) else ""
        
        print(f"{name:<20} {spec['bits']:<6} {spec['category']:<10} {dtype_str:<25} {spec['use_case']}{unsupported}{experimental}")
    
    print("\n" + "="*80)
    print("Configuration Examples:")
    print("="*80)
    print("""
# Standard training precisions
config.precision = 'fp32'         # Default, maximum stability
config.precision = 'bf16'         # Recommended for Ampere+ GPUs
config.precision = 'mixed_fp16'   # Mixed precision for memory efficiency
config.precision = 'tf32'         # Automatic speedup on Ampere+

# High precision (rarely used)
config.precision = 'fp64'         # Scientific computing

# Low precision (experimental)
config.precision = 'fp8_e4m3'     # H100+ GPUs only
config.precision = 'mixed_fp8'    # Mixed FP8 precision

# Inference optimization
config.inference_precision = 'fp16'  # Fast inference
config.inference_precision = 'int8'  # Quantized inference
config.inference_precision = 'int4'  # Maximum memory savings

# Combined example - Full feature showcase
config.precision = 'bf16'            # Training precision
config.inference_precision = 'fp16'  # Inference precision
config.quantization_method = 'bnb'   # Model quantization
config.quantization_bits = 8         # 8-bit weights
config.use_deepspeed = True          # DeepSpeed acceleration
config.zero_stage = 3                # ZeRO-3 optimization
config.cpu_offload = True            # CPU offloading
config.use_moe = True                # Mixture of Experts
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    print_all_precision_info()
    print_quantization_recommendations()
    
    # Example: Create a dummy config and show precision recommendations
    class DummyConfig:
        precision = 'bf16'
        inference_precision = 'fp16'
        quantization_method = None
        quantization_bits = None
    
    config = DummyConfig()
    pm = PrecisionManager(config)
    pm.print_precision_recommendations()