# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback as tb_module
import psutil
import gc
import json
import time
import math
import signal
import shutil
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import warnings
import os
import builtins
import sys

builtins.exit = sys.exit
builtins.quit = sys.exit  # just in case some code calls quit()


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utility modules
try:
    from utils.data_processing import process_oasst_data, validate_data_comprehensive, create_sample_data
    from utils.environment import validate_environment, estimate_training_time, get_system_info
    from utils.reporting import create_data_summary_report, create_training_report
    UTILS_AVAILABLE = True
    print("✓ Utils modules loaded successfully")
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"⚠ Utils modules not available: {e}")

# DeepSpeed imports
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available")

# Import our modules with fallbacks
try:
    from config.config_manager import Config, ConfigPresets
except ImportError:
    try:
        from config_manager import Config, ConfigPresets
    except ImportError:
        print("ERROR: Could not import config classes")
        sys.exit(1)

try:
    from core.tokenizer import ConversationTokenizer
    from core.model import DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, create_dataloader
except ImportError:
    try:
        from tokenizer import ConversationTokenizer
        from model import DeepSeekTransformer, DeepSeekConfig
        from core.dataset import ConversationDataset, create_dataloader, MultiDatasetManager
    except ImportError:
        print("ERROR: Could not import core modules")
        sys.exit(1)

# Import training infrastructure (orchestrator, trainer, checkpoint)
TRAINING_INFRASTRUCTURE_AVAILABLE = False
try:
    from training.orchestrator import AdaptiveTrainingOrchestrator
    from training.trainer import EnhancedConversationTrainer
    from training.checkpoint import CheckpointManager
    TRAINING_INFRASTRUCTURE_AVAILABLE = True
    print("✓ Advanced training infrastructure available (Orchestrator + Trainer + Checkpoint)")
except ImportError:
    try:
        from orchestrator import AdaptiveTrainingOrchestrator
        from trainer import EnhancedConversationTrainer
        from checkpoint import CheckpointManager
        TRAINING_INFRASTRUCTURE_AVAILABLE = True
        print("✓ Advanced training infrastructure available (Orchestrator + Trainer + Checkpoint)")
    except ImportError:
        print("⚠ Advanced training infrastructure not available - will use fallback")

"""
Path Validator Helper
Add this function to Main.py right after your imports and before main()
"""

def validate_data_paths(data_params: dict) -> bool:
    """
    Validate all data paths before training starts.
    Returns True if all paths are valid, False otherwise.
    """
    print("\n" + "="*80)
    print("VALIDATING DATA PATHS")
    print("="*80)
    
    training_mode = data_params.get('training_mode', 'finetuning_only')
    print(f"Training Mode: {training_mode}\n")
    
    all_valid = True
    checked_paths = []
    
    # Check base training paths
    if training_mode in ['base_only', 'hybrid', 'interleaved']:
        base_paths = data_params.get('base_training_paths', [])
        if base_paths:
            print("Base Training Paths:")
            for i, path_str in enumerate(base_paths, 1):
                path = Path(path_str)
                
                # Check if it's a directory (ERROR)
                if path.is_dir():
                    print(f"  ❌ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                    all_valid = False
                # Check if file exists
                elif path.exists() and path.is_file():
                    size = path.stat().st_size / (1024*1024)
                    print(f"  ✅ [{i}] {path_str} ({size:.2f} MB)")
                    checked_paths.append(str(path))
                else:
                    print(f"  ❌ [{i}] {path_str} - File not found!")
                    all_valid = False
    
    # Check fine-tuning paths
    if training_mode in ['finetuning_only', 'hybrid', 'interleaved']:
        ft_paths = data_params.get('finetuning_paths', [])
        if ft_paths:
            print("\nFine-tuning Paths:")
            for i, path_str in enumerate(ft_paths, 1):
                path = Path(path_str)
                
                # Check if it's a directory (ERROR)
                if path.is_dir():
                    print(f"  ❌ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                    all_valid = False
                # Check if file exists
                elif path.exists() and path.is_file():
                    size = path.stat().st_size / (1024*1024)
                    print(f"  ✅ [{i}] {path_str} ({size:.2f} MB)")
                    checked_paths.append(str(path))
                else:
                    print(f"  ❌ [{i}] {path_str} - File not found!")
                    all_valid = False
        else:
            print("\n❌ No fine-tuning paths specified for finetuning_only mode!")
            all_valid = False
    
    # Check eval paths
    eval_paths = []
    if training_mode == 'base_only':
        eval_paths = data_params.get('base_eval_paths', [])
    elif training_mode == 'finetuning_only':
        eval_paths = data_params.get('finetuning_eval_paths', [])
    
    if eval_paths:
        print("\nEvaluation Paths:")
        for i, path_str in enumerate(eval_paths, 1):
            path = Path(path_str)
            
            if path.is_dir():
                print(f"  ❌ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                all_valid = False
            elif path.exists() and path.is_file():
                size = path.stat().st_size / (1024*1024)
                print(f"  ✅ [{i}] {path_str} ({size:.2f} MB)")
            else:
                print(f"  ⚠️  [{i}] {path_str} - File not found (will use training data)")
    
    # Summary
    print("\n" + "="*80)
    if all_valid and checked_paths:
        print(f"✅ VALIDATION PASSED - All {len(checked_paths)} file(s) are valid")
        print("="*80 + "\n")
        return True
    else:
        print("❌ VALIDATION FAILED - Please fix the issues above")
        print("="*80 + "\n")
        return False
    
def validate_mps_compatibility(config) -> Tuple[bool, List[str]]:
    """
    Validate configuration for MPS (Apple Silicon) compatibility.
    
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check if we're actually on MPS
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return True, []  # Not MPS, no issues
    
    # Check DeepSpeed (not supported on MPS)
    if getattr(config, 'use_deepspeed', False):
        issues.append("DeepSpeed is not supported on MPS. Set use_deepspeed=False")
    
    # Check Flash Attention (not supported on MPS)
    if getattr(config, 'use_flash_attention', True):
        issues.append("Flash Attention is not supported on MPS. Set use_flash_attention=False")
    
    # Check precision (BF16 has limited support)
    precision = getattr(config, 'precision', 'fp32')
    if precision in ['bf16', 'mixed_bf16']:
        issues.append("BF16 precision has limited support on MPS. Use 'fp16' or 'fp32' instead")
    
    # Check model compilation (can be problematic)
    if getattr(config, 'compile', False):
        issues.append("Model compilation may cause issues on MPS. Consider setting compile=False")
    
    # Check batch size (MPS uses unified memory)
    batch_size = getattr(config, 'batch_size', 1)
    if batch_size > 8:
        issues.append(f"Large batch size ({batch_size}) may cause memory issues on MPS. Consider starting with batch_size=2-4")
    
    is_compatible = len(issues) == 0
    return is_compatible, issues


# Add this to your main() function right after data_params is defined:
# Example usage in main():
"""
    # After defining data_params:
    if not validate_data_paths(data_params):
        print("\n❌ Data path validation failed. Cannot continue.")
        print("Please check your file paths in the data_params configuration.\n")
        return 1
"""

def wrap_orchestrator_with_oom_protection(orchestrator, train_dataset, eval_dataset):
    """
    Wrap the orchestrator's adaptive training with OOM protection.
    
    This function catches OOM errors from the orchestrator and automatically
    adjusts batch size, then recreates the orchestrator with new settings.
    """
    original_batch_size = orchestrator.config.batch_size
    original_grad_accum = getattr(orchestrator.config, 'gradient_accumulation_steps', 1)
    min_batch_size = 1
    
    attempt = 0
    max_attempts = 10  # Prevent infinite loops
    
    print(f"\n{'='*80}")
    print(f"ADAPTIVE TRAINING WITH OOM PROTECTION")
    print(f"{'='*80}")
    print(f"Initial configuration:")
    print(f"  Batch size: {original_batch_size}")
    print(f"  Gradient accumulation: {original_grad_accum}")
    print(f"  Effective batch size: {original_batch_size * original_grad_accum}")
    print(f"  Max attempts: {max_attempts}")
    
    while orchestrator.config.batch_size >= min_batch_size and attempt < max_attempts:
        attempt += 1
        
        try:
            print(f"\n{'='*80}")
            print(f"TRAINING ATTEMPT {attempt}/{max_attempts}")
            print(f"{'='*80}")
            print(f"Current configuration:")
            print(f"  Batch size: {orchestrator.config.batch_size}")
            print(f"  Gradient accumulation: {orchestrator.config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {orchestrator.config.batch_size * orchestrator.config.gradient_accumulation_steps}")
            
            # Run adaptive training
            orchestrator.run_adaptive_training()
            
            # Success!
            print(f"\n{'='*80}")
            print(f"✓ TRAINING COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            break
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"])
            
            if is_oom:
                print(f"\n{'='*80}")
                print(f"⚠ OOM ERROR DETECTED IN ORCHESTRATOR (Attempt {attempt})")
                print(f"{'='*80}")
                print(f"Error message: {str(e)[:300]}")
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("✓ Cleared CUDA cache")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    print("✓ Cleared MPS cache")
                
                # Force garbage collection
                import gc
                gc.collect()
                print("✓ Ran garbage collection")
                
                # Adaptive strategy for batch size reduction
                current_batch = orchestrator.config.batch_size
                current_grad_accum = orchestrator.config.gradient_accumulation_steps
                
                # Strategy 1: If gradient accumulation is reasonable, increase it and reduce batch size
                if current_grad_accum < 16 and current_batch > 1:
                    new_batch_size = max(1, current_batch // 2)
                    new_grad_accum = min(32, current_grad_accum * 2)
                    strategy = "Halve batch size, double gradient accumulation"
                
                # Strategy 2: Just reduce batch size
                elif current_batch > 1:
                    new_batch_size = max(1, current_batch // 2)
                    new_grad_accum = current_grad_accum
                    strategy = "Halve batch size only"
                
                # Strategy 3: Can't reduce further
                else:
                    print(f"\n{'='*80}")
                    print(f"✗ CANNOT REDUCE BATCH SIZE FURTHER")
                    print(f"{'='*80}")
                    print(f"Current batch size: {current_batch}")
                    print(f"Current gradient accumulation: {current_grad_accum}")
                    print(f"\nSuggestions to reduce memory usage:")
                    print(f"  1. Reduce sequence length (current: {orchestrator.config.seq_length})")
                    print(f"  2. Reduce model size (layers: {orchestrator.config.num_layers}, hidden: {orchestrator.config.hidden_size})")
                    print(f"  3. Enable gradient checkpointing: config.gradient_checkpointing = True")
                    print(f"  4. Use lower precision: config.precision = 'fp16' or 'bf16'")
                    print(f"  5. Enable CPU offloading: config.cpu_offload = True")
                    print(f"  6. Use DeepSpeed ZeRO: config.use_deepspeed = True, config.zero_stage = 3")
                    if hasattr(orchestrator.config, 'use_moe') and orchestrator.config.use_moe:
                        print(f"  7. Reduce number of experts: config.num_experts (current: {orchestrator.config.num_experts})")
                    raise
                
                print(f"\n{'='*80}")
                print(f"APPLYING RECOVERY STRATEGY: {strategy}")
                print(f"{'='*80}")
                print(f"  Previous batch size: {current_batch}")
                print(f"  Previous gradient accumulation: {current_grad_accum}")
                print(f"  Previous effective batch: {current_batch * current_grad_accum}")
                print(f"  →")
                print(f"  New batch size: {new_batch_size}")
                print(f"  New gradient accumulation: {new_grad_accum}")
                print(f"  New effective batch: {new_batch_size * new_grad_accum}")
                
                # Update configuration
                orchestrator.config.batch_size = new_batch_size
                orchestrator.config.gradient_accumulation_steps = new_grad_accum
                
                # Recreate orchestrator with new configuration
                print(f"\nRecreating orchestrator with adjusted configuration...")
                try:
                    # Import here to avoid circular dependency
                    from training.orchestrator import AdaptiveTrainingOrchestrator
                    orchestrator = AdaptiveTrainingOrchestrator(orchestrator.config)
                    print(f"✓ Orchestrator recreated successfully")
                except ImportError:
                    from orchestrator import AdaptiveTrainingOrchestrator
                    orchestrator = AdaptiveTrainingOrchestrator(orchestrator.config)
                    print(f"✓ Orchestrator recreated successfully")
                
                print(f"\nPreparing to retry training (attempt {attempt + 1}/{max_attempts})...")
                
            else:
                # Not an OOM error, re-raise
                print(f"\n{'='*80}")
                print(f"✗ NON-OOM ERROR DETECTED")
                print(f"{'='*80}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)[:500]}")
                raise
        
        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print(f"TRAINING INTERRUPTED BY USER")
            print(f"{'='*80}")
            raise
        
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ UNEXPECTED ERROR")
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)[:500]}")
            raise
    
    # Check if we exhausted attempts
    if attempt >= max_attempts:
        print(f"\n{'='*80}")
        print(f"✗ MAXIMUM ATTEMPTS REACHED")
        print(f"{'='*80}")
        print(f"Training failed after {max_attempts} attempts")
        raise RuntimeError(f"Could not complete training after {max_attempts} OOM recovery attempts")
    
    # Report final configuration
    final_batch = orchestrator.config.batch_size
    final_grad_accum = orchestrator.config.gradient_accumulation_steps
    
    if final_batch != original_batch_size or final_grad_accum != original_grad_accum:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED WITH ADJUSTED CONFIGURATION")
        print(f"{'='*80}")
        print(f"Original → Final:")
        print(f"  Batch size: {original_batch_size} → {final_batch}")
        print(f"  Gradient accumulation: {original_grad_accum} → {final_grad_accum}")
        print(f"  Effective batch size: {original_batch_size * original_grad_accum} → {final_batch * final_grad_accum}")
        
        # Save optimal configuration
        try:
            optimal_config = {
                'batch_size': final_batch,
                'gradient_accumulation_steps': final_grad_accum,
                'effective_batch_size': final_batch * final_grad_accum,
                'original_batch_size': original_batch_size,
                'original_gradient_accumulation': original_grad_accum,
                'attempts_needed': attempt,
                'device': str(orchestrator.trainer.device) if hasattr(orchestrator, 'trainer') else 'unknown'
            }
            
            optimal_path = Path("optimal_batch_config.json")
            with open(optimal_path, 'w') as f:
                json.dump(optimal_config, f, indent=2)
            print(f"\n✓ Saved optimal configuration to {optimal_path}")
            print(f"  Use these settings for future runs to avoid OOM errors")
        except Exception as e:
            print(f"⚠ Could not save optimal configuration: {e}")
    
    return orchestrator


def validate_precision_support(precision: str, device: torch.device) -> Tuple[bool, str]:
    """Validate if the requested precision is supported by the hardware."""
    if precision in ['fp32', 'float32']:
        return True, ""
    
    if device.type == 'cpu':
        if precision in ['fp16', 'mixed_fp16']:
            return False, f"FP16 precision '{precision}' is not supported on CPU. Use 'fp32' or 'bf16' instead."
        elif precision in ['bf16', 'mixed_bf16']:
            try:
                test_tensor = torch.randn(2, 2, dtype=torch.bfloat16)
                _ = test_tensor + test_tensor
                return True, ""
            except:
                return False, f"BF16 precision '{precision}' is not supported on this CPU. Use 'fp32' instead."
    
    if device.type == 'mps':
        # MPS supports FP16 and FP32, but BF16 support is limited
        if precision in ['fp16', 'mixed_fp16']:
            return True, ""
        elif precision in ['bf16', 'mixed_bf16']:
            return False, f"BF16 precision '{precision}' is not fully supported on MPS. Use 'fp16' or 'fp32' instead."
        return True, ""
    
    if device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        
        if precision in ['fp16', 'mixed_fp16']:
            if major > 5 or (major == 5 and minor >= 3):
                return True, ""
            else:
                return False, f"FP16 precision '{precision}' requires compute capability >= 5.3. " \
                             f"Your GPU has {major}.{minor}. Use 'fp32' instead."
        
        elif precision in ['bf16', 'mixed_bf16']:
            if major >= 8:
                return True, ""
            else:
                return False, f"BF16 precision '{precision}' requires compute capability >= 8.0 (Ampere GPU or newer). " \
                             f"Your GPU has {major}.{minor}. Use 'fp16' or 'fp32' instead."
    
    return True, ""


def config_to_deepseek_config(config: Config):
    """
    Convert training Config to DeepSeekConfig.
    
    Args:
        config: Training configuration object
        
    Returns:
        DeepSeekConfig object
    """
    return DeepSeekConfig(
        vocab_size=getattr(config, 'vocab_size', 50257),
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 12),
        num_heads=getattr(config, 'num_heads', 12),
        num_kv_heads=getattr(config, 'num_kv_heads', None),
        intermediate_size=getattr(config, 'intermediate_size', None),
        seq_length=getattr(config, 'seq_length', 2048),
        dropout=getattr(config, 'dropout', 0.0),
        rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        init_std=getattr(config, 'init_std', 0.02),
        use_stable_embedding=getattr(config, 'use_stable_embedding', True),
        tie_word_embeddings=getattr(config, 'tie_word_embeddings', True),
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


def print_banner(title: str, width: int = 80):
    """Print a formatted banner."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def print_system_diagnostics():
    """Print comprehensive system diagnostics using utils."""
    print_banner("SYSTEM DIAGNOSTICS & ENVIRONMENT VALIDATION")
    
    if UTILS_AVAILABLE:
        system_info = get_system_info()
        
        print_section("Hardware Information")
        print(f"  Python Version: {system_info.get('python_version', 'Unknown')[:70]}")
        print(f"  PyTorch Version: {system_info.get('pytorch_version', 'Unknown')}")
        print(f"  CUDA Available: {system_info.get('cuda_available', False)}")
        print(f"  MPS Available: {system_info.get('mps_available', False)}")
        
        # Display primary acceleration method
        if system_info.get('cuda_available'):
            print(f"  Primary Accelerator: NVIDIA CUDA")
        elif system_info.get('mps_available'):
            print(f"  Primary Accelerator: Apple Silicon (MPS)")
        else:
            print(f"  Primary Accelerator: CPU Only")
        
        # CUDA-specific information
        if system_info.get('cuda_available'):
            print(f"  CUDA Version: {system_info.get('cuda_version', 'Unknown')}")
            print(f"  GPU Count: {system_info.get('gpu_count', 0)}")
            print(f"  GPU Model: {system_info.get('gpu_name', 'Unknown')}")
            print(f"  GPU Memory: {system_info.get('gpu_memory_gb', 0):.2f} GB")
            
            # Additional GPU info
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    compute_cap = torch.cuda.get_device_capability(i)
                    print(f"  GPU {i} Details:")
                    print(f"    Name: {props.name}")
                    print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                    print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
                    print(f"    Multi-Processors: {props.multi_processor_count}")
        
        # MPS-specific information
        if system_info.get('mps_available'):
            print(f"  MPS Device: {system_info.get('device_type', 'Apple Silicon')}")
            print(f"  MPS Backend: {system_info.get('mps_backend', 'Metal Performance Shaders')}")
            print(f"  Unified Memory Architecture: Yes")
            
            # Check PyTorch version for MPS compatibility
            pytorch_version = system_info.get('pytorch_version', '0.0.0')
            try:
                major, minor = pytorch_version.split('.')[:2]
                version_num = float(f"{major}.{minor}")
                if version_num >= 2.0:
                    print(f"  MPS Support Level: Full (PyTorch >= 2.0)")
                elif version_num >= 1.12:
                    print(f"  MPS Support Level: Beta (PyTorch >= 1.12)")
                else:
                    print(f"  MPS Support Level: Not Available (Need PyTorch >= 1.12)")
            except:
                print(f"  MPS Support Level: Unknown")
            
            # MPS capabilities and limitations
            print(f"  MPS Capabilities:")
            print(f"    - FP32 Training: ✓ Supported")
            print(f"    - FP16 Training: ✓ Supported")
            print(f"    - BF16 Training: ✗ Limited Support (use FP16 instead)")
            print(f"    - Flash Attention: ✗ Not Supported (CUDA only)")
            print(f"    - DeepSpeed: ✗ Not Supported (CUDA only)")
            print(f"    - Gradient Checkpointing: ✓ Supported")
            print(f"    - Mixed Precision: ✓ Supported (FP16)")
        
        print_section("System Resources")
        if system_info.get('system_memory_gb'):
            total_mem = system_info.get('system_memory_gb', 0)
            avail_mem = system_info.get('available_memory_gb', 0)
            used_mem = total_mem - avail_mem
            mem_percent = (used_mem / total_mem * 100) if total_mem > 0 else 0
            print(f"  Total System Memory: {total_mem:.2f} GB")
            print(f"  Available Memory: {avail_mem:.2f} GB")
            print(f"  Used Memory: {used_mem:.2f} GB ({mem_percent:.1f}%)")
            
            # Add MPS-specific memory note
            if system_info.get('mps_available'):
                print(f"  Note: MPS uses unified memory (shared with system)")
                recommended_mem = 16.0  # GB
                if total_mem < recommended_mem:
                    print(f"  ⚠️  Warning: {recommended_mem:.0f}GB+ recommended for MPS training")
        
        if system_info.get('cpu_count'):
            print(f"  CPU Cores: {system_info.get('cpu_count', 0)}")
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    print(f"  CPU Frequency: {cpu_freq.current:.0f} MHz")
                cpu_percent = psutil.cpu_percent(interval=0.1)
                print(f"  CPU Usage: {cpu_percent:.1f}%")
            except:
                pass
        
        if system_info.get('disk_space_gb'):
            disk_free = system_info.get('disk_space_gb', 0)
            try:
                disk_usage = shutil.disk_usage(".")
                disk_total = disk_usage.total / 1e9
                disk_used = disk_usage.used / 1e9
                disk_percent = (disk_used / disk_total * 100) if disk_total > 0 else 0
                print(f"  Total Disk Space: {disk_total:.2f} GB")
                print(f"  Used Disk Space: {disk_used:.2f} GB ({disk_percent:.1f}%)")
                print(f"  Free Disk Space: {disk_free:.2f} GB")
            except:
                print(f"  Free Disk Space: {disk_free:.2f} GB")
        
        print_section("Environment Validation")
        issues = validate_environment()
        if issues:
            print("  Detected Issues:")
            for i, issue in enumerate(issues, 1):
                # Color code issues
                if "MPS" in issue or "Apple Silicon" in issue:
                    symbol = "ℹ️ "
                elif "insufficient" in issue.lower() or "low" in issue.lower():
                    symbol = "⚠️  "
                elif "slow" in issue.lower() or "not available" in issue.lower():
                    symbol = "⚠️  "
                else:
                    symbol = "   "
                print(f"    {symbol}{i}. {issue}")
        else:
            print("  ✓ All environment checks passed successfully")
        
        # Check for required libraries
        print_section("Library Availability")
        libraries = {
            'DeepSpeed': DEEPSPEED_AVAILABLE,
            'Training Infrastructure': TRAINING_INFRASTRUCTURE_AVAILABLE,
            'Utils': UTILS_AVAILABLE,
        }
        
        # Add MPS-specific library notes
        if system_info.get('mps_available'):
            libraries['Flash Attention (CUDA only)'] = False
            libraries['MPS Fallback'] = True  # Always available with MPS
        
        for lib, available in libraries.items():
            status = "Available" if available else "Not Available"
            symbol = "✓" if available else "✗"
            
            # Special handling for expected unavailability on MPS
            if system_info.get('mps_available') and 'CUDA only' in lib:
                symbol = "ℹ️ "
                status = "Not Available (Expected on MPS)"
            
            print(f"  {symbol} {lib}: {status}")
        
        # Device-specific recommendations
        print_section("Hardware-Specific Recommendations")
        
        if system_info.get('mps_available'):
            print("  Apple Silicon (MPS) Detected:")
            print("    1. Use FP16 precision for optimal performance")
            print("    2. Disable DeepSpeed (not supported)")
            print("    3. Disable Flash Attention (CUDA only)")
            print("    4. Start with smaller batch sizes")
            print("    5. Monitor unified memory usage")
            print("    6. Consider disabling model compilation initially")
            print("    7. Enable MPS fallback for unsupported ops")
            print("")
            print("  Recommended Configuration:")
            print("    config.precision = 'fp16'")
            print("    config.use_deepspeed = False")
            print("    config.use_flash_attention = False")
            print("    config.compile = False")
            print("    config.batch_size = 2  # Start small")
            
        elif system_info.get('cuda_available'):
            gpu_memory = system_info.get('gpu_memory_gb', 0)
            print("  NVIDIA CUDA Detected:")
            
            if gpu_memory >= 80:
                print("    - High-end GPU detected (80GB+)")
                print("    - Can train large models with full features")
                print("    - Flash Attention recommended")
                print("    - DeepSpeed ZeRO-3 for very large models")
            elif gpu_memory >= 40:
                print("    - Professional GPU detected (40GB+)")
                print("    - Can train medium to large models")
                print("    - Flash Attention recommended")
                print("    - Consider DeepSpeed ZeRO-2/3")
            elif gpu_memory >= 16:
                print("    - Consumer high-end GPU detected (16GB+)")
                print("    - Can train small to medium models")
                print("    - Flash Attention recommended")
                print("    - Consider gradient checkpointing")
            elif gpu_memory >= 8:
                print("    - Consumer mid-range GPU detected (8GB+)")
                print("    - Limited to small models")
                print("    - Gradient checkpointing essential")
                print("    - Small batch sizes required")
            else:
                print("    - Limited GPU memory detected")
                print("    - Consider CPU training or upgrading GPU")
            
            # Check compute capability
            if torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability(0)
                major = compute_cap[0]
                
                if major >= 8:
                    print("    - Ampere or newer: BF16 supported")
                    print("    - Recommended precision: mixed_bf16")
                elif major >= 7:
                    print("    - Volta/Turing: FP16 supported")
                    print("    - Recommended precision: mixed_fp16")
                else:
                    print("    - Older architecture: FP32 only")
                    print("    - Recommended precision: fp32")
        else:
            print("  CPU Only Detected:")
            print("    - Training will be significantly slower")
            print("    - Recommended for debugging only")
            print("    - Consider cloud GPU instances for production")
            print("    - Use smaller models and batch sizes")
        
        # Memory recommendations
        print_section("Memory Recommendations")
        if system_info.get('mps_available'):
            total_mem = system_info.get('system_memory_gb', 0)
            if total_mem >= 32:
                print("  ✓ Excellent memory (32GB+) for MPS training")
            elif total_mem >= 16:
                print("  ✓ Good memory (16GB+) for MPS training")
            elif total_mem >= 8:
                print("  ⚠️  Limited memory (8GB) - use small models only")
            else:
                print("  ⚠️  Insufficient memory (<8GB) - not recommended")
        elif system_info.get('cuda_available'):
            gpu_mem = system_info.get('gpu_memory_gb', 0)
            sys_mem = system_info.get('system_memory_gb', 0)
            print(f"  GPU Memory: {gpu_mem:.1f}GB")
            print(f"  System Memory: {sys_mem:.1f}GB")
            if sys_mem < 16:
                print("  ⚠️  Consider 16GB+ system RAM for optimal performance")
        
    else:
        print("  Utils not available - showing basic diagnostics only")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  Device: Apple Silicon with MPS")
            try:
                import psutil
                total_mem = psutil.virtual_memory().total / 1e9
                print(f"  Unified Memory: {total_mem:.2f} GB")
            except:
                print(f"  Unified Memory: Unknown")
    
    print("="*80)


def prepare_and_validate_data(config, tokenizer):
    """
    Prepare and validate training data with comprehensive reporting.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        
    Returns:
        Dictionary containing dataset statistics
    """
    print_banner("DATA PREPARATION & VALIDATION")
    
    train_data_path = Path(config.train_data_path)
    eval_data_path = Path(config.eval_data_path)
    
    # Process OASST data if raw data is available
    if UTILS_AVAILABLE and hasattr(config, 'raw_oasst_path'):
        raw_path = Path(config.raw_oasst_path)
        if raw_path.exists() and not train_data_path.exists():
            print_section("Processing Raw OASST Data")
            print(f"  Input File: {raw_path}")
            print(f"  Output File: {train_data_path}")
            print(f"  Max Conversations: {config.max_conversations_per_file}")
            
            processing_start = time.time()
            num_processed = process_oasst_data(
                str(raw_path),
                str(train_data_path),
                max_conversations=config.max_conversations_per_file
            )
            processing_time = time.time() - processing_start
            
            print(f"  Processed {num_processed:,} conversations in {processing_time:.2f} seconds")
            print(f"  Processing rate: {num_processed/processing_time:.0f} conversations/sec")
    
    # Create sample data if needed
    if not train_data_path.exists():
        print_section("Creating Sample Data")
        if UTILS_AVAILABLE:
            print(f"  Training data not found: {train_data_path}")
            print(f"  Creating sample data for testing...")
            num_samples = 200
            create_sample_data(str(train_data_path), num_conversations=num_samples)
            print(f"  Created {num_samples} sample conversations")
        else:
            raise FileNotFoundError(f"Training data not found: {train_data_path}")
    
    # Validate data
    datasets_info = {}
    
    if UTILS_AVAILABLE:
        print_section("Validating Training Data")
        validation_start = time.time()
        train_stats = validate_data_comprehensive(str(train_data_path), tokenizer, max_check=5000)
        validation_time = time.time() - validation_start
        
        print(f"\n  File Information:")
        print(f"    Path: {train_data_path}")
        print(f"    Size: {train_stats['file_info'].get('size_mb', 0):.2f} MB")
        print(f"    Last Modified: {train_stats['file_info'].get('modified', 'Unknown')}")
        print(f"    Validation Time: {validation_time:.2f} seconds")
        
        print(f"\n  Conversation Statistics:")
        conv_stats = train_stats['conversation_stats']
        print(f"    Total Lines: {conv_stats.get('total_lines', 0):,}")
        print(f"    Valid Conversations: {conv_stats.get('valid_conversations', 0):,}")
        print(f"    Invalid Conversations: {conv_stats.get('invalid_conversations', 0):,}")
        print(f"    Success Rate: {train_stats['quality_metrics'].get('success_rate', 0):.2%}")
        print(f"    Avg Messages/Conversation: {conv_stats.get('avg_messages_per_conversation', 0):.1f}")
        print(f"    Max Messages: {conv_stats.get('max_messages', 0)}")
        print(f"    Min Messages: {conv_stats.get('min_messages', 0)}")
        
        print(f"\n  Token Statistics:")
        token_stats = train_stats['token_stats']
        print(f"    Average Tokens: {token_stats.get('avg_tokens', 0):.0f}")
        print(f"    Median Tokens: {token_stats.get('median_tokens', 0):.0f}")
        print(f"    Max Tokens: {token_stats.get('max_tokens', 0):,}")
        print(f"    Min Tokens: {token_stats.get('min_tokens', 0):,}")
        print(f"    Std Dev Tokens: {token_stats.get('std_tokens', 0):.1f}")
        
        print(f"\n  Role Distribution:")
        role_dist = conv_stats.get('role_distribution', {})
        total_messages = sum(role_dist.values())
        for role, count in sorted(role_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            print(f"    {role.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n  Quality Metrics:")
        quality = train_stats['quality_metrics']
        print(f"    Error Rate: {quality.get('error_rate', 0):.2%}")
        print(f"    Total Quality Issues: {quality.get('total_quality_issues', 0)}")
        
        issues = quality.get('quality_issues_sample', [])
        if issues:
            print(f"    Sample Issues (first 5):")
            for i, issue in enumerate(issues[:5], 1):
                print(f"      {i}. {issue}")
        
        datasets_info['train'] = train_stats
        
        # Validate eval data if different
        if eval_data_path.exists() and eval_data_path != train_data_path:
            print_section("Validating Evaluation Data")
            eval_stats = validate_data_comprehensive(str(eval_data_path), tokenizer, max_check=2000)
            print(f"  Eval Conversations: {eval_stats['conversation_stats'].get('valid_conversations', 0):,}")
            print(f"  Eval Success Rate: {eval_stats['quality_metrics'].get('success_rate', 0):.2%}")
            datasets_info['eval'] = eval_stats
        
        # Generate comprehensive data report
        if getattr(config, 'generate_data_reports', True):
            print_section("Generating Data Summary Report")
            report_paths = [str(train_data_path)]
            if eval_data_path.exists() and eval_data_path != train_data_path:
                report_paths.append(str(eval_data_path))
            
            report_file = "data_summary_report.html"
            create_data_summary_report(report_paths, tokenizer, report_file)
            print(f"  HTML report saved: {report_file}")
    else:
        print("  Utils not available - skipping detailed validation")
        print(f"  Training data exists: {train_data_path.exists()}")
    
    print("="*80)
    return datasets_info


def estimate_and_display_training_time(config, dataset_size: int):
    """
    Estimate and display training time using utils.
    
    Args:
        config: Training configuration
        dataset_size: Number of samples in dataset
    """
    if not UTILS_AVAILABLE:
        print("  Utils not available - skipping estimation")
        return
    
    print_section("Time Estimates")
    
    try:
        estimates = estimate_training_time(config, dataset_size)
        
        # Display basic estimates
        print(f"  Total Tokens to Process: {estimates['total_tokens']:,}")
        print(f"  Estimated Throughput: {estimates['tokens_per_second']:,.0f} tokens/second")
        print(f"  Device Type: {estimates['device_type']}")
        
        # Time estimates
        print(f"\n  Estimated Training Time:")
        hours = estimates['estimated_hours']
        
        # Convert to human-readable format
        if hours < 0.1:
            human_readable = f"{hours * 60:.1f} minutes"
        elif hours < 1:
            human_readable = f"{hours * 60:.0f} minutes"
        elif hours < 24:
            human_readable = f"{hours:.1f} hours"
        else:
            days = int(hours // 24)
            remaining_hours = hours % 24
            human_readable = f"{days} days, {remaining_hours:.1f} hours"
        
        print(f"    Total Hours: {hours:.2f}")
        print(f"    Total Days: {estimates['estimated_days']:.2f}")
        print(f"    Human Readable: {human_readable}")
        
        # Memory estimates
        print(f"\n  Memory Estimates:")
        print(f"    Estimated Memory Needed: {estimates['estimated_memory_needed_gb']:.2f} GB")
        print(f"    Memory Utilization: {estimates['memory_utilization']:.1%}")
        
        if estimates.get('memory_warning'):
            print(f"    ⚠️  WARNING: High memory utilization expected!")
            print(f"       Consider reducing batch size or enabling gradient checkpointing")
        else:
            print(f"    ✓ Memory utilization within safe limits")
        
        # Model info
        print(f"\n  Model Information:")
        print(f"    Total Parameters: {estimates['model_parameters']:,}")
        
        # Device-specific info
        print(f"\n  Device-Specific Details:")
        if estimates['device_type'] == 'cuda':
            print(f"    CUDA acceleration enabled")
            print(f"    Recommended precision: mixed_fp16 or mixed_bf16")
        elif estimates['device_type'] == 'mps':
            print(f"    Apple Silicon (MPS) acceleration")
            print(f"    Recommended precision: fp16")
            print(f"    Note: Some operations may fall back to CPU")
        else:
            print(f"    CPU only - training will be slow")
            print(f"    Consider using a GPU for production training")
        
        # Optimization recommendations
        print(f"\n  Optimization Recommendations:")
        
        if estimates['memory_utilization'] > 0.85:
            print(f"    Memory Optimization:")
            print(f"      - Reduce batch size")
            print(f"      - Enable gradient checkpointing")
            print(f"      - Use gradient accumulation")
            print(f"      - Consider mixed precision training")
        
        if estimates['estimated_hours'] > 48:
            print(f"    Training Time Optimization:")
            print(f"      - Consider using multiple GPUs")
            print(f"      - Enable mixed precision training")
            print(f"      - Use DeepSpeed ZeRO optimization")
            print(f"      - Reduce dataset size for initial experiments")
        
        if estimates['estimated_hours'] < 1:
            print(f"    ✓ Training time is reasonable")
        elif estimates['estimated_hours'] < 12:
            print(f"    Training time is moderate")
        else:
            print(f"    Training will take significant time - ensure stable environment")
        
        # Training schedule recommendation
        print(f"\n  Recommended Training Schedule:")
        if hours < 2:
            print(f"    Single session training recommended")
        elif hours < 12:
            print(f"    Can complete in one work session")
            print(f"    Ensure checkpoint saving every 1-2 hours")
        elif hours < 24:
            print(f"    Day-long training session")
            print(f"    Enable checkpoint saving every hour")
            print(f"    Monitor progress regularly")
        else:
            days = int(hours // 24) + 1
            print(f"    Multi-day training ({days} days)")
            print(f"    Enable checkpoint saving every 2-4 hours")
            print(f"    Set up monitoring and alerting")
            print(f"    Consider cloud training for reliability")
    
    except Exception as e:
        print(f"  Error estimating training time: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing without time estimates...")


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def setup_signal_handlers(orchestrator):
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        orchestrator: Training orchestrator instance
    """
    def signal_handler(sig, frame):
        print("\n" + "="*80)
        print("RECEIVED INTERRUPT SIGNAL - INITIATING GRACEFUL SHUTDOWN")
        print("="*80)
        print("Saving current state...")
        
        try:
            if orchestrator:
                orchestrator._save_meta_learning_state()
                print("Meta-learning state saved")
        except Exception as e:
            print(f"Error saving state: {e}")
        
        print("Cleanup complete. Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_and_setup_experiment(config) -> Path:
    """
    Validate configuration and setup experiment directory.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to experiment directory
    """
    print_section("Experiment Setup")
    
    experiment_dir = Path(f"experiments/{config.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Experiment Directory: {experiment_dir}")
    print(f"  Experiment Name: {config.experiment_name}")
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'reports', 'metrics']
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(exist_ok=True)
    print(f"  Created subdirectories: {', '.join(subdirs)}")
    
    # Save configuration
    config_path = experiment_dir / "config.yaml"
    config.save(str(config_path))
    print(f"  Configuration saved: {config_path}")
    
    # Save configuration as JSON for easier parsing
    config_json_path = experiment_dir / "config.json"
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  Configuration (JSON) saved: {config_json_path}")
    
    return experiment_dir


def save_experiment_metadata(experiment_dir: Path, config, model, datasets_info):
    """
    Save comprehensive experiment metadata.
    
    Args:
        experiment_dir: Path to experiment directory
        config: Training configuration
        model: Model instance
        datasets_info: Dataset statistics
    """
    print_section("Saving Experiment Metadata")
    
    metadata = {
        'experiment_name': config.experiment_name,
        'created_at': datetime.now().isoformat(),
        'training_config': {
            'epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'precision': config.precision,
            'gradient_accumulation': config.gradient_accumulation_steps,
        },
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'seq_length': config.seq_length,
            'use_moe': config.use_moe,
            'num_experts': config.num_experts if config.use_moe else None,
        }
    }
    
    # Add model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['model_parameters'] = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }
    
    # Save system info
    if UTILS_AVAILABLE:
        system_info_path = experiment_dir / "system_info.json"
        with open(system_info_path, 'w') as f:
            json.dump(get_system_info(), f, indent=2, default=str)
        print(f"  System info saved: {system_info_path}")
        
        # Save dataset statistics
        if datasets_info:
            data_stats_path = experiment_dir / "data_statistics.json"
            with open(data_stats_path, 'w') as f:
                json.dump(datasets_info, f, indent=2, default=str)
            print(f"  Dataset statistics saved: {data_stats_path}")
    
    # Save metadata
    metadata_path = experiment_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Experiment metadata saved: {metadata_path}")

    # Add this to Main.py to fix multi-dataset support

def load_checkpoint_for_continuation(checkpoint_path: str, orchestrator) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        orchestrator: Training orchestrator instance
        
    Returns:
        Dictionary with checkpoint info (start_epoch, global_step, etc.)
    """
    print("\n" + "="*80)
    print("LOADING CHECKPOINT FOR CONTINUATION")
    print("="*80)
    
    checkpoint_path = Path(checkpoint_path)
    
    # Find checkpoint file
    if checkpoint_path.is_dir():
        checkpoints = sorted(checkpoint_path.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_file = checkpoints[-1]  # Latest checkpoint
        print(f"Found {len(checkpoints)} checkpoints, using latest: {checkpoint_file.name}")
    else:
        checkpoint_file = checkpoint_path
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    print(f"\nLoading from: {checkpoint_file}")
    print(f"File size: {checkpoint_file.stat().st_size / 1e6:.2f} MB")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Restore orchestrator state
    if hasattr(orchestrator, 'load_checkpoint'):
        orchestrator.load_checkpoint(str(checkpoint_file))
        print("✓ Orchestrator state restored")
    
    # Extract info
    info = {
        'start_epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'checkpoint_path': str(checkpoint_file)
    }
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {info['start_epoch']}")
    print(f"  Global Step: {info['global_step']}")
    print(f"  Best Loss: {info['best_loss']:.4f}")
    print("="*80)
    
    return info

def setup_multi_dataset_training(config, data_params):
    """
    Setup training with multi-dataset support.
    
    This replaces the direct data_path usage with MultiDatasetManager.
    """
    from core.dataset import MultiDatasetManager
    
    print_section("Multi-Dataset Setup")
    
    # Initialize dataset manager
    dataset_manager = MultiDatasetManager(data_params)
    
    # Validate all datasets
    if data_params.get('validate_datasets', True):
        print("Validating datasets...")
        validation_stats = dataset_manager.validate_all_datasets()
        
        if validation_stats.get('validation_errors'):
            print("Validation errors found:")
            for error in validation_stats['validation_errors']:
                print(f"  ❌ {error}")
        
        print(f"\nTraining datasets: {len(validation_stats.get('train_datasets', []))}")
        for ds in validation_stats.get('train_datasets', []):
            print(f"  - {Path(ds['path']).name}: {ds['conversations']:,} conversations ({ds['size_mb']:.1f} MB)")
        
        print(f"\nEvaluation datasets: {len(validation_stats.get('eval_datasets', []))}")
        for ds in validation_stats.get('eval_datasets', []):
            print(f"  - {Path(ds['path']).name}: {ds['conversations']:,} conversations ({ds['size_mb']:.1f} MB)")
        
        print(f"\nTotal training conversations: {validation_stats.get('total_train_conversations', 0):,}")
        print(f"Total evaluation conversations: {validation_stats.get('total_eval_conversations', 0):,}")
    
    # Prepare combined datasets
    print("\nPreparing combined training data...")
    train_data_path = dataset_manager.prepare_training_data(
        force_rebuild=data_params.get('force_rebuild', False)
    )
    print(f"Training data ready: {train_data_path}")
    
    eval_data_path = None
    if dataset_manager.eval_paths:
        print("\nPreparing combined evaluation data...")
        eval_data_path = dataset_manager.prepare_evaluation_data()
        print(f"Evaluation data ready: {eval_data_path}")
    
    # Update config with final paths
    config.train_data_path = train_data_path
    config.eval_data_path = eval_data_path if eval_data_path else train_data_path
    
    return train_data_path, eval_data_path

def auto_adjust_epochs_chinchilla(config, model, dataset):
    """
    Chinchilla-style automatic epoch adjustment based on dataset size and model parameters.
    
    Formula: N_opt ≈ 20 * P (where P = model parameters in billions)
    
    Args:
        config: Training configuration object
        model: Model instance (to count parameters)
        dataset: Dataset instance (to estimate tokens)
        
    Returns:
        Updated config with adjusted num_epochs
    """
    print("\n" + "="*80)
    print("🧠 CHINCHILLA-STYLE EPOCH SCALING")
    print("="*80)
    
    # Get model parameter count
    try:
        P = sum(p.numel() for p in model.parameters())
        P_billions = P / 1e9
        print(f"📊 Model Parameters: {P:,} ({P_billions:.2f}B)")
    except Exception as e:
        print(f"⚠️  Could not count model parameters: {e}")
        print("   Skipping auto-epoch scaling")
        return config
    
    # Estimate dataset tokens
    try:
        # Method 1: Try to get from dataset directly
        if hasattr(dataset, 'total_tokens'):
            dataset_tokens = dataset.total_tokens
        # Method 2: Estimate from dataset size and sequence length
        elif hasattr(dataset, '__len__'):
            num_samples = len(dataset)
            seq_length = getattr(config, 'seq_length', 2048)
            dataset_tokens = num_samples * seq_length
        else:
            print("⚠️  Could not estimate dataset size")
            print("   Skipping auto-epoch scaling")
            return config
        
        print(f"📚 Dataset Tokens: {dataset_tokens:,} ({dataset_tokens/1e9:.2f}B)")
    except Exception as e:
        print(f"⚠️  Could not estimate dataset tokens: {e}")
        print("   Skipping auto-epoch scaling")
        return config
    
    # Chinchilla scaling: N_opt ≈ 20 * P (20 tokens per parameter)
    N_opt = int(20 * P)  # Optimal total tokens to see
    print(f"🎯 Chinchilla Optimal Tokens: {N_opt:,} ({N_opt/1e9:.2f}B)")
    
    # Calculate needed epochs
    if dataset_tokens <= 0:
        print("⚠️  Invalid dataset token count")
        return config
    
    tokens_per_epoch = dataset_tokens
    optimal_epochs = max(1, round(N_opt / tokens_per_epoch))
    
    # Get current epoch setting
    old_epochs = getattr(config, 'num_epochs', 3)
    
    # Apply constraints
    min_epochs = getattr(config, 'min_auto_epochs', 1)
    max_epochs = getattr(config, 'max_auto_epochs', 50)
    final_epochs = max(min_epochs, min(optimal_epochs, max_epochs))
    
    print(f"\n📈 Epoch Calculation:")
    print(f"   Tokens per epoch: {tokens_per_epoch:,}")
    print(f"   Optimal epochs (unconstrained): {optimal_epochs}")
    print(f"   Epoch constraints: {min_epochs} - {max_epochs}")
    print(f"   Original config: {old_epochs} epochs")
    print(f"   ➡️  Adjusted to: {final_epochs} epochs")
    
    # Calculate total training tokens
    total_tokens = tokens_per_epoch * final_epochs
    chinchilla_ratio = (total_tokens / N_opt) * 100
    
    print(f"\n🔢 Training Token Budget:")
    print(f"   Total tokens (new): {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"   Chinchilla target: {N_opt:,} ({N_opt/1e9:.2f}B)")
    print(f"   Coverage: {chinchilla_ratio:.1f}% of optimal")
    
    if chinchilla_ratio < 50:
        print(f"   ⚠️  WARNING: Significantly under Chinchilla recommendation")
        print(f"      Consider increasing max_auto_epochs or dataset size")
    elif chinchilla_ratio > 150:
        print(f"   ⚠️  WARNING: Exceeding Chinchilla recommendation")
        print(f"      May lead to overfitting - consider early stopping")
    else:
        print(f"   ✅ Within reasonable range of Chinchilla scaling")
    
    # Update config
    config.num_epochs = final_epochs
    
    print("="*80 + "\n")
    
    return config

def main():
    """Main training function with advanced features and comprehensive logging."""
    
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS
    # =================================================
    
    # Base model configuration
    config_choice = 'debug'  # Options: 'debug', 'debug_200m', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'
    
    # Training mode selection
    use_adaptive_training = TRAINING_INFRASTRUCTURE_AVAILABLE  # Orchestrator with AI-driven optimization

    # Training parameters
    training_params = {
        'use_moe': True,
        'use_mod': True,
        'num_epochs': 20,
        
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        
        'use_lr_scheduler': True,
        'lr_scheduler': "constant", # cosine, constant, or linear
        'warmup_ratio': 0.0001,
        
        'batch_size': 20,
        'gradient_accumulation_steps': 8,
        
        'precision': "fp32",
        'inference_precision': "fp16",
        
        # Comment these out if your not using MoE
        'num_experts': 8,
        'moe_top_k': 1,
        
        'compile': True,
        'max_memory_usage': 0.85,
        'save_every_n_batches': 1000,
        'eval_every_n_batches': 500,
        'use_flash_attention': True,
        'gradient_checkpointing': True,
        'num_workers': 0,
        'save_total_limit': 5,
        'weight_decay': 0.01,
    }

    adaptive_lr_params = {
        'enable_adaptive_lr': True,           # Master switch for adaptive LR adjustments
        'allow_scheduler_override': True,     # Allow orchestrator to override scheduler
        'min_override_threshold': 0.2,        # Only override if change > 20%
        'emergency_override_enabled': True,   # Always allow emergency LR reductions
        'log_lr_decisions': True,             # Log all LR decision making
    }
    
    # Data configuration
    data_params = {
        # ===================================================================
        # BASE TRAINING (Pre-training on raw text like The Pile, C4, etc. Works on .txt and .jsonl)
        # ===================================================================

        'base_training_paths': [
            'datasets/wikipedia_1.txt',
            'datasets/wikipedia_2.txt',
            'datasets/stackoverflow_1.txt',
            'datasets/stackoverflow_2.txt',
            'datasets/gutenberg_1.txt',
            'datasets/arxiv_1.txt',
            'datasets/arxiv_2.txt',
            'datasets/pubmed_1.txt',
            'datasets/pubmed_2.txt',
            'datasets/pubmed_3.txt',
            'datasets/pubmed_4.txt',  
            'datasets/openwebtext_1.txt',    
            'datasets/openwebtext_1.txt',  
            'datasets/ccnews_1.txt',
        ],

        'base_eval_paths': [ # Only .jsonl
            'datasets/oasst1_validation.jsonl',
        ],

        # ===================================================================
        # FINE-TUNING (Instruction tuning on conversations. Only works on .jsonl)
        # ===================================================================
        'finetuning_paths': [
            'datasets/oasst1_train.jsonl',
            'datasets/oasst1_train_part2.jsonl',
            'datasets/oasst1_train_part3.jsonl',
        ],

        'finetuning_eval_paths': [
            'datasets/oasst1_validation.jsonl',
        ],

        # ===================================================================
        # TRAINING MODE
        # ===================================================================
        'training_mode': 'hybrid',  # Options:
        # - 'base_only': Only base/pre-training on raw text
        # - 'finetuning_only': Only instruction/chat fine-tuning (default)
        # - 'hybrid': Sequential - base training THEN fine-tuning
        # - 'interleaved': Mixed - alternate between base and fine-tuning

        # For 'interleaved' mode only:
        'base_finetuning_ratio': 0.7,  # 70% base, 30% fine-tuning

        # ===================================================================
        # DATASET PROCESSING
        # ===================================================================
        'max_conversations_per_dataset': None,
        'validate_datasets': True,
        'cache_combined_dataset': True,
        'streaming_threshold_gb': 10.0,  # Use streaming for files > 10GB
    }
    # Validate data paths FIRST
    if not validate_data_paths(data_params):
        print("\n❌ Data path validation failed. Cannot continue.")
        print("Please check your file paths in the data_params configuration.\n")
        return 1
    
    # DeepSpeed configuration
    deepspeed_params = {
        'use_deepspeed': False,
        'cpu_offload': True,
        'cpu_offload_optimizer': False,
        'cpu_offload_parameters': False,
        'zero_stage': 2,
        'nvme_path': None,
        'max_grad_norm': 1.0,
    }
    
    # Quantization configuration
    quantization_params = {
        'quantization_method': None,  # Options: None, 'bnb', 'gptq', 'quanto'
        'quantization_bits': None,  # Options: None, 4, 8
    }
    # Add these to your monitoring_params or create a new checkpoint_params section:
    checkpoint_params = {
        'resume_from_checkpoint': None,  # Path to checkpoint file or directory (e.g., 'experiments/my_exp/checkpoints/checkpoint_epoch_5.pt')
        'resume_training': False,        # Set to True to continue training from checkpoint
        'reset_optimizer': False,        # Set to True to reset optimizer state (keeps model weights only)
        'reset_scheduler': False,        # Set to True to reset learning rate scheduler
    }
    
    # Monitoring and logging
    monitoring_params = {
        'log_level': "INFO",
        'experiment_name': f'Advanced_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'early_stopping_patience': 5,
        'backup_every_n_hours': 12,
        'enable_wandb': True,
        'wandb_project': 'deepseek-moe-training',
        'wandb_entity': 'matiasnhmb',
        'health_check_interval': 50,
        'log_every_n_steps': 50,
    }
    
    # Advanced features configuration
    advanced_features = {
        'enable_data_validation': True,
        'generate_data_reports': True,
        'estimate_training_time': True,
        'generate_training_reports': True,
        'auto_tune_batch_size': False,
        'continuous_checkpointing': True,
        'enable_profiling': False,
        'save_optimizer_states': True,
    }
    chinchilla_params = {
        'auto_epoch_scaling': True,      # Enable/disable auto-scaling
        'min_auto_epochs': 1,            # Minimum epochs (safety)
        'max_auto_epochs': 50,           # Maximum epochs (safety)
    }
    
    # =================================================
    # END CONFIGURATION SECTION

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print_banner("MPS COMPATIBILITY CHECK")
        is_compatible, issues = validate_mps_compatibility(config)
        
        if not is_compatible:
            print("MPS Compatibility Issues Detected:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\nApplying automatic fixes...")
            config.apply_device_optimizations('mps')
            
            # Re-validate
            is_compatible, remaining_issues = validate_mps_compatibility(config)
            if remaining_issues:
                print("\nRemaining issues after auto-fix:")
                for issue in remaining_issues:
                    print(f"  ⚠️  {issue}")
        else:
            print("✓ Configuration is MPS compatible")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, monitoring_params['log_level']),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    print_banner("DEEPSEEK MOE TRANSFORMER - ADVANCED TRAINING SYSTEM")
    print(f"Experiment: {monitoring_params['experiment_name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Adaptive AI-Driven' if use_adaptive_training else 'Standard'} Training")
    print(f"")
    print(f"System Status:")
    print(f"  Training Infrastructure: {'Available' if TRAINING_INFRASTRUCTURE_AVAILABLE else 'Not Available'}")
    print(f"  Utils Modules: {'Available' if UTILS_AVAILABLE else 'Not Available'}")
    print(f"  DeepSpeed: {'Available' if DEEPSPEED_AVAILABLE else 'Not Available'}")
    print(f"  CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}")
    if torch.cuda.is_available():
        print(f"  GPU Count: {torch.cuda.device_count()}")
    print("="*80)
    
    try:
        # Step 1: System diagnostics
        print_banner("STEP 1: SYSTEM DIAGNOSTICS")
        print_system_diagnostics()
        
        # Step 2: Create base configuration
        print_banner("STEP 2: CREATING CONFIGURATION")
        print(f"Loading configuration preset: {config_choice}")
        
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
            print(f"Base configuration loaded successfully")
        else:
            raise ValueError(f"Unknown config preset: {config_choice}")

        # After: config = getattr(ConfigPresets, config_choice)()
        
        # FORCE REMOVE ANY LIMITS
        config.limit_train_batches = None
        config.max_train_steps = None  
        config.fast_dev_run = False

        print("="*80)
        print("DEBUG: FORCING FULL DATASET TRAINING")
        print("="*80)   
        
        # Apply all parameter overrides
        all_params = {
            **training_params, 
            **data_params, 
            **deepspeed_params, 
            **quantization_params,
            **monitoring_params,
            **advanced_features
        }
        
        print_section("Applying Parameter Overrides")
        print(f"Total parameters to override: {len(all_params)}")
        
        override_count = 0
        for key, value in all_params.items():
            if value is not None:
                old_value = getattr(config, key, None)
                setattr(config, key, value)
                if old_value is not None and old_value != value and key not in ['raw_oasst_path']:
                    print(f"  {key}: {old_value} -> {value}")
                    override_count += 1
        
        print(f"Applied {override_count} configuration overrides")

        # Apply Chinchilla scaling parameters
        print("\nApplying Chinchilla scaling parameters...")
        for key, value in chinchilla_params.items():
            setattr(config, key, value)
            print(f"  {key}: {value}")
        
        # Step 3: Validate precision support
        print_banner("STEP 3: VALIDATING PRECISION SUPPORT")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_precision = training_params.get('precision', 'fp32')
        inference_precision = training_params.get('inference_precision', 'fp16')
        
        print(f"Device: {device}")
        print(f"Training Precision: {training_precision}")
        print(f"Inference Precision: {inference_precision}")
        
        # Validate training precision
        is_supported, error_msg = validate_precision_support(training_precision, device)
        if not is_supported:
            print(f"\nERROR: Precision Validation Failed")
            print(f"  {error_msg}")
            print(f"\nTraining cannot continue with unsupported precision.")
            return 1
        else:
            print(f"Training precision validated successfully")
        
        # Validate inference precision
        is_supported, error_msg = validate_precision_support(inference_precision, device)
        if not is_supported:
            print(f"\nWARNING: Inference precision not supported")
            print(f"  {error_msg}")
            print(f"  Using training precision for inference instead")
            config.inference_precision = training_precision
        else:
            print(f"Inference precision validated successfully")
        
        # Step 4: Validate configuration
        print_banner("STEP 4: VALIDATING CONFIGURATION")
        try:
            config.validate()
            print("Configuration validation passed")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Sequence length: {config.seq_length}")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Epochs: {config.num_epochs}")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return 1
        
        # Step 5: Initialize tokenizer
        print_banner("STEP 5: INITIALIZING TOKENIZER")
        tokenizer = ConversationTokenizer()
        config.vocab_size = tokenizer.vocab_size
        print(f"Tokenizer initialized successfully")
        print(f"  Vocabulary size: {config.vocab_size:,}")
        print(f"  Special tokens: {len(tokenizer.special_tokens) if hasattr(tokenizer, 'special_tokens') else 'N/A'}")
        
        # Step 6: Data preparation and validation (ADVANCED)
        datasets_info = None
        if advanced_features.get('enable_data_validation'):
            datasets_info = prepare_and_validate_data(config, tokenizer)
        else:
            print_banner("STEP 6: DATA PREPARATION (BASIC)")
            print("Skipping advanced data validation")
        
        # Complete replacement for Step 7 in Main.py (around line 1270-1350)
        # This handles both single and multiple dataset files correctly

        # Step 7: Setup datasets
        print_banner("STEP 7: SETTING UP DATASETS")
        
        # Use the HybridDatasetManager from dataset.py
        from core.dataset import HybridDatasetManager, setup_datasets

        # First, transfer data_params to config attributes
        # This ensures the HybridDatasetManager can read them
        config.base_training_paths = data_params.get('base_training_paths', [])
        config.base_eval_paths = data_params.get('base_eval_paths', [])
        config.finetuning_paths = data_params.get('finetuning_paths', [])
        config.finetuning_eval_paths = data_params.get('finetuning_eval_paths', [])
        config.training_mode = data_params.get('training_mode', 'finetuning_only')
        config.data_cache_dir = data_params.get('data_cache_dir', 'data/cache')
        config.streaming_threshold_gb = data_params.get('streaming_threshold_gb', 10.0)

        # Additional config settings
        config.base_finetuning_ratio = data_params.get('base_finetuning_ratio', 0.5)
        config.max_conversations_per_dataset = data_params.get('max_conversations_per_dataset', None)

        print(f"\nTraining mode: {config.training_mode}")
        print(f"Fine-tuning paths: {config.finetuning_paths}")
        print(f"Fine-tuning eval paths: {config.finetuning_eval_paths}")

        # Use the setup_datasets function which handles everything
        print("\nSetting up datasets using HybridDatasetManager...")
        try:
            train_dataset, eval_dataset = setup_datasets(config, tokenizer)

            print(f"\n✓ Datasets loaded successfully!")
            print(f"  Training dataset: {len(train_dataset):,} samples")
            if eval_dataset != train_dataset:
                print(f"  Evaluation dataset: {len(eval_dataset):,} samples")
            else:
                print(f"  Using training data for evaluation")

        except Exception as e:
            print(f"\n✗ Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()

            # Provide helpful error messages
            print("\n" + "="*80)
            print("TROUBLESHOOTING DATASET ERROR")
            print("="*80)
            print("\nPlease check:")
            print("1. All file paths in finetuning_paths exist:")
            for i, path in enumerate(config.finetuning_paths, 1):
                exists = Path(path).exists()
                status = "✓" if exists else "✗"
                print(f"   {status} [{i}] {path}")

            print("\n2. Files are valid JSONL format")
            print("3. Directory permissions allow reading")

        # Try to give specific advice based on the error
            if "IsADirectoryError" in str(e):
                print("\n⚠️  ERROR: A directory path was passed instead of a file path")
                print("   Check that all paths in finetuning_paths point to actual files, not directories")

            sys.exit(1)

        # Display dataset statistics if available
        print("\n" + "="*80)
        print("DATASET & TRAINING CALCULATION VERIFICATION")
        print("="*80)
        print(f"Dataset size: {len(train_dataset):,} conversations")
        print(f"Batch size: {config.batch_size}")
        print(f"Expected batches per epoch: {len(train_dataset) / config.batch_size:.2f}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Expected optimizer steps per epoch: {len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps):.2f}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Total epochs: {config.num_epochs}")
        print(f"Total optimizer steps for training: {(len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs:.0f}")
        print("="*80 + "\n")
        
        # Step 8: Estimate training time (ADVANCED)
        print_banner("STEP 8: ESTIMATE TRAINING TIME")
        if advanced_features.get('estimate_training_time'):
            estimate_and_display_training_time(config, len(train_dataset))
        
        
        # Step 9: Initialize model
        print_banner("STEP 9: INITIALIZING MODEL")
        print("Creating model configuration...")
        model_config = config_to_deepseek_config(config)
        
        print("Initializing model architecture...")
        model = DeepSeekTransformer(model_config)
        

        def init_weights_for_fp16(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Use smaller std for FP16 stability
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        model.apply(init_weights_for_fp16)
        print("✓ Applied FP16-safe weight initialization")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Architecture: DeepSeek Transformer")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"  Model Size (FP32): {total_params * 4 / 1e9:.2f} GB")
        print(f"  Model Size (FP16): {total_params * 2 / 1e9:.2f} GB")
        
        if config.use_moe:
            print(f"\nMixture of Experts Configuration:")
            print(f"  Number of Experts: {config.num_experts}")
            print(f"  Top-K Routing: {config.moe_top_k}")
            print(f"  Capacity Factor: {config.capacity_factor}")
            print(f"  Load Balancing Weight: {config.load_balancing_weight}")

        # Step 9.5: Auto-adjust epochs using Chinchilla scaling
        if getattr(config, 'auto_epoch_scaling', False):
            print_banner("STEP 9.5: CHINCHILLA EPOCH SCALING")
            print(f"Before Chinchilla: {config.num_epochs} epochs")
            config = auto_adjust_epochs_chinchilla(config, model, train_dataset)
            print(f"After Chinchilla: {config.num_epochs} epochs")
        else:
            print_banner("STEP 9.5: CHINCHILLA SCALING (DISABLED)")
            print(f"auto_epoch_scaling is False - using manual epochs: {config.num_epochs}")
        
        # Step 10: Initialize training system
        print_banner("STEP 10: INITIALIZING TRAINING SYSTEM")
        
        orchestrator = None
        
        if use_adaptive_training and TRAINING_INFRASTRUCTURE_AVAILABLE:
            print("Initializing Adaptive Training Orchestrator")
            print("\nAdvanced Features:")
            print("  - AI-driven hyperparameter optimization")
            print("  - Real-time performance monitoring")
            print("  - Adaptive learning rate scheduling")
            print("  - Advanced checkpointing system")
            print("  - Meta-learning capabilities")
            print("  - Automatic recovery from failures")
            print("  - Performance profiling and analysis")
            
            try:
                orchestrator = AdaptiveTrainingOrchestrator(config)
                print("\nOrchestrator initialized successfully")
                print("\n" + "="*80)
                print("🔍 TRAINER VERIFICATION")
                print("="*80)

                # Initialize training system
                orchestrator.initialize_training()

                # Verify trainer is real
                trainer_type = type(orchestrator.trainer).__name__
                print(f"Trainer type: {trainer_type}")

                if trainer_type == 'AdaptiveTrainer':
                    print("❌ CRITICAL ERROR: Using fallback trainer!")
                    print("   Real EnhancedConversationTrainer failed to load")
                    print("   Training will NOT work!")
                    sys.exit(1)

                # Verify train method exists and is callable
                if not hasattr(orchestrator.trainer, 'train'):
                    print("❌ CRITICAL ERROR: Trainer has no train method!")
                    sys.exit(1)

                if not callable(orchestrator.trainer.train):
                    print("❌ CRITICAL ERROR: Trainer.train is not callable!")
                    sys.exit(1)

                print("✅ Trainer verification passed")
                print(f"✅ Trainer class: {orchestrator.trainer.__class__.__module__}.{trainer_type}")
                print("="*80 + "\n")
            except Exception as e:
                print(f"\nERROR: Failed to initialize orchestrator: {e}")
                import traceback
                traceback.print_exc()
                return 1
            
        else:
            print("ERROR: Advanced training infrastructure required but not available")
            print("Please ensure the following modules are available:")
            print("  - training.orchestrator")
            print("  - training.trainer")
            print("  - training.checkpoint")
            return 1
        
        # Step 11: Setup signal handlers
        print_banner("STEP 11: SETTING UP SIGNAL HANDLERS")
        setup_signal_handlers(orchestrator)
        print("Signal handlers configured for graceful shutdown")
        print("  SIGINT (Ctrl+C): Save state and exit")
        print("  SIGTERM: Save state and exit")
        
        # Step 12: Create experiment directory and save metadata
        print_banner("STEP 12: EXPERIMENT SETUP")
        experiment_dir = validate_and_setup_experiment(config)
        save_experiment_metadata(experiment_dir, config, model, datasets_info)
        
        # Step 13: Display training configuration summary
        print_banner("STEP 13: TRAINING CONFIGURATION SUMMARY")
        
        print_section("Model Configuration")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Attention Heads: {config.num_heads}")
        print(f"  Sequence Length: {config.seq_length}")
        print(f"  MoE: {'Enabled' if config.use_moe else 'Disabled'}")
        
        print_section("Training Configuration")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Min Learning Rate: {config.min_lr}")
        print(f"  LR Scheduler: {config.lr_scheduler}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  Precision: {config.precision}")
        print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
        
        print_section("Optimization Configuration")
        if quantization_params.get('quantization_method'):
            print(f"  Quantization: {quantization_params['quantization_method']} ({quantization_params['quantization_bits']}-bit)")
        else:
            print(f"  Quantization: Disabled")
        print(f"  Flash Attention: {config.use_flash_attention}")
        print(f"  Model Compilation: {config.compile}")
        
        print_section("Data Configuration")
        print(f"  Training Samples: {len(train_dataset):,}")
        print(f"  Evaluation Samples: {len(eval_dataset):,}")
        print(f"  Steps per Epoch: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)}")
        print(f"  Total Training Steps: {(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs}")
        
        # Step 14: Start training
        print_banner("STEP 14: STARTING ADAPTIVE TRAINING")
        print(f"Training begins at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment directory: {experiment_dir}")
        print("")
        print("Press Ctrl+C at any time to gracefully stop training and save progress")
        print("="*80)
        
        # ✅ CRITICAL FIX: Initialize trainer BEFORE wrapping with OOM protection
        print("="*80)
        print("INITIALIZING TRAINER")
        print("="*80)
        
        try:
            orchestrator.initialize_training()
            
            if orchestrator.trainer is None:
                print("❌ CRITICAL: Trainer initialization returned None!")
                return 1
            
            # Check if using fallback trainer (bad sign)
            trainer_type = type(orchestrator.trainer).__name__
            if trainer_type == 'AdaptiveTrainer':
                print("⚠️  WARNING: Using fallback AdaptiveTrainer!")
                print("   Real trainer initialization failed silently")
                print("   Training may not work properly")
            
            print(f"✅ Trainer initialized: {trainer_type}")
            print(f"✅ Trainer has train method: {hasattr(orchestrator.trainer, 'train')}")
            
            # Safe device check
            try:
                if hasattr(orchestrator.trainer, 'device'):
                    print(f"✅ Model device: {orchestrator.trainer.device}")
                else:
                    print(f"⚠️  Device attribute not available (fallback trainer)")
            except Exception as e:
                print(f"⚠️  Could not get device: {e}")
            
            print(f"✅ Model parameters: {sum(p.numel() for p in orchestrator.model.parameters()):,}")
            print("="*80)
            
        except Exception as e:
            print("❌ CRITICAL: Trainer initialization failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        training_start_time = time.time()
        
        # Run adaptive training with OOM protection
        try:
            orchestrator = wrap_orchestrator_with_oom_protection(
                orchestrator, 
                train_dataset, 
                eval_dataset
            )
        except KeyboardInterrupt:
            print("\n" + "="*80)
            print("TRAINING INTERRUPTED BY USER")
            print("="*80)
            raise
        except Exception as e:
            print("\n" + "="*80)
            print("TRAINING ERROR")
            print("="*80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        total_training_hours = total_training_time / 3600
        
        # Step 15: Training completion summary
        print_banner("TRAINING COMPLETED SUCCESSFULLY!")
        
        print_section("Training Summary")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Started: {datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Completed: {datetime.fromtimestamp(training_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Time: {total_training_hours:.2f} hours ({total_training_time/60:.1f} minutes)")
        
        # Get adaptive training status
        status = orchestrator.get_adaptive_status()
        
        print_section("Adaptive Training Statistics")
        print(f"  Adaptive Decisions Made: {status.get('adaptive_decisions_made', 'N/A')}")
        print(f"  Metrics Collected: {status.get('metrics_collected', 'N/A')}")
        print(f"  Meta-Learning Runs: {status.get('meta_learning_runs', 'N/A')}")
        print(f"  Checkpoints Saved: {status.get('checkpoints_saved', 'N/A')}")
        
        # Step 16: Generate final reports (ADVANCED)
        if advanced_features.get('generate_training_reports') and UTILS_AVAILABLE:
            print_banner("STEP 16: GENERATING TRAINING REPORTS")
            try:
                create_training_report(str(experiment_dir))
                print(f"Training report generated: {experiment_dir}/training_report.html")
            except Exception as e:
                print(f"Warning: Could not generate training report: {e}")
        
        # Step 17: Save final summary
        print_section("Saving Final Summary")
        
        summary = {
            'experiment_name': config.experiment_name,
            'training_completed': datetime.now().isoformat(),
            'training_start_time': datetime.fromtimestamp(training_start_time).isoformat(),
            'training_end_time': datetime.fromtimestamp(training_end_time).isoformat(),
            'total_training_time_seconds': total_training_time,
            'total_training_time_hours': total_training_hours,
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': total_params - trainable_params,
            },
            'configuration': {
                'model_preset': config_choice,
                'moe_enabled': config.use_moe,
                'num_experts': config.num_experts if config.use_moe else None,
                'quantization': quantization_params.get('quantization_method'),
                'precision': config.precision,
                'training_mode': 'adaptive',
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'epochs': config.num_epochs,
            },
            'dataset_info': {
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
            },
            'adaptive_training_status': status,
            'advanced_features_used': {k: v for k, v in advanced_features.items() if v},
        }
        
        summary_path = experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Training summary saved: {summary_path}")
        
        # Final success message
        print_banner("ALL OPERATIONS COMPLETED SUCCESSFULLY")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Total execution time: {(time.time() - training_start_time)/3600:.2f} hours")
        print("")
        print("Thank you for using the DeepSeek MoE Training System!")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("TRAINING INTERRUPTED BY USER")
        print("="*80)
        print("Saving emergency state...")
        
        if 'orchestrator' in locals() and orchestrator:
            try:
                orchestrator._save_meta_learning_state()
                print("Meta-learning state saved successfully")
            except Exception as e:
                print(f"Error saving state: {e}")
        
        print("Graceful shutdown complete")
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("TRAINING FAILED WITH ERROR")
        print("="*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\nFull Traceback:")
        import traceback
        traceback.print_exc()
        print("="*80)
        return 1
        
    finally:
        print("\n" + "="*80)
        print("CLEANUP AND RESOURCE RELEASE")
        print("="*80)
        
        # Cleanup orchestrator
        if 'orchestrator' in locals() and orchestrator:
            try:
                print("Cleaning up orchestrator...")
                orchestrator.cleanup()
                print("Orchestrator cleanup complete")
            except Exception as e:
                print(f"Orchestrator cleanup error: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            print("Clearing GPU memory cache...")
            torch.cuda.empty_cache()
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  GPU Memory Allocated: {memory_allocated:.2f} GB")
                print(f"  GPU Memory Reserved: {memory_reserved:.2f} GB")
            except:
                pass
        
        # Run garbage collection
        print("Running garbage collection...")
        gc.collect()
        
        # Final system status
        if UTILS_AVAILABLE:
            try:
                final_info = get_system_info()
                if final_info.get('available_memory_gb'):
                    print(f"  Available System Memory: {final_info.get('available_memory_gb', 0):.2f} GB")
            except:
                pass
        
        print("Cleanup complete")
        print("="*80)


if __name__ == "__main__":
    exit(main())