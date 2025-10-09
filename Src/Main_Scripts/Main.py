# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
import time
import math
import signal
import shutil
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
        from dataset import ConversationDataset, create_dataloader
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
        return
    
    print_banner("TRAINING TIME & RESOURCE ESTIMATION")
    
    estimates = estimate_training_time(config, dataset_size)
    
    print_section("Time Estimates")
    print(f"  Total Tokens to Process: {estimates['total_tokens']:,}")
    print(f"  Estimated Throughput: {estimates['tokens_per_second']:,.0f} tokens/second")
    print(f"  Estimated Training Time:")
    print(f"    Hours: {estimates['estimated_hours']:.1f}")
    print(f"    Days: {estimates['estimated_days']:.2f}")
    
    # Convert to more readable format
    hours = estimates['estimated_hours']
    if hours < 1:
        print(f"    Human Readable: {hours * 60:.0f} minutes")
    elif hours < 24:
        print(f"    Human Readable: {hours:.1f} hours")
    else:
        days = int(hours // 24)
        remaining_hours = hours % 24
        print(f"    Human Readable: {days} days, {remaining_hours:.1f} hours")
    
    print_section("Resource Utilization")
    print(f"  Estimated Memory Utilization: {estimates['memory_utilization']:.1%}")
    
    if estimates.get('memory_warning'):
        print(f"  ⚠ WARNING: High memory utilization expected!")
        print(f"    Consider reducing batch size or enabling gradient checkpointing")
    else:
        print(f"  Memory utilization within safe limits")
    
    # Additional resource recommendations
    print_section("Optimization Recommendations")
    if estimates['memory_utilization'] > 0.85:
        print(f"  Memory:")
        print(f"    - Reduce batch size")
        print(f"    - Enable gradient checkpointing")
        print(f"    - Use gradient accumulation")
    
    if estimates['estimated_hours'] > 48:
        print(f"  Training Time:")
        print(f"    - Consider using multiple GPUs")
        print(f"    - Enable mixed precision training")
        print(f"    - Use DeepSpeed ZeRO optimization")
    
    print("="*80)


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
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'lr_scheduler': "cosine",
        'batch_size': 20,
        'gradient_accumulation_steps': 8,
        'precision': "fp16",
        'inference_precision': "int8",
        'compile': True,
        'max_memory_usage': 0.85,
        'save_every_n_batches': 1000,
        'eval_every_n_batches': 500,
        'use_flash_attention': True,
        'gradient_checkpointing': True,
        'num_workers': 2,
        'save_total_limit': 5,
        'weight_decay': 0.01,
    }
    
    # Data configuration
    data_params = {
        'train_data_path': 'oasst1_data/oasst1_train.jsonl',
        'eval_data_path': 'oasst1_data/oasst1_train.jsonl',
        'raw_oasst_path': 'raw_data/oasst1.jsonl',  # Optional: raw OASST data to process
        'max_conversations_per_file': 50000,
    }
    
    # DeepSpeed configuration
    deepspeed_params = {
        'use_deepspeed': False,
        'cpu_offload': False,
        'cpu_offload_optimizer': False,
        'cpu_offload_parameters': False,
        'zero_stage': 2,
        'nvme_path': None,
        'max_grad_norm': 1.0,
    }
    
    # Quantization configuration
    quantization_params = {
        'quantization_method': 'bnb',  # Options: None, 'bnb', 'gptq', 'quanto'
        'quantization_bits': 8,  # Options: None, 4, 8
    }
    
    # Monitoring and logging
    monitoring_params = {
        'log_level': "INFO",
        'experiment_name': f'Advanced_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'early_stopping_patience': 5,
        'backup_every_n_hours': 12,
        'enable_wandb': False,
        'wandb_project': None,
        'wandb_entity': None,
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
        
        # Step 7: Setup datasets
        print_banner("STEP 7: SETTING UP DATASETS")
        
        train_data_path = Path(data_params['train_data_path'])
        if not train_data_path.exists():
            print(f"ERROR: Training data not found: {train_data_path}")
            return 1
        
        print(f"Loading training dataset from: {train_data_path}")
        train_dataset = ConversationDataset(
            str(train_data_path), tokenizer, config, "train"
        )
        print(f"Training dataset loaded: {len(train_dataset):,} samples")
        
        eval_dataset = None
        eval_data_path = Path(data_params['eval_data_path'])
        if eval_data_path.exists() and eval_data_path != train_data_path:
            print(f"Loading evaluation dataset from: {eval_data_path}")
            eval_dataset = ConversationDataset(
                str(eval_data_path), tokenizer, config, "eval"
            )
            print(f"Evaluation dataset loaded: {len(eval_dataset):,} samples")
        else:
            print("Using training data for evaluation")
            eval_dataset = train_dataset
        
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
        if advanced_features.get('estimate_training_time'):
            estimate_and_display_training_time(config, len(train_dataset))
        
        # Step 9: Initialize model
        print_banner("STEP 9: INITIALIZING MODEL")
        print("Creating model configuration...")
        model_config = config_to_deepseek_config(config)
        
        print("Initializing model architecture...")
        model = DeepSeekTransformer(model_config)
        
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
            except Exception as e:
                print(f"\nERROR: Failed to initialize orchestrator: {e}")
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
        
        training_start_time = time.time()
        
        # Run adaptive training
        try:
            orchestrator.run_adaptive_training()
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