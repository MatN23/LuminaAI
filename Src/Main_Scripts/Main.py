# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# DeepSpeed imports
try:
    import deepspeed
    import torch
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
    from training.orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
except ImportError:
    try:
        from orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
    except ImportError:
        AdaptiveTrainingOrchestrator = None
        TrainingMetrics = None
        print("Warning: Orchestrator not available, using basic training")

try:
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
except ImportError:
    # Create fallback functions
    def process_oasst_data(*args, **kwargs):
        print("Data processing not available")
        return False
    
    def validate_data_comprehensive(*args, **kwargs):
        return {"status": "validation_not_available"}
    
    def validate_environment():
        return "environment_validation_not_available"
    
    def estimate_training_time(*args, **kwargs):
        return {"estimated_hours": 0}
    
    def create_data_summary_report(*args, **kwargs):
        return "report_not_available"

try:
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
except ImportError:
    try:
        from tokenizer import ConversationTokenizer
        from model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
        from dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
    except ImportError:
        print("ERROR: Could not import core modules")
        sys.exit(1)

try:
    from training.trainer import EnhancedConversationTrainer, DeepSpeedConfigGenerator, debug_dataloader
except ImportError:
    try:
        from trainer import EnhancedConversationTrainer, DeepSpeedConfigGenerator, debug_dataloader
    except ImportError:
        print("Warning: Enhanced trainer not available")
        EnhancedConversationTrainer = None
        DeepSpeedConfigGenerator = None
        debug_dataloader = None


def safe_model_memory_footprint(model):
    """Get model memory footprint with comprehensive error handling."""
    try:
        # Try the model's built-in method first
        if hasattr(model, 'get_memory_footprint'):
            return model.get_memory_footprint()
    except Exception as e:
        print(f"Built-in memory footprint calculation failed: {e}")
    
    # Fallback calculation
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'total_size_mb': total_size / (1024 * 1024),
            'method': 'fallback_calculation'
        }
    except Exception as e:
        print(f"Fallback memory calculation also failed: {e}")
        return {
            'total_parameters': 'unknown',
            'total_size_mb': 'unknown',
            'error': str(e)
        }


def benchmark_actual_performance(model, tokenizer, config, dist_manager, samples=20):
    """Measure real training performance for accurate time estimation."""
    print("MEASURING ACTUAL PERFORMANCE...")
    
    # Create realistic test data
    test_inputs = []
    for i in range(samples):
        # Create varied length test sequences
        base_length = config.seq_length // 4 + (i * 10)
        test_text = f"Training benchmark sample {i}. This is a realistic test sequence. " * (base_length // 50)
        
        try:
            tokens = tokenizer.encode_conversation({"messages": [
                {"role": "user", "content": test_text}
            ]})
            if len(tokens) > config.seq_length:
                tokens = tokens[:config.seq_length]
            else:
                tokens.extend([tokenizer.pad_token_id] * (config.seq_length - len(tokens)))
            
            test_inputs.append(torch.tensor(tokens, dtype=torch.long))
        except Exception as e:
            print(f"Error creating test input {i}: {e}")
            tokens = torch.randint(0, min(tokenizer.get_vocab_size(), 1000), (config.seq_length,), dtype=torch.long)
            test_inputs.append(tokens)
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_inputs, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup for measurement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = model.to(device)
        model.train()
    except Exception as e:
        print(f"Could not move model to device: {e}")
        device = torch.device('cpu')
        model = model.to(device)
    
    # Create optimizer matching training config
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.01)
        )
    except Exception as e:
        print(f"Could not create optimizer: {e}")
        return {
            'single_step_time': 1.0,
            'tokens_per_step': 100,
            'error': str(e)
        }
    
    # Warmup runs (don't count these)
    print("Warming up...")
    warmup_batches = min(3, len(test_loader))
    for i, batch in enumerate(test_loader):
        if i >= warmup_batches:
            break
            
        try:
            batch = batch.to(device, non_blocking=True)
            attention_mask = (batch != tokenizer.pad_token_id).long() if hasattr(tokenizer, 'pad_token_id') else torch.ones_like(batch)
            
            # Simple forward pass
            outputs = model(batch, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        except Exception as e:
            print(f"Warmup error: {e}")
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clear memory stats and synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # ACTUAL MEASUREMENT
    print("Measuring training performance...")
    measurement_start = time.time()
    
    total_tokens_processed = 0
    total_steps = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 5:  # Limit measurement batches
            break
            
        step_start = time.time()
        
        try:
            batch = batch.to(device, non_blocking=True)
            batch_size, seq_len = batch.shape
            
            attention_mask = (batch != tokenizer.pad_token_id).long() if hasattr(tokenizer, 'pad_token_id') else torch.ones_like(batch)
            
            # Forward pass
            outputs = model(batch, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation simulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            total_tokens_processed += batch_size * seq_len
            total_samples += batch_size
            total_steps += 1
            
        except Exception as e:
            print(f"Measurement error on batch {batch_idx}: {e}")
            continue
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    measurement_end = time.time()
    total_measurement_time = measurement_end - measurement_start
    
    # Calculate per-step performance
    single_step_time = total_measurement_time / max(total_steps, 1)
    tokens_per_step = total_tokens_processed / max(total_steps, 1)
    samples_per_step = total_samples / max(total_steps, 1)
    
    # Memory statistics
    peak_memory_gb = 0
    current_memory_gb = 0
    if torch.cuda.is_available():
        try:
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            current_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        except:
            pass
    
    performance_stats = {
        'single_step_time': single_step_time,
        'tokens_per_step': tokens_per_step,
        'samples_per_step': samples_per_step,
        'total_time': total_measurement_time,
        'tokens_processed': total_tokens_processed,
        'samples_processed': total_samples,
        'steps_completed': total_steps,
        'peak_memory_gb': peak_memory_gb,
        'current_memory_gb': current_memory_gb,
        'world_size': dist_manager.world_size
    }
    
    print(f"Performance measurement complete!")
    print(f"   Measured {total_steps} steps in {total_measurement_time:.1f}s")
    print(f"   Single step time: {single_step_time:.3f}s")
    print(f"   Tokens per step: {tokens_per_step:,.0f}")
    print(f"   Memory usage: {current_memory_gb:.1f}GB (peak: {peak_memory_gb:.1f}GB)")
    
    return performance_stats


class DeepSpeedIntegrationManager:
    """Manages DeepSpeed integration and distributed training setup."""
    
    def __init__(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.rank = int(os.environ.get('RANK', 0))
        self.is_distributed = self.world_size > 1
        
    def setup_distributed_environment(self):
        """Setup distributed training environment."""
        if not self.is_distributed:
            logging.info("Single GPU/CPU training mode")
            return
        
        logging.info(f"Distributed training setup:")
        logging.info(f"  World size: {self.world_size}")
        logging.info(f"  Global rank: {self.rank}")
        logging.info(f"  Local rank: {self.local_rank}")
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            
        # Initialize process group for non-DeepSpeed communication if needed
        if not DEEPSPEED_AVAILABLE:
            try:
                import torch.distributed as dist
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
                logging.info("PyTorch distributed initialized")
            except Exception as e:
                logging.error(f"Failed to initialize PyTorch distributed: {e}")
    
    def should_log(self) -> bool:
        """Check if this process should log (only rank 0)."""
        return self.rank == 0
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for this process."""
        info = {
            'rank': self.rank,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'is_distributed': self.is_distributed
        }
        
        if torch.cuda.is_available():
            try:
                info.update({
                    'device': f'cuda:{self.local_rank}',
                    'gpu_name': torch.cuda.get_device_name(self.local_rank),
                    'gpu_memory_gb': torch.cuda.get_device_properties(self.local_rank).total_memory / 1e9
                })
            except Exception as e:
                info['device'] = 'cuda'
                info['gpu_error'] = str(e)
        else:
            info['device'] = 'cpu'
        
        return info


class EnhancedResourceManager:
    """Enhanced resource management with DeepSpeed optimizations."""
    
    def __init__(self, distributed_manager: DeepSpeedIntegrationManager):
        self.distributed_manager = distributed_manager
        self.system_info = self._gather_comprehensive_system_info()
        
    def _gather_comprehensive_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information across all nodes."""
        info = {
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'distributed_info': self.distributed_manager.get_device_info()
        }
        
        # GPU information per device
        if torch.cuda.is_available():
            info['gpu_info'] = []
            try:
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    info['gpu_info'].append({
                        'device_id': i,
                        'name': gpu_props.name,
                        'memory_gb': gpu_props.total_memory / 1e9,
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                        'multiprocessor_count': gpu_props.multi_processor_count
                    })
            except Exception as e:
                info['gpu_info'] = [{'error': str(e)}]
        
        # Storage information
        try:
            disk_usage = psutil.disk_usage('.')
            info['disk_free_gb'] = disk_usage.free / (1024**3)
            info['disk_total_gb'] = disk_usage.total / (1024**3)
        except Exception:
            info.update({'disk_free_gb': 100, 'disk_total_gb': 500})
        
        return info
    
    def create_deepspeed_optimization_strategy(self, config: Config, model_size_gb: float, 
                                             sequence_length: int) -> Dict[str, Any]:
        """Create comprehensive DeepSpeed optimization strategy."""
        strategy = {
            'use_deepspeed': DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False),
            'zero_stage': 3,  # Default to ZeRO-3
            'cpu_offload': False,
            'nvme_offload': False,
            'moe_optimizations': {},
            'precision_strategy': 'bf16',
            'gradient_accumulation_strategy': {},
            'communication_optimizations': {}
        }
        
        if not strategy['use_deepspeed']:
            return strategy
        
        # Analyze memory requirements
        total_gpu_memory = sum(gpu.get('memory_gb', 0) for gpu in self.system_info.get('gpu_info', []))
        available_cpu_memory = self.system_info['available_memory_gb']
        
        # Memory-based decisions
        memory_pressure_ratio = model_size_gb / max(total_gpu_memory, 1.0)
        
        if memory_pressure_ratio > 0.8:  # High memory pressure
            strategy.update({
                'zero_stage': 3,
                'cpu_offload': True,
                'aggressive_cpu_offload': True,
                'precision_strategy': 'fp16',  # More memory efficient
                'activation_checkpointing': True
            })
            logging.info("High memory pressure detected - enabling aggressive optimizations")
            
        elif memory_pressure_ratio > 0.5:  # Moderate memory pressure
            strategy.update({
                'zero_stage': 3,
                'cpu_offload': True,
                'precision_strategy': 'bf16'
            })
            logging.info("Moderate memory pressure - enabling CPU offload")
            
        else:  # Low memory pressure
            strategy.update({
                'zero_stage': 2,  # Less aggressive
                'precision_strategy': 'bf16'
            })
            logging.info("Low memory pressure - using standard ZeRO-2")
        
        # NVMe offloading if path is provided and memory is very constrained
        nvme_path = getattr(config, 'nvme_path', None)
        if nvme_path and memory_pressure_ratio > 1.0:
            strategy['nvme_offload'] = True
            strategy['nvme_path'] = nvme_path
            logging.info(f"Enabling NVMe offload to {nvme_path}")
        
        # MoE-specific optimizations
        if hasattr(config, 'use_moe') and config.use_moe and DeepSpeedConfigGenerator:
            num_experts = getattr(config, 'num_experts', 8)
            world_size = self.distributed_manager.world_size
            
            try:
                moe_config = DeepSpeedConfigGenerator.create_moe_config_with_expert_parallelism(
                    world_size, num_experts, model_size_gb, sequence_length
                )
                strategy['moe_optimizations'] = moe_config['moe']
            except Exception as e:
                logging.warning(f"MoE configuration failed: {e}")
            
            # Adjust other settings for MoE
            if sequence_length > 100000:  # Very long sequences with MoE
                strategy['gradient_accumulation_strategy'] = {
                    'recommended_micro_batch_size': 1,
                    'recommended_gradient_accumulation': max(config.batch_size * 8, 64),
                    'reason': 'Long sequences with MoE require small micro batches'
                }
        
        # Communication optimizations for multi-node
        if self.distributed_manager.world_size > 8:  # Multi-node likely
            strategy['communication_optimizations'] = {
                'overlap_comm': True,
                'allgather_bucket_size': 5e8,
                'reduce_bucket_size': 5e8,
                'communication_data_type': strategy['precision_strategy'],
                'contiguous_gradients': True
            }
        
        return strategy
    
    def _estimate_model_memory_usage(self, config: Config) -> float:
        """Estimate model memory usage in GB."""
        try:
            # Basic parameter count
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_layers', 12)
            vocab_size = getattr(config, 'vocab_size', 50257)
            
            # Rough parameter estimation
            params = (
                vocab_size * hidden_size +  # Embeddings
                num_layers * (
                    hidden_size * hidden_size * 4 +  # Attention
                    hidden_size * getattr(config, 'intermediate_size', hidden_size * 4) * 2  # FFN
                )
            )
            
            # Account for MoE if enabled
            if hasattr(config, 'use_moe') and config.use_moe:
                num_experts = getattr(config, 'num_experts', 8)
                params += num_layers * num_experts * hidden_size * getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Memory usage: parameters + gradients + optimizer states + activations
            # Parameters: 4 bytes per param (fp32) or 2 bytes (fp16/bf16)
            param_memory = params * 2 / 1e9  # Assume mixed precision
            
            # Gradients: same size as parameters
            grad_memory = param_memory
            
            # Optimizer states (Adam): 8 bytes per parameter
            optimizer_memory = params * 8 / 1e9
            
            # Activations (rough estimate based on sequence length and batch size)
            batch_size = getattr(config, 'batch_size', 8)
            seq_length = getattr(config, 'seq_length', 2048)
            activation_memory = batch_size * seq_length * hidden_size * num_layers * 4 / 1e9
            
            total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
            
            return total_memory
            
        except Exception as e:
            logging.debug(f"Memory estimation failed: {e}")
            return 10.0  # Conservative default


class AdaptiveDeepSpeedWizard:
    """Enhanced configuration wizard with DeepSpeed integration."""
    
    def __init__(self, resource_manager: EnhancedResourceManager):
        self.resource_manager = resource_manager
    
    def auto_configure_deepspeed(self, config: Config, 
                                natural_description: Optional[str] = None,
                                manual_overrides: Dict[str, Any] = None) -> Config:
        """Automatically configure DeepSpeed based on system analysis, with proper override handling."""
        
        # Apply manual overrides BEFORE auto-configuration
        if manual_overrides:
            print("="*60)
            print("APPLYING MANUAL CONFIGURATION OVERRIDES")
            print("="*60)
            
            for param, value in manual_overrides.items():
                if hasattr(config, param):
                    old_value = getattr(config, param)
                    setattr(config, param, value)
                    print(f"OVERRIDE: {param}: {old_value} -> {value}")
                else:
                    # Create new attribute for DeepSpeed-specific settings
                    setattr(config, param, value)
                    print(f"NEW ATTRIBUTE: {param} = {value}")
            
            print("="*60)
        
        # Estimate model size from config
        try:
            model_size_gb = self.resource_manager._estimate_model_memory_usage(config)
            sequence_length = getattr(config, 'seq_length', 2048)
        except Exception as e:
            logging.warning(f"Could not estimate model size: {e}")
            model_size_gb = 10.0  # Default estimate
            sequence_length = 2048
        
        # Analyze system and create strategy (ONLY if no manual overrides for these specific params)
        strategy = self.resource_manager.create_deepspeed_optimization_strategy(
            config, model_size_gb, sequence_length
        )
        
        # Apply optimizations to config, but RESPECT manual overrides
        apply_deepspeed_optimizations(config, strategy)
        
        # RESTORE manual overrides if they were set
        if manual_overrides:
            for param, value in manual_overrides.items():
                if hasattr(config, param):
                    setattr(config, param, value)
                    print(f"PRESERVED MANUAL OVERRIDE: {param} = {value}")
        
        # Log final strategy
        print("="*60)
        print("FINAL DEEPSPEED CONFIGURATION")
        print("="*60)
        print(f"Model size estimate: {model_size_gb:.1f}GB")
        print(f"ZeRO stage: {getattr(config, 'zero_stage', 'N/A')}")
        print(f"CPU offload: {getattr(config, 'cpu_offload', False)}")
        print(f"CPU offload optimizer: {getattr(config, 'cpu_offload_optimizer', False)}")
        print(f"Precision: {getattr(config, 'precision', 'auto')}")
        print(f"Use DeepSpeed: {getattr(config, 'use_deepspeed', False)}")
        print("="*60)
        
        return config


def apply_deepspeed_optimizations(config: Config, strategy: Dict[str, Any]):
    """Apply DeepSpeed optimization strategy to config."""
    
    # Enable DeepSpeed
    config.use_deepspeed = strategy.get('use_deepspeed', True)
    
    # ZeRO configuration
    if not hasattr(config, 'zero_stage') or config.zero_stage == 0:
        config.zero_stage = strategy.get('zero_stage', 3)
    
    # CPU offloading
    if not hasattr(config, 'cpu_offload'):
        config.cpu_offload = strategy.get('cpu_offload', False)
        config.cpu_offload_optimizer = strategy.get('cpu_offload', False)
        config.aggressive_cpu_offload = strategy.get('aggressive_cpu_offload', False)
    
    # NVMe offloading
    if strategy.get('nvme_offload', False):
        config.nvme_path = strategy.get('nvme_path')
    
    # Precision strategy
    if strategy.get('precision_strategy') and not hasattr(config, 'precision'):
        config.precision = strategy['precision_strategy']
    
    # Gradient accumulation adjustments
    grad_strategy = strategy.get('gradient_accumulation_strategy', {})
    if grad_strategy:
        if 'recommended_micro_batch_size' in grad_strategy:
            config.batch_size = grad_strategy['recommended_micro_batch_size']
        if 'recommended_gradient_accumulation' in grad_strategy:
            config.gradient_accumulation_steps = grad_strategy['recommended_gradient_accumulation']
        logging.info(f"Adjusted batch size to {config.batch_size}, grad accumulation to {config.gradient_accumulation_steps}")
        logging.info(f"Reason: {grad_strategy.get('reason', 'DeepSpeed optimization')}")
    
    # MoE optimizations
    moe_opts = strategy.get('moe_optimizations', {})
    if moe_opts:
        # Apply MoE-specific settings
        for key, value in moe_opts.items():
            if hasattr(config, key):
                setattr(config, key, value)
        logging.info("Applied MoE optimizations to configuration")


def create_enhanced_directory_structure():
    """Create comprehensive directory structure including DeepSpeed."""
    directories = [
        'data', 'data/shards', 'data/processed', 'data/cache',
        'checkpoints', 'checkpoints/best', 'checkpoints/emergency', 'checkpoints/deepspeed',
        'experiments', 'experiments/archive',
        'logs', 'logs/adaptive', 'logs/performance', 'logs/deepspeed',
        'backups', 'backups/configs', 'backups/models',
        'reports', 'reports/adaptive', 'reports/performance', 'reports/moe',
        'monitoring', 'monitoring/metrics', 'monitoring/visualizations', 'monitoring/routing'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)


def setup_deepspeed_logging():
    """Setup enhanced logging for DeepSpeed training."""
    log_dir = Path('logs/deepspeed')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create separate log files for different components
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.FileHandler(log_dir / f'moe_routing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=handlers
    )
    
    # Set DeepSpeed logging level
    if DEEPSPEED_AVAILABLE:
        deepspeed_logger = logging.getLogger('deepspeed')
        deepspeed_logger.setLevel(logging.INFO)


def config_to_deepseek_config(config: Config):
    """Convert training Config to DeepSeekConfig with MoE support."""
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
        
        # MoE configuration
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


def create_dummy_training_data(file_path: Path, num_samples: int = 50):
    """Create dummy training data for testing."""
    print(f"Creating dummy training data: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_conversations = []
    for i in range(num_samples):
        conversation = {
            "messages": [
                {"role": "user", "content": f"This is test question number {i+1}. Can you help me with a simple task?"},
                {"role": "assistant", "content": f"Of course! I'd be happy to help you with test response {i+1}. This is a sample response for training purposes."}
            ]
        }
        dummy_conversations.append(conversation)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for conv in dummy_conversations:
            f.write(json.dumps(conv) + '\n')
    
    print(f"Created {num_samples} dummy conversations in {file_path}")


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"CUDA cleanup error: {e}")


def create_fallback_trainer(model, tokenizer, config):
    """Create a basic fallback trainer if enhanced trainer fails."""
    
    class BasicTrainer:
        def __init__(self, model, tokenizer, config):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Try to move model to device
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                print(f"Could not move model to {self.device}: {e}")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
            
            # Basic optimizer
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=config.learning_rate,
                    weight_decay=getattr(config, 'weight_decay', 0.01)
                )
            except Exception as e:
                print(f"Could not create optimizer: {e}")
                self.optimizer = None
        
        def train(self, train_dataset, eval_dataset=None):
            print("="*60)
            print("RUNNING WITH BASIC FALLBACK TRAINER")
            print("="*60)
            print("This is a minimal trainer for testing purposes only")
            print("For full training, fix the enhanced trainer import")
            
            if not self.optimizer:
                print("No optimizer available, cannot train")
                return
            
            try:
                # Create basic dataloader
                from torch.utils.data import DataLoader
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                
                self.model.train()
                
                for epoch in range(min(2, config.num_epochs)):  # Limit epochs
                    print(f"Basic training epoch {epoch + 1}")
                    
                    for batch_idx, batch in enumerate(train_loader):
                        if batch_idx >= 5:  # Limit batches
                            break
                        
                        try:
                            # Move to device
                            if isinstance(batch, dict):
                                batch = {k: v.to(self.device) for k, v in batch.items()}
                                input_ids = batch.get('input_ids')
                                attention_mask = batch.get('attention_mask')
                            else:
                                input_ids = batch.to(self.device)
                                attention_mask = None
                            
                            if input_ids is None:
                                continue
                            
                            # Forward pass
                            outputs = self.model(input_ids, attention_mask)
                            
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs
                            
                            # Simple loss calculation
                            labels = input_ids
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            
                            # Backward pass
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                            if batch_idx % 2 == 0:
                                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                                
                        except Exception as e:
                            print(f"Training step error: {e}")
                            continue
                
                print("Basic training completed")
                
            except Exception as e:
                print(f"Basic training failed: {e}")
                traceback.print_exc()
    
    return BasicTrainer(model, tokenizer, config)


def validate_deepspeed_config(config):
    """Validate DeepSpeed configuration for common issues."""
    issues = []
    
    if getattr(config, 'use_deepspeed', False) and not DEEPSPEED_AVAILABLE:
        issues.append("DeepSpeed requested but not available")
    
    if getattr(config, 'cpu_offload', False) and not getattr(config, 'use_deepspeed', False):
        issues.append("CPU offloading requires DeepSpeed")
    
    if getattr(config, 'zero_stage', 0) > 0 and not getattr(config, 'use_deepspeed', False):
        issues.append("ZeRO stages require DeepSpeed")
    
    # Check batch size configuration
    batch_size = getattr(config, 'batch_size', 1)
    grad_accum = getattr(config, 'gradient_accumulation_steps', 1)
    if batch_size * grad_accum > 64:
        issues.append(f"Very large effective batch size: {batch_size * grad_accum}")
    
    # Check memory requirements
    seq_length = getattr(config, 'seq_length', 2048)
    if seq_length > 8192 and not getattr(config, 'gradient_checkpointing', False):
        issues.append("Long sequences without gradient checkpointing may cause OOM")
    
    return issues


def main():
    """Enhanced main function with comprehensive error handling and CPU offloading fixes."""
    
    # HARDCODED CONFIGURATION - MODIFY THESE PARAMETERS
    # =================================================
    
    # Base model configuration - select from ConfigPresets
    config_choice = 'debug'  # Options: 'debug', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'
    
    # Override specific parameters (set to None to use preset defaults)
    override_params = {
        'use_moe': True,        # Enable MoE
        'num_epochs': 3,        # Training epochs (reduced for testing)
        'learning_rate': 1e-4,  # Learning rate
        'batch_size': 1,        # Micro batch size
        'gradient_accumulation_steps': 16,  # Reduced for testing
        'train_data_path': 'oasst1_data/oasst1_train.jsonl',
        'eval_data_path': 'data/eval.jsonl',

        'capacity_factor': 1.0,
        'load_balancing_weight': 0.08,
    }
    
    # DeepSpeed and optimization settings - THESE ARE MANUAL OVERRIDES
    manual_deepspeed_overrides = {
        'use_deepspeed': DEEPSPEED_AVAILABLE,
        'cpu_offload': True,             # FORCE CPU offloading
        'cpu_offload_optimizer': True,   # FORCE CPU optimizer offloading
        'cpu_offload_parameters': True,  # FORCE CPU parameter offloading
        'zero_stage': 2,                 # FORCE ZeRO-3
        'nvme_path': None,               # Set to NVMe path if available
    }
    
    # Optimization flags
    optimize_for_long_sequences = True
    check_environment = True
    estimate_time = True
    dry_run = False  # Set to True to test configuration without training
    
    # Data processing flags
    validate_data_path = None  # Set to data file path to validate
    process_oasst = None      # Set to (input_file, output_file) tuple to process
    create_report = True
    
    # =================================================
    # END HARDCODED CONFIGURATION
    
    experiment_name = f'DeepSpeed_MoE_{override_params.get("num_experts", 8)}E_CPU_Offload_Fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Setup distributed environment first
    dist_manager = DeepSpeedIntegrationManager()
    dist_manager.setup_distributed_environment()
    
    # Only log from rank 0 in distributed training
    should_log = dist_manager.should_log()
    
    if should_log:
        # Create enhanced directory structure
        create_enhanced_directory_structure()
        
        # Setup enhanced logging
        setup_deepspeed_logging()
        
        print("\n" + "="*80)
        print("FIXED CPU OFFLOADING DEEPSPEED TRAINING SYSTEM - VERSION 2.0")
        print("="*80)
        print(f"Experiment: {experiment_name}")
        print(f"DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print("="*80)
        
        # System information
        resource_manager = EnhancedResourceManager(dist_manager)
        print("\nSystem Information:")
        for key, value in resource_manager.system_info.items():
            if key == 'gpu_info' and isinstance(value, list):
                print(f"  {key}:")
                for i, gpu in enumerate(value):
                    if 'error' in gpu:
                        print(f"    GPU {i}: Error - {gpu['error']}")
                    else:
                        print(f"    GPU {i}: {gpu.get('name', 'Unknown')} ({gpu.get('memory_gb', 0):.1f}GB)")
            elif key != 'distributed_info':
                print(f"  {key}: {value}")
        
        print("\nDistributed Information:")
        dist_info = resource_manager.system_info['distributed_info']
        for key, value in dist_info.items():
            print(f"  {key}: {value}")
    
    try:
        # STEP 1: Create base configuration
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 1: CREATING BASE CONFIGURATION")
            print(f"{'='*60}")
        
        # Get base preset
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
        else:
            raise ValueError(f"Unknown config preset: {config_choice}")
        
        if should_log:
            print(f"Base configuration loaded: {config_choice}")
            print(f"  Default use_deepspeed: {getattr(config, 'use_deepspeed', False)}")
            print(f"  Default cpu_offload: {getattr(config, 'cpu_offload', False)}")
            print(f"  Default zero_stage: {getattr(config, 'zero_stage', 0)}")
        
        # STEP 2: Apply parameter overrides
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 2: APPLYING PARAMETER OVERRIDES")
            print(f"{'='*60}")
        
        if override_params:
            for key, value in override_params.items():
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    if should_log:
                        print(f"  {key}: {old_value} -> {value}")
                else:
                    if should_log:
                        print(f"  WARNING: Unknown parameter '{key}' ignored")
        
        # Update experiment name
        config.experiment_name = experiment_name
        
        # STEP 3: Apply DeepSpeed overrides BEFORE auto-configuration
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 3: APPLYING DEEPSPEED MANUAL OVERRIDES")
            print(f"{'='*60}")
        
        # Apply manual DeepSpeed overrides
        for key, value in manual_deepspeed_overrides.items():
            if hasattr(config, key):
                old_value = getattr(config, key)
                setattr(config, key, value)
                if should_log:
                    print(f"  MANUAL OVERRIDE: {key}: {old_value} -> {value}")
            else:
                # Create new attribute for DeepSpeed-specific settings
                setattr(config, key, value)
                if should_log:
                    print(f"  NEW ATTRIBUTE: {key} = {value}")
        
        # STEP 4: Auto-configure with resource manager (should preserve manual overrides)
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 4: AUTO-CONFIGURING WITH RESOURCE MANAGER")
            print(f"{'='*60}")
        
        wizard = AdaptiveDeepSpeedWizard(resource_manager)
        config = wizard.auto_configure_deepspeed(
            config, 
            natural_description=f"Training {config_choice} model with forced CPU offloading",
            manual_overrides=manual_deepspeed_overrides
        )
        
        # STEP 5: Validate configuration
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 5: CONFIGURATION VALIDATION")
            print(f"{'='*60}")
            
            # Validate DeepSpeed configuration
            config_issues = validate_deepspeed_config(config)
            if config_issues:
                print("Configuration Issues Found:")
                for issue in config_issues:
                    print(f"  WARNING: {issue}")
            else:
                print("Configuration validation passed")
            
            # Show final critical settings
            critical_settings = [
                'use_deepspeed', 'cpu_offload', 'cpu_offload_optimizer', 
                'cpu_offload_parameters', 'zero_stage', 'precision',
                'batch_size', 'gradient_accumulation_steps'
            ]
            
            print("\nFinal Critical Settings:")
            for setting in critical_settings:
                value = getattr(config, setting, "NOT SET")
                print(f"  {setting}: {value}")
        
        # Environment validation
        if check_environment:
            if should_log:
                print(f"\n{'='*60}")
                print("ENVIRONMENT VALIDATION")
                print(f"{'='*60}")
            
            try:
                env_status = validate_environment()  # Fixed: removed config parameter
                if should_log:
                    print(f"Environment validation: {env_status}")
            except Exception as e:
                if should_log:
                    print(f"Environment validation failed: {e}")
                # Continue anyway - not critical
                env_status = "validation_failed"
        
        # Initialize training components
        if should_log:
            print(f"\n{'='*60}")
            print("INITIALIZING TRAINING COMPONENTS")
            print(f"{'='*60}")
        
        # Initialize tokenizer
        try:
            tokenizer = ConversationTokenizer()
            if hasattr(config, 'vocab_size'):
                config.vocab_size = tokenizer.vocab_size
            if should_log:
                print(f"Tokenizer initialized: vocab_size={config.vocab_size}")
        except Exception as e:
            if should_log:
                print(f"Tokenizer initialization failed: {e}")
            raise
        
        # Initialize model with safe memory calculation
        try:
            model_config = config_to_deepseek_config(config)
            model = DeepSeekTransformer(model_config)
            
            # Get model info safely
            if should_log:
                try:
                    memory_info = safe_model_memory_footprint(model)
                    print(f"Model initialized:")
                    print(f"  Total parameters: {memory_info.get('total_parameters', 'Unknown')}")
                    if 'total_size_mb' in memory_info and memory_info['total_size_mb'] != 'unknown':
                        print(f"  Memory footprint: {memory_info['total_size_mb']:.1f} MB")
                    if 'method' in memory_info:
                        print(f"  Calculation method: {memory_info['method']}")
                    if 'error' in memory_info:
                        print(f"  Calculation error: {memory_info['error']}")
                except Exception as mem_error:
                    print(f"Model initialized, but memory calculation failed: {mem_error}")
                    print(f"  Model type: {type(model)}")
        except Exception as e:
            if should_log:
                print(f"Model initialization failed: {e}")
            raise
        
        # Performance benchmarking
        if estimate_time and not dry_run:
            if should_log:
                print(f"\n{'='*60}")
                print("PERFORMANCE BENCHMARKING")
                print(f"{'='*60}")
            
            try:
                performance_stats = benchmark_actual_performance(
                    model, tokenizer, config, dist_manager, samples=5  # Reduced samples
                )
                
                # Estimate total training time
                if 'error' not in performance_stats:
                    total_batches = 100  # Rough estimate for testing
                    estimated_total_time = performance_stats['single_step_time'] * total_batches
                    
                    if should_log:
                        print(f"Training time estimate: {estimated_total_time/3600:.2f} hours")
                else:
                    if should_log:
                        print(f"Performance benchmarking had errors: {performance_stats.get('error', 'Unknown')}")
                        
            except Exception as e:
                if should_log:
                    print(f"Performance benchmarking failed: {e}")
        
        # Data validation and creation
        if should_log:
            print(f"\n{'='*60}")
            print("DATA SETUP")
            print(f"{'='*60}")
        
        # Check if training data exists, create if not
        train_data_path = Path(config.train_data_path)
        if not train_data_path.exists():
            if should_log:
                print(f"Training data not found at {config.train_data_path}")
            create_dummy_training_data(train_data_path, num_samples=20)
        else:
            if should_log:
                print(f"Training data found: {train_data_path}")
        
        # Validate data if requested
        if validate_data_path and Path(validate_data_path).exists():
            try:
                validation_results = validate_data_comprehensive(validate_data_path, tokenizer)
                if should_log:
                    print(f"Data validation completed: {validation_results}")
            except Exception as e:
                if should_log:
                    print(f"Data validation failed: {e}")
        
        # Training execution
        if not dry_run:
            if should_log:
                print(f"\n{'='*80}")
                print("STARTING TRAINING EXECUTION")
                print(f"{'='*80}")
            
            # Initialize trainer with enhanced capabilities and fallback
            trainer = None
            try:
                if EnhancedConversationTrainer:
                    trainer = EnhancedConversationTrainer(model, tokenizer, config, logging.getLogger(__name__))
                    if should_log:
                        print("Enhanced trainer initialized successfully")
                else:
                    raise ImportError("Enhanced trainer not available")
                    
            except Exception as e:
                if should_log:
                    print(f"Enhanced trainer failed, creating fallback trainer: {e}")
                trainer = create_fallback_trainer(model, tokenizer, config)
            
            # Debug training setup if available
            if hasattr(trainer, 'debug_training_setup'):
                try:
                    trainer.debug_training_setup()
                except Exception as e:
                    if should_log:
                        print(f"Debug setup failed: {e}")
            
            # Setup datasets
            try:
                train_dataset = ConversationDataset(
                    str(train_data_path), tokenizer, config, "train"
                )
                if should_log:
                    print(f"Training dataset loaded: {len(train_dataset)} samples")
                
                # Load evaluation dataset if available
                eval_dataset = None
                eval_data_path = Path(config.eval_data_path)
                if eval_data_path.exists():
                    eval_dataset = ConversationDataset(
                        str(eval_data_path), tokenizer, config, "eval"
                    )
                    if should_log:
                        print(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
                else:
                    if should_log:
                        print("No evaluation dataset found, skipping")
                        
            except Exception as e:
                if should_log:
                    print(f"Dataset loading failed: {e}")
                raise
            
            # Create data loaders for debugging if available
            if should_log and debug_dataloader:
                try:
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                    debug_dataloader(test_loader, tokenizer, max_batches=2)
                except Exception as e:
                    print(f"Dataloader debugging failed: {e}")
            
            # Start training
            try:
                if should_log:
                    print(f"\nStarting training with:")
                    print(f"  CPU offloading: {getattr(config, 'cpu_offload', False)}")
                    print(f"  DeepSpeed enabled: {getattr(config, 'use_deepspeed', False)}")
                    print(f"  ZeRO stage: {getattr(config, 'zero_stage', 'N/A')}")
                    print(f"  Trainer type: {type(trainer).__name__}")
                
                trainer.train(train_dataset, eval_dataset)
                
                if should_log:
                    print("Training completed successfully!")
                    
            except KeyboardInterrupt:
                if should_log:
                    print("Training interrupted by user")
            except Exception as e:
                if should_log:
                    print(f"Training failed: {e}")
                    traceback.print_exc()
                raise
        
        else:
            if should_log:
                print(f"\n{'='*80}")
                print("DRY RUN COMPLETED - NO TRAINING EXECUTED")
                print("Configuration successfully validated and components initialized")
                print(f"{'='*80}")
        
        # Save final configuration
        if should_log:
            config_save_path = f"experiments/{experiment_name}/final_config.yaml"
            Path(config_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if hasattr(config, 'save'):
                    config.save(config_save_path)
                    print(f"Final configuration saved: {config_save_path}")
                else:
                    # Manual save as JSON if YAML save not available
                    import json
                    config_dict = {
                        attr: getattr(config, attr) 
                        for attr in dir(config) 
                        if not attr.startswith('_') and not callable(getattr(config, attr))
                    }
                    with open(config_save_path.replace('.yaml', '.json'), 'w') as f:
                        json.dump(config_dict, f, indent=2, default=str)
                    print(f"Final configuration saved as JSON: {config_save_path.replace('.yaml', '.json')}")
            except Exception as e:
                print(f"Failed to save configuration: {e}")
                
            # Print configuration summary
            print(f"\n{'='*60}")
            print("CONFIGURATION SUMMARY")
            print(f"{'='*60}")
            print(f"Experiment: {experiment_name}")
            print(f"Model: {config_choice} with {getattr(config, 'num_experts', 8)} MoE experts")
            print(f"CPU Offloading: {getattr(config, 'cpu_offload', False)}")
            print(f"CPU Optimizer Offloading: {getattr(config, 'cpu_offload_optimizer', False)}")
            print(f"CPU Parameter Offloading: {getattr(config, 'cpu_offload_parameters', False)}")
            print(f"ZeRO Stage: {getattr(config, 'zero_stage', 'N/A')}")
            print(f"Precision: {getattr(config, 'precision', 'auto')}")
            print(f"Micro Batch Size: {getattr(config, 'batch_size', 'N/A')}")
            print(f"Gradient Accumulation: {getattr(config, 'gradient_accumulation_steps', 'N/A')}")
            print(f"Learning Rate: {getattr(config, 'learning_rate', 'N/A')}")
            print(f"Epochs: {getattr(config, 'num_epochs', 'N/A')}")
            print(f"{'='*60}")
    
    except Exception as e:
        if should_log:
            print(f"CRITICAL ERROR: {e}")
            print("Full traceback:")
            traceback.print_exc()
        
        # Emergency cleanup
        cleanup_memory()
        raise
    
    finally:
        # Final cleanup
        cleanup_memory()
        if should_log:
            print("Final cleanup completed")


if __name__ == "__main__":
    main()