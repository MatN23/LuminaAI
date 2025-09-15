# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
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

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from training.orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
    from training.trainer import EnhancedConversationTrainer, DeepSpeedConfigGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


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
            info.update({
                'device': f'cuda:{self.local_rank}',
                'gpu_name': torch.cuda.get_device_name(self.local_rank),
                'gpu_memory_gb': torch.cuda.get_device_properties(self.local_rank).total_memory / 1e9
            })
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
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                info['gpu_info'].append({
                    'device_id': i,
                    'name': gpu_props.name,
                    'memory_gb': gpu_props.total_memory / 1e9,
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                    'multiprocessor_count': gpu_props.multi_processor_count
                })
        
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
            'use_deepspeed': DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', True),
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
        total_gpu_memory = sum(gpu['memory_gb'] for gpu in self.system_info.get('gpu_info', []))
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
        if hasattr(config, 'use_moe') and config.use_moe:
            num_experts = getattr(config, 'num_experts', 8)
            world_size = self.distributed_manager.world_size
            
            moe_config = DeepSpeedConfigGenerator.create_moe_config_with_expert_parallelism(
                world_size, num_experts, model_size_gb, sequence_length
            )
            strategy['moe_optimizations'] = moe_config['moe']
            
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
                                natural_description: Optional[str] = None) -> Config:
        """Automatically configure DeepSpeed based on system analysis."""
        
        # Estimate model size from config
        try:
            model_size_gb = self.resource_manager._estimate_model_memory_usage(config)
            sequence_length = getattr(config, 'seq_length', 2048)
        except Exception as e:
            logging.warning(f"Could not estimate model size: {e}")
            model_size_gb = 10.0  # Default estimate
            sequence_length = 2048
        
        # Analyze system and create strategy
        strategy = self.resource_manager.create_deepspeed_optimization_strategy(
            config, model_size_gb, sequence_length
        )
        
        # Apply optimizations to config
        apply_deepspeed_optimizations(config, strategy)
        
        # Log strategy
        logging.info("DeepSpeed Optimization Strategy:")
        logging.info(f"  Model size estimate: {model_size_gb:.1f}GB")
        logging.info(f"  ZeRO stage: {strategy['zero_stage']}")
        logging.info(f"  CPU offload: {strategy['cpu_offload']}")
        logging.info(f"  Precision: {strategy['precision_strategy']}")
        
        if strategy.get('moe_optimizations'):
            moe = strategy['moe_optimizations']
            logging.info(f"  MoE experts: {moe.get('num_experts', 'N/A')}")
            logging.info(f"  Expert parallel size: {moe.get('expert_parallel_size', 'N/A')}")
        
        return config


def apply_deepspeed_optimizations(config: Config, strategy: Dict[str, Any]):
    """Apply DeepSpeed optimization strategy to config."""
    
    # Enable DeepSpeed
    config.use_deepspeed = strategy.get('use_deepspeed', True)
    
    # ZeRO configuration
    config.zero_stage = strategy.get('zero_stage', 3)
    
    # CPU offloading
    config.cpu_offload = strategy.get('cpu_offload', False)
    config.cpu_offload_optimizer = strategy.get('cpu_offload', False)
    config.aggressive_cpu_offload = strategy.get('aggressive_cpu_offload', False)
    
    # NVMe offloading
    if strategy.get('nvme_offload', False):
        config.nvme_path = strategy.get('nvme_path')
    
    # Precision strategy
    if strategy.get('precision_strategy'):
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


def main():
    """Enhanced main function with comprehensive DeepSpeed integration."""
    
    # HARDCODED CONFIGURATION - MODIFY THESE PARAMETERS
    # =================================================
    
    # Base model configuration - select from ConfigPresets
    config_choice = 'debug'  # Options: 'debug', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'
    
    # Override specific parameters (set to None to use preset defaults)
    override_params = {
        'use_moe': True,        # Enable MoE
        'num_epochs': 10,       # Training epochs
        'learning_rate': 1e-4, # Learning rate
        'batch_size': 1,       # Micro batch size
        'gradient_accumulation_steps': 4,
        'train_data_path': 'oasst1_data/oasst1_train.jsonl',
        'eval_data_path': 'data/eval.jsonl',
    }
    
    # DeepSpeed and optimization settings
    enable_deepspeed = DEEPSPEED_AVAILABLE
    enable_cpu_offload = True
    enable_aggressive_optimization = True
    nvme_path = None  # Set to NVMe path if available, e.g., '/tmp/deepspeed_nvme'
    zero_stage = 3    # ZeRO optimization stage (1, 2, or 3)
    
    # Optimization flags
    optimize_for_long_sequences = True
    check_environment = True
    estimate_time = True
    dry_run = False  # Set to True to test configuration without training
    
    # Data processing flags
    validate_data_path = None  # Set to data file path to validate
    process_oasst = None      # Set to (input_file, output_file) tuple to process
    create_report = False
    
    # =================================================
    # END HARDCODED CONFIGURATION
    
    experiment_name = f'DeepSpeed_MoE_{override_params.get("num_experts", 8)}E_Experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
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
        print("ADAPTIVE AI TRAINING SYSTEM WITH DEEPSPEED INTEGRATION")
        print("   Self-Improving • Intelligent • Production-Ready • Distributed")
        print("="*80)
    
    # Initialize enhanced resource manager
    resource_manager = EnhancedResourceManager(dist_manager)
    
    if should_log:
        # Display comprehensive system information
        print(f"\nDistributed Training Information:")
        dist_info = dist_manager.get_device_info()
        print(f"   World size: {dist_info['world_size']}")
        print(f"   Global rank: {dist_info['rank']}")
        print(f"   Local rank: {dist_info['local_rank']}")
        print(f"   Device: {dist_info.get('device', 'cpu')}")
        
        if 'gpu_name' in dist_info:
            print(f"   GPU: {dist_info['gpu_name']} ({dist_info['gpu_memory_gb']:.1f}GB)")
        
        print(f"\nSystem Resources:")
        system_info = resource_manager.system_info
        print(f"   RAM: {system_info['memory_gb']:.1f}GB ({system_info['memory_usage_percent']:.1f}% used)")
        print(f"   CPU cores: {system_info['cpu_count']}")
        print(f"   Storage: {system_info['disk_free_gb']:.1f}GB free")
        
        if system_info.get('gpu_info'):
            print(f"   Total GPUs: {len(system_info['gpu_info'])}")
            for gpu in system_info['gpu_info']:
                print(f"     GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    try:
        # Environment validation
        if check_environment and should_log:
            logging.info("Validating training environment...")
            issues = validate_environment()
            if issues:
                logging.warning("Environment issues found:")
                for issue in issues:
                    logging.warning(f"     - {issue}")
            else:
                logging.info("Environment validation passed!")
            
            if dry_run:
                return 0
        
        # Data processing
        if process_oasst:
            input_file, output_file = process_oasst
            try:
                count = process_oasst_data(input_file, output_file)
                if should_log:
                    logging.info(f"Successfully processed {count:,} conversations")
                return 0
            except Exception as e:
                if should_log:
                    logging.error(f"Data processing failed: {e}")
                return 1
        
        # Load and configure base model
        try:
            # Get base configuration from preset
            config_map = {
                'debug': ConfigPresets.debug,
                'b1': ConfigPresets.b1,
                'b7': ConfigPresets.b7,
                'b14': ConfigPresets.b14,
                'b50': ConfigPresets.b50,
                'b100': ConfigPresets.b100,
                'b200': ConfigPresets.b200,
                'b300': ConfigPresets.b300,
            }
            
            if config_choice not in config_map:
                raise ValueError(f"Invalid config choice: {config_choice}")
            
            config = config_map[config_choice]()

            # Apply parameter overrides
            if override_params:
                for param, value in override_params.items():
                    if value is not None and hasattr(config, param):
                        setattr(config, param, value)
                        if should_log:
                            logging.info(f"Override: {param} = {value}")
            
            # Set experiment name
            config.experiment_name = experiment_name
            
            # Add DeepSpeed-specific attributes if not present
            if not hasattr(config, 'use_deepspeed'):
                config.use_deepspeed = enable_deepspeed
            if not hasattr(config, 'zero_stage'):
                config.zero_stage = zero_stage
            if not hasattr(config, 'cpu_offload'):
                config.cpu_offload = enable_cpu_offload
            if nvme_path and not hasattr(config, 'nvme_path'):
                config.nvme_path = nvme_path
            
            # Enhanced DeepSpeed configuration
            if enable_deepspeed and config.use_deepspeed:
                wizard = AdaptiveDeepSpeedWizard(resource_manager)
                config = wizard.auto_configure_deepspeed(config)
                
                if should_log:
                    logging.info(f"DeepSpeed optimization applied")
            
            config.validate()
            if should_log:
                logging.info(f"Configuration loaded and optimized: {config_choice}")
            
        except Exception as e:
            if should_log:
                logging.error(f"Configuration error: {e}")
                logging.error(traceback.format_exc())
            return 1
        
        # Initialize tokenizer - FIXED: Moved after config initialization
        try:
            tokenizer = ConversationTokenizer(model_name="gpt-4")
            config.vocab_size = tokenizer.get_vocab_size()  # Ensure targets are in range
            if should_log:
                logging.info("Tokenizer initialized successfully")
        except Exception as e:
            if should_log:
                logging.error(f"Failed to initialize tokenizer: {e}")
            return 1
        
        # Data validation
        if validate_data_path:
            try:
                if should_log:
                    logging.info(f"Validating data: {validate_data_path}")
                
                validate_path = Path(validate_data_path)
                if not validate_path.exists():
                    shard_files = list(validate_path.parent.glob(f"{validate_path.stem}_shard_*.jsonl"))
                    if shard_files:
                        if should_log:
                            logging.info(f"Found {len(shard_files)} shard files")
                        validate_data_path = str(shard_files[0])
                    else:
                        if should_log:
                            logging.error(f"File not found: {validate_data_path}")
                        return 1
                
                stats = validate_data_comprehensive(validate_data_path, tokenizer)
                
                if stats and should_log:
                    logging.info("Validation Results:")
                    for key, value in stats.items():
                        if isinstance(value, dict):
                            continue
                        elif isinstance(value, float) and 0 <= value <= 1:
                            logging.info(f"   {key.replace('_', ' ').title()}: {value:.2%}")
                        elif isinstance(value, (int, float)):
                            logging.info(f"   {key.replace('_', ' ').title()}: {value:,}")
                
                return 0
                
            except Exception as e:
                if should_log:
                    logging.error(f"Data validation failed: {e}")
                return 1
        
        # Time estimation
        if estimate_time and should_log:
            try:
                # Simple time estimation
                try:
                    with open(config.train_data_path, 'r') as f:
                        dataset_size = sum(1 for _ in f)
                except:
                    dataset_size = 10000
                
                total_tokens = dataset_size * config.seq_length
                if config.use_moe:
                    # MoE models are typically slower due to routing overhead
                    tokens_per_second = 100  # Conservative estimate
                else:
                    tokens_per_second = 500
                
                estimated_hours = total_tokens / (tokens_per_second * 3600)
                
                print(f"\nTraining Time Estimate:")
                print(f"   Total tokens: {total_tokens:,}")
                print(f"   Estimated throughput: {tokens_per_second:,} tokens/sec")
                print(f"   Estimated time: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
                
                if config.use_moe:
                    print(f"   Note: MoE routing may add 20-50% overhead")
                
            except Exception as e:
                logging.error(f"Time estimation failed: {e}")
        
        # Dry run
        if dry_run:
            if should_log:
                model_size_gb = resource_manager._estimate_model_memory_usage(config)
                print(f"\nDry Run Summary:")
                print(f"   Configuration: {config_choice}")
                print(f"   MoE: {config.use_moe} ({config.num_experts} experts)" if config.use_moe else "   MoE: Disabled")
                print(f"   DeepSpeed: {'Enabled' if getattr(config, 'use_deepspeed', False) else 'Disabled'}")
                print(f"   CPU offload: {'Enabled' if getattr(config, 'cpu_offload', False) else 'Disabled'}")
                print(f"   Sequence length: {config.seq_length:,}")
                print(f"   Estimated model size: {model_size_gb:.1f}GB")
                print(f"   World size: {dist_manager.world_size}")
                print(f"\nDry run completed - ready for training!")
            return 0
        
        # Initialize model and trainer
        try:
            if should_log:
                print(f"\nInitializing model and training system...")
            
            # Create model
            deepseek_config = config_to_deepseek_config(config)
            model = DeepSeekTransformer(deepseek_config)
            
            # Create enhanced trainer with DeepSpeed support
            trainer = EnhancedConversationTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config,
                logger=None  # We'll use the built-in logging
            )
            
            if should_log:
                print(f"\nTraining Configuration:")
                print(f"   Experiment: {config.experiment_name}")
                print(f"   Model: ~{estimate_parameters(deepseek_config):,} parameters")
                print(f"   Training mode: {'DeepSpeed' if getattr(trainer, 'use_deepspeed', False) else 'Standard PyTorch'}")
                print(f"   World size: {dist_manager.world_size}")
                print(f"   Sequence length: {config.seq_length:,}")
                print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps * dist_manager.world_size}")
                
                if config.use_moe:
                    print(f"   MoE experts: {config.num_experts}")
                    print(f"   MoE routing: top-{config.moe_top_k}")
                    if hasattr(config, 'capacity_factor'):
                        print(f"   Capacity factor: {config.capacity_factor}")
                
                if getattr(trainer, 'use_deepspeed', False):
                    print(f"   CPU offload: {'Enabled' if getattr(config, 'cpu_offload', False) else 'Disabled'}")
                    print(f"   ZeRO stage: {getattr(config, 'zero_stage', 'Not set')}")
            
            # Load datasets - FIXED: Use correct parameter name
            train_dataset = ConversationDataset(
                config.train_data_path,
                tokenizer,
                config,
            )
            
            eval_dataset = None
            if Path(config.eval_data_path).exists():
                eval_dataset = ConversationDataset(
                        config.eval_data_path,
                    tokenizer,
                    config,
                )
            
            # Run training
            if should_log:
                print(f"\n" + "="*80)
                print(f"STARTING ENHANCED TRAINING")
                print(f"="*80)
            
            start_time = datetime.now()
            
            # Optimize for long sequences if enabled
            if optimize_for_long_sequences and getattr(trainer, 'use_deepspeed', False):
                trainer.optimize_for_sequence_length(config.seq_length)
            
            trainer.train(train_dataset, eval_dataset)
            
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            if should_log:
                print(f"\nTraining completed successfully!")
                print(f"   Duration: {training_duration:.1f} seconds ({training_duration/3600:.2f} hours)")
                
                # Get MoE diagnostics if available
                if config.use_moe and getattr(trainer, 'use_deepspeed', False):
                    try:
                        moe_diagnostics = trainer.get_moe_diagnostics()
                        if 'recommendations' in moe_diagnostics and moe_diagnostics['recommendations']:
                            print(f"\nMoE Routing Recommendations:")
                            for rec in moe_diagnostics['recommendations']:
                                print(f"   - {rec}")
                    except Exception as e:
                        logging.debug(f"Could not get MoE diagnostics: {e}")
                
                # Memory statistics
                try:
                    memory_stats = trainer.get_memory_stats()
                    if 'gpu' in memory_stats:
                        gpu_stats = memory_stats['gpu']
                        print(f"\nFinal Memory Usage:")
                        print(f"   GPU allocated: {gpu_stats['allocated_gb']:.2f}GB")
                        print(f"   GPU reserved: {gpu_stats['reserved_gb']:.2f}GB")
                        print(f"   Peak GPU usage: {gpu_stats['max_allocated_gb']:.2f}GB")
                except Exception as e:
                    logging.debug(f"Could not get memory stats: {e}")
            
            return 0
            
        except KeyboardInterrupt:
            if should_log:
                print(f"\nTraining interrupted by user")
            return 1
        except Exception as e:
            if should_log:
                logging.error(f"Training failed: {e}")
                logging.error(traceback.format_exc())
            return 1
        
    except Exception as e:
        if should_log:
            logging.error(f"Main execution failed: {e}")
            logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # For DeepSpeed, we need to handle the distributed launch
    if DEEPSPEED_AVAILABLE and len(sys.argv) > 1 and '--local_rank' in ' '.join(sys.argv):
        # DeepSpeed distributed launch
        import torch
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
    
    exit_code = main()
    
    # Clean up distributed training
    if DEEPSPEED_AVAILABLE:
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except:
            pass
    
    exit(exit_code)