# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import traceback
import psutil
import gc
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from training.orchestrator import TrainingOrchestrator
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig  # Added DeepSeekConfig
    from core.dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


def config_to_deepseek_config(config):
    """Convert training Config to DeepSeekConfig."""
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


def setup_logging_basic():
    """Setup basic logging before full system initialization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directory_structure():
    """Create necessary directory structure including sharding directories."""
    directories = [
        'data',
        'data/shards',  # For sharded datasets
        'checkpoints', 
        'experiments',
        'logs',
        'backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)


def check_system_resources():
    """Check system resources and recommend configuration."""
    # Get system information
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    # Check GPU memory if available
    gpu_memory_gb = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass
    
    logging.info(f"System Resources:")
    logging.info(f"  RAM: {memory_gb:.1f} GB")
    logging.info(f"  CPU cores: {cpu_count}")
    logging.info(f"  GPU memory: {gpu_memory_gb:.1f} GB" if gpu_memory_gb > 0 else "  GPU: Not available")
    
    # Recommend sharding configuration
    if memory_gb < 16:
        recommended_shard_size = 256
        recommended_workers = min(2, cpu_count // 2)
        max_memory_usage = memory_gb * 0.6
    elif memory_gb < 64:
        recommended_shard_size = 512
        recommended_workers = min(4, cpu_count // 2)
        max_memory_usage = memory_gb * 0.7
    else:
        recommended_shard_size = 1024
        recommended_workers = min(8, cpu_count)
        max_memory_usage = memory_gb * 0.8
    
    logging.info(f"Recommended sharding config:")
    logging.info(f"  Shard size: {recommended_shard_size} MB")
    logging.info(f"  Workers: {recommended_workers}")
    logging.info(f"  Max memory usage: {max_memory_usage:.1f} GB")
    
    return {
        'max_shard_size_mb': recommended_shard_size,
        'num_workers': recommended_workers,
        'max_memory_usage_gb': max_memory_usage
    }


def auto_detect_dataset_strategy(data_path: str) -> str:
    """Automatically detect the best dataset loading strategy."""
    path = Path(data_path)
    
    if not path.exists():
        return "file_not_found"
    
    # Check if it's already sharded
    parent_dir = path.parent
    shard_files = list(parent_dir.glob("*_shard_*.jsonl"))
    if shard_files:
        total_shard_size = sum(f.stat().st_size for f in shard_files) / (1024**3)
        if total_shard_size > 10:  # > 10GB in shards
            return "streaming"
        else:
            return "sharded"
    
    # Check file size
    file_size_gb = path.stat().st_size / (1024**3)
    
    if file_size_gb < 0.5:  # < 500MB
        return "memory"
    elif file_size_gb < 10:  # < 10GB
        return "sharded"
    else:  # >= 10GB
        return "streaming"


def test_inference_precision(orchestrator):
    """Test different inference precision settings."""
    if not orchestrator.trainer:
        logging.warning("No trainer available for precision testing")
        return
    
    logging.info("\n" + "="*70)
    logging.info("TESTING INFERENCE PRECISION")
    logging.info("="*70)
    
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "Write a short Python function to add two numbers."
    ]
    
    try:
        # Test generation with multiple precisions
        for i, prompt in enumerate(test_prompts, 1):
            logging.info(f"\nTest Prompt {i}: {prompt}")
            logging.info("-" * 50)
            
            # Generate with different precisions
            responses = orchestrator.trainer.generate_with_multiple_precisions(prompt)
            
            for precision, response in responses.items():
                logging.info(f"{precision.upper()}: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Benchmark different precisions
        logging.info("\n" + "="*50)
        logging.info("PRECISION BENCHMARK")
        logging.info("="*50)
        
        benchmark_results = orchestrator.trainer.benchmark_inference_precision(
            test_prompts=test_prompts[:2],  # Use fewer prompts for benchmark
            max_new_tokens=50
        )
        
        # Display benchmark results
        for precision, results in benchmark_results.items():
            if results['success']:
                logging.info(f"{precision.upper()} Performance:")
                logging.info(f"  Tokens/second: {results['tokens_per_second']:.1f}")
                logging.info(f"  Avg time/prompt: {results['avg_time_per_prompt']:.3f}s")
                logging.info(f"  Peak memory: {results['peak_memory_mb']:.1f}MB")
            else:
                logging.warning(f"{precision.upper()}: Failed to complete benchmark")
                
    except Exception as e:
        logging.error(f"Precision testing failed: {e}")


def create_sharding_aware_config(base_config, system_resources: dict, dataset_strategy: str):
    """Create configuration optimized for the detected dataset strategy."""
    
    # Add sharding configuration to base config
    base_config.max_shard_size_mb = system_resources['max_shard_size_mb']
    base_config.max_memory_usage_gb = system_resources['max_memory_usage_gb']
    base_config.enable_memory_mapping = True
    base_config.shard_shuffle = True
    
    # Adjust batch size and workers based on strategy
    if dataset_strategy == "streaming":
        # For streaming datasets, use smaller batches and more workers
        base_config.batch_size = min(base_config.batch_size, 1)
        base_config.gradient_accumulation_steps = max(base_config.gradient_accumulation_steps, 8)
        base_config.num_workers = system_resources['num_workers']
        logging.info("Optimized config for streaming (massive dataset)")
        
    elif dataset_strategy == "sharded":
        # For sharded datasets, balance batch size and workers
        base_config.num_workers = max(2, system_resources['num_workers'] // 2)
        logging.info("Optimized config for sharded (large dataset)")
        
    else:  # memory strategy
        # For in-memory datasets, can use larger batches
        base_config.num_workers = min(4, system_resources['num_workers'])
        logging.info("Optimized config for in-memory (small dataset)")
    
    return base_config


def main():
    """Enhanced main function with comprehensive sharding support for any dataset size."""
    # Setup basic logging first
    setup_logging_basic()
    
    # Create directory structure
    create_directory_structure()
    
    # Check system resources
    system_resources = check_system_resources()
    
    # Hardcoded arguments with enhanced sharding support
    config_choice = 'debug'
    config_file = None
    train_data = 'oasst1_data/oasst1_train.jsonl'
    eval_data = 'data/eval.jsonl'
    epochs = 10
    lr = 3e-4
    batch_size = 2
    grad_accum = 4
    precision = 'auto'
    inference_precision = 'auto'
    experiment_name = 'LuminaAI_Experiment_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    resume = None
    seed = 42
    test_generation = True
    test_precision = True
    validate_data = None
    create_report = False
    process_oasst = None
    max_conversations = None
    check_environment = False
    estimate_time = False
    dry_run = False
    
    # New sharding-specific options
    force_streaming = False  # Force streaming mode even for smaller datasets
    disable_sharding = False  # Disable sharding (for testing)
    shard_size_mb = None  # Override shard size
    
    # Auto-detect dataset strategy
    dataset_strategy = auto_detect_dataset_strategy(train_data)
    logging.info(f"Detected dataset strategy: {dataset_strategy}")
    
    # Environment validation
    if check_environment:
        logging.info("Checking training environment...")
        issues = validate_environment()
        if issues:
            logging.warning("Environment issues found:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Environment looks good!")
        
        if not dry_run:
            return 0
    
    # Data processing
    if process_oasst:
        input_file, output_file = process_oasst
        try:
            count = process_oasst_data(input_file, output_file, max_conversations)
            logging.info(f"Successfully processed {count} conversations")
            return 0
        except Exception as e:
            logging.error(f"Data processing failed: {e}")
            return 1
    
    # Initialize tokenizer and log its details
    try:
        tokenizer = ConversationTokenizer(model_name="gpt-4")
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer: {e}")
        return 1
    
    # Data validation with sharding awareness
    if validate_data:
        try:
            logging.info("Data Validation Results:")
            
            # Check if we're validating a sharded dataset
            validate_path = Path(validate_data)
            if not validate_path.exists():
                # Check for sharded files
                shard_files = list(validate_path.parent.glob(f"{validate_path.stem}_shard_*.jsonl"))
                if shard_files:
                    logging.info(f"Found {len(shard_files)} shard files for validation")
                    # Validate first shard as representative
                    validate_data = str(shard_files[0])
                else:
                    logging.error(f"Data validation failed - no file found: {validate_data}")
                    return 1
            
            stats = validate_data_comprehensive(validate_data, tokenizer)
            
            if stats is None:
                logging.error("Data validation failed - no stats returned")
                return 1
            
            # Log validation results with safe access to keys
            try:
                # Handle conversation stats
                if 'conversation_stats' in stats:
                    conv_stats = stats['conversation_stats']
                    valid_convs = conv_stats.get('valid_conversations', conv_stats.get('total_conversations', 0))
                    logging.info(f"  Valid conversations: {valid_convs:,}")
                elif 'total_conversations' in stats:
                    logging.info(f"  Total conversations: {stats['total_conversations']:,}")
                
                # Handle quality metrics
                if 'quality_metrics' in stats:
                    qual_metrics = stats['quality_metrics']
                    success_rate = qual_metrics.get('success_rate', qual_metrics.get('validation_success_rate', 0))
                    logging.info(f"  Success rate: {success_rate:.2%}")
                elif 'success_rate' in stats:
                    logging.info(f"  Success rate: {stats['success_rate']:.2%}")
                
                # Handle token stats
                if 'token_stats' in stats:
                    token_stats = stats['token_stats']
                    avg_tokens = token_stats.get('avg_tokens', token_stats.get('average_tokens', 0))
                    logging.info(f"  Average tokens: {avg_tokens:.1f}")
                elif 'avg_tokens' in stats:
                    logging.info(f"  Average tokens: {stats['avg_tokens']:.1f}")
                elif 'average_tokens' in stats:
                    logging.info(f"  Average tokens: {stats['average_tokens']:.1f}")
                
                # Log dataset strategy
                if 'sharding_strategy' in stats:
                    logging.info(f"  Loading strategy: {stats['sharding_strategy']}")
                
                # Log additional useful stats
                for key, value in stats.items():
                    if key not in ['conversation_stats', 'quality_metrics', 'token_stats', 'sharding_strategy']:
                        if isinstance(value, (int, float)):
                            if isinstance(value, float) and 0 <= value <= 1:
                                logging.info(f"  {key.replace('_', ' ').title()}: {value:.2%}")
                            else:
                                logging.info(f"  {key.replace('_', ' ').title()}: {value:,}")
                        elif isinstance(value, str):
                            logging.info(f"  {key.replace('_', ' ').title()}: {value}")
                            
            except Exception as stat_error:
                logging.warning(f"Error processing stats: {stat_error}")
                logging.info(f"Raw stats keys: {list(stats.keys())}")
            
            if create_report:
                create_data_summary_report([validate_data], tokenizer)
            
            return 0
            
        except FileNotFoundError:
            logging.error(f"Data validation failed: File '{validate_data}' not found")
            return 1
        except KeyError as e:
            logging.error(f"Data validation failed: Missing key {e}")
            logging.debug(f"Available keys: {list(stats.keys()) if 'stats' in locals() else 'No stats available'}")
            return 1
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            logging.debug(traceback.format_exc())
            return 1
    
    # Load configuration
    try:
        if config_file:
            config = Config.load(config_file)
        else:
            config_map = {
                'debug': ConfigPresets.debug,
                'b1': ConfigPresets.b1,
                'b7': ConfigPresets.b7,
                'b14': ConfigPresets.b14,
                'b50': ConfigPresets.b50,
                'b100': ConfigPresets.b100,
                'b200': ConfigPresets.b200,
                'b300': ConfigPresets.b300,
                'b3_inference': ConfigPresets.b3_inference,
                'b6_quality': ConfigPresets.b6_quality,
                'm120_speed': ConfigPresets.m120_speed,
                'm70_memory': ConfigPresets.m70_memory
            }
            config = config_map[config_choice]()
        
        # Apply system resource optimizations
        config = create_sharding_aware_config(config, system_resources, dataset_strategy)
        
        # Apply hardcoded overrides
        if epochs is not None:
            config.num_epochs = epochs
        if lr is not None:
            config.learning_rate = lr
        if batch_size is not None:
            config.batch_size = batch_size
        if grad_accum is not None:
            config.gradient_accumulation_steps = grad_accum
        if precision is not None:
            config.precision = precision
        if inference_precision is not None:
            config.inference_precision = inference_precision
        if experiment_name is not None:
            config.experiment_name = experiment_name
        if shard_size_mb is not None:
            config.max_shard_size_mb = shard_size_mb
        
        config.train_data_path = train_data
        config.eval_data_path = eval_data
        config.seed = seed
        
        # Override lr_scheduler
        config.lr_scheduler = None
        
        # Re-validate after overrides
        config.validate()
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Training time estimation with sharding awareness
    if estimate_time:
        try:
            # Estimate dataset size more accurately for sharded datasets
            if Path(config.train_data_path).exists():
                with open(config.train_data_path, 'r') as f:
                    dataset_size = sum(1 for _ in f)
            else:
                # Check for sharded files
                train_path = Path(config.train_data_path)
                shard_files = list(train_path.parent.glob(f"{train_path.stem}_shard_*.jsonl"))
                if shard_files:
                    dataset_size = 0
                    for shard_file in shard_files:
                        try:
                            with open(shard_file, 'r') as f:
                                dataset_size += sum(1 for _ in f)
                        except Exception:
                            continue
                    logging.info(f"Counted {dataset_size:,} conversations across {len(shard_files)} shards")
                else:
                    dataset_size = 10000  # Default estimate
            
            # Create DeepSeek config for parameter estimation
            deepseek_config = config_to_deepseek_config(config)
            estimates = estimate_training_time(config, dataset_size)  # Pass the main 'config', not 'deepseek_config'
            
            logging.info("Training Time Estimates:")
            logging.info(f"  Dataset size: {dataset_size:,} conversations")
            logging.info(f"  Dataset strategy: {dataset_strategy}")
            logging.info(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
            logging.info(f"  Total tokens: {estimates['total_tokens']:,}")
            logging.info(f"  Throughput: {estimates['tokens_per_second']:,} tokens/sec")
            logging.info(f"  Memory utilization: {estimates['memory_utilization']:.1%}")
            logging.info(f"  Training precision: {config.precision}")
            logging.info(f"  Inference precision: {config.inference_precision}")
            logging.info(f"  Shard size: {config.max_shard_size_mb}MB")
            
            if estimates['memory_warning']:
                logging.warning("  High memory utilization expected - consider reducing batch size")
            
            if dataset_strategy == "streaming":
                logging.info("  Note: Streaming mode will minimize memory usage")
            
        except Exception as e:
            logging.error(f"Time estimation failed: {e}")
            return 1
    
    # Dry run
    if dry_run:
        logging.info("Dry run completed successfully!")
        logging.info(f"Would use training precision: {config.precision}")
        logging.info(f"Would use inference precision: {config.inference_precision}")
        logging.info(f"Would use dataset strategy: {dataset_strategy}")
        logging.info(f"Would use shard size: {config.max_shard_size_mb}MB")
        return 0
    
    # Main training with sharding support
    logging.info("="*80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING WITH SHARDING")
    logging.info("="*80)
    
    try:
        # Memory monitoring setup
        initial_memory = psutil.virtual_memory().percent
        logging.info(f"Initial memory usage: {initial_memory:.1f}%")
        
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Create DeepSeek config for parameter estimation  
        deepseek_config = config_to_deepseek_config(config)
        
        # Log configuration with sharding info
        logging.info(f"Configuration: {config_choice}")
        logging.info(f"Model parameters: ~{estimate_parameters(deepseek_config):,}")
        logging.info(f"Experiment: {config.experiment_name}")
        logging.info(f"Training precision: {config.precision}")
        logging.info(f"Inference precision: {config.inference_precision}")
        logging.info(f"Dataset strategy: {dataset_strategy}")
        logging.info(f"Shard configuration:")
        logging.info(f"  Max shard size: {config.max_shard_size_mb}MB")
        logging.info(f"  Max memory usage: {config.max_memory_usage_gb:.1f}GB")
        logging.info(f"  Workers: {config.num_workers}")
        
        # Run training with memory monitoring
        memory_before = psutil.virtual_memory().percent
        
        orchestrator.run_training()
        
        memory_after = psutil.virtual_memory().percent
        logging.info(f"Memory usage: {memory_before:.1f}% -> {memory_after:.1f}%")
        
        # Test precision if requested
        if test_precision and orchestrator.trainer:
            test_inference_precision(orchestrator)
        
        # Test generation with sharding-aware memory management
        if test_generation and orchestrator.trainer:
            logging.info("\n" + "="*60)
            logging.info("TESTING GENERATION WITH SHARDING SUPPORT")
            logging.info("="*60)
            
            test_prompts = [
                "Hello, how are you today?",
                "What is machine learning?",
                "Write a simple Python function to calculate factorial.",
                "Explain the concept of recursion in programming.",
                "What are the benefits of using transformers in NLP?"
            ]
            
            # Memory monitoring during generation
            memory_before_gen = psutil.virtual_memory().percent
            
            # Test with different inference precisions
            for precision_test in ["fp32", "fp16", "bf16"]:
                logging.info(f"\n--- Testing with {precision_test.upper()} inference precision ---")
                
                # Set inference precision
                try:
                    orchestrator.trainer.set_inference_precision(precision_test)
                except Exception as e:
                    logging.warning(f"Failed to set {precision_test} precision: {e}")
                    continue
                
                for i, prompt in enumerate(test_prompts[:3], 1):  # Test first 3 prompts
                    logging.info(f"\nTest {i}/3 ({precision_test.upper()}):")
                    logging.info(f"User: {prompt}")
                    try:
                        # Monitor memory during generation
                        mem_before = psutil.virtual_memory().percent
                        
                        response = orchestrator.trainer.generate(
                            prompt,
                            max_new_tokens=100,
                            temperature=0.7
                        )
                        
                        mem_after = psutil.virtual_memory().percent
                        
                        logging.info(f"Assistant: {response}")
                        logging.info(f"Memory: {mem_before:.1f}% -> {mem_after:.1f}%")
                        
                        # Force garbage collection to manage memory
                        if mem_after > 85:  # If memory usage is high
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except ImportError:
                                pass
                            
                    except Exception as e:
                        logging.error(f"Generation failed with {precision_test}: {e}")
                    logging.info("-" * 50)
            
            memory_after_gen = psutil.virtual_memory().percent
            logging.info(f"\nGeneration memory impact: {memory_before_gen:.1f}% -> {memory_after_gen:.1f}%")
            
            # Reset to original inference precision
            try:
                orchestrator.trainer.set_inference_precision(config.inference_precision)
                logging.info(f"\nReset inference precision to: {config.inference_precision}")
            except Exception as e:
                logging.warning(f"Failed to reset inference precision: {e}")
        
        # Final memory cleanup
        final_memory = psutil.virtual_memory().percent
        logging.info(f"\nFinal memory usage: {final_memory:.1f}%")
        
        if final_memory > 90:
            logging.warning("High memory usage detected - running cleanup...")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        logging.info("\nTraining and testing completed successfully!")
        logging.info(f"Dataset strategy used: {dataset_strategy}")
        logging.info(f"Final inference precision: {orchestrator.trainer.inference_precision}")
        
        # Summary of sharding benefits
        if dataset_strategy in ["sharded", "streaming"]:
            logging.info("\nSharding Benefits Achieved:")
            logging.info(f"  Memory efficient loading: Yes")
            logging.info(f"  Scalable to massive datasets: Yes")
            logging.info(f"  Parallel processing: Yes")
            logging.info(f"  Automatic strategy selection: Yes")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Cleanup on interrupt
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        # Cleanup on error
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return 1


def test_sharding_system():
    """Test the sharding system with different dataset sizes."""
    logging.info("="*60)
    logging.info("TESTING SHARDING SYSTEM")
    logging.info("="*60)
    
    try:
        # Test with different file sizes
        test_cases = [
            ("b1", 1000000),    # 100 conversations
            ("b7", 1000000000000),   # 1,000 conversations
            ("b14", 10000000000), # 10,000 conversations
            ("b50", 50000000000000), # 50,000 conversations
            ("b100", 100000000000), # 100,000 conversations 
            ("b200", 200000000000000),  # 200,000 conversations
            ("b300", 300000000000000)   # 300,000 conversations
        ]
        
        for test_name, conv_count in test_cases:
            logging.info(f"\nTesting {test_name} dataset simulation ({conv_count:,} conversations)")
            
            # Create mock conversations
            mock_conversations = []
            for i in range(conv_count):
                mock_conv = {
                    'conversation_id': f'test_{i}',
                    'messages': [
                        {'role': 'user', 'content': f'Test question {i}', 'turn': 1},
                        {'role': 'assistant', 'content': f'Test response {i}', 'turn': 2}
                    ],
                    'total_turns': 2
                }
                mock_conversations.append(mock_conv)
            
            # Test memory usage
            import sys
            memory_mb = sys.getsizeof(mock_conversations) / (1024 * 1024)
            logging.info(f"  Memory usage: {memory_mb:.1f}MB")
            
            # Simulate sharding decision
            if memory_mb < 100:
                strategy = "memory"
            elif memory_mb < 1000:
                strategy = "sharded"
            else:
                strategy = "streaming"
            
            logging.info(f"  Recommended strategy: {strategy}")
            
            # Cleanup
            del mock_conversations
            gc.collect()
    
    except Exception as e:
        logging.error(f"Sharding test failed: {e}")


if __name__ == "__main__":
    # Add sharding system test option
    if len(sys.argv) > 1 and sys.argv[1] == "--test-sharding":
        setup_logging_basic()
        test_sharding_system()
        exit(0)
    
    exit(main())