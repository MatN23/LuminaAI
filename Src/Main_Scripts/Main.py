# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import traceback
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
    from core.model import estimate_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


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
    """Create necessary directory structure."""
    directories = [
        'data',
        'checkpoints', 
        'experiments',
        'logs',
        'backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def main():
    """Enhanced main function with comprehensive CLI and error handling."""
    # Setup basic logging first
    setup_logging_basic()
    
    # Create directory structure
    create_directory_structure()
    
    # Hardcoded arguments instead of argparse
    config_choice = 'medium'
    config_file = None
    train_data = 'oasst1_data/oasst1_train.jsonl'
    eval_data = 'data/eval.jsonl'
    epochs = 100
    lr = 1e-5
    batch_size = 2
    grad_accum = 4
    precision = 'fp16'
    experiment_name = 'hardcoded_experiment'
    resume = None
    seed = 42
    test_generation = True
    validate_data = None  # Skip validation since no validation.jsonl file exists
    create_report = False
    process_oasst = None
    max_conversations = None
    check_environment = False
    estimate_time = False
    dry_run = False  # Actually train the model
    dry_run = False
    
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
        # Note: ConversationTokenizer already logs its initialization details
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer: {e}")
        return 1
    
    # Data validation
    if validate_data:
        try:
            logging.info("Data Validation Results:")
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
                
                # Log any additional useful stats
                for key, value in stats.items():
                    if key not in ['conversation_stats', 'quality_metrics', 'token_stats']:
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
                'small': ConfigPresets.small,
                'medium': ConfigPresets.medium,
                'large': ConfigPresets.large,
            }
            config = config_map[config_choice]()
        
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
        if experiment_name is not None:
            config.experiment_name = experiment_name
        
        config.train_data_path = train_data
        config.eval_data_path = eval_data
        config.seed = seed
        
        # Re-validate after overrides
        config.validate()
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Training time estimation
    if estimate_time:
        try:
            # Estimate dataset size
            if Path(config.train_data_path).exists():
                with open(config.train_data_path, 'r') as f:
                    dataset_size = sum(1 for _ in f)
            else:
                dataset_size = 10000  # Default estimate
            
            estimates = estimate_training_time(config, dataset_size)
            
            logging.info("Training Time Estimates:")
            logging.info(f"  Dataset size: {dataset_size:,} conversations")
            logging.info(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
            logging.info(f"  Total tokens: {estimates['total_tokens']:,}")
            logging.info(f"  Throughput: {estimates['tokens_per_second']:,} tokens/sec")
            logging.info(f"  Memory utilization: {estimates['memory_utilization']:.1%}")
            
            if estimates['memory_warning']:
                logging.warning("  ⚠️  High memory utilization expected - consider reducing batch size")
            
            if not dry_run:
                return 0
        except Exception as e:
            logging.error(f"Time estimation failed: {e}")
            return 1
    
    # Dry run
    if dry_run:
        logging.info("Dry run completed successfully!")
        return 0
    
    # Main training
    logging.info("="*80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("="*80)
    
    try:
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Log configuration
        logging.info(f"Configuration: {config_choice}")
        logging.info(f"Model parameters: ~{estimate_parameters(config):,}")
        logging.info(f"Experiment: {config.experiment_name}")
        
        # Run training
        orchestrator.run_training()
        
        # Test generation if requested
        if test_generation and orchestrator.trainer:
            logging.info("\n" + "="*60)
            logging.info("TESTING GENERATION")
            logging.info("="*60)
            
            test_prompts = [
                "Hello, how are you today?",
                "What is machine learning?",
                "Write a simple Python function to calculate factorial.",
                "Explain the concept of recursion in programming.",
                "What are the benefits of using transformers in NLP?"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                logging.info(f"\nTest {i}/5:")
                logging.info(f"User: {prompt}")
                try:
                    response = orchestrator.trainer.generate(prompt)
                    logging.info(f"Assistant: {response}")
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                logging.info("-" * 50)
        
        logging.info("\n🎉 Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())