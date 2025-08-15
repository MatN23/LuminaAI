# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
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
    """Simplified main function with default settings."""
    # Setup basic logging first
    setup_logging_basic()
    
    # Create directory structure
    create_directory_structure()
    
    # ================================
    # EASY CONFIGURATION SECTION
    # ================================
    # Modify these variables to customize behavior:
    
    config_preset = 'large'          # Options: 'debug', 'small', 'medium', 'large'
    train_data_path = 'oasst1_data/train.jsonl'  # Path to training data
    eval_data_path = 'oasst1_data/validation_conversations.jsonl'  # Path to evaluation data
    seed = 42
    test_generation = True           # Set to False to skip generation testing
    check_environment = True         # Set to False to skip environment check
    estimate_time = True            # Set to False to skip time estimation
    
    # Advanced overrides (set to None to use preset defaults)
    epochs_override = None          # e.g., 10
    learning_rate_override = None   # e.g., 1e-4
    batch_size_override = None      # e.g., 8
    
    # ================================
    # END CONFIGURATION SECTION
    # ================================
    
    logging.info("="*80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("="*80)
    logging.info(f"Using configuration preset: {config_preset}")
    
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
    
    # Load default configuration
    try:
        config_map = {
            'debug': ConfigPresets.debug,
            'small': ConfigPresets.small,
            'medium': ConfigPresets.medium,
            'large': ConfigPresets.large,
        }
        config = config_map[config_preset]()
        
        # Set default paths and settings
        config.train_data_path = train_data_path
        config.eval_data_path = eval_data_path
        config.seed = seed
        
        # Apply overrides if provided
        if epochs_override is not None:
            config.num_epochs = epochs_override
            logging.info(f"Override: epochs = {epochs_override}")
        if learning_rate_override is not None:
            config.learning_rate = learning_rate_override
            logging.info(f"Override: learning_rate = {learning_rate_override}")
        if batch_size_override is not None:
            config.batch_size = batch_size_override
            logging.info(f"Override: batch_size = {batch_size_override}")
        
        # Validate configuration
        config.validate()
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Training time estimation (if enabled and data exists)
    if estimate_time:
        try:
            if Path(config.train_data_path).exists():
                with open(config.train_data_path, 'r') as f:
                    dataset_size = sum(1 for _ in f)
            else:
                dataset_size = 1000  # Default estimate for debug mode
                logging.warning(f"Training data not found at {config.train_data_path}")
                logging.info(f"Using estimated dataset size: {dataset_size}")
            
            estimates = estimate_training_time(config, dataset_size)
            
            logging.info("Training Time Estimates:")
            logging.info(f"  Dataset size: {dataset_size:,} conversations")
            logging.info(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
            logging.info(f"  Total tokens: {estimates['total_tokens']:,}")
            logging.info(f"  Throughput: {estimates['tokens_per_second']:,} tokens/sec")
            logging.info(f"  Memory utilization: {estimates['memory_utilization']:.1%}")
            
            if estimates['memory_warning']:
                logging.warning("  ⚠️  High memory utilization expected - consider reducing batch size")
                
        except Exception as e:
            logging.warning(f"Could not estimate training time: {e}")
    
    # Main training
    try:
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Log configuration details
        logging.info("\nTraining Configuration:")
        logging.info(f"  Preset: {config_preset}")
        logging.info(f"  Model parameters: ~{estimate_parameters(config):,}")
        logging.info(f"  Experiment: {config.experiment_name}")
        logging.info(f"  Epochs: {config.num_epochs}")
        logging.info(f"  Learning rate: {config.learning_rate}")
        logging.info(f"  Batch size: {config.batch_size}")
        logging.info(f"  Precision: {getattr(config, 'precision', 'fp32')}")
        
        # Run training
        logging.info("\nStarting training...")
        orchestrator.run_training()
        
        # Test generation if enabled
        if test_generation and hasattr(orchestrator, 'trainer') and orchestrator.trainer:
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
                logging.info(f"\nTest {i}/{len(test_prompts)}:")
                logging.info(f"User: {prompt}")
                try:
                    response = orchestrator.trainer.generate(prompt)
                    logging.info(f"Assistant: {response}")
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                logging.info("-" * 50)
        
        logging.info("\n🎉 Training completed successfully!")
        logging.info("="*80)
        
        # Final summary
        logging.info("Training Summary:")
        logging.info(f"  Configuration: {config_preset}")
        logging.info(f"  Experiment: {config.experiment_name}")
        logging.info(f"  Data: {config.train_data_path}")
        logging.info(f"  Checkpoints saved to: checkpoints/")
        logging.info(f"  Logs saved to: logs/")
        logging.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("\n❌ Training interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        logging.error(f"\n❌ Training failed: {e}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())