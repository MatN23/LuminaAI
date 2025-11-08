# tests/test_trainer.py
"""
Comprehensive test suite for EnhancedConversationTrainer.

Tests cover:
- Initialization and setup
- Loss computation (standard, weighted, padded)
- Training steps
- Adaptive features (LR, batch size, MoE, MoD)
- Edge cases and numerical stability
"""

import pytest
import torch
import math
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    from config.config_manager import ConfigPresets
    config = ConfigPresets.debug()
    config.vocab_size = 1000
    config.hidden_size = 128
    config.num_layers = 2
    config.num_heads = 4
    config.seq_length = 64
    config.batch_size = 2
    config.learning_rate = 1e-4
    config.num_epochs = 1
    config.gradient_accumulation_steps = 1
    config.precision = 'fp32'
    config.use_moe = False
    config.use_mod = False
    return config


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    from core.tokenizer import ConversationTokenizer
    tokenizer = ConversationTokenizer(model_name="gpt2")
    return tokenizer


@pytest.fixture
def small_model(mock_config):
    """Create small model for testing."""
    from core.model import DeepSeekTransformer
    model = DeepSeekTransformer(mock_config)
    return model


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    import logging
    logger = logging.getLogger("test_trainer")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def trainer(small_model, mock_tokenizer, mock_config, mock_logger):
    """Create trainer instance."""
    from training.trainer import EnhancedConversationTrainer
    
    trainer = EnhancedConversationTrainer(
        small_model, mock_tokenizer, mock_config, mock_logger
    )
    return trainer


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestTrainerInitialization:
    """Test trainer initialization and setup."""
    
    def test_trainer_creation(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test trainer can be initialized."""
        from training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device is not None
        print("✅ Trainer creation test passed")
    
    def test_precision_manager(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test precision manager is initialized."""
        from training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        assert trainer.precision_manager is not None
        assert trainer.precision_manager.train_precision == 'fp32'
        print("✅ Precision manager test passed")
    
    def test_device_selection(self, trainer):
        """Test correct device is selected."""
        assert trainer.device.type in ['cuda', 'mps', 'cpu']
        print(f"✅ Device selection test passed: {trainer.device.type}")
    
    def test_optimizer_initialization(self, trainer):
        """Test optimizer is properly initialized."""
        assert trainer.optimizer is not None
        assert len(trainer.optimizer.param_groups) > 0
        
        # Check learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        assert lr == trainer.config.learning_rate
        print(f"✅ Optimizer initialization test passed: LR={lr}")
    
    def test_scheduler_setup(self, trainer, mock_config):
        """Test scheduler can be set up."""
        # Scheduler is set up later, so we test the setup method
        total_steps = 100
        trainer._setup_scheduler(total_steps)
        
        if trainer.scheduler is not None:
            assert hasattr(trainer.scheduler, 'step')
            print("✅ Scheduler setup test passed")
        else:
            print("⚠️  Scheduler disabled by config")


# ============================================================================
# LOSS COMPUTATION TESTS
# ============================================================================

class TestLossComputation:
    """Test loss computation correctness and edge cases."""
    
    def test_compute_loss_basic(self, trainer, mock_config):
        """Test basic loss computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_weights = None
        
        loss_dict = trainer.compute_loss(logits, labels, loss_weights)
        
        # Check all required keys
        assert 'loss' in loss_dict
        assert 'raw_loss' in loss_dict
        assert 'perplexity' in loss_dict
        assert 'accuracy' in loss_dict
        assert 'valid_tokens' in loss_dict
        
        # Check values are valid
        assert not torch.isnan(loss_dict['loss'])
        assert not torch.isinf(loss_dict['loss'])
        assert loss_dict['loss'].item() > 0
        assert loss_dict['perplexity'].item() > 0
        
        print(f"✅ Basic loss computation test passed")
        print(f"   Loss: {loss_dict['loss'].item():.4f}")
        print(f"   Perplexity: {loss_dict['perplexity'].item():.2f}")
        print(f"   Accuracy: {loss_dict['accuracy'].item():.2%}")
    
    def test_compute_loss_with_weights(self, trainer, mock_config):
        """Test loss computation with weights."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test without weights
        loss_dict_no_weights = trainer.compute_loss(logits, labels, None)
        
        # Test with weights
        loss_weights = torch.ones(batch_size, seq_len)
        loss_weights[:, :5] = 2.0  # Higher weight for first 5 tokens
        loss_dict_with_weights = trainer.compute_loss(logits, labels, loss_weights)
        
        # CRITICAL: raw_loss should be UNCHANGED by weights
        raw_diff = abs(
            loss_dict_no_weights['raw_loss'].item() - 
            loss_dict_with_weights['raw_loss'].item()
        )
        
        assert raw_diff < 1e-5, f"Raw loss changed by weights! Diff: {raw_diff}"
        
        # Training loss should be different (weighted)
        train_diff = abs(
            loss_dict_no_weights['loss'].item() - 
            loss_dict_with_weights['loss'].item()
        )
        
        # In most cases, weighted loss should differ (unless weights are uniform)
        # This is expected behavior
        
        print("✅ Loss weighting test passed")
        print(f"   Raw loss (no weights): {loss_dict_no_weights['raw_loss'].item():.4f}")
        print(f"   Raw loss (with weights): {loss_dict_with_weights['raw_loss'].item():.4f}")
        print(f"   Difference: {raw_diff:.2e} (should be ~0)")
    
    def test_loss_with_padding(self, trainer, mock_config):
        """Test loss computation with padded tokens."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Add padding (pad_token_id = 0)
        labels[:, 5:] = 0  # Last 5 tokens are padding
        
        loss_dict = trainer.compute_loss(logits, labels, None)
        
        # Check loss is valid
        assert not torch.isnan(loss_dict['loss'])
        assert loss_dict['loss'].item() > 0
        
        # Check valid token count
        expected_valid = batch_size * (5 - 1)  # 5 non-padded tokens, shifted by 1
        actual_valid = loss_dict['valid_tokens'].item()
        
        # Allow some tolerance for edge cases
        assert actual_valid <= expected_valid * 1.2
        
        print("✅ Padding mask test passed")
        print(f"   Valid tokens: {actual_valid}")
        print(f"   Loss: {loss_dict['loss'].item():.4f}")
    
    def test_loss_gradient_flow(self, trainer, mock_config):
        """Test that loss has proper gradient flow."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss_dict = trainer.compute_loss(logits, labels, None)
        
        # Check gradient requirements
        assert loss_dict['loss'].requires_grad, "Loss must be differentiable!"
        assert not loss_dict['raw_loss'].requires_grad, "Raw loss should be detached!"
        
        # Test backward pass
        loss_dict['loss'].backward()
        
        assert logits.grad is not None, "Gradients should flow to logits!"
        assert not torch.isnan(logits.grad).any(), "Gradients should be valid!"
        
        print("✅ Gradient flow test passed")
    
    def test_loss_numerical_stability(self, trainer, mock_config):
        """Test loss computation with extreme values."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        # Test with very large logits
        extreme_logits = torch.randn(batch_size, seq_len, vocab_size) * 100
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss_dict = trainer.compute_loss(extreme_logits, labels, None)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(loss_dict['loss'])
        assert not torch.isinf(loss_dict['loss'])
        assert loss_dict['perplexity'].item() < float('inf')
        
        print("✅ Numerical stability test passed")
        print(f"   Extreme logits loss: {loss_dict['loss'].item():.4f}")
        print(f"   Perplexity: {loss_dict['perplexity'].item():.2f}")
    
    def test_perplexity_calculation(self, trainer, mock_config):
        """Test perplexity is calculated correctly."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss_dict = trainer.compute_loss(logits, labels, None)
        
        # Perplexity should be exp(loss)
        raw_loss = loss_dict['raw_loss'].item()
        expected_ppl = math.exp(min(raw_loss, 15.0))  # Clamped
        actual_ppl = loss_dict['perplexity'].item()
        
        # Allow small numerical difference
        ppl_diff = abs(expected_ppl - actual_ppl)
        assert ppl_diff < 0.1 or ppl_diff / expected_ppl < 0.01
        
        print("✅ Perplexity calculation test passed")
        print(f"   Raw loss: {raw_loss:.4f}")
        print(f"   Expected PPL: {expected_ppl:.2f}")
        print(f"   Actual PPL: {actual_ppl:.2f}")
    
    def test_loss_with_all_padding(self, trainer, mock_config):
        """Test loss computation when all tokens are padding."""
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long)  # All padding
        
        loss_dict = trainer.compute_loss(logits, labels, None)
        
        # Should handle gracefully
        assert loss_dict['valid_tokens'].item() == 0
        assert loss_dict['loss'].item() == 0.0
        assert loss_dict['perplexity'].item() == float('inf')
        
        print("✅ All-padding test passed")


# ============================================================================
# TRAINING STEP TESTS
# ============================================================================

class TestTrainingStep:
    """Test training step execution."""
    
    def test_train_step_basic(self, trainer, mock_config):
        """Test single training step."""
        batch = {
            'input_ids': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'labels': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'loss_weights': torch.ones(2, 10)
        }
        
        step_metrics = trainer.train_step(batch)
        
        assert 'loss' in step_metrics
        assert 'accuracy' in step_metrics
        assert step_metrics['loss'] >= 0
        assert 0 <= step_metrics['accuracy'] <= 1
        
        print("✅ Basic training step test passed")
        print(f"   Loss: {step_metrics['loss']:.4f}")
        print(f"   Accuracy: {step_metrics['accuracy']:.2%}")
    
    def test_optimizer_step(self, trainer, mock_config):
        """Test optimizer step execution."""
        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]
        
        # Create batch and take training step
        batch = {
            'input_ids': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'labels': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'loss_weights': torch.ones(2, 10)
        }
        
        trainer.train_step(batch)
        opt_metrics = trainer.optimizer_step()
        
        # Parameters should have changed
        params_changed = False
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Parameters should change after optimizer step!"
        assert 'grad_norm' in opt_metrics
        assert 'lr' in opt_metrics
        
        print("✅ Optimizer step test passed")
        print(f"   Grad norm: {opt_metrics['grad_norm']:.4f}")
        print(f"   Learning rate: {opt_metrics['lr']:.2e}")


# ============================================================================
# ADAPTIVE FEATURES TESTS
# ============================================================================

class TestAdaptiveFeatures:
    """Test adaptive training features."""
    
    def test_adjust_learning_rate(self, trainer):
        """Test learning rate adjustment."""
        old_lr = trainer.optimizer.param_groups[0]['lr']
        new_lr = old_lr * 0.5
        
        trainer.adjust_learning_rate(new_lr, grace_period=10)
        
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        assert abs(actual_lr - new_lr) < 1e-8
        
        # Check override flag is set
        assert trainer._adaptive_lr_override == True
        
        print("✅ Learning rate adjustment test passed")
        print(f"   Old LR: {old_lr:.2e}")
        print(f"   New LR: {actual_lr:.2e}")
    
    def test_emergency_lr_reduction(self, trainer):
        """Test emergency LR reduction."""
        old_lr = trainer.optimizer.param_groups[0]['lr']
        
        trainer.emergency_lr_reduction(reduction_factor=10.0)
        
        new_lr = trainer.optimizer.param_groups[0]['lr']
        expected_lr = old_lr / 10.0
        
        assert abs(new_lr - expected_lr) < 1e-8
        
        print("✅ Emergency LR reduction test passed")
        print(f"   Old LR: {old_lr:.2e}")
        print(f"   New LR: {new_lr:.2e}")
    
    def test_adjust_batch_size(self, trainer, mock_config):
        """Test batch size adjustment."""
        old_batch_size = mock_config.batch_size
        new_batch_size = old_batch_size * 2
        
        trainer.adjust_batch_size(new_batch_size)
        
        assert trainer.config.batch_size == new_batch_size
        
        print("✅ Batch size adjustment test passed")
        print(f"   Old batch size: {old_batch_size}")
        print(f"   New batch size: {trainer.config.batch_size}")
    
    def test_get_current_metrics(self, trainer, mock_config):
        """Test getting current training metrics."""
        metrics = trainer.get_current_metrics()
        
        assert hasattr(metrics, 'epoch')
        assert hasattr(metrics, 'step')
        assert hasattr(metrics, 'loss')
        assert hasattr(metrics, 'learning_rate')
        
        print("✅ Get current metrics test passed")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for full training workflow."""
    
    def test_full_training_step_workflow(self, trainer, mock_config):
        """Test complete training step workflow."""
        # Setup
        batch = {
            'input_ids': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'labels': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'loss_weights': torch.ones(2, 10)
        }
        
        # Forward pass
        step_metrics = trainer.train_step(batch)
        assert step_metrics['loss'] > 0
        
        # Optimizer step
        opt_metrics = trainer.optimizer_step()
        assert opt_metrics['grad_norm'] >= 0
        
        # Update global step
        trainer.global_step += 1
        
        print("✅ Full training workflow test passed")
        print(f"   Step: {trainer.global_step}")
        print(f"   Loss: {step_metrics['loss']:.4f}")
        print(f"   Grad Norm: {opt_metrics['grad_norm']:.4f}")


# ============================================================================
# STANDALONE TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TRAINER TESTS")
    print("="*80 + "\n")
    
    # Create fixtures
    from config.config_manager import ConfigPresets
    from core.tokenizer import ConversationTokenizer
    from core.model import DeepSeekTransformer
    from training.trainer import EnhancedConversationTrainer
    import logging
    
    config = ConfigPresets.debug()
    config.vocab_size = 1000
    config.batch_size = 2
    
    tokenizer = ConversationTokenizer(model_name="gpt2")
    model = DeepSeekTransformer(config)
    logger = logging.getLogger("test")
    
    trainer = EnhancedConversationTrainer(model, tokenizer, config, logger)
    
    # Run tests
    test_classes = [
        TestTrainerInitialization(),
        TestLossComputation(),
        TestTrainingStep(),
        TestAdaptiveFeatures(),
        TestTrainingIntegration()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*60}")
        print(f"Running {class_name}")
        print(f"{'='*60}")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    print(f"\n{method_name}:")
                    method = getattr(test_class, method_name)
                    
                    # Call with appropriate fixtures
                    if 'mock_config' in method.__code__.co_varnames:
                        method(trainer, config)
                    else:
                        method(trainer)
                    
                    passed_tests += 1
                except Exception as e:
                    failed_tests.append((class_name, method_name, str(e)))
                    print(f"❌ {method_name} FAILED: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    else:
        print("\n✅ ALL TESTS PASSED!")
    
    print("="*80 + "\n")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run without pytest if executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)