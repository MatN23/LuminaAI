import pytest
import torch
from unittest.mock import Mock


class TestTrainerInitialization:
    """Test trainer initialization."""
    
    def test_trainer_creation(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test trainer can be initialized."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device is not None
    
    def test_precision_manager(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test precision manager is initialized."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        assert trainer.precision_manager is not None
        assert trainer.precision_manager.train_precision == 'fp32'


class TestLossComputation:
    """Test loss computation."""
    
    def test_compute_loss(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test loss computation."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_weights = torch.ones(batch_size, seq_len)
        
        loss_dict = trainer.compute_loss(logits, labels, loss_weights)
        
        assert 'loss' in loss_dict
        assert 'raw_loss' in loss_dict
        assert 'perplexity' in loss_dict
        assert 'accuracy' in loss_dict
        assert not torch.isnan(loss_dict['loss'])
    
    def test_loss_with_padding(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test loss computation with padded tokens."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        batch_size = 2
        seq_len = 10
        vocab_size = mock_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 5:] = 0  # Pad last 5 tokens
        loss_weights = torch.ones(batch_size, seq_len)
        
        loss_dict = trainer.compute_loss(logits, labels, loss_weights)
        
        assert not torch.isnan(loss_dict['loss'])
        assert loss_dict['loss'].item() > 0


class TestTrainingStep:
    """Test training step."""
    
    def test_train_step(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test single training step."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
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


class TestAdaptiveFeatures:
    """Test adaptive training features."""
    
    def test_adjust_learning_rate(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test learning rate adjustment."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        old_lr = trainer.optimizer.param_groups[0]['lr']
        new_lr = old_lr * 0.5
        
        trainer.adjust_learning_rate(new_lr)
        
        assert trainer.optimizer.param_groups[0]['lr'] == new_lr
    
    def test_adjust_batch_size(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test batch size adjustment."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        old_batch_size = mock_config.batch_size
        new_batch_size = old_batch_size * 2
        
        trainer.adjust_batch_size(new_batch_size)
        
        assert trainer.config.batch_size == new_batch_size