import pytest
import torch
from pathlib import Path


class TestEndToEnd:
    """End-to-end training tests."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires full setup")
    def test_minimal_training_run(
        self, temp_dir, sample_conversation_data,
        mock_tokenizer, mock_config, mock_logger
    ):
        """Test minimal training run end-to-end."""
        try:
            from Main_Scripts.core.model import DeepSeekTransformer, DeepSeekConfig
            from Main_Scripts.training.trainer import EnhancedConversationTrainer
            from Main_Scripts.core.dataset import FastConversationDataset, create_dataloader
            
            # Create model
            model_config = DeepSeekConfig(
                vocab_size=mock_config.vocab_size,
                hidden_size=mock_config.hidden_size,
                num_layers=mock_config.num_layers,
                num_heads=mock_config.num_heads,
                num_kv_heads=mock_config.num_kv_heads,
                seq_length=mock_config.seq_length
            )
            model = DeepSeekTransformer(model_config)
            
            # Create dataset
            dataset = FastConversationDataset(
                sample_conversation_data,
                mock_tokenizer,
                mock_config
            )
            
            # Create trainer
            mock_config.num_epochs = 1
            trainer = EnhancedConversationTrainer(
                model, mock_tokenizer, mock_config, mock_logger
            )
            
            # Train
            trainer.train(dataset)
            
            # Should complete without errors
            assert True
            
        except ImportError:
            pytest.skip("Required modules not available")