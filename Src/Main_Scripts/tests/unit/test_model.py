# tests/unit/test_model.py
import pytest
import torch
from core.model import TransformerModel, estimate_parameters, RMSNorm, RotaryEmbedding


class TestModel:
    """Test model components."""
    
    def test_parameter_estimation(self, test_config):
        """Test parameter count estimation."""
        estimated = estimate_parameters(test_config)
        assert estimated > 0
        assert isinstance(estimated, int)
        
        # Create actual model and compare
        model = TransformerModel(test_config)
        actual = sum(p.numel() for p in model.parameters())
        
        # Estimation should be reasonably close (within 20%)
        assert abs(estimated - actual) / actual < 0.2
    
    def test_model_creation(self, test_config):
        """Test model initialization."""
        model = TransformerModel(test_config)
        
        assert model.config == test_config
        assert len(model.layers) == test_config.num_layers
        assert model.embed_tokens.num_embeddings == test_config.vocab_size
        
        # Check weight tying
        assert model.embed_tokens.weight is model.lm_head.weight
    
    def test_forward_pass(self, model, test_config):
        """Test model forward pass."""
        batch_size = 2
        seq_length = 10
        
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_length))
        
        # Test basic forward pass
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_length, test_config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_forward_with_attention_mask(self, model, test_config):
        """Test forward pass with attention mask."""
        batch_size = 2
        seq_length = 10
        
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens
        
        logits = model(input_ids, attention_mask=attention_mask)
        
        assert logits.shape == (batch_size, seq_length, test_config.vocab_size)
        assert not torch.isnan(logits).any()
    
    def test_hidden_states_output(self, model, test_config):
        """Test returning hidden states."""
        batch_size = 2
        seq_length = 10
        
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_length))
        
        logits, hidden_states = model(input_ids, return_hidden_states=True)
        
        assert len(hidden_states) == test_config.num_layers
        for hidden_state in hidden_states:
            assert hidden_state.shape == (batch_size, seq_length, test_config.hidden_size)
    
    def test_invalid_input_handling(self, model, test_config):
        """Test handling of invalid inputs."""
        batch_size = 2
        seq_length = 10
        
        # Input with tokens >= vocab_size (should be clamped)
        invalid_input = torch.full((batch_size, seq_length), test_config.vocab_size + 100)
        
        # Should not crash
        logits = model(invalid_input)
        assert logits.shape == (batch_size, seq_length, test_config.vocab_size)
        assert not torch.isnan(logits).any()


class TestRMSNorm:
    """Test RMSNorm implementation."""
    
    def test_rmsnorm_creation(self):
        """Test RMSNorm initialization."""
        norm = RMSNorm(64)
        assert norm.weight.shape == (64,)
        assert torch.allclose(norm.weight, torch.ones(64))
    
    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        
        # Check normalization property (approximately unit variance)
        variance = output.pow(2).mean(-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=1e-5)


class TestRotaryEmbedding:
    """Test RoPE implementation."""
    
    def test_rope_creation(self):
        """Test RoPE initialization."""
        rope = RotaryEmbedding(64, max_seq_len=1024)
        assert rope.dim == 64
        assert rope.max_seq_len == 1024
        assert rope._cached_seq_len == 1024
    
    def test_rope_forward(self):
        """Test RoPE forward pass."""
        rope = RotaryEmbedding(64)
        
        cos, sin = rope(10, torch.device('cpu'))
        
        assert cos.shape == (10, 64)
        assert sin.shape == (10, 64)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()
    
    def test_rope_cache_expansion(self):
        """Test RoPE cache expansion."""
        rope = RotaryEmbedding(64, max_seq_len=100)
        
        # Request longer sequence
        cos, sin = rope(200, torch.device('cpu'))
        
        assert cos.shape[0] >= 200
        assert rope._cached_seq_len >= 200


# tests/unit/test_dataset.py
import pytest
import torch
from core.dataset import ConversationDataset, create_dataloader


class TestDataset:
    """Test dataset functionality."""
    
    def test_dataset_creation(self, temp_data_file, tokenizer, test_config):
        """Test dataset initialization."""
        dataset = ConversationDataset(temp_data_file, tokenizer, test_config, "train")
        
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.stats['valid_conversations'] > 0
    
    def test_dataset_getitem(self, temp_data_file, tokenizer, test_config):
        """Test dataset item retrieval."""
        dataset = ConversationDataset(temp_data_file, tokenizer, test_config, "train")
        
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'labels' in item
        assert 'attention_mask' in item
        assert 'loss_weights' in item
        
        # Check tensor properties
        seq_len = test_config.seq_length - 1
        assert item['input_ids'].shape == (seq_len,)
        assert item['labels'].shape == (seq_len,)
        assert item['attention_mask'].shape == (seq_len,)
        assert item['loss_weights'].shape == (seq_len,)
    
    def test_dataset_statistics(self, temp_data_file, tokenizer, test_config):
        """Test dataset statistics computation."""
        dataset = ConversationDataset(temp_data_file, tokenizer, test_config, "train")
        
        stats = dataset.get_stats()
        
        assert 'total_loaded' in stats
        assert 'valid_conversations' in stats
        assert 'avg_token_length' in stats
        assert stats['total_loaded'] > 0
        assert stats['valid_conversations'] > 0
    
    def test_dataloader_creation(self, temp_data_file, tokenizer, test_config):
        """Test dataloader creation."""
        dataset = ConversationDataset(temp_data_file, tokenizer, test_config, "train")
        dataloader = create_dataloader(dataset, test_config, shuffle=True)
        
        assert dataloader.batch_size == test_config.batch_size
        assert dataloader.drop_last == True
        
        # Test iteration
        batch = next(iter(dataloader))
        assert isinstance(batch, dict)
        assert batch['input_ids'].shape[0] == test_config.batch_size