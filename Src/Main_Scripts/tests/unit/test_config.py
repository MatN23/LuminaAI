import pytest
import tempfile
from pathlib import Path
from config.config_manager import Config, ConfigPresets


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = Config()
        assert config.hidden_size == 512
        assert config.num_layers == 8
        assert config.vocab_size > 0
    
    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = Config(hidden_size=64, num_heads=4)
        config.validate()  # Should not raise
        
        # Invalid config - hidden_size not divisible by num_heads
        with pytest.raises(AssertionError):
            config = Config(hidden_size=63, num_heads=4)
            config.validate()
        
        # Invalid precision
        with pytest.raises(AssertionError):
            config = Config(precision="fp64")
            config.validate()
    
    def test_config_serialization(self):
        """Test config save/load."""
        config = Config(experiment_name="test_save")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            config.save(temp_path)
            assert Path(temp_path).exists()
            
            # Load config
            loaded_config = Config.load(temp_path)
            assert loaded_config.experiment_name == "test_save"
            assert loaded_config.hidden_size == config.hidden_size
        
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def test_presets(self):
        """Test configuration presets."""
        debug_config = ConfigPresets.debug()
        assert debug_config.experiment_name == "debug_run"
        assert debug_config.num_epochs == 100
        
        large_config = ConfigPresets.large()
        assert large_config.hidden_size == 2048
        assert large_config.num_layers == 24
    
    def test_post_init_processing(self):
        """Test post-initialization processing."""
        config = Config(vocab_size=1000)  # Not multiple of 64
        # Should be adjusted to nearest multiple of 64
        assert config.vocab_size % 64 == 0
        assert config.vocab_size >= 1000


# tests/unit/test_tokenizer.py
import pytest
from core.tokenizer import ConversationTokenizer


class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer creates successfully."""
        tokenizer = ConversationTokenizer()
        assert tokenizer.vocab_size > 50000
        assert tokenizer.base_vocab_size > 0
        assert len(tokenizer.special_tokens) > 0
    
    def test_conversation_encoding(self, tokenizer, sample_conversation):
        """Test conversation encoding."""
        tokens = tokenizer.encode_conversation(sample_conversation)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)
        
        # Should contain special tokens
        assert tokenizer.special_tokens["<|im_start|>"] in tokens
        assert tokenizer.special_tokens["<|im_end|>"] in tokens
    
    def test_empty_conversation(self, tokenizer):
        """Test handling of empty conversations."""
        empty_conv = {"messages": []}
        tokens = tokenizer.encode_conversation(empty_conv)
        assert tokens == []
        
        # Missing messages field
        invalid_conv = {}
        tokens = tokenizer.encode_conversation(invalid_conv)
        assert tokens == []
    
    def test_malformed_messages(self, tokenizer):
        """Test handling of malformed messages."""
        malformed = {
            "messages": [
                {"role": "user", "content": ""},  # Empty content
                {"role": "invalid_role", "content": "Hello"},  # Invalid role
                {"content": "Missing role"},  # Missing role
            ]
        }
        tokens = tokenizer.encode_conversation(malformed)
        # Should handle gracefully without crashing
        assert isinstance(tokens, list)
    
    def test_decode_functionality(self, tokenizer):
        """Test token decoding."""
        # Test with valid base vocabulary tokens
        test_tokens = [1, 2, 3, 4, 5]
        decoded = tokenizer.decode(test_tokens)
        assert isinstance(decoded, str)
        
        # Test with special tokens
        special_tokens = [tokenizer.special_tokens["<|im_start|>"]]
        decoded = tokenizer.decode(special_tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)
    
    def test_role_token_mapping(self, tokenizer):
        """Test role token retrieval."""
        user_token = tokenizer.get_role_token('user')
        assistant_token = tokenizer.get_role_token('assistant')
        system_token = tokenizer.get_role_token('system')
        
        assert user_token == tokenizer.special_tokens["<|user|>"]
        assert assistant_token == tokenizer.special_tokens["<|assistant|>"]
        assert system_token == tokenizer.special_tokens["<|system|>"]
        
        # Test fallback for unknown role
        unknown_token = tokenizer.get_role_token('unknown')
        assert unknown_token == tokenizer.special_tokens["<|user|>"]