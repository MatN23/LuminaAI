# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.
import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Src', 'Main_Scripts'))

from config.config_manager import Config, ConfigPresets
from core.tokenizer import ConversationTokenizer
from core.model import TransformerModel
from core.dataset import ConversationDataset


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with minimal resources."""
    return Config(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        seq_length=128,
        intermediate_size=128,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        experiment_name="test_experiment"
    )


@pytest.fixture(scope="session")
def tokenizer():
    """Test tokenizer instance."""
    return ConversationTokenizer()


@pytest.fixture(scope="session")
def model(test_config):
    """Test model instance."""
    return TransformerModel(test_config)


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return {
        "conversation_id": "test_001",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
        ]
    }


@pytest.fixture
def temp_data_file(sample_conversation):
    """Create temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write multiple conversations
        for i in range(10):
            conv = sample_conversation.copy()
            conv["conversation_id"] = f"test_{i:03d}"
            f.write(json.dumps(conv) + '\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
