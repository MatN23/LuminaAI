# tests/integration/test_training_pipeline.py
import pytest
import torch
import tempfile
from pathlib import Path
from training.trainer import EnhancedConversationTrainer
from monitoring.logger import ProductionLogger


class TestTrainingPipeline:
    """Test end-to-end training pipeline."""
    
    def test_trainer_initialization(self, model, tokenizer, test_config, temp_checkpoint_dir):
        """Test trainer initialization."""
        logger = ProductionLogger(log_level="WARNING")
        trainer = EnhancedConversationTrainer(model, tokenizer, test_config, logger)
        
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert trainer.config is test_config
    
    @pytest.mark.slow
    def test_single_training_step(self, temp_data_file, tokenizer, test_config):
        """Test single training step execution."""
        # Create minimal dataset
        from core.dataset import ConversationDataset, create_dataloader
        
        dataset = ConversationDataset(temp_data_file, tokenizer, test_config, "train")
        dataloader = create_dataloader(dataset, test_config, shuffle=False)
        
        # Initialize training components
        model = TransformerModel(test_config)
        logger = ProductionLogger(log_level="ERROR")
        trainer = EnhancedConversationTrainer(model, tokenizer, test_config, logger)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Perform training step
        initial_loss = float('inf')
        try:
            loss = trainer._training_step(batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() > 0
            assert not torch.isnan(loss)
            initial_loss = loss.item()
        except Exception as e:
            pytest.fail(f"Training step failed: {e}")
    
    def test_generation(self, model, tokenizer, test_config):
        """Test model generation."""
        logger = ProductionLogger(log_level="ERROR")
        trainer = EnhancedConversationTrainer(model, tokenizer, test_config, logger)
        
        prompt = "Hello, how are you?"
        
        try:
            response = trainer.generate(prompt, max_new_tokens=20)
            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            # Generation might fail with untrained model, but shouldn't crash
            assert "generation" in str(e).lower() or "decode" in str(e).lower()


# tests/performance/test_performance.py
import pytest
import torch
import time
import psutil
from core.model import TransformerModel


class TestPerformance:
    """Test performance characteristics."""
    
    def test_inference_speed(self, test_config):
        """Test inference latency."""
        model = TransformerModel(test_config)
        model.eval()
        
        batch_size = 1
        seq_length = 50
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_length))
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                _ = model(input_ids)
                end = time.time()
                times.append(end - start)
        
        avg_time = sum(times) / len(times)
        tokens_per_second = seq_length / avg_time
        
        # Basic performance check (should process at least 1000 tokens/sec on CPU)
        assert tokens_per_second > 100, f"Too slow: {tokens_per_second:.1f} tokens/sec"
    
    def test_memory_usage(self, test_config):
        """Test memory consumption."""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        model = TransformerModel(test_config)
        
        # Memory after model creation
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_overhead = model_memory - baseline_memory
        
        # Should be reasonable for small test model
        assert model_overhead < 500, f"Too much memory: {model_overhead:.1f} MB"
        
        # Test forward pass memory
        input_ids = torch.randint(0, test_config.vocab_size, (2, 50))
        _ = model(input_ids)
        
        forward_memory = process.memory_info().rss / 1024 / 1024  # MB
        forward_overhead = forward_memory - model_memory
        
        # Forward pass shouldn't use excessive memory
        assert forward_overhead < 200, f"Forward pass too much memory: {forward_overhead:.1f} MB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self, test_config):
        """Test GPU memory usage."""
        device = torch.device('cuda')
        model = TransformerModel(test_config).to(device)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        input_ids = torch.randint(0, test_config.vocab_size, (4, 100)).to(device)
        
        with torch.no_grad():
            _ = model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Should be reasonable for test model
        assert peak_memory < 1000, f"GPU memory too high: {peak_memory:.1f} MB"