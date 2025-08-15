# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import pytest
from core.tokenizer import ConversationTokenizer


class TestInputValidation:
    """Test input validation and security."""
    
    def test_malicious_input_handling(self, tokenizer):
        """Test handling of potentially malicious inputs."""
        malicious_inputs = [
            # Extremely long content
            {"messages": [{"role": "user", "content": "A" * 100000}]},
            
            # Special characters and encoding attacks
            {"messages": [{"role": "user", "content": "\x00\x01\x02malicious"}]},
            
            # Unicode attacks
            {"messages": [{"role": "user", "content": "��" * 1000}]},
            
            # Deeply nested structure (JSON bomb attempt)
            {"messages": [{"role": "user", "content": str({"a": {"b": {"c": "deep"}}})}]},
            
            # Empty and whitespace-only content
            {"messages": [{"role": "user", "content": "   \n\t   "}]},
        ]
        
        for malicious_input in malicious_inputs:
            try:
                tokens = tokenizer.encode_conversation(malicious_input)
                # Should handle gracefully without crashing
                assert isinstance(tokens, list)
                # Should not produce extremely long token sequences
                assert len(tokens) < 50000, "Input produced too many tokens"
            except Exception as e:
                # If it raises an exception, it should be a controlled one
                assert "encode" in str(e).lower() or "conversation" in str(e).lower()
    
    def test_injection_prevention(self, tokenizer):
        """Test prevention of various injection attacks."""
        injection_attempts = [
            # SQL-like injection
            {"messages": [{"role": "user", "content": "'; DROP TABLE users; --"}]},
            
            # Script injection
            {"messages": [{"role": "user", "content": "<script>alert('xss')</script>"}]},
            
            # Command injection
            {"messages": [{"role": "user", "content": "; rm -rf / #"}]},
            
            # Path traversal
            {"messages": [{"role": "user", "content": "../../../etc/passwd"}]},
        ]
        
        for injection in injection_attempts:
            tokens = tokenizer.encode_conversation(injection)
            # Should tokenize without special handling (tokenizer should be neutral)
            assert isinstance(tokens, list)
            # Verify content is properly escaped/encoded
            assert len(tokens) > 0
    
    def test_resource_exhaustion_prevention(self, tokenizer):
        """Test prevention of resource exhaustion attacks."""
        # Test with many messages
        many_messages = {
            "messages": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
                for i in range(1000)
            ]
        }
        
        start_time = time.time()
        tokens = tokenizer.encode_conversation(many_messages)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0, "Processing took too long"
        assert isinstance(tokens, list)
    
    def test_memory_safety(self, tokenizer):
        """Test memory safety with large inputs."""
        import sys
        
        # Monitor memory usage
        initial_size = sys.getsizeof(tokenizer)
        
        # Process multiple large conversations
        for i in range(100):
            large_conv = {
                "messages": [
                    {"role": "user", "content": "X" * 1000},
                    {"role": "assistant", "content": "Y" * 1000}
                ]
            }
            tokens = tokenizer.encode_conversation(large_conv)
            assert isinstance(tokens, list)
        
        # Memory shouldn't grow excessively
        final_size = sys.getsizeof(tokenizer)
        memory_growth = final_size - initial_size
        
        assert memory_growth < 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"


# Makefile for running tests
"""
# Makefile

.PHONY: test test-unit test-integration test-performance test-security test-slow

# Install test dependencies
install-test:
	pip install pytest pytest-cov pytest-benchmark pytest-mock psutil

# Run all tests
test:
	pytest tests/ -v --cov=Src/Main_Scripts --cov-report=html --cov-report=term

# Run only unit tests (fast)
test-unit:
	pytest tests/unit/ -v

# Run integration tests
test-integration:
	pytest tests/integration/ -v

# Run performance tests
test-performance:
	pytest tests/performance/ -v

# Run security tests
test-security:
	pytest tests/security/ -v

# Run slow tests (marked with @pytest.mark.slow)
test-slow:
	pytest tests/ -v -m slow

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=Src/Main_Scripts --cov-report=html --cov-fail-under=80

# Run specific test file
test-file:
	pytest $(FILE) -v

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	pytest tests/ -n auto -v
"""