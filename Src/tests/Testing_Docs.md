# Complete Testing Guide 🧪

## Quick Start - Run All Tests

```bash
# 1. Install test dependencies
pip install pytest pytest-cov pytest-mock psutil pytest-benchmark pytest-xdist

# 2. Run everything
make test

# 3. View coverage report
open htmlcov/index.html
```

## 📁 Test Structure Setup

First, create the test directory structure:

```bash
mkdir -p tests/{unit,integration,performance,security}
mkdir -p tests/fixtures
mkdir -p tests/data

# Copy all test files from the test suite artifact
# tests/
# ├── conftest.py                    # Test configuration and fixtures
# ├── unit/
# │   ├── test_config.py            # Configuration tests
# │   ├── test_tokenizer.py         # Tokenizer tests
# │   ├── test_model.py             # Model architecture tests
# │   └── test_dataset.py           # Dataset tests
# ├── integration/
# │   └── test_training_pipeline.py # End-to-end training tests
# ├── performance/
# │   └── test_performance.py       # Speed and memory tests
# └── security/
#     └── test_input_validation.py  # Security validation tests
```

## 🚀 Running Different Test Categories

### 1. Unit Tests (Fast - 30 seconds)
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_model.py -v

# Run specific test method
pytest tests/unit/test_model.py::TestModel::test_forward_pass -v

# Run with coverage
pytest tests/unit/ -v --cov=Src/Main_Scripts --cov-report=html
```

**Example output:**
```
tests/unit/test_config.py::TestConfig::test_config_creation PASSED
tests/unit/test_config.py::TestConfig::test_config_validation PASSED
tests/unit/test_tokenizer.py::TestTokenizer::test_conversation_encoding PASSED
tests/unit/test_model.py::TestModel::test_forward_pass PASSED
================================ 15 passed in 12.34s ================================
```

### 2. Integration Tests (Medium - 2-5 minutes)
```bash
# Run integration tests
pytest tests/integration/ -v

# Run with specific config
pytest tests/integration/test_training_pipeline.py -v --tb=short
```

### 3. Performance Tests (Slow - 5-10 minutes)
```bash
# Run performance tests
pytest tests/performance/ -v

# Skip GPU tests if no GPU available
pytest tests/performance/ -v -m "not gpu"

# Run with benchmarking
pytest tests/performance/ -v --benchmark-only
```

### 4. Security Tests (Fast - 1-2 minutes)
```bash
# Run security tests
pytest tests/security/ -v

# Run with detailed output
pytest tests/security/test_input_validation.py -v -s
```

## 🎯 Focused Testing Scenarios

### Test Specific Component
```bash
# Test only the tokenizer
pytest tests/unit/test_tokenizer.py -v

# Test model with different configs
pytest tests/unit/test_model.py -k "test_parameter_estimation" -v

# Test security validation
pytest tests/security/ -k "malicious" -v
```

### Test with Different Python Versions
```bash
# Using tox (install with: pip install tox)
tox

# Or manually with different Python versions
python3.9 -m pytest tests/unit/
python3.10 -m pytest tests/unit/
python3.11 -m pytest tests/unit/
```

### Parallel Testing (Faster)
```bash
# Install parallel testing
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto -v

# Run with specific number of workers
pytest tests/ -n 4 -v
```

## 📊 Coverage Analysis

### Generate Coverage Reports
```bash
# HTML coverage report
pytest tests/ --cov=Src/Main_Scripts --cov-report=html

# Terminal coverage report
pytest tests/ --cov=Src/Main_Scripts --cov-report=term-missing

# XML coverage for CI/CD
pytest tests/ --cov=Src/Main_Scripts --cov-report=xml
```

### Coverage Thresholds
```bash
# Fail if coverage below 80%
pytest tests/ --cov=Src/Main_Scripts --cov-fail-under=80

# Show lines missing coverage
pytest tests/ --cov=Src/Main_Scripts --cov-report=term-missing
```

**Example coverage output:**
```
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
Src/Main_Scripts/config/config_manager.py    145      8    94%   234-241
Src/Main_Scripts/core/model.py               312     15    95%   445-450, 523-528
Src/Main_Scripts/core/tokenizer.py            89      3    97%   156-158
Src/Main_Scripts/security/auth.py            234     12    95%   345-352, 421-426
----------------------------------------------------------------
TOTAL                                        892     45    95%
```

## 🔍 Debugging Failed Tests

### Verbose Output
```bash
# Show print statements and detailed errors
pytest tests/unit/test_model.py -v -s

# Show local variables on failure
pytest tests/unit/test_model.py -v -l

# Drop into debugger on failure
pytest tests/unit/test_model.py --pdb
```

### Test Specific Conditions
```bash
# Only run tests marked as "slow"
pytest tests/ -m "slow" -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Run tests matching pattern
pytest tests/ -k "test_security or test_validation" -v
```

### GPU-Specific Testing
```bash
# Run only GPU tests (if available)
pytest tests/performance/test_performance.py::TestPerformance::test_gpu_memory_efficiency -v

# Skip GPU tests on CPU-only machines
pytest tests/ -m "not gpu" -v
```

## 🛠️ Setting Up Test Environment

### 1. Install Dependencies
```bash
# Create test requirements file
cat > test-requirements.txt << EOF
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-benchmark>=4.0.0
pytest-xdist>=3.3.0
pytest-timeout>=2.1.0
psutil>=5.9.0
memory-profiler>=0.61.0
EOF

# Install test dependencies
pip install -r test-requirements.txt
```

### 2. Configure pytest.ini
```bash
# Create pytest configuration
cat > pytest.ini << EOF
[tool:pytest]
minversion = 7.0
addopts = 
    -ra -q 
    --strict-markers 
    --strict-config
    --cov-report=term-missing
    --cov-report=html
    --timeout=300
testpaths = tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU
    integration: marks tests as integration tests
    security: marks tests as security tests
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
EOF
```

### 3. Test Data Setup
```bash
# Create test data directory
mkdir -p tests/data

# Create sample test data
python -c "
import json
from pathlib import Path

# Create sample conversations
test_conversations = []
for i in range(100):
    conv = {
        'conversation_id': f'test_{i:03d}',
        'messages': [
            {'role': 'user', 'content': f'Test message {i}'},
            {'role': 'assistant', 'content': f'Test response {i}'}
        ]
    }
    test_conversations.append(conv)

# Save test data
with open('tests/data/test_conversations.jsonl', 'w') as f:
    for conv in test_conversations:
        f.write(json.dumps(conv) + '\n')

print('Test data created')
"
```

## 🔧 Continuous Integration Testing

### GitHub Actions Workflow
```bash
# Test in CI/CD pipeline
.github/workflows/test.yml

# Manual trigger
gh workflow run "CI/CD Pipeline"

# Check workflow status
gh run list
```

### Local CI Simulation
```bash
# Simulate CI environment
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 bash -c "
pip install -r requirements.txt &&
pip install -r test-requirements.txt &&
pytest tests/ -v --cov=Src/Main_Scripts --cov-fail-under=80
"
```

## 🚨 Security Testing

### Run Security Scans
```bash
# Input validation security tests
pytest tests/security/test_input_validation.py -v

# Authentication security tests  
pytest tests/security/ -k "auth" -v

# Run with security focus
pytest tests/security/ -v --tb=short
```

### Additional Security Tools
```bash
# Install security tools
pip install bandit safety

# Run Bandit security scanner
bandit -r Src/Main_Scripts/ -f json -o security-report.json

# Check for known vulnerabilities
safety check -r requirements.txt

# Run together
bandit -r Src/Main_Scripts/ && safety check -r requirements.txt
```

## 📈 Performance Testing

### Benchmark Specific Functions
```bash
# Run benchmarks
pytest tests/performance/ --benchmark-only

# Save benchmark results
pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

# Performance regression tests
pytest tests/performance/test_performance.py::TestPerformance::test_inference_speed -v
```

### Memory Testing
```bash
# Monitor memory usage during tests
pytest tests/performance/ -v --tb=short

# Profile memory usage
python -m memory_profiler tests/performance/test_performance.py
```

## 🔄 Test-Driven Development Workflow

### 1. Write Failing Test First
```bash
# Create new test
echo "def test_new_feature():
    assert False  # TODO: implement feature
" >> tests/unit/test_new_feature.py

# Run to see it fail
pytest tests/unit/test_new_feature.py -v
```

### 2. Implement Feature
```bash
# Write minimal code to pass test
# Then run tests again
pytest tests/unit/test_new_feature.py -v
```

### 3. Refactor and Re-test
```bash
# Run full test suite after changes
pytest tests/ -v --cov=Src/Main_Scripts
```

## 📝 Test Output Examples

### Successful Test Run
```
========================= test session starts =========================
platform linux -- Python 3.10.0, pytest-7.4.0, pluggy-1.0.0
cachedir: .pytest_cache
rootdir: /app, configfile: pytest.ini
plugins: cov-4.1.0, mock-3.11.0, benchmark-4.0.0
collected 45 items

tests/unit/test_config.py::TestConfig::test_config_creation PASSED    [  2%]
tests/unit/test_config.py::TestConfig::test_config_validation PASSED  [  4%]
tests/unit/test_tokenizer.py::TestTokenizer::test_tokenizer_initialization PASSED [  7%]
tests/unit/test_model.py::TestModel::test_forward_pass PASSED         [ 91%]
tests/security/test_input_validation.py::TestInputValidation::test_malicious_input_handling PASSED [100%]

========================= 45 passed in 23.45s =========================

---------- coverage: platform linux, python 3.10.0-final-0 -----------
Name                                        Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
Src/Main_Scripts/config/config_manager.py    145      8    94%   234-241
Src/Main_Scripts/core/model.py               312     15    95%   445-450
---------------------------------------------------------------------------
TOTAL                                         892     45    95%

Required test coverage of 80% reached. Total coverage: 95.05%
```

### Failed Test Example
```
========================= test session starts =========================

tests/unit/test_model.py::TestModel::test_forward_pass FAILED         [ 50%]

================================= FAILURES =================================
_______________________ TestModel.test_forward_pass _______________________

    def test_forward_pass(self, model, test_config):
        input_ids = torch.randint(0, test_config.vocab_size, (2, 10))
        logits = model(input_ids)
>       assert logits.shape == (2, 10, test_config.vocab_size)
E       AssertionError: assert torch.Size([2, 10, 1024]) == (2, 10, 512)

tests/unit/test_model.py:89: AssertionError
========================= short test summary info =========================
FAILED tests/unit/test_model.py::TestModel::test_forward_pass - AssertionError: assert torch.Size([2, 10, 1024]) == (2, 10, 512)
========================= 1 failed, 44 passed in 18.32s =========================
```

## 🎬 Complete Testing Commands Reference

```bash
# Quick test commands
make test                    # Run all tests
make test-unit              # Unit tests only
make test-integration       # Integration tests only
make test-security          # Security tests only
make test-performance       # Performance tests only
make test-coverage          # With coverage report

# Advanced testing
pytest tests/ -v --cov=Src/Main_Scripts --cov-fail-under=80 --cov-report=html
pytest tests/ -n auto -v    # Parallel execution
pytest tests/ -k "not slow" # Skip slow tests
pytest tests/ --pdb         # Debug on failure
pytest tests/ -x             # Stop on first failure
pytest tests/ --lf           # Run last failed tests only
```

This comprehensive testing setup ensures your production system is thoroughly validated before deployment! 🚀