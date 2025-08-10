# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn.functional as F
import logging
import re
import gc
import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import math
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation with sampling parameters."""
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_greedy: bool = False
    repetition_penalty: float = 1.1
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    min_length: int = 1
    do_sample: bool = True

# ULTRA-AGGRESSIVE memory management (from Train.py)
@contextmanager
def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup."""
    try:
        yield
    finally:
        # Force Python garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def setup_device():
    """Setup device with ultra-conservative memory management (from Train.py)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        
        # Much more conservative memory fraction
        torch.cuda.set_per_process_memory_fraction(0.70)  # Only use 70% of GPU memory
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    return device

device = setup_device()

def ensure_tensor_device(tensor, target_device):
    """Ensure tensor is on the target device."""
    if tensor.device != target_device:
        return tensor.to(target_device)
    return tensor

def validate_tokenizer(tokenizer):
    """Validate that the tokenizer is working correctly."""
    test_text = "Hello, how are you?"
    try:
        # Test required methods exist
        required_methods = ['encode', 'decode', 'vocab_size']
        for method in required_methods:
            if not hasattr(tokenizer, method):
                print(f"❌ Missing required method: {method}")
                return False
        
        # Test required attributes
        required_attrs = ['vocab', 'id_to_token']
        for attr in required_attrs:
            if not hasattr(tokenizer, attr):
                print(f"❌ Missing required attribute: {attr}")
                return False
        
        # Test tokenization methods
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"🔍 Tokenizer test:")
        print(f"   Original: {test_text}")
        print(f"   Encoded: {encoded[:10]}..." if len(encoded) > 10 else f"   Encoded: {encoded}")
        print(f"   Decoded: {decoded}")
        print(f"   Vocab size: {tokenizer.vocab_size()}")
        
        return len(encoded) > 0 and decoded.strip()
    except Exception as e:
        print(f"❌ Tokenizer validation failed: {e}")
        return False

def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1) -> torch.Tensor:
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits
    
    score = torch.gather(logits, 1, input_ids)
    # If score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    if temperature <= 0.0:
        temperature = 1.0
    return logits / temperature

def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits to only keep top k tokens."""
    if top_k <= 0:
        return logits
    
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float('inf')
    return logits

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling."""
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    return logits

def sample_next_token(logits: torch.Tensor, config: GenerationConfig) -> int:
    """Sample next token using the specified sampling strategy."""
    if config.use_greedy:
        return torch.argmax(logits, dim=-1).item()
    
    # Apply temperature
    logits = apply_temperature(logits, config.temperature)
    
    # Apply top-k filtering
    if config.top_k > 0:
        logits = top_k_filtering(logits, config.top_k)
    
    # Apply top-p filtering
    if config.top_p < 1.0:
        logits = top_p_filtering(logits, config.top_p)
    
    # Sample from the filtered distribution
    if config.do_sample:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    else:
        next_token = torch.argmax(logits, dim=-1).item()
    
    return next_token

# =====================================================================
# AGENT TOOLS AND CAPABILITIES
# =====================================================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class AgentTool(ABC):
    """Abstract base class for agent tools."""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class FileSystemTool(AgentTool):
    """Tool for file system operations."""
    
    def name(self) -> str:
        return "filesystem"
    
    def description(self) -> str:
        return "Read, write, list files and directories"
    
    def execute(self, action: str, path: str = "", content: str = "") -> ToolResult:
        try:
            path_obj = Path(path) if path else Path.cwd()
            
            if action == "read":
                if not path_obj.exists():
                    return ToolResult(False, None, f"File not found: {path}")
                content = path_obj.read_text(encoding='utf-8')
                return ToolResult(True, content, metadata={"size": len(content)})
            
            elif action == "write":
                path_obj.write_text(content, encoding='utf-8')
                return ToolResult(True, f"Written {len(content)} characters to {path}")
            
            elif action == "list":
                if path_obj.is_file():
                    return ToolResult(True, [str(path_obj)])
                items = [str(item) for item in path_obj.iterdir()]
                return ToolResult(True, items)
            
            elif action == "exists":
                return ToolResult(True, path_obj.exists())
            
            else:
                return ToolResult(False, None, f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(False, None, str(e))

class CalculatorTool(AgentTool):
    """Tool for mathematical calculations."""
    
    def name(self) -> str:
        return "calculator"
    
    def description(self) -> str:
        return "Perform mathematical calculations"
    
    def execute(self, expression: str) -> ToolResult:
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/().() ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return ToolResult(False, None, "Expression contains invalid characters")
            
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, None, f"Calculation error: {e}")

class TaskMemoryTool(AgentTool):
    """Tool for storing and retrieving task-related information."""
    
    def __init__(self):
        self.memory = {}
        self.task_history = []
    
    def name(self) -> str:
        return "memory"
    
    def description(self) -> str:
        return "Store and retrieve information across tasks"
    
    def execute(self, action: str, key: str = "", value: Any = None) -> ToolResult:
        try:
            if action == "store":
                self.memory[key] = value
                return ToolResult(True, f"Stored {key}")
            
            elif action == "retrieve":
                if key in self.memory:
                    return ToolResult(True, self.memory[key])
                return ToolResult(False, None, f"Key not found: {key}")
            
            elif action == "list":
                return ToolResult(True, list(self.memory.keys()))
            
            elif action == "clear":
                self.memory.clear()
                return ToolResult(True, "Memory cleared")
            
            elif action == "log_task":
                task_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "task": value,
                    "status": key  # using key as status
                }
                self.task_history.append(task_entry)
                return ToolResult(True, "Task logged")
            
            elif action == "get_history":
                return ToolResult(True, self.task_history[-10:])  # Last 10 tasks
            
            else:
                return ToolResult(False, None, f"Unknown memory action: {action}")
                
        except Exception as e:
            return ToolResult(False, None, str(e))

class WebSearchTool(AgentTool):
    """Tool for web search (simulated for MVP)."""
    
    def name(self) -> str:
        return "web_search"
    
    def description(self) -> str:
        return "Search the web for information"
    
    def execute(self, query: str) -> ToolResult:
        # Simulated web search for MVP
        try:
            # In a real implementation, you'd use an actual search API
            simulated_results = [
                f"Search result 1 for '{query}': Lorem ipsum information about {query}",
                f"Search result 2 for '{query}': Additional details regarding {query}",
                f"Search result 3 for '{query}': More context about {query}"
            ]
            return ToolResult(True, simulated_results, metadata={"query": query, "count": len(simulated_results)})
        except Exception as e:
            return ToolResult(False, None, str(e))

# =====================================================================
# AGENTIC REASONING AND PLANNING
# =====================================================================

@dataclass
class Task:
    """Represents a task to be executed by the agent."""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    steps: List[str] = None
    results: List[Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.results is None:
            self.results = []
        if self.created_at is None:
            self.created_at = datetime.now()

class TaskPlanner:
    """Plans and decomposes tasks into actionable steps."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure model is on correct device
        self.model = self.model.to(device)
        
        # Default generation config for planning (more conservative)
        self.generation_config = GenerationConfig(
            max_tokens=30,  # Reduced for planning
            temperature=0.3,  # Lower temperature for more focused planning
            top_p=0.8,
            top_k=30,
            use_greedy=False,
            do_sample=True,
            repetition_penalty=1.2  # Higher penalty for planning
        )
    
    def plan_task(self, task_description: str, available_tools: List[str]) -> List[str]:
        """Decompose a task into actionable steps."""
        try:
            # Create a planning prompt
            tools_list = ", ".join(available_tools)
            planning_prompt = f"""Task: {task_description}
Tools: {tools_list}
Plan:
1."""
            
            # Generate plan using the model
            plan_response = self._generate_with_model(planning_prompt, self.generation_config)
            
            # Parse the response into steps
            steps = self._parse_plan_response(plan_response)
            
            # Fallback if parsing fails
            if not steps:
                steps = [f"Execute task: {task_description}"]
            
            return steps
        except Exception as e:
            logger.error(f"Error in task planning: {e}")
            return [f"Execute task: {task_description}"]
    
    def _generate_with_model(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using the model with memory-efficient sampling."""
        try:
            self.model.eval()
            with torch.no_grad():
                input_tokens = self.tokenizer.encode(prompt)
                if not input_tokens:
                    return ""
                
                input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
                input_ids = ensure_tensor_device(input_ids, device)
                
                # Set up stop tokens
                eos_tokens = [
                    config.pad_token_id,
                    self.tokenizer.vocab.get("</s>", -1),
                    self.tokenizer.vocab.get("<eos>", -1),
                    self.tokenizer.vocab.get("<|endoftext|>", -1)
                ]
                eos_tokens = [t for t in eos_tokens if t != -1]
                
                generated_tokens = []
                current_input = input_ids
                
                for step in range(config.max_tokens):
                    # Check sequence length limit
                    max_seq_len = getattr(self.model.config, 'seq_length', 512)
                    if current_input.size(1) >= max_seq_len:
                        break
                    
                    # Get model output
                    output = self.model(current_input)
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    elif isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    next_token_logits = logits[0, -1, :].clone()
                    
                    # Apply repetition penalty
                    if config.repetition_penalty != 1.0:
                        next_token_logits = apply_repetition_penalty(
                            next_token_logits.unsqueeze(0), 
                            current_input, 
                            config.repetition_penalty
                        ).squeeze(0)
                    
                    # Sample next token
                    next_token_id = sample_next_token(next_token_logits, config)
                    
                    # Check for stop conditions
                    if next_token_id in eos_tokens:
                        break
                    
                    # Add token to generated sequence
                    generated_tokens.append(next_token_id)
                    
                    # Update input for next iteration
                    next_token_tensor = torch.tensor([[next_token_id]], device=device)
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
                
                # Decode generated tokens
                if generated_tokens:
                    response = self.tokenizer.decode(generated_tokens)
                    return response.strip()
                
                return ""
                    
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return ""
        finally:
            # Clean up GPU memory
            with ultra_memory_cleanup():
                pass
    
    def _parse_plan_response(self, response: str) -> List[str]:
        """Parse the model's response into actionable steps."""
        steps = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered steps
            if re.match(r'^\d+\.', line):
                step = re.sub(r'^\d+\.\s*', '', line)
                if step:
                    steps.append(step)
            elif line and not steps:  # If no numbered format, just use the content
                steps.append(line)
        
        return steps[:3]  # Limit to 3 steps for MVP

# =====================================================================
# CORE AGENT CLASS
# =====================================================================

class WordLevelAgent:
    """Core agentic AI that can autonomously execute tasks."""
    
    def __init__(self, model, tokenizer, metadata):
        self.model = model.to(device)  # Ensure model is on correct device
        self.tokenizer = tokenizer
        self.metadata = metadata
        
        # Default generation config (aligned with Train.py conservative approach)
        self.generation_config = GenerationConfig(
            max_tokens=30,  # Reduced for memory efficiency
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            use_greedy=False,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        # Validate tokenizer
        if not validate_tokenizer(tokenizer):
            logger.warning("⚠️ Tokenizer validation failed - responses may be poor quality")
        
        # Initialize tools
        self.tools = {
            "filesystem": FileSystemTool(),
            "calculator": CalculatorTool(),
            "memory": TaskMemoryTool(),
            "web_search": WebSearchTool()
        }
        
        # Initialize planner
        self.planner = TaskPlanner(model, tokenizer)
        
        # Task management
        self.active_tasks = {}
        self.completed_tasks = []
        
        logger.info(f"🤖 Agent initialized with {len(self.tools)} tools")
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown generation parameter: {key}")
    
    def list_capabilities(self) -> Dict[str, str]:
        """List agent capabilities."""
        return {name: tool.description() for name, tool in self.tools.items()}
    
    def execute_task(self, task_description: str, auto_plan: bool = True) -> Task:
        """Execute a task autonomously."""
        task_id = f"task_{len(self.active_tasks) + len(self.completed_tasks) + 1}"
        task = Task(id=task_id, description=task_description)
        
        logger.info(f"🎯 Starting task: {task_description}")
        
        try:
            task.status = "in_progress"
            self.active_tasks[task_id] = task
            
            # Plan the task if auto_plan is enabled
            if auto_plan:
                available_tools = list(self.tools.keys())
                task.steps = self.planner.plan_task(task_description, available_tools)
                logger.info(f"📋 Planned {len(task.steps)} steps")
            
            # Execute each step
            for i, step in enumerate(task.steps):
                logger.info(f"⚡ Step {i+1}: {step}")
                
                try:
                    result = self._execute_step(step)
                    task.results.append(result)
                    
                    if result.success:
                        logger.info(f"✅ Step {i+1} completed")
                    else:
                        logger.warning(f"⚠️ Step {i+1} failed: {result.error}")
                        
                except Exception as e:
                    error_result = ToolResult(False, None, str(e))
                    task.results.append(error_result)
                    logger.error(f"❌ Step {i+1} error: {e}")
            
            # Determine final status
            successful_steps = sum(1 for r in task.results if r.success)
            if successful_steps == len(task.steps):
                task.status = "completed"
                logger.info(f"🎉 Task completed successfully!")
            elif successful_steps > 0:
                task.status = "partially_completed"
                logger.info(f"⚠️ Task partially completed ({successful_steps}/{len(task.steps)} steps)")
            else:
                task.status = "failed"
                logger.error(f"❌ Task failed")
            
            # Log task completion
            self.tools["memory"].execute("log_task", task.status, task_description)
            
        except Exception as e:
            task.status = "failed"
            logger.error(f"❌ Task execution error: {e}")
            task.results.append(ToolResult(False, None, str(e)))
        
        finally:
            # Move to completed tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self.completed_tasks.append(task)
        
        return task
    
    def _execute_step(self, step: str) -> ToolResult:
        """Execute a single step of a task."""
        # Parse the step to determine tool and action
        step_lower = step.lower()
        
        # Simple pattern matching to determine which tool to use
        if any(word in step_lower for word in ["calculate", "math", "compute", "+", "-", "*", "/"]):
            # Extract mathematical expression
            math_pattern = r'[\d+\-*/().\s]+'
            matches = re.findall(math_pattern, step)
            if matches:
                expression = max(matches, key=len).strip()
                return self.tools["calculator"].execute(expression=expression)
        
        elif any(word in step_lower for word in ["file", "read", "write", "save", "load"]):
            if "read" in step_lower:
                # Extract filename
                files = re.findall(r'[\w.-]+\.\w+', step)
                if files:
                    return self.tools["filesystem"].execute("read", path=files[0])
            elif "write" in step_lower or "save" in step_lower:
                # Extract filename and content (simplified)
                files = re.findall(r'[\w.-]+\.\w+', step)
                if files:
                    content = f"Content generated for: {step}"
                    return self.tools["filesystem"].execute("write", path=files[0], content=content)
            elif "list" in step_lower:
                return self.tools["filesystem"].execute("list", path=".")
        
        elif any(word in step_lower for word in ["search", "find", "lookup"]):
            # Extract search query
            query_match = re.search(r'search(?:\s+for)?\s+(.+)', step_lower)
            if query_match:
                query = query_match.group(1).strip()
                return self.tools["web_search"].execute(query=query)
        
        elif any(word in step_lower for word in ["remember", "store", "save", "memory"]):
            # Store information in memory
            key = f"info_{len(self.tools['memory'].memory) + 1}"
            return self.tools["memory"].execute("store", key=key, value=step)
        
        else:
            # Default: try to use the step as a general instruction
            return ToolResult(True, f"Executed: {step}", metadata={"type": "general"})
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return task
        
        return None
    
    def list_tasks(self) -> Dict[str, List[Task]]:
        """List all tasks."""
        return {
            "active": list(self.active_tasks.values()),
            "completed": self.completed_tasks[-10:]  # Last 10 completed
        }

# =====================================================================
# MODEL MANAGER FOR LOADING MODELS (Updated from Train.py)
# =====================================================================

@dataclass
class ModelMetadata:
    """Model metadata structure (from Train.py)."""
    model_name: str = ""
    version: str = ""
    created_at: str = ""
    last_modified: str = ""
    model_config: dict = None
    training_config: dict = None
    dataset_info: dict = None
    performance_metrics: dict = None
    model_size_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    training_time_hours: float = 0.0
    epochs_trained: int = 0
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    hardware_used: str = ""
    pytorch_version: str = ""
    cuda_version: str = None
    notes: str = ""
    tags: list = None

class ModelManager:
    """Lightweight model manager for loading trained models."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def list_models(self) -> List[Dict]:
        """List all available models."""
        models = []
        
        for model_dir in self.save_dir.iterdir():
            if model_dir.is_dir():
                try:
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        models.append({
                            'id': model_dir.name,
                            'model_name': metadata.get('model_name', 'Unknown'),
                            'version': metadata.get('version', 'v1.0'),
                            'created_at': metadata.get('created_at', ''),
                            'best_loss': metadata.get('best_loss', float('inf')),
                            'epochs_trained': metadata.get('epochs_trained', 0),
                            'total_parameters': metadata.get('total_parameters', 0),
                            'precision_type': metadata.get('performance_metrics', {}).get('precision_type', 'unknown'),
                            'mixed_precision_used': metadata.get('performance_metrics', {}).get('mixed_precision_used', False),
                            'deepspeed_used': metadata.get('performance_metrics', {}).get('deepspeed_used', False),
                            'zero_stage': metadata.get('performance_metrics', {}).get('zero_stage', 'N/A')
                        })
                except Exception as e:
                    logger.warning(f"Could not load metadata for {model_dir.name}: {e}")
                    continue
        
        # Sort by best loss
        models.sort(key=lambda x: x.get('best_loss', float('inf')))
        return models
    
    def load_model(self, model_id: str):
        """Load a model by ID."""
        model_path = self.save_dir / model_id
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        # Load metadata
        metadata_file = model_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Model metadata not found: {model_id}")
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        # Load tokenizer
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found: {model_id}")
        
        with open(tokenizer_file, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Recreate tokenizer
        tokenizer = ImprovedTokenizer()
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.trained = True
        
        # Load model config
        model_config_dict = metadata_dict.get('model_config', {})
        model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()
        
        # Create model
        model = MiniTransformer(model_config)
        
        # Load model weights (handle both regular and DeepSpeed checkpoints)
        model_file = model_path / "model.pth"
        checkpoint_dir = model_path / "checkpoint"
        
        if model_file.exists():
            # Regular PyTorch model
            with ultra_memory_cleanup():
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict)
                del state_dict
        elif checkpoint_dir.exists():
            # DeepSpeed checkpoint
            logger.warning("DeepSpeed checkpoint detected, loading as regular model")
            pytorch_model = checkpoint_dir / "pytorch_model.bin"
            if pytorch_model.exists():
                with ultra_memory_cleanup():
                    state_dict = torch.load(pytorch_model, map_location='cpu')
                    model.load_state_dict(state_dict)
                    del state_dict
            else:
                raise FileNotFoundError(f"Model weights not found in DeepSpeed checkpoint: {model_id}")
        else:
            raise FileNotFoundError(f"Model weights not found: {model_id}")
        
        # Create metadata object
        metadata = ModelMetadata(**metadata_dict)
        
        logger.info(f"✅ Loaded model: {model_id}")
        logger.info(f"   Model: {metadata.model_name} {metadata.version}")
        logger.info(f"   Parameters: {metadata.total_parameters:,}")
        logger.info(f"   Best loss: {metadata.best_loss:.4f}")
        logger.info(f"   Vocab size: {tokenizer.vocab_size()}")
        
        # Log training details if available
        if hasattr(metadata, 'performance_metrics') and metadata.performance_metrics:
            pm = metadata.performance_metrics
            logger.info(f"   Precision: {pm.get('precision_type', 'unknown')}")
            logger.info(f"   Mixed Precision: {pm.get('mixed_precision_used', False)}")
            logger.info(f"   DeepSpeed: {pm.get('deepspeed_used', False)}")
            if pm.get('deepspeed_used'):
                logger.info(f"   ZeRO Stage: {pm.get('zero_stage', 'N/A')}")
        
        return model, tokenizer, metadata

# =====================================================================
# IMPROVED TOKENIZER (FROM TRAIN.PY)
# =====================================================================

class ImprovedTokenizer:
    """Improved tokenizer with better stability (from Train.py)."""
    
    def __init__(self):
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, 
            "<user>": 4, "<assistant>": 5, "\n": 6, " ": 7
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.target_vocab_size = 5000  # Aligned with Train.py
        self.trained = False
    
    def train_from_text(self, text, vocab_size=None, min_freq=2):
        """Train tokenizer with smaller vocabulary."""
        if vocab_size:
            self.target_vocab_size = min(vocab_size, 5000)
        
        # Character and word frequency counting
        char_freq = {}
        word_freq = {}
        
        for line in text.split('\n'):
            for char in line:
                if char.isprintable() and char not in self.vocab:
                    char_freq[char] = char_freq.get(char, 0) + 1
            
            words = line.lower().split()
            for word in words:
                if word not in self.vocab:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add frequent characters first
        current_id = len(self.vocab)
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        for char, freq in sorted_chars:
            if freq >= min_freq and current_id < self.target_vocab_size // 2:
                self.vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= min_freq and current_id < self.target_vocab_size:
                self.vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        self.trained = True
        logger.info(f"Tokenizer trained with {len(self.vocab)} tokens")
    
    def encode(self, text):
        """Encode text with fallback to character-level."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")
        
        tokens = []
        words = text.split()
        
        for word in words:
            if word.lower() in self.vocab:
                tokens.append(self.vocab[word.lower()])
            else:
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab["<unk>"])
            
            if " " in self.vocab:
                tokens.append(self.vocab[" "])
        
        if tokens and tokens[-1] == self.vocab.get(" ", -1):
            tokens.pop()
        
        return tokens
    
    def decode(self, token_ids):
        """Decode with better text reconstruction."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<pad>", "<bos>", "<eos>"]:
                    tokens.append(token)
        
        text = ""
        for token in tokens:
            if token == " ":
                text += " "
            elif len(token) == 1:
                text += token
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        
        return text.strip()
    
    def vocab_size(self):
        return len(self.vocab)

# =====================================================================
# MODEL CONFIG AND ARCHITECTURE CLASSES (FROM TRAIN.PY)
# =====================================================================

@dataclass
class ModelConfig:
    """Model configuration (from Train.py)."""
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    seq_length: int = 1024
    dropout: float = 0.1
    model_type: str = "transformer"
    tokenizer_type: str = "improved"
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False

# Import model architecture from Train.py concepts
class MiniTransformer(torch.nn.Module):
    """Simplified transformer for ultra-low VRAM (architecture from Train.py)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embeddings = torch.nn.Parameter(torch.zeros(config.seq_length, config.hidden_size))
        
        # Transformer layers
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(MiniTransformerBlock(config))
        
        # Output layers
        self.ln_final = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings to reduce parameters
        self.lm_head.weight = self.embeddings.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization."""
        torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.pos_embeddings, mean=0.0, std=0.01)
        
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        pos_embeddings = self.pos_embeddings[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        hidden_states = token_embeddings + pos_embeddings
        
        # Apply layers with memory cleanup
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            # Periodic cleanup for memory efficiency
            if i % 2 == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class MiniTransformerBlock(torch.nn.Module):
    """Ultra-simplified transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.attn = MiniAttention(config)
        self.mlp = MiniMLP(config)
        
        self.dropout = torch.nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Pre-norm with immediate cleanup
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)
        del x_norm
        x = residual + self.dropout(attn_out)
        del residual, attn_out
        
        # MLP block
        residual = x
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        del x_norm
        x = residual + self.dropout(mlp_out)
        del residual, mlp_out
        
        return x

class MiniAttention(torch.nn.Module):
    """Memory-efficient attention."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = torch.nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)
        
        # Register causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.seq_length, config.seq_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        del qkv
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        del q, k
        
        # Apply causal mask
        scores.masked_fill_(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        del scores
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        del attn_weights, v
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        del attn_output
        
        return output

class MiniMLP(torch.nn.Module):
    """Smaller MLP block."""
    
    def __init__(self, config):
        super().__init__()
        intermediate_size = max(config.hidden_size * 2, 128)  # Smaller for memory
        
        self.fc1 = torch.nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =====================================================================
# AGENTIC CHAT INTERFACE (Updated with Train.py settings)
# =====================================================================

class AgenticWordAIChat:
    """Agentic chat interface aligned with Train.py memory efficiency."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.agent = None
        self.conversation_history = []
        self.autonomous_mode = False
    
    def load_model(self, model_id: str) -> bool:
        """Load a model and initialize the agent."""
        try:
            with ultra_memory_cleanup():
                model, tokenizer, metadata = self.model_manager.load_model(model_id)
                self.agent = WordLevelAgent(model, tokenizer, metadata)
            logger.info(f"✅ Agentic chat ready with model: {metadata.model_name} {metadata.version}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _generate_conversational_response(self, user_input: str, debug_mode: bool = False) -> str:
        """Generate a conversational response with ultra-efficient context."""
        try:
            # Minimal context for memory efficiency
            if self.conversation_history:
                recent_context = " ".join(self.conversation_history[-1:])  # Only last exchange
                context = f"{recent_context} <user> {user_input}\n<assistant>"
            else:
                context = f"<user> {user_input}\n<assistant>"
            
            # Limit context length
            context = context[-200:] if len(context) > 200 else context
            
            if debug_mode:
                print(f"🔍 Context: {context[:100]}...")
            
            # Ultra-conservative config for memory efficiency
            conv_config = GenerationConfig(
                max_tokens=25,  # Reduced for memory
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                use_greedy=False,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            return self._generate_with_model_safe(context, conv_config, debug_mode)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def _generate_with_model_safe(self, prompt: str, config: GenerationConfig, debug_mode: bool = False) -> str:
        """Ultra-safe generation with Train.py memory management."""
        try:
            self.agent.model.eval()
            
            with torch.no_grad():
                with ultra_memory_cleanup():
                    # Encode input
                    input_tokens = self.agent.tokenizer.encode(prompt)
                    if not input_tokens:
                        return "I couldn't process your input."
                    
                    if debug_mode:
                        print(f"🔍 Input tokens: {len(input_tokens)}")
                        print(f"🔍 Generation config: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
                    
                    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
                    input_ids = ensure_tensor_device(input_ids, device)
                    
                    # Set up stop tokens
                    eos_tokens = [
                        config.pad_token_id,
                        self.agent.tokenizer.vocab.get("</s>", -1),
                        self.agent.tokenizer.vocab.get("<eos>", -1),
                        self.agent.tokenizer.vocab.get("<|endoftext|>", -1)
                    ]
                    eos_tokens = [t for t in eos_tokens if t != -1]
                    
                    generated_tokens = []
                    current_input = input_ids
                    
                    for step in range(config.max_tokens):
                        try:
                            # Check sequence length limit
                            max_seq_len = getattr(self.agent.model.config, 'seq_length', 512)
                            if current_input.size(1) >= max_seq_len:
                                break
                            
                            # Get model output
                            with ultra_memory_cleanup():
                                output = self.agent.model(current_input)
                            
                            # Handle different output formats
                            if hasattr(output, 'logits'):
                                logits = output.logits
                            elif isinstance(output, tuple):
                                logits = output[0]
                            else:
                                logits = output
                            
                            next_token_logits = logits[0, -1, :].clone()
                            
                            # Apply repetition penalty
                            if config.repetition_penalty != 1.0:
                                next_token_logits = apply_repetition_penalty(
                                    next_token_logits.unsqueeze(0), 
                                    current_input, 
                                    config.repetition_penalty
                                ).squeeze(0)
                            
                            # Sample next token
                            next_token_id = sample_next_token(next_token_logits, config)
                            
                            # Check for stop conditions
                            if next_token_id in eos_tokens:
                                break
                            
                            # Add token to generated sequence
                            generated_tokens.append(next_token_id)
                            
                            # Update input for next iteration
                            next_token_tensor = torch.tensor([[next_token_id]], device=device)
                            current_input = torch.cat([current_input, next_token_tensor], dim=1)
                            
                            # Cleanup intermediate tensors
                            del logits, next_token_logits, output
                        
                        except Exception as step_error:
                            if debug_mode:
                                print(f"🔍 Step {step} error: {step_error}")
                            break
                    
                    # Decode generated tokens
                    if generated_tokens:
                        response = self.agent.tokenizer.decode(generated_tokens)
                        
                        # Clean up response
                        response = response.strip()
                        response = re.sub(r'<[^>]*>', '', response)
                        response = re.sub(r'\s+', ' ', response)
                        
                        # Filter gibberish
                        if self._is_gibberish(response):
                            if debug_mode:
                                print(f"🔍 Detected gibberish: {response[:50]}...")
                            return "I'm having trouble generating a coherent response."
                        
                        if debug_mode:
                            print(f"🔍 Generated tokens: {generated_tokens}")
                            print(f"🔍 Final response: {response}")
                        
                        return response if response else "I'm not sure how to respond to that."
                    else:
                        return "I couldn't generate a response."
        
        except Exception as e:
            if debug_mode:
                print(f"🔍 Generation error: {e}")
            logger.error(f"Model generation error: {e}")
            return "Sorry, I encountered an error while thinking."
        
        finally:
            # Ultra-aggressive cleanup (from Train.py)
            with ultra_memory_cleanup():
                pass
    
    def _is_gibberish(self, text: str) -> bool:
        """Detect if generated text is likely gibberish."""
        if not text or len(text.strip()) < 2:
            return True
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.5:
                return True
        
        # Check for random character sequences
        if len(text) > 20 and ' ' not in text:
            return True
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-')
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # Check for repeated patterns
        if len(re.findall(r'(\w{1,3})\1{3,}', text)) > 0:
            return True
        
        return False
    
    def chat(self):
        """Interactive agentic chat interface with Train.py optimizations."""
        if not self.agent:
            # Auto-load best model
            models = self.model_manager.list_models()
            if not models:
                print("❌ No models available. Please train a model first.")
                return
            
            best_model = min(models, key=lambda x: x['best_loss'])
            if not self.load_model(best_model['id']):
                print("❌ Failed to load model.")
                return
        
        print("\n" + "="*70)
        print("🤖 AGENTIC AI SYSTEM (Train.py Optimized)")
        print("="*70)
        print("🔧 Agent Capabilities:")
        capabilities = self.agent.list_capabilities()
        for name, desc in capabilities.items():
            print(f"   • {name}: {desc}")
        
        print(f"\n🔤 Tokenization: Improved - Vocab: {self.agent.tokenizer.vocab_size():,}")
        
        # Show training information from metadata
        if hasattr(self.agent.metadata, 'performance_metrics') and self.agent.metadata.performance_metrics:
            pm = self.agent.metadata.performance_metrics
            print(f"\n📊 Model Training Info:")
            print(f"   • Precision: {pm.get('precision_type', 'unknown')}")
            print(f"   • Mixed Precision: {pm.get('mixed_precision_used', False)}")
            print(f"   • DeepSpeed: {pm.get('deepspeed_used', False)}")
            if pm.get('deepspeed_used'):
                print(f"   • ZeRO Stage: {pm.get('zero_stage', 'N/A')}")
            print(f"   • Gradient Checkpointing: {pm.get('gradient_checkpointing_used', False)}")
        
        # Show current generation settings
        config = self.agent.generation_config
        print(f"\n⚙️ Generation Settings (Memory Optimized):")
        print(f"   • Temperature: {config.temperature}")
        print(f"   • Top-p: {config.top_p}")
        print(f"   • Top-k: {config.top_k}")
        print(f"   • Max tokens: {config.max_tokens}")
        print(f"   • Greedy: {config.use_greedy}")
        print(f"   • Repetition penalty: {config.repetition_penalty}")
        
        print("\n💬 Commands:")
        print("   • Normal chat - Just type your message")
        print("   • /task <description> - Execute a task autonomously")
        print("   • /auto on/off - Toggle autonomous mode")
        print("   • /tasks - List all tasks")
        print("   • /status <task_id> - Check task status")
        print("   • /capabilities - Show agent capabilities")
        print("   • /tokenize <text> - Test tokenization")
        print("   • /generation - Show/modify generation settings")
        print("   • /presets - Load generation presets")
        print("   • /memory - Show memory usage")
        print("   • /clear - Clear conversation history")
        print("   • /debug - Toggle debug mode")
        print("   • /simple <message> - Simple response without history")
        print("   • /exit - Exit chat")
        print("="*70)
        print(f"🧠 Autonomous mode: {'ON' if self.autonomous_mode else 'OFF'}")
        print(f"💾 Memory management: Ultra-aggressive (Train.py optimized)")
        print("-"*70)
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == "/exit":
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == "/debug":
                    debug_mode = not debug_mode
                    print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                
                elif user_input.lower() == "/clear":
                    self.conversation_history = []
                    with ultra_memory_cleanup():
                        pass
                    print("🗑️ Conversation history cleared and memory cleaned!")
                    continue
                
                elif user_input.lower() == "/memory":
                    print(f"\n💾 Memory Status:")
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        cached = torch.cuda.memory_reserved() / 1024**3
                        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                        print(f"   • GPU Allocated: {allocated:.2f}GB")
                        print(f"   • GPU Cached: {cached:.2f}GB")
                        print(f"   • GPU Peak: {max_allocated:.2f}GB")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        allocated = torch.mps.current_allocated_memory() / 1024**3
                        print(f"   • MPS Allocated: {allocated:.2f}GB")
                    else:
                        print("   • Running on CPU")
                    
                    print(f"   • Conversation history: {len(self.conversation_history)} entries")
                    continue
                
                elif user_input.lower() == "/capabilities":
                    print("\n🔧 Agent Capabilities:")
                    for name, desc in self.agent.list_capabilities().items():
                        print(f"   • {name}: {desc}")
                    continue
                
                elif user_input.lower() == "/generation":
                    self._handle_generation_settings()
                    continue
                
                elif user_input.lower() == "/presets":
                    self._handle_generation_presets()
                    continue
                
                elif user_input.startswith("/tokenize "):
                    # Test tokenization
                    text = user_input[10:].strip()
                    if text:
                        encoded = self.agent.tokenizer.encode(text)
                        decoded = self.agent.tokenizer.decode(encoded)
                        
                        print(f"\n🔤 Tokenization Test:")
                        print(f"   Original: {text}")
                        print(f"   Token IDs: {encoded}")
                        print(f"   Decoded: {decoded}")
                        print(f"   Compression ratio: {len(text.split())}/{len(encoded)} = {len(text.split())/len(encoded):.2f}x")
                    else:
                        print("❌ Please provide text to tokenize")
                    continue
                
                elif user_input.lower() == "/tasks":
                    tasks = self.agent.list_tasks()
                    print(f"\n📋 Active Tasks: {len(tasks['active'])}")
                    for task in tasks['active']:
                        print(f"   • {task.id}: {task.description} ({task.status})")
                    print(f"\n✅ Recent Completed Tasks: {len(tasks['completed'])}")
                    for task in tasks['completed'][-5:]:
                        print(f"   • {task.id}: {task.description} ({task.status})")
                    continue
                
                elif user_input.startswith("/status "):
                    task_id = user_input.split(" ", 1)[1].strip()
                    task = self.agent.get_task_status(task_id)
                    if task:
                        print(f"\n📊 Task {task_id}:")
                        print(f"   Description: {task.description}")
                        print(f"   Status: {task.status}")
                        print(f"   Steps: {len(task.steps)}")
                        print(f"   Results: {len(task.results)} completed")
                    else:
                        print(f"❌ Task {task_id} not found")
                    continue
                
                elif user_input.startswith("/auto "):
                    mode = user_input.split(" ", 1)[1].strip().lower()
                    if mode == "on":
                        self.autonomous_mode = True
                        print("🧠 Autonomous mode enabled - I'll execute tasks automatically!")
                    elif mode == "off":
                        self.autonomous_mode = False
                        print("💬 Autonomous mode disabled - Back to normal chat")
                    else:
                        print("❌ Use '/auto on' or '/auto off'")
                    continue
                
                elif user_input.startswith("/simple "):
                    simple_message = user_input[8:].strip()
                    with ultra_memory_cleanup():
                        response = self._generate_simple_response(simple_message, debug_mode)
                    print(f"🤖 AI: {response}")
                    continue
                
                elif user_input.startswith("/task "):
                    task_description = user_input[6:].strip()
                    if task_description:
                        print(f"🎯 Executing task: {task_description}")
                        with ultra_memory_cleanup():
                            task = self.agent.execute_task(task_description)
                        print(f"📊 Task {task.id} {task.status}")
                        
                        # Show results
                        successful_results = [r for r in task.results if r.success]
                        if successful_results:
                            print("✅ Successful results:")
                            for i, result in enumerate(successful_results[:3]):  # Show first 3
                                print(f"   {i+1}. {str(result.data)[:100]}{'...' if len(str(result.data)) > 100 else ''}")
                    else:
                        print("❌ Please specify a task description")
                    continue
                
                # Regular conversation or autonomous task detection
                if self.autonomous_mode and self._is_task_request(user_input):
                    print("🧠 I detect this is a task - executing autonomously...")
                    with ultra_memory_cleanup():
                        task = self.agent.execute_task(user_input)
                    print(f"📊 Task {task.id} {task.status}")
                    
                    # Provide conversational response about the task
                    if task.status == "completed":
                        response = f"I've successfully completed your task! I executed {len(task.steps)} steps."
                    else:
                        response = f"I attempted your task but encountered some issues. Status: {task.status}"
                    
                    print(f"🤖 AI: {response}")
                
                else:
                    # Normal conversational response with memory cleanup
                    with ultra_memory_cleanup():
                        response = self._generate_conversational_response(user_input, debug_mode)
                    print(f"🤖 AI: {response}")
                
                # Add to conversation history (keep minimal for memory)
                self.conversation_history.append(f"<user> {user_input}")
                self.conversation_history.append(f"<assistant> {response if 'response' in locals() else 'Task executed'}")
                
                # Trim history aggressively for memory
                if len(self.conversation_history) > 4:  # Only keep 2 exchanges
                    self.conversation_history = self.conversation_history[-4:]
                
                # Periodic memory cleanup
                if len(self.conversation_history) % 2 == 0:
                    with ultra_memory_cleanup():
                        pass
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
                # Emergency memory cleanup on error
                with ultra_memory_cleanup():
                    pass
    
    def _handle_generation_settings(self):
        """Handle generation settings modification (memory-conscious)."""
        config = self.agent.generation_config
        print(f"\n⚙️ Current Generation Settings (Memory Optimized):")
        print(f"   1. Temperature: {config.temperature}")
        print(f"   2. Top-p: {config.top_p}")
        print(f"   3. Top-k: {config.top_k}")
        print(f"   4. Greedy: {config.use_greedy}")
        print(f"   5. Repetition penalty: {config.repetition_penalty}")
        print(f"   6. Max tokens: {config.max_tokens} (memory limited)")
        print(f"   7. Do sample: {config.do_sample}")
        
        try:
            choice = input("\nEnter setting number to modify (or press Enter to skip): ").strip()
            if not choice:
                return
            
            choice = int(choice)
            
            if choice == 1:
                temp = float(input("Enter temperature (0.1-2.0): "))
                self.agent.update_generation_config(temperature=max(0.1, min(2.0, temp)))
            elif choice == 2:
                top_p = float(input("Enter top-p (0.1-1.0): "))
                self.agent.update_generation_config(top_p=max(0.1, min(1.0, top_p)))
            elif choice == 3:
                top_k = int(input("Enter top-k (1-100): "))
                self.agent.update_generation_config(top_k=max(1, min(100, top_k)))
            elif choice == 4:
                greedy = input("Use greedy decoding? (y/n): ").lower().startswith('y')
                self.agent.update_generation_config(use_greedy=greedy)
            elif choice == 5:
                penalty = float(input("Enter repetition penalty (1.0-1.5): "))
                self.agent.update_generation_config(repetition_penalty=max(1.0, min(1.5, penalty)))
            elif choice == 6:
                max_tokens = int(input("Enter max tokens (10-50 for memory efficiency): "))
                self.agent.update_generation_config(max_tokens=max(10, min(50, max_tokens)))
                print("⚠️ Note: Higher token counts may cause memory issues")
            elif choice == 7:
                do_sample = input("Enable sampling? (y/n): ").lower().startswith('y')
                self.agent.update_generation_config(do_sample=do_sample)
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Invalid input")
    
    def _handle_generation_presets(self):
        """Handle generation preset selection (memory-optimized)."""
        presets = {
            "1": ("Creative (Low Memory)", {"temperature": 0.9, "top_p": 0.95, "top_k": 40, "use_greedy": False, "repetition_penalty": 1.1, "max_tokens": 25}),
            "2": ("Balanced (Ultra Memory)", {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "use_greedy": False, "repetition_penalty": 1.1, "max_tokens": 30}),
            "3": ("Focused (Memory Safe)", {"temperature": 0.3, "top_p": 0.8, "top_k": 30, "use_greedy": False, "repetition_penalty": 1.2, "max_tokens": 20}),
            "4": ("Precise (Minimal Memory)", {"temperature": 0.1, "top_p": 0.7, "top_k": 20, "use_greedy": False, "repetition_penalty": 1.3, "max_tokens": 15}),
            "5": ("Greedy (Fastest)", {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "use_greedy": True, "repetition_penalty": 1.0, "max_tokens": 20})
        }
        
        print("\n🎛️ Generation Presets (Memory Optimized):")
        for key, (name, settings) in presets.items():
            print(f"   {key}. {name}")
            print(f"      Temperature: {settings['temperature']}, Top-p: {settings['top_p']}, "
                  f"Top-k: {settings['top_k']}, Max tokens: {settings['max_tokens']}")
        
        try:
            choice = input("\nSelect preset (1-5): ").strip()
            if choice in presets:
                name, settings = presets[choice]
                self.agent.update_generation_config(**settings)
                print(f"✅ Applied {name} preset")
            else:
                print("❌ Invalid choice")
        except Exception as e:
            print(f"❌ Error applying preset: {e}")
    
    def _is_task_request(self, text: str) -> bool:
        """Detect if user input is a task request."""
        task_indicators = [
            "calculate", "compute", "math", "solve",
            "create", "write", "save", "file",
            "search", "find", "look up", "lookup",
            "remember", "store", "list", "show me"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in task_indicators)
    
    def _generate_simple_response(self, user_input: str, debug_mode: bool = False) -> str:
        """Generate a simple response without conversation history (ultra memory-efficient)."""
        try:
            # Very simple prompt
            prompt = f"Human: {user_input}\nAI:"
            
            if debug_mode:
                print(f"🔍 Simple prompt: {prompt}")
            
            # Ultra-conservative config for memory efficiency
            simple_config = GenerationConfig(
                max_tokens=15,  # Very limited for memory
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                use_greedy=False,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            return self._generate_with_model_safe(prompt, simple_config, debug_mode)
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return "I'm having trouble generating a response. Please try again."

# =====================================================================
# MAIN FUNCTION (Updated for Train.py compatibility)
# =====================================================================

def main():
    """Main function for the agentic AI system (Train.py optimized)."""
    print("\n" + "="*70)
    print("🤖 AGENTIC AI SYSTEM (Train.py Optimized)")
    print("   Autonomous Task Execution + Conversational AI")
    print("   🔤 Improved Tokenization for Enhanced Efficiency")
    print("   ⚙️ Advanced Sampling: Temperature, Top-p, Top-k, Greedy")
    print("   💾 Ultra-Aggressive Memory Management")
    print("   🚀 Compatible with Train.py Models")
    print("="*70)
    print(f"🔧 Device: {device}")
    print(f"🐍 PyTorch: {torch.__version__}")
    
    # Apply CUDA optimizations (from Train.py)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        print("🔧 CUDA optimizations applied (Train.py style)")
    
    print("="*70)
    
    # Initialize model manager
    try:
        model_manager = ModelManager("models")
    except Exception as e:
        print(f"❌ Failed to initialize ModelManager: {e}")
        print("💡 Make sure you have the models directory")
        return 1
    
    # Check for available models
    try:
        models = model_manager.list_models()
        if not models:
            print("❌ No trained models found!")
            print("\n📝 To get started:")
            print("   1. Run 'python Train.py' to train a model")
            print("   2. Then run this script to start the agentic AI")
            return 1
        
        print(f"✅ Found {len(models)} trained model(s)")
        
        # Show model info with Train.py details
        for model in models[:3]:  # Show first 3 models
            print(f"   📁 {model['id']}: {model.get('model_name', 'Unknown')} (loss: {model.get('best_loss', 'N/A'):.4f})")
            print(f"      📊 Parameters: {model.get('total_parameters', 0):,} | Epochs: {model.get('epochs_trained', 0)}")
            
            # Show training details
            precision = model.get('precision_type', 'unknown')
            mixed_precision = model.get('mixed_precision_used', False)
            deepspeed = model.get('deepspeed_used', False)
            zero_stage = model.get('zero_stage', 'N/A')
            
            print(f"      🔧 Precision: {precision} | Mixed: {mixed_precision} | DeepSpeed: {deepspeed}")
            if deepspeed and zero_stage != 'N/A':
                print(f"      🚀 ZeRO Stage: {zero_stage}")
    
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return 1
    
    # Initialize agentic chat with ultra memory management
    try:
        with ultra_memory_cleanup():
            agentic_chat = AgenticWordAIChat(model_manager)
        
        print("\n🧠 Initializing agentic AI system...")
        agentic_chat.chat()
        return 0
        
    except Exception as e:
        logger.error(f"Agentic chat system error: {e}")
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Final cleanup
        with ultra_memory_cleanup():
            pass

if __name__ == "__main__":
    exit(main())