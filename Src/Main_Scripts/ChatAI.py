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
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import shared components
from model_manager import ModelManager, ModelMetadata
from word_transformer import WordTransformer, WordTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """Setup the best available device with proper error handling."""
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        else:
            device = torch.device("cpu")
            logger.info("Using device: CPU")
        return device
    except Exception as e:
        logger.warning(f"Error setting up device: {e}. Falling back to CPU.")
        return torch.device("cpu")

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
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"🔍 Tokenizer test:")
        print(f"   Original: {test_text}")
        print(f"   Encoded: {encoded[:10]}..." if len(encoded) > 10 else f"   Encoded: {encoded}")
        print(f"   Decoded: {decoded}")
        return len(encoded) > 0 and decoded.strip()
    except Exception as e:
        print(f"❌ Tokenizer validation failed: {e}")
        return False

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
    
    def __init__(self, model: WordTransformer, tokenizer: WordTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure model is on correct device
        self.model = self.model.to(device)
    
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
            plan_response = self._generate_with_model(planning_prompt, max_length=100)
            
            # Parse the response into steps
            steps = self._parse_plan_response(plan_response)
            
            # Fallback if parsing fails
            if not steps:
                steps = [f"Execute task: {task_description}"]
            
            return steps
        except Exception as e:
            logger.error(f"Error in task planning: {e}")
            return [f"Execute task: {task_description}"]
    
    def _generate_with_model(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the model with proper device handling."""
        try:
            self.model.eval()
            with torch.no_grad():
                input_tokens = self.tokenizer.encode(prompt)
                if not input_tokens:
                    return ""
                
                input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
                input_ids = ensure_tensor_device(input_ids, device)
                
                # Simple greedy generation to avoid sampling issues
                for _ in range(max_length):
                    if input_ids.size(1) >= getattr(self.model.config, 'seq_length', 512):
                        break
                    
                    logits = self.model(input_ids)
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                    
                    # Use greedy decoding for more stable results
                    next_token_id = torch.argmax(logits[0, -1, :]).item()
                    
                    # Check for end token
                    if next_token_id == self.tokenizer.vocab.get("</s>", -1) or next_token_id == 0:
                        break
                    
                    next_token_tensor = torch.tensor([[next_token_id]], device=device)
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                
                # Decode only the generated part
                generated_tokens = input_ids[0][len(self.tokenizer.encode(prompt)):].tolist()
                response = self.tokenizer.decode(generated_tokens)
                return response.strip()
                    
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return ""
    
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
    
    def __init__(self, model: WordTransformer, tokenizer: WordTokenizer, metadata: ModelMetadata):
        self.model = model.to(device)  # Ensure model is on correct device
        self.tokenizer = tokenizer
        self.metadata = metadata
        
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
        if any(word in step_lower for word in ["calculate", "math", "compute", "+"  , "-", "*", "/"]):
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
# AGENTIC CHAT INTERFACE
# =====================================================================

class AgenticWordAIChat:
    """Agentic chat interface that can autonomously execute tasks."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.agent = None
        self.conversation_history = []
        self.autonomous_mode = False
    
    def load_model(self, model_id: str) -> bool:
        """Load a model and initialize the agent."""
        try:
            model, tokenizer, metadata = self.model_manager.load_model(model_id)
            self.agent = WordLevelAgent(model, tokenizer, metadata)
            logger.info(f"✅ Agentic chat ready with model: {metadata.model_name} {metadata.version}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def chat(self):
        """Interactive agentic chat interface."""
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
        print("🤖 AGENTIC WORD-LEVEL AI SYSTEM")
        print("="*70)
        print("🔧 Agent Capabilities:")
        capabilities = self.agent.list_capabilities()
        for name, desc in capabilities.items():
            print(f"   • {name}: {desc}")
        
        print("\n💬 Commands:")
        print("   • Normal chat - Just type your message")
        print("   • /task <description> - Execute a task autonomously")
        print("   • /auto on/off - Toggle autonomous mode")
        print("   • /tasks - List all tasks")
        print("   • /status <task_id> - Check task status")
        print("   • /capabilities - Show agent capabilities")
        print("   • /clear - Clear conversation history")
        print("   • /debug - Toggle debug mode")
        print("   • /simple <message> - Simple response without history")
        print("   • /exit - Exit chat")
        print("="*70)
        print(f"🧠 Autonomous mode: {'ON' if self.autonomous_mode else 'OFF'}")
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
                    print("🗑️ Conversation history cleared!")
                    continue
                
                elif user_input.lower() == "/capabilities":
                    print("\n🔧 Agent Capabilities:")
                    for name, desc in self.agent.list_capabilities().items():
                        print(f"   • {name}: {desc}")
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
                    response = self._generate_simple_response(simple_message, debug_mode)
                    print(f"🤖 AI: {response}")
                    continue
                
                elif user_input.startswith("/task "):
                    task_description = user_input[6:].strip()
                    if task_description:
                        print(f"🎯 Executing task: {task_description}")
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
                    task = self.agent.execute_task(user_input)
                    print(f"📊 Task {task.id} {task.status}")
                    
                    # Provide conversational response about the task
                    if task.status == "completed":
                        response = f"I've successfully completed your task! I executed {len(task.steps)} steps."
                    else:
                        response = f"I attempted your task but encountered some issues. Status: {task.status}"
                    
                    print(f"🤖 AI: {response}")
                
                else:
                    # Normal conversational response
                    response = self._generate_conversational_response(user_input, debug_mode)
                    print(f"🤖 AI: {response}")
                
                # Add to conversation history
                self.conversation_history.append(f"<user> {user_input}")
                self.conversation_history.append(f"<bot> {response if 'response' in locals() else 'Task executed'}")
                
                # Trim history
                if len(self.conversation_history) > 10:  # Reduced history size
                    self.conversation_history = self.conversation_history[-10:]
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
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
        """Generate a simple response without conversation history."""
        try:
            # Very simple prompt
            prompt = f"Human: {user_input}\nAI:"
            
            if debug_mode:
                print(f"🔍 Simple prompt: {prompt}")
            
            return self._generate_with_model_safe(prompt, max_tokens=30, debug_mode=debug_mode)
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return "I'm having trouble generating a response. Please try again."
    
    def _generate_conversational_response(self, user_input: str, debug_mode: bool = False) -> str:
        """Generate a conversational response with limited context."""
        try:
            # Build minimal context
            if self.conversation_history:
                recent_context = " ".join(self.conversation_history[-2:])  # Only last exchange
                context = f"{recent_context} <user> {user_input}\n<bot>"
            else:
                context = f"<user> {user_input}\n<bot>"
            
            if debug_mode:
                print(f"🔍 Context: {context[:100]}...")
            
            return self._generate_with_model_safe(context, max_tokens=50, debug_mode=debug_mode)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def _generate_with_model_safe(self, prompt: str, max_tokens: int = 50, debug_mode: bool = False) -> str:
        """Safely generate text with extensive error handling and debugging."""
        try:
            self.agent.model.eval()
            
            with torch.no_grad():
                # Encode input
                input_tokens = self.agent.tokenizer.encode(prompt)
                if not input_tokens:
                    return "I couldn't process your input."
                
                if debug_mode:
                    print(f"🔍 Input tokens: {len(input_tokens)}")
                
                input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
                input_ids = ensure_tensor_device(input_ids, device)
                
                generated_tokens = []
                current_input = input_ids
                
                for step in range(max_tokens):
                    try:
                        # Get model output
                        with torch.no_grad():
                            output = self.agent.model(current_input)
                        
                        # Handle different output formats
                        if hasattr(output, 'logits'):
                            logits = output.logits
                        elif isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        
                        # Get next token (greedy decoding)
                        next_token_logits = logits[0, -1, :]
                        next_token_id = torch.argmax(next_token_logits).item()
                        
                        # Check for stop conditions
                        if next_token_id in [0, self.agent.tokenizer.vocab.get("</s>", -1), self.agent.tokenizer.vocab.get("<eos>", -1)]:
                            break
                        
                        # Add token to generated sequence
                        generated_tokens.append(next_token_id)
                        
                        # Update input for next iteration
                        next_token_tensor = torch.tensor([[next_token_id]], device=device)
                        current_input = torch.cat([current_input, next_token_tensor], dim=1)
                        
                        # Stop if we hit max sequence length
                        if current_input.size(1) >= getattr(self.agent.model.config, 'seq_length', 512):
                            break
                    
                    except Exception as step_error:
                        if debug_mode:
                            print(f"🔍 Step {step} error: {step_error}")
                        break
                
                # Decode generated tokens
                if generated_tokens:
                    response = self.agent.tokenizer.decode(generated_tokens)
                    
                    # Clean up response
                    response = response.strip()
                    response = re.sub(r'<[^>]*>', '', response)  # Remove any XML tags
                    response = re.sub(r'\s+', ' ', response)     # Normalize whitespace
                    
                    # Filter out obvious gibberish patterns
                    if self._is_gibberish(response):
                        if debug_mode:
                            print(f"🔍 Detected gibberish: {response[:50]}...")
                        return "I'm having trouble generating a coherent response. The model may need more training."
                    
                    if debug_mode:
                        print(f"🔍 Generated: {response}")
                    
                    return response if response else "I'm not sure how to respond to that."
                else:
                    return "I couldn't generate a response."
        
        except Exception as e:
            if debug_mode:
                print(f"🔍 Generation error: {e}")
            logger.error(f"Model generation error: {e}")
            return "Sorry, I encountered an error while thinking."
    
    def _is_gibberish(self, text: str) -> bool:
        """Detect if generated text is likely gibberish."""
        if not text or len(text.strip()) < 2:
            return True
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.5:  # More than 50% repetition
                return True
        
        # Check for random character sequences
        if len(text) > 20 and ' ' not in text:  # Long string without spaces
            return True
        
        # Check for excessive special characters or numbers
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-')
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # Check for known gibberish patterns
        gibberish_patterns = [
            r'encrypted_text',
            r'predicted_token',
            r'ondrop',
            r'setitems\d+',
            r'_image[a-z]+',
            r'alignment absorption',
            r'\b[a-z]{15,}\b',  # Very long nonsense words
        ]
        
        for pattern in gibberish_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    """Main function for the agentic AI system."""
    print("\n" + "="*70)
    print("🤖 AGENTIC WORD-LEVEL AI SYSTEM")
    print("   Autonomous Task Execution + Conversational AI")
    print("="*70)
    print(f"🔧 Device: {device}")
    print(f"🐍 PyTorch: {torch.__version__}")
    print("="*70)
    
    # Initialize model manager
    try:
        model_manager = ModelManager("models")
    except Exception as e:
        print(f"❌ Failed to initialize ModelManager: {e}")
        print("💡 Make sure you have the model_manager.py file and models directory")
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
        
        # Show model info
        for model in models[:3]:  # Show first 3 models
            print(f"   📁 {model['id']}: {model.get('model_name', 'Unknown')} (loss: {model.get('best_loss', 'N/A'):.4f})")
    
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return 1
    
    # Initialize agentic chat
    agentic_chat = AgenticWordAIChat(model_manager)
    
    try:
        agentic_chat.chat()
        return 0
    except Exception as e:
        logger.error(f"Agentic chat system error: {e}")
        print(f"❌ System error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())