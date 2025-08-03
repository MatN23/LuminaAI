# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from model_manager import ModelManager
from subword_transformer import SubwordTransformer, SubwordTokenizer

logger = logging.getLogger(__name__)

class GenerationDebugger:
    """Debug tool for analyzing model generation issues."""
    
    def __init__(self, model_id: str = "latest", models_dir: str = "models"):
        self.model_manager = ModelManager(models_dir)
        self.device = self._setup_device()
        
        # Load model
        try:
            self.model, self.tokenizer, self.metadata = self.model_manager.load_model(model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded model: {self.metadata.model_name}")
            print(f"   Loss: {self.metadata.best_loss:.4f}")
            print(f"   Perplexity: {self.metadata.best_perplexity:.2f}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def debug_tokenizer(self):
        """Debug tokenizer behavior."""
        print("\n" + "="*60)
        print("🔤 TOKENIZER DEBUGGING")
        print("="*60)
        
        # Test basic tokenization
        test_phrases = [
            "Hello world",
            "How are you today?",
            "This is a test",
            "<user> Hello <assistant>",
            "The quick brown fox"
        ]
        
        print(f"Vocabulary size: {self.tokenizer.vocab_size():,}")
        print(f"Number of BPE merges: {len(self.tokenizer.merges):,}")
        
        # Show special tokens
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<user>", "<assistant>"]
        print(f"\nSpecial tokens:")
        for token in special_tokens:
            token_id = self.tokenizer.vocab.get(token, "NOT FOUND")
            print(f"  {token}: {token_id}")
        
        print(f"\nTokenization examples:")
        for phrase in test_phrases:
            tokens = self.tokenizer.tokenize(phrase)
            encoded = self.tokenizer.encode(phrase)
            decoded = self.tokenizer.decode(encoded)
            
            print(f"\nInput: '{phrase}'")
            print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"  IDs: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
            print(f"  Decoded: '{decoded}'")
            print(f"  Match: {'✅' if phrase.lower().strip() == decoded.lower().strip() else '❌'}")
    
    def debug_model_predictions(self, prompt: str = "<user> Hello"):
        """Debug raw model predictions."""
        print("\n" + "="*60)
        print("🧠 MODEL PREDICTION DEBUGGING")
        print("="*60)
        
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        print(f"Prompt: '{prompt}'")
        print(f"Input IDs: {input_ids[0].tolist()}")
        print(f"Input tokens: {[self.tokenizer.id_to_token.get(id.item(), '<UNK>') for id in input_ids[0]]}")
        
        with torch.no_grad():
            # Get raw logits
            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :]  # Last position
            
            # Analyze predictions
            probs = F.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 20)
            
            print(f"\nTop 20 predicted tokens:")
            for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
                token = self.tokenizer.id_to_token.get(idx.item(), '<UNK>')
                print(f"  {i+1:2d}. {token:15s} (ID:{idx.item():4d}) - {prob.item():.4f}")
            
            # Check for problematic patterns
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_prob = probs.max().item()
            
            print(f"\nPrediction analysis:")
            print(f"  Entropy: {entropy:.4f} (higher = more uncertain)")
            print(f"  Max probability: {max_prob:.4f}")
            print(f"  Temperature effect: {'Low confidence' if entropy > 5 else 'High confidence'}")
    
    def debug_generation_step_by_step(self, prompt: str = "<user> Hello", max_steps: int = 10):
        """Debug generation step by step."""
        print("\n" + "="*60)
        print("🔍 STEP-BY-STEP GENERATION DEBUGGING")
        print("="*60)
        
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        generated = input_ids.clone()
        
        print(f"Starting prompt: '{prompt}'")
        print(f"Initial tokens: {[self.tokenizer.id_to_token.get(id.item(), '<UNK>') for id in input_ids[0]]}")
        
        for step in range(max_steps):
            with torch.no_grad():
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :]
                
                # Test different sampling strategies
                strategies = {
                    "greedy": self._greedy_sample(next_token_logits),
                    "temp_0.7": self._temperature_sample(next_token_logits, 0.7),
                    "temp_1.0": self._temperature_sample(next_token_logits, 1.0),
                    "top_k_50": self._top_k_sample(next_token_logits, 50),
                }
                
                print(f"\nStep {step + 1}:")
                print(f"  Current sequence: {self.tokenizer.decode(generated[0].tolist())}")
                
                for strategy_name, next_token_id in strategies.items():
                    token = self.tokenizer.id_to_token.get(next_token_id, '<UNK>')
                    prob = F.softmax(next_token_logits, dim=-1)[next_token_id].item()
                    print(f"  {strategy_name:10s}: {token:15s} (prob: {prob:.4f})")
                
                # Use greedy for continuation
                next_token_id = strategies["greedy"]
                next_token = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop on end token
                if next_token_id == self.tokenizer.vocab.get("</s>", -1):
                    print(f"  → Stopped on </s> token")
                    break
        
        final_text = self.tokenizer.decode(generated[0].tolist())
        print(f"\nFinal generated text: '{final_text}'")
    
    def _greedy_sample(self, logits):
        return torch.argmax(logits, dim=-1).item()
    
    def _temperature_sample(self, logits, temperature):
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def _top_k_sample(self, logits, k):
        top_k_logits, top_k_indices = torch.topk(logits, k)
        probs = F.softmax(top_k_logits, dim=-1)
        selected = torch.multinomial(probs, 1)
        return top_k_indices[selected].item()
    
    def test_generation_parameters(self, prompt: str = "<user> Hello"):
        """Test different generation parameters to find best settings."""
        print("\n" + "="*60)
        print("🎛️ GENERATION PARAMETER TESTING")
        print("="*60)
        
        test_configs = [
            {"temperature": 0.1, "top_k": 0, "top_p": 1.0, "name": "Very Conservative"},
            {"temperature": 0.5, "top_k": 20, "top_p": 0.9, "name": "Conservative"},
            {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "name": "Balanced"},
            {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "name": "Creative"},
            {"temperature": 1.5, "top_k": 0, "top_p": 0.95, "name": "Very Creative"},
        ]
        
        for config in test_configs:
            print(f"\n{config['name']} (temp={config['temperature']}, top_k={config['top_k']}, top_p={config['top_p']}):")
            
            try:
                generated = self.generate_with_params(
                    prompt, 
                    max_length=20,
                    temperature=config['temperature'],
                    top_k=config['top_k'],
                    top_p=config['top_p']
                )
                print(f"  Result: '{generated}'")
            except Exception as e:
                print(f"  Error: {e}")
    
    def generate_with_params(self, prompt: str, max_length: int = 50, 
                           temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
        """Generate text with specific parameters."""
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= self.metadata.model_config.seq_length:
                    break
                
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop on end token
                if next_token.item() == self.tokenizer.vocab.get("</s>", -1):
                    break
        
        # Decode only the generated part
        response_ids = generated[0][input_ids.size(1):].tolist()
        return self.tokenizer.decode(response_ids)
    
    def analyze_training_data_format(self):
        """Analyze if the model learned the expected conversation format."""
        print("\n" + "="*60)
        print("📚 TRAINING DATA FORMAT ANALYSIS")
        print("="*60)
        
        # Test if model learned conversation patterns
        test_prompts = [
            "<user>",
            "<user> Hello",
            "<user> How are you?",
            "<user> What is your name?",
            "<assistant>",
        ]
        
        for prompt in test_prompts:
            print(f"\nTesting prompt: '{prompt}'")
            
            # Get model's immediate prediction
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                top_5_probs, top_5_indices = torch.topk(F.softmax(next_token_logits, dim=-1), 5)
                
                print("  Top 5 next tokens:")
                for prob, idx in zip(top_5_probs, top_5_indices):
                    token = self.tokenizer.id_to_token.get(idx.item(), '<UNK>')
                    print(f"    {token:15s} ({prob.item():.4f})")
    
    def full_diagnosis(self):
        """Run complete diagnosis of generation issues."""
        print("🚨 RUNNING FULL GENERATION DIAGNOSIS")
        print("="*70)
        
        try:
            # 1. Tokenizer check
            self.debug_tokenizer()
            
            # 2. Model predictions
            self.debug_model_predictions()
            
            # 3. Step-by-step generation
            self.debug_generation_step_by_step()
            
            # 4. Parameter testing
            self.test_generation_parameters()
            
            # 5. Training format analysis
            self.analyze_training_data_format()
            
            print("\n" + "="*70)
            print("✅ DIAGNOSIS COMPLETE")
            print("="*70)
            
            # Recommendations
            print("\n🔧 RECOMMENDATIONS:")
            
            if self.metadata.best_loss > 3.0:
                print("❌ High loss detected - model needs more training")
            elif self.metadata.best_loss > 2.0:
                print("⚠️ Moderate loss - consider training longer or adjusting hyperparameters")
            else:
                print("✅ Loss looks reasonable")
            
            if self.metadata.best_perplexity > 50:
                print("❌ Very high perplexity - model struggling to learn patterns")
            elif self.metadata.best_perplexity > 20:
                print("⚠️ High perplexity - model still learning")
            else:
                print("✅ Perplexity in reasonable range")
            
            print("\n💡 Try these fixes:")
            print("1. Use lower temperature (0.1-0.5) for more coherent output")
            print("2. Try greedy decoding (temperature=0) to see best predictions")
            print("3. Check if model learned conversation format with <user>/<assistant> tokens")
            print("4. Consider training longer if loss is still high")
            print("5. Verify tokenizer encode/decode cycle works correctly")
            
        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run generation debugging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug model generation issues")
    parser.add_argument("--model", default="latest", help="Model ID to debug")
    parser.add_argument("--prompt", default="<user> Hello", help="Test prompt")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnosis only")
    
    args = parser.parse_args()
    
    try:
        debugger = GenerationDebugger(args.model)
        
        if args.quick:
            debugger.debug_tokenizer()
            debugger.test_generation_parameters(args.prompt)
        else:
            debugger.full_diagnosis()
            
    except Exception as e:
        print(f"❌ Failed to run debugger: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())