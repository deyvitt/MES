# =============================================================================
# system/inference.py
# =============================================================================
import torch
from typing import Dict, List, Optional, Union
import time

class MambaInferenceEngine:
    """Optimized inference engine for Mamba swarm"""
    
    def __init__(self, swarm_engine):
        self.swarm_engine = swarm_engine
        self.config = swarm_engine.config
        
        # Inference optimizations
        self.use_half_precision = True
        self.use_torch_compile = hasattr(torch, 'compile')
        
        # Apply optimizations
        self._optimize_models()
    
    def _optimize_models(self):
        """Apply inference optimizations"""
        if self.use_half_precision and self.config.device != 'cpu':
            # Convert to half precision for faster inference
            for specialist in self.swarm_engine.tlm_manager.specialists.values():
                specialist.model = specialist.model.half()
            self.swarm_engine.aggregator = self.swarm_engine.aggregator.half()
        
        if self.use_torch_compile:
            try:
                # Compile models for faster inference (PyTorch 2.0+)
                for specialist in self.swarm_engine.tlm_manager.specialists.values():
                    specialist.model = torch.compile(specialist.model)
                self.swarm_engine.aggregator = torch.compile(self.swarm_engine.aggregator)
                print("Models compiled for faster inference")
            except Exception as e:
                print(f"Could not compile models: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 100, 
                temperature: float = 0.7, top_k: int = 50) -> Dict:
        """
        Generate text response with advanced sampling
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Dict with generated text and metadata
        """
        start_time = time.time()
        
        # Process through swarm
        result = self.swarm_engine.process_request(prompt, max_tokens)
        
        if not result['success']:
            return result
        
        # Add inference metadata
        result.update({
            'temperature': temperature,
            'top_k': top_k,
            'inference_time': time.time() - start_time,
            'tokens_per_second': max_tokens / (time.time() - start_time)
        })
        
        return result
    
    def stream_generate(self, prompt: str, max_tokens: int = 100):
        """
        Stream generation token by token (placeholder implementation)
        """
        # This would implement streaming generation
        # For now, return the full response
        result = self.generate(prompt, max_tokens)
        yield result['response']
    
    def chat_completion(self, messages: List[Dict], max_tokens: int = 100) -> Dict:
        """
        Chat completion interface similar to OpenAI API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Chat completion response
        """
        # Convert messages to single prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        result = self.generate(prompt, max_tokens)
        
        if result['success']:
            # Format as chat completion
            return {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': result['response']
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(result['response'].split()),
                    'total_tokens': len(prompt.split()) + len(result['response'].split())
                },
                'model': 'mamba-swarm-70m',
                'inference_time': result.get('inference_time', 0)
            }
        else:
            return {
                'error': result.get('error', 'Unknown error'),
                'success': False
            }
    
    def _format_chat_prompt(self, messages: List[Dict]) -> str:
        """Format chat messages into a single prompt"""
        formatted = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted += f"System: {content}\n"
            elif role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
        
        formatted += "Assistant: "
        return formatted 