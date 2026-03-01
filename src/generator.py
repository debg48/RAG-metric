import requests
import json
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class OllamaGenerator:
    """Wrapper for local Ollama API generation."""
    
    def __init__(self, model_name: str = "goekdenizguelmez/JOSIEFIED-Qwen3:4b", 
                 host: str = "http://localhost:11434", 
                 temperature: float = 0.3,
                 max_tokens: int = 150):
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"{self.host}/api/generate"
        
    def _build_prompt(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Construct the RAG prompt."""
        context_str = ""
        for i, p in enumerate(passages):
            context_str += f"[{i+1}] {p['text']}\n"
            
        prompt = f"""You are a helpful and precise assistant. Answer the following question strictly based on the provided context passages. If the answer is not in the context, say "I don't know based on the context." Keep your answer concise and direct.

Context passages:
{context_str}

Question: {query}
Answer:"""
        return prompt
        
    def _build_confidence_prompt(self, query: str, passages: List[Dict[str, Any]], answer: str) -> str:
        """Construct prompt to elicit self-reported confidence."""
        context_str = ""
        for i, p in enumerate(passages):
            context_str += f"[{i+1}] {p['text']}\n"
            
        prompt = f"""Context passages:
{context_str}

Question: {query}
Generated Answer: {answer}

Based on the context passages provided, how confident are you that the Generated Answer is correct and fully supported by the context? 
Provide ONLY a single float number between 0.0 and 1.0, where 0.0 means completely unsure/incorrect, and 1.0 means absolutely certain/correct. Do not output any other text."""
        return prompt

    def generate(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Generate answer directly given query and context passages."""
        prompt = self._build_prompt(query, passages)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return "Generation Error"
            
    def generate_with_confidence(self, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Generate answer and then ask model for its confidence."""
        # 1. Generate answer
        answer = self.generate(query, passages)
        if answer == "Generation Error":
            return answer, 0.0
            
        # 2. Ask for confidence
        conf_prompt = self._build_confidence_prompt(query, passages, answer)
        
        payload = {
            "model": self.model_name,
            "prompt": conf_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for more stable confidence score
                "num_predict": 10
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            conf_str = response.json().get("response", "").strip()
            
            # Try to parse float
            import re
            match = re.search(r"0\.\d+|1\.0|0|1", conf_str)
            if match:
                confidence = float(match.group())
            else:
                confidence = 0.5  # Neutral fallback if parsing fails
            
            # Clip to valid range just in case
            confidence = max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to extract confidence: {e}")
            confidence = 0.5
            
        return answer, confidence

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = OllamaGenerator()
    passages = [{"text": "Paris is the capital of France and has a population of 2.1 million."}]
    
    # Simple test (requires Ollama to be running)
    try:
        ans, conf = gen.generate_with_confidence("What is the capital of France?", passages)
        print(f"Answer: {ans}")
        print(f"Confidence: {conf}")
    except Exception as e:
        print("Could not test generator. Make sure Ollama is running.")
