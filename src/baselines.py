import numpy as np
import logging
from typing import List, Dict, Any, Callable

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class BaselineSignals:
    """Computes baseline hallucination predictors to compare against RSI."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model for Baselines: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
    def compute_entropy_proxy(self, a_i_list: List[str]) -> float:
        """
        Since Ollama doesn't reliably expose token-level logprobs, we use the average 
        pairwise semantic distance among the perturbed answers as a proxy for Generation Entropy.
        High proxy entropy = high semantic variance = likely hallucination.
        """
        if not a_i_list or len(a_i_list) < 2:
            return 0.0
            
        self._load_model()
        texts = [t if t.strip() else " " for t in a_i_list]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Compute all pairwise cosine similarities
        n = len(embeddings)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                sim = np.clip(sim, -1.0, 1.0)
                distances.append(1.0 - sim)
                
        if not distances:
            return 0.0
            
        return float(np.mean(distances))

    def compute_doc_similarity(self, query: str, passages: List[Dict[str, Any]]) -> float:
        """
        Compute mean cosine similarity between query and retrieved passages.
        Low doc similarity might indicate a high risk of hallucination (poor retrieval).
        """
        if not query or not passages:
            return 0.0
            
        self._load_model()
        
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        p_texts = [p["text"] for p in passages]
        p_embs = self.model.encode(p_texts, normalize_embeddings=True)
        
        similarities = np.dot(p_embs, q_emb)
        similarities = np.clip(similarities, -1.0, 1.0)
        
        return float(np.mean(similarities))
        
    def get_self_confidence(self, generator, query: str, passages: List[Dict[str, Any]]) -> float:
        """
        Uses the generator's built-in self-reporting mechanism.
        """
        _, conf = generator.generate_with_confidence(query, passages)
        return conf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseline = BaselineSignals()
    
    ais = ["The capital is Paris.", "Paris is the capital of France.", "London is the capital."]
    entropy = baseline.compute_entropy_proxy(ais)
    print(f"Entropy Proxy: {entropy}")
    
    passages = [{"text": "Paris is in France."}, {"text": "London is in the UK."}]
    doc_sim = baseline.compute_doc_similarity("What is the capital?", passages)
    print(f"Doc Sim: {doc_sim}")
