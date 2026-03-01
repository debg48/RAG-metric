import numpy as np
import logging
from typing import List, Dict, Any, Union

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class RSIComputer:
    """Computes Retrieval Sensitivity Index (RSI) between baseline answer A_0 and perturbed answers A_i."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model for RSI: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
    def compute_rsi(self, a0: str, a_i_list: List[str]) -> Dict[str, Any]:
        """
        Compute divergence metrics between the baseline answer and perturbed answers.
        Uses cosine similarity. Divergence = 1 - cosine_sim
        """
        if not a0 or not a_i_list:
            logger.warning("Empty answers provided to RSI computation.")
            return {
                "rsi_mean": 0.0,
                "rsi_variance": 0.0,
                "rsi_max": 0.0,
                "divergences": []
            }
            
        self._load_model()
        
        # Encode all answers together for efficiency
        texts = [a0] + a_i_list
        # Replace empty strings with a space to avoid completely zero embeddings leading to NaNs
        texts = [t if t.strip() else " " for t in texts]
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        a0_emb = embeddings[0]
        a_i_embs = embeddings[1:]
        
        # Compute cosine similarities
        # Since embeddings are L2 normalized, dot product is cosine similarity
        similarities = np.dot(a_i_embs, a0_emb)
        
        # Clip to [-1, 1] to avoid float precision weirdness
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Divergence is 1 - similarity (range 0 to 2)
        divergences = 1.0 - similarities
        
        result = {
            "rsi_mean": float(np.mean(divergences)),
            "rsi_variance": float(np.var(divergences)),
            "rsi_max": float(np.max(divergences)),
            "divergences": [float(d) for d in divergences]
        }
        
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    computer = RSIComputer()
    
    a0 = "Pairs is the capital of France and its population is 2.1 million."
    a_i_list = [
        "The capital of France is Paris, home to 2.1M people.",  # highly similar
        "I don't know based on the context.",                    # divergent
        "France's capital is London.",                           # divergent (hallucinated)
        "Paris is the capital of France and its population is 2.1 million." # identical
    ]
    
    stats = computer.compute_rsi(a0, a_i_list)
    print(f"RSI Mean: {stats['rsi_mean']:.4f}")
    print(f"RSI Max: {stats['rsi_max']:.4f}")
    print(f"Divergences: {stats['divergences']}")
