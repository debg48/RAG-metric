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
            
    def compute_rsi(self, a0: str, a_i_list: List[str], 
                    entropy: float = 0.0, 
                    a0_evidence_overlap: float = 0.0, 
                    ai_evidence_overlaps: List[float] = None) -> Dict[str, Any]:
        """
        Compute divergence metrics between the baseline answer and perturbed answers.
        Now includes Entropy-Normalized RSI, Evidence-Grounded RSI over overlap drops,
        and Rank-Weighted RSI.
        """
        if ai_evidence_overlaps is None:
            ai_evidence_overlaps = [0.0] * len(a_i_list)
            
        if not a0 or not a_i_list:
            logger.warning("Empty answers provided to RSI computation.")
            return {
                "rsi_mean": 0.0,
                "rsi_variance": 0.0,
                "rsi_max": 0.0,
                "rsi_norm": 0.0,
                "rsi_evidence": 0.0,
                "rsi_weighted": 0.0,
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
        
        # 1. Standard RSI 
        rsi_mean = float(np.mean(divergences))
        
        # 2. Entropy-Normalized RSI (isolate retrieval instability from token uncertainty)
        EPSILON = 1e-5
        rsi_norm = rsi_mean / (entropy + EPSILON)
        
        # 3. Evidence-Grounded RSI (Claim-Evidence Drift)
        # We want to measure how much the Answer-Evidence alignment DROPS under perturbation.
        # If mean perturbed overlap is LOWER than baseline overlap, this captures grounding collapse.
        mean_ai_overlap = float(np.mean(ai_evidence_overlaps))
        rsi_evidence = max(0.0, a0_evidence_overlap - mean_ai_overlap)
        
        # 4. Rank-Weighted Expected RSI
        # SLMs heavily depend on top docs. Hardcode weights based on the 6 standard perturbations:
        # [remove_top1, remove_top2, replace_top1, replace_adversarial, shuffle, inject]
        # We highly weight the first 4 (rank-1/2 changes) and lower weight the context noise.
        weights = np.ones(len(divergences))
        if len(divergences) == 6:
             weights = np.array([0.25, 0.25, 0.20, 0.20, 0.05, 0.05])
        elif len(divergences) > 0:
             # Fallback: equal weighting if perturbations don't match exactly 6
             weights = np.ones(len(divergences)) / len(divergences)
             
        rsi_weighted = float(np.sum(divergences * weights))
        
        result = {
            "rsi_mean": rsi_mean,
            "rsi_variance": float(np.var(divergences)),
            "rsi_max": float(np.max(divergences)),
            "rsi_norm": float(rsi_norm),
            "rsi_evidence": float(rsi_evidence),
            "rsi_weighted": float(rsi_weighted),
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
