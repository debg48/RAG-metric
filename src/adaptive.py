import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def adaptive_policy(query: str, rsi_score: float, threshold: float, retriever, generator, base_k: int = 5) -> Dict[str, Any]:
    """
    If RSI > threshold (high instability / hallucination risk),
    re-retrieve with a larger k (e.g. 2*k) and regenerate the answer.
    """
    if rsi_score <= threshold:
        # Trust the baseline, no action needed
        return {
            "triggered": False,
            "new_answer": None,
            "cost_api_calls": 0
        }
        
    logger.info(f"RSI={rsi_score:.3f} > {threshold:.3f}. Triggering adaptive re-retrieval.")
    
    # Re-retrieve with expanded scope
    new_passages = retriever.retrieve(query, top_k=base_k * 2)
    
    # Or, ideally, we build a "chain-of-thought" prompt here since we know it's hard,
    # but for simplicity we just give it more context.
    new_answer = generator.generate(query, new_passages)
    
    return {
        "triggered": True,
        "new_answer": new_answer,
        "cost_api_calls": 1
    }

def evaluate_adaptive_policy(results_baseline: List[Dict[str, Any]], 
                             results_adaptive: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compare outcomes before and after applying the adaptive policy.
    """
    if not results_baseline or not results_adaptive:
        return {}
        
    total = len(results_baseline)
    triggered_count = sum(1 for r in results_adaptive if r["triggered"])
    
    old_hallucinations = sum(1 for r in results_baseline if r["is_hallucinated"])
    new_hallucinations = sum(1 for r in results_adaptive if r["is_hallucinated"])
    
    old_f1 = sum(r["f1_score"] for r in results_baseline) / total
    new_f1 = sum(r["f1_score"] for r in results_adaptive) / total
    
    return {
        "total_queries": total,
        "triggered_queries": triggered_count,
        "extra_compute_pct": (triggered_count / total) * 100,
        "base_hallucination_rate": (old_hallucinations / total) * 100,
        "new_hallucination_rate": (new_hallucinations / total) * 100,
        "hallucination_reduction_pct": ((old_hallucinations - new_hallucinations) / max(1, old_hallucinations)) * 100,
        "base_f1": float(old_f1),
        "new_f1": float(new_f1),
        "f1_gain": float(new_f1 - old_f1)
    }

def sweep_thresholds(baseline_data: pd.DataFrame, thresholds: List[float]) -> List[Dict[str, float]]:
    """Simulate the cost-benefit of different thresholds (for Fig 8)."""
    # This requires running the adaptive regenerate on all triggered instances...
    # For a post-hoc sweep without re-running LLM calls, we would need to have pre-computed 
    # the "expanded-k" answer for all queries.
    # We will implement this logic in `run_pipeline.py` by pre-computing a fallback answer.
    pass
