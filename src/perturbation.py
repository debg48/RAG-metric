import random
import copy
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

def remove_top1(query: str, passages: List[Dict[str, Any]], retriever: Any = None) -> List[Dict[str, Any]]:
    """Remove the rank-1 document, returning ranks 2 through k."""
    if not passages or len(passages) <= 1:
        return passages
    return passages[1:]

def replace_top1(query: str, passages: List[Dict[str, Any]], retriever: Any) -> List[Dict[str, Any]]:
    """Replace rank-1 document with rank-(k+1) from the retriever."""
    if not passages or not retriever:
        return passages
        
    k = len(passages)
    # Fetch k+1 docs to get the next one down
    extended_results = retriever.retrieve(query, top_k=k+1)
    
    if len(extended_results) > k:
        new_passages = copy.deepcopy(passages)
        new_passages[0] = extended_results[k]  # Replace index 0 with rank k+1
        return new_passages
    else:
        return passages

def shuffle_order(query: str, passages: List[Dict[str, Any]], retriever: Any = None) -> List[Dict[str, Any]]:
    """Randomly permute the order of retrieved passages. Fixed seed per query for reproducibility."""
    if not passages:
        return passages
        
    # Use query hash as seed so it's deterministic per query but random across queries
    seed = hash(query) % (2**32 - 1)
    rng = random.Random(seed)
    
    shuffled = copy.deepcopy(passages)
    rng.shuffle(shuffled)
    return shuffled

def inject_irrelevant(query: str, passages: List[Dict[str, Any]], retriever: Any) -> List[Dict[str, Any]]:
    """Inject a random irrelevant document at the top.
    We approximate 'irrelevant' by fetching from the bottom of the corpus or randomly if retriever allows,
    but here we just pull a random doc from the corpus that isn't in top-100.
    """
    if not passages or not retriever or not hasattr(retriever, 'corpus'):
        return passages
        
    # Get top 100 to avoid them
    top_100 = retriever.retrieve(query, top_k=100)
    top_100_ids = {p["id"] for p in top_100}
    
    # Find a random doc not in top 100
    corpus = retriever.corpus
    seed = hash(query) % (2**32 - 1)
    rng = random.Random(seed)
    
    attempts = 0
    irrelevant_doc = None
    while attempts < 20:
        idx = rng.randint(0, len(corpus) - 1)
        doc = corpus[idx]
        if doc["id"] not in top_100_ids:
            # We found an irrelevant one
            irrelevant_doc = {
                "id": doc["id"],
                "text": str(doc["text"]),
                "score": 0.0,
                "rank": 0,
                "retriever": "injected"
            }
            break
        attempts += 1
        
    if irrelevant_doc:
        new_passages = copy.deepcopy(passages)
        # Prepend irrelevant doc
        new_passages.insert(0, irrelevant_doc)
        # Keep k+1 passage
        return new_passages 
    else:
        return passages

def remove_top2(query: str, passages: List[Dict[str, Any]], retriever: Any = None) -> List[Dict[str, Any]]:
    """Remove the rank-1 and rank-2 documents."""
    if not passages or len(passages) <= 2:
        return passages[-1:] if passages else [] # Leave at least something, or empty
    return passages[2:]

def replace_with_adversarial(query: str, passages: List[Dict[str, Any]], retriever: Any) -> List[Dict[str, Any]]:
    """Replace rank-1 document with a highly contradictory/adversarial distractor if possible."""
    # We approximate this by finding a document that is semantically dissimilar to the top-1 doc
    # but still somewhat related to the query, OR just a hard negative from the bottom of top-100.
    if not passages or not retriever or not hasattr(retriever, 'corpus'):
        return passages
        
    top_100 = retriever.retrieve(query, top_k=100)
    if len(top_100) > 50:
        # Take a hard negative from rank 50-100
        seed = hash(query) % (2**32 - 1)
        rng = random.Random(seed)
        distractor = top_100[rng.randint(50, min(99, len(top_100)-1))]
        
        new_passages = copy.deepcopy(passages)
        new_passages[0] = distractor
        return new_passages
    return passages

# Registry of perturbation names to functions
PERTURBATION_REGISTRY: Dict[str, Callable] = {
    "remove_top1": remove_top1,
    "remove_top2": remove_top2,
    "replace_top1": replace_top1,
    "replace_adversarial": replace_with_adversarial,
    "shuffle_order": shuffle_order,
    "inject_irrelevant": inject_irrelevant
}

def apply_perturbations(query: str, passages: List[Dict[str, Any]], retriever: Any, active_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Apply all active perturbations to a set of passages."""
    results = {}
    for p_type in active_types:
        if p_type in PERTURBATION_REGISTRY:
            results[p_type] = PERTURBATION_REGISTRY[p_type](query, passages, retriever)
        else:
            logger.warning(f"Unknown perturbation type requested: {p_type}")
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.retrieval import DenseRetriever
    corpus = [{"id": f"doc_{i}", "text": f"Text {i}"} for i in range(100)]
    r = DenseRetriever()
    r.build_index(corpus)
    
    query = "test query"
    base_docs = r.retrieve(query, top_k=3)
    print("Base:", [d["id"] for d in base_docs])
    
    print("Remove top-1:", [d["id"] for d in remove_top1(query, base_docs)])
    print("Replace top-1:", [d["id"] for d in replace_top1(query, base_docs, r)])
    print("Shuffle:", [d["id"] for d in shuffle_order(query, base_docs)])
    print("Inject:", [d["id"] for d in inject_irrelevant(query, base_docs, r)])
