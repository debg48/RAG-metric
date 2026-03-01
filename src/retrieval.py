import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Any

# Conditional imports to defer loading heavy models
try:
    import faiss
except ImportError:
    pass
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass
    
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    pass

logger = logging.getLogger(__name__)

class DenseRetriever:
    """FAISS-based dense retriever using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.index = None
        self.corpus = []
        self.corpus_texts = []
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
    def build_index(self, corpus: List[Dict[str, str]]):
        """Build FAISS index from a corpus of passages."""
        self._load_model()
        self.corpus = corpus
        self.corpus_texts = [doc["text"] for doc in corpus]
        
        logger.info(f"Encoding {len(self.corpus_texts)} passages...")
        # Normalize embeddings for inner product to be equivalent to cosine similarity
        embeddings = self.model.encode(self.corpus_texts, normalize_embeddings=True, show_progress_bar=True)
        
        d = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP (dim={d})...")
        self.index = faiss.IndexFlatIP(d)
        self.index.add(np.array(embeddings).astype("float32"))
        logger.info("Dense index built.")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top k passages for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        self._load_model()
        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.corpus):
                results.append({
                    "id": self.corpus[idx]["id"],
                    "text": self.corpus[idx]["text"],
                    "score": float(score),
                    "rank": i + 1,
                    "retriever": "dense"
                })
        return results


class SparseRetriever:
    """BM25 sparse retriever."""
    
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.corpus_texts = []
        
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
        
    def build_index(self, corpus: List[Dict[str, str]]):
        """Build BM25 index from a corpus of passages."""
        logger.info(f"Building BM25 index on {len(corpus)} passages...")
        self.corpus = corpus
        self.corpus_texts = [doc["text"] for doc in corpus]
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Sparse index built.")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top k passages for a query."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "id": self.corpus[idx]["id"],
                "text": self.corpus[idx]["text"],
                "score": float(doc_scores[idx]),
                "rank": i + 1,
                "retriever": "sparse"
            })
        return results

class HybridRetriever:
    """Combines Dense and Sparse retrievers via Reciprocal Rank Fusion."""
    
    def __init__(self, dense_model: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.dense = DenseRetriever(model_name=dense_model, device=device)
        self.sparse = SparseRetriever()
        
    def build_index(self, corpus: List[Dict[str, str]]):
        self.dense.build_index(corpus)
        self.sparse.build_index(corpus)
        
    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # Retrieve more candidates before fusion
        retrieve_k = max(top_k * 2, 100)
        dense_results = self.dense.retrieve(query, retrieve_k)
        sparse_results = self.sparse.retrieve(query, retrieve_k)
        
        # Apply Reciprocal Rank Fusion (RRF)
        scores = {}
        docs = {}
        
        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            if doc_id not in scores:
                scores[doc_id] = 0
                docs[doc_id] = {"id": doc_id, "text": res["text"]}
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        for rank, res in enumerate(sparse_results):
            doc_id = res["id"]
            if doc_id not in scores:
                scores[doc_id] = 0
                docs[doc_id] = {"id": doc_id, "text": res["text"]}
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        # Sort and return top_k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for i, doc_id in enumerate(sorted_ids[:top_k]):
            res = docs[doc_id]
            res["score"] = float(scores[doc_id])
            res["rank"] = i + 1
            res["retriever"] = "hybrid"
            results.append(res)
            
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    corpus = [
        {"id": "doc_0", "text": "The apple M4 chip features advanced CPU capabilities."},
        {"id": "doc_1", "text": "Hallucination in RAG is an active research area."},
        {"id": "doc_2", "text": "Retrieval Augmented Generation combines search and LLMs."}
    ]
    
    retriever = DenseRetriever()
    retriever.build_index(corpus)
    res = retriever.retrieve("What is RAG?", top_k=2)
    print("Dense Top 2:", res)
    
    sparse = SparseRetriever()
    sparse.build_index(corpus)
    res = sparse.retrieve("M4 CPU capabilities", top_k=2)
    print("Sparse Top 2:", res)
