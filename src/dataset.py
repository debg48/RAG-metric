import json
import logging
from typing import List, Dict, Any
from datasets import load_dataset
import numpy as np

logger = logging.getLogger(__name__)

def load_squad_subset(sample_size: int = 400, seed: int = 42) -> List[Dict]:
    """Load a subset of SQuAD v2 validation set."""
    logger.info(f"Loading SQuAD v2 validation set...")
    
    dataset = load_dataset("squad_v2", split="validation")
    
    # -1 means full dataset
    if sample_size == -1 or sample_size >= len(dataset):
        logger.info(f"Using FULL SQuAD v2 dataset ({len(dataset)} samples).")
        sampled = dataset
    else:
        logger.info(f"Sampling {sample_size} samples.")
        sampled = dataset.shuffle(seed=seed).select(range(sample_size))
        
    logger.info(f"Selected {len(sampled)} SQuAD questions.")
    
    results = []
    for item in sampled:
        results.append({
            "id": f"squad_{item['id']}",
            "question": item["question"],
            "context": item["context"], # For building corpus
            "answers": item["answers"]["text"],
            "dataset": "squad2"
        })
        
    return results

def load_medquad_subset(sample_size: int = 350, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load a subset of MedQuAD dataset.
    """
    logger.info(f"Loading {sample_size} samples from MedQuAD dataset...")
    
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    dataset = dataset.shuffle(seed=seed)
    
    if sample_size > 0 and sample_size < len(dataset):
        sampled = dataset.select(range(sample_size))
    else:
        sampled = dataset
        
    logger.info(f"Selected {len(sampled)} MedQuAD questions.")
    
    results = []
    for item in sampled:
        results.append({
            "id": f"medquad_{item['qtype']}_{hash(item['Question'])}",
            "question": item["Question"],
            "context": item["Answer"], # MedQuAD's answer is often explanatory/contextual
            "answers": [item["Answer"]],
            "dataset": "medquad"
        })
        
    return results

def load_pubmedqa_subset(sample_size: int = 350, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load a subset of PubMedQA labeled dataset.
    """
    logger.info(f"Loading {sample_size} samples from PubMedQA dataset...")
    
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    dataset = dataset.shuffle(seed=seed)
    
    if sample_size > 0 and sample_size < len(dataset):
        sampled = dataset.select(range(sample_size))
    else:
        sampled = dataset
        
    logger.info(f"Selected {len(sampled)} PubMedQA questions.")
    
    results = []
    for i, item in enumerate(sampled):
        # PubMedQA provides 'CONTEXTS' as a list of strings
        context = " ".join(item["CONTEXTS"]) if isinstance(item["CONTEXTS"], list) else str(item["CONTEXTS"])
        
        results.append({
            "id": f"pubmedqa_{i}",
            "question": item["QUESTION"],
            "context": context,
            "answers": [item["LONG_ANSWER"]], 
            "dataset": "pubmedqa"
        })
        
    return results

def build_passage_corpus(squad_data: List[Dict[str, Any]] = None, medquad_data: List[Dict[str, Any]] = None, pubmedqa_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Build a unified passage corpus from the loaded data subsets.
    """
    logger.info("Building unified passage corpus from selected samples...")
    unique_contexts = set()
    
    if squad_data:
        for item in squad_data:
            unique_contexts.add(item["context"])
            
    if medquad_data:
        for item in medquad_data:
            unique_contexts.add(item["context"])
            
    if pubmedqa_data:
        for item in pubmedqa_data:
            unique_contexts.add(item["context"])
        
    corpus = [{"id": f"doc_{i}", "text": text} for i, text in enumerate(unique_contexts)]
    logger.info(f"Built corpus with {len(corpus)} unique passages.")
    
    return corpus

def load_mixed_datasets(squad_size: int = 450, medquad_size: int = 175, pubmedqa_size: int = 175, seed: int = 42) -> tuple:
    squad_data = load_squad_subset(squad_size, seed) if squad_size > 0 else []
    medquad_data = load_medquad_subset(medquad_size, seed) if medquad_size > 0 else []
    pubmedqa_data = load_pubmedqa_subset(pubmedqa_size, seed) if pubmedqa_size > 0 else []
    
    all_queries = squad_data + medquad_data + pubmedqa_data
    corpus = build_passage_corpus(squad_data, medquad_data, pubmedqa_data)
    
    return all_queries, corpus

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    queries, corpus = load_mixed_datasets(squad_size=2, medquad_size=2, pubmedqa_size=2)
    print(f"Total queries: {len(queries)}")
    print(f"Total corpus docs: {len(corpus)}")
