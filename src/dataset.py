import json
import logging
from typing import List, Dict, Any
from datasets import load_dataset
import numpy as np

logger = logging.getLogger(__name__)

def load_squad_subset(sample_size: int = 450, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load a subset of the SQuAD v2 validation set.
    """
    logger.info(f"Loading {sample_size} samples from SQuAD v2 validation set...")
    
    dataset = load_dataset("squad_v2", split="validation")
    answerable = dataset.filter(lambda example: len(example["answers"]["text"]) > 0)
    answerable = answerable.shuffle(seed=seed)
    
    if sample_size > 0 and sample_size < len(answerable):
        sampled = answerable.select(range(sample_size))
    else:
        sampled = answerable
        
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

def load_hotpotqa_subset(sample_size: int = 350, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load a subset of HotpotQA validation set.
    """
    logger.info(f"Loading {sample_size} samples from HotpotQA validation set...")
    
    # HotpotQA has a 'distractor' split often used as validation
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    dataset = dataset.shuffle(seed=seed)
    
    if sample_size > 0 and sample_size < len(dataset):
        sampled = dataset.select(range(sample_size))
    else:
        sampled = dataset
        
    logger.info(f"Selected {len(sampled)} HotpotQA questions.")
    
    results = []
    for item in sampled:
        # HotpotQA context is a dict with keys 'title' (list of strings) and 'sentences' (list of list of strings)
        titles = item["context"]["title"]
        sentences_lists = item["context"]["sentences"]
        
        flat_context = " ".join([
            f"{title}: {' '.join(sents)}" 
            for title, sents in zip(titles, sentences_lists)
        ])
        
        results.append({
            "id": f"hotpot_{item['id']}",
            "question": item["question"],
            "context": flat_context,
            "answers": [item["answer"]], # HotpotQA provides a single string answer
            "dataset": "hotpot_qa"
        })
        
    return results

def build_passage_corpus(squad_data: List[Dict[str, Any]] = None, hotpot_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Build a unified passage corpus from the loaded data subsets.
    """
    logger.info("Building unified passage corpus from selected samples...")
    unique_contexts = set()
    
    if squad_data:
        for item in squad_data:
            unique_contexts.add(item["context"])
            
    if hotpot_data:
        for item in hotpot_data:
            unique_contexts.add(item["context"])
        
    corpus = [{"id": f"doc_{i}", "text": text} for i, text in enumerate(unique_contexts)]
    logger.info(f"Built corpus with {len(corpus)} unique passages.")
    
    return corpus

def load_mixed_datasets(squad_size: int = 450, hotpot_size: int = 350, seed: int = 42) -> tuple:
    squad_data = load_squad_subset(squad_size, seed) if squad_size > 0 else []
    hotpot_data = load_hotpotqa_subset(hotpot_size, seed) if hotpot_size > 0 else []
    
    all_queries = squad_data + hotpot_data
    corpus = build_passage_corpus(squad_data, hotpot_data)
    
    return all_queries, corpus

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    queries, corpus = load_mixed_datasets(squad_size=2, hotpot_size=2)
    print(f"Total queries: {len(queries)}")
    print(f"Total corpus docs: {len(corpus)}")
