import re
import string
import collections
from typing import List, Callable
import logging
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
        
    def white_space_fix(text):
        return ' '.join(text.split())
        
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
        
    def lower(text):
        return text.lower()
        
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> List[str]:
    """Get tokens from normalized string."""
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold: str, a_pred: str) -> int:
    """Exact match of normalized answers."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute word-level F1 score over normalized answers."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
        
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rouge_l(a_gold: str, a_pred: str) -> float:
    """Compute ROUGE-L f-measure for long medical text."""
    if not a_gold.strip() or not a_pred.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(a_gold, a_pred)
    return scores['rougeL'].fmeasure

def metric_max_over_ground_truths(metric_fn: Callable, prediction: str, ground_truths: List[str]) -> float:
    """Compute metric against all ground truths, take the max."""
    if not ground_truths:
        return metric_fn("", prediction)
        
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(ground_truth, prediction)
        scores_for_ground_truths.append(score)
        
    return max(scores_for_ground_truths)

def compute_evidence_overlap(prediction: str, evidence_texts: List[str]) -> float:
    """
    Compute word overlap between generated answer and the retrieved evidence.
    Simple ratio of generated tokens that appear in the context.
    """
    pred_toks = get_tokens(prediction)
    if not pred_toks:
        return 0.0
        
    # Combine all evidence into one set of tokens
    combined_evidence = set()
    for text in evidence_texts:
        combined_evidence.update(get_tokens(text))
        
    overlap_count = sum(1 for tok in pred_toks if tok in combined_evidence)
    overlap_ratio = overlap_count / len(pred_toks)
    
    return overlap_ratio

def label_hallucination(prediction: str, 
                        ground_truths: List[str], 
                        evidence_texts: List[str],
                        em_threshold: float = 0.0,
                        f1_threshold: float = 0.3,
                        overlap_threshold: float = 0.3,
                        dataset_type: str = "squad2") -> bool:
    """
    Determine if the answer is a hallucination.
    Operational definition:
    1. Not correct (F1 or ROUGE < threshold)
    2. AND potentially not supported by text (evidence overlap < threshold)
    Note: A 'I don't know' answer might have low score and low overlap, but is it a hallucination?
    Usually we consider 'I don't know' as a safe refusal, not a hallucination.
    """
    pred_norm = normalize_answer(prediction)
    
    # Handle safe refusals
    refusal_phrases = ["i dont know", "not in the context", "cannot answer", "no information"]
    if any(phrase in pred_norm for phrase in refusal_phrases):
        # Refusals are not hallucinations, they are safe abstain actions.
        return False
        
    best_em = metric_max_over_ground_truths(compute_exact, prediction, ground_truths)
    
    scoring_fn = compute_rouge_l if dataset_type in ["medquad", "pubmedqa"] else compute_f1
    best_score = metric_max_over_ground_truths(scoring_fn, prediction, ground_truths)
    
    overlap = compute_evidence_overlap(prediction, evidence_texts)
    
    # A hallucination occurs when the model gives a wrong answer AND makes things up
    # However, for pure generation evaluation, often just Score < threshold is considered hallucination
    # if it's not a refusal.
    # We use a combined definition here.
    
    is_wrong = best_score < f1_threshold
    
    # Optional strict definition: it's a hallucination if it's wrong AND has low overlap with text
    # (meaning it pulled external knowledge)
    # is_unsupported = overlap < overlap_threshold
    # is_hallucinated = is_wrong and is_unsupported
    
    # For now, let's stick to the simpler SQuAD benchmark definition:
    # If it's wrong (and not a refusal), it's a hallucination (making up incorrect info).
    is_hallucinated = is_wrong
    
    return is_hallucinated

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pred = "Paris is the capital."
    gts = ["Paris", "France's capital is Paris"]
    evidences = ["Paris, the capital city of France, has a population of 2.1 million."]
    
    f1 = metric_max_over_ground_truths(compute_f1, pred, gts)
    ol = compute_evidence_overlap(pred, evidences)
    lbl = label_hallucination(pred, gts, evidences)
    
    print(f"F1: {f1}, Overlap: {ol}, Hallucinated: {lbl}")
