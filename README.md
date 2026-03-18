# Retrieval Sensitivity Index (RSI) Experimental Pipeline

This repository contains the full experimental code to evaluate and validate the **Retrieval Sensitivity Index (RSI)**, as presented in the paper *"Retrieval Sensitivity Index - A Sensitivity-Based Analysis of Hallucination in Medical RAG Systems for Small Language Models"*. RSI is a scalar, perturbation-based metric designed to measure instance-level retrieval-conditioned instability in RAG (Retrieval-Augmented Generation) systems, thereby predicting the probability of hallucination, especially for Small Language Models (SLMs).

The experimental setup is explicitly designed to be runnable on consumer hardware (e.g., Apple M4 with limited memory) using local language models via Ollama.

## Core Features

- **Models Evaluated:** Qwen2.5-1.5B-Instruct (`qwen2.5:1.5b-instruct` - **Recommended**) and TinyLlama-1.1B (`tinyllama`). Support for any other ultra-lightweight Ollama model.
- **Datasets Supported:** SQuAD v2 (Short-span factoid QA), MedQuAD (Long-form medical explanations), and PubMedQA (Evidence-based biomedical reasoning).
- **Dual-Stream Retrieval:** FAISS Dense Retrieval (`all-MiniLM-L6-v2`) and BM25 Sparse Retrieval, combined using Reciprocal Rank Fusion (RRF).
- **Perturbation Strategies (6):** 
  - *Rank-based:* Remove Top-1, Remove Top-2, Replace Top-1 (with rank $k+1$).
  - *Noise-based:* Replace Top-1 with Adversarial Distractor, Shuffle Document Order, Inject Irrelevant Document.
- **Resiliency:** Built-in **Checkpoint/Resume** support. Interrupted runs can be resumed from the same folder without losing progress.
- **Metrics Evaluated:** 
  - *RSI Variants:* Standard RSI, Entropy-Normalized RSI, Evidence-Grounded RSI, and Rank-Weighted RSI.
  - *Uncertainty Baselines:* Token Entropy (Proxy via embedding variance), Self-Reported Confidence, and Query-Document Similarity.
  - *Answer Quality:* Exact Match, unigram F1 (SQuAD), and ROUGE-L F-measure (Medical datasets).
- **Statistical Analytics:** Outputs comprehensive CSV tables and 13 generated publication-ready plots (Violin plots, ROC/AUC Curves, Cost-Benefit Adaptive Thresholding, etc.).

## Prerequisites

1. Install [Ollama](https://ollama.com/)
2. Pull the required SLMs used in the paper:

   ```bash
   ollama pull qwen2.5:1.5b-instruct
   ollama pull tinyllama
   ```

3. Python 3.9+ with `pip`

## Setup

```bash
# Clone or navigate to the repository
cd RAG-metric

# Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running the Pipeline

The primary orchestrator is `run_pipeline.py`. It automates dataset downloading, FAISS indexing, LLM generation, statistical evaluations, and plot generation. All results are stored in the `results/` folder.

See `commands.txt` for specific examples of full-scale experimental runs.

### Example Quick Test (Smoke Test)

```bash
python run_pipeline.py --sample-size 5
```

## Repository Structure

```text
├── config.yaml             # Hyperparameters & settings
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Main orchestrator script
├── commands.txt            # End-to-end execution commands
├── ablation/               # Contains scripts for ablation studies
├── plots/
│   └── plot_results.py     # Source for the 13 generated paper figures
└── src/                    # Core modules
    ├── adaptive.py         # Adaptive re-retrieval policy code
    ├── baselines.py        # Entropy proxy, Doc Sim, and Self-confidence
    ├── dataset.py          # SQuAD & Medical QA loaders
    ├── evaluation.py       # AUC, t-tests, bootstrap CI
    ├── generator.py        # Ollama API wrappers
    ├── labeling.py         # EM/F1/ROUGE-L hallucination heuristics
    ├── perturbation.py     # 6 implementation strategies (includes adversarial & top-2)
    ├── retrieval.py        # FAISS / BM25 class wrappers
    └── rsi.py              # RSI variants logic
```

## Citation

If you use this code for your research, please refer to the paper:
**"Retrieval Sensitivity Index - A Sensitivity-Based Analysis of Hallucination in Medical RAG Systems for Small Language Models"**
