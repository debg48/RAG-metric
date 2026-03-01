# Retrieval Sensitivity Index (RSI) Experimental Pipeline

This repository contains the full experimental code to evaluate and validate the **Retrieval Sensitivity Index (RSI)**. RSI is a scalar metric designed to measure instance-level retrieval-conditioned instability in RAG (Retrieval-Augmented Generation) systems, thereby predicting the probability of hallucination.

The experimental setup is explicitly designed to be runnable on consumer hardware (e.g., Apple M4 with limited memory) using local language models via Ollama.

## Core Features

- **Models Supported:** Qwen3 4B (`goekdenizguelmez/JOSIEFIED-Qwen3:4b`), Llama 3 8B Instruct Q4 (`llama3:8b-instruct-q4_0`), or any other Ollama-compatible model.
- **Datasets Supported:** SQuAD v2 (Extractive/Abstractive QA) and HotpotQA (Multi-hop QA).
- **Retrievers:** FAISS Dense Retrieval (`all-MiniLM-L6-v2`) and BM25 Sparse Retrieval.
- **Perturbation Strategies (4):** Rank-1 Removal, Rank-1 Replacement with Rank $k+1$, Document Order Shuffling, and Irrelevant Document Injection.
- **Metrics Evaluated:** RSI (Mean, Variance, Max), Token Entropy (Proxy via embedding variance), Calibrated Self-Confidence, Document Similarity, Exact Match (EM), and Word-level F1.
- **Statistical Analytics:** Outputs comprehensive CSV tables and 13 generated publication-ready plots (Violin plots, ROC/AUC Curves, Cost-Benefit Adaptive Thresholding, etc.).

## Prerequisites

1. Install [Ollama](https://ollama.com/)
2. Pull required models:

   ```bash
   ollama run goekdenizguelmez/JOSIEFIED-Qwen3:4b
   ollama run llama3:8b-instruct-q4_0
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
    ├── dataset.py          # SQuAD & HotpotQA loaders
    ├── evaluation.py       # AUC, t-tests, bootstrap CI
    ├── generator.py        # Ollama API wrappers
    ├── labeling.py         # EM/F1 hallucination heuristics
    ├── perturbation.py     # 4 implementation strategies
    ├── retrieval.py        # FAISS / BM25 class wrappers
    └── rsi.py              # Divergence / Cosine Similarity logic
```
