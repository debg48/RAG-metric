"""
Calculate mean divergence per perturbation type for each experiment run.
Outputs a summary table as both a printed table and a CSV file.
"""

import os
import pandas as pd
import glob

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DIV_COLS = [
    "div_remove_top1",
    "div_remove_top2",
    "div_replace_top1",
    "div_replace_adversarial",
    "div_shuffle_order",
    "div_inject_irrelevant",
]

CLEAN_NAMES = {
    "div_remove_top1": "Remove Top-1",
    "div_remove_top2": "Remove Top-2",
    "div_replace_top1": "Replace Top-1",
    "div_replace_adversarial": "Replace Adversarial",
    "div_shuffle_order": "Shuffle Order",
    "div_inject_irrelevant": "Inject Irrelevant",
}


def get_run_label(run_dir: str) -> str:
    """Extract a human-readable label from the run directory name."""
    basename = os.path.basename(run_dir)
    parts = basename.split("_")
    # e.g. qwen2.5_1.5b-instruct_SQuAD_20260305_145632
    # or   tinyllama_SQuAD_20260305_200703
    if "qwen" in basename.lower():
        model = "Qwen 2.5"
        dataset = parts[2]
    elif "tinyllama" in basename.lower():
        model = "TinyLlama"
        dataset = parts[1]
    else:
        model = parts[0]
        dataset = parts[1]
    return f"{model} ({dataset})"


def compute_means():
    """Compute mean divergence per perturbation for all available runs."""
    run_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")))

    rows = []
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "raw", "full_results.csv")
        if not os.path.exists(csv_path):
            continue

        label = get_run_label(run_dir)
        df = pd.read_csv(csv_path)

        available = [c for c in DIV_COLS if c in df.columns]
        if not available:
            continue

        means = df[available].mean()
        row = {"Run": label}
        for col in DIV_COLS:
            row[CLEAN_NAMES[col]] = round(means[col], 4) if col in means else None
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    summary = compute_means()

    # Print to console
    print("\n=== Mean Divergence per Perturbation Type ===\n")
    print(summary.to_string(index=False))

    # Save as CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perturbation_means.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
