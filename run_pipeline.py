import os
import random
import yaml
import json
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, List
import time

# Local imports
from src.dataset import load_mixed_datasets, build_passage_corpus
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.generator import OllamaGenerator
from src.perturbation import apply_perturbations
from src.rsi import RSIComputer
from src.labeling import label_hallucination, compute_exact, metric_max_over_ground_truths, compute_f1, compute_evidence_overlap
from src.baselines import BaselineSignals
from src.evaluation import compare_groups, compute_correlations, compute_roc_auc, bootstrap_auc
from src.adaptive import adaptive_policy
from plots.plot_results import PlotGenerator

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "pipeline.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("pipeline")

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(config: Dict[str, Any], logger: logging.Logger):
    logger.info("Initializing Pipeline Constraints (Apple M4 / CPU)")
    
    # Setup paths
    paths = config["paths"]
    os.makedirs(paths["raw_logs_dir"], exist_ok=True)
    os.makedirs(paths["tables_dir"], exist_ok=True)
    os.makedirs(paths["figures_dir"], exist_ok=True)
    
    # 1. Load Dataset
    ds_conf = config["dataset"]
    sample_size = ds_conf.get("sample_size", 450)
    dataset_flag = ds_conf.get("dataset_flag", "squad")  # squad | hotpot | both
    
    if dataset_flag == "squad":
        squad_size, medquad_size, pubmedqa_size = sample_size, 0, 0
    elif dataset_flag == "medical":
        squad_size, medquad_size, pubmedqa_size = 0, int(sample_size * 0.5), sample_size - int(sample_size * 0.5)
    elif dataset_flag == "both":
        squad_size = int(sample_size * 0.56)   # ~56% SQuAD
        medical_size = sample_size - squad_size
        medquad_size = int(medical_size * 0.5)
        pubmedqa_size = medical_size - medquad_size
    elif dataset_flag == "medquad":
        squad_size, medquad_size, pubmedqa_size = 0, sample_size, 0
    elif dataset_flag == "pubmedqa":
        squad_size, medquad_size, pubmedqa_size = 0, 0, sample_size
    else:
        raise ValueError(f"Unknown dataset flag: {dataset_flag}. Use 'squad', 'medical', 'both', 'medquad', or 'pubmedqa'.")
    
    logger.info(f"1. Loading dataset(s): {dataset_flag} (squad={squad_size}, medquad={medquad_size}, pubmedqa={pubmedqa_size})")
    queries, corpus = load_mixed_datasets(squad_size, medquad_size, pubmedqa_size, ds_conf.get("random_seed", 42))
    
    # 2. Build Retriever
    logger.info("2. Building Retriever...")
    ret_conf = config["retrieval"]
    if ret_conf["use_bm25"]:
        retriever = HybridRetriever(dense_model=ret_conf["embedding_model"], device="cpu")
    else:
        retriever = DenseRetriever(model_name=ret_conf["embedding_model"], device="cpu")
    retriever.build_index(corpus)
    
    # 3. Initializing Generator and Metric Modules
    logger.info("3. Initializing Generator and Metrics...")
    gen_conf = config["generator"]
    generator = OllamaGenerator(
        model_name=gen_conf["model_name"],
        temperature=gen_conf["temperature"],
        max_tokens=gen_conf["max_tokens"]
    )
    
    rsi_computer = RSIComputer(model_name=ret_conf["embedding_model"])
    baselines = BaselineSignals(model_name=ret_conf["embedding_model"])
    
    # Determine active perturbations
    pert_conf = config["perturbation"]
    active_perturbations = [k for k, v in pert_conf.items() if v]
    
    # Result collection
    checkpoint_path = os.path.join(paths["raw_logs_dir"], "results_partial.jsonl")
    results = []
    processed_ids = set()
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Checking for existing progress in {checkpoint_path}...")
        with open(checkpoint_path, "r") as f:
            for line in f:
                res = json.loads(line)
                results.append(res)
                processed_ids.add(res["qid"])
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} queries already processed.")

    logger.info(f"4. Processing {len(queries)} queries...")
    for item in tqdm(queries, desc="Pipeline Run"):
        qid = item["id"]
        if qid in processed_ids:
            continue
            
        q_text = item["question"]
        gts = item["answers"]
        
        # Retrieval
        base_passages = retriever.retrieve(q_text, top_k=ret_conf["top_k"])
        
        # Generation A_0
        start_time = time.time()
        a_0, conf = generator.generate_with_confidence(q_text, base_passages)
        gen_time = time.time() - start_time
        
        # Compute baseline evidence overlap
        a0_evidence_overlap = compute_evidence_overlap(a_0, [p["text"] for p in base_passages])
        
        # Pre-compute for adaptive policy cost sweep (k*2)
        expanded_passages = retriever.retrieve(q_text, top_k=ret_conf["top_k"] * 2)
        a_expanded = generator.generate(q_text, expanded_passages)
        
        # Perturbations A_i
        perturbed_contexts = apply_perturbations(q_text, base_passages, retriever, active_perturbations)
        
        a_i_dict = {}
        ai_evidence_overlaps = []
        for p_type, p_passages in perturbed_contexts.items():
            a_i = generator.generate(q_text, p_passages)
            a_i_dict[p_type] = a_i
            
            # Compute perturbed evidence overlap
            ai_overlap = compute_evidence_overlap(a_i, [p["text"] for p in p_passages])
            ai_evidence_overlaps.append(ai_overlap)
            
        a_i_list = list(a_i_dict.values())
            
        # Metrics Computation
        rsi_stats = rsi_computer.compute_rsi(
            a_0, a_i_list, 
            a0_evidence_overlap=a0_evidence_overlap, 
            ai_evidence_overlaps=ai_evidence_overlaps
        )
        
        # Store individual divergence per perturbation
        for i, p_type in enumerate(a_i_dict.keys()):
            if i < len(rsi_stats["divergences"]):
                rsi_stats[f"div_{p_type}"] = rsi_stats["divergences"][i]
            
        entropy = baselines.compute_entropy_proxy(a_i_list)
        doc_sim = baselines.compute_doc_similarity(q_text, base_passages)
        
        # Quality Labeling
        evidence_texts = [p["text"] for p in base_passages]
        em = metric_max_over_ground_truths(compute_exact, a_0, gts)
        f1 = metric_max_over_ground_truths(compute_f1, a_0, gts)
        
        # Expanded quality (for adaptive policy)
        expanded_em = metric_max_over_ground_truths(compute_exact, a_expanded, gts)
        expanded_f1 = metric_max_over_ground_truths(compute_f1, a_expanded, gts)
        
        # Label hallucination based on config thresholds
        lbl_conf = config["labeling"]
        is_hallucinated = label_hallucination(
            a_0, gts, evidence_texts,
            em_threshold=lbl_conf["em_threshold"],
            f1_threshold=lbl_conf["f1_threshold"],
            overlap_threshold=lbl_conf["evidence_overlap_threshold"]
        )
        
        is_hallucinated_expanded = label_hallucination(
            a_expanded, gts, [p["text"] for p in expanded_passages],
            em_threshold=lbl_conf["em_threshold"],
            f1_threshold=lbl_conf["f1_threshold"],
            overlap_threshold=lbl_conf["evidence_overlap_threshold"]
        )
        
        rec = {
            "qid": qid,
            "question": q_text,
            "baseline_answer": a_0,
            "ground_truths": gts,
            "exact_match": em,
            "f1_score": f1,
            "is_hallucinated": is_hallucinated,
            "confidence": conf,
            "entropy_proxy": entropy,
            "doc_similarity": doc_sim,
            "a_expanded": a_expanded,
            "expanded_f1": expanded_f1,
            "expanded_is_hallucinated": is_hallucinated_expanded,
            "gen_time_sec": gen_time
        }
        rec.update(rsi_stats)
        
        # Save to checkpoint
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            
        results.append(rec)
        
    # Save raw results
    logger.info("5. Saving raw instance logs...")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(paths["raw_logs_dir"], "full_results.csv"), index=False)
    
    # Check if we have results to process
    if df.empty:
        logger.warning("No results to evaluate. Exiting.")
        return
        
    # 6. Statistical Evaluation
    logger.info("6. Running statistical evaluation...")
    y_true = df["is_hallucinated"].tolist()
    
    # Compare correct vs hallucinated
    t_test_res = compare_groups(
        df[df["is_hallucinated"] == False]["rsi_mean"].tolist(),
        df[df["is_hallucinated"] == True]["rsi_mean"].tolist()
    )
    
    # Correlations
    corrs = compute_correlations(df["rsi_mean"].tolist(), df["f1_score"].tolist())
    
    # ROC AUC
    roc_rsi_mean = compute_roc_auc(y_true, df["rsi_mean"].tolist())
    roc_rsi_max = compute_roc_auc(y_true, df["rsi_max"].tolist())
    roc_rsi_var = compute_roc_auc(y_true, df["rsi_variance"].tolist())
    
    # New metrics
    roc_rsi_norm = compute_roc_auc(y_true, df["rsi_norm"].tolist())
    roc_rsi_evidence = compute_roc_auc(y_true, df["rsi_evidence"].tolist())
    roc_rsi_weighted = compute_roc_auc(y_true, df["rsi_weighted"].tolist())
    
    roc_entropy = compute_roc_auc(y_true, df["entropy_proxy"].tolist())
    roc_conf = compute_roc_auc(y_true, (1.0 - df["confidence"]).tolist())  # Low config = High risk
    roc_docsim = compute_roc_auc(y_true, (1.0 - df["doc_similarity"]).tolist())
    
    # Bootstrap CI
    logger.info("Computing bootstrap CIs (this may take a moment)...")
    ci_rsi = bootstrap_auc(y_true, df["rsi_mean"].tolist(), n_boot=500)
    ci_ent = bootstrap_auc(y_true, df["entropy_proxy"].tolist(), n_boot=500)
    ci_cnf = bootstrap_auc(y_true, (1.0 - df["confidence"]).tolist(), n_boot=500)
    
    stats_summary = {
        "t_test_p_value": t_test_res.get("p_value", 1.0),
        "cohens_d": t_test_res.get("cohens_d", 0.0),
        "rsi_f1_pearson": corrs.get("pearson_r", 0.0),
        "auc_rsi_mean": roc_rsi_mean["auc"],
        "auc_rsi_max": roc_rsi_max["auc"],
        "auc_rsi_variance": roc_rsi_var["auc"],
        "auc_rsi_norm": roc_rsi_norm["auc"],
        "auc_rsi_evidence": roc_rsi_evidence["auc"],
        "auc_rsi_weighted": roc_rsi_weighted["auc"],
        "auc_entropy": roc_entropy["auc"],
        "auc_confidence": roc_conf["auc"],
        "auc_doc_sim": roc_docsim["auc"],
        "opt_rsi_threshold": roc_rsi_mean.get("optimal_threshold", 0.5)
    }
    
    with open(os.path.join(paths["tables_dir"], "eval_summary.json"), "w") as f:
        json.dump(stats_summary, f, indent=2)
        
    # 7. Adaptive Policy Cost-Benefit sweep
    logger.info("7. Sweeping Adaptive Policy thresholds...")
    thresholds = np.linspace(0.0, df["rsi_max"].max(), 20)
    sweep_results = []
    
    total = len(df)
    base_hallucination_rate = (df["is_hallucinated"].sum() / total) * 100
    
    for t in thresholds:
        triggered = df["rsi_mean"] > t
        
        # New outcomes = fallback where triggered, else baseline
        new_is_hallucinated = df["is_hallucinated"].copy()
        new_is_hallucinated[triggered] = df.loc[triggered, "expanded_is_hallucinated"]
        
        new_f1 = df["f1_score"].copy()
        new_f1[triggered] = df.loc[triggered, "expanded_f1"]
        
        sweep_results.append({
            "threshold": float(t),
            "extra_compute_pct": (triggered.sum() / total) * 100,
            "base_hallucination_rate": base_hallucination_rate,
            "new_hallucination_rate": (new_is_hallucinated.sum() / total) * 100,
            "new_mean_f1": float(new_f1.mean())
        })
        
    pd.DataFrame(sweep_results).to_csv(os.path.join(paths["tables_dir"], "adaptive_sweep.csv"), index=False)
    
    # 8. Plotting Generation
    logger.info("8. Generating publication plots...")
    plotter = PlotGenerator(out_dir=paths["figures_dir"])
    
    try:
        plotter.plot_fig1_rsi_distribution(df)
        plotter.plot_fig2_rsi_vs_f1_scatter(df)
        plotter.plot_fig3_rsi_em_box(df)
        plotter.plot_fig4_roc_comparison(df)
        
        ci_dict = {
            "RSI (Mean)": ci_rsi,
            "Entropy": ci_ent,
            "1 - Confidence": ci_cnf
        }
        plotter.plot_fig5_auc_bootstrap_ci(ci_dict)
        
        plotter.plot_fig6_precision_recall(df)
        
        # Predict using optimal threshold from Youden's J
        y_pred = df["rsi_mean"] > roc_rsi_mean["optimal_threshold"]
        plotter.plot_fig7_confusion_matrix(y_true, y_pred.tolist())
        
        plotter.plot_fig8_adaptive_cost_benefit(sweep_results)
        plotter.plot_fig10_perturbation_heatmap(df)
        plotter.plot_fig13_correlation_matrix(df)
    except Exception as e:
        logger.error(f"Plotting failed (likely due to insufficient samples containing both classes): {e}")

    logger.info("Pipeline Complete. Results saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Run RSI Experimental Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--sample-size", type=int, help="Override sample size in config for quick testing")
    parser.add_argument("--model", type=str, help="Override Ollama model name in config")
    parser.add_argument("--dataset", type=str, choices=["squad", "medical", "both", "medquad", "pubmedqa"], help="Which dataset(s) to run on")
    parser.add_argument("--resume", type=str, help="Path to a previous run directory to resume from")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.sample_size:
        config["dataset"]["sample_size"] = args.sample_size
    if args.model:
        config["generator"]["model_name"] = args.model
    if args.dataset:
        config["dataset"]["dataset_flag"] = args.dataset
        
    if args.resume:
        # Resume mode: Use provided path
        base_results = args.resume
        logger_name = f"pipeline_resume_{os.path.basename(base_results)}"
        logger = setup_logger(base_results)
        # Update paths in config for the experiment to use this folder
        config["paths"]["raw_logs_dir"] = os.path.join(base_results, "raw")
        config["paths"]["tables_dir"] = os.path.join(base_results, "tables")
        config["paths"]["figures_dir"] = os.path.join(base_results, "figures")
        logger.info(f"Resuming experiment from: {base_results}")
    else:
        # New run mode: Generate unique run ID
        import datetime
        model_safe_name = config["generator"]["model_name"].replace("/", "_").replace(":", "_")
        dataset_safe_name = config["dataset"]["dataset_flag"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{model_safe_name}_{dataset_safe_name}_{timestamp}"
        
        # Update paths in config
        base_results = os.path.join(config["paths"]["results_dir"], run_id)
        config["paths"]["raw_logs_dir"] = os.path.join(base_results, "raw")
        config["paths"]["tables_dir"] = os.path.join(base_results, "tables")
        config["paths"]["figures_dir"] = os.path.join(base_results, "figures")
            
        log_dir = base_results
        logger = setup_logger(log_dir)
    
    run_experiment(config, logger)
