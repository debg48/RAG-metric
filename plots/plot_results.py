import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)

# Configure paper-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    'figure.autolayout': True,
    'figure.dpi': 300
})

class PlotGenerator:
    def __init__(self, out_dir: str = "results/figures"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
    def _save(self, fig, name: str):
        fig.savefig(os.path.join(self.out_dir, f"{name}.png"), format='png')
        fig.savefig(os.path.join(self.out_dir, f"{name}.pdf"), format='pdf')
        plt.close(fig)
        logger.info(f"Saved figure: {name}")

    def plot_fig1_rsi_distribution(self, df: pd.DataFrame):
        """Fig 1: Violin + box plot of RSI by hallucination label"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # We need a proper grouping column
        df['Group'] = df['is_hallucinated'].map({True: 'Hallucinated', False: 'Correct'})
        
        sns.violinplot(
            data=df, x='Group', y='rsi_mean', 
            inner=None, color=".8", ax=ax, alpha=0.5
        )
        sns.boxplot(
            data=df, x='Group', y='rsi_mean',
            width=0.3, boxprops={'zorder': 2}, ax=ax,
            palette=['#2ecc71', '#e74c3c'] # Green for correct, Red for hallucinated
        )
        
        ax.set_title("RSI Distribution: Correct vs. Hallucinated Outputs")
        ax.set_ylabel("Retrieval Sensitivity Index (RSI)")
        ax.set_xlabel("")
        
        self._save(fig, "fig1_rsi_distribution")

    def plot_fig2_rsi_vs_f1_scatter(self, df: pd.DataFrame):
        """Fig 2: RSI vs F1 scatter with regression"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df['Group'] = df['is_hallucinated'].map({True: 'Hallucinated', False: 'Correct'})
        
        sns.regplot(
            data=df, x='rsi_mean', y='f1_score', 
            scatter=False, color='black', ax=ax,
            line_kws={'linestyle':'--'}
        )
        sns.scatterplot(
            data=df, x='rsi_mean', y='f1_score',
            hue='Group', palette=['#2ecc71', '#e74c3c'],
            alpha=0.7, ax=ax, s=100
        )
        
        ax.set_title("Answer Quality (F1) vs. Retrieval Sensitivity (RSI)")
        ax.set_xlabel("RSI")
        ax.set_ylabel("F1 Score")
        
        self._save(fig, "fig2_rsi_vs_f1_scatter")

    def plot_fig3_rsi_em_box(self, df: pd.DataFrame):
        """Fig 3: RSI by Exact Match"""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.boxplot(
            data=df, x='exact_match', y='rsi_mean',
            palette='Blues', ax=ax
        )
        ax.set_title("RSI grouped by Exact Match")
        ax.set_xlabel("Exact Match (0/1)")
        ax.set_ylabel("RSI")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Match (0)', 'Exact Match (1)'])
        
        self._save(fig, "fig3_rsi_vs_em_box")

    def plot_fig4_roc_comparison(self, df: pd.DataFrame):
        """Fig 4: ROC curves for all predictors"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        y_true = df['is_hallucinated'].astype(int)
        
        predictors = {
            'RSI (Mean)': df['rsi_mean'],
            'RSI (Norm)': df['rsi_norm'],
            'RSI (Evidence)': df['rsi_evidence'],
            'RSI (Weighted)': df['rsi_weighted'],
            'Entropy Proxy': df['entropy_proxy'],
            'Doc Sim (Inv)': 1.0 - df['doc_similarity'], # Lower sim -> more likely to hallucinate
            'Confidence (Inv)': 1.0 - df['confidence'] # Lower conf -> more likely to hallucinate
        }
        
        colors = ['#e74c3c', '#d35400', '#27ae60', '#8e44ad', '#3498db', '#f39c12', '#2c3e50']
        
        for (name, scores), color in zip(predictors.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
        ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        
        self._save(fig, "fig4_roc_comparison")
        
    def plot_fig5_auc_bootstrap_ci(self, ci_data: Dict[str, Tuple[float, float, float]]):
        """Fig 5: Bootstrap CI forest plot
        ci_data format: {'RSI': (mean, lower, upper), ...}
        """
        if not ci_data:
            return
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        names = list(ci_data.keys())
        means = [v[0] for v in ci_data.values()]
        lower = [v[1] for v in ci_data.values()]
        upper = [v[2] for v in ci_data.values()]
        
        y_pos = np.arange(len(names))
        
        err_lower = np.array(means) - np.array(lower)
        err_upper = np.array(upper) - np.array(means)
        
        ax.errorbar(means, y_pos, xerr=[err_lower, err_upper], fmt='o', 
                   color='black', capsize=5, capthick=2, markersize=8)
                   
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('AUC (95% CI)')
        ax.set_title('Bootstrap Confidence Intervals for AUC')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        self._save(fig, "fig5_auc_bootstrap_ci")

    def plot_fig6_precision_recall(self, df: pd.DataFrame):
        """Fig 6: PR Curves"""
        fig, ax = plt.subplots(figsize=(8, 8))
        y_true = df['is_hallucinated'].astype(int)
        
        predictors = {
            'RSI (Mean)': df['rsi_mean'],
            'RSI (Norm)': df['rsi_norm'],
            'RSI (Evidence)': df['rsi_evidence'],
            'RSI (Weighted)': df['rsi_weighted'],
            'Entropy Proxy': df['entropy_proxy'],
            'Confidence (Inv)': 1.0 - df['confidence']
        }
        
        colors = ['#e74c3c', '#d35400', '#27ae60', '#8e44ad', '#3498db', '#2c3e50']
        
        for (name, scores), color in zip(predictors.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, scores)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=color, lw=2, label=f'{name} (AP = {ap:.3f})')
            
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        self._save(fig, "fig6_precision_recall")

    def plot_fig7_confusion_matrix(self, y_true: List[bool], y_pred: List[bool]):
        """Fig 7: Confusion Matrix"""
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Correct (Pred)', 'Hallucinated (Pred)'],
                   yticklabels=['Correct (True)', 'Hallucinated (True)'])
                   
        ax.set_title('Decision Boundary at Optimal Threshold')
        self._save(fig, "fig7_confusion_matrix")

    def plot_fig8_adaptive_cost_benefit(self, threshold_metrics: List[Dict[str, float]]):
        """Fig 8: Cost-benefit curve"""
        if not threshold_metrics:
            return
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df = pd.DataFrame(threshold_metrics)
        # x: extra_cost_pct, y: hall_rate
        
        ax.plot(df['extra_compute_pct'], df['new_hallucination_rate'], marker='o', linestyle='-', color='#8e44ad', linewidth=2)
        
        # Annotate points with threshold values
        for i, row in df.iterrows():
            if i % max(1, len(df)//5) == 0:  # Annotate subset
                ax.annotate(f"$\\tau={row['threshold']:.2f}$", 
                           (row['extra_compute_pct'], row['new_hallucination_rate']),
                           textcoords="offset points", xytext=(0,10), ha='center')
                           
        ax.axhline(y=df['base_hallucination_rate'].iloc[0], color='r', linestyle='--', label='Baseline Hallucination Rate')
        
        ax.set_xlabel('% Extra Retrieval Calls')
        ax.set_ylabel('Resulting Hallucination Rate')
        ax.set_title('Adaptive Re-retrieval Cost-Benefit Analysis')
        ax.legend()
        
        self._save(fig, "fig8_adaptive_cost_benefit")

    def plot_fig10_perturbation_heatmap(self, df: pd.DataFrame):
        """Fig 10: Divergence Matrix heatmap (subset of 20 queries)"""
        # We need per-perturbation data
        if 'div_remove_top1' not in df.columns:
            return
            
        cols_of_interest = [c for c in df.columns if c.startswith('div_')]
        if not cols_of_interest:
            return
            
        # Sample 20 random queries, sorted by label
        sample = df.sample(min(20, len(df)), random_state=42).sort_values('is_hallucinated')
        data = sample[cols_of_interest].values
        labels = sample['is_hallucinated'].map({True: 'H', False: 'C'}).values
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Clean column names
        col_names = [c.replace('div_', '').replace('_', ' ').title() for c in cols_of_interest]
        
        sns.heatmap(data, cmap='YlOrRd', ax=ax,
                   xticklabels=col_names,
                   yticklabels=[f"Q{i} ({l})" for i, l in enumerate(labels)])
                   
        ax.set_title('Per-Perturbation Divergence (Sample)')
        self._save(fig, "fig10_perturbation_heatmap")

    def plot_fig13_correlation_matrix(self, df: pd.DataFrame):
        """Fig 13: Correlation heatmap"""
        cols = ['rsi_mean', 'entropy_proxy', 'confidence', 'doc_similarity', 'f1_score', 'exact_match']
        available_cols = [c for c in cols if c in df.columns]
        
        if len(available_cols) < 2:
            return
            
        corr = df[available_cols].corr(method='spearman')
        
        # Clean names
        clean_names = {
            'rsi_mean': 'RSI',
            'entropy_proxy': 'Entropy Proxy',
            'confidence': 'Confidence',
            'doc_similarity': 'Doc Sim',
            'f1_score': 'F1 Score',
            'exact_match': 'Exact Match'
        }
        corr = corr.rename(index=clean_names, columns=clean_names)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax,
                   square=True, linewidths=.5)
                   
        ax.set_title('Spearman Correlation Matrix of Prediction Signals')
        self._save(fig, "fig13_correlation_matrix")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Generate dummy data and test plots
    N = 100
    np.random.seed(42)
    df = pd.DataFrame({
        'rsi_mean': np.random.beta(2, 5, N) * 2,
        'f1_score': np.random.uniform(0, 1, N),
        'exact_match': np.random.choice([0, 1], N),
        'entropy_proxy': np.random.uniform(0, 0.5, N),
        'confidence': np.random.uniform(0.5, 1.0, N),
        'doc_similarity': np.random.uniform(0.3, 0.9, N),
        'is_hallucinated': np.random.choice([False, True], N, p=[0.7, 0.3])
    })
    # Correlate F1 and RSI for synthetic data
    df.loc[df['is_hallucinated'] == True, 'rsi_mean'] += 0.5
    df.loc[df['is_hallucinated'] == True, 'f1_score'] *= 0.3
    
    pg = PlotGenerator(out_dir="results/figures_test")
    pg.plot_fig1_rsi_distribution(df)
    pg.plot_fig2_rsi_vs_f1_scatter(df)
    pg.plot_fig3_rsi_em_box(df)
    pg.plot_fig4_roc_comparison(df)
    pg.plot_fig6_precision_recall(df)
    pg.plot_fig13_correlation_matrix(df)
    print("Test plots generated.")
