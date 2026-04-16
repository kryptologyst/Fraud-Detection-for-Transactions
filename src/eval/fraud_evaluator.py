"""Comprehensive evaluation metrics for fraud detection."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots


logger = logging.getLogger(__name__)


class FraudEvaluator:
    """Comprehensive evaluation for fraud detection models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_proba),
            "auc_pr": average_precision_score(y_true, y_proba),
        }
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        })
        
        # Precision at different recall levels
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Precision at 1%, 5%, 10% recall
        for recall_target in [0.01, 0.05, 0.10]:
            idx = np.where(recall >= recall_target)[0]
            if len(idx) > 0:
                metrics[f"precision_at_{int(recall_target*100)}pct_recall"] = precision[idx[0]]
        
        # Precision at top K%
        n_samples = len(y_proba)
        for k in [1, 5, 10]:
            k_samples = max(1, int(n_samples * k / 100))
            top_k_indices = np.argsort(y_proba)[-k_samples:]
            top_k_precision = y_true[top_k_indices].mean()
            metrics[f"precision_at_top_{k}pct"] = top_k_precision
        
        # Alert volume metrics
        fraud_rate = y_true.mean()
        alert_rate = y_pred.mean()
        metrics.update({
            "fraud_rate": fraud_rate,
            "alert_rate": alert_rate,
            "alert_volume_per_1k": alert_rate * 1000,
            "precision_lift": metrics["precision"] / fraud_rate if fraud_rate > 0 else 0,
        })
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model at multiple thresholds.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary with metrics for each threshold
        """
        logger.info("Evaluating model at multiple thresholds...")
        
        # Get predictions
        y_proba = model.predict_proba(X_test)
        
        results = {}
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self.calculate_metrics(y_test, y_pred, y_proba, threshold)
            results[f"threshold_{threshold}"] = metrics
        
        self.results = results
        return results
    
    def create_leaderboard(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a leaderboard comparing different models/thresholds.
        
        Args:
            results: Results from evaluate_model
            
        Returns:
            DataFrame with leaderboard
        """
        leaderboard_data = []
        
        for model_name, metrics in results.items():
            leaderboard_data.append({
                "Model": model_name,
                "AUCPR": metrics.get("auc_pr", 0),
                "Precision@1%": metrics.get("precision_at_1pct_recall", 0),
                "Precision@5%": metrics.get("precision_at_5pct_recall", 0),
                "Precision@Top1%": metrics.get("precision_at_top_1pct", 0),
                "Precision@Top5%": metrics.get("precision_at_top_5pct", 0),
                "F1-Score": metrics.get("f1", 0),
                "Alert Rate": metrics.get("alert_rate", 0),
                "Alert Volume/1K": metrics.get("alert_volume_per_1k", 0),
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values("AUCPR", ascending=False)
        
        return leaderboard
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Figure:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Figure:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {auc_pr:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Add baseline (fraud rate)
        fraud_rate = y_true.mean()
        fig.add_hline(
            y=fraud_rate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (Fraud Rate = {fraud_rate:.3f})"
        )
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Fraud'],
            y=['Actual Normal', 'Actual Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> Figure:
        """Plot threshold analysis showing precision, recall, and F1 vs threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        thresholds = np.linspace(0.01, 0.99, 50)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Threshold Analysis']
        )
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precision_scores,
            mode='lines',
            name='Precision',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recall_scores,
            mode='lines',
            name='Recall',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1-Score',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Threshold Analysis',
            xaxis_title='Threshold',
            yaxis_title='Score',
            width=700,
            height=500
        )
        
        return fig
    
    def generate_report(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Get predictions
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba, threshold)
        
        # Generate report
        report = f"""
# Fraud Detection Model Evaluation Report

## Model Performance Summary
- **Model Type**: {getattr(model, 'model_type', 'Unknown')}
- **Threshold**: {threshold}
- **Test Samples**: {len(y_test):,}
- **Fraud Rate**: {metrics['fraud_rate']:.3f}
- **Alert Rate**: {metrics['alert_rate']:.3f}

## Key Metrics
- **AUCPR**: {metrics['auc_pr']:.3f}
- **AUC-ROC**: {metrics['auc_roc']:.3f}
- **Precision**: {metrics['precision']:.3f}
- **Recall**: {metrics['recall']:.3f}
- **F1-Score**: {metrics['f1']:.3f}
- **Specificity**: {metrics['specificity']:.3f}

## Precision at Different Recall Levels
- **Precision@1% Recall**: {metrics.get('precision_at_1pct_recall', 0):.3f}
- **Precision@5% Recall**: {metrics.get('precision_at_5pct_recall', 0):.3f}
- **Precision@10% Recall**: {metrics.get('precision_at_10pct_recall', 0):.3f}

## Precision at Top K%
- **Precision@Top1%**: {metrics['precision_at_top_1pct']:.3f}
- **Precision@Top5%**: {metrics['precision_at_top_5pct']:.3f}
- **Precision@Top10%**: {metrics['precision_at_top_10pct']:.3f}

## Confusion Matrix
- **True Negatives**: {metrics['true_negatives']:,}
- **False Positives**: {metrics['false_positives']:,}
- **False Negatives**: {metrics['false_negatives']:,}
- **True Positives**: {metrics['true_positives']:,}

## Operational Metrics
- **Alert Volume per 1K Transactions**: {metrics['alert_volume_per_1k']:.1f}
- **Precision Lift**: {metrics['precision_lift']:.1f}x
- **False Positive Rate**: {metrics['false_positive_rate']:.3f}
- **False Negative Rate**: {metrics['false_negative_rate']:.3f}

## Recommendations
"""
        
        # Add recommendations based on metrics
        if metrics['precision'] < 0.3:
            report += "- **Low Precision**: Consider increasing threshold or improving feature engineering\n"
        
        if metrics['recall'] < 0.5:
            report += "- **Low Recall**: Consider decreasing threshold or using ensemble methods\n"
        
        if metrics['alert_volume_per_1k'] > 50:
            report += "- **High Alert Volume**: Consider increasing threshold to reduce false positives\n"
        
        if metrics['precision_lift'] < 2:
            report += "- **Low Precision Lift**: Model may not be significantly better than random\n"
        
        report += "\n---\n"
        report += "*This is a research/educational tool. Not suitable for production fraud detection.*\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
