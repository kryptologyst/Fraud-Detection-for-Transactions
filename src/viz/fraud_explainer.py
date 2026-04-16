"""Explainability and interpretability for fraud detection models."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class FraudExplainer:
    """Explain fraud detection model predictions using SHAP and other methods."""
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize explainer.
        
        Args:
            model: Trained fraud detection model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def _prepare_shap_explainer(self, X: pd.DataFrame) -> None:
        """Prepare SHAP explainer for the model.
        
        Args:
            X: Feature matrix
        """
        logger.info("Preparing SHAP explainer...")
        
        # Use different explainers based on model type
        if hasattr(self.model, 'model_type'):
            if self.model.model_type == "xgboost":
                self.explainer = shap.TreeExplainer(self.model.model)
            elif self.model.model_type == "lightgbm":
                self.explainer = shap.TreeExplainer(self.model.model)
            elif self.model.model_type == "random_forest":
                self.explainer = shap.TreeExplainer(self.model.model)
            elif self.model.model_type == "neural_network":
                # Use KernelExplainer for neural networks
                background = shap.sample(X, 100)  # Sample background
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                # Fallback to KernelExplainer
                background = shap.sample(X, 100)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        else:
            # Fallback to KernelExplainer
            background = shap.sample(X, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
    
    def explain_predictions(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000,
        save_shap_values: bool = True,
    ) -> Dict[str, Any]:
        """Explain model predictions using SHAP.
        
        Args:
            X: Feature matrix to explain
            max_samples: Maximum number of samples to explain
            save_shap_values: Whether to save SHAP values
            
        Returns:
            Dictionary with explanation results
        """
        logger.info(f"Explaining predictions for {len(X)} samples...")
        
        # Limit samples for performance
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
            logger.info(f"Sampling {max_samples} samples for explanation")
        else:
            X_sample = X
        
        # Prepare explainer if not already done
        if self.explainer is None:
            self._prepare_shap_explainer(X_sample)
        
        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            # Tree-based models
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use fraud class
        else:
            # Kernel-based explainer
            shap_values = self.explainer(X_sample)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values[:, :, 1]  # Use fraud class
        
        if save_shap_values:
            self.shap_values = shap_values
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Get top contributing features for each prediction
        explanations = []
        for i in range(len(X_sample)):
            sample_shap = shap_values[i]
            top_features = np.argsort(np.abs(sample_shap))[-5:][::-1]  # Top 5 features
            
            explanation = {
                'sample_idx': i,
                'prediction': self.model.predict_proba(X_sample.iloc[[i]])[0],
                'top_features': [
                    {
                        'feature': self.feature_names[idx],
                        'shap_value': sample_shap[idx],
                        'feature_value': X_sample.iloc[i, idx]
                    }
                    for idx in top_features
                ]
            }
            explanations.append(explanation)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance_df,
            'explanations': explanations,
            'X_sample': X_sample
        }
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame, top_k: int = 20) -> Figure:
        """Plot feature importance.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            top_k: Number of top features to show
            
        Returns:
            Plotly figure
        """
        top_features = feature_importance_df.head(top_k)
        
        fig = go.Figure(data=go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color=top_features['importance'], colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f'Top {top_k} Feature Importance',
            xaxis_title='SHAP Importance',
            yaxis_title='Features',
            width=800,
            height=max(400, top_k * 20)
        )
        
        return fig
    
    def plot_shap_summary(self, shap_values: np.ndarray, X_sample: pd.DataFrame) -> Figure:
        """Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X_sample: Sample features
            
        Returns:
            Plotly figure
        """
        # Calculate mean absolute SHAP values for summary
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_shap)[::-1]
        top_features = sorted_indices[:20]  # Top 20 features
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['SHAP Values Distribution', 'Feature Importance'],
            horizontal_spacing=0.1
        )
        
        # SHAP values distribution
        for i, feature_idx in enumerate(top_features[:10]):  # Top 10 for clarity
            fig.add_trace(
                go.Scatter(
                    x=shap_values[:, feature_idx],
                    y=[self.feature_names[feature_idx]] * len(shap_values),
                    mode='markers',
                    name=self.feature_names[feature_idx],
                    opacity=0.6,
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # Feature importance bar chart
        fig.add_trace(
            go.Bar(
                x=[mean_shap[idx] for idx in top_features],
                y=[self.feature_names[idx] for idx in top_features],
                orientation='h',
                name='Importance',
                marker=dict(color='lightblue')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='SHAP Summary Plot',
            width=1200,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_waterfall(self, explanation: Dict[str, Any], sample_idx: int = 0) -> Figure:
        """Plot waterfall chart for individual prediction explanation.
        
        Args:
            explanation: Explanation dictionary from explain_predictions
            sample_idx: Index of sample to explain
            
        Returns:
            Plotly figure
        """
        sample_explanation = explanation['explanations'][sample_idx]
        
        # Sort features by SHAP value magnitude
        sorted_features = sorted(
            sample_explanation['top_features'],
            key=lambda x: abs(x['shap_value']),
            reverse=True
        )
        
        # Create waterfall data
        base_value = 0.5  # Assume base probability
        cumulative = base_value
        
        waterfall_data = []
        waterfall_data.append({
            'feature': 'Base',
            'shap_value': 0,
            'cumulative': base_value
        })
        
        for feature in sorted_features:
            cumulative += feature['shap_value']
            waterfall_data.append({
                'feature': feature['feature'],
                'shap_value': feature['shap_value'],
                'cumulative': cumulative
            })
        
        # Create waterfall plot
        fig = go.Figure(go.Waterfall(
            name="Fraud Probability",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(waterfall_data) - 2) + ["total"],
            x=[item['feature'] for item in waterfall_data],
            y=[item['shap_value'] for item in waterfall_data],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"Prediction Explanation (Sample {sample_idx})",
            showlegend=False,
            width=800,
            height=500
        )
        
        return fig
    
    def generate_explanation_report(
        self,
        X: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        max_samples: int = 100,
    ) -> str:
        """Generate comprehensive explanation report.
        
        Args:
            X: Feature matrix
            y_true: True labels (optional)
            max_samples: Maximum samples to explain
            
        Returns:
            Report string
        """
        logger.info("Generating explanation report...")
        
        # Get explanations
        explanations = self.explain_predictions(X, max_samples=max_samples)
        
        # Generate report
        report = f"""
# Fraud Detection Model Explanation Report

## Model Interpretability Analysis

### Feature Importance (Top 10)
"""
        
        top_features = explanations['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
        
        report += f"""
### Sample Explanations

Analyzed {len(explanations['explanations'])} sample predictions:

"""
        
        # Show explanations for a few samples
        for i, explanation in enumerate(explanations['explanations'][:5]):
            report += f"""
#### Sample {i+1}
- **Predicted Fraud Probability**: {explanation['prediction']:.3f}
- **Top Contributing Features**:
"""
            for feature in explanation['top_features']:
                direction = "increases" if feature['shap_value'] > 0 else "decreases"
                report += f"  - {feature['feature']}: {direction} fraud probability by {abs(feature['shap_value']):.3f}\n"
        
        report += """
### Key Insights

1. **Most Important Features**: The top contributing features provide insights into fraud patterns
2. **Feature Interactions**: SHAP values show how individual features contribute to predictions
3. **Model Transparency**: Understanding feature contributions helps validate model decisions

### Recommendations

- Monitor top features for data quality and drift
- Investigate high-contributing features for business insights
- Use explanations to build trust in model decisions
- Consider feature engineering based on importance rankings

---
*This explanation is for research/educational purposes. Not suitable for production fraud detection.*
"""
        
        return report
    
    def get_rule_based_explanations(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Generate rule-based explanations for predictions.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            List of rule-based explanations
        """
        predictions = self.model.predict_proba(X)
        fraud_predictions = predictions >= threshold
        
        rules = []
        
        # Rule 1: High amount transactions
        high_amount_mask = X['amount'] > X['amount'].quantile(0.95)
        high_amount_fraud = fraud_predictions[high_amount_mask].mean()
        if high_amount_fraud > 0.3:
            rules.append({
                'rule': 'High Amount Transactions',
                'condition': f'amount > {X["amount"].quantile(0.95):.2f}',
                'fraud_rate': high_amount_fraud,
                'description': f'Transactions above 95th percentile have {high_amount_fraud:.1%} fraud rate'
            })
        
        # Rule 2: Night transactions
        night_mask = X['time_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23])
        night_fraud = fraud_predictions[night_mask].mean()
        if night_fraud > 0.2:
            rules.append({
                'rule': 'Night Transactions',
                'condition': 'time_of_day in [0,1,2,3,4,5,22,23]',
                'fraud_rate': night_fraud,
                'description': f'Night transactions have {night_fraud:.1%} fraud rate'
            })
        
        # Rule 3: Unusual merchant categories
        merchant_fraud_rates = X.groupby('merchant_category').apply(
            lambda x: fraud_predictions[x.index].mean()
        )
        high_risk_merchants = merchant_fraud_rates[merchant_fraud_rates > 0.15]
        
        for merchant, fraud_rate in high_risk_merchants.items():
            rules.append({
                'rule': f'High-Risk Merchant: {merchant}',
                'condition': f'merchant_category == "{merchant}"',
                'fraud_rate': fraud_rate,
                'description': f'{merchant} transactions have {fraud_rate:.1%} fraud rate'
            })
        
        # Rule 4: Device trust issues
        if 'device_trusted' in X.columns:
            untrusted_mask = X['device_trusted'] == 0
            untrusted_fraud = fraud_predictions[untrusted_mask].mean()
            if untrusted_fraud > 0.2:
                rules.append({
                    'rule': 'Untrusted Device',
                    'condition': 'device_trusted == 0',
                    'fraud_rate': untrusted_fraud,
                    'description': f'Untrusted devices have {untrusted_fraud:.1%} fraud rate'
                })
        
        return rules
