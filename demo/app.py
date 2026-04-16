"""Streamlit demo application for fraud detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_data import generate_transaction_data
from src.models.fraud_detector import FraudDetector
from src.eval.fraud_evaluator import FraudEvaluator
from src.viz.fraud_explainer import FraudExplainer
from src.utils import set_seed, setup_logging, obfuscate_ip, obfuscate_email


# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .fraud-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .normal-transaction {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Educational Tool Only</h4>
    <p>This is a demonstration tool for research and educational purposes. 
    <strong>NOT suitable for production fraud detection.</strong> All data is synthetic and contains no real personal information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Data generation parameters
        st.subheader("Data Generation")
        n_users = st.slider("Number of Users", 100, 5000, 1000)
        n_transactions = st.slider("Number of Transactions", 1000, 50000, 10000)
        fraud_rate = st.slider("Target Fraud Rate (%)", 1, 20, 5) / 100
        
        # Model parameters
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "lightgbm", "neural_network", "random_forest"],
            index=0
        )
        use_graph_features = st.checkbox("Use Graph Features", value=True)
        use_imbalanced_learning = st.checkbox("Use Imbalanced Learning", value=True)
        
        # Evaluation parameters
        st.subheader("Evaluation")
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)
        
        # Generate data button
        if st.button("🔄 Generate Data", type="primary"):
            with st.spinner("Generating synthetic transaction data..."):
                set_seed(42)
                users_df, transactions_df = generate_transaction_data(
                    n_users=n_users,
                    n_transactions=n_transactions,
                    fraud_rate=fraud_rate,
                    seed=42
                )
                
                st.session_state.users_df = users_df
                st.session_state.transactions_df = transactions_df
                st.session_state.data_generated = True
                st.session_state.model_trained = False
                
                st.success(f"Generated {len(transactions_df):,} transactions for {len(users_df):,} users")
                st.rerun()
        
        # Train model button
        if st.session_state.data_generated and not st.session_state.model_trained:
            if st.button("🤖 Train Model", type="primary"):
                with st.spinner("Training fraud detection model..."):
                    # Initialize model
                    model = FraudDetector(
                        model_type=model_type,
                        use_imbalanced_learning=use_imbalanced_learning,
                        use_graph_features=use_graph_features,
                        random_state=42
                    )
                    
                    # Train model
                    model.fit(st.session_state.transactions_df)
                    
                    # Initialize evaluator and explainer
                    evaluator = FraudEvaluator()
                    explainer = FraudExplainer(model, model.feature_names)
                    
                    st.session_state.current_model = model
                    st.session_state.evaluator = evaluator
                    st.session_state.explainer = explainer
                    st.session_state.model_trained = True
                    
                    st.success("Model trained successfully!")
                    st.rerun()
    
    # Main content
    if not st.session_state.data_generated:
        st.info("👈 Please generate data using the sidebar to get started.")
        return
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(st.session_state.transactions_df):,}")
    
    with col2:
        fraud_count = st.session_state.transactions_df['is_fraud'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        fraud_rate_actual = st.session_state.transactions_df['is_fraud'].mean()
        st.metric("Actual Fraud Rate", f"{fraud_rate_actual:.1%}")
    
    with col4:
        st.metric("Unique Users", f"{st.session_state.transactions_df['user_id'].nunique():,}")
    
    # Data visualization
    st.subheader("Transaction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution
        fig_amount = go.Figure()
        
        normal_amounts = st.session_state.transactions_df[~st.session_state.transactions_df['is_fraud']]['amount']
        fraud_amounts = st.session_state.transactions_df[st.session_state.transactions_df['is_fraud']]['amount']
        
        fig_amount.add_trace(go.Histogram(
            x=normal_amounts,
            name='Normal Transactions',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_amount.add_trace(go.Histogram(
            x=fraud_amounts,
            name='Fraudulent Transactions',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_amount.update_layout(
            title="Transaction Amount Distribution",
            xaxis_title="Amount ($)",
            yaxis_title="Count",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Time distribution
        fig_time = go.Figure()
        
        normal_times = st.session_state.transactions_df[~st.session_state.transactions_df['is_fraud']]['time_of_day']
        fraud_times = st.session_state.transactions_df[st.session_state.transactions_df['is_fraud']]['time_of_day']
        
        fig_time.add_trace(go.Histogram(
            x=normal_times,
            name='Normal Transactions',
            opacity=0.7,
            nbinsx=24
        ))
        
        fig_time.add_trace(go.Histogram(
            x=fraud_times,
            name='Fraudulent Transactions',
            opacity=0.7,
            nbinsx=24
        ))
        
        fig_time.update_layout(
            title="Transaction Time Distribution",
            xaxis_title="Hour of Day",
            yaxis_title="Count",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Model evaluation
    if st.session_state.model_trained:
        st.header("🤖 Model Performance")
        
        # Get predictions
        y_true = st.session_state.transactions_df['is_fraud']
        y_proba = st.session_state.current_model.predict_proba(st.session_state.transactions_df)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Precision", f"{precision:.3f}")
        
        with col2:
            st.metric("Recall", f"{recall:.3f}")
        
        with col3:
            st.metric("F1-Score", f"{f1:.3f}")
        
        with col4:
            st.metric("AUC-ROC", f"{auc_roc:.3f}")
        
        with col5:
            st.metric("AUC-PR", f"{auc_pr:.3f}")
        
        # Performance plots
        st.subheader("Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_roc:.3f})',
                line=dict(color='blue', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=500,
                height=400
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # Precision-Recall Curve
            from sklearn.metrics import precision_recall_curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recall_curve,
                y=precision_curve,
                mode='lines',
                name=f'PR Curve (AUC = {auc_pr:.3f})',
                line=dict(color='green', width=2)
            ))
            
            # Add baseline
            fraud_rate = y_true.mean()
            fig_pr.add_hline(
                y=fraud_rate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Baseline ({fraud_rate:.3f})"
            )
            
            fig_pr.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                width=500,
                height=400
            )
            
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance")
        
        try:
            # Get explanations
            explanations = st.session_state.explainer.explain_predictions(
                st.session_state.transactions_df, max_samples=1000
            )
            
            # Plot feature importance
            fig_importance = st.session_state.explainer.plot_feature_importance(
                explanations['feature_importance'], top_k=15
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating explanations: {str(e)}")
        
        # Interactive prediction
        st.header("🎯 Interactive Prediction")
        
        st.subheader("Test Individual Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample a random transaction
            if st.button("🎲 Random Transaction"):
                sample_transaction = st.session_state.transactions_df.sample(1)
                st.session_state.sample_transaction = sample_transaction
        
        with col2:
            # Manual transaction input
            st.write("Or create a custom transaction:")
            
            amount = st.number_input("Amount ($)", min_value=1.0, max_value=10000.0, value=100.0)
            time_of_day = st.slider("Time of Day (Hour)", 0, 23, 12)
            merchant_category = st.selectbox(
                "Merchant Category",
                ["grocery", "gas_station", "restaurant", "online_retail", "electronics", 
                 "jewelry", "travel", "gambling", "cryptocurrency", "cash_advance"]
            )
            device_trusted = st.selectbox("Device Trusted", [True, False])
            location_match = st.selectbox("Location Match", [True, False])
            
            if st.button("🔍 Predict"):
                # Create custom transaction
                custom_transaction = pd.DataFrame([{
                    'amount': amount,
                    'time_of_day': time_of_day,
                    'merchant_category': merchant_category,
                    'device_trusted': device_trusted,
                    'location_match': location_match,
                    'day_of_week': 1,  # Default values
                    'account_age_days': 365,
                    'credit_score': 650,
                    'country': 'US',
                    'device_type': 'mobile_app',
                    'device_id': 'custom_device',
                    'ip_hash': 'custom_ip',
                    'user_id': 'custom_user',
                    'time_since_last_transaction': 24,
                    'is_fraud': False  # Unknown for prediction
                }])
                
                st.session_state.sample_transaction = custom_transaction
        
        # Display prediction results
        if 'sample_transaction' in st.session_state:
            transaction = st.session_state.sample_transaction
            
            # Get prediction
            fraud_prob = st.session_state.current_model.predict_proba(transaction)[0]
            is_fraud_predicted = fraud_prob >= threshold
            
            # Display transaction details
            st.subheader("Transaction Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Amount**: ${transaction['amount'].iloc[0]:.2f}")
                st.write(f"**Time**: {transaction['time_of_day'].iloc[0]:02d}:00")
                st.write(f"**Merchant**: {transaction['merchant_category'].iloc[0]}")
            
            with col2:
                st.write(f"**Device Trusted**: {'Yes' if transaction['device_trusted'].iloc[0] else 'No'}")
                st.write(f"**Location Match**: {'Yes' if transaction['location_match'].iloc[0] else 'No'}")
                st.write(f"**Country**: {transaction['country'].iloc[0]}")
            
            with col3:
                st.write(f"**Device Type**: {transaction['device_type'].iloc[0]}")
                st.write(f"**Account Age**: {transaction['account_age_days'].iloc[0]} days")
                st.write(f"**Credit Score**: {transaction['credit_score'].iloc[0]}")
            
            # Display prediction
            st.subheader("Prediction Result")
            
            if is_fraud_predicted:
                st.markdown(f"""
                <div class="fraud-alert">
                <h3>🚨 FRAUD ALERT</h3>
                <p><strong>Fraud Probability:</strong> {fraud_prob:.1%}</p>
                <p><strong>Threshold:</strong> {threshold:.1%}</p>
                <p>This transaction has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-transaction">
                <h3>✅ NORMAL TRANSACTION</h3>
                <p><strong>Fraud Probability:</strong> {fraud_prob:.1%}</p>
                <p><strong>Threshold:</strong> {threshold:.1%}</p>
                <p>This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show actual label if available
            if 'is_fraud' in transaction.columns and transaction['is_fraud'].iloc[0] is not None:
                actual_fraud = transaction['is_fraud'].iloc[0]
                if actual_fraud:
                    st.warning("⚠️ **Actual Label**: This transaction was actually fraudulent!")
                else:
                    st.success("✅ **Actual Label**: This transaction was actually legitimate!")
    
    else:
        st.info("👈 Please train a model using the sidebar to see performance metrics.")


if __name__ == "__main__":
    main()
