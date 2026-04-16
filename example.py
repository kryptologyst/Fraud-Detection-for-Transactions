#!/usr/bin/env python3
"""Simple example demonstrating the fraud detection system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.synthetic_data import generate_transaction_data
from src.models.fraud_detector import FraudDetector
from src.eval.fraud_evaluator import FraudEvaluator
from src.utils import set_seed, setup_logging


def main():
    """Run a simple fraud detection example."""
    
    # Setup logging
    setup_logging("INFO")
    
    print("🛡️  Fraud Detection System Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Generate synthetic data
    print("📊 Generating synthetic transaction data...")
    users_df, transactions_df = generate_transaction_data(
        n_users=1000,
        n_transactions=10000,
        fraud_rate=0.05,
        seed=42
    )
    
    print(f"✅ Generated {len(transactions_df):,} transactions for {len(users_df):,} users")
    print(f"📈 Fraud rate: {transactions_df['is_fraud'].mean():.1%}")
    
    # Split data (time-based to avoid leakage)
    transactions_df = transactions_df.sort_values('timestamp')
    split_idx = int(len(transactions_df) * 0.8)
    
    train_df = transactions_df.iloc[:split_idx]
    test_df = transactions_df.iloc[split_idx:]
    
    print(f"📚 Training set: {len(train_df):,} transactions")
    print(f"🧪 Test set: {len(test_df):,} transactions")
    
    # Train XGBoost model
    print("\n🤖 Training XGBoost fraud detection model...")
    model = FraudDetector(
        model_type="xgboost",
        use_imbalanced_learning=True,
        use_graph_features=True,
        random_state=42
    )
    
    model.fit(train_df)
    print("✅ Model training completed")
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    evaluator = FraudEvaluator()
    
    # Get predictions
    y_true = test_df['is_fraud']
    y_proba = model.predict_proba(test_df)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Calculate key metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    
    print("\n📈 Model Performance:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"   AUC-ROC:   {auc_roc:.3f}")
    print(f"   AUC-PR:    {auc_pr:.3f}")
    
    # Test individual prediction
    print("\n🎯 Testing individual transaction prediction...")
    
    # Sample a transaction
    sample_transaction = test_df.sample(1)
    fraud_prob = model.predict_proba(sample_transaction)[0]
    is_fraud_predicted = fraud_prob >= 0.5
    
    print(f"\n📋 Sample Transaction:")
    print(f"   Amount: ${sample_transaction['amount'].iloc[0]:.2f}")
    print(f"   Time: {sample_transaction['time_of_day'].iloc[0]:02d}:00")
    print(f"   Merchant: {sample_transaction['merchant_category'].iloc[0]}")
    print(f"   Device Trusted: {'Yes' if sample_transaction['device_trusted'].iloc[0] else 'No'}")
    
    print(f"\n🔮 Prediction:")
    print(f"   Fraud Probability: {fraud_prob:.1%}")
    print(f"   Predicted: {'🚨 FRAUD' if is_fraud_predicted else '✅ Normal'}")
    
    # Show actual label
    actual_fraud = sample_transaction['is_fraud'].iloc[0]
    print(f"   Actual: {'🚨 FRAUD' if actual_fraud else '✅ Normal'}")
    
    # Check if prediction was correct
    correct = (is_fraud_predicted == actual_fraud)
    print(f"   Result: {'✅ Correct' if correct else '❌ Incorrect'}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\n⚠️  DISCLAIMER: This is a research/educational tool.")
    print("   NOT suitable for production fraud detection.")
    print("   All data is synthetic and contains no real personal information.")


if __name__ == "__main__":
    main()
