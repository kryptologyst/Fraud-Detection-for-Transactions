"""Tests for fraud detection system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.data.synthetic_data import generate_transaction_data, TransactionDataGenerator
from src.models.fraud_detector import FraudDetector
from src.eval.fraud_evaluator import FraudEvaluator
from src.utils import set_seed, get_device


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_generate_transaction_data(self):
        """Test transaction data generation."""
        users_df, transactions_df = generate_transaction_data(
            n_users=100,
            n_transactions=1000,
            fraud_rate=0.05,
            seed=42
        )
        
        assert len(users_df) == 100
        assert len(transactions_df) == 1000
        assert 'is_fraud' in transactions_df.columns
        assert 'amount' in transactions_df.columns
        assert 'user_id' in transactions_df.columns
        
        # Check fraud rate is approximately correct
        fraud_rate = transactions_df['is_fraud'].mean()
        assert 0.03 <= fraud_rate <= 0.07  # Allow some variance
    
    def test_data_generator(self):
        """Test TransactionDataGenerator class."""
        generator = TransactionDataGenerator(seed=42)
        users_df, transactions_df = generator.generate_dataset(
            n_users=50,
            n_transactions=500,
            fraud_rate=0.1
        )
        
        assert len(users_df) == 50
        assert len(transactions_df) == 500
        assert transactions_df['is_fraud'].mean() > 0


class TestFraudDetector:
    """Test fraud detection models."""
    
    def test_fraud_detector_initialization(self):
        """Test fraud detector initialization."""
        detector = FraudDetector(
            model_type="xgboost",
            use_imbalanced_learning=True,
            use_graph_features=True,
            random_state=42
        )
        
        assert detector.model_type == "xgboost"
        assert detector.use_imbalanced_learning is True
        assert detector.use_graph_features is True
        assert detector.random_state == 42
    
    def test_fraud_detector_training(self):
        """Test fraud detector training."""
        # Generate small dataset for testing
        users_df, transactions_df = generate_transaction_data(
            n_users=50,
            n_transactions=500,
            fraud_rate=0.1,
            seed=42
        )
        
        detector = FraudDetector(
            model_type="random_forest",  # Use faster model for testing
            use_imbalanced_learning=False,
            use_graph_features=False,
            random_state=42
        )
        
        # Train model
        detector.fit(transactions_df)
        
        # Test prediction
        probabilities = detector.predict_proba(transactions_df.head(10))
        predictions = detector.predict(transactions_df.head(10))
        
        assert len(probabilities) == 10
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in probabilities)
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Generate test data
        users_df, transactions_df = generate_transaction_data(
            n_users=50,
            n_transactions=500,
            fraud_rate=0.1,
            seed=42
        )
        
        detector = FraudDetector(
            model_type="random_forest",
            use_imbalanced_learning=False,
            use_graph_features=False,
            random_state=42
        )
        
        detector.fit(transactions_df)
        
        # Evaluate model
        metrics = detector.evaluate(transactions_df)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        
        # Check metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1


class TestFraudEvaluator:
    """Test fraud evaluation metrics."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = FraudEvaluator()
        assert evaluator.results == {}
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        evaluator = FraudEvaluator()
        
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.3, 0.1, 0.9, 0.2, 0.1, 0.7, 0.2])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        
        # Check specific values
        assert metrics['accuracy'] == 0.8  # 8/10 correct
        assert metrics['precision'] == 1.0  # 3/3 predicted fraud are correct
        assert metrics['recall'] == 0.75  # 3/4 actual fraud detected


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        # Should be one of the expected devices
        assert str(device) in ['cpu', 'cuda', 'mps']


if __name__ == "__main__":
    pytest.main([__file__])
