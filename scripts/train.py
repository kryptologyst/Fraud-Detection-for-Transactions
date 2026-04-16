#!/usr/bin/env python3
"""Training script for fraud detection models."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.synthetic_data import generate_transaction_data
from src.models.fraud_detector import FraudDetector
from src.eval.fraud_evaluator import FraudEvaluator
from src.utils import set_seed, setup_logging, load_config, save_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--save-data", action="store_true", help="Save generated data")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_file = config.get("logging.log_file", None)
    setup_logging(config.get("logging.level", "INFO"), log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting fraud detection model training...")
    logger.info(f"Configuration: {args.config}")
    
    # Set random seed
    seed = config.get("data.seed", 42)
    set_seed(seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    logger.info("Generating synthetic transaction data...")
    users_df, transactions_df = generate_transaction_data(
        n_users=config.get("data.n_users", 1000),
        n_transactions=config.get("data.n_transactions", 10000),
        fraud_rate=config.get("data.fraud_rate", 0.05),
        seed=seed
    )
    
    logger.info(f"Generated {len(transactions_df):,} transactions for {len(users_df):,} users")
    logger.info(f"Fraud rate: {transactions_df['is_fraud'].mean():.3f}")
    
    # Save data if requested
    if args.save_data:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        users_df.to_parquet(data_dir / "users.parquet", index=False)
        transactions_df.to_parquet(data_dir / "transactions.parquet", index=False)
        
        logger.info(f"Data saved to {data_dir}")
    
    # Split data
    logger.info("Splitting data into train/test sets...")
    
    # Use time-based split to avoid data leakage
    transactions_df = transactions_df.sort_values('timestamp')
    split_idx = int(len(transactions_df) * (1 - args.test_size))
    
    train_df = transactions_df.iloc[:split_idx]
    test_df = transactions_df.iloc[split_idx:]
    
    logger.info(f"Train set: {len(train_df):,} transactions")
    logger.info(f"Test set: {len(test_df):,} transactions")
    
    # Initialize model
    model_type = config.get("model.type", "xgboost")
    logger.info(f"Initializing {model_type} model...")
    
    model = FraudDetector(
        model_type=model_type,
        use_imbalanced_learning=config.get("features.use_imbalanced_learning", True),
        use_graph_features=config.get("features.use_graph_features", True),
        random_state=seed
    )
    
    # Train model
    logger.info("Training model...")
    model.fit(train_df)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = FraudEvaluator()
    
    # Evaluate at multiple thresholds
    thresholds = config.get("evaluation.thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    results = evaluator.evaluate_model(model, test_df, test_df['is_fraud'], thresholds)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results)
    logger.info("Model evaluation completed")
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(leaderboard.to_string(index=False))
    
    # Generate detailed report
    report = evaluator.generate_report(
        model, test_df, test_df['is_fraud'], threshold=0.5
    )
    
    # Save model and results
    model_path = output_dir / f"{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    results_path = output_dir / f"{model_type}_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    leaderboard_path = output_dir / f"{model_type}_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    
    report_path = output_dir / f"{model_type}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Leaderboard saved to {leaderboard_path}")
    logger.info(f"Report saved to {report_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
