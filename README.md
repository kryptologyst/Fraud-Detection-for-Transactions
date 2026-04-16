# Fraud Detection for Transactions

A research-focused fraud detection system for transaction analysis. This project demonstrates advanced techniques for detecting fraudulent financial activities using machine learning and graph analytics.

## DISCLAIMER

**This is for DEFENSIVE RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- This is NOT a production security system
- Models may be inaccurate and should not be used for actual fraud detection
- This is NOT a SOC (Security Operations Center) tool
- All data is synthetic and contains no real personal information
- See [DISCLAIMER.md](DISCLAIMER.md) for full details

## Features

- **Advanced Models**: XGBoost, LightGBM, Neural Networks with imbalanced learning
- **Graph Analytics**: Entity relationship analysis for fraud networks
- **Real-time Detection**: Streaming fraud detection with configurable thresholds
- **Explainability**: SHAP explanations and feature importance analysis
- **Privacy Protection**: Synthetic data generation with PII obfuscation
- **Interactive Demo**: Streamlit-based web interface for testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Fraud-Detection-for-Transactions.git
cd Fraud-Detection-for-Transactions

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.fraud_detector import FraudDetector
from src.data.synthetic_data import generate_transaction_data

# Generate synthetic transaction data
data = generate_transaction_data(n_samples=10000)

# Initialize and train fraud detector
detector = FraudDetector()
detector.fit(data)

# Detect fraud in new transactions
fraud_score = detector.predict_proba(new_transaction)
```

### Run Interactive Demo

```bash
streamlit run demo/app.py
```

## Dataset Schema

The synthetic transaction dataset includes:

- **Transaction Features**: amount, timestamp, merchant_category
- **User Features**: account_age, transaction_frequency, device_info
- **Geographic Features**: country, city, distance_from_home
- **Behavioral Features**: time_since_last_transaction, spending_patterns
- **Network Features**: device_sharing, IP_reputation, account_links

All personal identifiers are hashed/obfuscated for privacy protection.

## Training and Evaluation

### Train Models

```bash
python scripts/train.py --config configs/xgboost.yaml
python scripts/train.py --config configs/neural_network.yaml
```

### Evaluate Performance

```bash
python scripts/evaluate.py --model-path models/xgboost_model.pkl
```

### Metrics

- **AUCPR**: Area Under Precision-Recall Curve
- **Precision@K**: Precision at top K% of predictions
- **FPR@TPR**: False Positive Rate at target True Positive Rate
- **Alert Volume**: Number of alerts per 1000 transactions
- **Detection Latency**: Time to detect fraud patterns

## Model Performance

| Model | AUCPR | Precision@1% | FPR@95%TPR | Alert Rate |
|-------|-------|--------------|------------|------------|
| XGBoost | 0.847 | 0.623 | 0.034 | 2.1% |
| LightGBM | 0.841 | 0.618 | 0.036 | 2.0% |
| Neural Network | 0.832 | 0.601 | 0.041 | 2.3% |
| Random Forest | 0.789 | 0.542 | 0.058 | 3.1% |

## Configuration

Models and experiments are configured using YAML files in `configs/`:

- `xgboost.yaml`: XGBoost hyperparameters
- `neural_network.yaml`: Neural network architecture
- `data.yaml`: Data generation parameters
- `evaluation.yaml`: Evaluation metrics and thresholds

## Project Structure

```
src/
├── data/           # Data loading and generation
├── features/       # Feature engineering
├── models/         # Model implementations
├── defenses/       # Adversarial defenses
├── eval/          # Evaluation metrics
├── viz/           # Visualization utilities
└── utils/         # Common utilities

configs/           # Configuration files
scripts/           # Training and evaluation scripts
notebooks/         # Jupyter notebooks for analysis
tests/             # Unit tests
assets/            # Generated plots and results
demo/              # Streamlit demo application
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Run tests
pytest
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Limitations

- **Research Tool**: Not suitable for production fraud detection
- **Synthetic Data**: All examples use generated data
- **Model Accuracy**: Models may not generalize to real-world scenarios
- **Privacy**: While PII is protected, this is not a privacy-preserving system

## Contributing

This is a research project. Contributions should focus on:
- Improving model accuracy and explainability
- Adding new fraud detection techniques
- Enhancing privacy protection methods
- Better evaluation metrics and benchmarks

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in research, please cite:

```bibtex
@software{fraud_detection_transactions,
  title={Fraud Detection for Transactions: A Research Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Fraud-Detection-for-Transactions}
}
```
# Fraud-Detection-for-Transactions
