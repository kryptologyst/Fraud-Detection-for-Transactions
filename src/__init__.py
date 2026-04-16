"""Main source package."""

from .data import generate_transaction_data, TransactionDataGenerator
from .models import FraudDetector, FraudNeuralNetwork
from .eval import FraudEvaluator
from .viz import FraudExplainer
from .utils import set_seed, get_device, setup_logging, Config

__version__ = "1.0.0"
__all__ = [
    "generate_transaction_data",
    "TransactionDataGenerator", 
    "FraudDetector",
    "FraudNeuralNetwork",
    "FraudEvaluator",
    "FraudExplainer",
    "set_seed",
    "get_device", 
    "setup_logging",
    "Config"
]
