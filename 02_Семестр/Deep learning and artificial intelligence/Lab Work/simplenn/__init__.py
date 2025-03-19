"""
SimpleNN - легковесный нейросетевой фреймворк на Python.
"""

# Импортируем основные компоненты, чтобы они были доступны из корневого пакета
from simplenn.core.tensor import Tensor
from simplenn.core.activation import (
    Activation, Sigmoid, ReLU, LeakyReLU, Tanh, Softmax
)
from simplenn.core.loss import (
    Loss, MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy
)
from simplenn.core.layer import (
    Layer, Dense, Dropout, BatchNormalization
)
from simplenn.core.optimizer import (
    Optimizer, SGD, MomentumSGD, Adagrad, RMSprop, 
    Adam, GradientClipping
)
from simplenn.core.model import Model
from simplenn.utils.data_utils import DataUtils

__version__ = "0.1.0"
__author__ = "SimpleNN Team"