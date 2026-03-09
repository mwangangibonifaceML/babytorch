import numpy as np

from minitorch.tensor.tensor import Tensor
from minitorch.zexamples.loan_default.src.components.data_transformation import DataTransformation
from minitorch.zexamples.loan_default.src.pipeline.model_pipeline import LoanDefaultPredictor
from minitorch.losses.losses import BinaryCrossEntropyLoss
from minitorch.optimizers.optim import SGD
from minitorch.train.training import Trainer