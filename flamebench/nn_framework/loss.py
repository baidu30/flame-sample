import torch
from torch import Tensor

class TorchLoss:

    def __init__(self, *args, **kwargs):
        pass
            

    def getLoss(self, labels: Tensor, predictions: Tensor, features: Tensor, **kwargs) -> Tensor:
        return torch.nn.MSELoss()(predictions, labels)