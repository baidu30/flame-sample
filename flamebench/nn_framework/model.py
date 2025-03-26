import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

from .loss import TorchLoss
from ..dataset_tools.container import Container
from tqdm import tqdm


class TorchModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss: TorchLoss = TorchLoss()
        self.optim: torch.optim.Optimizer|None = None

        # TODO: To consider whether or not to implement simple functions or reuse TorchLoss for evaluation metrics
        self.metricFunc: list[TorchLoss] = list()


    def fit(self, trainDS: Container, validDS: Container, epochs: int, batch: int = 64, verbose: bool = False) -> tuple[list[float], list[list[float]]]:
        
        assert self.optim is not None, "Optimizer not yet set! Use `set_optim` method to set the optimizer."
        assert trainDS.device == validDS.device, f"Device used for training and validation containers must be the same. Received {trainDS.device} and {validDS.device}."

        self.to(device=trainDS.device)
        trainDL: DataLoader = DataLoader(trainDS, batch)
        validDL: DataLoader = DataLoader(validDS, validDS.__len__())

        training_hist: list = list()
        validation_hist: list = list()

        for epoch in range(epochs):

            running_loss = 0.0

            with tqdm(total=len(trainDL), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", disable=verbose==False) as pbar:
                dataSize = 0
                for batchIdx, data in enumerate(trainDL):
                    lossOutput = self.step(data)
                    running_loss += lossOutput.item()
                    dataSize += data[0].shape[0]
                    pbar.set_postfix(loss=running_loss/dataSize)
                    pbar.update(1)

                metrics = self.evaluate(validDL, verbose)

                training_hist.append(running_loss/dataSize)
                validation_hist.append(metrics)

        return (training_hist, validation_hist)


    def step(self, data: tuple):

        features, labels = data

        prediction = self(features)

        lossOutput = self.loss.getLoss(prediction, labels, features)
        self.optim.zero_grad()
        lossOutput.backward()
        self.optim.step()

        return lossOutput

    
    def evaluate(self, data: DataLoader, verbose: bool = False) -> list[float]:

        metrics = list()

        if len(self.metricFunc) == 0:
            metricFuncs = [self.loss.getLoss]
        else:
            metricFuncs = self.metricFunc
        
        for batchIdx, data in enumerate(data):

            features, labels = data
            prediction = self(features)

        for metricIdx in range(len(metricFuncs)):
            metrics.append(metricFuncs[metricIdx](prediction, labels, features).item())

        if verbose:
            metricOutput: str = "Metrics = ["
            for metric in metrics:
                metricOutput += f"{metric}, "
            print(metricOutput + "]")

        return metrics


    def set_optim(self, optim: torch.optim.Optimizer):
        self.optim = optim

    
    def set_loss(self, loss: TorchLoss):
        self.loss = loss


    def set_seed(seed: int, deterministic: bool = False, benchmark: bool = False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


class MLPModel(TorchModel):

    def __init__(self, layers: list[int], activation: nn.Module = nn.GELU()):
        super().__init__()

        self.torchLayers: nn.ModuleList = nn.ModuleList()
        for idx in range(len(layers)-2):
            self.torchLayers.append(nn.Linear(layers[idx], layers[idx+1]))
            self.torchLayers.append(activation)
        self.torchLayers.append(nn.Linear(layers[-2], layers[-1]))


    def forward(self, x):
        
        for layer in self.torchLayers:
            x = layer(x)

        return x