import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch import Tensor
from torch import cuda
import torch
import copy

class Container(Dataset):

    def __init__(self, datasrc, device: str|None = None, dtype: torch.dtype = torch.float32):
        """
        Abstract container for general datasets that can be used for ODEBench

        Args:
            datasrc (any): Source(s) of data to be imported.
        """

        # Determine local device
        if device is not None:
            self.device: str = device
        else:
            self.device: str = "cuda" if cuda.is_available() else 'cpu'

        # Set data type to be used throughout
        self.dtype: torch.dtype = dtype

        # Matrix for data (n, m)
        self.data: Tensor|None = None

        # Load in dataset
        self.load_data(datasrc)

        # Convert to PyTorch Tensor type
        if not isinstance(self.data, Tensor) and self.data is not None:
            self.data = Tensor(self.data).to(device=self.device, dtype=self.dtype)

        # Access index for dataloading
        self.dataIdx: np.ndarray = np.array(list(range(self.data.shape[0])))


    def train_test_split(self, split_ratio: float) -> tuple[Dataset, Dataset]:
        """
        Split the data into training and testing set.
        """
        
        cutoffIdx: int = int(split_ratio*self.data.shape[0])

        trainDS: Container = copy.deepcopy(self)
        trainDS.dataIdx = trainDS.dataIdx[:cutoffIdx]

        validDS: Container = copy.deepcopy(self)
        validDS.dataIdx = validDS.dataIdx[cutoffIdx:]

        return trainDS, validDS
        

    def shuffle(self, seed: int|None = None):

        if seed is not None:
            np.random.seed(seed)

        self.dataIdx: np.ndarray = np.random.permutation(self.data.shape[0])
    
    
    def __len__(self) -> int:
        """
        Returns total length of dataset.

        Returns:
            int: Total number of data records.
        """

        return len(self.dataIdx)
    

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        Method used by PyTorch's `DataLoader` to load in data as an iterator. Index `idx` passed in by PyTorch is of `int` data type
        and new data is retrieved one at a time.

        Args:
            idx (int): Data retrieval index.

        Returns:
            tuple[Tensor, Tensor, dict[str, Tensor]]: A tuple of model features and labels all can defined by user.
        """

        return (self.getModelFeatures(self.dataIdx[idx]), self.getModelLabels(self.dataIdx[idx]))
    

    def getModelFeatures(self, idx: int) -> Tensor:

        return self.data[idx, :-1]
    

    def getModelLabels(self, idx: int) -> Tensor:

        return self.data[idx, [-1]]


    def load_data(self, datasrc):
        """
        Load data method that detects the user input for data and calls the correct loader method.

        Args:
            datasrc (any): Dataset may be numpy arrays, pandas dataframes, filepaths, or a list of the above.

        Raises:
            ValueError: Invalid dataset variables povided.
            NotImplementedError: Unsupported method.
        """

        if isinstance(datasrc, str):

            assert os.path.isfile(datasrc), f"Specified dataset is assumed to be a file address at, but no valid file is found at {datasrc}."
            fileSuffix: str = datasrc.split(".")[-1]

            if fileSuffix == "npy":
                self.load_npy(datasrc)
            elif fileSuffix == "yaml":
                self.load_yaml(datasrc)
            else:
                raise ValueError("File found but not a supported file type.")
            
        elif isinstance(datasrc, pd.DataFrame):

            raise NotImplementedError

        elif isinstance(datasrc, list):

            for data in datasrc:
                self.load_data(data)

        elif isinstance(datasrc, np.ndarray):

            self.data = datasrc.copy()
            
        else:

            raise NotImplementedError

    
    def load_npy(self, filename: str):

        raise NotImplementedError
    

    def load_yaml(self, filename: str):

        raise NotImplementedError
    

    def load_df(self, df: pd.DataFrame):

        raise NotImplementedError