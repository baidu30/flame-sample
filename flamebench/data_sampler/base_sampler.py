import numpy as np
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    def __init__(self, config):
        self.config = config
        self.data = None
        
    @abstractmethod
    def sample(self):
        """执行采样逻辑"""
        pass
    
    def save_as_npy(self, save_path):
        np.save(save_path, self.data)
        
    def get_metadata(self):
        return {
            'variables': self._get_variable_names(),
            'scenario': self.config.scenario_type
        }
    
    @abstractmethod
    def _get_variable_names(self):
        """返回数据变量名称列表"""
        pass