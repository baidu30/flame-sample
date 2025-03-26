from .base_sampler import BaseSampler
from cantera import Solution
import numpy as np

class ZeroDSampler(BaseSampler):
    def sample(self):
        # todo : how to sample data
        # gas = Solution(self.config.mechanism)
        # self.data = np.array(sampled_data)
        pass
        
    def _get_variable_names(self):
        return ['temperature', 'pressure', '...']