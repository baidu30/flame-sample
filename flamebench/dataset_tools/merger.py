import numpy as np

class DatasetMerger:
    @staticmethod
    def merge(datasets, axis=0):
        """合并多个npy数据集"""
        merged_data = np.concatenate(datasets, axis=axis)
        return merged_data