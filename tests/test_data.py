
import pytest
import numpy as np

from baby.tensor import Tensor
from baby.data import Dataset, DataLoader

class CompatibleDummyDataset(Dataset):
    """
    A dummy dataset whose __getitem__ returns a tuple of numpy arrays 
    with the same shape, making them stackable.
    """
    def __init__(self, num_samples=20, feature_len=4):
        super().__init__()
        self.num_samples = num_samples
        self.feature_len = feature_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index >= self.num_samples:
            raise IndexError("Index out of range")
        
        data1 = np.full((self.feature_len,), index, dtype=np.float32)
        data2 = np.full((self.feature_len,), -index, dtype=np.float32)
        
        return data1, data2
def test_custom_dataloader_iteration():
    dataset = CompatibleDummyDataset(num_samples=10, feature_len=4)
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    it = iter(loader)
    next(it) # Batch 1: indices 0,1,2,3
    next(it) # Batch 2: indices 4,5,6,7
    final_batch = next(it) # Batch 3: indices 8,9
    
    assert final_batch[0].shape == (2, 4)
    
    expected_final_data = np.array([
        [8., 8., 8., 8.],
        [9., 9., 9., 9.]
    ], dtype=np.float32)
    
    assert np.array_equal(final_batch[0].data, expected_final_data), "Final batch data mismatch"
def test_dataloader_shuffling():
    dataset = CompatibleDummyDataset(num_samples=100, feature_len=1)
    loader_1 = DataLoader(dataset, batch_size=10, shuffle=True)
    loader_2 = DataLoader(dataset, batch_size=10, shuffle=True)
    
    batch1 = next(iter(loader_1))[0].data
    batch2 = next(iter(loader_2))[0].data
    
    assert not np.array_equal(batch1, batch2), "Shuffle=True resulted in identical batches."
