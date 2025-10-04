
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
    """
    Tests that the DataLoader iterates and creates batches in the expected format:
    a tuple of Tensors, where each Tensor is a stacked sample.
    """
    dataset = CompatibleDummyDataset(num_samples=10, feature_len=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    first_batch = next(iter(loader))
    

    assert isinstance(first_batch, tuple)
    
    assert len(first_batch) == 4
    assert all(isinstance(t, Tensor) for t in first_batch)
    
    tensor_1 = first_batch[0]
    expected_shape = (2, 4) 
    assert tensor_1.shape == expected_shape
    
    expected_content_1 = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ], dtype=np.float32)
    assert np.array_equal(tensor_1.data, expected_content_1)
    
    tensor_2 = first_batch[1]
    assert tensor_2.shape == expected_shape
    
    expected_content_2 = np.array([
        [1., 1., 1., 1.],
        [-1., -1., -1., -1.]
    ], dtype=np.float32)
    assert np.array_equal(tensor_2.data, expected_content_2)