import numpy as np

from baby.utils import parse_mnist
from .tensor import Tensor
from typing import List, Optional
import gzip 
import struct 
class Dataset:
    """An Base class representing a dataset.

    This is the base class for all datasets. Your own custom dataset should
    inherit from this class and at least override the `__len__` method
    (which returns the size of the dataset) and the `__getitem__` method
    (which supports fetching a data sample at a given index).

    Args:
        transforms (list, optional): A list of functions or callable objects
            that take a data sample and return a transformed version. Applied
            in the order they are provided. Defaults to None.

    Example:
        >>> class MyNumberDataset(Dataset):
        ...     def __init__(self, numbers):
        ...         super().__init__()
        ...         self.numbers = numbers
        ...     def __len__(self):
        ...         return len(self.numbers)
        ...     def __getitem__(self, index):
        ...         # Returns a tuple of (number, number_squared)
        ...         return self.numbers[index], self.numbers[index] ** 2
        ...
        >>> dataset = MyNumberDataset([1, 2, 3, 4])
        >>> print(f"Dataset size: {len(dataset)}")
        Dataset size: 4
        >>> print(f"Third sample: {dataset[2]}")
        Third sample: (3, 9)
    """
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def apply_transform(self, x):
        if not self.transforms:
            return x
        for t in self.transforms:
            x = t(x)
        return x


class DataLoader:
    """Provides an iterator for easy batching, shuffling, and loading of data.

    This wraps a `Dataset` and allows you to easily loop over your data in
    mini-batches. It handles shuffling and makes sure the last batch is included
    even if the dataset size isn't perfectly divisible by the batch size.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            Defaults to 1.
        shuffle (bool, optional): Set to `True` to have the data reshuffled
            at every epoch (highly recommended for training). Defaults to `True`.

    Example:
        >>> # Create a simple dataset that works with this DataLoader's logic.
        >>> # __getitem__ must return items with the same shape to be stackable.
        >>> class MyStackableDataset(Dataset):
        ...     def __len__(self):
        ...         return 10
        ...     def __getitem__(self, index):
        ...         return np.array([index, index]), np.array([-index, -index])
        ...
        >>> dataset = MyStackableDataset()
        >>> # Create a DataLoader to iterate over the dataset in batches.
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=False)
        >>> # The loader can be used in a for-loop.
        >>> for batch in loader:
        ...     # The batch is a tuple of Tensors, one Tensor per sample.
        ...     print(f"Batch contains {len(batch)} samples.")
        ...     print(f"First sample's Tensor shape: {batch[0].shape}")
        ...     break
        Batch contains 4 samples.
        First sample's Tensor shape: (2, 2)
    """
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.batch_idx = 0
        self.num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        start = self.batch_idx * self.batch_size
        batch_indices = self.indices[start: start+self.batch_size]
        samples = [self.dataset[i] for i in batch_indices]


        unzipped_samples = zip(*samples)

        all_arrays = [np.stack(s) for s in unzipped_samples]

        batch = tuple(Tensor(arr) for arr in all_arrays)
        self.batch_idx += 1
        return batch
    

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, self.labels = parse_mnist(image_filesname=image_filename, label_filename=label_filename) 
        self.transforms = transforms


    def __getitem__(self, index) -> object:
        if isinstance(index, slice):
            #lets take [0:5]
            #self.images[0:5]
            # we get back (5,28,28)
            
            images_batch_flat = np.array(self.images[index], dtype=np.float32)
            
            # images_batch_reshaped = images_batch_flat.reshape(len(images_batch_flat), 784)
            images_batch_reshaped = images_batch_flat.reshape(-1, 28, 28, 1)
            #we convert into # (5,28,28,1)
            
            labels_batch = np.array(self.labels[index])
            return (images_batch_reshaped, labels_batch)
        
        else:  #single index , return directly 
            sample_image = self.images[index]
            sample_label = self.labels[index]
            
            np_sample_image = np.array(sample_image, dtype=np.float32).reshape(28, 28, 1)
            np_sample_label = np.array(sample_label)
            # np_sample_image = np.array(sample_image, dtype=np.float32).reshape(784)
            # np_sample_label = np.array(sample_label)

            if self.transforms is not None:
                for tform in self.transforms:
                    np_sample_image = tform(np_sample_image)

            return (np_sample_image, np_sample_label)
    def __len__(self) -> int:
        return len(self.images)