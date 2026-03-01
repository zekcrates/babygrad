import numpy as np

from baby.utils import parse_mnist
from .tensor import Tensor
from typing import List, Optional
import gzip 
import struct 
class Dataset:
    
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
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.batch_idx = 0
        # Ceiling division ensures we get the partial last batch
        self.num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
            
        start = self.batch_idx * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        
        samples = [self.dataset[i] for i in batch_indices]

        unzipped_samples = zip(*samples)
        
        all_arrays = [np.stack(s) for s in unzipped_samples]
        batch = tuple(Tensor(arr) for arr in all_arrays)
        
        self.batch_idx += 1
        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

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
    



class RandomFlipHorizontal:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # img shape: (3, 32, 32) -> (C, H, W)
        if np.random.rand() < self.p:
            # Flip along the last axis (Width)
            return img[:, :, ::-1] 
        return img

class RandomCrop:
    def __init__(self, padding=4):
        self.padding = padding

    def __call__(self, img):
        # img shape: (3, 32, 32)
        c, h, w = img.shape
        
        # 1. Pad only the H and W dimensions with zeros
        padded = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # 2. Pick a random top-left corner
        top = np.random.randint(0, 2 * self.padding + 1)
        left = np.random.randint(0, 2 * self.padding + 1)
        
        # 3. Crop back to original size
        return padded[:, top:top+h, left:left+w]

import os
import pickle
from typing import Optional, List
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    batch[b'filenames'] = [name.decode('utf-8') for name in batch[b'filenames']]
    return batch


class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder: str, train: bool,
                 p: Optional[int] = 0.5,
                 transforms: Optional[List] = None):
        super().__init__(transforms)
        self.base_folder = base_folder
        self.transforms = transforms
        self.p = p
        self.train = train

        if train:
            data_list = []
            label_list = []
            for i in range(1, 6):
                batch = unpickle(f"{base_folder}/data_batch_{i}")
                data_list.append(batch[b'data'])
                label_list.extend(batch[b'labels'])
            self.images = np.vstack(data_list)
            self.labels = np.array(label_list)
        else:
            batch = unpickle(f"{base_folder}/test_batch")
            self.images = batch[b'data']
            self.labels = np.array(batch[b'labels'])


    def __getitem__(self, index):
        if isinstance(index, slice):
            images_flat = np.array(self.images[index]) / 255
            images_reshaped = images_flat.reshape(-1, 3, 32, 32)
            labels_batch = np.array(self.labels[index])
            return (images_reshaped, labels_batch)

        
        sample_image = self.images[index] / 255
        sample_label = self.labels[index]

        new_sample_image = sample_image.reshape(3, 32, 32)

        if self.transforms:
            for tform in self.transforms:
                new_sample_image = tform(new_sample_image)

        return (new_sample_image, sample_label)

    def __len__(self):
        return len(self.images)
