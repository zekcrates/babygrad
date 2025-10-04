

import struct 
import gzip
import numpy as np 
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f :
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        image_data = np.frombuffer(f.read() , dtype=np.uint8)
        images = image_data.reshape(num_images, rows*cols)
        
        
        return images 
    
def load_mnist_labels(filename):
    with gzip.open(filename , "rb") as f :
        magic ,num_labels = struct.unpack('>II', f.read(8))
        #print("num labels : ", num_labels)

        label = np.frombuffer(f.read() , dtype=np.uint8) 
        return label 
    

def parse_mnist(image_filesname, label_filename):
    """
    Args:
        image_filename (str): name of  image.
        label_filename (str): name of  labels (0,1,2,3,...9)

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    train_images  =   load_mnist_images(image_filesname).astype(np.float32)
    train_images = (train_images - np.min(train_images)) / (np.max(train_images) - np.min(train_images))
    train_labels = load_mnist_labels(label_filename)    
    return train_images , train_labels 
