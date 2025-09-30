"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

from baby import Tensor, ops
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
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

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
    train_images  =   load_mnist_images(image_filesname)
    #print(train_images.shape)
    # train_images = train_images.reshape((10000, 784))
    train_images = train_images.astype(np.float32)
    train_images = (train_images - np.min(train_images)) / (np.max(train_images) - np.min(train_images))
    train_labels = load_mnist_labels(label_filename)
    
    ### END YOUR CODE
    return train_images , train_labels 


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (Tensor[np.float32])
    """

    log_sum_exp = ops.log(ops.exp(Z).sum(axes=1))
    correct = (Z*y_one_hot).sum(axes=1)
    loss = log_sum_exp - correct
    batch_size = Z.shape[0]
    return ops.summation(loss) / batch_size


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: Tensor[np.float32]
            W2: Tensor[np.float32]
    """

    num_examples = X.shape[0]
    
    num_classes = W2.shape[1]
    for i in range(0 ,num_examples,batch):
        x_batch = Tensor(X[i:i+batch]) 
        y_batch = y[i:i+batch]

        
        a = x_batch @ W1 
        b = ops.relu(a)
        logits  = b @ W2 


        y_one_hot = np.zeros((y_batch.shape[0], num_classes), dtype=np.float32)
        y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1
        y_one_hot = Tensor(y_one_hot)


        
        loss = softmax_loss(logits, y_one_hot)

        W1.grad = None
        W2.grad = None
        loss.backward() 
        # print("loss : ", loss)
        preds = logits.numpy().argmax(axis=1)
        acc = np.mean(preds == y_batch)

        print(f"loss: {loss.numpy():.4f}, acc: {acc*100:.2f}%")
        W1 = Tensor(W1.numpy() - lr * W1.grad)
        W2 = Tensor(W2.numpy() - lr * W2.grad)



    return (W1,W2)


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )

np.random.seed(1)
W1 = Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
W2 = Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
nn_epoch(X, y, W1, W2, lr=0.2, batch=100)