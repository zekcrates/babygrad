"""
A simple example of training a two-layer neural network on MNIST
using the babygrad library.
"""
import struct
import gzip
import numpy as np
from baby import Tensor, ops

def parse_mnist(image_filename, label_filename):
    """
    Reads an image and label file in MNIST format.

    Args:
        image_filename (str): Path to the gzipped image file.
        label_filename (str): Path to the gzipped label file.

    Returns:
        Tuple (X, y):
            X (np.ndarray): Images as a (num_examples, 784) array.
            y (np.ndarray): Labels as a (num_examples,) array.
    """
    with gzip.open(image_filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols)
    with gzip.open(label_filename, "rb") as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    normalized_images = images.astype(np.float32) / 255.0
    return normalized_images, labels


class SimpleNN:
    """A simple two-layer neural network."""
    def __init__(self, input_size, hidden_size, num_classes):
        self.W1 = Tensor(np.random.randn(input_size, hidden_size).astype(np.float32) / np.sqrt(hidden_size))
        self.W2 = Tensor(np.random.randn(hidden_size, num_classes).astype(np.float32) / np.sqrt(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the network."""
        # print("xshape : ",x.shape)
        z1 = x @ self.W1 # (128,784) @ (784, 100) -> (128,100)
        a1 = ops.relu(z1)
        logits = a1 @ self.W2  # (128,100) @ (100,10) -> (128,10)
        return logits

    def parameters(self):
        """Returns a list of all model parameters."""
        return [self.W1, self.W2]


def softmax_loss(logits: Tensor, y_true: Tensor) -> Tensor:
    """
    Computes the softmax cross-entropy loss.

    Args:
        logits (Tensor): The raw output scores from the model.
        y_true (Tensor): The one-hot encoded true labels.

    Returns:
        A scalar tensor representing the average loss.
    """
    batch_size = logits.shape[0]
    log_sum_exp = ops.log(ops.exp(logits).sum(axes=1))
    z_y = (logits * y_true).sum(axes=1)
    loss = log_sum_exp - z_y
    return loss.sum() / batch_size


def train_epoch(model: SimpleNN, X_train: np.ndarray, y_train: np.ndarray, lr: float, batch_size: int):
    """
    Runs a single training epoch for the model.
    """
    num_examples = X_train.shape[0]
    for i in range(0, num_examples, batch_size):
        x_batch_np = X_train[i:i+batch_size]
        y_batch_np = y_train[i:i+batch_size]

        x_batch = Tensor(x_batch_np)

        logits = model.forward(x_batch)

        num_classes = logits.shape[1]
        y_one_hot_np = np.zeros((y_batch_np.shape[0], num_classes), dtype=np.float32)
        y_one_hot_np[np.arange(y_batch_np.shape[0]), y_batch_np] = 1
        y_one_hot = Tensor(y_one_hot_np)

        loss = softmax_loss(logits, y_one_hot)
        for p in model.parameters():
            p.grad = None
        loss.backward()
        for p in model.parameters():
            p.data = p.data - lr * p.grad

        preds = logits.data.argmax(axis=1)
        acc = np.mean(preds == y_batch_np)

        print(f"  Batch {i//batch_size+1:3d}: Loss = {loss.data:.4f}, Accuracy = {acc*100:.2f}%")


if __name__ == "__main__":
    EPOCHS = 10
    LEARNING_RATE = 0.1
    
    BATCH_SIZE = 128  
    NUM_STEPS_TO_RUN = 20
    NUM_SAMPLES = BATCH_SIZE * NUM_STEPS_TO_RUN 

    INPUT_SIZE = 784
    HIDDEN_SIZE = 100
    NUM_CLASSES = 10

    print("Loading MNIST data...")
    X_train, y_train = parse_mnist("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    X_test, y_test = parse_mnist("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")
    
    print(f"Using a subset of {NUM_SAMPLES} images to run for {NUM_STEPS_TO_RUN} steps...")
    X_train = X_train[:NUM_SAMPLES]
    y_train = y_train[:NUM_SAMPLES]
    print("Data loaded.\n")

    np.random.seed(42) 
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_epoch(model, X_train, y_train, LEARNING_RATE, BATCH_SIZE)
        print("-" * 20)