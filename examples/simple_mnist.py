
import struct
import gzip
import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Module, Linear, ReLU, Sequential, SoftmaxLoss, LayerNorm1d, BatchNorm1d

def parse_mnist(image_filename, label_filename):
    """
    Reads an image and label file in MNIST format.
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


def train_epoch(model: Module, loss_fn: Module, optimizer: Adam, X_train: np.ndarray, y_train: np.ndarray, batch_size: int):
    """
    Runs a single training epoch for the model.
    """
    num_examples = X_train.shape[0]
    for i in range(0, num_examples, batch_size):
        x_batch_np = X_train[i:i+batch_size]
        y_batch_np = y_train[i:i+batch_size]
        x_batch = Tensor(x_batch_np)

        logits = model(x_batch)

        
        loss = loss_fn(logits, y_batch_np)

        optimizer.reset_grad()
        loss.backward()
        optimizer.step()

        preds = logits.data.argmax(axis=1)
        acc = np.mean(preds == y_batch_np)

        print(f"   Batch {i//batch_size+1:3d}: Loss = {loss.data:.4f}, Accuracy = {acc*100:.2f}%")


def evaluate(model: Module, X_test: np.ndarray, y_test: np.ndarray, batch_size: int):
    """
    Evaluates the model on the test set.
    """
    model.eval() 
    
    num_examples = X_test.shape[0]
    total_acc = 0
    total_count = 0
    
    for i in range(0, num_examples, batch_size):
        x_batch_np = X_test[i:i+batch_size]
        y_batch_np = y_test[i:i+batch_size]
        x_batch = Tensor(x_batch_np)

        logits = model(x_batch)
        preds = logits.data.argmax(axis=1)
        
        
        total_acc += np.sum(preds == y_batch_np)
        total_count += y_batch_np.shape[0]
        
    return total_acc / total_count
if __name__ == "__main__":
    EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_STEPS_TO_RUN = 20
    # NUM_SAMPLES = BATCH_SIZE * NUM_STEPS_TO_RUN
    INPUT_SIZE = 784
    HIDDEN_SIZE = 100
    NUM_CLASSES = 10

    print("Loading MNIST data...")
    X_train, y_train = parse_mnist("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    X_test,y_test = parse_mnist("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")
    # print(f"Using a subset of {NUM_SAMPLES} images to run for {NUM_STEPS_TO_RUN} steps...")
    # X_train = X_train[:NUM_SAMPLES]
    # y_train = y_train[:NUM_SAMPLES]
    print("Data loaded.\n")

    np.random.seed(42)
    
    model = Sequential(
        Linear(INPUT_SIZE, HIDDEN_SIZE),
        BatchNorm1d(HIDDEN_SIZE), # LayerNorm1d
        ReLU(),
        Linear(HIDDEN_SIZE, NUM_CLASSES)
    )
    
    loss_fn = SoftmaxLoss()
    
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        
        model.train()
        train_epoch(model, loss_fn, optimizer, X_train, y_train, BATCH_SIZE)

        test_acc = evaluate(model, X_test, y_test, BATCH_SIZE)

        print(f"--- Epoch {epoch+1}/{EPOCHS} --- Test Accuracy = {test_acc*100:.2f}% ---")
        print("-" * 20)