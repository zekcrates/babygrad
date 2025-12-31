import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Module, Conv, Linear, ReLU, Sequential, SoftmaxLoss, Flatten, BatchNorm1d
from baby.data import MNISTDataset, DataLoader

class SimpleCNN(Module):
    """
    A simple CNN for MNIST:
    Conv -> ReLU -> Flatten -> Linear -> BatchNorm -> ReLU -> Linear
    """
    def __init__(self):
        super().__init__()
        
        # Input: (batch, 1, 28, 28) in NCHW format
        self.conv1 = Conv(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True
        )
        
        self.relu1 = ReLU()
        self.flatten = Flatten()
        
        # After conv: (batch, 16, 28, 28)
        # After flatten: (batch, 16*28*28) = (batch, 12544)
        self.fc1 = Linear(16 * 28 * 28, 128)
        self.bn1 = BatchNorm1d(128)
        self.relu2 = ReLU()
        self.fc2 = Linear(128, 10)
    
    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        #remember in `ops.conv` we did the format (NHWC)
        # and i told in `nn.Conv` that some formats are (NCHW)
        # so we transposed first and then called `ops.conv`

        # But in our MNISTDataset we get in (NHWC) (-1,28,28,1)

        x = self.conv1(x)      # -> (batch, 16, 28, 28)
        x = self.relu1(x)      
        x = self.flatten(x)    # -> (batch, 12544)
        x = self.fc1(x)        
        x = self.bn1(x)        
        x = self.relu2(x)      
        x = self.fc2(x)        
        return x


def train_epoch(model, loss_fn, optimizer, train_loader):
    """Train for one epoch."""
    model.train()
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # x_batch shape: (batch, 28, 28, 1) from MNISTDataset
        # Need to convert to (batch, 1, 28, 28) for Conv
        

        #remember in `ops.conv` we did the format (NHWC)
        # and i told in `nn.Conv` that some formats are (NCHW)
        # so we transposed first and then called `ops.conv`

        # Transpose: (batch, 28, 28, 1) -> (batch, 1, 28, 28)
        x_reshaped = x_batch.data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        x_batch = Tensor(x_reshaped)
        
        # Forward pass
        logits = model(x_batch)
        
        # Compute loss
        loss = loss_fn(logits, y_batch.data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if batch_idx % 50 == 0:
            preds = logits.data.argmax(axis=1)
            acc = np.mean(preds == y_batch.data)
            print(f"   Batch {batch_idx:3d}: Loss = {loss.data:.4f}, Accuracy = {acc*100:.2f}%")


def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    
    total_correct = 0
    total_count = 0
    
    for x_batch, y_batch in test_loader:
        # Transpose to NCHW
        x_reshaped = x_batch.data.transpose(0, 3, 1, 2)
        x_batch = Tensor(x_reshaped)
        
        # Forward pass
        logits = model(x_batch)
        
        # Get predictions
        preds = logits.data.argmax(axis=1)
        
        total_correct += np.sum(preds == y_batch.data)
        total_count += y_batch.data.shape[0]
    
    return total_correct / total_count


if __name__ == "__main__":
    EPOCHS = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    
    print("Loading MNIST data...")
    train_dataset = MNISTDataset(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz"
    )
    
    test_dataset = MNISTDataset(
        "data/t10k-images-idx3-ubyte.gz",
        "data/t10k-labels-idx1-ubyte.gz"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Data loaded.\n")
    
    np.random.seed(42)
    
    model = SimpleCNN()
    loss_fn = SoftmaxLoss()
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...\n")
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_epoch(model, loss_fn, optimizer, train_loader)
        
        test_acc = evaluate(model, test_loader)
        print(f"\n>>> Test Accuracy = {test_acc*100:.2f}% <<<")
        print("-" * 40)
    
    print("\nSaving model...")
    model.save("cnn_mnist.pt")
    print("Done!")