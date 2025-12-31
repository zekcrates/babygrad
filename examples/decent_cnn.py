import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Module, Conv, Linear, ReLU, Sequential,Dropout, SoftmaxLoss, Flatten, BatchNorm1d
from baby.data import MNISTDataset, DataLoader
from baby.trainer import Trainer


class SimpleCNN(Module):
    """
    A simple CNN for MNIST:
    Conv -> ReLU -> Flatten -> Linear -> BatchNorm -> ReLU -> Linear
    """
    def __init__(self):
        super().__init__()
        
        # Using Sequential for cleaner structure
        self.model = Sequential(
            Conv(in_channels=1, out_channels=16, kernel_size=3, stride=1, bias=True),
            ReLU(),
            Flatten(),
            Linear(16 * 28 * 28, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 10)
        )
    
    def forward(self, x):
        # x shape: (batch, 28, 28, 1) from MNISTDataset in NHWC format
        # Need to convert to (batch, 1, 28, 28) for Conv (NCHW)
        
        # Transpose: (batch, 28, 28, 1) -> (batch, 1, 28, 28)
        x_data = x.data if isinstance(x, Tensor) else x
        x_reshaped = x_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        x = Tensor(x_reshaped)
        
        return self.model(x)


if __name__ == "__main__":
    EPOCHS = 6
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
    
    # Initialize model, loss, and optimizer
    model = SimpleCNN()
    loss_fn = SoftmaxLoss()
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader
    )
    
    print("Starting training...\n")
    
    # Train the model
    trainer.fit(epochs=EPOCHS)
    
    # Final evaluation
    print("\n" + "="*40)
    final_acc = trainer.evaluate(loader=test_loader)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
    print("="*40)
    
    print("\nSaving model...")
    model.save("cnn_mnist.pt")
    print("Done!")  