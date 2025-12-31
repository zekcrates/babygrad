import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Module, Conv, Linear, ReLU, Sequential, Dropout, SoftmaxLoss, Flatten, BatchNorm1d
from baby.data import DataLoader,CIFAR10Dataset
from baby.trainer import Trainer

class CIFAR10CNN(Module):
    """
    Architecture for CIFAR-10:
    Conv(3->16) -> ReLU -> Conv(16->32) -> ReLU -> Flatten -> Linear -> BN -> ReLU -> Dropout -> Linear
    """
    def __init__(self):
        super().__init__()
        
        # Math: 32 channels * 32 width * 32 height = 32,768
        flattened_size = 32768
        
        self.net = Sequential(
            # First Conv: Extracts color-based edges/blobs
            Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=True),
            ReLU(),
            
            # Second Conv: Combines edges into textures/simple shapes
            Conv(in_channels=16, out_channels=32, kernel_size=3, stride=1, bias=True),
            ReLU(),
            
            Flatten(),
            
            # Classification Head
            Linear(flattened_size, 256),
            BatchNorm1d(256),
            ReLU(),
            Dropout(p=0.4),
            Linear(256, 10)
        )
    
    def forward(self, x):
        #CIFAR10Dataset already provides (3, 32, 32), (N,C,H,W) so no transpose needed
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.net(x)

if __name__ == "__main__":
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64 
    
    print("Loading CIFAR-10 data...")
    train_dataset = CIFAR10Dataset(base_folder="data/cifar-10-batches-py", train=True)
    test_dataset = CIFAR10Dataset(base_folder="data/cifar-10-batches-py", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data loaded. Training samples: {len(train_dataset)}")
    
    np.random.seed(42)
    
    # Initialize components
    model = CIFAR10CNN()
    loss_fn = SoftmaxLoss()
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader
    )
    
    print("Starting CIFAR-10 training (Strap in, this is heavy!)...\n")
    
    trainer.fit(epochs=EPOCHS)
    
    print("\n" + "="*40)
    final_acc = trainer.evaluate(loader=test_loader)
    print(f"Final CIFAR-10 Test Accuracy: {final_acc*100:.2f}%")
    print("="*40)
    
    model.save("cifar10_model.pt")
    print("Model saved as cifar10_model.pt")