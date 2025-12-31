import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Module, Conv, Linear, ReLU, Sequential, Dropout,Residual, SoftmaxLoss, Flatten, BatchNorm1d
from baby.data import DataLoader,CIFAR10Dataset, RandomCrop, RandomFlipHorizontal
from baby.trainer import Trainer



class MiniResNet(Module):
    def __init__(self):
        super().__init__()
        
        # 1. Stem: Initial expansion (3 -> 32 channels)
        self.stem = Sequential(
            Conv(3, 32, kernel_size=3, stride=1, bias=True),
            ReLU()
        )
        
        # Helper for Residual Block
        def res_block():
            return Residual(
                Sequential(
                    Conv(32, 32, kernel_size=3, stride=1, bias=True),
                    ReLU(),
                    Conv(32, 32, kernel_size=3, stride=1, bias=True)
                )
            )

        # 2. Residual Layers: 2 blocks deep
        self.layers = Sequential(
            res_block(),
            ReLU(),
            res_block(),
            ReLU()
        )
        
        # 3. Head: Flatten and Classify
        # Input: 32 * 32 * 32 = 32,768
        self.head = Sequential(
            Flatten(),
            Linear(32768, 256),
            BatchNorm1d(256), 
            ReLU(),
            Dropout(p=0.3),
            Linear(256, 10)
        )

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        x = self.stem(x)
        x = self.layers(x)
        return self.head(x)

if __name__ == "__main__":
    EPOCHS = 10
    LEARNING_RATE = 0.0005 
    BATCH_SIZE = 64 
    train_transforms = [RandomFlipHorizontal(), RandomCrop(padding=4)]
    
    print("Loading CIFAR-10...")
    train_dataset = CIFAR10Dataset(base_folder="data/cifar-10-batches-py", train=True, transforms=train_transforms)
    test_dataset = CIFAR10Dataset(base_folder="data/cifar-10-batches-py", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    np.random.seed(42)
    model = MiniResNet()
    loss_fn = SoftmaxLoss()
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader
    )
    
    print("Starting ResNet training (No BatchNorm in blocks)...\n")
    trainer.fit(epochs=EPOCHS)