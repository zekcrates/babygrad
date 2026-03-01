
import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Dropout, Flatten, Linear, ReLU, Sequential, SoftmaxLoss, BatchNorm1d
from baby.data import MNISTDataset, DataLoader
from baby.trainer import Trainer  
from baby.compiler import print_graph


import time
if __name__ == "__main__":
    EPOCHS = 5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    INPUT_SIZE = 784
    HIDDEN_SIZE = 100
    NUM_CLASSES = 10

    print("Loading MNIST data via Dataset...")
    train_dataset = MNISTDataset("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    test_dataset = MNISTDataset("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaded.\n")
    print("Number of training samples:", len(train_dataset))
    print("Number of batches per epoch:", len(train_loader))
    np.random.seed(42)
    model = Sequential(
        Flatten(),
        Linear(INPUT_SIZE, HIDDEN_SIZE),
        BatchNorm1d(HIDDEN_SIZE),
        ReLU(),
        Dropout(), 
        Linear(HIDDEN_SIZE, NUM_CLASSES)
    )

    loss_fn = SoftmaxLoss()
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)


    print("\n==== Building Graph For Inspection ====")

    # Take one batch
    x_batch, y_batch = next(iter(train_loader))

    # Forward pass
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)

    # Print graph
    print_graph(loss)

    print("==== End Graph ====\n")


    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader=test_loader)
    start_time = time.time() 

    trainer.fit(EPOCHS)

    end_time = time.time() 
    total_time = end_time - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.2f} seconds")