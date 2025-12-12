# import numpy as np
# from baby import Tensor
# from baby.optim import Adam
# from baby.nn import Dropout, Flatten, Module, Linear, ReLU, Sequential, SoftmaxLoss, LayerNorm1d, BatchNorm1d
# from baby.data import MNISTDataset, DataLoader

# def train_epoch(model: Module, loss_fn: Module, optimizer: Adam, train_loader: DataLoader):
#     for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
#         logits = model(x_batch)
#         loss = loss_fn(logits, y_batch.data)

#         optimizer.reset_grad()
#         loss.backward()
#         optimizer.step()

#         preds = logits.data.argmax(axis=1)
#         acc = np.mean(preds == y_batch.data)

#         print(f"   Batch {batch_idx+1:3d}: Loss = {loss.data:.4f}, Accuracy = {acc*100:.2f}%")


# def evaluate(model: Module, test_loader: DataLoader):
#     model.eval()
#     total_correct = 0
#     total_count = 0
#     for x_batch, y_batch in test_loader:
#         logits = model(x_batch)
#         preds = logits.data.argmax(axis=1)
#         total_correct += np.sum(preds == y_batch.data)
#         total_count += y_batch.data.shape[0]
#     return total_correct / total_count


# if __name__ == "__main__":
#     EPOCHS = 5
#     LEARNING_RATE = 0.001
#     BATCH_SIZE = 128
#     INPUT_SIZE = 784
#     HIDDEN_SIZE = 100
#     NUM_CLASSES = 10

#     print("Loading MNIST data via Dataset...")
#     train_dataset = MNISTDataset("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
#     test_dataset = MNISTDataset("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     print("Data loaded.\n")

#     np.random.seed(42)

#     model = Sequential(
#         Flatten(),
#         Linear(INPUT_SIZE, HIDDEN_SIZE),
#         BatchNorm1d(HIDDEN_SIZE),
#         ReLU(),
#         Dropout(),
#         Linear(HIDDEN_SIZE, NUM_CLASSES)
#     )

#     loss_fn = SoftmaxLoss()
#     optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

#     for epoch in range(EPOCHS):
#         model.train()
#         print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
#         train_epoch(model, loss_fn, optimizer, train_loader)
#         test_acc = evaluate(model, test_loader)
#         print(f"--- Test Accuracy = {test_acc*100:.2f}% ---")
#         print("-" * 40)



import numpy as np
from baby import Tensor
from baby.optim import Adam
from baby.nn import Dropout, Flatten, Linear, ReLU, Sequential, SoftmaxLoss, BatchNorm1d
from baby.data import MNISTDataset, DataLoader
from baby.trainer import Trainer  # <--- The new import

if __name__ == "__main__":
    # --- Configuration ---
    EPOCHS = 5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    INPUT_SIZE = 784
    HIDDEN_SIZE = 100
    NUM_CLASSES = 10

    # --- Data Loading ---
    print("Loading MNIST data via Dataset...")
    # Your Dataset/DataLoader implementation is perfect for the Trainer
    train_dataset = MNISTDataset("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    test_dataset = MNISTDataset("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaded.\n")

    # --- Model Setup ---
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

    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader=test_loader)
    trainer.fit(EPOCHS)