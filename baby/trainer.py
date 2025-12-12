from baby.tensor import Tensor

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
    def fit(self, epochs: int):
        """
        Runs the training loop for the specified number of epochs.
        """
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in self.train_loader:
                if isinstance(batch, (list, tuple)):
                     x, y = batch
                else:
                     x, y = batch.x, batch.y
                
                if not isinstance(x, Tensor): x = Tensor(x)
                
                self.optimizer.zero_grad()
                
                pred = self.model(x)
                
                loss = self.loss_fn(pred, y)
                
                loss.backward()
                
                self.optimizer.step()
                
                total_loss += loss.data
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


            if self.val_loader is not None:
                val_acc = self.evaluate()
                print(f" | Val Acc: {val_acc*100:.2f}%")
            else:
                print()

    def evaluate(self):
        """Helper to check accuracy on validation set"""
        self.model.eval() 
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)): x, y = batch
            else: x, y = batch.x, batch.y
            
            if not isinstance(x, Tensor): x = Tensor(x)

            logits = self.model(x)
            if isinstance(y, Tensor):
                y_np = y.data
            else:
                y_np = y
            preds = logits.data.argmax(axis=1)
            correct += (preds == y_np).sum()
            total += y.shape[0]
            
        return correct / total