from baby.tensor import Tensor

# class Trainer:
#     def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.train_loader = train_loader
#         self.val_loader = val_loader
    
#     def fit(self, epochs: int, log_interval: int = 50):
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             num_batches = 0
            
#             print(f"--- Epoch {epoch+1}/{epochs} ---")
            
#             for batch_idx, batch in enumerate(self.train_loader):
#                 x, y = batch if isinstance(batch, (list, tuple)) else (batch.x, batch.y)
#                 if not isinstance(x, Tensor): x = Tensor(x)
                
#                 self.optimizer.zero_grad()
#                 pred = self.model(x)
#                 loss = self.loss_fn(pred, y)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.data
#                 num_batches += 1
                
#                 if batch_idx % log_interval == 0:
#                     print(f"  Batch {batch_idx:3d}: Loss = {loss.data:.4f}")
            
#             avg_loss = total_loss / num_batches
#             print(f"End of Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}", end="")

#             if self.val_loader is not None:
#                 val_acc = self.evaluate()
#                 print(f" | Val Acc: {val_acc*100:.2f}%")
#             else:
#                 print()

#     def evaluate(self, loader=None):
#         """
#         Helper to check accuracy on a specific loader or the default val_loader.
#         Leaves the model in eval mode after completion.
#         """
#         target_loader = loader if loader is not None else self.val_loader
#         if target_loader is None:
#             return 0.0
        
#         self.model.eval() 
#         correct = 0
#         total = 0
        
#         for batch in target_loader:
#             if isinstance(batch, (list, tuple)): 
#                 x, y = batch
#             else: 
#                 x, y = batch.x, batch.y
            
#             if not isinstance(x, Tensor): x = Tensor(x)

#             logits = self.model(x)
            
#             y_np = y.data if isinstance(y, Tensor) else y
#             preds = logits.data.argmax(axis=1)
            
#             correct += (preds == y_np).sum()
#             total += y_np.shape[0]
        
#         return correct / total

from baby.tensor import Tensor
import numpy as np

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
            
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
            for batch_idx, batch in enumerate(self.train_loader):
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
                
                # Print progress every 50 batches
                if batch_idx % 50 == 0:
                    # Calculate accuracy for this batch
                    y_np = y.data if isinstance(y, Tensor) else y
                    preds = pred.data.argmax(axis=1)
                    batch_acc = (preds == y_np).mean()
                    
                    print(f"  Batch {batch_idx:3d}: Loss = {loss.data:.4f} | Acc = {batch_acc*100:.2f}%")
            
            avg_loss = total_loss / num_batches
            print(f"End of Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}", end="")

            if self.val_loader is not None:
                val_acc = self.evaluate()
                print(f" | Val Acc: {val_acc*100:.2f}%")
            else:
                print()

    def evaluate(self, loader=None):
        """
        Helper to check accuracy on a specific loader or the default val_loader.
        Leaves the model in eval mode after completion.
        """
        target_loader = loader if loader is not None else self.val_loader
        if target_loader is None:
            return 0.0
        
        self.model.eval() 
        correct = 0
        total = 0
        
        for batch in target_loader:
            if isinstance(batch, (list, tuple)): 
                x, y = batch
            else: 
                x, y = batch.x, batch.y
            
            if not isinstance(x, Tensor): x = Tensor(x)

            logits = self.model(x)
            
            y_np = y.data if isinstance(y, Tensor) else y
            preds = logits.data.argmax(axis=1)
            
            correct += (preds == y_np).sum()
            total += y_np.shape[0]
        
        return correct / total