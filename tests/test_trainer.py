import pytest
import numpy as np

from baby.tensor import Tensor
from baby.nn import Linear, SoftmaxLoss, Sequential, Dropout
from baby.optim import Adam, SGD
from baby.trainer import Trainer

def test_trainer_mode_toggle():
    from baby.nn import Dropout, Sequential
    from baby.optim import SGD
    from baby.tensor import Tensor
    import numpy as np
    
    model = Sequential(Dropout(p=0.5))
    optimizer = SGD(model.parameters(), lr=0.1)
    
    def dummy_loss_fn(pred, y):
        return pred.sum()
    
    dummy_loader = [(np.ones((1,1)), np.array([0]))]
    
    trainer = Trainer(model, optimizer, dummy_loss_fn, 
                      train_loader=dummy_loader, 
                      val_loader=dummy_loader)

    trainer.fit(epochs=1)
    assert model.training == True
    
    trainer.evaluate()
    assert model.training == False
def test_trainer_overfit_convergence():
    x = np.random.randn(4, 5).astype(np.float32)
    y = np.array([0, 1, 2, 0])
    loader = [(x, y)] 
    
    model = Linear(5, 3)
    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = SoftmaxLoss()
    
    trainer = Trainer(model, optimizer, loss_fn, train_loader=loader)
    
    trainer.fit(epochs=50)
    
    acc = trainer.evaluate(loader=loader) 
    assert acc > 0.9, f"Trainer failed to overfit one batch. Final Acc: {acc}"

def test_trainer_input_parsing():
    model = Linear(2, 2)
    optimizer = SGD(model.parameters())
    loss_fn = SoftmaxLoss()

    class ObjectBatch:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    loader_tuple = [ (np.ones((2,2)), np.array([0, 1])) ]
    loader_obj = [ ObjectBatch(np.ones((2,2)), np.array([0, 1])) ]
    
    for loader in [loader_tuple, loader_obj]:
        trainer = Trainer(model, optimizer, loss_fn, train_loader=loader)
        trainer.fit(epochs=1)