import numpy as np
import pytest

from baby.tensor import Tensor
from baby.nn import Parameter
from baby.optim import SGD, Adam

def test_sgd_step():
    
    p_initial = np.array([10.0, -5.0])
    p = Parameter(p_initial.copy())

    gradient = np.array([2.0, -4.0])
    p.grad = gradient 

    lr = 0.1

    optimizer = SGD(params=[p], lr=lr)

    optimizer.step()

    p_expected = p_initial - lr * gradient

    assert np.allclose(p.data, p_expected), "SGD step did not produce the correct value."


def test_adam_step():
    
    p_initial = np.array([0.5])
    p = Parameter(p_initial.copy())
    
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    optimizer = Adam(params=[p], lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    
    g1 = np.array([0.2])
    p.grad = g1
    optimizer.step()

    m1 = (1 - beta1) * g1
    v1 = (1 - beta2) * (g1**2)
    m1_hat = m1 / (1 - beta1**1)
    v1_hat = v1 / (1 - beta2**1)
    p1_expected = p_initial - lr * m1_hat / (np.sqrt(v1_hat) + eps)

    assert np.allclose(p.data, p1_expected), "Adam value is incorrect after step 1."
    assert np.allclose(optimizer.m[p], m1), "Adam 'm' state is incorrect after step 1."
    assert np.allclose(optimizer.v[p], v1), "Adam 'v' state is incorrect after step 1."
    assert optimizer.t == 1, "Adam 't' state is incorrect after step 1."
    g2 = np.array([-0.3])
    p.grad = g2
    optimizer.step()    
    m2 = beta1 * m1 + (1 - beta1) * g2
    v2 = beta2 * v1 + (1 - beta2) * (g2**2)
    m2_hat = m2 / (1 - beta1**2)
    v2_hat = v2 / (1 - beta2**2)
    p2_expected = p1_expected - lr * m2_hat / (np.sqrt(v2_hat) + eps)

    assert np.allclose(p.data, p2_expected), "Adam value is incorrect after step 2."
    assert np.allclose(optimizer.m[p], m2), "Adam 'm' state is incorrect after step 2."
    assert np.allclose(optimizer.v[p], v2), "Adam 'v' state is incorrect after step 2."
    assert optimizer.t == 2, "Adam 't' state is incorrect after step 2."






def test_zero_grad_logic():
    from baby.nn import Parameter
    from baby.optim import SGD
    
    p = Parameter([1.0, 2.0], requires_grad=True)
    p.grad = np.array([0.5, 0.5]) 
    
    optimizer = SGD([p], lr=0.1)
    optimizer.zero_grad()
    
    assert p.grad is None or np.all(p.grad == 0),  "Grad should be reset to zeros, not None"
    assert p.grad is None or np.all(p.grad == 0), "zero_grad() failed to clear the gradients to zero."



def test_linear_convergence():
    from baby import Tensor
    from baby.nn import Parameter
    from baby.optim import SGD
    
    x = Tensor([[1.0], [2.0], [3.0]])
    target = Tensor([[2.0], [4.0], [6.0]])
    
    w = Parameter([[0.0]]) 
    optimizer = SGD([w], lr=0.01)
    
    initial_loss = None
    for i in range(100):
        optimizer.zero_grad()
        
        pred = x @ w
        loss = ((pred - target) ** 2).sum() 
        
        if i == 0: initial_loss = loss.data
        
        loss.backward()
        
        optimizer.step()
    
    assert loss.data < initial_loss, "Loss did not decrease during training."
    assert np.allclose(w.data, [[2.0]], atol=1e-2), f"W should be close to 2.0, got {w.data}"
    print(f"Convergence Test: PASSED. Final W: {w.data}")



def test_optimizer_parameter_list_handling():
    from baby.nn import Module, Parameter
    from baby.optim import Adam
    
    class SimpleModel(Module):
        def __init__(self):
            self.w1 = Parameter([1.0])
            self.w2 = Parameter([2.0])
            
    model = SimpleModel()
    try:
        optimizer = Adam(model.parameters(), lr=0.01)
    except Exception as e:
        pytest.fail(f"Optimizer failed to handle model.parameters(): {e}")