
import numpy as np
import pytest
from baby import Tensor, ops
import numdifftools as nd # Import the new library


# tests/conftest.py


@pytest.fixture(autouse=True)
def fix_random_seed():
    
    np.random.seed(0)
def numerical_gradient_check(op, *inputs):
    """
    Checks the gradient of an operation using the professional numdifftools library.
    """
    for i, an_input in enumerate(inputs):
        if not an_input.requires_grad:
            continue
            
        clean_inputs = [Tensor(x.data, requires_grad=True) for x in inputs]
        loss = op(*clean_inputs).sum()
        loss.backward()
        analytical_grad = clean_inputs[i].grad

        
        def f_wrapper(x_np):
            temp_inputs = list(inputs)
            temp_inputs[i] = Tensor(x_np.reshape(an_input.shape))
            return op(*temp_inputs).sum().data

        numerical_grad = nd.Gradient(f_wrapper)(an_input.data.flatten()).reshape(an_input.shape)

        # print(f"Checking gradient for input {i}...")
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3) 
        # print("OK!")


def test_add_backward():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    b = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x, y: x + y, a, b)

def test_mul_backward():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    b = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x, y: x * y, a, b)

def test_power_scalar_backward():
    a = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    numerical_gradient_check(lambda x: x ** 3, a)

def test_div_backward():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    b = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    numerical_gradient_check(lambda x, y: x / y, a, b)

def test_summation_backward_axis():
    a = Tensor(np.random.rand(5, 6, 7), requires_grad=True)
    numerical_gradient_check(lambda x: x.sum(axes=1), a)

def test_summation_backward_all():
    a = Tensor(np.random.rand(5, 6), requires_grad=True)
    numerical_gradient_check(lambda x: x.sum(), a)

def test_matmul_backward():
    a = Tensor(np.random.rand(5, 4), requires_grad=True)
    b = Tensor(np.random.rand(4, 7), requires_grad=True)
    numerical_gradient_check(lambda x, y: x @ y, a, b)

def test_reshape_backward():
    a = Tensor(np.random.rand(5, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x.reshape(4, 5), a)

def test_transpose_backward():
    a = Tensor(np.random.rand(5, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x.transpose(), a)

def test_broadcast_to_backward():
    a = Tensor(np.random.rand(5, 1), requires_grad=True)
    numerical_gradient_check(lambda x: x.broadcast_to((5, 4)), a)

def test_negate_backward():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x: -x, a)

def test_log_backward():
    a = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    numerical_gradient_check(ops.log, a)

def test_exp_backward():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(ops.exp, a)

def test_relu_backward():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    numerical_gradient_check(ops.relu, a)

def test_combined_ops():
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    b = Tensor(np.random.rand(4, 5), requires_grad=True)
    c = Tensor(np.random.rand(3, 5), requires_grad=True)
    func = lambda x, y, z: ops.relu(x @ y + z)
    numerical_gradient_check(func, a, b, c)