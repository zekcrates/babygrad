import numpy as np
import pytest
from baby import Tensor, ops

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def fix_random_seed():
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)


def compare_with_pytorch(baby_func, torch_func, *inputs, atol=1e-5, rtol=1e-5):
    """
    Compare forward and backward passes between baby autograd and PyTorch.
    
    Args:
        baby_func: Function using baby Tensor operations
        torch_func: Equivalent function using torch operations
        *inputs: Input arrays (numpy)
        atol, rtol: Tolerances for comparison
    """
    # Baby autograd
    baby_inputs = [Tensor(inp, requires_grad=True) for inp in inputs]
    baby_output = baby_func(*baby_inputs)
    baby_loss = baby_output.sum()
    baby_loss.backward()
    
    # PyTorch
    torch_inputs = [torch.tensor(inp, requires_grad=True, dtype=torch.float64) for inp in inputs]
    torch_output = torch_func(*torch_inputs)
    torch_loss = torch_output.sum()
    torch_loss.backward()
    
    # Compare forward pass
    np.testing.assert_allclose(
        baby_output.data, 
        torch_output.detach().numpy(), 
        atol=atol, rtol=rtol,
        err_msg="Forward pass mismatch"
    )
    
    # Compare gradients
    for i, (baby_inp, torch_inp) in enumerate(zip(baby_inputs, torch_inputs)):
        if baby_inp.requires_grad:
            np.testing.assert_allclose(
                baby_inp.grad, 
                torch_inp.grad.numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"Gradient mismatch for input {i}"
            )


# Basic operations
def test_add_vs_pytorch():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    compare_with_pytorch(
        lambda x, y: x + y,
        lambda x, y: x + y,
        a, b
    )


def test_multiply_vs_pytorch():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    compare_with_pytorch(
        lambda x, y: x * y,
        lambda x, y: x * y,
        a, b
    )


def test_matmul_vs_pytorch():
    a = np.random.rand(5, 4)
    b = np.random.rand(4, 7)
    compare_with_pytorch(
        lambda x, y: x @ y,
        lambda x, y: x @ y,
        a, b
    )


def test_power_scalar_vs_pytorch():
    a = np.random.rand(3, 4) + 0.1
    compare_with_pytorch(
        lambda x: x ** 3,
        lambda x: x ** 3,
        a
    )


def test_divide_vs_pytorch():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4) + 0.1
    compare_with_pytorch(
        lambda x, y: x / y,
        lambda x, y: x / y,
        a, b
    )


# Activation functions
def test_relu_vs_pytorch():
    a = np.random.randn(3, 4)
    compare_with_pytorch(
        ops.relu,
        torch.relu,
        a
    )


def test_sigmoid_vs_pytorch():
    a = np.random.randn(3, 4)
    compare_with_pytorch(
        ops.sigmoid,
        torch.sigmoid,
        a
    )


def test_tanh_vs_pytorch():
    a = np.random.randn(3, 4)
    compare_with_pytorch(
        ops.tanh,
        torch.tanh,
        a
    )


def test_exp_vs_pytorch():
    a = np.random.rand(3, 4)
    compare_with_pytorch(
        ops.exp,
        torch.exp,
        a
    )


def test_log_vs_pytorch():
    a = np.random.rand(3, 4) + 0.1
    compare_with_pytorch(
        ops.log,
        torch.log,
        a
    )


# Shape operations
def test_reshape_vs_pytorch():
    a = np.random.rand(6, 4)
    compare_with_pytorch(
        lambda x: x.reshape(4, 6),
        lambda x: x.reshape(4, 6),
        a
    )


def test_transpose_vs_pytorch():
    a = np.random.rand(5, 4)
    compare_with_pytorch(
        lambda x: x.transpose(),
        lambda x: x.t(),
        a
    )


def test_sum_axis_vs_pytorch():
    a = np.random.rand(5, 6, 7)
    compare_with_pytorch(
        lambda x: x.sum(axes=1),
        lambda x: x.sum(dim=1),
        a
    )


def test_sum_all_vs_pytorch():
    a = np.random.rand(5, 6)
    compare_with_pytorch(
        lambda x: x.sum(),
        lambda x: x.sum(),
        a
    )


def test_broadcast_to_vs_pytorch():
    a = np.random.rand(5, 1)
    compare_with_pytorch(
        lambda x: x.broadcast_to((5, 4)),
        lambda x: x.expand(5, 4),
        a
    )


# Advanced operations
def test_logsoftmax_vs_pytorch():
    a = np.random.randn(5, 10)
    compare_with_pytorch(
        ops.logsoftmax,
        lambda x: torch.log_softmax(x, dim=1),
        a,
        atol=1e-4  # Slightly higher tolerance for numerical stability
    )


def test_logsumexp_vs_pytorch():
    a = np.random.randn(5, 6, 7)
    compare_with_pytorch(
        lambda x: ops.logsumexp(x, axes=1),
        lambda x: torch.logsumexp(x, dim=1),
        a,
        atol=1e-4
    )


# Complex compositions
def test_mlp_layer_vs_pytorch():
    """Test a simple MLP layer: relu(X @ W + b)"""
    X = np.random.rand(3, 4)
    W = np.random.rand(4, 5)
    b = np.random.rand(3, 5)
    
    compare_with_pytorch(
        lambda x, w, bias: ops.relu(x @ w + bias),
        lambda x, w, bias: torch.relu(x @ w + bias),
        X, W, b
    )


def test_softmax_cross_entropy_pattern():
    """Test common pattern: -log_softmax then sum"""
    logits = np.random.randn(4, 10)
    
    compare_with_pytorch(
        lambda x: -ops.logsoftmax(x).sum(),
        lambda x: -torch.log_softmax(x, dim=1).sum(),
        logits,
        atol=1e-4
    )


def test_squared_error_pattern():
    """Test (x - y)^2 pattern"""
    x = np.random.rand(3, 4)
    y = np.random.rand(3, 4)
    
    compare_with_pytorch(
        lambda a, b: ((a - b) ** 2).sum(),
        lambda a, b: ((a - b) ** 2).sum(),
        x, y
    )


# Edge cases
def test_negate_vs_pytorch():
    a = np.random.rand(3, 4)
    compare_with_pytorch(
        lambda x: -x,
        lambda x: -x,
        a
    )


def test_scalar_operations_vs_pytorch():
    a = np.random.rand(3, 4)
    compare_with_pytorch(
        lambda x: (x + 5.0) * 3.0 - 2.0,
        lambda x: (x + 5.0) * 3.0 - 2.0,
        a
    )


def test_chained_matmul_vs_pytorch():
    """Test A @ B @ C"""
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    C = np.random.rand(4, 5)
    
    compare_with_pytorch(
        lambda a, b, c: a @ b @ c,
        lambda a, b, c: a @ b @ c,
        A, B, C
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])