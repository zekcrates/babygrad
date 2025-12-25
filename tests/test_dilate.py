import numpy as np
import pytest
import numdifftools as nd
from baby import Tensor, ops
from baby.ops import dilate, undilate

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
        
        # Ensure the analytical gradient is a numpy array for comparison
        analytical_grad = clean_inputs[i].grad
        if isinstance(analytical_grad, Tensor):
            analytical_grad = analytical_grad.data

        def f_wrapper(x_np):
            temp_inputs = list(inputs)
            temp_inputs[i] = Tensor(x_np.reshape(an_input.shape))
            return op(*temp_inputs).sum().data

        numerical_grad = nd.Gradient(f_wrapper)(an_input.data.flatten()).reshape(an_input.shape)

        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-2, rtol=1e-2)

### --- Forward Tests ---

def test_dilate_forward_simple():
    # 1D Case: [1, 2, 3] dilation=1 -> [1, 0, 2, 0, 3]
    a = Tensor([1, 2, 3])
    out = dilate(a, axes=(0,), dilation=1)
    expected = np.array([1, 0, 2, 0, 3])
    np.testing.assert_allclose(out.data, expected)

    # 2D Case: 2x2 dilation=1 -> 3x3
    b = Tensor([[1, 2], 
                [3, 4]])
    out_2d = dilate(b, axes=(0, 1), dilation=1)
    expected_2d = np.array([[1, 0, 2],
                            [0, 0, 0],
                            [3, 0, 4]])
    np.testing.assert_allclose(out_2d.data, expected_2d)

def test_dilate_selective_axes():
    # Only dilate axis 1
    a = Tensor([[1, 2], [3, 4]])
    # axis 1: 2 -> 2 + (2-1)*2 = 4
    out = dilate(a, axes=(1,), dilation=2)
    assert out.shape == (2, 4)
    expected = np.array([[1, 0, 0, 2],
                         [3, 0, 0, 4]])
    np.testing.assert_allclose(out.data, expected)

### --- Backward Tests (Numerical Gradient Checks) ---

def test_dilate_gradient_simple():
    # Test gradient for 1D dilation
    a = Tensor(np.random.randn(4), requires_grad=True)
    numerical_gradient_check(lambda x: dilate(x, axes=(0,), dilation=1), a)

def test_dilate_gradient_2d():
    # Test gradient for 2D dilation (typical for Conv backward)
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    numerical_gradient_check(lambda x: dilate(x, axes=(0, 1), dilation=1), a)

def test_dilate_gradient_high_dilation():
    # Test gradient with larger gaps
    a = Tensor(np.random.randn(2, 2), requires_grad=True)
    numerical_gradient_check(lambda x: dilate(x, axes=(0, 1), dilation=3), a)

def test_undilate_gradient_simple():
    # This checks if Undilate's backward (which should call Dilate) is correct
    # Input 5x5, undilate to 3x3
    a = Tensor(np.random.randn(5, 5), requires_grad=True)
    numerical_gradient_check(lambda x: undilate(x, axes=(0, 1), dilation=1), a)

def test_undilate_gradient_with_skipped_edges():
    # This is the "Edge Case" we discussed: 4x4 undilated to 2x2
    # If this passes, your Dilate/Undilate logic handles skipped pixels correctly
    a = Tensor(np.random.randn(4, 4), requires_grad=True)
    numerical_gradient_check(lambda x: undilate(x, axes=(0, 1), dilation=1), a)