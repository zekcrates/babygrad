
import numpy as np
import pytest
from baby import Tensor, ops
import numdifftools as nd 




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
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-2, rtol=1e-2) 
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



def test_sigmoid_backward():
    """Tests the backward pass of the sigmoid function."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    numerical_gradient_check(ops.sigmoid, a)

def test_tanh_backward():
    """Tests the backward pass of the tanh function."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    numerical_gradient_check(ops.tanh, a)






def test_logsoftmax_backward():
    """Tests the backward pass of the LogSoftmax function."""
    # Logits are often positive and negative, so randn is a good choice
    a = Tensor(np.random.randn(5, 10), requires_grad=True)
    numerical_gradient_check(ops.logsoftmax, a)


def test_logsumexp_backward_single_axis():
    """Tests LogSumExp backward pass when summing over a single axis."""
    a = Tensor(np.random.randn(5, 6, 7), requires_grad=True)
    # Use a lambda to pass the 'axes' argument
    numerical_gradient_check(lambda x: ops.logsumexp(x, axes=1), a)


def test_logsumexp_backward_multiple_axes():
    """Tests LogSumExp backward pass when summing over multiple axes."""
    a = Tensor(np.random.randn(5, 6, 7), requires_grad=True)
    numerical_gradient_check(lambda x: ops.logsumexp(x, axes=(0, 2)), a)

def test_logsumexp_backward_all_axes():
    """Tests LogSumExp backward pass when summing over all axes."""
    a = Tensor(np.random.randn(8, 3), requires_grad=True)
    numerical_gradient_check(lambda x: ops.logsumexp(x, axes=None), a)





def test_diamond_graph():
    a = Tensor([2.0], requires_grad=True)
    b = a * 2.0  # b = 4
    c = a * 3.0  # c = 6
    d = b + c    # d = 10
    
    d.backward()
    
    # Gradient should be: (1.0 * 2.0) + (1.0 * 3.0) = 5.0
    assert a.grad == 5.0

def test_gradient_accumulation():
    x = Tensor([10.0], requires_grad=True)
    y = x + x + x
    
    y.backward()
    
    # If your code says 1.0, you are overwriting. 
    # It must be 3.0.
    assert x.grad == 3.0


# def test_inplace_safety():
#     a = Tensor([2.0], requires_grad=True)
#     b = a * a # Needs original 'a' (2.0) for 2*a backward
    
#     # Manually messing with data
#     a.data += 1.0 
    
#     b.backward()
    
#     # If the engine used the new value (3.0), the grad will be 6.0 (Wrong).
#     # It should have used the value at the time of the forward pass (4.0).
#     assert a.grad == 4.0


def test_intermediate_gradients():
    x = Tensor([2.0], requires_grad=True)
    w = Tensor([3.0], requires_grad=True)
    
    hidden = x * w     # hidden is a "middle-man"
    output = hidden ** 2 # output = 36
    
    output.backward()
    
    # d(out)/d(hidden) = 2 * hidden = 12
    # d(out)/d(w) = d(out)/d(hidden) * d(hidden)/d(w) = 12 * 2 = 24
    assert w.grad == 24.0


def test_add_scalar_backward():
    a = Tensor(np.random.rand(3, 3), requires_grad=True)
    numerical_gradient_check(lambda x: x + 5.0, a)

def test_mul_scalar_backward():
    a = Tensor(np.random.rand(3, 3), requires_grad=True)
    numerical_gradient_check(lambda x: x * 3.0, a)

def test_div_scalar_backward():
    a = Tensor(np.random.rand(3, 3), requires_grad=True)
    numerical_gradient_check(lambda x: x / 2.0, a)

def test_sqrt_backward():
    # Keep values positive for sqrt
    a = Tensor(np.random.rand(3, 3) + 0.1, requires_grad=True)
    numerical_gradient_check(ops.sqrt, a)



def test_flip_backward():
    a = Tensor(np.random.rand(4, 4), requires_grad=True)
    # Test flipping on both axes
    numerical_gradient_check(lambda x: ops.flip(x, axes=(0, 1)), a)

def test_dilate_backward():
    a = Tensor(np.random.rand(2, 2), requires_grad=True)
    # Test dilation of 1 (adds 1 zero between pixels)
    numerical_gradient_check(lambda x: ops.dilate(x, axes=(0, 1), dilation=1), a)

def test_undilate_backward():
    # Create a 4x4 and undilate to 2x2
    a = Tensor(np.random.rand(4, 4), requires_grad=True)
    numerical_gradient_check(lambda x: ops.undilate(x, axes=(0, 1), dilation=1), a)


def test_add_scalar_backward():
    """Test AddScalar backward pass."""
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x + 5.0, a)

def test_mul_scalar_backward():
    """Test MulScalar backward pass."""
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x * 3.5, a)

def test_div_scalar_backward():
    """Test DivScalar backward pass."""
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x / 2.0, a)

def test_power_backward():
    """Test Power (tensor^tensor) backward pass."""
    a = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    b = Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
    numerical_gradient_check(lambda x, y: x ** y, a, b)

def test_sqrt_backward():
    """Test Sqrt backward pass."""
    a = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    numerical_gradient_check(ops.sqrt, a)

def test_transpose_with_axes_backward():
    """Test Transpose with specific axes."""
    a = Tensor(np.random.rand(2, 3, 4), requires_grad=True)
    numerical_gradient_check(lambda x: x.transpose(axes=(2, 0, 1)), a)

def test_broadcast_to_complex_backward():
    """Test BroadcastTo with multiple dimensions."""
    a = Tensor(np.random.rand(1, 3, 1), requires_grad=True)
    numerical_gradient_check(lambda x: x.broadcast_to((4, 3, 5)), a)

def test_summation_multiple_axes():
    """Test Summation over multiple axes."""
    a = Tensor(np.random.rand(5, 6, 7), requires_grad=True)
    numerical_gradient_check(lambda x: x.sum(axes=(0, 2)), a)

def test_matmul_batched():
    """Test MatMul with batched inputs."""
    a = Tensor(np.random.rand(2, 5, 4), requires_grad=True)
    b = Tensor(np.random.rand(2, 4, 7), requires_grad=True)
    numerical_gradient_check(lambda x, y: x @ y, a, b)



def test_diamond_graph():
    """Test gradient accumulation in diamond-shaped computation graph."""
    a = Tensor([2.0], requires_grad=True)
    b = a * 2.0  # b = 4
    c = a * 3.0  # c = 6
    d = b + c    # d = 10
    
    d.backward()
    
    # Gradient should be: (1.0 * 2.0) + (1.0 * 3.0) = 5.0
    assert a.grad == 5.0

def test_gradient_accumulation():
    """Test that gradients accumulate correctly when a tensor is used multiple times."""
    x = Tensor([10.0], requires_grad=True)
    y = x + x + x
    
    y.backward()
    
    # Should be 3.0, not 1.0 (testing for proper accumulation)
    assert x.grad == 3.0


def test_power_scalar_negative():
    """Test PowerScalar with negative exponent."""
    a = Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
    numerical_gradient_check(lambda x: x ** -2, a)

def test_broadcast_from_scalar():
    """Test broadcasting from a scalar."""
    a = Tensor(np.array([[5.0]]), requires_grad=True)
    numerical_gradient_check(lambda x: x.broadcast_to((3, 4)), a)

def test_relu_with_negatives():
    """Test ReLU with values that cross zero."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)  
    numerical_gradient_check(ops.relu, a)

def test_division_near_zero():
    """Test division with denominator not too close to zero."""
    a = Tensor(np.random.rand(3, 4), requires_grad=True)
    b = Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)  
    numerical_gradient_check(lambda x, y: x / y, a, b)

def test_log_positive_values():
    """Test log with strictly positive values."""
    a = Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
    numerical_gradient_check(ops.log, a)



def test_complex_composition_1():
    """Test complex composition: (x^2 + y) * exp(z)."""
    x = Tensor(np.random.rand(3, 4), requires_grad=True)
    y = Tensor(np.random.rand(3, 4), requires_grad=True)
    z = Tensor(np.random.rand(3, 4), requires_grad=True)
    func = lambda a, b, c: (a ** 2 + b) * ops.exp(c)
    numerical_gradient_check(func, x, y, z)

def test_complex_composition_2():
    """Test complex composition with multiple operations."""
    x = Tensor(np.random.rand(3, 4), requires_grad=True)
    y = Tensor(np.random.rand(4, 5), requires_grad=True)
    func = lambda a, b: ops.sigmoid((a @ b).sum(axes=1))
    numerical_gradient_check(func, x, y)

def test_nested_broadcasts():
    """Test nested broadcasting operations."""
    a = Tensor(np.random.rand(1, 4), requires_grad=True)
    b = Tensor(np.random.rand(3, 1), requires_grad=True)
    func = lambda x, y: (x.broadcast_to((3, 4)) + y.broadcast_to((3, 4))).sum()
    numerical_gradient_check(func, a, b)

