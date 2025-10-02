import pytest
import numpy as np

from baby.tensor import Tensor
from baby.nn import Module, Parameter, Linear, ReLU, Sequential, Dropout, Flatten, Residual
from baby import init
from tests.test_ops import numerical_gradient_check

class ParamLayer(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = Parameter(init.kaiming_normal(in_dim, out_dim))

    def forward(self, x):
        return x @ self.weight

def test_relu_layer():
    layer = ReLU()
    x = Tensor(np.array([[-1., 0.01, 2.], [3., -4., -0.01]]), requires_grad=True)
    output = layer(x)
    expected_forward = np.array([[0., 0.01, 2.], [3., 0., 0.]])
    assert np.allclose(output.data, expected_forward), "ReLU forward pass is incorrect."
    numerical_gradient_check(layer, x)

def test_flatten_layer():
    layer = Flatten()
    batch_size, c, h, w = 10, 3, 4, 5
    x = Tensor(np.random.randn(batch_size, c, h, w))
    output = layer(x)
    expected_shape = (batch_size, c * h * w)
    assert output.shape == expected_shape, "Flatten output shape is incorrect."
    assert np.allclose(output.data, x.data.reshape(expected_shape))

def test_dropout_modes():
    p_dropout = 0.5
    layer = Dropout(p=p_dropout)
    x = Tensor(np.random.rand(10, 10))

    layer.eval()
    output_eval = layer(x)
    assert np.array_equal(output_eval.data, x.data), "Dropout should do nothing in eval mode."

    layer.train()
    np.random.seed(0)
    output_train = layer(x)
    np.random.seed(0)
    mask_np = np.random.rand(*x.shape) <= (1 - p_dropout)
    expected_data = (x.data * mask_np) / (1 - p_dropout)
    assert np.allclose(output_train.data, expected_data), "Dropout train mode is incorrect."

def test_sequential_forward_and_params():
    l1 = ParamLayer(10, 5)
    l2 = ParamLayer(5, 2)
    model = Sequential(l1, ReLU(), l2)
    
    params = model.parameters()
    assert len(params) == 2, "Sequential module did not find all parameters."
    
    x = Tensor(np.random.randn(8, 10))
    expected_output = l2(ReLU()(l1(x)))
    model_output = model(x)
    assert np.allclose(model_output.data, expected_output.data), "Sequential forward pass is incorrect."

def test_residual_block():
    dim = 10
    inner_fn = ParamLayer(dim, dim)
    res_block = Residual(inner_fn)
    
    params = res_block.parameters()
    assert len(params) == 1, "Residual block did not find internal module's parameters."
    
    x = Tensor(np.random.randn(8, dim))
    expected_output = inner_fn(x) + x
    block_output = res_block(x)
    assert np.allclose(block_output.data, expected_output.data), "Residual block forward pass is incorrect."

def test_linear_init():
    in_features, out_features = 10, 5
    
    layer_with_bias = Linear(in_features, out_features, bias=True)
    assert isinstance(layer_with_bias.weight, Parameter)
    assert layer_with_bias.weight.shape == (in_features, out_features)
    assert isinstance(layer_with_bias.bias, Parameter)
    assert layer_with_bias.bias.shape == (1, out_features)

    layer_no_bias = Linear(in_features, out_features, bias=False)
    assert layer_no_bias.bias is None

def test_linear_forward():
    batch_size, in_features, out_features = 4, 10, 5
    layer = Linear(in_features, out_features, bias=True)
    
    w_np = np.random.randn(in_features, out_features).astype(np.float32)
    b_np = np.random.randn(1, out_features).astype(np.float32)
    layer.weight = Parameter(w_np)
    layer.bias = Parameter(b_np)
    
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)
    x = Tensor(x_np)
    
    output = layer(x)
    
    expected_output = x_np @ w_np + b_np
    assert isinstance(output, Tensor)
    assert np.allclose(output.data, expected_output), "Linear forward pass is incorrect."

def test_linear_backward():
    batch_size, in_features, out_features = 4, 10, 5
    
    layer = Linear(in_features, out_features, bias=True)
    x = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)
    
    def linear_op(inp, w, b):
        res = inp @ w
        res += b.broadcast_to(res.shape)
        return res
    
    numerical_gradient_check(linear_op, x, layer.weight, layer.bias)





def test_sigmoid_layer():
    """Tests the forward and backward pass of the Sigmoid layer."""
    # Arrange
    from baby.nn import Sigmoid # Import the new layer
    layer = Sigmoid()
    x_np = np.array([[-1., 0., 2.], [0.5, -4., 1.]])
    x = Tensor(x_np, requires_grad=True)
    
    # Act (Forward)
    output = layer(x)
    
    # Assert (Forward)
    expected_forward = 1 / (1 + np.exp(-x_np))
    assert np.allclose(output.data, expected_forward), "Sigmoid forward pass is incorrect."
    
    # Act & Assert (Backward using our numerical checker)
    numerical_gradient_check(layer, x)


def test_tanh_layer():
    """Tests the forward and backward pass of the Tanh layer."""

    from baby.nn import Tanh 
    layer = Tanh()
    x_np = np.array([[-1., 0., 2.], [0.5, -4., 1.]])
    x = Tensor(x_np, requires_grad=True)
    
    output = layer(x)
    
    expected_forward = np.tanh(x_np)
    assert np.allclose(output.data, expected_forward), "Tanh forward pass is incorrect."
    
    numerical_gradient_check(layer, x)