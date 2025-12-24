import pytest
import numpy as np

from baby.tensor import Tensor
from baby.nn import BatchNorm1d, LayerNorm1d, Module, Parameter, Linear, ReLU, Sequential, Dropout, Flatten, Residual, SoftmaxLoss
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
    
    output = layer(x)
    
    expected_forward = 1 / (1 + np.exp(-x_np))
    assert np.allclose(output.data, expected_forward), "Sigmoid forward pass is incorrect."
    
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



def test_softmax_loss():
    """Tests the forward and backward pass of the SoftmaxLoss layer."""
    batch_size, num_classes = 5, 4
    loss_fn = SoftmaxLoss()
    
    logits_np = np.random.randn(batch_size, num_classes).astype(np.float32)
    logits = Tensor(logits_np, requires_grad=True)
    
    y = np.random.randint(num_classes, size=batch_size)

    loss = loss_fn(logits, y)
    
    max_logits = np.max(logits_np, axis=1, keepdims=True)
    exp_logits = np.exp(logits_np - max_logits)
    sum_exp_logits = np.sum(exp_logits, axis=1)
    log_sum_exp_np = np.log(sum_exp_logits) + np.squeeze(max_logits)
    correct_logits = logits_np[np.arange(batch_size), y]
    expected_loss = np.sum(log_sum_exp_np - correct_logits) / batch_size
    
    assert isinstance(loss, Tensor)
    assert np.allclose(loss.data, expected_loss), "SoftmaxLoss forward pass is incorrect."
    numerical_gradient_check(lambda l: loss_fn(l, y), logits)





def test_layernorm1d_backward():
    """Tests the backward pass of the LayerNorm1d layer."""
    batch_size, dim = 8, 12
    layer = LayerNorm1d(dim)
    x = Tensor(np.random.randn(batch_size, dim), requires_grad=True)

    def layernorm1d_op(x, weight, bias):
        """Replicates the LayerNorm1d forward pass using base ops for gradient checking."""
        eps = layer.eps
        
        mean = x.sum(axes=1) / dim
        mean_reshaped = mean.reshape((batch_size, 1))
        x_minus_mean = x - mean_reshaped.broadcast_to(x.shape)
        
        var = (x_minus_mean**2).sum(axes=1) / dim
        var_reshaped = var.reshape((batch_size, 1))
        
        std_inv = (var_reshaped.broadcast_to(x.shape) + eps)**-0.5
        x_hat = x_minus_mean * std_inv
        
        return weight.reshape((1, dim)).broadcast_to(x.shape) * x_hat + \
               bias.reshape((1, dim)).broadcast_to(x.shape)

    numerical_gradient_check(layernorm1d_op, x, layer.weight, layer.bias)


def test_batchnorm1d_forward():
    """Tests the forward pass logic for both train and eval modes in BatchNorm1d."""
    dim = 10
    batch_size = 8
    momentum = 0.2
    eps = 1e-5

    layer = BatchNorm1d(dim, eps=eps, momentum=momentum)
    x_np = np.random.randn(batch_size, dim).astype(np.float32)
    x = Tensor(x_np)

    layer.eval()
    running_mean_np = np.random.randn(dim).astype(np.float32)
    running_var_np = np.random.rand(dim).astype(np.float32) 
    layer.running_mean.data = running_mean_np
    layer.running_var.data = running_var_np

    output_eval = layer(x)
    
    expected_x_hat_eval = (x_np - running_mean_np) / np.sqrt(running_var_np + eps)
    expected_output_eval = layer.weight.data * expected_x_hat_eval + layer.bias.data
    
    assert np.allclose(output_eval.data, expected_output_eval), "BatchNorm1d eval forward pass is incorrect."

    layer.train()
    initial_running_mean = layer.running_mean.data.copy()
    initial_running_var = layer.running_var.data.copy()

    output_train = layer(x)
    
    batch_mean_np = x_np.mean(axis=0)
    batch_var_np = x_np.var(axis=0) 
    expected_x_hat_train = (x_np - batch_mean_np) / np.sqrt(batch_var_np + eps)
    expected_output_train = layer.weight.data * expected_x_hat_train + layer.bias.data
    
    assert np.allclose(output_train.data, expected_output_train, atol=1e-6), "BatchNorm1d train forward pass is incorrect."

    expected_running_mean = (1 - momentum) * initial_running_mean + momentum * batch_mean_np
    expected_running_var = (1 - momentum) * initial_running_var + momentum * batch_var_np
    
    assert np.allclose(layer.running_mean.data, expected_running_mean), "BatchNorm1d running_mean update is incorrect."
    assert np.allclose(layer.running_var.data, expected_running_var), "BatchNorm1d running_var update is incorrect."


def test_batchnorm1d_backward():
    batch_size, dim = 16, 10
    layer = BatchNorm1d(dim)
    layer.train() 
    
    x = Tensor(np.random.randn(batch_size, dim), requires_grad=True)

    def batchnorm1d_op(x, weight, bias):
        eps = layer.eps
        
        mean = x.sum(axes=0) / batch_size
        mean_reshaped = mean.reshape((1, dim))
        x_minus_mean = x - mean_reshaped.broadcast_to(x.shape)
        
        var = (x_minus_mean**2).sum(axes=0) / batch_size
        var_reshaped = var.reshape((1, dim))
        std_inv = (var_reshaped.broadcast_to(x.shape) + eps)**-0.5
        x_hat = x_minus_mean * std_inv
        
        return weight.reshape((1, dim)).broadcast_to(x.shape) * x_hat + \
               bias.reshape((1, dim)).broadcast_to(x.shape)

    numerical_gradient_check(batchnorm1d_op, x, layer.weight, layer.bias)



def test_recursive_parameters():
    from baby.nn import Module, Linear, Sequential
    
    class SubBlock(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(10, 10)
    
    class BigModel(Module):
        def __init__(self):
            super().__init__()
            self.block = SubBlock()
            self.net = Sequential(Linear(10, 2))
            
    model = BigModel()
    params = list(model.parameters())
    
    assert len(params) == 4, f"Expected 4 parameters, found {len(params)}"



def test_mlp_classification_convergence():
    from baby import Tensor
    from baby.nn import Linear, ReLU, Sequential, SoftmaxLoss
    from baby.optim import Adam
    
    x = Tensor(np.random.randn(4, 5))
    y = np.array([0, 1, 2, 0]) # Target indices
    
    model = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 3)
    )
    loss_fn = SoftmaxLoss()
    optimizer = Adam(model.parameters(), lr=0.1)
    
    initial_loss = loss_fn(model(x), y).data
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        
    assert loss.data < initial_loss, "MLP Loss did not decrease. Check your SoftmaxLoss/Linear integration."
    print(f"MLP Integration: PASSED. Final Loss: {loss.data}")


def test_kaiming_init_stats():
    from baby import init
    
    in_dim, out_dim = 1000, 1000
    weights = init.kaiming_normal(in_dim, out_dim)
    
    expected_std = np.sqrt(2.0 / in_dim)
    actual_std = np.std(weights.data)
    
    assert np.allclose(actual_std, expected_std, rtol=0.1), \
        f"Kaiming init variance is wrong. Expected std {expected_std}, got {actual_std}"