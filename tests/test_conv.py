import pytest
import numpy as np
from baby.tensor import Tensor
from baby.ops import conv # Assumes your factory function is in ops
from tests.test_ops import numerical_gradient_check

import numpy as np

def reference_conv2d(A, B, stride=1, padding=0):
    """Simple nested loops to verify im2col logic."""
    if padding > 0:
        A = np.pad(A, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
    
    N, H, W, C_in = A.shape
    K, _, _, C_out = B.shape
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    
    out = np.zeros((N, H_out, W_out, C_out))
    
    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * stride, w * stride
                window = A[n, h_start:h_start+K, w_start:w_start+K, :]
                out[n, h, w, :] = np.sum(window[:, :, :, np.newaxis] * B, axis=(0, 1, 2))
                
    return out

def test_conv_forward_basic():
    # Setup: (Batch=1, H=5, W=5, C_in=1)
    A = np.arange(25).reshape(1, 5, 5, 1).astype(np.float32)
    # Kernel: (K=3, K=3, C_in=1, C_out=1) - All ones
    B = np.ones((3, 3, 1, 1)).astype(np.float32)
    
    res_baby = conv(Tensor(A), Tensor(B), stride=1, padding=0).data
    res_ref = reference_conv2d(A, B, stride=1, padding=0)
    
    assert res_baby.shape == (1, 3, 3, 1)
    assert np.allclose(res_baby, res_ref), "Basic Forward Conv Failed"

def test_conv_forward_padding():
    A = np.random.randn(2, 10, 10, 3).astype(np.float32)
    B = np.random.randn(3, 3, 3, 8).astype(np.float32) # 8 filters
    
    # Test with Padding=1
    res_baby = conv(Tensor(A), Tensor(B), stride=1, padding=1).data
    res_ref = reference_conv2d(A, B, stride=1, padding=1)
    
    assert res_baby.shape == (2, 10, 10, 8)
    assert np.allclose(res_baby, res_ref, atol=1e-5)

def test_conv_forward_stride():
    A = np.random.randn(1, 14, 14, 1).astype(np.float32)
    B = np.random.randn(3, 3, 1, 1).astype(np.float32)
    
    # Test with Stride=2
    res_baby = conv(Tensor(A), Tensor(B), stride=2, padding=0).data
    res_ref = reference_conv2d(A, B, stride=2, padding=0)
    
    # (14 - 3) // 2 + 1 = 11 // 2 + 1 = 6
    assert res_baby.shape == (1, 6, 6, 1)
    assert np.allclose(res_baby, res_ref, atol=1e-5)


def test_conv_backward_basic():
    """
    Test backward pass for a simple 1x1 stride, no padding convolution.
    Verifies dL/dA and dL/dB.
    """
    # Batch=1, H=5, W=5, C_in=1
    a = Tensor(np.random.randn(1, 5, 5, 1), requires_grad=True)
    # K=3, K=3, C_in=1, C_out=1
    b = Tensor(np.random.randn(3, 3, 1, 1), requires_grad=True)
    
    numerical_gradient_check(lambda x, y: conv(x, y, stride=1, padding=0), a, b)

def test_conv_backward_padding():
    """
    Verifies that gradients flow correctly through the zero-padding boundary.
    """
    a = Tensor(np.random.randn(1, 4, 4, 2), requires_grad=True)
    b = Tensor(np.random.randn(3, 3, 2, 2), requires_grad=True)
    
    numerical_gradient_check(lambda x, y: conv(x, y, stride=1, padding=1), a, b)

def test_conv_backward_stride():
    """
    The 'Math Boss': Verifies the dilation logic in the backward pass.
    If stride > 1, the output gradient must be dilated before the 
    backward convolution.
    """
    a = Tensor(np.random.randn(1, 7, 7, 1), requires_grad=True)
    b = Tensor(np.random.randn(3, 3, 1, 1), requires_grad=True)
    
    numerical_gradient_check(lambda x, y: conv(x, y, stride=2, padding=0), a, b)

def test_conv_backward_multi_filter():
    """
    Verifies that gradients are correctly summed across multiple 
    input and output channels.
    """
    batch, h, w, c_in = 2, 6, 6, 3
    k, c_out = 3, 4
    
    a = Tensor(np.random.randn(batch, h, w, c_in), requires_grad=True)
    b = Tensor(np.random.randn(k, k, c_in, c_out), requires_grad=True)
    
    numerical_gradient_check(lambda x, y: conv(x, y, stride=1, padding=1), a, b)