from typing import Optional
import numpy as np 
from .tensor import Tensor , NDArray, _ensure_tensor

class Op:
    
    def __call__(self, *inputs):
        tensor_inputs =  [_ensure_tensor(i) for i in inputs]
        
        requires_grad = any(t.requires_grad for t in tensor_inputs)

        input_data = [t.data for t in tensor_inputs]

        output_data = self.forward(*input_data)

        output_tensor = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:   
            
            output_tensor._op = self
            output_tensor._inputs = tensor_inputs
        return output_tensor
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, out_grad, node):
        raise NotImplementedError




class Add(Op):
    def forward(self, a: NDArray, b: NDArray):
        return a + b

    def backward(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return Add()(a, b)


class AddScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a + self.scalar

    def backward(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class Mul(Op):
    def forward(self, a: NDArray, b: NDArray):
        return a * b

    def backward(self, out_grad: Tensor, node: Tensor):
        a, b = node._inputs
        return out_grad * b, out_grad * a 


def multiply(a, b):
    return Mul()(a, b)


class MulScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a *self.scalar

    def backward(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)




class Pow(Op):
    """Op to element-wise raise a tensor to a power."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        return np.pow(a, b)
        
    

    
        
    def backward(self, out_grad, node):
        a, b = node._inputs
        one = Tensor(1.0)
        grad_a = multiply(multiply(out_grad, b), power(a, add_scalar(b, -1)))
        grad_b = multiply(multiply(out_grad, power(a, b)), log(a))
        return grad_a, grad_b

def power(a, b):
    return Pow()(a, b)

class PowerScalar(Op):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def forward(self, a: NDArray) -> NDArray:
        return np.power(a ,self.scalar)

    def backward(self, out_grad, node):
        inp = node._inputs[0]
        one = Tensor(1.0)
        grad = multiply(out_grad, multiply(Tensor(self.scalar), power_scalar(inp, self.scalar - 1)))
        return grad



def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class Div(Op):
    """Op to element-wise divide two nodes."""

    def forward(self, a, b):
        return a/b

    def backward(self, out_grad, node):
        x,y = node._inputs 
        grad_x = divide(out_grad, y)
        grad_y = multiply(negate(out_grad), divide(x, multiply(y, y)))
        return grad_x, grad_y



def divide(a, b):
    return Div()(a, b)


class DivScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        return np.array(a / self.scalar, dtype=a.dtype)

    def backward(self, out_grad, node):
        return  out_grad/self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(Op):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a):
        if self.axes is None :
                return np.swapaxes(a,-2,-1)
        else:
            if(len(self.axes))==2 :
                a1 , a2 = self.axes 
                return np.swapaxes(a, a1, a2)
            else:

                return np.transpose(a, axes=self.axes)

    def backward(self, out_grad, node):
        return transpose(out_grad, self.axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return a.reshape(self.shape)


    def backward(self, out_grad, node):
        a = node._inputs[0]
        return reshape(out_grad, a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return np.broadcast_to(a, self.shape)

    def backward(self, out_grad, node):
    
        a = node._inputs[0]
        original_shape = a.shape 
        converted_shape = out_grad.shape

        changed_shape = len(converted_shape) -len(original_shape)
        grad =out_grad
        for _ in range(changed_shape):
            grad = summation(grad, axes=0)

        for i, (orig_dim, new_dim) in enumerate(zip(original_shape, grad.shape)):
            if orig_dim ==1 and new_dim > 1 :
                grad = summation(grad, axes=i)
                new_shape = list(grad.shape)
                new_shape.insert(i, 1)  
                grad = reshape(grad, tuple(new_shape))

        return grad 
    


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(Op):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a):
        return np.sum(a, self.axes)


    def backward(self, out_grad, node):
        a = node._inputs[0]
        original_shape = a.shape 
        if self.axes is None:
            return out_grad.reshape((1,) * len(original_shape)).broadcast_to(original_shape)
        axes = self.axes 
        if isinstance(axes, int):
            axes = (axes,)
        axes = tuple([ax if ax >= 0 else ax + len(original_shape) for ax in axes])
        grad_shape = list(out_grad.shape)
        for ax in sorted(axes):
            grad_shape.insert(ax, 1)

        reshaped = reshape(out_grad, grad_shape)
        return broadcast_to(reshaped, original_shape)



def summation(a, axes=None):
    return Summation(axes)(a)

class MatMul(Op):
    def forward(self, a, b):
        return np.matmul(a, b)

    def backward(self, out_grad, node):
        a, b = node._inputs

        # If scalar grad (from summation), broadcast to output shape
        if len(out_grad.shape) == 0:
            out_grad = out_grad.broadcast_to(node.shape)

        grad_a = matmul(out_grad, transpose(b, axes=(-1, -2)))
        grad_b = matmul(transpose(a, axes=(-1, -2)), out_grad)

        # Reduce extra dims from broadcasting
        while len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=0)
        while len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=0)

        # Match original input shapes
        grad_a = grad_a.reshape(a.shape)
        grad_b = grad_b.reshape(b.shape)

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(Op):
    def forward(self, a):
        return -a



    def backward(self, out_grad, node):
        return  negate(out_grad)



def negate(a):
    return Negate()(a)


class Log(Op):
    def forward(self, a):
        return  np.log(a)


    def backward(self, out_grad, node):
        inp = node._inputs[0]
        
        return out_grad / inp

def log(a):
    return Log()(a)


class Exp(Op):
    def forward(self, a):
        return np.exp(a)

    def backward(self, out_grad, node):
        x = node._inputs[0]
        return   multiply(out_grad, exp(node._inputs[0])) 
    


def exp(a):
    return Exp()(a)


class ReLU(Op):
    def forward(self, a):
        ### BEGIN YOUR SOLUTION     
        a = a * (a>0)
        return a
        ### END YOUR SOLUTION

    def backward(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node._inputs[0]

        relu_grad = a.data > 0

        return out_grad * relu_grad
        ### END YOUR SOLUTION



def relu(a):
    return ReLU()(a)

