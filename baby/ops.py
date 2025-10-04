from typing import Optional
import numpy as np 
from .tensor import Tensor , NDArray, _ensure_tensor

class Function:
    
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




class Add(Function):
    def forward(self, a: NDArray, b: NDArray):
        return a + b

    def backward(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return Add()(a, b)


class AddScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a + self.scalar

    def backward(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class Mul(Function):
    def forward(self, a: NDArray, b: NDArray):
        return a * b

    def backward(self, out_grad: Tensor, node: Tensor):
        a, b = node._inputs
        return out_grad * b, out_grad * a 


def multiply(a, b):
    return Mul()(a, b)


class MulScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a *self.scalar

    def backward(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)




class Pow(Function):
    """Function to element-wise raise a tensor to a power."""

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

class PowerScalar(Function):
    """Function raise a tensor to an (integer) power."""

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


class Div(Function):
    """Function to element-wise divide two nodes."""

    def forward(self, a, b):
        return a/b

    def backward(self, out_grad, node):
        x,y = node._inputs 
        grad_x = divide(out_grad, y)
        grad_y = multiply(negate(out_grad), divide(x, multiply(y, y)))
        return grad_x, grad_y



def divide(a, b):
    return Div()(a, b)


class DivScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        return np.array(a / self.scalar, dtype=a.dtype)

    def backward(self, out_grad, node):
        return  out_grad/self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(Function):
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


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return a.reshape(self.shape)


    def backward(self, out_grad, node):
        a = node._inputs[0]
        return reshape(out_grad, a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(Function):
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


class Summation(Function):
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

class MatMul(Function):
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


class Negate(Function):
    def forward(self, a):
        return -a



    def backward(self, out_grad, node):
        return  negate(out_grad)



def negate(a):
    return Negate()(a)


class Log(Function):
    def forward(self, a):
        return  np.log(a)


    def backward(self, out_grad, node):
        inp = node._inputs[0]
        
        return out_grad / inp

def log(a):
    return Log()(a)


class Exp(Function):
    def forward(self, a):
        return np.exp(a)

    def backward(self, out_grad, node):
        x = node._inputs[0]
        return   multiply(out_grad, exp(node._inputs[0])) 
    


def exp(a):
    return Exp()(a)


class ReLU(Function):
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



class Sigmoid(Function):
    def forward(self, a):
        out = 1/(1+np.exp(-a))
        return out 
    
    def backward(self, out_grad, node):
        sigm_oid = node.data 
        local = sigm_oid * (1-sigm_oid)
        return out_grad * local 
    

def sigmoid(x):
    return Sigmoid()(x) 


class Tanh(Function):
    def forward(self,a):
        return np.tanh(a)
    def backward(self, out_grad, node):
        tan_h = node.data 
        local = 1- tan_h**2 
        return out_grad *local 
    

def tanh(x):
    return Tanh()(x)





class LogSoftmax(Function):
    def forward(self,a):
        max_a = np.max(a,axis=(1,), keepdims=True)
        shifted_a = a - max_a
        log_sum_exp  = np.log(np.sum(np.exp(shifted_a), axis=(1,), keepdims=True))
        return shifted_a - log_sum_exp
    def backward(self,out_grad, node):
        input_tensor = node._inputs[0] 
        axes = (1,)  #will change in future 
        new_shape = list(input_tensor.shape)

        new_shape[1] =1 # the 1st dimensiojn will get squeezed 
        new_shape = tuple(new_shape)

        sum_out_grad = summation(out_grad, axes=axes)
        reshaped_sum_out_grad =reshape(sum_out_grad, new_shape)
        broadcasted_sum_out_grad = broadcast_to(reshaped_sum_out_grad, input_tensor.shape)

        max_val = input_tensor.data.max(axis=axes, keepdims=True)
        max_tensor = Tensor(max_val, device=input_tensor.device , dtype=input_tensor.dtype)

        neg_max_tensor = negate(max_tensor)
        shifted = add(input_tensor,  broadcast_to(neg_max_tensor, input_tensor.shape))

        exp_shifted = exp(shifted)
        sum_exp = summation(exp_shifted,axes=axes)
        reshaped_sum_exp = reshape(sum_exp, new_shape)
        softmax_Z = divide(exp_shifted, broadcast_to(reshaped_sum_exp, input_tensor.shape))
        multiplied_term = multiply(softmax_Z, broadcasted_sum_out_grad)
        neg_multiplied_term = negate(multiplied_term)
        return add(out_grad, neg_multiplied_term)


def logsoftmax(a):
    return LogSoftmax()(a)



class LogSumExp(Function):
    def __init__(self, axes):
        self.axes =axes 
    def forward(self,a):
        max_a = np.max(a, axis=self.axes, keepdims=True)
        sub_a = a - max_a
        exp_sub = np.exp(sub_a)
        sum_exp = np.sum(exp_sub,axis=self.axes, keepdims=True)
        log_sum = max_a + np.log(sum_exp)

        return np.squeeze(log_sum, axis=self.axes)
    def backward(self, out_grad: Tensor, node: Tensor):
        a = node._inputs[0]

        new_shape = list(a.shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(a.shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        for axis in axes:
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        max_a_val = a.data.max(axis=self.axes, keepdims=True)
        max_a_tensor = Tensor(max_a_val, device=a.device, dtype=a.dtype)

        shifted_a = a - max_a_tensor
        exp_shifted_a = exp(shifted_a)
        
        sum_exp_shifted_a = summation(exp_shifted_a, self.axes)
        
        # Reshape the sum for division (same logic as before)
        reshaped_sum = reshape(sum_exp_shifted_a, new_shape)
        
        softmax = divide(exp_shifted_a, broadcast_to(reshaped_sum, a.shape))

        reshaped_out_grad = reshape(out_grad, new_shape)
        grad = multiply(broadcast_to(reshaped_out_grad, a.shape), softmax)
        
        return grad
    

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)




class Sqrt(Function):
    def forward(self, a):
        return  np.sqrt(a)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad / (2 * sqrt(a))

def sqrt(a):
    return Sqrt()(a)    