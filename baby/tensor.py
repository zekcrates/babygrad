
import numpy as np

NDArray = np.ndarray
def _ensure_tensor(val):
    return val if isinstance(val, Tensor) else Tensor(val, requires_grad=False)


class Tensor:
    """
    A tensor with automatic differentiation support.
    
    This is the core of Baby. It wraps a NumPy array and tracks
    operations for computing gradients automatically.
    """
    
    def __init__(self, data, *, device=None, dtype="float32", requires_grad=True):
        """
        Create a new tensor.
        
        Args:
            data: Array-like data (list, numpy array, or another Tensor)
            device: Device placement (currently ignored, CPU only)
            dtype: Data type for the array
            requires_grad: Whether to track gradients for this tensor
        
        Design decision: requires_grad defaults to True (unlike PyTorch)
        because this is a learning library - we want to see gradients!
        """
        if isinstance(data, Tensor):
            self.data = data.data.astype(dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        
        self.grad = None
        self.requires_grad = requires_grad
        
        self._op = None        # Operation that created this tensor
        self._inputs = []      # Parent tensors
        
        self._device = device if device else  "cpu"
    
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

        topo_order = []
        visited = set()
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._inputs:
                    build_topo(parent)
                topo_order.append(node)
        build_topo(self)
        
        # This dictionary will hold the gradient TENSOR for every node
        grads = {}
        
        # NEEDLE-STYLE: If no gradient is provided, default to a tensor of ones
        # with the same shape as the output. No error for non-scalars.
        if grad is None:
            grads[id(self)] = Tensor(np.ones_like(self.data))
        else:
            grads[id(self)] = _ensure_tensor(grad)
        
        for node in reversed(topo_order):
            out_grad = grads.get(id(node))
            if out_grad is None:
                continue

            if node.grad is None:
                node.grad = out_grad.data.copy()
            else:
                node.grad += out_grad.data

            if node._op:
                input_grads = node._op.backward(out_grad, node)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)
                
                for i, parent in enumerate(node._inputs):
                    if parent.requires_grad:
                        parent_id = id(parent)
                        if parent_id not in grads:
                            grads[parent_id] = input_grads[i]
                        else:
                            grads[parent_id] = grads[parent_id] + input_grads[i]

    def numpy(self):
        """
        Return the data as a NumPy array (detached from graph).
        
        This creates a copy, so modifications won't affect the tensor.
        
        Returns:
            np.ndarray: Copy of the tensor's data
        """
        return self.data.copy()
    
    def detach(self):
        """
        Create a new tensor with same data but no gradient tracking.
        
        Useful when you want to use values without building computation graph.
        
        Returns:
            Tensor: New tensor with requires_grad=False
        
        Example:
            >>> x = Tensor([1, 2, 3], requires_grad=True)
            >>> y = x.detach()  # y doesn't track gradients
            >>> z = y * 2       # This operation won't be in graph
        """
        return Tensor(self.data, requires_grad=False, dtype=str(self.dtype))
    
    # ========================================
    # PROPERTIES
    # ========================================
    
    @property
    def shape(self):
        """Shape of the tensor."""
        return self.data.shape
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self.data.dtype
    
    @property
    def ndim(self):
        """Number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self):
        """Total number of elements."""
        return self.data.size
    
    @property
    def device(self):
        """Device where tensor lives (currently always 'cpu')."""
        return self._device
    
    @property
    def T(self):
        """Transpose (swaps last two dimensions)."""
        return self.transpose()
    
    # ========================================
    # STRING REPRESENTATION
    # ========================================
    
    def __repr__(self):
        """
        Detailed representation showing data and gradient tracking.
        
        Example:
            >>> x = Tensor([1, 2, 3])
            >>> print(repr(x))
            Tensor([1. 2. 3.], requires_grad=True)
        """
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self):
        """
        Simple string representation (just the data).
        
        Example:
            >>> x = Tensor([1, 2, 3])
            >>> print(x)
            [1. 2. 3.]
        """
        return str(self.data)
    
    # ========================================
    # ARITHMETIC OPERATORS
    # ========================================
    
    def __add__(self, other):
        """Addition: a + b"""
        from .ops import Add
        return Add()(self, other)
    
    def __radd__(self, other):
        """Right addition: 5 + tensor"""
        return self.__add__(other)
    
    def __mul__(self, other):
        """Multiplication: a * b"""
        from .ops import Mul
        return Mul()(self, other)
    
    def __rmul__(self, other):
        """Right multiplication: 5 * tensor"""
        return self.__mul__(other)
    
    def __sub__(self, other):
        """Subtraction: a - b"""
        # a - b = a + (-b)
        return self.__add__(self.__neg__() if other is self else -other)
    
    def __rsub__(self, other):
        """Right subtraction: 5 - tensor"""
        from .ops import Add, Negate
        return Add()(Tensor(other, requires_grad=False), Negate()(self))
    
    def __truediv__(self, other):
        """Division: a / b"""
        from .ops import Div
        return Div()(self, other)
    
    def __rtruediv__(self, other):
        """Right division: 5 / tensor"""
        from .ops import Div
        return Div()(Tensor(other, requires_grad=False), self)
    
    def __pow__(self, exponent):
        """Power: a ** n"""
        from .ops import Pow
        return Pow(exponent)(self)
    
    def __neg__(self):
        """Negation: -a"""
        from .ops import Negate
        return Negate()(self)
    
    def __matmul__(self, other):
        """Matrix multiplication: a @ b"""
        from .ops import MatMul
        return MatMul()(self, other)
    
    def __rmatmul__(self, other):
        """Right matmul: array @ tensor"""
        from .ops import MatMul
        return MatMul()(Tensor(other, requires_grad=False), self)
    
    # ========================================
    # TENSOR OPERATIONS (METHODS)
    # ========================================
    
    def sum(self, axes=None):
        """
        Sum elements along given axis.
        
        Args:
            axis: Axis or axes to sum over (None = sum all)
            keepdims: Keep reduced dimensions as size 1
        
        Example:
            >>> x = Tensor([[1, 2], [3, 4]])
            >>> x.sum()  # 10
            >>> x.sum(axis=0)  # [4, 6]
        """
        from .ops import Summation
        return Summation(axes, )(self)
    
    # def mean(self, axis=None):
    #     """Mean of elements along given axis."""
    #     from .ops import 
    #     return Mean(axis)(self)
    
    def reshape(self, *shape):
        """
        Reshape tensor to new shape.
        
        Args:
            shape: New shape (can pass as tuple or separate args)
        
        Example:
            >>> x = Tensor([1, 2, 3, 4])
            >>> x.reshape(2, 2)  # [[1, 2], [3, 4]]
            >>> x.reshape((2, 2))  # Also works
        """
        from .ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape(shape)(self)
    
    def transpose(self, axes=None):
        """
        Transpose tensor.
        
        Args:
            axes: Permutation of axes (None = reverse all axes)
        
        Example:
            >>> x = Tensor([[1, 2], [3, 4]])
            >>> x.transpose()  # [[1, 3], [2, 4]]
        """
        from .ops import Transpose
        return Transpose(axes)(self)
    
    def broadcast_to(self, shape):
        """Broadcast tensor to new shape."""
        from .ops import BroadcastTo
        return BroadcastTo(shape)(self)


    # basic initializers 
    @classmethod
    def rand(cls, *shape, low=0.0, high=1.0, dtype="float32", requires_grad=False):
        """Generate random numbers uniform between low and high"""
        array = np.random.rand(*shape) * (high - low) + low
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
        """Generate random normal with specified mean and std deviation"""
        array = np.random.randn(*shape) * std + mean
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def constant(cls, *shape, c=1.0, dtype="float32", requires_grad=False):
        """Generate a constant Tensor"""
        array = np.ones(*shape) * c
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype="float32", requires_grad=False):
        """Generate an all-ones Tensor"""
        return cls.constant(*shape, c=1.0, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *shape, dtype="float32", requires_grad=False):
        """Generate an all-zeros Tensor"""
        return cls.constant(*shape, c=0.0, dtype=dtype, requires_grad=requires_grad)

# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def tensor(data, requires_grad=True, dtype="float32"):
    """
    Convenience function to create a tensor.
    
    Example:
        >>> x = baby.tensor([1, 2, 3])
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

def zeros_like(array: Tensor, *, requires_grad=False):
    """Generate a all-zeros Tensor with the same shape/dtype as another Tensor."""
    return Tensor.zeros(*array.shape, dtype=str(array.dtype), requires_grad=requires_grad)

def ones_like(array: Tensor, *, requires_grad=False):
    """Generate a all-ones Tensor with the same shape/dtype as another Tensor."""
    return Tensor.ones(*array.shape, dtype=str(array.dtype), requires_grad=requires_grad)
