from typing import Any
from baby import init, ops
from .tensor import Tensor
import numpy as np 

class Parameter(Tensor):
    """
    A special Tensor that tells a Module it is a learnable parameter.

    Think of this as a regular Tensor that has been "marked" as a weight or
    bias that should be updated by the optimizer during training. When you
    create a model, you should wrap all learnable tensors (like the weights
    and biases of a linear layer) in this class.

    Example:
        >>> # A regular tensor attribute - will be ignored by model.parameters()
        >>> self.some_data = Tensor([1, 2, 3])
        >>>
        >>> # A parameter - will be found and trained by the optimizer!
        >>> self.weights = Parameter(Tensor.randn(10, 5))
    """
    pass

def _get_parameters(obj) -> list[Parameter]:
    """
    A simple recursive function that finds all Parameter objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    params = []
    
    if isinstance(obj, Parameter):
        return [obj]
    
    if isinstance(obj, Module):
        return obj.parameters()

    if isinstance(obj, dict):
        for value in obj.values():
            params.extend(_get_parameters(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            params.extend(_get_parameters(item))

    return params

def _get_modules(obj) -> list['Module']:
    """
    A simple recursive function that finds all Module objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    modules = []
    
    if isinstance(obj, Module):
        return [obj]
        
    if isinstance(obj, dict):
        for value in obj.values():
            modules.extend(_get_modules(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            modules.extend(_get_modules(item))

    return modules


class Module:
    """
    The base class for all neural network modules (layers).
    """
    def __init__(self):
        self.training = True

    def parameters(self) -> list[Parameter]:
        """
        Returns a list of all parameters in the module and its submodules.
        """
        params = _get_parameters(self.__dict__)        
        unique_params = []
        seen_ids = set()
        for p in params:
            if id(p) not in seen_ids:
                unique_params.append(p)
                seen_ids.add(id(p))
        return unique_params

    def train(self):
        """Sets this module and all its submodules to training mode."""
        self.training = True
        # Find all child modules and recursively call train() on them.
        for m in _get_modules(self.__dict__):
            m.train()

    def eval(self):
        """Sets this module and all its submodules to evaluation mode."""
        self.training = False
        # Find all child modules and recursively call eval() on them.
        for m in _get_modules(self.__dict__):
            m.eval()

    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Makes the module callable like a function.

        This is a Python "magic method" that allows you to treat your module
        instance as a function. It automatically calls the `forward` method
        that you defined.

        Example:
            >>> model = MyModel(10, 2)
            >>> input_tensor = Tensor.randn(64, 10)
            >>> output = model(input_tensor)  # This calls model.forward(input_tensor)
        """
        return self.forward(*args, **kwargs)


class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    The ReLU function is defined as f(x) = max(0, x). It is a non-parametric
    layer, meaning it has no learnable weights.

    Example:
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()  # Apply activation after the linear layer
        )
    """
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    """
    A container that chains a sequence of modules together.

    The modules are applied in the order they are passed to the constructor.
    This is a convenient way to build simple, feed-forward models without
    having to write a custom `forward` method.

    Example:
        # A simple 2-layer MLP for MNIST
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        logits = model(input_tensor)
    """
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

class Dropout(Module):
    """
    A regularization layer to help prevent overfitting.

    During training (`.train()` mode), it randomly sets some input elements to
    zero with a probability of `p`. The remaining elements are scaled up by
    `1 / (1 - p)` to compensate ("inverted dropout").

    During evaluation (`.eval()` mode), this layer does nothing and just
    passes the input through.

    Example:
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2) # Drop 20% of activations during training
        )
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Probability of *keeping* an element is 1 - p
            mask = Tensor.randb(*x.shape, p=(1 - self.p))
            return (x * mask) / (1 - self.p)
        else:
            return x

class Flatten(Module):
    """
    Flattens a tensor by reshaping it to `(batch_size, -1)`.

    This is essential for transitioning from multi-dimensional layers (like
    convolutional layers) to 2D-input layers (like linear layers).

    Example:
        # A CNN might produce a feature map of shape (32, 64, 7, 7)
        # (batch_size, channels, height, width)
        
        model = nn.Sequential(
            nn.Conv2d(...),
            nn.ReLU(),
            nn.Flatten(), # Reshapes output to (32, 64 * 7 * 7) = (32, 3136)
            nn.Linear(3136, 10)
        )
    """
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        # Calculate the product of all dimensions except the first (batch)
        flat_dim = np.prod(x.shape[1:]).item()
        return x.reshape(batch_size, flat_dim)

class Residual(Module):
    """
    Creates a residual connection block, which implements `F(x) + x`.

    This block takes a submodule (`fn`) which defines the transformation `F`.
    The input `x` is passed through `fn`, and the original `x` is added to
    the output (a "skip connection"). This is a core component of architectures
    like ResNet.

    Example:
        # A simple residual block
        main_path = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        res_block = nn.Residual(main_path)
        output = res_block(input_tensor_of_shape_64)
    """
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
    


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias.
                               Defaults to True.

    Shape:
        - Input: `(batch_size, *, in_features)` where `*` means any number of
          additional dimensions.
        - Output: `(batch_size, *, out_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight (Parameter): The learnable weights of the module of shape
                            `(in_features, out_features)`.
        bias (Parameter):   The learnable bias of the module of shape `(out_features,)`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(
            init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype).reshape(
                (1, self.out_features))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        
        res = x @ self.weight

        if self.bias is not None:
            res += self.bias.broadcast_to(res.shape)
        
        return res