from typing import Any
from baby import init, ops
from .tensor import Tensor
import numpy as np 
import pickle 
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
    def __init__(self, data, *args, **kwargs):
        # Parameters always require gradients.
        kwargs['requires_grad'] = True
        super().__init__(data, *args, **kwargs)

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
    

    def state_dict(self):
        state_dic ={}
        for key,value in self.__dict__.items():
            if isinstance(value, Tensor) or isinstance(value, Parameter):
                state_dic[key] = value.data
            elif isinstance(value, Module):
                child_sd =value.state_dict()
                for k,v in child_sd.items():
                    state_dic[f"{key}.{k}"] = v


            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        child_sd = item.state_dict()
                        for k, v in child_sd.items():
                            state_dic[f"{key}.{i}.{k}"] = v
        return state_dic    
    def load_state_dict(self,state_dict):
            for key,value in self.__dict__.items():
                if isinstance(value, Parameter) or isinstance(value,Tensor):
                    if key in state_dict:

                        if (value.shape != state_dict[key].shape):
                            raise ValueError(f"Shape mismatch for {key}: expected {value.shape}, got {state_dict[key].shape}")
                        value.data = state_dict[key]
                 
                elif isinstance(value, Module):
                    prefix = f"{key}."
                    child_sd = {
                        k[len(prefix):]: v 
                        for k, v in state_dict.items() 
                        if k.startswith(prefix)
                    }
                    value.load_state_dict(child_sd)

                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, Module):
                            prefix = f"{key}.{i}."
                            child_sd = {
                                k[len(prefix):]: v 
                                for k, v in state_dict.items() 
                                if k.startswith(prefix)
                            }
                            item.load_state_dict(child_sd)


                
    def load(self, filename):
        with open(filename,'rb') as f:
            self.load_state_dict(pickle.load(f))

    def save(self,filename):
        with open(filename , 'wb') as f :
            pickle.dump(self.state_dict(), f)


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
    



class Tanh(Module):
    def forward(self, x: Tensor): 
        return ops.tanh(x)

class Sigmoid(Module):
    def forward(self,x: Tensor):
        return ops.sigmoid(x)
    


class SoftmaxLoss(Module):
    def forward(self, logits, y):
        """
        Calculates the softmax cross-entropy loss.

        Args:
            logits: A tensor of shape (batch_size, num_classes) containing the model's raw output.
            y: A list or numpy array of integers (batch_size,) containing the true class labels.
        """
        n, k = logits.shape
        
        y_one_hot = Tensor.one_hot(y, k, requires_grad=False)
        
        logsumexp_val = ops.logsumexp(logits, axes=(1,))
        
        h_y = (logits * y_one_hot).sum(axes=(1,))
        
        return (logsumexp_val - h_y).sum() / n


class MSELoss(Module):
    def forward(self, pred, target):
        """
        Calculates the Mean Squared Error.
        Args:
            pred: Tensor of shape (batch_size, output_dim)
            target: Tensor of shape (batch_size, output_dim)
        """
        return ((pred - target) ** 2).mean()

class LayerNorm1d(Module):
    def __init__(self,dim: int, eps: float=1e-5,device=None, dtype="float32"):
        super().__init__()
        self.dim = dim 
        self.eps = eps 
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype))
        self.bias = Parameter(Tensor.zeros(dim, dtype="float32"))
    
    def forward(self,x):
        mean = ops.summation(x, axes=(1,))/self.dim 
        mean_reshaped = ops.reshape(mean, (x.shape[0], 1))
        mean_broadcasted = ops.broadcast_to(mean_reshaped, x.shape)

        x_minus_mean = x - mean_broadcasted
        var= ops.summation(x_minus_mean**2 , axes=(1,))/self.dim 
        var_reshaped = ops.reshape(var, (x.shape[0], 1))
        var_broadcasted = ops.broadcast_to(var_reshaped,x.shape)
        std = ops.sqrt(var_broadcasted + self.eps)
        x_hat = x_minus_mean/std 

        weight_reshaped = ops.reshape(self.weight, (1,self.dim))
        bias_reshaped = ops.reshape(self.bias, (1, self.dim))
        weight_broadcasted = ops.broadcast_to(weight_reshaped, x.shape)
        bias_broadcasted = ops.broadcast_to(bias_reshaped, x.shape)
        out = weight_broadcasted * x_hat + bias_broadcasted

        return out 


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim 
        self.eps = eps 
        self.momentum = momentum
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype))
        self.bias = Parameter(Tensor.zeros(dim,  dtype=dtype))
        self.running_mean = Tensor.zeros(dim, dtype=dtype)
        self.running_var = Tensor.ones(dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = x.shape[0]            
            mean = ops.summation(x, axes=(0,)) / batch_size
            var = ops.summation((x - ops.broadcast_to(mean.reshape((1, self.dim)), x.shape))**2, axes=(0,)) / batch_size
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data
            mean_to_use = mean
            var_to_use = var
        else:
            mean_to_use = self.running_mean
            var_to_use = self.running_var
        mean_reshaped = mean_to_use.reshape((1, self.dim))
        var_reshaped = var_to_use.reshape((1, self.dim))
        std = ops.sqrt(var_reshaped + self.eps)
        x_hat = (x - ops.broadcast_to(mean_reshaped, x.shape)) / ops.broadcast_to(std, x.shape)
        weight_reshaped = self.weight.reshape((1, self.dim))
        bias_reshaped = self.bias.reshape((1, self.dim))
        
        return ops.broadcast_to(weight_reshaped, x.shape) * x_hat + ops.broadcast_to(bias_reshaped, x.shape)



class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output
    is the corresponding word embeddings.

    Args:
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.

    Example:
        >>> # An embedding module for a dictionary of 10 words, each with a 3-dim vector
        >>> embedding = nn.Embedding(10, 3)
        >>> # Input is a Tensor of integer indices
        >>> input_indices = Tensor([1, 4, 9, 2])
        >>> output = embedding(input_indices)
        >>> print(output.shape)
        (4, 3)
    """
    def __init__(self, num_embeddings: int, embeddings_dim: int ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.weight =Parameter(init.kaiming_uniform(num_embeddings, embeddings_dim))
    
    def forward(self, x ):
        embedded_data = self.weight.data[x.data.astype(int)]
        return Tensor(embedded_data)
    